from dotenv import load_dotenv
import os
import time
import pandas as pd
from pathlib import Path
from aws_sns import merge_overall_model_accuracy, publish_etf_signals_to_sns
from aws_aurora_dsql import dsql_execute_sql
from aws_dynamodb import dedupe_posts, load_batch_df_to_dynamodb_cli, load_df_to_dynamodb_cli, load_ids_from_dynamodb, add_id_to_processed_post_ids, input_df_columns_filter
from crawler import apify_crawler_default, apify_crawler_backup, customized_crawler, filter_posts_by_date_and_content_length, duplicate_posts_to_minute_boundaries, add_post_prefix_to_content
from aws_s3 import load_parquet_from_s3, save_df_to_s3_parquet
from etf_historical import get_stock_bars, extract_ids_and_contents, post_etf_joins, deduplicate_and_remove_existing_ids
from aws_bedrock import build_validator_user_prompt, query_bedrock_model, concurrent_job_with_prompt_caching, concurrent_job_with_prompt_caching_and_dynamic_workers, save_results_to_jsonl
from ml_model_deployment import calculate_processing_latency, merge_post_signal_and_validation_dfs, predict_symbol_combo_signals, load_xgboost_models_from_s3, load_selected_model_performance, score_decision_layer, symbol_voting_system
from ml_training_data_building import convert_validator_output_to_df, expand_json_output_to_metric_columns, join_etf_and_json_output, keep_only_x_and_y_columns, load_batch_output_jsonl_to_df, load_single_output_to_df, scale_input_metric_columns
from prompt.standard_metrics import STANDARD_METRICS
from exceptions import RestartProcess


def inference_init():
    deployed_models = load_xgboost_models_from_s3()

    df = customized_crawler(num_posts=20)
    processed_post_ids = set(df["id"].dropna())
    
    return deployed_models, processed_post_ids



def crawl_posts_and_preprocess(processed_post_ids: set[str]) -> tuple[pd.DataFrame, set[str]]:
    
    df = customized_crawler(num_posts=20)
    
    post_df = dedupe_posts(df, processed_post_ids, id_col="id")
    
    if post_df.empty:
        raise RestartProcess("No new posts to process. Restarting the process.")
    else:
        processed_post_ids = add_id_to_processed_post_ids(post_df, processed_post_ids)
    
        post_date = pd.to_datetime(post_df.iloc[0]["created_at"])
        post_start_date = (post_date - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        post_end_date = (post_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

        post_df = filter_posts_by_date_and_content_length(
            df=post_df,
            start_date=post_start_date,
            end_date=post_end_date,
            min_content_length=50,
            date_column="created_at",
            content_column="content",
        )
        
        if post_df.empty:
            raise RestartProcess("No posts after filtering by date and content length. Restarting the process.")
        else:
            post_df = duplicate_posts_to_minute_boundaries(
                df=post_df,
                datetime_column="created_at",
                post_duplicate=False,
            )

            post_df = add_post_prefix_to_content(post_df)
            
            return post_df, processed_post_ids
        


def generate_llm_custom_embedding_vector(post_df: pd.DataFrame) -> pd.DataFrame:
    
    ids, contents = extract_ids_and_contents(post_df)

    system_prompt = Path("src/prompt/system_prompt_v4.txt").read_text(encoding="utf-8")

    results = concurrent_job_with_prompt_caching_and_dynamic_workers(
        ids=ids,
        user_prompts=contents,
        system_prompt=system_prompt,
        model_id="us.anthropic.claude-opus-4-6-v1",
        region_name="us-east-1",
        temperature=0.4,
        max_tokens=2000,
        top_p=0.95,
        top_k=250,
        initial_workers=2,
        system_prompt_caching=True,
        batch_size=25,
        max_attempts=6,
        min_workers=1,
        max_workers=3,
        throttle_ratio_to_reduce=0.2,
        if_save_file=False,
        file_save_path="batch_in_progress.jsonl",
    )
    
    output_df = load_single_output_to_df(results)
    
    output_df = expand_json_output_to_metric_columns(output_df, STANDARD_METRICS)
    
    return output_df



def ml_model_inference(deployed_models: dict, output_df: pd.DataFrame, post_df: pd.DataFrame) -> pd.DataFrame:
    
    df_model = input_df_columns_filter(output_df)
    
    df_model = scale_input_metric_columns(df_model, STANDARD_METRICS)
    
    X = keep_only_x_and_y_columns(df_model)
    
    selected_model_df = load_selected_model_performance()
    
    prediction_result = predict_symbol_combo_signals(
        X=X,
        models=deployed_models,
        selected_model_df=selected_model_df,
        max_workers=8,
    )
    
    signal_prediction = symbol_voting_system(prediction_result, post_df)
    
    return signal_prediction



def llm_validation_and_signal_scoring(post_df: pd.DataFrame, signal_prediction: pd.DataFrame) -> pd.DataFrame:
    
    validator_user_prompt = build_validator_user_prompt(post_df, signal_prediction)
    validator_system_prompt = Path("src/prompt/validator_system_prompt.txt").read_text(encoding="utf-8")
    
    validation_result = query_bedrock_model(
        system_prompt=validator_system_prompt,
        user_prompt=validator_user_prompt,
        model_id="us.anthropic.claude-opus-4-6-v1",
        region_name="us-east-1",
        temperature=0.2,
        max_tokens=1000,
        top_p=0.95,
        top_k=250,
        system_prompt_caching=False,
    )
    
    validation_df = convert_validator_output_to_df(validation_result)
    
    merged_df = merge_post_signal_and_validation_dfs(signal_prediction, post_df, validation_df)
    
    merged_df = calculate_processing_latency(merged_df)
    
    load_batch_df_to_dynamodb_cli(merged_df, table_name="processed_records")
    
    merged_df = score_decision_layer(
        df=merged_df,
        symbol_threshold=0.7,
        combined_threshold=0.4,
    )
    
    if merged_df.empty:
        raise RestartProcess("No valid signals after scoring. Restarting the process.")
    else:
        load_batch_df_to_dynamodb_cli(merged_df, table_name="published_records")
        
        return merged_df
    


def publish_signals(merged_df: pd.DataFrame) -> None:
    
    merged_df = merge_overall_model_accuracy(merged_df)
    
    response = publish_etf_signals_to_sns(merged_df, topic_arn=os.getenv("AWS_SNS_TOPIC_ARN"))
    
    print(f"SNS publish response: {response}")
    
    

def main() -> None:
    
    load_dotenv()
    print("Starting inference pipeline...", flush=True)
    
    deployed_models, processed_post_ids = inference_init()
    
    while True:
        try:
            # processed_post_ids.pop()

            post_df, processed_post_ids = crawl_posts_and_preprocess(processed_post_ids)
            
            output_df = generate_llm_custom_embedding_vector(post_df)
            
            signal_prediction = ml_model_inference(deployed_models, output_df, post_df)
            
            merged_df = llm_validation_and_signal_scoring(post_df, signal_prediction)   

            publish_signals(merged_df)
            
            print("Process completed successfully.")

        except RestartProcess as e:
            print(f"Restarting process: {e}", flush=True)
            time.sleep(10)
            continue



if __name__ == "__main__":
    
     main()
    