from pathlib import Path
import pandas as pd
from aws_aurora_dsql import create_table_and_load_df_to_aurora, dsql_execute_sql
from aws_bedrock import query_bedrock_model
from aws_s3 import read_parquet_files_from_s3_prefix
from ml_training_data_building import convert_news_topic_matching_output_to_df
from dotenv import load_dotenv
import os
from news_ingestion_pipeline import fetch_news_thenewsapi


def build_news_metadata_for_vector_ingestion(
    news_df: pd.DataFrame,
    topic_mapping_df: pd.DataFrame,
) -> pd.DataFrame:
    topic_metadata_df = (
        topic_mapping_df
        .groupby("uuid")
        .agg(
            topics=("topic", list),
            max_topic_confidence=("confidence_score", "max"),
        )
        .reset_index()
    )

    return news_df.merge(
        topic_metadata_df,
        on="uuid",
        how="left",
    )



def build_vector_metadata_with_topic_scores(
    news_df: pd.DataFrame,
    topic_mapping_df: pd.DataFrame,
) -> pd.DataFrame:
    topic_metadata_df = (
        topic_mapping_df
        .groupby("uuid")
        .agg(
            topics=("topic", list),
            max_topic_confidence=("confidence_score", "max"),
        )
        .reset_index()
    )

    topic_score_df = (
        topic_mapping_df
        .pivot_table(
            index="uuid",
            columns="topic",
            values="confidence_score",
            aggfunc="max",
            fill_value=0.0,
        )
        .add_prefix("topic_score_")
        .reset_index()
    )

    metadata_df = news_df.merge(
        topic_metadata_df,
        on="uuid",
        how="left",
    )

    metadata_df = metadata_df.merge(
        topic_score_df,
        on="uuid",
        how="left",
    )

    metadata_df["topics"] = metadata_df["topics"].apply(
        lambda value: value if isinstance(value, list) else []
    )

    return metadata_df



if __name__ == "__main__":
    load_dotenv()
    
    # sql_query_2 = """
    # SELECT * FROM training_data.news_topic_matching
    # """
    
    # rows = dsql_execute_sql(
    #     host=os.getenv("AWS_AURORA_DB_HOST"),
    #     database="postgres",
    #     sql=sql_query_2,
    #     user="admin",
    #     region="us-east-1",
    #     profile="default",
    # )
    # print(rows)