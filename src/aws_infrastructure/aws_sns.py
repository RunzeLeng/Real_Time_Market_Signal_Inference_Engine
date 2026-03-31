import os
import boto3
import pandas as pd
from aws_aurora_dsql import dsql_execute_sql


def merge_overall_model_accuracy(merged_df: pd.DataFrame) -> pd.DataFrame:
    sql_query = """
    SELECT symbol, AVG(avg_valid_accuracy_high_confidence) AS model_accuracy
    FROM training_output.selected_model_performance
    GROUP BY symbol
    """
    
    rows = dsql_execute_sql(
        host=os.getenv("AWS_AURORA_DB_HOST"),
        database="postgres",
        sql=sql_query,
        user="admin",
        region="us-east-1",
        profile="default",
    )
    
    merged_df = merged_df.merge(
        rows, on="symbol", how="inner"
    )
    
    return merged_df



def publish_etf_signals_to_sns(
    merged_df: pd.DataFrame,
    topic_arn: str | None = None,
    subject: str = "ETF Market Signal Alert",
    region_name: str = "us-east-1",
) -> dict:
    sns_client = boto3.client("sns", region_name=region_name)

    message_lines = []

    message_lines.append(f"Post ID: {merged_df.iloc[0]['id']}")
    message_lines.append(f"Created At: {merged_df.iloc[0]['created_at']}")
    message_lines.append(f"Market Impact Score: {merged_df.iloc[0]['market_impact_score']}")
    message_lines.append(f"End-to-End Processing Latency: {merged_df.iloc[0]['latency']} seconds")

    message_lines.append("")
    message_lines.append(str(merged_df.iloc[0]["content"]))

    message_lines.append("")
    message_lines.append("ETF Signals:")

    for _, row in merged_df.iterrows():
        line = (
            f" - {row['symbol']}: {row['predicted_signal']}"
            f" | ML Model Accuracy={row['model_accuracy']:.1%}"
            f" | LLM Reasonableness Score={row['reasonableness_score']}"
            f" | Reason={row['brief_reason']}"
        )
        message_lines.append(line)

    message = "\n".join(message_lines)

    response = sns_client.publish(
        TopicArn=topic_arn,
        Subject=subject,
        Message=message,
    )

    return response