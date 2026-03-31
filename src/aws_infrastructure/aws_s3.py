from io import BytesIO
import os
from pathlib import PurePosixPath
import boto3
from dotenv import load_dotenv
import pandas as pd
from crawler import post_filtering, post_formating, etf_formating


def load_parquet_from_s3(
    bucket_name: str,
    object_key: str,
    num_posts: int = 10000
) -> pd.DataFrame:
    """
    Read a parquet file from S3 and keep only the ID, created_by, and content columns.
    """
    
    s3_client = boto3.client("s3")
    response = s3_client.get_object(Bucket=bucket_name, Key=object_key)

    df = pd.read_parquet(BytesIO(response["Body"].read()))
    
    if "etf" not in object_key.lower():
        df = post_filtering(df[["id", "created_at", "content"]].copy(), num_posts=num_posts)
        df = post_formating(df, column="created_at")
    else:
        df = etf_formating(df, column="timestamp")
    
    print(df)
    return df



def load_group_parquet_from_s3(
    bucket_name: str,
    object_key: str,
    num_posts: int = 10000,
) -> pd.DataFrame:
    s3_client = boto3.client("s3")

    response = s3_client.list_objects_v2(Bucket=bucket_name)
    matching_keys = []

    for obj in response.get("Contents", []):
        key = obj["Key"]
        file_name = PurePosixPath(key).name

        if key.lower().endswith(".parquet") and object_key.lower() in file_name.lower():
            matching_keys.append(key)

    if not matching_keys:
        raise ValueError(f"No parquet files found with keyword: {object_key}")

    df_list = []

    for key in matching_keys:
        response = s3_client.get_object(Bucket=bucket_name, Key=key)
        df_part = pd.read_parquet(BytesIO(response["Body"].read()))
        df_list.append(df_part)

    df = pd.concat(df_list, ignore_index=True)

    if "etf" not in object_key.lower():
        df = post_filtering(df[["id", "created_at", "content"]].copy(), num_posts=num_posts)
        df = post_formating(df, column="created_at")
    else:
        df = etf_formating(df, column="timestamp")

    print(df)
    return df



def save_df_to_s3_parquet(df: pd.DataFrame, bucket_name: str, object_key: str) -> None:
    buffer = BytesIO()
    df.to_parquet(buffer, index=False)
    buffer.seek(0)

    s3_client = boto3.client("s3")
    s3_client.put_object(
        Bucket=bucket_name,
        Key=object_key,
        Body=buffer.getvalue()
    )



if __name__ == "__main__":
    load_dotenv()
    load_parquet_from_s3(
        bucket_name=os.getenv("AWS_S3_BUCKET_NAME"),
        object_key=os.getenv("AWS_S3_OBJECT_KEY_ETF"),
    )