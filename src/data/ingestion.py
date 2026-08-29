import os
import shutil
from pathlib import Path
from typing import Tuple, List
import random
from src.logger import logger
import boto3

import os

S3_BUCKET = os.getenv("S3_BUCKET", "your-medical-mlops-bucket")
S3_REGION = os.getenv("AWS_REGION", "ap-south-1")

S3_BUCKET = os.getenv("S3_BUCKET")

S3 = boto3.client("s3")


def upload_to_s3(bucket_name: str) -> bool:
    """
    Check if an S3 bucket exists.

    Args:
        bucket_name (str): The name of the S3 bucket."""
    local_dir = Path("data/raw")
    if not local_dir.exists():
        logger.error(f"Local directory does not exist: {local_dir}")
        raise FileNotFoundError(f"Local directory not found: {local_dir}")
    if not S3_BUCKET:
        logger.error("S3_BUCKET environment variable is not set.")
        raise ValueError("S3_BUCKET environment variable is not set.")
    for file_path in local_dir.rglob("*"):
        if file_path.is_file():
            s3_key = str(file_path.relative_to(local_dir))
            try:
                S3.upload_file(str(file_path), bucket_name, s3_key)
            except Exception as e:
                logger.error(f"Failed to upload {file_path} to S3: {e}")
                raise e
    print(f"Uploaded to s3://{S3_BUCKET}/{s3_key}")



    
    # Placeholder for actual S3 bucket existence check
    # In a real implementation, you would use boto3 or another library to check the bucket
    logger.info(f"Checking if S3 bucket exists: {bucket_name}")
    return True  # Assume the bucket exists for this placeholder


def ingest_data():
    print("Starting data ingestion...")
    dataset_path = Path("data/raw")
    upload_to_s3(dataset_path)
    print("Data ingestion completed.")
    
    