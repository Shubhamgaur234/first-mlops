import boto3
import os
from dotenv import load_dotenv
from pathlib import Path
from src.constants import AWS_ACCESS_KEY_ID_ENV_KEY, AWS_SECRET_ACCESS_KEY_ENV_KEY, REGION_NAME

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

class S3Client:
    """
    S3Client is a class to connect to AWS S3 using boto3.
    It reads the AWS credentials from environment variables.
    """
    def __init__(self, region_name=REGION_NAME):
        aws_access_key_id = os.getenv(AWS_ACCESS_KEY_ID_ENV_KEY)
        aws_secret_access_key = os.getenv(AWS_SECRET_ACCESS_KEY_ENV_KEY)
        region_name = os.getenv("AWS_DEFAULT_REGION") or os.getenv("AWS_REGION") or region_name

        if not aws_access_key_id or not aws_secret_access_key:
            raise Exception(f"AWS credentials not found. Please set {AWS_ACCESS_KEY_ID_ENV_KEY} and {AWS_SECRET_ACCESS_KEY_ENV_KEY} environment variables.")

        self.s3_resource = boto3.resource(
            's3',
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=region_name
        )
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=region_name
        )
