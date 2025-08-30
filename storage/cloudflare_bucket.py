import boto3
import pandas as pd
from botocore.client import Config
from io import StringIO
from logger.bs_logger import bs_logger

from config import CLOUDFLARE_ACCOUNT_ID, CLOUDFLARE_R2_ACCESS_KEY, CLOUDFLARE_R2_SECRET_KEY

# Cloudflare R2 configuration
access_key = CLOUDFLARE_R2_ACCESS_KEY
secret_key = CLOUDFLARE_R2_SECRET_KEY
endpoint_url = f'https://{CLOUDFLARE_ACCOUNT_ID}.r2.cloudflarestorage.com'
bucket_name = 'blocksurvey-ai-data-analysis'

# Setup boto3 client
cloudflare_client = boto3.client(
    's3',
    endpoint_url=endpoint_url,
    aws_access_key_id=access_key,
    aws_secret_access_key=secret_key,
    config=Config(signature_version='s3v4'),
    region_name='auto'
)

def read_file(file_key):
    try:
        response = cloudflare_client.get_object(Bucket=bucket_name, Key=file_key)
        file_content = response['Body'].read().decode('utf-8')
        df = pd.read_csv(StringIO(file_content))
        return df
    except cloudflare_client.exceptions.NoSuchKey:
        bs_logger.error("File not found in R2 bucket. Please check the file key and bucket name.")
        exit()
    except Exception as e:
        bs_logger.error(f"An error occurred: {e}")
        exit()

def delete_file(file_key):
    try:
        cloudflare_client.delete_object(Bucket=bucket_name, Key=file_key)
    except cloudflare_client.exceptions.NoSuchKey:
        bs_logger.error("File not found in R2 bucket. Please check the file key and bucket name.")
        exit()
    except Exception as e:
        bs_logger.error(f"An error occurred: {e}")
        exit()