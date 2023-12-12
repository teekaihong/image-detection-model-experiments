import os
import boto3
from tqdm import tqdm
from boto3.s3.transfer import S3Transfer
from dotenv import load_dotenv

load_dotenv(override=True)

s3_client = boto3.client(
    's3',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
)

s3_tr_config = boto3.s3.transfer.TransferConfig(
    max_concurrency=10,
    use_threads=True
)

def list_all_objects(s3_client, bucket_name, prefix):
    paginator = s3_client.get_paginator('list_objects_v2')
    paginated_response = paginator.paginate(Bucket=bucket_name, Prefix=prefix, StartAfter=prefix)
    response_list = [x for x in tqdm(paginated_response, desc='listing s3 objects')]
    # check if response_list is empty
    if 'Contents' not in response_list[0]:
        return []
    s3_files = [contents for response in response_list for contents in response['Contents']]
    return s3_files
