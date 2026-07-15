import argparse
import os
import boto3
import sys
import pandas as pd
import time

def get_array_status_summary(parent_job_id):
    batch = boto3.client('batch')
    
    response = batch.describe_jobs(jobs=[parent_job_id])
    
    if not response['jobs']:
        return "Job not found."
    
    job_detail = response['jobs'][0]
    
    # Check if it's actually an array job
    if 'arrayProperties' in job_detail and 'statusSummary' in job_detail['arrayProperties']:
        return job_detail['arrayProperties']['statusSummary']
    else:
        return "No status summary found (it might not be a parent array job)."


def batch_complete(parent_job_id):
    summary = get_array_status_summary(parent_job_id)
    done_count = summary['SUCCEEDED'] + summary['FAILED']
    total_count = sum(list(summary.values()))
    return done_count == total_count


import boto3
from urllib.parse import urlparse

def s3_folder_exists(s3_full_path):
    """
    Checks if an S3 folder exists given a full 's3://bucket/path/to/folder' string.
    """
    # Parse the S3 URL
    parsed = urlparse(s3_full_path)
    if parsed.scheme != 's3':
        raise ValueError("Path must start with s3://")
    
    bucket_name = parsed.netloc
    prefix = parsed.path.lstrip('/')
    
    # Ensure prefix ends with a slash to avoid partial matches
    if prefix and not prefix.endswith('/'):
        prefix += '/'
        
    s3 = boto3.client('s3')
    
    # List objects with the prefix, limit to 1 for performance
    response = s3.list_objects_v2(
        Bucket=bucket_name,
        Prefix=prefix,
        MaxKeys=1
    )
    
    return 'Contents' in response


import boto3
from urllib.parse import urlparse

def list_s3_files(s3_full_path):
    """
    Returns a list of full s3:// paths for all files in the given S3 folder.
    """
    parsed = urlparse(s3_full_path)
    if parsed.scheme != 's3':
        raise ValueError("Path must start with s3://")
    
    bucket_name = parsed.netloc
    prefix = parsed.path.lstrip('/')
    
    # Ensure prefix ends with a slash if it's meant to be a folder
    if prefix and not prefix.endswith('/'):
        prefix += '/'
        
    s3 = boto3.client('s3')
    paginator = s3.get_paginator('list_objects_v2')
    
    file_list = []
    
    # Paginator handles the "next token" logic automatically
    for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
        if 'Contents' in page:
            for obj in page['Contents']:
                # Construct the full s3:// path for each object
                file_path = f"s3://{bucket_name}/{obj['Key']}"
                file_list.append(file_path)
                
    return file_list





def run_batch_array(s3_path, generation, train_folds, val_folds):

    output_path = "{}/median_objectives/gen_{}".format(s3_path, generation)
    if not s3_folder_exists(output_path): 
        df_tasks = pd.read_csv("{}/populations/gen_{}.csv".format(s3_path, generation))
        array_size = df_tasks.shape[0]

        batch = boto3.client('batch', region_name='us-west-2')
        job_def_name = "simulation-job-def"

        # 1. Define resources within the script
        # This replaces the need for the external job-definition.json
        print(f"Registering/Updating Job Definition: {job_def_name}...")
        batch.register_job_definition(
            jobDefinitionName=job_def_name,
            type='container',
            containerProperties={
                'image': '129861351772.dkr.ecr.us-west-2.amazonaws.com/simulation:latest',
                'vcpus': 1,
                'memory': 1024,
                'jobRoleArn': 'arn:aws:iam::129861351772:role/ecsTaskExecutionRole',
                'executionRoleArn': 'arn:aws:iam::129861351772:role/ecsTaskExecutionRole',
                
            }
        )

        # 2. Submit the job
        cmd = [
            "python3", "python_scripts/regime_navigator_2d.py",
            "--s3_path", s3_path,
            "--generation", str(generation),
            "--train_folds", *map(str, train_folds),
            "--val_folds", *map(str, val_folds)
        ]

        print("Submitting array job...")
        response = batch.submit_job(
            jobName=f"sim-gen-{generation}",
            jobQueue="batch-arm-192-queue",
            jobDefinition=job_def_name,
            arrayProperties={'size': array_size},
            containerOverrides={'command': cmd}
        )

        batch_id = response['jobId']
        print(f"Successfully submitted! Job ID: {batch_id}")
        while not batch_complete(batch_id):
            time.sleep(15)
    else:
        print("output already exists")


def main():
    parser = argparse.ArgumentParser()
    # ... (Keep your existing arguments)
    parser.add_argument('--s3_path', required=True)
    parser.add_argument('--train_folds', type=int, nargs='+', default=[])
    parser.add_argument('--val_folds', type=int, nargs='+', default=[])
    args = parser.parse_args()

    for generation in range(2):
        run_batch_array(args.s3_path, generation, args.train_folds, args.val_folds)
        output_path = "{}/median_objectives/gen_{}".format(args.s3_path, generation)

        df = pd.DataFrame()
        for s3_path in ["gen_0/{}".format(file) for file in os.listdir('gen_0')]: #list_s3_files(output_path):
            df_add = pd.read_parquet(s3_path)
            df = pd.concat([df, df_add])
        
            
        df_obj = pd.pivot_table(df.reset_index(), values = 'value', index = 'sim_id', columns = ['mode', 'objective'])
        df_obj.columns = ['_'.join(col) for col in df_obj.columns]
        df_opt_obj = df_obj[['train_mean_regret', 'train_regret_quantile']]

        df_tasks = pd.read_parquet("{}/populations/gen_{}.parquet".format(args.s3_path, generation))
        df_pop = df_opt_obj.join(df_tasks, how = 'inner')

        zzz=1
           



    

if __name__ == "__main__":
    main()