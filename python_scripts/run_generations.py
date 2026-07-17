import argparse
import os
import boto3
import sys
import pandas as pd
import time

import numpy as np
from pymoo.core.problem import Problem
import awswrangler as wr

from utils import get_dna_hash
from pymoo.util.display.multi import MultiObjectiveOutput
from pymoo.core.population import Population
from apsa_ngsa2 import APSANGSA2
from surrogate_models import FastStackedSurrogate, HeterogeneousEnsemble, SurrogateProblem

import boto3
from botocore.exceptions import ClientError
from urllib.parse import urlparse

def s3_file_exists(s3_path: str) -> bool:
    """
    Checks if a file exists at the given S3 path.
    
    :param s3_path: The full S3 path (e.g., 's3://my-bucket/path/to/file.txt')
    :return: True if exists, False otherwise.
    """
    # Parse the S3 URI
    parsed = urlparse(s3_path)
    bucket_name = parsed.netloc
    key = parsed.path.lstrip('/')
    
    s3_client = boto3.client('s3')
    
    try:
        s3_client.head_object(Bucket=bucket_name, Key=key)
        return True
    except ClientError as e:
        # If the error code is 404, the file does not exist
        if e.response['Error']['Code'] == "404":
            return False
        else:
            # If it's a different error (e.g., 403 Forbidden), re-raise it
            raise e


def get_array_status_summary(parent_job_id):
    batch = boto3.client('batch')
    
    response = batch.describe_jobs(jobs=[parent_job_id])
    
    if not response['jobs']:
        return {}
    
    job_detail = response['jobs'][0]
    
    # Check if it's actually an array job
    if 'arrayProperties' in job_detail and 'statusSummary' in job_detail['arrayProperties']:
        return job_detail['arrayProperties']['statusSummary']
    else:
        return {}


def batch_complete(parent_job_id, num_jobs):
    summary = {}
    while len(summary) == 0: 
        summary = get_array_status_summary(parent_job_id)
    total_count = sum(list(summary.values()))
    while total_count < num_jobs:
        total_count = sum(list(summary.values()))
    
    done_count = sum([summary[key] for key in ['SUCCEEDED', 'FAILED'] if key in summary])
    
    return done_count == total_count, done_count


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





def run_batch_array(image_arn, s3_path, generation, train_folds, val_folds):

    output_path = "{}/median_objectives/gen_{}".format(s3_path, generation)
    if not s3_folder_exists(output_path): 
        df_tasks = pd.read_parquet("{}/populations/gen_{}.parquet".format(s3_path, generation))
        num_jobs = array_size = df_tasks.shape[0]

        batch = boto3.client('batch', region_name='us-west-2')
        job_def_name = "simulation-job-def"

        # 1. Define resources within the script
        # This replaces the need for the external job-definition.json
        print(f"Registering/Updating Job Definition: {job_def_name}...")
        batch.register_job_definition(
            jobDefinitionName=job_def_name,
            type='container',
            containerProperties={
                'image': image_arn,
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
        print(f"Successfully submitted! Job ID: {batch_id} for generation {generation}")
        t1 = time.time() 
        print('batch complete: {}'.format(batch_complete(batch_id, num_jobs)), flush = True)
        while not batch_complete(batch_id, num_jobs)[0]:
            time.sleep(30)
            num_complete = batch_complete(batch_id, num_jobs)[1]
            print('waiting on batch {} seconds, {}/{} complete'.format(time.time() - t1, num_complete, num_jobs))
        print(f"completed Job ID: {batch_id} for generation {generation}", flush = True)
    else:
        print("output already exists for generation {}".format(generation), flush = True)
    

def get_objectives(s3_path, generation, obj_columns = ['train_mean_regret', 'train_regret_quantile']):
    output_path = "{}/median_objectives/gen_{}/".format(s3_path, generation)
    df = wr.df = wr.s3.read_parquet(
        path=output_path,
        dataset=True
    )
   
    df_obj = pd.pivot_table(df.reset_index(), values = 'value', index = 'sim_id', columns = ['mode', 'objective'])
    df_obj.columns = ['_'.join(col) for col in df_obj.columns]
    

    return df_obj[obj_columns]


class BatchArrayProblem(Problem):

    def __init__(self,  image_arn, s3_path, train_folds, val_folds, param_names, xl, xu):
        
        self.__dict__.update({k: v for k, v in locals().items() if k != 'self'})
        self.generation = 0

        super().__init__(n_var = len(param_names), n_obj = 2, xl=self.xl, xu=self.xu, elementwise_evaluation=False)
        

    def _evaluate(self, x, out, *args, **kwargs):
        df_tasks = pd.DataFrame(x, columns = self.param_names)
        df_tasks.index = df_tasks.apply(get_dna_hash, axis = 1)
        
        df_tasks.to_parquet("{}/populations/gen_{}.parquet".format(self.s3_path, self.generation))
        run_batch_array(self.image_arn, self.s3_path, self.generation, self.train_folds, self.val_folds)
        df_obj = get_objectives(self.s3_path, self.generation)

        df_pop = df_tasks[[]].join(df_obj, how = 'left')
        out["F"] = df_pop.values




def main():
    parser = argparse.ArgumentParser()
    # ... (Keep your existing arguments)
    parser.add_argument('--s3_path', required=True)
    parser.add_argument('--train_folds', type=int, nargs='+', default=[])
    parser.add_argument('--val_folds', type=int, nargs='+', default=[])
    args = parser.parse_args()

    param_names = [
        'dollar_ret_1p', 'dollar_ret_6p', 'dollar_ret_13p', 'dollar_ret_26p',
       'avg_eps_1q', 'avg_eps_2q', 'avg_eps_4q', 'avg_eps_8q', 'threshold',
       'beta', 'mom_decay', 'qual_decay', 'risk_macro_weights_0',
       'risk_macro_weights_1', 'risk_macro_weights_2', 'risk_macro_weights_3',
       'temporal_macro_weights_0', 'temporal_macro_weights_1',
       'temporal_macro_weights_2', 'temporal_macro_weights_3', 'max_voo'
    ]
    image_arn = "129861351772.dkr.ecr.us-west-2.amazonaws.com/simulation:latest"
    
    df_initial = pd.read_parquet('sim_results/initial_pop_2d.parquet')
    s3_pop_file = "{}/populations/gen_0.parquet".format(args.s3_path)
    if not s3_file_exists(s3_pop_file):
        df_initial.to_parquet('s3_pop_file')
    num_vars = df_initial.shape[1]
    
    # Indices: 0-7: PCA, 8: Threshold, 9: Beta, 10-11: Decay, 12-15: Macro Weights
    xl= np.array([
        -2, -2, -2, -2,  # Mom PCA
        -2, -2, -2, -2,  # Qual PCA
        -2.0,            # Threshold (Index 8: expanded from 0.1)
        0.5,             # Beta (Index 9)
        -1, -1,          # Decays
        -1, -1, -1, -1,   # risk Macro Weights
        -1, -1, -1, -1,  # temporal Macro Weights
        .05              #max_voo
    ])

    xu = np.array([
        2, 2, 2, 2,      # Mom PCA
        2, 2, 2, 2,      # Qual PCA
        2.0,             # Threshold (Index 8: expanded from 0.9)
        15.0,            # Beta (Index 9: expanded from 2.0)
        1, 1,            # Decays
        1, 1, 1, 1,       # risk Macro Weights
        1, 1, 1, 1,       # temporal Macro Weights
        .6                 #max_voo
    ])


    master_problem = BatchArrayProblem(image_arn, args.s3_path, args.train_folds, args.val_folds, param_names, xl, xu)
    models = [FastStackedSurrogate(num_vars) for i in range(2)]
    ensemble = HeterogeneousEnsemble(models)
    surrogate_problem = SurrogateProblem(ensemble, master_problem)

    algorithm = APSANGSA2(
        pop_size=210,
        n_infills = 215,
        surr_n_gen=30,
        surr_thresholds = {'min_point': .1, 'max_point': .01},
        surr_eps_elim=1e-6,
        output=MultiObjectiveOutput(),
        surr_tolerance = .1,
        surrogate_problem = surrogate_problem,
        master_problem = master_problem,
        surrogate_memory_generations = 3
    )
    
    
    

    
    for gen in range(150):
        
        
        
        master_problem.generation = gen
        
        task_path = "{}/populations/gen_{}.parquet".format(args.s3_path, gen)
        if not s3_file_exists(task_path):
            pop = algorithm.ask()
        else:
            df_tasks = pd.read_parquet(task_path)
            X = df_tasks.values
            pop = Population.new("X", X) 

        
        F = master_problem.evaluate(pop.get("X"))
        valid = ~np.isnan(F).any(axis=1)
        
        valid_pop = pop[valid]
        valid_pop.set("F", F[valid])
        
        algorithm.tell(infills=valid_pop, gen = gen)
        
    
           



    

if __name__ == "__main__":
    main()