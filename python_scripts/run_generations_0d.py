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
from pymoo.algorithms.moo.nsga2 import NSGA2
from apsa_ngsa2 import APSANGSA2
from surrogate_models import FastStackedSurrogate, HeterogeneousEnsemble, SurrogateProblem





from utils import s3_file_exists, s3_folder_exists, s3_list_files




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
    time.sleep(1)
    summary = get_array_status_summary(parent_job_id)
    while not ( hasattr(summary, '__len__') and len(summary)>0):
        print('waiting for summary', flush = True)
        summary = get_array_status_summary(parent_job_id)
        time.sleep(1)
    total_count = sum(list(summary.values()))
    while total_count < num_jobs:
        print('waiting for total count to add up {}/{}'.format(total_count, num_jobs), flush = True)
        summary = get_array_status_summary(parent_job_id)
        total_count = sum(list(summary.values()))
        time.sleep(1)
    
    done_count = sum([summary[key] for key in ['SUCCEEDED', 'FAILED'] if key in summary])
    
    return done_count == total_count, done_count


import time
import boto3

def wait_for_batch_job(job_id):
    """
    Waits for an AWS Batch array job to reach a terminal state.
    """
    client = boto3.client('batch')
    
    while True:
        response = client.describe_jobs(jobs=[job_id])
        if not response['jobs']:
            raise Exception(f"Job {job_id} not found.")
            
        job = response['jobs'][0]
        status = job.get('status')
        
        # 'SUCCEEDED' or 'FAILED' are terminal states for the parent job
        # Note: If the parent job status is 'FAILED', it means the job couldn't be scheduled.
        # Array jobs themselves usually reach 'SUCCEEDED' once all children are processed,
        # regardless of whether the children succeeded or failed.
        if status in ['SUCCEEDED', 'FAILED']:
            print(f"Job {job_id} reached terminal state: {status}")
            break
            
        print(f"Job {job_id} status is {status}... waiting 30s")
        time.sleep(30)

    # Return status summary for your logs
    return job.get('arrayProperties', {}).get('statusSummary', {})














def run_batch_array(image_arn, s3_path, generation, train_folds, val_folds):

    output_path = "{}/median_objectives/gen={}".format(s3_path, generation)
    if not s3_folder_exists(output_path): 
        df_tasks = pd.read_parquet("{}/populations/gen_{}.parquet".format(s3_path, generation))
        num_jobs = array_size = df_tasks.shape[0]

        batch = boto3.client('batch', region_name='us-west-2')
        job_def_name = "simulation-job-def"

        # 1. Define resources within the script
        # This replaces the need for the external job-definition.json
        print(f"Registering/Updating Job Definition: {job_def_name}...", flush = True)
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

        print("Submitting array job...", flush = True)
        response = batch.submit_job(
            jobName=f"sim-gen-{generation}",
            jobQueue="batch-arm-192-queue",
            jobDefinition=job_def_name,
            arrayProperties={'size': array_size},
            containerOverrides={'command': cmd}
        )

        batch_id = response['jobId']
        print(f"Successfully submitted! Job ID: {batch_id} for generation {generation}", flush = True)
        t1 = time.time() 
        num_complete = batch_complete(batch_id, num_jobs)
        print('batch complete: {}'.format(num_complete), flush = True)
        while not batch_complete(batch_id, num_jobs)[0]:
            time.sleep(30)
            num_complete = batch_complete(batch_id, num_jobs)[1]
            print('waiting on batch {} seconds, {}/{} complete'.format(time.time() - t1, num_complete, num_jobs), flush = True)
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
       'avg_eps_1q', 'avg_eps_2q', 'avg_eps_4q', 'avg_eps_8q',  'max_voo'
    ]
    image_arn = "129861351772.dkr.ecr.us-west-2.amazonaws.com/simulation:latest"
    
    # df_initial = pd.read_parquet('sim_results/initial_pop_2d.parquet')
    # s3_pop_file = "{}/populations/gen_0.parquet".format(args.s3_path)
    # if not s3_file_exists(s3_pop_file):
    #     df_initial.to_parquet(s3_pop_file)
    # num_vars = df_initial.shape[1]
    
    # Indices: 0-7: PCA, 8: Threshold, 9: Beta, 10-11: Decay, 12-15: Macro Weights
    xl= np.array([
        0, 0, 0, 0,  # Mom PCA
        0, 0, 0, 0,  # Qual PCA
        .05              #max_voo
    ])

    xu = np.array([
        1, 1, 1, 1,      # Mom PCA
        1, 1, 1, 1,      # Qual PCA     
        .6                 #max_voo
    ])

    num_vars = len(xl)
    master_problem = BatchArrayProblem(image_arn, args.s3_path, args.train_folds, args.val_folds, param_names, xl, xu)
    
    

    algorithm = NSGA2(pop_size = 20)
    algorithm.setup(master_problem)
    
    

    
    for gen in range(3):
        
        
        
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