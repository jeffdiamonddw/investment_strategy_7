import functools
import logging
import joblib
import numpy as np
import pandas as pd

import boto3
from utils import get_dna_hash


OUTPUT_FOLDER = 's3://jdinvestment/2d_test_1'
N_SAMPLES = 3
PERTURBATION_CV = .01


if __name__ == "__main__":
    
    logging.getLogger('botocore.credentials').setLevel(logging.WARNING)
    my_boto3_session = boto3.Session()

   
    
    df_initial = pd.read_parquet('s3://jdinvestment/sim_results/pareto_initial_pop_for_2d.parquet').iloc[:, 6:]
    df_initial.columns = [col.replace('macro_weights', 'risk_macro_weights') for col in df_initial.columns]
    df_new_weights = df_initial[[col for col in df_initial.columns if 'macro' in col]]\
        .rename(columns = {col: col.replace('risk', 'temporal') for col in df_initial.columns if 'macro' in col})
    df_initial = pd.concat([df_initial, df_new_weights], axis = 1)
    df_initial['max_voo'] = .05

    parent_sim_ids = df_initial.apply(get_dna_hash, axis = 1)
    

    df_out = pd.DataFrame()
    for i in range(N_SAMPLES - 1):
            # Apply Gaussian perturbation based on input CV[cite: 1]
            noise = np.random.normal(0, PERTURBATION_CV * np.abs(df_initial.values), size=df_initial.shape)
            df_add = df_initial + noise
            df_add['parent_sim_id'] = parent_sim_ids
            df_out = pd.concat([df_out, df_add])
    df_initial['parent_sim_id'] = parent_sim_ids
    df_out = pd.concat([df_initial, df_out])
    df_out.to_csv("{}/populations/gen_0.csv".format(OUTPUT_FOLDER))
            
           
            
            



    