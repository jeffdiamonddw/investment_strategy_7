
import argparse
import joblib
import os
import time


import sys
import boto3
import logging

import functools
import numpy as np
import pandas as pd
import awswrangler as wr
import logging
import s3fs


from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.population import Population

from regime_navigator_1d import get_rn_problem_params
from simulate_stock_rotation import simulate
from objective_functions import mean_annualized_return, weighted_quantile, weighted_mean, WeightedRegretApplyer, WeightedRegimeApplyer, apply_objectives
from regime_navigator_1d import get_rn_problem_params, RegimeNavigator1D
from utils import get_dna_hash, save_to_zarr




from pymoo.config import Config
Config.warnings['not_compiled'] = False

import zarr
import xarray as xr

from utils import write_to_s3
from smart_open import open as smart_open





TICKER_FILE = "strategy/multi_dim_stock_list.csv"
OUTPUT_FOLDER = 's3://jdinvestment/2d_test_1'








# --- 2. SUBCLASS THE WRAPPER PROBLEM ---
class RegimeNavigator2D(RegimeNavigator1D):
    """
    Inherits from RegimeNavigatorProblem but swaps out self.base 
    
    
    Uses programmatic introspection to filter and forward exact keyword 
    arguments to constructors, entirely eliminating argument order errors.
    """
    def __init__(
            self, 
            mom_kit, 
            qual_kit, 
            df_macro, 
            data_features, 
            df_price, 
            params, 
            periods, 
            output_folder, 
            holdings=None,
            objective_functions_dict = None,
            objective_sense = None
        ):
        
       self.__dict__.update({k: v for k, v in locals().items() if k != 'self'})


    
       


    def run_simulation(self, w_mom_vals, w_qual_vals, threshold, beta, mom_decay, qual_decay, df_macro_weights, max_voo, period_key, sim_id, session=None, holdings=None):
        
        
        macro_weights_risk = df_macro_weights.values.flatten()[:4]
        macro_weights_temporal = df_macro_weights.values.flatten()[4:]
        
        macro_weights_risk /= macro_weights_risk.sum()
        macro_weights_temporal /= macro_weights_temporal.sum()
        period = self.periods[period_key]
        
        keep_date = (period['val_start_date'] <= self.df_macro.index) & (self.df_macro.index <= period['end_date'])
        df_macro = self.df_macro.loc[keep_date]

        s_risk_aversion_full = self.df_macro.dot(macro_weights_risk).rename("risk_aversion") 
        risk_aversion_mean = s_risk_aversion_full.mean()
        s_risk_aversion = s_risk_aversion_full.loc[
            (self.df_macro.index >= period['val_start_date']) & (self.df_macro.index <= period['end_date'])
        ]  

        s_temporal_full = self.df_macro.dot(macro_weights_temporal).rename("temporal") 
        temporal_mean = s_temporal_full.mean()
        s_temporal = s_temporal_full.loc[
            (self.df_macro.index >= period['val_start_date']) & (self.df_macro.index <= period['end_date'])
        ]  
        
        
        s_quality_weight = 1 / (1 + (-beta * (s_risk_aversion - threshold)).astype(float).apply(np.exp))
        mom_num_periods = np.array([int(col.split('_')[-1][:-1]) for col in self.mom_kit['columns']])
        qual_num_periods = np.array([int(col.split('_')[-1][:-1]) for col in self.qual_kit['columns']])
        
        df_mom_decay = pd.DataFrame(np.exp((- mom_decay * (risk_aversion_mean - s_risk_aversion).values[:, None] * mom_num_periods).astype(float)), index=s_risk_aversion.index, columns = self.mom_kit['columns'])
        df_qual_decay = pd.DataFrame(np.exp((- qual_decay * (risk_aversion_mean - s_risk_aversion).values[:, None] * qual_num_periods).astype(float)), index=s_risk_aversion.index, columns = self.qual_kit['columns'])
        
        
        w_dict = {}
        df_mom = pd.DataFrame({self.mom_kit['columns'][j]: [w_mom_vals[j]] for j in range(len(w_mom_vals))})
        df_qual = pd.DataFrame({self.qual_kit['columns'][j]: [w_qual_vals[j]] for j in range(len(w_mom_vals))})
        
        df_mom_weights = df_mom_decay.mul(df_mom.iloc[0], axis=1).mul(1 - s_quality_weight, axis=0)
        df_qual_weights = df_qual_decay.mul(df_qual.iloc[0], axis=1).mul(s_quality_weight, axis=0)
        df_weights = pd.concat([df_mom_weights, df_qual_weights], axis = 1)
        #df_weights = df_weights.div(df_weights.sum(axis = 1), axis = 0)
        df_weights.to_csv('temp/weights.csv')
       #********************************************************************************************************************************* 

        df_holdings = simulate(self.df_price, self.params, self.data_features, df_weights, period,  sim_id, session = session, holdings = holdings)
        
      
        return df_holdings
    

    def evaluate(self, x):
        
        x_numeric = x.X if hasattr(x, "X") else x
        sim_id = get_dna_hash(x_numeric)
        # CHANGE self.mom_kit to self.mom_kit
        w_mom = self.mom_kit['scaler'].inverse_transform(
            self.mom_kit['pca'].inverse_transform(x_numeric[0:4].reshape(1, -1))
        ).flatten()
        
        # CHANGE self.qual_kit to self.qual_kit
        w_qual = self.qual_kit['scaler'].inverse_transform(
            self.qual_kit['pca'].inverse_transform(x_numeric[4:8].reshape(1, -1))
        ).flatten()
        w_mom, w_qual = np.clip(w_mom, 0, 1), np.clip(w_qual, 0, 1)

        opt_threshold = x_numeric[8]
        opt_beta = x_numeric[9]
        mom_decay = x_numeric[10]
        qual_decay = x_numeric[11]
        df_macro_weights = pd.DataFrame(x_numeric[12:20].reshape(2,4), index = ['risk_weights', 'temporal_weights'], columns = self.df_macro.columns)
        max_voo = x_numeric[20]

        
        
        df_sim = pd.DataFrame()
        for key in self.periods:
            df_period = self.run_simulation(w_mom, w_qual, opt_threshold, opt_beta, mom_decay, qual_decay, df_macro_weights, max_voo, key, sim_id, holdings = self.holdings)
            df_sim = pd.concat([df_sim, df_period]).reset_index(drop=True)
        
        
        
        ticker_cols = [c for c in df_sim.columns if c not in ['date', 'sim_id'] ]
        total_value_series = df_sim.set_index('date')[ticker_cols].apply(pd.to_numeric, errors='coerce').sum(axis=1).rename('value')
        
        
        
        return df_sim, total_value_series
        
        





if __name__ == "__main__":
    
    num_samples = 3
    perturbation_cv = .01
    
    t1 = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--s3_path', required=True)
    parser.add_argument('--generation', type=int, required=True)
    parser.add_argument('--train_folds', type=int, nargs='+', default=[],
                        help='List of training fold integers (e.g., --train_fold 0 1 2)')
    parser.add_argument('--val_folds', type=int, nargs='+', default=[],
                        help='List of validation fold integers (e.g., --val_fold 3 4)')
    args = parser.parse_args()

    s3_path = args.s3_path
    generation = args.generation
    task_index = int(os.environ.get('AWS_BATCH_JOB_ARRAY_INDEX', 0))
   
    
    logging.getLogger('botocore.credentials').setLevel(logging.WARNING)
    my_boto3_session = boto3.Session()
    s3 = s3fs.S3FileSystem(session=my_boto3_session)

    periods = {
        'train': {'train_start_date': pd.to_datetime('2006-01-01'), 'val_start_date': pd.to_datetime('2008-01-01'), 'end_date': pd.to_datetime('2026-06-06')},
    }
    
    
    df_folds = pd.read_parquet('strategy/folds.parquet')
    
 
    
    df_train_folds = df_folds.loc[df_folds.fold_index.isin(args.train_folds)]
    df_val_folds = df_folds.loc[df_folds.fold_index.isin(args.val_folds)]

    weighting_func_quantile = functools.partial(weighted_quantile, .1)
    start_date = min(df_folds.start_date)
    end_date = max(df_folds.end_date)
    agg_func = functools.partial(mean_annualized_return, start_date, end_date)

    
  

    objective_functions_dict = {
        'train': {
            'regret_quantile': WeightedRegretApplyer(df_train_folds, agg_func, weighting_func_quantile, 'max_annualized_return', 'max'),
            'mean_regret' : WeightedRegretApplyer(df_train_folds, agg_func, weighted_mean, 'max_annualized_return', 'max'),
            'quantile': WeightedRegimeApplyer(df_train_folds, agg_func, weighting_func_quantile),
            'mean': WeightedRegimeApplyer(df_train_folds, agg_func, weighted_mean)
        },
        'val': {
            'regret_quantile': WeightedRegretApplyer(df_val_folds, agg_func, weighting_func_quantile, 'max_annualized_return', 'max'),
            'mean_regret' : WeightedRegretApplyer(df_val_folds, agg_func, weighted_mean, 'max_annualized_return', 'max'),
            'quantile': WeightedRegimeApplyer(df_val_folds, agg_func, weighting_func_quantile),
            'mean': WeightedRegimeApplyer(df_val_folds, agg_func, weighted_mean)
        }
    }
   

    
    objective_sense = {'regret_quantile': 'min', 'mean_regret': 'min'}
    

    principal = [408000]
    
    
    params = {
            'principal': principal, 'max_frac': .05, 'feature_horizon_weeks': 104,
            'min_price': 5, 'trade_fee': 7, 'objective_sensitivity': 0.144, 'obj_threshold': 0,
            'start_date': pd.to_datetime('Jan 1, 2005'), 'end_date': pd.Timestamp.now()
        }
    

   
    holdings = None
    
    problem_args = get_rn_problem_params(
        periods,
        momentum_file = "s3://jdinvestment/simulation_data/momentum.nc", 
        quality_file = "s3://jdinvestment/simulation_data/quality.nc",
        gic_file = "s3://jdinvestment/simulation_data/gic_data.nc",
        macro_file = "s3://jdinvestment/simulation_data/macro_signals.parquet",
        manifold_file = "s3://jdinvestment/sim_results/manifold_triple_threat.csv",
        output_folder = None,
        params = params,
        holdings = holdings


    )  
    
   

    regime_navigator = RegimeNavigator2D(**problem_args)

    s_task = pd.read_csv("{}/populations/gen_{}.csv".format(s3_path, generation)).iloc[task_index]
    parameter_cols = [name for name in s_task.index if 'sim_id' not in name]
    x = s_task.loc[parameter_cols].values
    sim_id = get_dna_hash(x)
    print('pre-time: {}'.format(time.time() - t1))
    
    parent_sim_id = sim_id
    perturbed_x = x
    df_evaluations = pd.DataFrame()
    for sample in range(num_samples):
        sim_id = get_dna_hash(perturbed_x)
        df_history, total_value_series = regime_navigator.evaluate(perturbed_x)
    
        logging.getLogger('botocore.credentials').setLevel(logging.WARNING)
        my_boto3_session = boto3.Session()
        
        wr.s3.to_parquet(
                df=df_history,
                path="{}/holdings/sim_{}.parquet".format(s3_path, sim_id),
                dataset=False,
                index = True,
                boto3_session=my_boto3_session 
        )
        
        df_values = pd.DataFrame(total_value_series).transpose()
        df_values.index = [sim_id]
        df_values['parent_sim_id'] = parent_sim_id
        df_values['generation'] = generation
        df_values.columns = df_values.columns.astype(str)


        wr.s3.to_parquet(
                df=df_values,
                path='{}/portfolio_values/sim_{}.parquet'.format(s3_path, sim_id),
                dataset=False,
                index = True,
                boto3_session=my_boto3_session 
        )

        
        
        df_evaluation = apply_objectives(objective_functions_dict, total_value_series)
        df_evaluation['sim_id'] = sim_id
        df_evaluation['parent_sim_id'] = parent_sim_id
        wr.s3.to_parquet(
                df=df_evaluation,
                path='{}/objectives/sim_{}.parquet'.format(s3_path, sim_id),
                dataset=False,
                index = True,
                boto3_session=my_boto3_session 
        )
        df_evaluations = pd.concat([df_evaluations, df_evaluation])

        noise = np.random.normal(0, perturbation_cv * np.abs(x), size=x.shape)
        perturbed_x = x + noise

    df_agg = df_evaluations[['mode', 'objective', 'value']].groupby(['mode', 'objective']).agg('median')
    df_agg['sim_id'] = parent_sim_id
    wr.s3.to_parquet(
            df=df_agg,
            path='{}/median_objectives/sim_{}.parquet'.format(s3_path, sim_id),
            dataset=False,
            index = True,
            boto3_session=my_boto3_session 
    )
        
    
    


     
