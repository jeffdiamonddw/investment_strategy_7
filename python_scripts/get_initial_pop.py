import os

import pandas as pd
import numpy as np
import xarray as xr
import joblib
from scipy.stats import spearmanr
from paretoset import paretoset
import functools

from objective_functions import mean_annualized_return, weighted_quantile, weighted_mean, WeightedRegretApplyer, WeightedRegimeApplyer, apply_objectives
from utils import get_dna_hash


def get_pareto_layers(df, sense,  num_layers):
    df_out = df.copy()
    df_out['layer'] = num_layers 
    df_current = df.copy()
    
    
    for layer_idx in range(num_layers):
        # paretoset returns a boolean mask for the current non-dominated front
        mask = paretoset(df_current.values, sense=sense)
        layer_ids = df_current.index[mask]
        other_ids = df_current.index[~mask] # Define sense (max/min) for your objectives
        
        # Store the current layer
        df_out.loc[layer_ids, 'layer'] = layer_idx
        
        # Remove these points and move to the next layer
        df_current = df_current.loc[other_ids]
        
    return df_out   

if not os.path.isfile('sim_results/best_pop_1d.parquet'):
    
    df_history = pd.read_parquet('s3://jdinvestment/median_17_year_holdings.parquet').set_index('sim_id')
    df_history.columns = pd.to_datetime(df_history.columns)
    df_params = pd.read_parquet('s3://jdinvestment/evaluations.parquet').set_index('sim_id')
    df_folds = pd.read_parquet('strategy/folds.parquet')

    train_folds = [1,2,3]
    df_train_folds = df_folds.loc[df_folds.fold_index.isin(train_folds)]

    weighting_func_quantile = functools.partial(weighted_quantile, .1)
    start_date = min(df_folds.start_date)
    end_date = max(df_folds.end_date)
    agg_func = functools.partial(mean_annualized_return, start_date, end_date)

    regret_quantile_func =  WeightedRegretApplyer(df_train_folds, agg_func, weighting_func_quantile, 'max_annualized_return', 'max')
    mean_regret_func =  WeightedRegretApplyer(df_train_folds, agg_func, weighted_mean, 'max_annualized_return', 'max')


    regret_quantile = df_history.apply(regret_quantile_func, axis = 1)
    mean_regret = df_history.apply(mean_regret_func, axis = 1)
    df_obj = pd.DataFrame({'regret_quantile': regret_quantile, 'mean_regret': mean_regret}, index = df_history.index)
    df_obj.to_parquet('sim_results/median_17_year_regime_objectives.parquet')


    num_initial = 210
    df_best = get_pareto_layers(df_obj, sense = ['min', 'min'], num_layers = 5).sort_values(by = 'layer').iloc[:num_initial]
    df_initial = df_params.loc[df_best.index].iloc[:, -16:]
    df_initial.to_parquet('sim_results/best_pop_1d.parquet')
    df_pop = df_initial
else:
    df_pop = pd.read_parquet('sim_results/best_pop_1d.parquet')

df_pop = df_pop.rename(columns = {col : col.replace('macro_weights', 'risk_macro_weights') for col in df_pop.columns if 'macro_weights' in col})
df_add = df_pop.loc[:, [col for col in df_pop.columns if 'macro_weights' in col]].rename(columns = {col: col.replace('risk', 'temporal') for col in df_pop.columns if 'macro_weights' in col})
df_initial = pd.concat([df_pop, df_add], axis = 1)
df_initial['max_voo'] = .05
df_initial.index = df_initial.apply(get_dna_hash, axis = 1)
df_initial.to_parquet('sim_results/initial_pop_2d.parquet')
df_initial.to_parquet('s3://jdinvestment/2d_test_2/populations/gen_0.parquet')


