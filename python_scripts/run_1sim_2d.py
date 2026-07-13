import functools
import time, random
import os

import numpy as np
import pandas as pd
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.population import Population
from pymoo.operators.sampling.rnd import FloatRandomSampling




from objective_functions import  mean_annual_drawdown_integral, mean_annualized_return, pct_change_quantile, worst_annual_return
from regime_navigator_2d import get_rn_problem_params, RegimeNavigator2D

  



TICKER_FILE = "strategy/multi_dim_stock_list.csv"
HOLDINGS_FOLDER = "s3://jdinvestment/median_17_year_holdings"
EVAL_FOLDER = "s3://jdinvestment/median_17_year_evaluations"
INITIAL_POP_FILE = "sim_results/max_annualized_return.csv"
OUTPUT_FOLDER = "s3://jdinvestment/median_17_year_1sim"





if __name__ == "__main__":

    df_pop = pd.read_csv(INITIAL_POP_FILE)
    #df_pop = df_starting_pop.loc[df_starting_pop['rank'] == 0]


    weight_cols = [
        'dollar_ret_1p', 'dollar_ret_6p', 'dollar_ret_13p',
       'dollar_ret_26p', 'avg_eps_1q', 'avg_eps_2q', 'avg_eps_4q',
       'avg_eps_8q', 'threshold', 'beta', 'mom_decay', 'qual_decay',
       'macro_weights_0', 'macro_weights_1', 'macro_weights_2',
       'macro_weights_3'
    ]

    X_initial_1d =[float(x) for x in df_pop[weight_cols].values.flatten()]
    X_initial_2d = np.array(X_initial_1d + X_initial_1d[-4:] + [.05])
    X_initial_2d = X_initial_2d.reshape(1, len(X_initial_2d))
    initial_pop = Population.new("X", X_initial_2d)
    
    periods = {
        '17_year': {'train_start_date': pd.to_datetime('Jan 1, 2005'), 'val_start_date': pd.to_datetime('Jan 1, 2008'), 'end_date': pd.to_datetime('July 31, 2026')},
    }
    
    
    tickers = list(pd.read_csv(TICKER_FILE).symbol) + ['GIC']
    objective_functions_dict = {
        'worst_annual_return': functools.partial(worst_annual_return, pd.to_datetime('2008-01-01'), pd.to_datetime('2025-01-01'), tickers),
        'drawdown_integral': functools.partial(mean_annual_drawdown_integral, pd.to_datetime('2008-01-01'), pd.to_datetime('2025-01-01'), tickers),
        'annualized_return': functools.partial(mean_annualized_return, pd.to_datetime('2008-01-01'), pd.to_datetime('2025-01-01'), tickers),
        'worst_annual_4wk': functools.partial(pct_change_quantile, pd.to_datetime('2008-01-01'), pd.to_datetime('2025-01-01'), tickers, 1/13),
    }
    objective_sense = {'drawdown_integral': 'min', 'annualized_return': 'max', 'worst_annual_4wk': 'max'}

    principal = [408000]
    params = {
        'principal': principal, 'max_frac': .05, 'feature_horizon_weeks': 104,
        'min_price': 5, 'trade_fee': 7, 'objective_sensitivity': 0.144, 'obj_threshold': 0,
        'start_date': pd.to_datetime('Jan 1, 2005'), 'end_date': pd.Timestamp.now()
    }
    

    problem_args = get_rn_problem_params(
        periods,
        momentum_file = "simulation_data/momentum.nc", 
        quality_file = "simulation_data/quality.nc",
        gic_file = "simulation_data/gic_data.nc",
        macro_file = "s3://jdinvestment/simulation_data/macro_signals.parquet",
        manifold_file = "s3://jdinvestment/sim_results/manifold_triple_threat.csv",
        output_folder = OUTPUT_FOLDER,
        params = params,
        


    ) 
    
   
    problem_args['objective_functions_dict'] = objective_functions_dict
    problem_args['objective_sense'] = objective_sense
    
    
    problem = RegimeNavigator2D(**problem_args)    # The 'Wrapped' class + Sim Args
            
    
    result = problem.evaluate(initial_pop.get("X")[0])
    zzz=1
    
    
    
    