
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
from simulate_stock_rotation import simulate, get_proposal, optimize
from objective_functions import mean_annualized_return, weighted_quantile, weighted_mean, WeightedRegretApplyer, WeightedRegimeApplyer, apply_objectives, RegretPercentile, FoldPercentile, FoldApplyer, worst_annual_drawdown_integral
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
            objective_sense = None,
            df_dividend = None,
            now = False,
            hold = False
        ):
        
       self.__dict__.update({k: v for k, v in locals().items() if k != 'self'})

       if holdings is None:
            self.holdings = pd.DataFrame(0.0, index = df_price.index, columns = range(len(params['principal'])))
            self.holdings.loc['CASH', :] = np.floor(params['principal']).astype(int)


    
       


    def get_simulation_args(self, w_mom, w_qual, threshold, beta, mom_decay, qual_decay, df_macro_weights, max_voo, sim_id, max_frac):
        
        
        macro_weights_risk = df_macro_weights.values.flatten()[:4]
        macro_weights_temporal = df_macro_weights.values.flatten()[4:]
        
        macro_weights_risk /= macro_weights_risk.sum()
        macro_weights_temporal /= macro_weights_temporal.sum()
        period = self.periods
        
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

        mom_cols = ['dollar_ret_1p', 'dollar_ret_6p', 'dollar_ret_13p', 'dollar_ret_26p']
        qual_cols = ['avg_eps_1q', 'avg_eps_2q', 'avg_eps_4q', 'avg_eps_8q']
        mom_num_periods = np.array([int(col.split('_')[-1][:-1]) for col in mom_cols])
        qual_num_periods = np.array([int(col.split('_')[-1][:-1]) for col in qual_cols])
        
        df_mom_decay = pd.DataFrame(np.exp((- mom_decay * (risk_aversion_mean - s_risk_aversion).values[:, None] * mom_num_periods).astype(float)), index=s_risk_aversion.index, columns = self.mom_kit['columns'])
        df_qual_decay = pd.DataFrame(np.exp((- qual_decay * (risk_aversion_mean - s_risk_aversion).values[:, None] * qual_num_periods).astype(float)), index=s_risk_aversion.index, columns = self.qual_kit['columns'])
        
        
        
        
        df_mom = pd.DataFrame({mom_cols[j]: [w_mom[j]] for j in range(len(w_mom))})
        df_qual = pd.DataFrame({qual_cols[j]: [w_qual[j]] for j in range(len(w_mom))})
        
        df_mom_weights = df_mom_decay.mul(df_mom.iloc[0], axis=1).mul(1 - s_quality_weight, axis=0)
        df_qual_weights = df_qual_decay.mul(df_qual.iloc[0], axis=1).mul(s_quality_weight, axis=0)
        df_weights = pd.concat([df_mom_weights, df_qual_weights], axis = 1)
        df_weights = df_weights.div(df_weights.sum(axis = 1), axis = 0)
        df_weights.to_csv('temp/weights.csv')
       #********************************************************************************************************************************* 

        simulate_args = {
            'df_price' : self.df_price, 
            '_params': self.params, 
            'data_features': self.data_features, 
            'df_weights': df_weights, 
            'period': period, 
            'sim_id': sim_id , 
            'holdings': self.holdings, 
            'max_frac': max_frac, 
            'max_voo' : max_voo, 
            'df_dividend': self.df_dividend , 
            'now': self.now , 
            'hold': self.hold 

        }
        return simulate_args

    
    def get_proposal_args(self, x): #self, w_mom, w_qual, threshold, beta, mom_decay, qual_decay, df_macro_weights, max_voo, sim_id, holdings, max_frac):

        sim_params = self.extract_params(x)
        simulation_args = self.get_simulation_args(**sim_params)

        holdings_start = self.holdings

        
        params = self.params
        data_features = simulation_args['data_features']
        df_features = data_features.isel(date = -1).to_pandas().transpose()
        feature_weights = dict(simulation_args['df_weights'].iloc[-1])
        max_voo = simulation_args['max_voo']
        max_frac = simulation_args['max_frac']

        result = {
            'holdings_start': holdings_start, 
            'params': params, 
            'df_features': df_features, 
            'feature_weights': feature_weights, 
            'max_voo': max_voo,
           'max_frac': max_frac

        }
        return result

    def get_proposal(self, x):
        proposal_args = self.get_proposal_args(x)
        proposed_holdings = get_proposal(**proposal_args)
        return proposed_holdings

    def update_proposal(self,x):
        proposal_args = self.get_proposal_args(x)
        df_price_live = pd.read_parquet('live_quotes_cache.parquet').set_index('symbol')
        live_sell_price = df_price_live['bid_price']; live_sell_price.loc['CASH'] = 1
        live_buy_price = df_price_live['ask_price']; live_buy_price.loc['CASH'] = 1
        live_price = .5 * (live_sell_price + live_buy_price)
        for key in ['holdings_start', 'df_features']:
            proposal_args[key] = proposal_args[key].loc[list(df_price_live.index) + ['CASH']]

        holdings = proposal_args['holdings_start']
        invested = holdings.loc[[s for s in holdings.index if s != 'CASH']].values.sum()
        if invested == 0:
            proposal_args['live_price'] = live_buy_price
            proposed_holdings = get_proposal(**proposal_args)
            buy = np.maximum(0, proposed_holdings - self.holdings)

            buy = buy.loc[buy.sum(1) >0]
            buy_cost = (buy * live_buy_price.loc[buy.index].values.reshape(len(buy.index), 1)).sum()
            proposed_holdings.loc['CASH'] = self.holdings.loc['CASH'] - buy_cost
            sell = pd.DataFrame()

        else:
            proposal_args['live_price'] = live_price
            proposed_holdings = get_proposal(**proposal_args)


            proposed_holdings = get_proposal(**proposal_args)
            holdings_symbols = self.holdings.loc[self.holdings.sum(1) !=0].index
            symbols = sorted(list(set(holdings_symbols).union(proposed_holdings.index)))
            holdings = self.holdings.loc[symbols]
            proposed_holdings_aug = 0 * holdings.copy()
            proposed_holdings_aug.loc[proposed_holdings.index] = proposed_holdings
            

        

            sell = - np.minimum(0, proposed_holdings_aug - holdings)
            sell = sell.loc[sell.sum(1) != 0]
            buy = np.maximum(0, proposed_holdings_aug - holdings)
            buy = buy.loc[buy.sum(1) != 0]

            live_price.loc[sell.index] = live_sell_price.loc[sell.index]
            live_price.loc[buy.index] = live_buy_price.loc[buy.index]

            proposal_args['live_price'] = live_price
            proposed_holdings = get_proposal(**proposal_args)

            sell = - np.minimum(0, proposed_holdings_aug - holdings)
            buy = np.maximum(0, proposed_holdings_aug - holdings)


            sell_revenue = (sell * live_sell_price.loc[sell.index].values.reshape(len(sell.index), 1)).sum()
            buy_cost = (buy * live_buy_price.loc[buy.index].values.reshape(len(buy.index), 1)).sum()
            num_trades = (sell.loc[(sell.index != 'CASH')]>0).values.sum() + (buy.loc[(buy.index != 'CASH')]>0).values.sum()
            cash_made = sell_revenue - buy_cost - 6.95 * num_trades
            proposed_holdings.loc['CASH'] = self.holdings.loc['CASH'] + cash_made

        return proposed_holdings, sell, buy, live_sell_price, live_buy_price

        
        

        
       
        
        



    def run_simulation(self, w_mom, w_qual, threshold, beta, mom_decay, qual_decay, df_macro_weights, max_voo, sim_id, max_frac):
            
        
            simulation_args = self.get_simulation_args(w_mom, w_qual, threshold, beta, mom_decay, qual_decay, df_macro_weights, max_voo, sim_id, max_frac)
            df_holdings_history, proposed_holdings, data_share_holdings, live_sell_price, live_buy_price = simulate(**simulation_args)
      
            return df_holdings_history, proposed_holdings, data_share_holdings, live_sell_price, live_buy_price



    def extract_params(self, x):
         
        x_numeric = x.X if hasattr(x, "X") else x
        sim_id = get_dna_hash(x_numeric)
        # CHANGE self.mom_kit to self.mom_kit
        w_mom = x_numeric[:4]
        
        # CHANGE self.qual_kit to self.qual_kit
        w_qual =x_numeric[5:9]
        w_mom, w_qual = np.clip(w_mom, 0, 1), np.clip(w_qual, 0, 1)

        threshold = x_numeric[8]
        beta = x_numeric[9]
        mom_decay = x_numeric[10]
        qual_decay = x_numeric[11]
        df_macro_weights = pd.DataFrame(x_numeric[12:20].reshape(2,4), index = ['risk_weights', 'temporal_weights'], columns = self.df_macro.columns)
        max_voo = x_numeric[20]
        if len(x_numeric) >= 22:
            max_frac = x_numeric[21]
        else:
            max_frac = .05
        # max_frac = .05
        
        sim_params = {
            'w_mom': w_mom, 
            'w_qual': w_qual, 
            'threshold': threshold, 
            'beta': beta, 
            'mom_decay': mom_decay, 
            'qual_decay': qual_decay, 
            'df_macro_weights': df_macro_weights, 
            'max_voo': max_voo, 
            'sim_id': sim_id,  
            'max_frac': max_frac
        }

        return sim_params

    

    def evaluate(self, x):
        
        
        
        sim_params = self.extract_params(x)
        df_sim, proposed_holdings, data_share_holdings, live_sell_price, live_buy_price = self.run_simulation(**sim_params)
    
        ticker_cols = [c for c in df_sim.columns if c not in ['date', 'sim_id'] ]
        total_value_series = df_sim.set_index('date')[ticker_cols].apply(pd.to_numeric, errors='coerce').sum(axis=1).rename('value')
        
        
        
        return df_sim, proposed_holdings, total_value_series, data_share_holdings, live_sell_price, live_buy_price
        
        





if __name__ == "__main__":
    
    DATA_PATH = "s3://jdinvestment/simulation_data_533"

    num_samples = 3
    perturbation_cv = .01

    
    t1 = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--s3_path', required=True)
    parser.add_argument('--generation', type=int, required=False)
    parser.add_argument('--train_folds', type=int, nargs='+', default=[],
                        help='List of training fold integers (e.g., --train_fold 0 1 2)')
    parser.add_argument('--val_folds', type=int, nargs='+', default=[],
                        help='List of validation fold integers (e.g., --val_fold 3 4)')
    parser.add_argument('--holdings', type = str, required=False)
    parser.add_argument('--update', action='store_true', help='runs optimization for next time period with live price data')
    parser.add_argument('--propose', action='store_true', help='runs optimization for current plan with last closing price')
    parser.add_argument('--hold', action='store_true', help='runs simulation just holding existing holdings')
    

    args = parser.parse_args()

    s3_path = args.s3_path
    generation = args.generation
    task_index = int(os.environ.get('AWS_BATCH_JOB_ARRAY_INDEX', 0))
   
    
    logging.getLogger('botocore.credentials').setLevel(logging.WARNING)
    my_boto3_session = boto3.Session()
    s3 = s3fs.S3FileSystem(session=my_boto3_session)

    periods = {'train_start_date': pd.to_datetime('2006-01-01'), 'val_start_date': pd.to_datetime('2008-01-01'), 'end_date': pd.to_datetime('2026-10-01')}
    
    
    
    df_folds = pd.read_parquet("{}/folds.parquet".format(s3_path))
  
 
    
    df_train_folds = df_folds.loc[df_folds.fold_index.isin(args.train_folds)]
    df_val_folds = df_folds.loc[df_folds.fold_index.isin(args.val_folds)]

    weighting_func_quantile = functools.partial(weighted_quantile, .1)
    start_date = min(df_folds.start_date)
    end_date = max(df_folds.end_date)
    agg_func = functools.partial(mean_annualized_return, start_date, end_date)

    
  

    objective_functions_dict = {
        'train': {
            'regret_quantile': WeightedRegretApplyer(df_train_folds, agg_func, weighting_func_quantile, 'voo_return', 'max'),
            'mean_regret' : WeightedRegretApplyer(df_train_folds, agg_func, weighted_mean, 'voo_return', 'max'),
            'quantile': WeightedRegimeApplyer(df_train_folds, agg_func, weighting_func_quantile),
            'mean': WeightedRegimeApplyer(df_train_folds, agg_func, weighted_mean),
        },
        'val': {
            'regret_quantile': WeightedRegretApplyer(df_val_folds, agg_func, weighting_func_quantile, 'voo_return', 'max'),
            'mean_regret' : WeightedRegretApplyer(df_val_folds, agg_func, weighted_mean, 'voo_return', 'max'),
            'quantile': WeightedRegimeApplyer(df_val_folds, agg_func, weighting_func_quantile),
            'mean': WeightedRegimeApplyer(df_val_folds, agg_func, weighted_mean)
        }
    }
   

    
    objective_sense = {'regret_quantile': 'min', 'mean_regret': 'min'}
    

    #principal = [23958.38]  
    #principal = [sum([15312.67, 238478.05]), 43828.5]
    #principal = [297619.22]  #CAD [21312, 331911, 61000, 33345]
    #principal = [280000]
    #principal = [237552.61, 43497.29]
    #principal = [24180.63]
    principal = [237552.61, 43497.29]
    #principal = [555999.79]
    
    
    params = {
            'principal': principal, 'max_frac': .05, 'feature_horizon_weeks': 104,
            'min_price': 5, 'trade_fee': 7, 'objective_sensitivity': 0.01, 'obj_threshold': 0,
            'start_date': pd.to_datetime('Jan 1, 2005'), 'end_date': pd.Timestamp.now()
        }
    

   
    holdings = args.holdings
    if holdings is not None:
        holdings = pd.read_parquet(holdings)
    
    problem_args = get_rn_problem_params(
        momentum_file = "{}/momentum.nc".format(DATA_PATH), 
        quality_file = "{}/quality.nc".format(DATA_PATH),
        bil_file = "{}/bil_data.nc".format(DATA_PATH),
        macro_file = "s3://jdinvestment/simulation_data/macro_signals.parquet".format(DATA_PATH),
        manifold_file = "s3://jdinvestment/sim_results/manifold_triple_threat.csv",
        output_folder = None,
        params = params,
        holdings = holdings


    ) 


    
    
    problem_args['periods'] = periods
    problem_args['hold'] = args.hold

    regime_navigator = RegimeNavigator2D(**problem_args)

    s_task = pd.read_parquet("{}/populations/gen_{}.parquet".format(s3_path, generation)).iloc[task_index]
    parameter_cols = [name for name in s_task.index if 'sim_id' not in name]
    x = s_task.loc[parameter_cols].values
    sim_id = get_dna_hash(x)

   

    #add percentile of 28-day regret as objective
    s_voo_pct_change = problem_args['df_price'].loc['VOO'].pct_change()
    s_voo_pct_change.index = pd.to_datetime(s_voo_pct_change.index)
    objective_functions_dict['train']['28_day_regret'] = lambda s_val: RegretPercentile(s_voo_pct_change, df_folds, args.train_folds, quantile = .9)(s_val.pct_change())
    objective_functions_dict['train']['28_day_percentile'] = lambda s_val: FoldPercentile(df_folds, args.train_folds, quantile = 1/13)(s_val.pct_change())
    objective_functions_dict['train']['drawdown'] = lambda s_val: FoldApplyer(df_folds, args.train_folds, myfunc = worst_annual_drawdown_integral)(s_val)
    objective_functions_dict['val']['28_day_regret'] = lambda s_val: RegretPercentile(s_voo_pct_change, df_folds, args.val_folds, quantile = .9)(s_val.pct_change())
    objective_functions_dict['val']['28_day_percentile'] = lambda s_val: FoldPercentile(df_folds, args.val_folds, quantile = 1/13)(s_val.pct_change())
    objective_functions_dict['val']['drawdown'] =  lambda s_val: FoldApplyer(df_folds, args.val_folds, myfunc = worst_annual_drawdown_integral)(s_val)
    objective_sense['28_day_regret'] = 'min'
    objective_sense['drawdown'] = 'min'

    
    #jeff temp
    #total_value_series = pd.read_parquet('temp/total_value_series.parquet')['value']
    #df_evaluation = apply_objectives(objective_functions_dict, total_value_series)

    print('pre-time: {}'.format(time.time() - t1), flush = True)
    
    parent_sim_id = sim_id
    perturbed_x = x
    df_evaluations = pd.DataFrame()

    if args.propose:
        proposed_holdings = regime_navigator.get_proposal(x)
        proposed_holdings.to_parquet("{}/proposed_holdings.parquet".format(args.s3_path))
    elif args.update:
        proposed_holdings, sell, buy, live_sell_price, live_buy_price = regime_navigator.update_proposal(x)
        zzz=1
        

    else:

        for sample in range(num_samples):
            sim_id = get_dna_hash(perturbed_x)
            df_history, proposed_holdings, total_value_series, data_share_holdings, live_sell_price, live_buy_price = regime_navigator.evaluate(perturbed_x)

            tvs = total_value_series
            num_years = (tvs.index[-1] - tvs.index[0]).days/365.25
            rtn = (tvs.iloc[-1]/tvs.iloc[0])**(1/num_years)-1
            print('rtn {}:'.format(rtn))


            logging.getLogger('botocore.credentials').setLevel(logging.WARNING)
            my_boto3_session = boto3.Session()

            df_holdings = df_history.set_index('date').iloc[:-1, :-1]

            proposed_holdings.to_parquet("{}/proposed_holdings/sim_{}.parquet".format(s3_path, sim_id))
            
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
                path='{}/median_objectives/gen={}/sim_{}.parquet'.format(s3_path, generation, parent_sim_id),
                dataset=False,
                index = True,
                boto3_session=my_boto3_session 
        )


   
        
    
    


     
