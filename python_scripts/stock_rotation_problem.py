"""
Triple Threat Strategy: Dry Run Version
1. MINIMAL POPULATION: Set to 3 to verify pipeline integrity.
2. MINIMAL GENERATIONS: Set to 2 to test the ask/tell loop transitions.
3. OUTPUT VISIBLE: SuppressOutput is disabled to monitor engine prints.
"""

import numpy as np

# --- 1. DRY RUN CONSTANTS ---
ANNUAL_RISK_PERCENTILE = 1/13
NUM_WORKERS = 1
N_OFFSPRINGS = 188
TIMEOUT = 180
POP_SIZE = 180   # Minimal for testing
GEN_COUNT =  600# Minimal for testing
EVAL_FOLDER = "s3://jdinvestment/new_evaluations_9"
HOLDINGS_FOLDER = "s3://jdinvestment/new_holdings_history_9"
CHECKPOINT_URI = "s3://jdinvestment/checkpoints/checkpoint_740.pkl"
TARGET_COMPLETIONS = 90


# Shared Search Space Configuration
VAR_COUNT = 16
OBJ_COUNT = 5

# Indices: 0-7: PCA, 8: Threshold, 9: Beta, 10-11: Decay, 12-15: Macro Weights
XL_DEFAULT = np.array([
    -2, -2, -2, -2,  # Mom PCA
    -2, -2, -2, -2,  # Qual PCA
    -2.0,            # Threshold (Index 8: expanded from 0.1)
    0.5,             # Beta (Index 9)
    -1, -1,          # Decays
    -1, -1, -1, -1   # Macro Weights
])

XU_DEFAULT = np.array([
    2, 2, 2, 2,      # Mom PCA
    2, 2, 2, 2,      # Qual PCA
    2.0,             # Threshold (Index 8: expanded from 0.9)
    15.0,            # Beta (Index 9: expanded from 2.0)
    1, 1,            # Decays
    1, 1, 1, 1       # Macro Weights
])



import os
import boto3 
import signal
import time
import random
from smart_open import open as smart_open


os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import logging
import psutil  # You may need to add this to your Docker image/environment

# Setup basic logging to stdout (which AWS Batch sends to CloudWatch)
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("TripleThreatMonitor")



import numpy as np
import os
import pandas as pd
import xarray as xr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import time
# Pymoo imports
from pymoo.core.problem import Problem


# Internal Logic Imports
from simulate_stock_rotation import get_gic_eps, simulate

import awswrangler as wr
import fsspec
import pickle


import resource
import platform
import contextlib


import multiprocessing as mp
import numpy as np
import time
from pymoo.core.problem import Problem



# --- 2. UTILITIES ---

import multiprocessing as mp
import numpy as np
import time
from pymoo.core.problem import Problem

from optimization_manager import OptimizationManager

from utils import get_dna_hash




# Sentinel to signal workers to shut down
STOP_SIGNAL = "STOP"


def get_direction_sign(label):
    """Returns -1 for 'max' and +1 for 'min'."""
    return -1 if label.lower() == 'max' else 1 if label.lower() == 'min' else None






@contextlib.contextmanager
def monitor_peak_memory():
    """
    Context manager to measure the peak Resident Set Size (RSS) 
    of the current process and its children.
    """
    try:
        yield
    finally:
        # Get peak memory usage
        usage = resource.getrusage(resource.RUSAGE_SELF)
        peak_bytes = usage.ru_maxrss
        
        # On macOS, ru_maxrss is in bytes. 
        # On Linux, it is in kilobytes.
        if platform.system() != 'Darwin':
            peak_bytes *= 1024
            
        peak_mb = peak_bytes / (1024 * 1024)
        print(f"\n--- Memory Report ---")
        print(f"Peak Resident Set Size: {peak_mb:.2f} MB")
        print(f"---------------------")


def save_checkpoint(algorithm, path):
    """
    Saves the algorithm state to Local or S3 using fsspec.
    """
    # Use 'wb' for binary write; fsspec handles the protocol (s3 vs local)
    with fsspec.open(path, 'wb') as f:
        pickle.dump(algorithm, f)
    print(f"--- [CHECKPOINT] State saved to {path} ---")

def load_checkpoint(path):
    """
    Loads the algorithm state if it exists.
    """
    # Check existence via fsspec
    fs, fs_path = fsspec.core.url_to_fs(path)
    if fs.exists(fs_path):
        with fsspec.open(path, 'rb') as f:
            return pickle.load(f)
    return None








def build_kit(df, cols):
    s = StandardScaler().fit(df[cols])
    p = PCA(n_components=4).fit(s.transform(df[cols]))
    return {'scaler': s, 'pca': p, 'columns': cols}




class StockRotationProblem(Problem):
   
    def __init__(
            self, 
            data_features, 
            df_price, 
            params,
            periods, 
            holdings_folder, 
            eval_folder, 
            weight_columns, 
            objective_functions_dict, 
            objective_sense, 
            holdings = None, 
            xl=XL_DEFAULT, 
            xu=XU_DEFAULT,
        ):

        # Store these so they exist for the __getstate__ / __setstate__ logic
        self.data_features = data_features
        self.df_price = df_price
        self.params = params
        self.periods = periods
        self.holdings_folder = holdings_folder
        self.eval_folder = eval_folder

        self.weight_columns = weight_columns
        self.obj_funcs = objective_functions_dict
        self.objective_sense = objective_sense
        self.holdings = holdings
        
        
        
        super().__init__(n_var=len(xl), n_obj=len(objective_sense), n_constr=0, xl=xl, xu=xu)


    def __getstate__(self):
        """
        Exclude massive dataframes/xarrays from the pickle file.
        This keeps the checkpoint file tiny (KBs instead of GBs).
        """
        state = self.__dict__.copy()
        # Remove the heavy hitters
        keys_to_drop = ['df_macro', 'data_features', 'df_price', 'base']
        for key in keys_to_drop:
            state[key] = None
        return state

    def __setstate__(self, state):
        """Called when unpickling; we will re-inject data in the main loop."""
        self.__dict__.update(state)


    def run_simulation(self, w_mom_vals, w_qual_vals, period_key, sim_id, session = None, holdings = None):
        period = self.periods[period_key]
        dates = self.df_price.columns
        df_weights = pd.DataFrame(len(dates) * (w_mom_vals + w_qual_vals), index = dates, columns = list(self.data_features.band))
        df_holdings = simulate(self.df_price, self.params, self.data_features, df_weights, period, self.holdings_folder, sim_id, session = session, holdings = holdings)
        
        
        return df_holdings
        
    
    def _evaluate(self, x, out, *args, **kwargs):
        t1 = time.time()
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
        macro_weights = x_numeric[12:]/sum(abs(x_numeric[12:]))

        
        
        df_sim = pd.DataFrame()
        for key in self.periods:
            df_period = self.run_simulation(w_mom, w_qual, opt_threshold, opt_beta, mom_decay, qual_decay, macro_weights, key, sim_id, holdings = self.holdings)
            df_sim = pd.concat([df_sim, df_period]).reset_index(drop=True)
        if df_sim.empty:
            out["F"] = [9999999] * 5
            return

        ticker_cols = [c for c in df_sim.columns if c not in ['sim_id', 'date', 'total_value', 'pct_change']]
        df_sim['total_value'] = df_sim[ticker_cols].apply(pd.to_numeric, errors='coerce').sum(axis=1)
        
        out_columns = ['sim_id'] + list(self.obj_funcs.keys()) + self.weight_columns
        obj_results = {name: func(df_sim) for name, func in self.obj_funcs.items()}
        
        values = [sim_id] + list(obj_results.values()) + list(x)
        df_out = pd.DataFrame({out_columns[i]: [values[i]] for i in range(len(out_columns))})

        save_result_agnostic(df_out, self.eval_folder)

        
        out["F"] = [get_direction_sign(self.objective_sense[key]) * obj_results[key] for key in self.objective_sense.keys()] 

        print('time: {}'.format(time.time() - t1))



def get_sr_problem_params(
        periods,
        momentum_file = "simulation_data/momentum.nc", 
        quality_file = "simulation_data/quality.nc",
        gic_file = "simulation_data/gic_data.nc",
        params = None,
        output_folder = None
        


):
   
    with smart_open(momentum_file, 'rb') as fp:
        da_mom = xr.open_dataset(fp).to_array().squeeze()
    with smart_open(quality_file, 'rb') as fp:
        da_qual = xr.open_dataset(fp).to_array().squeeze()
    _data_features = xr.concat([da_mom, da_qual], dim='band')
    with smart_open(gic_file, 'rb') as fp:
        da_mom_gic = xr.open_dataarray(fp)
    da_qual_gic = get_gic_eps(da_mom_gic)
    data_features = xr.concat([_data_features, xr.concat([da_mom_gic, da_qual_gic], dim='band')], dim='symbol', join = 'inner').drop_sel(band = 'price_end')
    
    df_price = da_mom.sel(band='price_end').to_pandas()
    
    
    
    
    
    if params is None:
        params = {
            'principal': [327000, 60000, 21000], 'max_frac': .05, 'feature_horizon_weeks': 104,
            'min_price': 5, 'trade_fee': 7, 'objective_sensitivity': 0.144, 'obj_threshold': 0,
            'start_date': pd.to_datetime('Jan 1, 2005'), 'end_date': pd.Timestamp.now()
        }
    
    


    
    problem_args = (data_features, df_price, params, periods, output_folder)
    arg_names = ('data_features', 'df_price', 'params', 'periods', 'output_folder')
    
    return dict(zip(arg_names, problem_args))

    

