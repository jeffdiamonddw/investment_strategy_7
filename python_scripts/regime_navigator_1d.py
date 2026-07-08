# --- Standard Library ---
import inspect
import logging
import os
import time
import functools

# --- 3rd Party ---
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.population import Population


# --- Custom ---

from stock_rotation_problem import get_sr_problem_params
from optimization_manager import OptimizationManager
from utils import get_dna_hash



from optimization_manager import OptimizationManager
from simulate_stock_rotation import simulate
from objective_functions import mean_annualized_return
from perturbation_executor import PerturbationExecutor
from simulation_manager import SimulationManager


NUM_WORKERS = 4
POP_SIZE = 4
N_OFFSPRING = 4 
TARGET_COMPLETIONS_PER_ROUND = 4
TIMEOUT_SEC = 180
GEN_COUNT = 5
SATISFICING_SHARPNESS = 20

TICKER_FILE = "strategy/multi_dim_stock_list.csv"
\




# Indices: 0-7: PCA, 8: Threshold, 9: Beta, 10-11: Decay, 12-15: Macro Weights
XL= np.array([
    -2, -2, -2, -2,  # Mom PCA
    -2, -2, -2, -2,  # Qual PCA
    -2.0,            # Threshold (Index 8: expanded from 0.1)
    0.5,             # Beta (Index 9)
    -1, -1,          # Decays
    -1, -1, -1, -1   # Macro Weights
])

XU = np.array([
    2, 2, 2, 2,      # Mom PCA
    2, 2, 2, 2,      # Qual PCA
    2.0,             # Threshold (Index 8: expanded from 0.9)
    15.0,            # Beta (Index 9: expanded from 2.0)
    1, 1,            # Decays
    1, 1, 1, 1       # Macro Weights
])


# Setup basic logging to stdout (which AWS Batch sends to CloudWatch)
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("TripleThreatMonitor")



def get_direction_sign(label):
    """Returns -1 for 'max' and +1 for 'min'."""
    return -1 if label.lower() == 'max' else 1 if label.lower() == 'min' else None









def save_result_agnostic(df, path):
    """
    Saves a single evaluation result (row) to a destination.
    
    Local Logic: 
        Appends the row to a single CSV file.
    S3 Logic: 
        Writes a unique CSV to the directory using the sim_id as the filename.
    """
    if path.startswith("s3://"):
        # Ensure we have a sim_id to name the file
        if 'sim_id' in df.columns and not df.empty:
            # We take the first sim_id (assuming one eval per call)
            sim_id = df['sim_id'].iloc[0]
            
            # Clean path to ensure it's treated as a directory
            base_dir = path if path.endswith('/') else path + '/'
            s3_path = f"{base_dir}sim_{sim_id}.csv"
            
            # Use awswrangler to write the individual file
            wr.s3.to_csv(df=df, path=s3_path, index=False)
        else:
            raise ValueError("S3 destination requires 'sim_id' in DataFrame for unique naming.")
            
    else:
        # Local Logic: Standard file append
        # Create directory if it doesn't exist
        dir_name = os.path.dirname(path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
            
        # Append to the file (write header only if file doesn't exist)
        file_exists = os.path.isfile(path)
        df.to_csv(path, mode='a', index=False, header=not file_exists)



def build_kit(df, cols):
    s = StandardScaler().fit(df[cols])
    p = PCA(n_components=4).fit(s.transform(df[cols]))
    return {'scaler': s, 'pca': p, 'columns': cols}




class RegimeNavigator1D:
   
    def __init__(
            self, 
            mom_kit, 
            qual_kit, 
            df_macro, 
            data_features, 
            df_price, 
            params,
            periods, 
            holdings_folder, 
            eval_folder,  
            objective_functions_dict, 
            objective_sense, 
            holdings = None, 
            

        ):

       self.__dict__.update({k: v for k, v in locals().items() if k != 'self'})
        
        
        
   


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


    def run_simulation(self, w_mom_vals, w_qual_vals, threshold, beta, mom_decay, qual_decay, macro_weights, period_key, sim_id, session = None, holdings = None):
        period = self.periods[period_key]
        
        keep_date = (period['val_start_date'] <= df_macro.index) & (df_macro.index <= period['end_date'])
        df_macro = self.df_macro.loc[keep_date]

        s_risk_aversion_full = self.df_macro.dot(macro_weights).rename("risk_aversion") 
        risk_aversion_mean = s_risk_aversion_full.mean()
        s_risk_aversion = s_risk_aversion_full.loc[
            (self.df_macro.index >= period['val_start_date']) & (self.df_macro.index <= period['end_date'])
        ]  
        
        
        s_quality_weight = 1 / (1 + np.exp(-beta * (s_risk_aversion - threshold)))
        mom_num_periods = np.array([int(col.split('_')[-1][:-1]) for col in self.mom_kit['columns']])
        qual_num_periods = np.array([int(col.split('_')[-1][:-1]) for col in self.qual_kit['columns']])
        df_mom_decay = pd.DataFrame(np.exp(- mom_decay * (risk_aversion_mean - s_risk_aversion).values[:, None] * mom_num_periods), index=s_risk_aversion.index, columns = self.mom_kit['columns'])
        df_qual_decay = pd.DataFrame(np.exp(- qual_decay * (risk_aversion_mean - s_risk_aversion).values[:, None] * qual_num_periods), index=s_risk_aversion.index, columns = self.qual_kit['columns'])
        
        
        w_dict = {}
        df_mom = pd.DataFrame({self.mom_kit['columns'][j]: [w_mom_vals[j]] for j in range(len(w_mom_vals))})
        df_qual = pd.DataFrame({self.qual_kit['columns'][j]: [w_qual_vals[j]] for j in range(len(w_mom_vals))})
        
        df_mom_weights = df_mom_decay.mul(df_mom.iloc[0], axis=1).mul(1 - s_quality_weight, axis=0)
        df_qual_weights = df_qual_decay.mul(df_qual.iloc[0], axis=1).mul(s_quality_weight, axis=0)
        df_weights = pd.concat([df_mom_weights, df_qual_weights], axis = 1)
        df_holdings = simulate(self.df_price, self.params, self.data_features, df_weights, period, self.holdings_folder, sim_id, session = session, holdings = holdings)
        
        
        return df_holdings
        
    
    def evaluate(self, x):
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
        macro_weights = x_numeric[12:]

        
        
        df_sim = pd.DataFrame()
        for key in self.periods:
            df_period = self.run_simulation(w_mom, w_qual, opt_threshold, opt_beta, mom_decay, qual_decay, macro_weights, key, sim_id, holdings = self.holdings)
            df_sim = pd.concat([df_sim, df_period]).reset_index(drop=True)
        

        ticker_cols = [c for c in df_sim.columns if c not in ['date', 'sim_id'] ]
        total_value_series = df_sim.set_index('date')[ticker_cols].apply(pd.to_numeric, errors='coerce').sum(axis=1).rename('value')
        
        
        
        return total_value_series

      




def align_dataframe_to_dates(df, target_dates, max_days=7):
    # 1. Combine indexes
    combined_index = df.index.union(target_dates).sort_values()
    df_combined = df.reindex(combined_index)
    
    # 2. Interpolate
    df_interpolated = df_combined.interpolate(method='time')
    
    # 3. Extrapolation Logic:
    # ffill() extends the last valid point forward
    # bfill() extends the first valid point backward
    df_interpolated = df_interpolated.ffill().bfill()
    
    # 4. Filter to target dates
    df_final = df_interpolated.reindex(target_dates)
    
    # 5. Apply Proximity Filter (within 7 days)
    idx = df.index.get_indexer(target_dates, method='nearest')
    nearest_dates = df.index[idx]
    diffs = (pd.Series(target_dates) - pd.Series(nearest_dates)).abs()
    valid_mask = diffs <= pd.Timedelta(days=max_days)
    
    return df_final[valid_mask.values]

def get_rn_problem_params(
        periods,
        momentum_file = "s3://jdinvestment//momentum.nc", 
        quality_file = "s3://jdinvestment/simulation_data/quality.nc",
        gic_file = "s3://jdinvestment/simulation_data/gic_data.nc",
        macro_file = "s3://jdinvestment/simulation_data/macro_signals.parquet",
        manifold_file = "s3://jdinvestment/sim_results/manifold_triple_threat.csv",
        output_folder = None,
        params = None,
        holdings = None
        


):
    

    # 1. Capture all incoming argument values from this local scope
    local_vars = locals()
    clean_args = {k: v for k, v in local_vars.items() if k != 'self'}

    # 2. Dynamic Parent (Wrapper framework) Init Call via Keyword Unpacking
    sr_sig = inspect.signature(get_sr_problem_params)
    sr_kwargs = {
        k: v for k, v in clean_args.items() 
        if k in sr_sig.parameters
    }
    sr_problem_params = get_sr_problem_params(**sr_kwargs)
   
    
    
    if macro_file.endswith('.csv'):
        df_macro = pd.read_csv(macro_file, index_col=0)
        df_macro.index = pd.to_datetime(df_macro.index)
    else:
        df_macro = pd.read_parquet(macro_file)
    

    
    
    dates = sr_problem_params['df_price'].columns
    df_macro = align_dataframe_to_dates(df_macro, dates)
    
    if params is None:
        params = {
            'principal': [408000], 'max_frac': .05, 'feature_horizon_weeks': 104,
            'min_price': 5, 'trade_fee': 7, 'objective_sensitivity': 0.144, 'obj_threshold': 0,
            'start_date': pd.to_datetime('Jan 1, 2005'), 'end_date': pd.Timestamp.now()
        }
    
    

    print("Initializing problem kits...")
    df_man = pd.read_csv(manifold_file)
    mom_cols = ['dollar_ret_1p', 'dollar_ret_6p', 'dollar_ret_13p', 'dollar_ret_26p']
    qual_cols = ['avg_eps_1q', 'avg_eps_2q', 'avg_eps_4q', 'avg_eps_8q']
    
    df_elite = df_man.nlargest(int(len(df_man) * 0.10), 'f4_terminal')
    mom_kit = build_kit(df_elite, mom_cols)
    qual_kit = build_kit(df_elite, qual_cols)

    problem_params = sr_problem_params.copy()
    problem_params.update({'mom_kit': mom_kit, 'qual_kit': qual_kit, 'df_macro': df_macro})

    
    return problem_params

    
if __name__ == '__main__':


    periods = {
            'train': {'train_start_date': pd.to_datetime('2006-01-01'), 'val_start_date': pd.to_datetime('2008-01-01'), 'end_date': pd.to_datetime('2024-12-31')},
        }
        


    objective_functions_dict = {
        'annualized_return':functools.partial(annualized_return, periods['train']['val_start_date'], periods['train']['end_date']),
        'drawdown_integral': functools.partial(mean_annual_drawdown_integral,periods['train']['val_start_date'], periods['train']['end_date']),
        'worst_annual_4wk': functools.partial(pct_change_quantile, periods['train']['val_start_date'], periods['train']['end_date'], 1/13),
    }
    objective_sense = {'annualized_return': 'max', 'drawdown_integral': 'min', 'worst_annual_4wk': 'max'}
    satisficing_threshold = {
        'drawdown_integral': .004,
        'worst_annual_4wk': -.025

    }


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
    


    
    

    # Attempt to load existing progress
    algorithm = load_checkpoint(CHECKPOINT_URI)
    
    

    if True: #algorithm is None:
        print("Initial Startup: Building RANDOM population...")
        
        
        # 4. Initialize Manager and NSGA2 (Default sampling is FloatRandomSampling)
        simulation_manager = SimulationManager(
            
            workhorse_cls=PerturbationExecutor, 
            workhorse_args= dict(workhorse_cls = RegimeNavigator1D, workhorse_args = problem_args, n_samples = 3, perturbation_cv = 0.01),      # The 'Wrapped' class + Sim Args
            num_workers=NUM_WORKERS,
            timeout_sec=TIMEOUT_SEC,
            target_completions_per_round = TARGET_COMPLETIONS_PER_ROUND,
            target_completions_per_generation = POP_SIZE
        )
        problem = OptimizationManager(
            sim_manager = simulation_manager, 
            objective_dict = objective_functions_dict,
            objective_sense = objective_sense,
            thresholds = satisficing_threshold,
            xl = XL,
            xu = XU,
            output_file = OUTPUT_FILE
        )
    
        

        # 2. Initialize NSGA2 with the forced Mating object
        algorithm = NSGA2(
            pop_size=POP_SIZE,
            eliminate_duplicates=True
        )
        algorithm.setup(problem, termination=('n_gen', GEN_COUNT))
        algorithm.mating.n_offsprings = N_OFFSPRING
                # 1. Force the algorithm's internal n_offsprings property
        algorithm.n_offsprings = N_OFFSPRING


        # 2. Force-initialize the population if it's None (The "Warm-up" step)
        if algorithm.pop is None:
            # Use the sampling operator to create the first 4 individuals
            algorithm.pop = algorithm.initialization.do(algorithm.problem, POP_SIZE)
            # Important: Evaluate the initial population so they have fitness values
            algorithm.evaluator.eval(algorithm.problem, algorithm.pop)

        save_checkpoint(algorithm, CHECKPOINT_URI)
            
    else:
        print(f"Resuming from Generation {algorithm.n_gen}...")
    
        # 4. Initialize Manager and NSGA2 (Default sampling is FloatRandomSampling)
        simulation_manager = SimulationManager(
            
            workhorse_cls=PerturbationExecutor, 
            workhorse_args= dict(workhorse_cls = RegimeNavigator2D, workhorse_args = problem_args, n_samples = 3, perturbation_cv = 0.01),      # The 'Wrapped' class + Sim Args
            num_workers=NUM_WORKERS,
            timeout_sec=TIMEOUT_SEC,
            target_completions_per_round = TARGET_COMPLETIONS_PER_ROUND,
            target_completions_per_generation = POP_SIZE
        )
        problem = OptimizationManager(
            sim_manager = simulation_manager, 
            objective_dict = objective_functions_dict,
            thresholds = satisficing_threshold,
            xl = XL,
            xu = XU
        )
        
        # Sync the generation counts
        problem.n_gen = algorithm.n_gen 
        algorithm.problem = problem

        # 2. FORCE OVERRIDE TERMINATION
        # We re-import the termination factory to ensure it's fresh
        from pymoo.termination import get_termination
        algorithm.termination = get_termination("n_gen", GEN_COUNT)
        
        # 3. CRITICAL: Reset the 'has_finished' flag if it exists
        algorithm.has_terminated = False

    # --- 4. EXECUTION LOOP ---
    first_gen = True
    gen = 0
    while algorithm.has_next():
        infills = algorithm.ask()
    
        print("*****************************************    Generation {} *****************************************************************************".format(gen))
        gen += 1
        
        # if first_gen and len(infills) < N_OFFSPRING:
        #     n_missing = N_OFFSPRING - len(infills)
        #     # Create random samples in the same space
        #     # (Assuming your problem defines XL and XU)
        #     random_samples = np.random.uniform(problem.xl, problem.xu, size=(n_missing, problem.n_var))
            
        #     new_infills = Population.new("X", random_samples)
            
        #     # Combine the actual algorithm infills with our bootstrap infills
        #     infills = Population.merge(infills, new_infills)
            
        #     first_gen = False
            
        # In a parallel version, you'd use your input_queue logic here. 
        # For this dry run, it uses the standard evaluator.
        algorithm.evaluator.eval(algorithm.problem, infills)
        

        valid_infills = Population()
        for ind in infills:
            # Only keep the individual if F is not None
            if np.isnan(ind.get("F")).sum() == 0:
                valid_infills = Population.merge(valid_infills, ind)

        algorithm.tell(infills=valid_infills)
        
        # Checkpoint at the end of every successful generation
        save_checkpoint(algorithm, CHECKPOINT_URI)
        # print(f"Gen {algorithm.n_gen} Success. Checkpoint saved.")

    print("--- Optimization Complete ---")
    #algorithm.problem.cleanup()
    # At the very end of your script
    
    print("Optimization Complete")