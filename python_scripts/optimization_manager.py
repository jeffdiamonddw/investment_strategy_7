import numpy as np
import pandas as pd

from pymoo.core.problem import Problem

from objective_functions import get_direction_sign
from utils import get_dna_hash
from objective_functions import apply_objectives
import xarray as xr
import s3fs



class SatisficingTestManager(Problem):
    def __init__(self, n_var=16, n_obj=2):
        # Setup: 16 parameters, 2 objectives
        super().__init__(n_var=n_var, n_obj=n_obj, xl=-2, xu=2)
        self.thresholds = {'f1': 0.5, 'f2': 0.5}

    def _evaluate(self, x, out, *args, **kwargs):
        # 1. Create a simple synthetic objective (the "truth")
        # Minimizing the distance from the center (0,0)
        f1 = np.sum(x**2, axis=1)
        f2 = np.sum((x-1)**2, axis=1)
        
        # 2. Combine into a DataFrame to reuse your transformation logic
        df = pd.DataFrame({'f1': f1, 'f2': f2})
        
        # 3. Apply your specific satisficing logic
        # This will 'flatten' results worse than the threshold
        for key in self.thresholds:
            df[key] = smooth_non_decreasing_satisficing_min(
                df[key], self.thresholds[key], sharpness=20.0
            )
            
        out["F"] = df.values


import numpy as np

def test_satisficing_smooth_min(values, threshold, sharpness=10.0):
    """
    Tuned for the range of [0, 2] used in your SatisficingTestManager.
    
    - 'values' are the raw outputs from the ZDT1-style test problem.
    - 'threshold' is the target value (e.g., 0.5).
    - 'sharpness' controls how aggressively it flattens the penalty.
    """
    # Softplus-based smoothing
    # This creates a smooth transition at the threshold, 
    # where values > threshold get heavily penalized (flattened),
    # and values < threshold track the original value.
    
    # We use a logistic-style activation to flatten values above the threshold
    penalty = (1.0 / sharpness) * np.log(1 + np.exp(sharpness * (values - threshold)))
    
    # Return the 'satisficed' objective value
    return values + penalty






def smooth_non_decreasing_satisficing_min(values, threshold, sharpness=20.0):
    """
    Normalized for Minimization:
    - Values that are 'better' (more negative) than the threshold stay as-is.
    - Values 'worse' (closer to positive infinity) are flattened at the threshold.
    """
    # 1. Flip signs to convert back to the 'increasing' logic
    # -values becomes the magnitude of the objective
    # -threshold becomes the target
    neg_values = -values
    neg_threshold = -threshold
    
    # 2. Use the same robust logic, now working on the flipped space
    scale = np.abs(neg_threshold) + 1e-6
    rel_delta = (neg_threshold - neg_values) / scale
    
    return - (neg_threshold - (scale / sharpness) * np.log(1 + np.exp(sharpness * rel_delta)))



def one_sided_compressor(values, threshold, strength, sense='max'):
    values = np.array(values, dtype=np.float64)
    # Calibrate sharpness
    s = np.log(np.exp(strength) - 1 + 1e-8) / (0.1 * (np.abs(threshold) + 1e-8))
    
    if sense == 'max':
        # Floor: Flat for values < T, Identity for values > T
        arg = s * (values - threshold)
        # Stable Softplus:
        softplus = np.where(arg > 50, arg, np.log1p(np.exp(np.clip(arg, None, 50))))
        return threshold + (1.0 / s) * softplus
        
    else: # sense == 'min'
        # Ceiling: Identity for values < T, Flat for values > T
        arg = s * (threshold - values)
        # Stable Softplus:
        softplus = np.where(arg > 50, arg, np.log1p(np.exp(np.clip(arg, None, 50))))
        return threshold - (1.0 / s) * softplus
    


class SatisficingTestManager(Problem):
    def __init__(self, n_var=16, n_obj=2):
        # Setup: 16 parameters, 2 objectives
        super().__init__(n_var=n_var, n_obj=n_obj, xl=-2, xu=2)
        self.thresholds = {'f1': 18, 'f2': 31}

    def _evaluate(self, x, out, *args, **kwargs):
        # 1. Create a simple synthetic objective (the "truth")
        # Minimizing the distance from the center (0,0)
        f1 = np.sum(x**2, axis=1)
        f2 = np.sum((x-1)**2, axis=1)
        
        # 2. Combine into a DataFrame to reuse your transformation logic
        df = pd.DataFrame({'f1': f1, 'f2': f2})
        
        # 3. Apply your specific satisficing logic
        # This will 'flatten' results worse than the threshold
        for key in self.thresholds:
            df[key] =  np.minimum(100, test_satisficing_smooth_min(
                df[key], self.thresholds[key], sharpness=20.0
            ))
            
        out["F"] = df.values 


def save_to_zarr(ds, store_path, append_dim='sim_id'):
    fs = s3fs.S3FileSystem()
    
    # Check if the Zarr store already exists on S3
    if fs.exists(store_path):
        # Mode 'a' (append) is standard for existing stores
        ds.to_zarr(store_path, append_dim=append_dim, mode='a')
    else:
        # Mode 'w' (write) is required for the first creation
        ds.to_zarr(store_path, mode='w')  
    
    
class OptimizationManager(Problem):
    def __init__(
            self, 
            sim_manager, 
            objective_dict,
            objective_sense,
            thresholds,
            xl,
            xu,
            output_folder
        ):
        
        self.__dict__.update({k: v for k, v in locals().items() if k != 'self'})
        self.obj_count = len(objective_dict)
        self.var_count = len(xl)
        self.output_folder= output_folder
        self.first_write = True

        
        super().__init__(n_var=self.var_count, n_obj=self.obj_count, n_constr=0, xl=self.xl, xu=self.xu)

    def _evaluate(self, x, out, *args, **kwargs):
        # 1. DELEGATE: SimulationManager handles the heavy lifting
        # Returns the DataFrames as requested
        pop_results = self.sim_manager.evaluate(x)
        keep = [x is not None for x in pop_results]
        pop_results = [x for x in pop_results if x is not None]
        sim_ids = np.array([get_dna_hash(y) for y in x ])[keep]
        
        # 2. AGGREGATE: OptimizationManager controls the math
        # Median is calculated here
        data_metrics_list = []
        for individual_result in pop_results:
            data_list = []
            for perturbation_result in individual_result:
                df_add = apply_objectives(self.objective_dict, perturbation_result)
                data_add = xr.DataArray(df_add.values, dims=['mode', 'objective'], coords={'mode': df_add.index, 'objective': df_add.columns})
                data_list.append(data_add)
            da_metrics = xr.concat(data_list, dim = 'perturbation').median(dim = 'perturbation')
            data_metrics_list.append(da_metrics)
        data_metrics = xr.concat(data_metrics_list, dim = 'sim_id')
        data_metrics.coords['sim_id'] =  sim_ids
        save_to_zarr(data_metrics, "{}/objectives.zarr".format(self.output_folder))

        pop_results_array = np.array([[list(s_permutation.values) for s_permutation in sim_id_result] for sim_id_result in pop_results])
        data_values = xr.DataArray(pop_results_array, coords = {'sim_id': sim_ids, 'perturbation': range(len(pop_results[0])), 'date': pop_results[0][0].index})
        save_to_zarr(data_values, "{}/portfolio_values.zarr".format(self.output_folder))
        
        # 4. REPORT: Tell pymoo the results
        out["F"] = np.full((len(x), len(self.objective_sense)), np.nan)
        out["F"][keep,:] = data_metrics.sel(mode = 'train', objective = list(self.objective_sense.keys())).values
        
        
