import numpy as np
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

import warnings
from pymoo.core.problem import Problem

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.spatial import KDTree
from scipy.stats import qmc

# --- Third-Party ---
import pandas as pd
from pymoo.core.individual import Individual
from pymoo.core.population import Population
from pymoo.core.problem import Problem
from pymoo.indicators.hv import HV

import joblib

import io
import boto3

from urllib.parse import urlparse

import numpy as np
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

import numpy as np


class FastStackedSurrogate:
    def __init__(self, n_vars, rf_buffer_size=500):
        self.X_history = np.zeros((0, n_vars))
        self.F_history = np.zeros((0,1))
        self.rf_buffer_size = rf_buffer_size
        
        # Parallel Base Model: Global anchor with warm_start
        self.base_model = RandomForestRegressor(
            n_estimators=100, 
            warm_start=True, 
            n_jobs=-1,  # Parallelize tree building
            random_state=42
        )
        
        self.residual_booster = None
        # Standard XGBoost Booster (Non-DART)
        self.res_params = {
            'objective': 'reg:squarederror',
            'nthread': -1,        # Parallelize boosting rounds
            'max_depth': 3,
            'learning_rate': 0.05,
            'reg_lambda': 10.0,
            'min_child_weight': 5
        }
        
      

    def fit(self, train_x, train_y, **kwargs):
        train_x = np.array(train_x)
        train_y = np.array(train_y).reshape(-1, 1)
        
        # 1. Update full history
        self.X_history = np.vstack([self.X_history, train_x])
        self.F_history = np.vstack([self.F_history, train_y])
        
        # 2. RF Buffer Logic
        if len(self.X_history) > self.rf_buffer_size:
            indices = np.arange(len(self.X_history))[-self.rf_buffer_size:]
            x_rf, y_rf = self.X_history[indices], self.F_history[indices]
        else:
            x_rf, y_rf = self.X_history, self.F_history
            
        # 3. Complexity Cap
        MAX_ESTIMATORS = 300 
        if self.base_model.n_estimators < MAX_ESTIMATORS:
            self.base_model.n_estimators += 10
            
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.ensemble")
            self.base_model.fit(x_rf, y_rf.ravel())
        
        # 4. Calculate Residuals
        rf_preds = self.base_model.predict(self.X_history).reshape(-1, 1)
        residuals = self.F_history - rf_preds
        
        # 5. Standard Incremental Boosting
        dtrain = xgb.DMatrix(self.X_history, label=residuals)
        self.residual_booster = xgb.train(
            params=self.res_params,
            dtrain=dtrain,
            num_boost_round=20, 
            xgb_model=self.residual_booster # Warm-start the booster
        )

    def predict(self, x):
        x = np.atleast_2d(x)
        base_pred = self.base_model.predict(x).reshape(-1, 1)
        res_pred = self.residual_booster.predict(xgb.DMatrix(x)).reshape(-1, 1)
        return base_pred + res_pred
    


class HeterogeneousEnsemble:
    def __init__(self, models):
        self.models = models
        
        # Determine capability at initialization
        self.is_bayesian = all(hasattr(m, 'posterior') and callable(m.posterior) 
                               for m in self.models)

    def fit(self, X, F, n_epochs=5, **kwargs):
        for i, model in enumerate(self.models):
            model.fit(X, F[:, i:i+1], n_epochs=n_epochs, **kwargs)

    def predict(self, X):
        return np.column_stack([model.predict(X) for model in self.models])

    def posterior(self, X, **kwargs):
        """
        Returns a PosteriorList if models are Bayesian.
        The user is responsible for checking .is_bayesian before calling.
        """
        return PosteriorList(*[m.posterior(X, **kwargs) for m in self.models])


# --- 3. SURROGATE PROBLEM (Validator) ---
class SurrogateProblem(Problem):
    def __init__(self, model, master_problem):
        # Contract validation
        
        
        super().__init__(n_var=master_problem.n_var, n_obj=master_problem.n_obj, 
                         xl=master_problem.xl, xu=master_problem.xu, elementwise_evaluation=False)
        self.model = model

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = self.model.predict(x)

    def fit(self, X, F, n_epochs=5, **kwargs):
        self.model.fit(X, F, n_epochs=n_epochs, **kwargs)


def calculate_distance_to_pareto(Z_trace):
    """
    Calculates Euclidean distances in the Z-normalized space.
    Z_trace: (N, n_obj) normalized objective values
    """
    nds = NonDominatedSorting()
    # Find points on the front using normalized values
    front_indices = get_pareto_front_indices(Z_trace)
    pareto_front = Z_trace[front_indices]
    
    # Distance calculation remains the same, but now operates on Z-scores
    diff = Z_trace[:, np.newaxis, :] - pareto_front[np.newaxis, :, :]
    dist_matrix = np.linalg.norm(diff, axis=2)
    distances = np.min(dist_matrix, axis=1)
    
    return distances

def get_pareto_front_indices(costs):
    """
    Highly optimized Pareto front identification in O(N log N).
    """
    # 1. Sort points by first objective (ascending)
    # If first objectives are equal, sort by second, etc.
    indices = np.lexsort((costs[:, 1], costs[:, 0]))
    sorted_costs = costs[indices]
    
    # 2. Identify non-dominated points
    is_efficient = np.ones(len(costs), dtype=bool)
    
    # For a point to be dominated, it must be worse in all objectives 
    # than a point already seen. In a sorted list, we only need to 
    # check the current minimum of the subsequent objectives.
    min_so_far = np.full(costs.shape[1], np.inf)
    
    for i in range(len(sorted_costs) - 1, -1, -1):
        # If the point is worse than the current minimum in ANY objective, 
        # it is dominated (assuming minimization)
        if np.any(sorted_costs[i] > min_so_far):
            is_efficient[indices[i]] = False
        else:
            min_so_far = np.minimum(min_so_far, sorted_costs[i])
            
    return is_efficient


def get_validation_indices_from_weights(weights, val_ratio=0.1):
    """
    Selects validation indices using existing weights.
    
    weights: (N,) array of importance weights
    val_ratio: Proportion of total data to use for validation
    """
    n_samples = len(weights)
    n_val = int(n_samples * val_ratio)
    
    # Normalize weights to create a valid probability distribution
    probs = weights / weights.sum()
    
    # Sample indices based on the probability distribution
    # This keeps your validation set "density-matched" to your training focus
    val_indices = np.random.choice(
        n_samples, 
        size=n_val, 
        replace=False, 
        p=probs
    )
    
    return val_indices
    
    
class ResidualBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        # Each block performs a non-linear transformation
        self.block = nn.Sequential(
            nn.Linear(d, d),
            nn.Softplus(),
            nn.Linear(d, d),
            nn.Softplus()
        )


    def forward(self, x):
        # The "residual" path: Adds the original input to the transformed version
        return x + self.block(x)
    

class SurrogateManifoldProblem(Problem):
    def __init__(self, n_var, n_obj, xl, xu, expected_mean, expected_std, alpha = 3.0, saved_model_path = None):
        
        if saved_model_path:
            # We use the static load method and update the current instance's __dict__
            # to point to the loaded object's attributes
            loaded_obj = SurrogateManifoldProblem.load(saved_model_path)
            self.__dict__.update(loaded_obj.__dict__)
            return
       
        
        
        super().__init__(n_var=n_var, n_obj=n_obj, xl=-5.0, xu=5.0, elementwise_evaluation=False)
        self.alpha = alpha
        self.n_var = n_var
        self.n_obj = n_obj
        self.xl = xl
        self.xu = xu
        self.expected_mean = expected_mean
        self.expected_std = expected_std
        

        
        self.X_trace = np.empty((0, self.n_var))
        self.F_trace = np.empty((0, self.n_obj))
        
       
        # 2. Build and Train the Oracle
        self.model = self._build_model(self.n_var, self.n_obj)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.005)
        self.criterion = torch.nn.MSELoss(reduction='none')
        

        
        
        

    def _build_model(self, in_d, out_d):
        """
        Builds a deeper, residual-based network for better representation 
        learning in 16D space.
        """
        return nn.Sequential(
            nn.Linear(in_d, 256),
            nn.Softplus(),
            ResidualBlock(256),
            ResidualBlock(256),
            nn.Linear(256, out_d)
        )



    def fit(self, X, F, epochs=2000, min_epochs = 200):
        
        
        self.X_trace = np.vstack([self.X_trace, X])
        self.F_trace = np.vstack([self.F_trace, F])
        self.Z_trace = (self.F_trace - self.expected_mean)/self.expected_std
        
        
    
        
        
        distances = calculate_distance_to_pareto(self.Z_trace)

        # Normalize distances to [0, 1] for stable alpha selection
        dist_norm = (distances - distances.min()) / (distances.max() - distances.min() + 1e-8)

        # Alpha controls the focus. A higher alpha (e.g., 5.0) makes the model 
        # care EXCLUSIVELY about the frontier. A lower alpha (e.g., 1.0) is smoother.
        self.weights = np.exp(-self.alpha * dist_norm) 
       
        
        
        
        
        # 1. Setup Data Splits
        val_indices = get_validation_indices_from_weights(self.weights, val_ratio=0.1)
        all_indices = np.arange(len(self.weights))
        train_indices = np.setdiff1d(all_indices, val_indices)
        
        X_train, F_train, W_train = torch.FloatTensor(X[train_indices]), torch.FloatTensor(self.Z_trace[train_indices]), torch.FloatTensor(self.weights[train_indices]).view(-1, 1)
        X_val = torch.FloatTensor(X[val_indices])
        raw_targets_train = torch.FloatTensor(self.F_trace[train_indices])
        raw_targets_val = torch.FloatTensor(self.F_trace[val_indices])
        
        # Stopping criteria state
        best_val_mads = np.full(F_train.shape[1], np.inf)
        patience, epochs_no_improve = 20, 0
        divergence_threshold = 1.05 # 5% degradation limit
        
        for epoch in range(max_epochs := 2000 if epochs is None else epochs):
            self.optimizer.zero_grad()
            predictions_train = self.model(X_train)
            loss = (self.criterion(predictions_train, F_train) * W_train).mean()
            loss.backward()
            self.optimizer.step()
            
            with torch.no_grad():
                preds_val = self.model(X_val)
                preds_train = self.model(X_train)
                
                scaled_val = torch.FloatTensor(self.expected_mean) + (preds_val * torch.FloatTensor(self.expected_std))
                scaled_train = torch.FloatTensor(self.expected_mean) + (preds_train * torch.FloatTensor(self.expected_std))
                
                train_mad = torch.mean(torch.abs(scaled_train - raw_targets_train), dim=0).cpu().numpy()
                val_mad = torch.mean(torch.abs(scaled_val - raw_targets_val), dim=0).cpu().numpy()
                
                gap_ratio = val_mad / (train_mad + 1e-8)
                divergence_threshold = 1.5 # Adjust based on how much "gap" you tolerate
                if epoch > min_epochs and np.any(gap_ratio > divergence_threshold):
                    print(f"Overfitting/Divergence detected: ValMAD is {divergence_threshold}x higher than TrainMAD.")
                    break
                
                # Check 2: Plateau (No improvement across all objectives)
                if np.any(val_mad < best_val_mads):
                    best_val_mads = np.minimum(best_val_mads, val_mad)
                    epochs_no_improve = 0
                    torch.save(self.model.state_dict(), 'best_model.pth')
                else:
                    epochs_no_improve += 1
            
            if epoch % 10 == 0:
                
                outputs = [epoch] + list(np.vstack((train_mad, val_mad)).transpose().flatten())     
                print("Epoch: {}   TrainMAD/ValMAD: {}/{} {}/{} ".format(*outputs))
                
            if epoch > min_epochs and epochs_no_improve >= patience:
                print(f"Plateau reached at epoch {epoch}. Stopping.")
                self.model.load_state_dict(torch.load('best_model.pth'))
                break
            zzz=1


    def predict(self, x):
        with torch.no_grad():
            result = self.model(torch.from_numpy(x).float()).detach().numpy()
        return result

    def _evaluate(self, x, out, *args, **kwargs):
        
        
        # 3. Predict via Oracle
        with torch.no_grad():
            out["F"] = self.model(x).numpy()

    

    def sample(self, n_samples=500):
        sampler = qmc.LatinHypercube(d=self.n_var)
        X = qmc.scale(sampler.random(n=n_samples), self.xl, self.xu).astype(np.float32)
        out = {}
        self._evaluate(X, out)
        return X, out["F"]
    

    


    def save(self, filepath_base):
        # 1. Prepare data (remove model from pickle)
        temp_model = self.model
        self.model = None
        
        # 2. Check if S3
        if filepath_base.startswith("s3://"):
            parsed = urlparse(filepath_base)
            bucket = parsed.netloc
            key_prefix = parsed.path.lstrip('/')
            s3 = boto3.client('s3')
            
            # Save Model
            model_buf = io.BytesIO()
            torch.save(temp_model.state_dict(), model_buf)
            s3.put_object(Bucket=bucket, Key=f"{key_prefix}_model.pth", Body=model_buf.getvalue())
            
            # Save Metadata
            meta_buf = io.BytesIO()
            joblib.dump(self, meta_buf)
            s3.put_object(Bucket=bucket, Key=f"{key_prefix}_meta.pkl", Body=meta_buf.getvalue())
        else:
            # Local save
            torch.save(temp_model.state_dict(), f"{filepath_base}_model.pth")
            joblib.dump(self, f"{filepath_base}_meta.pkl")
            
        # Restore model
        self.model = temp_model

    @staticmethod
    def load(filepath_base, X_trace, F_trace):
        model_state = None
        obj = None
        
        # 1. Check if S3
        if filepath_base.startswith("s3://"):
            parsed = urlparse(filepath_base)
            bucket = parsed.netloc
            key_prefix = parsed.path.lstrip('/')
            s3 = boto3.client('s3')
            
            # Load Metadata
            meta_obj = s3.get_object(Bucket=bucket, Key=f"{key_prefix}_meta.pkl")
            obj = joblib.load(io.BytesIO(meta_obj['Body'].read()))
            
            # Load Model
            model_obj = s3.get_object(Bucket=bucket, Key=f"{key_prefix}_model.pth")
            model_state = torch.load(io.BytesIO(model_obj['Body'].read()))
        else:
            # Local load
            obj = joblib.load(f"{filepath_base}_meta.pkl")
            model_state = torch.load(f"{filepath_base}_model.pth")
            
        # 2. Reconstruct Model
        obj.model = obj._build_model(obj.X_trace.shape[1], obj.F_trace.shape[1])
        obj.model.load_state_dict(model_state)
        obj.model.eval()
        
        return obj