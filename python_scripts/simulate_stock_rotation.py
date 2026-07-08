import inspect
import itertools
import logging
import os
import sys
import time

import awswrangler as wr
import numpy as np
import pandas as pd
import pyomo.environ as pyo
from pyomo.common.timing import HierarchicalTimer
from pyomo.contrib.appsi.solvers import Highs
from pyomo.core import *



    
def solve_with_timer(solver, model, max_time):    
    
    # Initialize the timer
    timer = HierarchicalTimer()

    timer.start("Total")

    # --- Model Construction ---
    timer.start("Construction")
    # ... your model building code here ...
    timer.stop("Construction")

    # --- Solving ---
    timer.start("Solver")
    t1 = time.time()
    solver.solve(model, tee=True)
    
    timer.stop("Solver")

    timer.stop("Total")

    # Print the breakdown
    total_time = time.time() - t1
    if total_time > max_time:
        print(timer, flush = True)
        zzz=1


def dataframe_to_dict(R):
    return {(symbol, date) : R.loc[symbol, date] for symbol, date in itertools.product(R.index, R.columns)}

def var_to_dataframe(var):
    records = []
    for index in var:
        val = value(var[index])
        idx_tuple = index if isinstance(index, tuple) else (index,)
        records.append((*idx_tuple, val))
    col_names = [f"dim_{i+1}" for i in range(len(records[0]) - 1)] + ["value"]
    return pd.DataFrame(records, columns=col_names)

def var_to_pivot_table(var, index_name = 'symbol', columns_name = 'account'):
    df = var_to_dataframe(var).rename(columns = {'dim_1': index_name, 'dim_2': columns_name})
    return pd.pivot_table(df, index = index_name, columns = columns_name, values = 'value')



class SuppressOutput:
    def __enter__(self):
        # Save original file descriptors
        self.stdout_fd = os.dup(sys.stdout.fileno())
        self.stderr_fd = os.dup(sys.stderr.fileno())
        
        # Open devnull
        self.devnull = os.open(os.devnull, os.O_WRONLY)
        
        # Redirect stdout/stderr to devnull
        os.dup2(self.devnull, sys.stdout.fileno())
        os.dup2(self.devnull, sys.stderr.fileno())
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original FDs
        os.dup2(self.stdout_fd, sys.stdout.fileno())
        os.dup2(self.stderr_fd, sys.stderr.fileno())
        
        # Close saved FDs and devnull
        os.close(self.stdout_fd)
        os.close(self.stderr_fd)
        os.close(self.devnull)








def optimize(_params, df_features, current_price, holdings, budget, feature_weights, prev_sol=None, max_voo = None):
    
    logging.getLogger('pyomo.util.infeasible').setLevel(logging.INFO)
    budget = np.maximum(0, budget)
    
    params = _params.copy()
    params['feature_values'] = df_features.fillna(0)
    params['current_price'] = current_price
    params['holdings'] = holdings
    params['budget'] = budget
    keep_stocks = (params['current_price'].values > params['min_price']).flatten() | (params['current_price'].index == 'GIC').flatten()
    drop_stock_list = params['current_price'].index[~keep_stocks]
    for key in ['current_price', 'feature_values', 'holdings']:
        params[key] = params[key].loc[keep_stocks]

    model = pyo.ConcreteModel()
    model.stock = pyo.Set(initialize = list(params['current_price'].index))
    model.feature = pyo.Set(initialize = list(params['feature_values'].columns))
    model.account = pyo.Set(initialize = range(len(params['principal'])))
    model.feature_values = pyo.Param(model.stock, model.feature, initialize = dataframe_to_dict(params['feature_values']), domain = pyo.Reals)
    model.current_price = pyo.Param(model.stock, initialize = dict(pd.DataFrame(params['current_price']).iloc[:,0]), domain = pyo.NonNegativeReals)
    model.M = pyo.Param(initialize = 1e7)
    model.holdings = pyo.Param(model.stock, model.account, initialize = dataframe_to_dict(params['holdings']))
    model.budget = pyo.Param(model.account, initialize = params['budget'])
    model.feature_weight = pyo.Param(model.feature, initialize = feature_weights)

    model.x = pyo.Var(model.stock, model.account, domain=pyo.NonNegativeIntegers)
    model.t = pyo.Var(model.stock, model.account, domain=pyo.Binary)

    def budget_constraint(model, a):
        return sum(model.x[s,a] * model.current_price[s] for s in model.stock) + sum(params['trade_fee'] * model.t[s,a] for s in model.stock) <= model.budget[a]
    model.budget_constraint = pyo.Constraint(model.account, rule = budget_constraint)

    def max_frac_constraint(model, stock):
       if stock == 'GIC': 
           return pyo.Constraint.Feasible
       elif stock == 'VOO' and max_voo is not None:
           return sum(model.x[stock, a] * model.current_price[stock] for a in model.account) <= max_voo * sum(params['budget'])
       else:
           return sum(model.x[stock, a] * model.current_price[stock] for a in model.account) <= params['max_frac'] * sum(params['budget'])
    model.max_frac_constraint = pyo.Constraint(model.stock, rule = max_frac_constraint)

    def obj_expression(model):
        return - sum(model.feature_weight[w] * model.feature_values[s, w] * model.x[s,a] for w in model.feature for s in model.stock for a in model.account)
    model.OBJ = pyo.Objective(rule=obj_expression, sense = pyo.minimize)

    # --- WARM START INJECTION ---
    solver = Highs()
    solver.highs_options["time_limit"] = 1.0
    #solver.highs_options["threads"] = 1
    
    # Update the injection block at the start of your optimize function:
    if prev_sol:
        for var_name, values_dict in prev_sol.items():
            if hasattr(model, var_name):
                var_obj = model.find_component(var_name)
                # values_dict is {index: value}
                for index, val in values_dict.items():
                    if val is not None:
                        var_obj[index].set_value(val)

    with SuppressOutput():
        solver.solve(model)
    
    obj_value = pyo.value(model.OBJ)
    
    # --- PHASE 2 ---
    if obj_value < _params['obj_threshold']:
        threshold_value = (1 - params['objective_sensitivity']) * obj_value
        def obj_near_optimal_constraint(model):
            return - sum(model.feature_weight[w] * model.feature_values[s, w] * model.x[s,a] for w in model.feature for s in model.stock for a in model.account) <= threshold_value
        model.near_optimal_constraint = pyo.Constraint(rule = obj_near_optimal_constraint)
        model.del_component(model.OBJ)
        model.num_trades_obj = pyo.Objective(rule=lambda m: sum(m.t[s,a] for s in m.stock for a in m.account), sense = pyo.minimize)
        
        with SuppressOutput(): 
            solver.solve(model)
        
        df_sol = var_to_pivot_table(model.x).loc[params['current_price'].index, :]
    else:
        df_sol = pd.DataFrame(0, index = params['current_price'].index, columns = range(len(params['principal'])))

    _investment = np.matmul(pd.DataFrame(params['current_price']).loc[params['current_price'].index != 'GIC'].transpose().values, df_sol.loc[df_sol.index != 'GIC'].values)
    num_trades = (df_sol != params['holdings']).values.sum()
    df_sol = df_sol.astype('float')
    df_sol.loc['GIC', :] = (params['budget'] - _investment - float(params['trade_fee'] * num_trades)).flatten()
    df_sol = pd.concat([df_sol, pd.DataFrame(0, index = drop_stock_list, columns = df_sol.columns)]).loc[holdings.index]
    
    # Capture new solution for next loop
   # Replace the old new_sol line with this:
    new_sol = {}
    for v in model.component_objects(pyo.Var):
        # .get_values() returns a dictionary of {index: value}
        # We store this under the variable's name
        new_sol[v.name] = v.get_values()
    
    return df_sol, num_trades * params['trade_fee'], obj_value, new_sol




        
        
        
        
def simulate(df_price, _params, data_features, df_weights, period, sim_id = None, session = None, holdings = None, max_voo = None):
    
    params = _params.copy()
    params.update(period)
    val_start_dates = df_weights.index[:-1]
    val_end_dates = df_weights.index[1:]
    time_tups = list(zip(val_start_dates, val_end_dates))
    
    if holdings is None:
        holdings = pd.DataFrame(0.0, index = df_price.index, columns = range(len(params['principal'])))
        holdings.loc['GIC', :] = params['principal']

    
    
    history = []
    df_holdings_history = pd.DataFrame()
    df_holdings_shares = pd.DataFrame()
    last_val_start_date = time_tups[0][0]
    for val_start_date, val_end_date in time_tups:
        
        holdings_shares = pd.DataFrame((holdings.sum(axis = 1))).transpose()
        holdings_shares.loc[:, 'date'] = val_start_date
        df_holdings_shares = pd.concat([df_holdings_shares, holdings_shares])
        
    
        current_price = df_price.loc[:, val_start_date].copy(); current_price.loc['GIC'] = 1
        budget = (holdings.values * pd.DataFrame(current_price).fillna(0).values).sum(axis = 0)
        df_features = data_features.sel(date = val_start_date).to_pandas().transpose()
        feature_weights = dict(df_weights.loc[val_start_date])
        new_sol = None
        holdings, _, _  , new_sol = optimize(params, df_features, current_price, holdings, budget, feature_weights, new_sol, max_voo = max_voo)
        
        holdings_out = pd.DataFrame((holdings.sum(axis = 1).values * current_price)).transpose().reset_index().rename(columns = {'index': 'date'})
        holdings_out.loc[:, 'sim_id'] = sim_id

        df_holdings_history = pd.concat([df_holdings_history, holdings_out])
        
        

        if val_end_date in data_features.date:
            gic_multiplier = 1 + np.array(data_features.sel(symbol = 'GIC', date = val_end_date, band = 'dollar_ret_1p'))
        else:
            gic_multiplier = 1.025**(1/13)

        stocks = sorted(list(set(holdings.index).difference(['GIC'])))
        v_start = (holdings.loc[stocks] * df_price.loc[stocks, [val_start_date]].values).sum(axis=0) + holdings.loc['GIC']
        v_end = (holdings.loc[stocks] * df_price.loc[stocks, [val_end_date]].values).sum(axis=0) + gic_multiplier * holdings.loc['GIC']
        history.append((val_start_date, val_end_date, v_start.sum(), v_end.sum()))
        gic_frac = holdings.loc['GIC'].values.sum()/v_start.sum()
        print(sim_id, val_start_date, val_end_date, v_start.sum(), v_end.sum(), gic_frac, flush = True)
        
    stagger_delay = (int(sim_id, 16) % 5000) / 1000.0
    time.sleep(stagger_delay)
    
    
    return df_holdings_history

def get_gic_eps(data_gic):
    df_gic = data_gic.sel(symbol = 'GIC').to_pandas().transpose().iloc[:, 1:]
    df_gic['avg_eps_1q'] = (1 + df_gic.dollar_ret_1p)**((365/4)/(28))-1
    df_gic['avg_eps_2q'] = (1 + df_gic.dollar_ret_6p)**((365/2)/(6*28))-1
    df_gic['avg_eps_4q'] = (1 + df_gic.dollar_ret_13p)**(365/(13*28))-1
    df_gic['avg_eps_8q'] = (1 + df_gic.dollar_ret_26p)**((2*365)/(26*28))-1
    return df_gic[[c for c in df_gic.columns if 'eps' in c]].stack().to_xarray().expand_dims(symbol=['GIC']).transpose('band','symbol','date')

def interpolate_to_4week_grid(da, anchor_date):
    days = (pd.to_datetime(da.date) - pd.to_datetime(anchor_date)).days
    da_numeric = da.assign_coords(date=days)
    new_coords = np.arange(int(days.min()//28)*28, int(days.max()//28+1)*28, 28)
    da_interp = da_numeric.interp(date=new_coords, method="linear")
    return da_interp.assign_coords(date=pd.to_datetime(anchor_date) + pd.to_timedelta(da_interp.date.values, unit='D'))

# --- Main Cluster Exploration Script ---

