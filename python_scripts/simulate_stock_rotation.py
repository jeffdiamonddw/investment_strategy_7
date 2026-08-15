import inspect
import itertools
import logging
import os
import sys
import time
import joblib

import awswrangler as wr
import numpy as np
import pandas as pd
import pyomo.environ as pyo
from pyomo.common.timing import HierarchicalTimer
from pyomo.contrib.appsi.solvers import Highs
from pyomo.core import *


import pandas as pd
import pyomo.environ as pyo

import pandas as pd
import numpy as np
import xarray as xr


def allocate_stocks_heuristic(
    B: pd.Series, P: pd.Series, account_values: list
) -> pd.DataFrame:
    """Heuristically allocates buy orders (B) across accounts based on stock prices (P)

    and account total dollar values to maximize future flexibility and handle integer constraints.

    Parameters:
    - B: pandas Series of total shares to buy per stock (index = ticker)
    - P: pandas Series of stock prices (index = ticker)
    - account_values: list of floats representing the total dollar amount in each account

    Returns:
    - b_df: pandas DataFrame of allocated integer buy shares (index = ticker, columns = account integers 0, 1, 2...)
    """
    # Setup data structures
    tickers = B.index.tolist()
    num_accounts = len(account_values)
    account_indices = list(range(num_accounts))

    # Initialize allocation DataFrame with zeros
    b_df = pd.DataFrame(0, index=tickers, columns=account_indices)

    # Track remaining cash/capacity for each account
    remaining_cash = list(account_values)
    total_cash = sum(account_values)

    # Sort stocks by price in descending order (highest price first to allocate to larger accounts)
    sorted_tickers = P.sort_values(ascending=False).index.tolist()

    # Sort accounts by size in descending order (largest account first)
    sorted_accounts = sorted(
        account_indices, key=lambda k: account_values[k], reverse=True
    )

    # Step 1: Greedy allocation favoring large accounts for high-priced stocks
    for i in sorted_tickers:
        shares_needed = int(B[i])
        if shares_needed <= 0:
            continue

        price = P[i]

        # Try to fulfill this stock's buy order by distributing across accounts,
        # giving preference to larger accounts first.
        for k in sorted_accounts:
            if shares_needed <= 0:
                break

            # Calculate how many shares this account can afford based on remaining cash
            max_shares_affordable = int(remaining_cash[k] // price)

            if max_shares_affordable > 0:
                # Take as many as needed or as many as can be afforded
                shares_to_allocate = min(shares_needed, max_shares_affordable)

                # Assign shares
                b_df.loc[i, k] += shares_to_allocate
                remaining_cash[k] -= shares_to_allocate * price
                shares_needed -= shares_to_allocate

        # If any shares remain unallocated due to strict cash limits,
        # dump the remainder into the largest account anyway to satisfy total demand (B_i)
        if shares_needed > 0:
            largest_account = sorted_accounts[0]
            b_df.loc[i, largest_account] += shares_needed
            remaining_cash[largest_account] -= shares_needed * price

    # Step 2: Ensure exact totals match original B vector (repair step for any rounding/spillover adjustments)
    current_totals = b_df.sum(axis=1)
    for i in tickers:
        diff = int(B[i] - current_totals[i])
        if diff != 0:
            # Adjust using the largest account
            largest_account = sorted_accounts[0]
            b_df.loc[i, largest_account] += diff

    return b_df



def solve_stock_allocation(
    _S: pd.Series, _B: pd.Series, _P: pd.Series, _H: pd.DataFrame, M: float = 1000000.0, trade_cost = 7
):
    """Solves the stock allocation MILP problem using Pyomo and CBC.

    Parameters:
    - S: pandas Series of total shares to sell per stock (index = ticker)
    - B: pandas Series of total shares to buy per stock (index = ticker)
    - P: pandas Series of stock prices (index = ticker)
    - H: pandas DataFrame representing the current number of shares for each stock in each account (index = ticker, columns = account integers)
    - M: Big-M constant
    - trade_cost: cost of each trade

    Returns:
    - sold: pandas DataFrame of allocated sell shares 
    - bought: pandas DataFrame of allocated buy shares 
    """

    out = {'S': _S, 'B':_B, 'P':_P, 'H': _H}
    with open('temp/allocation_test.joblib','wb') as fp:
        joblib.dump(out, fp)

    _H = np.maximum(0, _H)

    target_buy = (_B * _P.values).sum()
    target_sell = (_S * _P.values).sum()



    sell_stocks = _S.index[_S.values>0]
    single_account_stocks = _H.index[(_H>0).sum(1) == 1]
    single_account_sell = list(set(single_account_stocks).intersection(sell_stocks))

    cash = pd.Series({account: 0.0 for account in _H.columns})
    sold = 0 * _H
    num_trades = 0
    for stock in single_account_sell:
        account = np.where(_H.loc[stock] > 0)[0][0]
        _H.loc[stock, account] -= _S.loc[stock]
        cash.loc[account] += _S.loc[stock] * _P.loc[stock]
        sold.loc[stock, account] += _S.loc[stock]
        _S.loc[stock] = 0
        num_trades+=1
        

    sell_all_stocks = _H.index[(_H.sum(1) == _S) & (_S.values > 0)]
    for stock in sell_all_stocks:
        for account in _H.columns:
            if _H.loc[stock, account] > 0:
                cash[account] += _H.loc[stock, account] * _P.loc[stock]
                sold.loc[stock, account] += _H.loc[stock, account]
                _S.loc[stock] -= _H.loc[stock, account] 
                _H.loc[stock, account] = 0
                num_trades+=1 

    keep = (_S.values>0) | (_B.values > 0)
    H = _H.copy().loc[keep]
    P = _P.copy().loc[keep]
    S = _S.copy().loc[keep].astype(int)
    B = _B.copy().loc[keep].astype(int)


                    

    stocks = H.index.tolist()
    accounts = H.columns.tolist()

    # Initialize Pyomo model
    model = pyo.ConcreteModel()

    model.I = pyo.Set(initialize=stocks)
    model.K = pyo.Set(initialize=accounts)

    # Variables
    model.x = pyo.Var(model.I, model.K, domain=pyo.Binary)
    model.s = pyo.Var(model.I, model.K, domain=pyo.NonNegativeIntegers)
    model.b = pyo.Var(model.I, model.K, domain=pyo.NonNegativeIntegers)


    model.S = pyo.Param(model.I, initialize = {key: int(S.loc[key]) for key in S.index}, domain = pyo.NonNegativeIntegers)
    model.B = pyo.Param(model.I,  initialize = {key: int(B.loc[key]) for key in B.index}, domain = pyo.NonNegativeIntegers)
    model.P = pyo.Param(model.I,  initialize = {key: P.loc[key] for key in P.index}, domain = pyo.NonNegativeReals)

    # Objective: maximize the value of the stocks we can actually buy 
    model.obj = pyo.Objective(
        expr=sum(P[i] * model.b[i, k] for i in model.I for k in model.K), sense=pyo.maximize
    )

    # Constraints
    #1. s_ik <= H_ik  (can't sell more than we have for any account)
    model.c1 = pyo.Constraint(
    model.I, model.K, rule=lambda m, i, k: m.s[i, k] <= H.loc[i, k]
    )

    # 2. s_ik <= M * x_ik  can't sell any if no transactions
    model.c2 = pyo.Constraint(
        model.I, model.K, rule=lambda m, i, k: m.s[i, k] <= M * m.x[i, k]
    )

    # 3. b_ik <= M * x_ik (can't buy any if no transactions)
    model.c3 = pyo.Constraint(
        model.I, model.K, rule=lambda m, i, k: m.b[i, k] <= M * m.x[i, k]
    )

    #4. sum_ik b_ik * P_i +  trade_cost * sum_ik x_ik <= sum_ik s_ik * P_i
    # sum_ik (b_ik * Pi + trade_cost *x_ik - s_ik * P_i) <= 0
    #total of what we buy + trade costs must be <= what we make by selling
    model.c4 = pyo.Constraint(
        model.K,
    rule = lambda m, k: sum(m.b[i,k] * P[i] + trade_cost * m.x[i,k] - m.s[i,k] * P[i]   for i in m.I ) <= cash[k]
    )

    # 5. sum_k s_ik = S_i  (sell everything we want to)
    model.c5 = pyo.Constraint(
        model.I, rule=lambda m, i: sum(m.s[i, k] for k in m.K) <= model.S[i]
    )

    #6. sum_k b_ik <= B_i (don't buy more than we want to)
    model.c6 = pyo.Constraint(
        model.I, rule=lambda m, i: sum(m.b[i, k] for k in m.K) <= model.B[i]
    )

    # Solve
    solver = pyo.SolverFactory("cbc")
    solver.options["seconds"] = 10
    solver.options["ratioGap"] = 1 - .01

    results = solver.solve(model, tee=False)

    # Check if solved successfully and optimally
    is_optimal = (
        results.solver.status == pyo.SolverStatus.ok
        and results.solver.termination_condition
        == pyo.TerminationCondition.optimal
    )


    

    # Extract results into dataframes with the exact shape/index/columns of H
    s_data = {
        k: {i: pyo.value(model.s[i, k]) for i in stocks} for k in accounts
    }
    b_data = {
        k: {i: pyo.value(model.b[i, k]) for i in stocks} for k in accounts
    }

    s_df = pd.DataFrame(s_data, index=stocks)[accounts]
    b_df = pd.DataFrame(b_data, index=stocks)[accounts]

    total_bought = (b_df * P.values.reshape(len(P),1)).values.sum()

    print((target_buy - total_bought)/target_buy)

    #7 sum_ik b_ik P_i >= .99 * total_bought
    model.c7 = pyo.Constraint(
        rule = lambda m: sum( model.b[i,k] * P[i] for i in model.I for k in model.K) >= .99 * total_bought
    )
    model.del_component(model.obj)
    model.obj = pyo.Objective(
        expr=sum( model.x[i, k] for i in model.I for k in model.K), 
        sense=pyo.minimize
    )

    results = solver.solve(model, tee=False)


    #Extract results into dataframes with the exact shape/index/columns of H
    s_data = {
        k: {i: pyo.value(model.s[i, k]) for i in stocks} for k in accounts
    }
    b_data = {
        k: {i: pyo.value(model.b[i, k]) for i in stocks} for k in accounts
    }

    s_df = pd.DataFrame(s_data, index=stocks)[accounts]
    b_df = pd.DataFrame(b_data, index=stocks)[accounts]

    sold.loc[s_df.index] += s_df
    bought = 0 * sold
    bought.loc[b_df.index] = b_df.values
    total_sold = (sold * _P.values.reshape(len(_P),1)).values.sum()
    total_bought = (b_df * P.values.reshape(len(P),1)).values.sum()
    target_buy = (_B * _P.values).sum()
    
    #print((target_buy - total_bought)/target_buy)
    #print((target_sell - total_sold)/target_sell)
    #print((sold >0).values.sum() + (b_df>0).values.sum())

    return sold, bought
    
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








def optimize(_params, df_features, current_price, _holdings, _budget, feature_weights, prev_sol=None, max_voo = None):
    
    df_scores =pd.DataFrame(df_features.values/current_price.values.reshape(current_price.shape[0],1), index = df_features.index)
    s_scores = pd.Series(np.matmul(df_scores.values, np.array(list(feature_weights.values())).reshape(len(feature_weights), 1)).flatten(), index = df_scores.index)

    holdings = pd.DataFrame(_holdings.sum(axis = 1))
    budget = [sum(_budget)]

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
    model.account = pyo.Set(initialize = range(len(budget)))
    model.feature_values = pyo.Param(model.stock, model.feature, initialize = dataframe_to_dict(params['feature_values']), domain = pyo.Reals)
    model.current_price = pyo.Param(model.stock, initialize = dict(pd.DataFrame(params['current_price']).iloc[:,0]), domain = pyo.NonNegativeReals)
    model.M = pyo.Param(initialize = 1e7)
    #model.holdings = pyo.Param(model.stock, model.account, initialize = dataframe_to_dict(holdings))
    
    model.holdings = pyo.Param(
        model.stock,
        model.account,
        initialize={(s, a): holdings.loc[s, a] for s in model.stock for a in model.account},
    )
    model.budget = pyo.Param(model.account, initialize = budget)
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

    solver = pyo.SolverFactory('cbc')
    solver.options["seconds"] = 1
    solver.options["threads"] = 1  # Force CBC to use exactly one core
    solver.options["ratioGap"] = 1 - params['objective_sensitivity']

   

    
    

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
        df_sol = pd.DataFrame(0, index = params['current_price'].index, columns = range(len(budget)))

    _investment = np.matmul(pd.DataFrame(params['current_price']).loc[params['current_price'].index != 'GIC'].transpose().values, df_sol.loc[df_sol.index != 'GIC'].values)
    num_trades = (df_sol != params['holdings']).values.sum()
    df_sol = df_sol.astype('float')
    df_sol.loc['GIC', :] = (params['budget'] - _investment - float(params['trade_fee'] * num_trades)).flatten()
    df_sol = pd.concat([df_sol, pd.DataFrame(0, index = drop_stock_list, columns = df_sol.columns)]).loc[holdings.index]
    df_sol = df_sol.astype(int)
    
    initial_investment = _holdings.loc[_holdings.index != 'GIC'].values.sum()
    if initial_investment == 0:
        df_allocation = allocate_stocks_heuristic(df_sol[0], current_price, _budget)
        num_trades = (df_allocation != _holdings).values.sum()
    else:
        B = np.maximum(0, df_sol - holdings.sum(1).values.reshape(df_sol.shape[0],1))[0]  #what to buy
        S = np.maximum(0, holdings.sum(1).values.reshape(df_sol.shape[0],1) - df_sol)[0] #what to sell
        sold, bought = solve_stock_allocation(S, B, current_price, _holdings)
        transaction_costs = _params['trade_fee'] * ((sold >0).sum(0) + (bought>0).sum(0))
        cash = ((sold  - bought).values * current_price.values.reshape(current_price.shape[0], 1)).sum(0) - transaction_costs
        bought.loc['GIC'] += cash.values.astype(int)
        df_allocation = _holdings + bought - sold
        num_trades = ((sold>0) | (bought > 0)).values.sum()
        obj_value = None
        
    
    return df_allocation, num_trades * params['trade_fee'], obj_value




        
        
        
        
def simulate(df_price, _params, data_features, df_weights, period, sim_id = None, session = None, holdings = None, max_voo = None):
    
    params = _params.copy()
    params.update(period)
    val_start_dates = df_weights.index[:-1]
    val_end_dates = df_weights.index[1:]
    time_tups = list(zip(val_start_dates, val_end_dates)) + [(val_end_dates[-1], None)]
    
    if holdings is None:
        holdings = pd.DataFrame(0.0, index = df_price.index, columns = range(len(params['principal'])))
        holdings.loc['GIC', :] = params['principal']
    holdings_start = holdings.copy()
    
    
    history = []
    df_holdings_history = pd.DataFrame()
    df_holdings_shares = pd.DataFrame()
    last_val_start_date = time_tups[0][0]
    new_sol = None
    all_holdings = {}
    for val_start_date, val_end_date in time_tups:
        
        if val_start_date > max(data_features.coords['date'].to_pandas()):
            break

        if val_end_date is None:
            holdings = holdings_start
        
        all_holdings[val_start_date] = holdings
        holdings_shares = pd.DataFrame((holdings.sum(axis = 1))).transpose()
        holdings_shares.loc[:, 'date'] = val_start_date
        df_holdings_shares = pd.concat([df_holdings_shares, holdings_shares])
        
    
        current_price = df_price.loc[:, val_start_date].copy(); current_price.loc['GIC'] = 1
        budget = (holdings.values * pd.DataFrame(current_price).fillna(0).values).sum(axis = 0)
        df_features = data_features.sel(date = val_start_date).to_pandas().transpose()
        feature_weights = dict(df_weights.loc[val_start_date])
        
        holdings,   _, _  = optimize(params, df_features, current_price, holdings, budget, feature_weights, new_sol, max_voo = max_voo)
        
        if val_end_date is None:
            final_holdings = holdings

        holdings_out = pd.DataFrame((holdings.sum(axis = 1).values * current_price)).transpose().reset_index().rename(columns = {'index': 'date'})
        holdings_out.loc[:, 'sim_id'] = sim_id

        df_holdings_history = pd.concat([df_holdings_history, holdings_out])
        
        

        if val_end_date in data_features.date:
            gic_multiplier = 1 + np.array(data_features.sel(symbol = 'GIC', date = val_end_date, band = 'dollar_ret_1p'))
        else:
            gic_multiplier = 1 + np.array(data_features.sel(symbol = 'GIC', date = val_start_date, band = 'dollar_ret_1p'))

        stocks = sorted(list(set(holdings.index).difference(['GIC'])))
        v_start = (holdings.loc[stocks] * df_price.loc[stocks, [val_start_date]].values).sum(axis=0) + holdings.loc['GIC']

        if val_end_date is not None:
            v_end = (holdings.loc[stocks] * df_price.loc[stocks, [val_end_date]].values).sum(axis=0) + gic_multiplier * holdings.loc['GIC']
            history.append((val_start_date, val_end_date, v_start.sum(), v_end.sum()))
            gic_frac = holdings.loc['GIC'].values.sum()/v_start.sum()
            voo_frac = holdings.loc['VOO'].values.sum()/v_start.sum()
            print(sim_id, val_start_date, val_end_date, v_start.sum(), v_end.sum(), gic_frac, voo_frac, flush = True)
        
    stagger_delay = (int(sim_id, 16) % 5000) / 1000.0
    time.sleep(stagger_delay)

    holdings_arrays = []
    for d, df in sorted(all_holdings.items()):
        # Convert each pandas DataFrame to a 2D DataArray (stock x account)
        da = xr.DataArray(df.values, dims=["symbol", "account"], coords=dict(symbol=df.index, account=df.columns))
        holdings_arrays.append(da)

    # Concatenate along a new 'date' dimension
    combined_holdings = xr.concat(holdings_arrays, dim=pd.Index(all_holdings.keys(), name="date"))
    
    return df_holdings_history, final_holdings, combined_holdings

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

