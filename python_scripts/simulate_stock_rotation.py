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

from live_update_listener import get_last_close


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
    __S: pd.Series, __B: pd.Series, __P: pd.Series, __H: pd.DataFrame, M: float = 1000000.0, trade_cost = 7
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

    

    _S = __S.copy()
    _B = __B.copy()
    _P = __P.copy()
    _H = __H.copy()

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

    if sum(keep) == 0:
        bought = 0 * sold
        return sold, bought

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








def optimize(_params, df_features, _current_price, __holdings, _budget, feature_weights, max_frac = .05, max_voo = None):
    
    df_scores =pd.DataFrame(df_features.values/_current_price.values.reshape(_current_price.shape[0],1), index = df_features.index)
    s_scores = pd.Series(np.matmul(df_scores.values, np.array(list(feature_weights.values())).reshape(len(feature_weights), 1)).flatten(), index = df_scores.index).sort_values(ascending = False)

    _holdings = __holdings.loc[__holdings.index != 'CASH'].copy()
    _cash = __holdings.loc['CASH']
    current_price = pd.DataFrame(_current_price.loc[_current_price.index != 'CASH'])

    holdings = pd.DataFrame(_holdings.sum(axis = 1))
    budget = [sum(_budget)]
    

    logging.getLogger('pyomo.util.infeasible').setLevel(logging.INFO)
    budget = np.maximum(0, budget)
    
    params = _params.copy()
    params['feature_values'] = df_features.fillna(0)
    params['current_price'] = current_price
    params['holdings'] = holdings
    params['budget'] = budget
    is_keep_stock = (current_price.values > params['min_price']).flatten() | (current_price.index == 'BIL').flatten()
    keep_stocks = current_price.index[is_keep_stock]
    drop_stock_list = sorted(list(set(current_price.index).difference(keep_stocks)))
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
           return sum(model.x[stock, a] * model.current_price[stock] for a in model.account) <= max_frac * sum(params['budget'])
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
        params['objective_sensitivity'] = .1
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

    _investment = np.matmul(pd.DataFrame(params['current_price']).loc[params['current_price'].index != 'CASH'].transpose().values, df_sol.loc[df_sol.index != 'CASH'].values)
    num_trades = (df_sol != params['holdings']).values.sum()
    df_sol = df_sol.astype('float')
    df_sol.loc['CASH', :] = (params['budget'] - _investment - float(params['trade_fee'] * num_trades)).flatten()
    df_sol = pd.concat([df_sol, pd.DataFrame(0, index = drop_stock_list, columns = df_sol.columns)]).loc[holdings.index]
    df_sol = df_sol.astype(int)
    
    initial_investment = _holdings.values.sum()
    if initial_investment == 0:
        df_allocation = allocate_stocks_heuristic(df_sol[0], current_price.iloc[:,0], _budget)
        num_trades = (df_allocation != _holdings).sum(0)
        buy_cost = (df_allocation * current_price.values).sum(0)
        df_allocation.loc['CASH'] = np.floor(
            (_cash - buy_cost).sum() - params['trade_fee'] * num_trades
        ).astype(int)
    else:
        sum_holdings = holdings.sum(1).values.reshape(df_sol.shape[0],1)
        B = np.maximum(0, df_sol - sum_holdings)  #what to buy
        S = - np.minimum(0, df_sol - sum_holdings) #what to sell
        sold, bought = solve_stock_allocation(S[0], B[0], current_price.iloc[:,0], _holdings)
        sold_revenue = (sold * current_price.values.reshape(len(current_price),1)).values.sum(0)
        bought_cost = (bought * current_price.values.reshape(len(current_price),1)).values.sum(0)
        transaction_costs = _params['trade_fee'] * ((sold >0).sum(0) + (bought>0).sum(0))
        cash_made = np.floor(sold_revenue - bought_cost - transaction_costs).astype(int)
        df_allocation = _holdings + bought - sold
        df_allocation.loc['CASH'] = np.floor(_cash + cash_made).astype(int)
        
        
        
        obj_value = None
        
    
    return df_allocation, num_trades * params['trade_fee'], obj_value




def get_proposal(holdings_start, df_price, params, df_features, feature_weights, max_voo, max_frac):

    if holdings_start is None:
        holdings_start = pd.DataFrame(0.0, index = df_price.index, columns = range(len(params['principal'])))
        holdings_start.loc['CASH', :] = np.floor(params['principal']).astype(int)


    tickers = [s for s in holdings_start.index if s != 'CASH']
    price_start = get_last_close(tickers)
    price_start.loc['CASH'] = 1
    budget_start = (holdings_start * price_start.values.reshape(len(price_start), 1)).sum(0)
    proposed_holdings, _ , _= optimize(params, df_features, price_start, holdings_start, budget_start, feature_weights, max_voo = max_voo, max_frac = max_frac)  
    return proposed_holdings      
        
        
        
def simulate(df_price, _params, data_features, df_weights, period, sim_id = None, session = None, holdings = None, max_frac = None, max_voo = None, df_dividend = None, now = False, hold = False):
    

    df_pending_dividends = pd.DataFrame()

    params = _params.copy()
    params.update(period)
    val_start_dates = df_weights.index[:-1]
    val_end_dates = df_weights.index[1:]
    time_tups = list(zip(val_start_dates, val_end_dates)) + [(val_end_dates[-1], None)]

    if now:
        time_tups = time_tups[-1:]
    
    if holdings is None:
        holdings = pd.DataFrame(0.0, index = df_price.index, columns = range(len(params['principal'])))
        holdings.loc['CASH', :] = np.floor(params['principal']).astype(int)
    holdings_start = holdings.copy()

   
    
    history = []
    df_holdings_history = pd.DataFrame()
    df_holdings_shares = pd.DataFrame()
    start_holdings = {}
    for val_start_date, val_end_date in time_tups:
        
        if val_start_date > max(data_features.coords['date'].to_pandas()):
            break

        df_features = data_features.sel(date = val_start_date).to_pandas().transpose()
        feature_weights = dict(df_weights.loc[val_start_date])
        
        last_price = df_price.loc[:, val_start_date].copy(); last_price.loc['CASH'] = 1
        last_budget = (holdings.values * pd.DataFrame(last_price.loc[holdings.index]).fillna(0).values).sum(axis = 0)
    
        if now:
            tickers = [s for s in holdings_start.index if s != 'CASH']
            price_start = get_last_close(tickers)
            price_start.loc['CASH'] = 1
            budget_start = (holdings_start * price_start.values.reshape(len(price_start), 1)).sum(0)
            proposed_holdings, num_trades, obj_value = optimize(params, df_features, price_start, holdings_start, budget_start, feature_weights, max_voo = max_voo, max_frac = max_frac)


            df_price_live = pd.read_parquet('live_quotes_cache.parquet').set_index('symbol')
    
            
            live_sell_price = df_price_live['bid_price']; live_sell_price.loc['CASH'] = 1
            live_buy_price = df_price_live['ask_price']; live_buy_price.loc['CASH'] = 1
            live_budget = (holdings.values * pd.DataFrame(live_sell_price.loc[holdings.index]).fillna(0).values).sum(axis = 0)
            
            current_price = live_sell_price.loc[holdings.index]
            budget = live_budget
        else:
            current_price =last_price
            budget = last_budget
            live_sell_price = live_buy_price = None
            
        
        holdings_shares = pd.DataFrame((holdings.sum(axis = 1))).transpose()
        holdings_shares.loc[:, 'date'] = val_start_date
        df_holdings_shares = pd.concat([df_holdings_shares, holdings_shares])
        
    
        
        
        

      
            
            
        old_value = (holdings * current_price.values.reshape(len(current_price),1)).values.sum()
        
        if not hold:
            holdings,   _, _  = optimize(params, df_features, current_price, holdings, budget, feature_weights, max_voo = max_voo, max_frac = max_frac)

        if val_end_date is not None:

            if df_pending_dividends.shape[0] > 0:
                is_payed = (df_pending_dividends.payment_date >= val_start_date) & (df_pending_dividends.payment_date <= val_end_date)
                df_payed = df_pending_dividends.loc[is_payed]
                df_pending_dividends = df_pending_dividends.loc[~is_payed]
                pending_payed = {}
                for account in holdings:
                    pending_payed[account] = df_payed.loc[df_payed.account == account, 'total'].sum()
                    holdings.loc['CASH', account] += int(pending_payed[account])
                    
            stocks_held = holdings.index[holdings.sum(1) > 0]
            new_paid, df_pending = get_payed_dividends(df_dividend, holdings.loc[stocks_held], val_start_date, val_end_date)
            df_pending_dividends = pd.concat([df_pending_dividends, df_pending])
            for account in holdings:
                 holdings.loc['CASH', account] += np.floor(new_paid[account]).astype(int) 


        if val_end_date is not None:
            
            new_price = df_price.loc[:, val_end_date].copy(); new_price.loc['CASH'] = 1

            new_value = (holdings * new_price.values.reshape(len(current_price),1)).values.sum() 
            cash_value = holdings.loc['CASH'].sum()

            print(val_start_date, val_end_date, old_value, new_value, cash_value/new_value)
        
        if val_end_date is None:
            

            if now:
                df_buy = np.maximum(proposed_holdings - holdings_start,0)
                df_sell = - np.minimum(proposed_holdings - holdings_start, 0)
                num_trades = (df_buy.loc[df_buy.index != 'CASH'] > 0).values.sum(0) + (df_sell.loc[df_sell.index != 'CASH'] > 0).values.sum(0)
                sell_revenue = np.nansum(df_sell.values * live_sell_price.loc[df_sell.index].values.reshape(len(live_sell_price),1), axis = 0)
                buy_cost = np.nansum(proposed_holdings.values * pd.DataFrame(live_buy_price.loc[holdings.index]).fillna(0).values, axis =0)
                proposed_holdings.loc['CASH'] = np.floor(sell_revenue - buy_cost - 6.95 * num_trades).astype(int)
            else:
                proposed_holdings = holdings
            

        holdings_add = pd.DataFrame((holdings.sum(axis = 1).values * current_price)).transpose().reset_index().rename(columns = {'index': 'date'})
        holdings_add.loc[:, 'sim_id'] = sim_id

        df_holdings_history = pd.concat([df_holdings_history, holdings_add])
        start_holdings[val_start_date] = holdings
        

        

        

        if val_end_date is not None:
            history.append((val_start_date, val_end_date, old_value, new_value))
            
        
    stagger_delay = (int(sim_id, 16) % 5000) / 1000.0
    time.sleep(stagger_delay)

    share_holdings_arrays = []
    for d, df in sorted(start_holdings.items()):
        # Convert each pandas DataFrame to a 2D DataArray (stock x account)
        da = xr.DataArray(df.values, dims=["symbol", "account"], coords=dict(symbol=df.index, account=df.columns))
        share_holdings_arrays.append(da)

    # Concatenate along a new 'date' dimension
    data_share_holdings = xr.concat(share_holdings_arrays, dim=pd.Index(start_holdings.keys(), name="date"))
    
    return df_holdings_history, proposed_holdings, data_share_holdings, live_sell_price, live_buy_price

def get_bil_eps(data_bil):
    df_bil = data_bil.sel(symbol = 'BIL').to_pandas().transpose().iloc[:, 1:]
    df_bil['avg_eps_1q'] = (1 + df_bil.dollar_ret_1p)**((365/4)/(28))-1
    df_bil['avg_eps_2q'] = (1 + df_bil.dollar_ret_6p)**((365/2)/(6*28))-1
    df_bil['avg_eps_4q'] = (1 + df_bil.dollar_ret_13p)**(365/(13*28))-1
    df_bil['avg_eps_8q'] = (1 + df_bil.dollar_ret_26p)**((2*365)/(26*28))-1
    return df_bil[[c for c in df_bil.columns if 'eps' in c]].stack().to_xarray().expand_dims(symbol=['BIL']).transpose('band','symbol','date')

def interpolate_to_4week_grid(da, anchor_date):
    days = (pd.to_datetime(da.date) - pd.to_datetime(anchor_date)).days
    da_numeric = da.assign_coords(date=days)
    new_coords = np.arange(int(days.min()//28)*28, int(days.max()//28+1)*28, 28)
    da_interp = da_numeric.interp(date=new_coords, method="linear")
    return da_interp.assign_coords(date=pd.to_datetime(anchor_date) + pd.to_timedelta(da_interp.date.values, unit='D'))


def get_payed_dividends(df_dividend, holdings, start_date, end_date):

    payed = {}
    df_pending = pd.DataFrame()
    for account in holdings.columns:
        stocks_held = holdings.index[holdings[account] > 0]
        df = df_dividend.copy().loc[df_dividend.ticker.isin(stocks_held) & (df_dividend.ex_dividend_date >= start_date) & (df_dividend.ex_dividend_date <= end_date)]
        dividends_payed = df.loc[(df.payment_date >= start_date) & (df.payment_date <= end_date)].copy()
        dividends_payed['holdings'] = holdings.loc[dividends_payed.ticker, account].copy().values
        dividends_payed['total'] = dividends_payed.amount * dividends_payed.holdings
        payed[account] = dividends_payed.total.sum()

        dividends_pending = df.loc[(df.payment_date > end_date)].copy()
        dividends_pending['holdings'] = holdings.loc[dividends_pending.ticker, account].copy().values
        dividends_pending['total'] = dividends_pending.amount * dividends_pending.holdings
        dividends_pending['account'] = account
        df_pending = pd.concat([df_pending, dividends_pending])
    return payed, df_pending


# --- Main Cluster Exploration Script ---

