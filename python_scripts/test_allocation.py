import joblib
import numpy as np
import pandas as pd
import pyomo.environ as pyo

with open('temp/allocation_test.joblib', 'rb') as fp:
    vars = joblib.load(fp)

_S = vars['S']
_B = vars['B']
_P = vars['P']
_H = vars['H']
_H = np.maximum(0, _H)
M = 100000000
trade_cost = 7

target_buy = (_B * _P.values).sum()
target_sell = (_S * _P.values).sum()



sell_stocks = _S.index[_S.values>0]
single_account_stocks = _H.index[(_H>0).sum(1) == 1]
single_account_sell = list(set(single_account_stocks).intersection(sell_stocks))

cash = pd.Series({account: 0 for account in _H.columns})
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
    for account in H.columns:
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
bought = b_df
total_sold = (sold * _P.values.reshape(len(_P),1)).values.sum()
total_bought = (b_df * P.values.reshape(len(P),1)).values.sum()
target_buy = (_B * _P.values).sum()

print((target_buy - total_bought)/target_buy)
print((target_sell - total_sold)/target_sell)
print((sold >0).values.sum() + (b_df>0).values.sum())


