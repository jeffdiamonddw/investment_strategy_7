import joblib
from simulate_stock_rotation import optimize

with open('temp/out_1.joblib', 'rb') as fp:
    (params, df_features, current_price, holdings, budget, feature_weights, max_voo) = joblib.load(fp)
    holdings,   _, _  = optimize(params, df_features, current_price, holdings, budget, feature_weights, max_voo = max_voo)