from objective_functions import *
import functools

total_value_series = pd.read_parquet('temp/total_value_series.parquet')['value']


s3_path = 's3://jdinvestment/2d_run_4'
df_folds = pd.read_parquet("{}/folds.parquet".format(s3_path))
  
train_folds = [0]
val_folds = [1,2]

df_train_folds = df_folds.loc[df_folds.fold_index.isin(train_folds)]
df_val_folds = df_folds.loc[df_folds.fold_index.isin(val_folds)]

weighting_func_quantile = functools.partial(weighted_quantile, .1)
start_date = min(df_folds.start_date)
end_date = max(df_folds.end_date)
agg_func = functools.partial(mean_annualized_return, start_date, end_date)


objective_functions_dict = {
        'train': {
            'regret_quantile': WeightedRegretApplyer(df_train_folds, agg_func, weighting_func_quantile, 'voo_return', 'max'),
            'mean_regret' : WeightedRegretApplyer(df_train_folds, agg_func, weighted_mean, 'voo_return', 'max'),
            'quantile': WeightedRegimeApplyer(df_train_folds, agg_func, weighting_func_quantile),
            'mean': WeightedRegimeApplyer(df_train_folds, agg_func, weighted_mean)
        },
        'val': {
            'regret_quantile': WeightedRegretApplyer(df_val_folds, agg_func, weighting_func_quantile, 'voo_return', 'max'),
            'mean_regret' : WeightedRegretApplyer(df_val_folds, agg_func, weighted_mean, 'voo_return', 'max'),
            'quantile': WeightedRegimeApplyer(df_val_folds, agg_func, weighting_func_quantile),
            'mean': WeightedRegimeApplyer(df_val_folds, agg_func, weighted_mean)
        }
    }




df_evaluation = apply_objectives(objective_functions_dict, total_value_series)
zzz=1