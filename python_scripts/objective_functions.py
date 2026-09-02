import pandas as pd
import numpy as np


def worst_annual_return(start_date, end_date, total_value_series):
    df = total_value_series.loc[(total_value_series.index >= start_date) & (total_value_series.index <= end_date)].reset_index()
    df['year'] = df.date.dt.year,
    df2 = df[['year', 'value']].groupby('year').agg(lambda x: (x.values[-1]/x.values[0])**(len(x)/13)) - 1
    return df2['value'].min()


def mean_annual_drawdown_integral(start_date, end_date, total_value_series):
    df = total_value_series.loc[(total_value_series.index >= start_date) & (total_value_series.index <= end_date)].reset_index()
    df['year'] = df.date.dt.year
    df2 = df[['year', 'value']].groupby('year').agg(lambda x: np.maximum(0, x.values[0] - x.values).sum()/(len(x) * x.values[0]))
    return df2['value'].mean()

def worst_annual_drawdown_integral(start_date, end_date, total_value_series):
    if start_date is None:
       start_date = total_value_series.index[0]
    if end_date is None:
       end_date = total_value_series.index[-1]
    df = total_value_series.loc[(total_value_series.index >= start_date) & (total_value_series.index <= end_date)].reset_index()
    df['year'] = df.date.dt.year
    df2 = df[['year', 'value']].groupby('year').agg(lambda x: np.maximum(0, x.values[0] - x.values).sum()/(len(x) * x.values[0]))
    return df2['value'].max()


def mean_drawdown(start_date, end_date, total_value_series):
    tvs = total_value_series.loc[(total_value_series.index >= start_date) & (total_value_series.index <= end_date)]
    result = (np.maximum(0, tvs.values[0] - tvs.values)/tvs.values[0]).mean()
    return result

def mean_annualized_return(start_date, end_date, total_value_series):
    if start_date is None:
        start_date = total_value_series.index[0]
    if end_date is None:
        end_date = total_value_series.index[-1]
    tvs = total_value_series.loc[(total_value_series.index >= start_date) & (total_value_series.index <= end_date)]
    if len(tvs) < 2:
        return np.nan
    num_years = (tvs.index.max() - tvs.index.min()).days/365
    result = (tvs.iloc[-1]/tvs.iloc[0])**(1/num_years) - 1
    return result

def terminal_value(start_date, end_date,total_value_series):
    tvs = total_value_series.loc[(total_value_series.index >= start_date) & (total_value_series.index <= end_date)]
    result = tvs.iloc[-1]
    return result


def pct_change_quantile(start_date, end_date, quantile, total_value_series):
    tvs = total_value_series.loc[(total_value_series.index >= start_date) & (total_value_series.index <= end_date)]
    result =tvs.pct_change().dropna().quantile(quantile)
    return result

def get_direction_sign(label):
    """Returns -1 for 'max' and +1 for 'min'."""
    return -1 if label.lower() == 'max' else 1 if label.lower() == 'min' else None


def apply_objectives(objective_dict, a_series):
    result = {}
    for key1 in objective_dict:
        for key2, func  in objective_dict[key1].items():
            result[(key1, key2)] = func(a_series)
    df_out = pd.DataFrame(pd.Series(result)).reset_index()
    df_out.columns = ['mode', 'objective', 'value']
    return df_out






import pandas as pd


class WeightedRegretApplyer:

  def __init__(self, df_folds, agg_func, weighting_func, regret_col, sense):
    """Creates a function that maps a series to a dictionary of aggregated

    values, one for each regime. This is ideal for establishing the
    'best-case' benchmark (S*_R) for regret calculations.

    Parameters:
    - regime_df: DataFrame with 'start', 'end', and 'regime_id' columns.
    - agg_func: Function to apply (e.g., np.max, np.mean).
    """

    self.df_folds = df_folds.reset_index(drop=True)
    self.agg_func = agg_func
    self.weighting_func = weighting_func
    self.regret_col = regret_col
    self.sense = sense

    # FIX: Standardize interval bounds to datetime64[us] to match the call indexer
    start_dates = pd.to_datetime(self.df_folds["start_date"]).astype(
        "datetime64[us]"
    )
    end_dates = pd.to_datetime(self.df_folds["end_date"]).astype(
        "datetime64[us]"
    )

    self.intervals = pd.IntervalIndex.from_arrays(
        start_dates, end_dates, closed="both", name="regime_id"
    )

  def __call__(self, a_series):
    # Standardize incoming series index to match interval precision
    a_series.index = pd.to_datetime(a_series.index).floor("D").astype(
        "datetime64[us]"
    )
    regime_index = self.intervals.get_indexer(a_series.index)

    # Filter out dates that don't fall into any regime (indexer == -1)
    valid_mask = regime_index != -1

    df = pd.DataFrame(a_series.loc[valid_mask].rename("value"))
    df["regime_index"] = regime_index[valid_mask]
    df_agg = df.groupby("regime_index").agg(self.agg_func)
    df_agg["weight"] = (
        self.df_folds.end_date - self.df_folds.start_date
    ).dt.days.iloc[df_agg.index]
    #jeff      df_agg = df_agg.loc[df_agg.isnull().sum(1) == 0]
    df_agg["weight"] /= df_agg["weight"].sum()

    df_agg["best"] = self.df_folds[self.regret_col].loc[df_agg.index]
    if self.sense == "max":
      df_agg["regret"] = df_agg["best"] - df_agg["value"]
    else:
      df_agg["regret"] = df_agg["value"] - df_agg["best"]

    result = self.weighting_func(df_agg.regret, df_agg.weight)
    return result


class WeightedRegimeApplyer:

  def __init__(self, df_folds, agg_func, weighting_func):
    """Creates a function that maps a series to a dictionary of aggregated

    values, one for each regime. This is ideal for establishing the
    'best-case' benchmark (S*_R) for regret calculations.

    Parameters:
    - regime_df: DataFrame with 'start', 'end', and 'regime_id' columns.
    - agg_func: Function to apply (e.g., np.max, np.mean).
    """

    self.df_folds = df_folds.reset_index(drop=True)
    self.agg_func = agg_func
    self.weighting_func = weighting_func

    # FIX: Standardize interval bounds to datetime64[us] to match the call indexer
    start_dates = pd.to_datetime(self.df_folds["start_date"]).astype(
        "datetime64[us]"
    )
    end_dates = pd.to_datetime(self.df_folds["end_date"]).astype(
        "datetime64[us]"
    )

    self.intervals = pd.IntervalIndex.from_arrays(
        start_dates, end_dates, closed="both", name="regime_id"
    )

  def __call__(self, a_series):
    # Standardize incoming series index to match interval precision
    a_series.index = pd.to_datetime(a_series.index).floor("D").astype(
        "datetime64[us]"
    )
    regime_index = self.intervals.get_indexer(a_series.index)

    # Filter out dates that don't fall into any regime (indexer == -1)
    valid_mask = regime_index != -1

    df = pd.DataFrame(a_series.loc[valid_mask].rename("value"))
    df["regime_index"] = regime_index[valid_mask]
    df_agg = df.groupby("regime_index").agg(self.agg_func)
    df_agg["weight"] = (
        self.df_folds.end_date - self.df_folds.start_date
    ).dt.days.iloc[df_agg.index]
    #jeff df_agg = df_agg.loc[df_agg.isnull().sum(1) == 0]

    df_agg["weight"] /= df_agg["weight"].sum()

    result = self.weighting_func(df_agg.value, df_agg.weight)
    return result
    


class RegimeApplyer:
    
    def __init__(self, df_folds, agg_func):

        """
        Creates a function that maps a series to a dictionary of aggregated 
        values, one for each regime. This is ideal for establishing the 
        'best-case' benchmark (S*_R) for regret calculations.
        
        Parameters:
        - regime_df: DataFrame with 'start', 'end', and 'regime_id' columns.
        - agg_func: Function to apply (e.g., np.max, np.mean).
        """

        self.df_folds = df_folds.reset_index(drop = True)
        self.agg_func = agg_func
       
     

        self.intervals = pd.IntervalIndex.from_arrays(
        df_folds['start_date'], 
        df_folds['end_date'], 
        closed='both', 
        name='regime_id'
        )
    
    def __call__(self, a_series):
        a_series.index = a_series.index.floor('D').astype('datetime64[us]')
        regime_index = self.intervals.get_indexer(a_series.index)
        
        # Filter out dates that don't fall into any regime (indexer == -1)
        valid_mask = regime_index != -1

        df = pd.DataFrame(a_series.loc[valid_mask].rename('value'))
        df['regime_index'] = regime_index[valid_mask]
        df_agg = df.groupby('regime_index').agg(self.agg_func)
        df_agg = df_agg.loc[df_agg.isnull().sum(axis = 1) == 0]
        df_agg.index = self.df_folds.iloc[df_agg.index]['description']
        
        
        
        return pd.Series(df_agg.iloc[:,0])
    


def weighted_quantile(quantile, values, weights):
    """
    Interpolates the 'value' at a given quantile by 
    dynamically calculating cumulative weights.
    """
    df = pd.DataFrame({'value': values, 'weight': weights}).sort_values(by = 'value')
    df['weight'] /= df['weight'].sum()
    df['cum_weight'] = df['weight'].cumsum()
    value_at_quantile = np.interp(quantile, df['cum_weight'], df['value'])
    
    return value_at_quantile

def weighted_mean(values, weights):
    if values.notnull().sum() == 0:
       return np.nan
    weighted_mean = (values * weights).sum() / weights.sum()
    return weighted_mean


def extract_fold_regimes(df_fold, a_series):

    intervals = pd.IntervalIndex.from_arrays(
            df_fold['start_date'], 
            df_fold['end_date'], 
            closed='both', 
            name='regime_id'
    )
    a_series.index = a_series.index.floor('D').astype('datetime64[us]')
    regime_index = intervals.get_indexer(a_series.index)
    valid_mask = regime_index != -1
    df = pd.DataFrame(a_series.loc[valid_mask].rename('value'))
    df['regime_index'] = regime_index[valid_mask]

    df_out = pd.DataFrame()
    for regime_index in df.regime_index.drop_duplicates():
        df1 = df.loc[df.regime_index == regime_index]
        previous_date = df1.index.min() - pd.Timedelta(days = 28)
        previous_value = a_series.loc[previous_date]
        df0 = pd.DataFrame({'regime_index': [regime_index], 'value': [previous_value], 'date': [previous_date]}).set_index('date')
        df_add = pd.concat([df0, df1])
        df_out = pd.concat([df_out, df_add])

    return df_out


import pandas as pd

def get_elements_by_folds(df_fold, fold_indices, series):
    """
    Returns elements of a time-indexed series where the date falls between 
    the start_date and end_date of specified fold indices.
    
    Parameters:
    - df_fold (pd.DataFrame): Contains 'fold_index', 'start_date', 'end_date'
    - fold_indices (list): List of fold indices to filter by
    - series (pd.Series): Series with date index
    
    Returns:
    - pd.Series: Filtered subset of the input series
    """
    # Make copies and ensure datetime types for robust comparison
    df = df_fold.copy()
    df['start_date'] = pd.to_datetime(df['start_date'])
    df['end_date'] = pd.to_datetime(df['end_date'])
    
    s = series.copy()
    s.index = pd.to_datetime(s.index)
    
    # Filter the fold rows for the specified indices
    selected_folds = df[df['fold_index'].isin(fold_indices)]
    
    # Build a boolean mask for dates falling within any of the selected fold intervals
    mask = pd.Series(False, index=s.index)
    for _, row in selected_folds.iterrows():
        mask = mask | ((s.index >= row['start_date']) & (s.index <= row['end_date']))
        
    return s[mask]



class RegretPercentile:

    def __init__(self, s_reference, df_fold, folds, quantile = .1):
      self.reference = s_reference
      self.df_fold = df_fold
      self.folds = folds
      self.quantile = quantile

    def __call__(self, s_values):
       s_val = get_elements_by_folds(self.df_fold, self.folds, s_values)
       s_ref = get_elements_by_folds(self.df_fold, self.folds, self.reference)
       return (s_ref - s_val).quantile(self.quantile)

class FoldPercentile:
    def __init__(self, df_fold, folds, quantile = .1):
         self.df_fold = df_fold
         self.folds = folds
         self.quantile = quantile
   
    def __call__(self, s_values):
        s_val = get_elements_by_folds(self.df_fold, self.folds, s_values)
        return s_val.quantile(self.quantile)

class FoldApplyer:
    def __init__(self, df_fold, folds, myfunc):
         self.df_fold = df_fold
         self.folds = folds
         self.myfunc = myfunc
   
    def __call__(self, s_values):
        s_val = get_elements_by_folds(self.df_fold, self.folds, s_values)
        return self.myfunc(None, None, s_val)
       


    