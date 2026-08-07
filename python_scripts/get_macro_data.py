import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# --- CONFIGURATION ---
API_TOKEN = '693327461e9541.04731237'
FRED_KEY = '835b84250468b5de8c18889b86369f7c'
START_DATE = '2004-01-01'
END_DATE = datetime.now().strftime('%Y-%m-%d')
ROTATION_DAYS = 28

import xarray as xr

def fetch_vix(ticker):
    url = f"https://eodhd.com/api/eod/{ticker}.INDX?api_token={API_TOKEN}&fmt=json&from={START_DATE}"
    r = requests.get(url)
    if r.status_code == 200:
        df = pd.DataFrame(r.json())
        df['date'] = pd.to_datetime(df['date'])
        return df[['date', 'close']].rename(columns={'close': ticker})
    return pd.DataFrame()

def fetch_ust_yields():
    all_years = []
    current_year = datetime.now().year
    for year in range(2005, current_year + 1):
        url = f"https://eodhd.com/api/ust/yield-rates?api_token={API_TOKEN}&filter[year]={year}&fmt=json"
        r = requests.get(url)
        if r.status_code == 200:
            response_json = r.json()
            data = response_json.get('data', [])
            if isinstance(data, list) and len(data) > 0:
                year_df = pd.DataFrame.from_records(data)
                all_years.append(year_df)
    if not all_years: return pd.DataFrame()
    df = pd.concat(all_years, ignore_index=True)
    df['date'] = pd.to_datetime(df['date'])
    df_pivot = pd.pivot_table(df, index='date', values='rate', columns='tenor')
    if '2Y' in df_pivot.columns and '10Y' in df_pivot.columns:
        df_pivot['y10'] = pd.to_numeric(df_pivot['10Y'], errors='coerce')
        df_pivot['y2'] = pd.to_numeric(df_pivot['2Y'], errors='coerce')
        df_pivot['YIELD_SPREAD'] = df_pivot['y10'] - df_pivot['y2']
        return df_pivot.reset_index()[['date', 'YIELD_SPREAD']]
    return pd.DataFrame()

def fetch_fed_rate():
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {"series_id": "DFF", "api_key": FRED_KEY, "file_type": "json", "observation_start": START_DATE}
    r = requests.get(url, params=params)
    if r.status_code == 200:
        data = r.json().get('observations', [])
        df = pd.DataFrame(data)
        if 'date' in df.columns and 'value' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            return df.dropna(subset=['value'])[['date', 'value']].rename(columns={'value': 'FED_RATE'})
    return pd.DataFrame()

def fetch_reconstructed_hy_spread(treasury_df):
    hyg_url = f"https://eodhd.com/api/eod/HYG.US?api_token={API_TOKEN}&fmt=json&from={START_DATE}&period=d&order=a"
    r = requests.get(hyg_url)
    if r.status_code == 200:
        df_h = pd.DataFrame(r.json())
        df_h['date'] = pd.to_datetime(df_h['date'])
        df_merged = df_h.merge(treasury_df[['date', 'YIELD_SPREAD']], on='date', how='inner')
        df_merged['HY_SPREAD'] = (1 / df_merged['close']) * 100
        return df_merged[['date', 'HY_SPREAD']]
    return pd.DataFrame()


# 1. DOWNLOAD DATA STREAMS
print("Downloading raw inputs...")
yields = fetch_ust_yields()
hy_spread = fetch_reconstructed_hy_spread(yields)
fed_rate = fetch_fed_rate()
vix = fetch_vix("VIX")
vix3m = fetch_vix("VIX3M")

# --- APPLY REPORTING & AVAILABILITY LAGS ---
# Shift publication dates forward by 1 trading day so day t's data is only accessible on t+1
fed_rate['date'] = fed_rate['date'] + timedelta(days=1)
yields['date'] = yields['date'] + timedelta(days=1)

# 2. MERGE INTO MASTER DATAFRAME
print("Aligning continuous daily records with publication lags...")
df_master = vix.merge(vix3m, on='date', how='outer')
df_master = df_master.merge(yields, on='date', how='outer')
df_master = df_master.merge(fed_rate, on='date', how='outer')
df_master = df_master.merge(hy_spread, on='date', how='outer')
df_master = df_master.sort_values('date').ffill().dropna()

# 3. CALCULATE DERIVED SIGNALS (Correct Contango/Backwardation orientation)
df_master['VIX_RATIO'] = df_master['VIX3M'] / df_master['VIX']

signals = ['VIX_RATIO', 'YIELD_SPREAD', 'HY_SPREAD','FED_RATE' ]

# --- STEP 4: BASELINE BACKGROUND Z-NORMALIZATION (365-day Daily Window) ---
print("Establishing background baseline distributions (365-day Daily Lookback)...")
df_daily_z = pd.DataFrame(index=df_master.date)
for sig in signals:
    mean_365 = df_master[sig].rolling(window=365, min_periods=28).mean()
    std_365 = df_master[sig].rolling(window=365, min_periods=28).std()
    df_daily_z[sig] = (df_master[sig].values - mean_365.values) / (std_365.values + 1e-6)

# --- STEP 5: CALCULATE HIGH-FIDELITY MOMENT MATRIX ---
df_daily_features = pd.DataFrame(index = df_master.date)
for sig in signals:
    # Component 1: Level (Order 0, Moment 1) -> 28-day rolling average position
    df_daily_features.loc[:, f'{sig}_0_1'] = df_daily_z[sig].rolling(window = 28, min_periods = 1).mean().values
    
    # Component 2: Volatility (Order 0, Moment 2) -> Zero-Centered Path Volatility
    df_daily_features.loc[:, f'{sig}_0_2'] = df_daily_z[sig].rolling(window=28, min_periods = 1).std().values - 1.0
    
    # Component 3: Velocity (Order 1, Moment 1) -> 28-day relative momentum shift
    df_daily_features.loc[:, f'{sig}_1_1'] = df_daily_z[sig].diff(periods = 28).values
    
# 6. DOWN-SAMPLE TO 4-WEEK HEARTBEAT DATES
print("Down-sampling processed features to 28-day heartbeats...")
rotation_dates = []
current = pd.to_datetime(START_DATE)
end_dt = pd.to_datetime(END_DATE)

while current <= end_dt:
    idx_pos = df_daily_features.index.get_indexer([current], method='nearest')[0]
    rotation_dates.append(df_daily_features.index[idx_pos])
    current += timedelta(days=ROTATION_DAYS)

# Slice the fully-calculated daily matrix strictly at the heartbeat timestamps
df_heartbeat = df_daily_features.loc[rotation_dates].copy().drop_duplicates()

# --- STEP 7: EXTRACT SIMULATION-SAFE TACTICAL ACCELERATION ---
for sig in signals:
    # Compute rolling standard deviation on momentum shifts
    raw_accel_series = pd.Series(df_heartbeat[f'{sig}_1_1'].rolling(window=5, min_periods=1).std().values)
    
    # Apply an expanding median window to eliminate look-ahead bias / data leakage
    expanding_median = raw_accel_series.expanding(min_periods=1).median()
    
    # Store zero-centered parameter mapping safely
    df_heartbeat.loc[:, f'{sig}_1_2'] = (raw_accel_series - expanding_median).values

# Save final structured dataset
df_heartbeat = df_heartbeat.dropna().reset_index()
df_heartbeat.columns = [col.lower() for col in df_heartbeat.columns]
df_heartbeat.set_index('date', inplace = True)
df_heartbeat.to_parquet('s3://jdinvestment/simulation_data/macro_data.parquet')