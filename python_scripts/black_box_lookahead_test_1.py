import os
import io
import json
import warnings
import requests
import numpy as np
import pandas as pd
import xarray as xr

# Silence internal warnings
warnings.filterwarnings("ignore", message=".*Timestamp.utcnow is deprecated.*")

# --- CONFIGURATION ---
API_KEY = '693327461e9541.04731237' 
TICKER_FILE = 'strategy/multi_dim_stock_list.csv'
MOMENTUM_PATH = 'simulation_data/momentum.nc'
QUALITY_PATH = 'simulation_data/quality.nc'
GIC_PATH = 'simulation_data/gic_data.nc'

RAW_CACHE_DIR = 'raw_cache'
RAW_MOMENTUM_DIR = os.path.join(RAW_CACHE_DIR, 'momentum')
RAW_QUALITY_DIR = os.path.join(RAW_CACHE_DIR, 'quality')
RAW_HOLDINGS_DIR = os.path.join(RAW_CACHE_DIR, 'holdings')
RAW_BOC_PATH = os.path.join(RAW_CACHE_DIR, 'boc_data.csv')

PIN_DATE = '2005-09-05'
DAYS_INTERVAL = 28

MOM_BANDS = ['price_end', 'dollar_ret_1p', 'dollar_ret_6p', 'dollar_ret_13p', 'dollar_ret_26p']
QUAL_BANDS = ['avg_eps_1q', 'avg_eps_2q', 'avg_eps_4q', 'avg_eps_8q']

LEVERAGE_MAP = {
    'SH': -1.0, 'SDS': -2.0, 'SPXU': -3.0, 'PSQ': -1.0, 'QID': -2.0, 
    'SQQQ': -3.0, 'BTAL': -1.0, 'SVXY': 0.5, 'CTA': -1.0, 'DBMF': -1.0, 
    'KMLM': -1.0, 'PFIX': -1.0, 'CYA': -1.0, 'RSBT': -1.0, 'FMF': -1.0
}


# ==========================================
# PART 1: RAW DATA PULL & CACHING TO DISK
# ==========================================

def pull_raw_boc_data(start_date='2005-01-01'):
    """Pulls Bank Rate and 1-Year GIC raw CSV from BoC and saves to disk."""
    os.makedirs(RAW_CACHE_DIR, exist_ok=True)
    series_ids = "V80691310,V80691339"
    url = f"https://www.bankofcanada.ca/valet/observations/{series_ids}/csv?start_date={start_date}"
    
    try:
        print(f"Pulling raw BoC data...")
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        with open(RAW_BOC_PATH, 'w', encoding='utf-8') as f:
            f.write(response.text)
        print(f"Saved raw BoC data to {RAW_BOC_PATH}")
    except Exception as e:
        print(f"Failed to pull raw BoC data: {e}")


def pull_raw_momentum_data(tickers):
    """Pulls raw EOD price CSVs for tickers and saves them to disk."""
    os.makedirs(RAW_MOMENTUM_DIR, exist_ok=True)
    print(f"Pulling raw Momentum data for {len(tickers)} tickers from EODHD...")
    
    for j, ticker in enumerate(tickers):
        try:
            url = f'https://eodhd.com/api/eod/{ticker}.US?api_token={API_KEY}&fmt=csv&from={PIN_DATE}'
            response = requests.get(url, timeout=10)
            if response.status_code != 200:
                print(f"    ! Failed to fetch {ticker} (Status: {response.status_code})")
                continue
            
            file_path = os.path.join(RAW_MOMENTUM_DIR, f"{ticker}.csv")
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(response.text)
                
            print(f"Saved raw momentum for {ticker} ({j+1}/{len(tickers)})")
        except Exception as e:
            print(f"    ! Error pulling raw momentum for {ticker}: {e}")


def pull_raw_quality_data(tickers):
    """Pulls raw fundamentals JSON and ETF holdings CSVs for quality logic and saves to disk."""
    os.makedirs(RAW_QUALITY_DIR, exist_ok=True)
    os.makedirs(RAW_HOLDINGS_DIR, exist_ok=True)
    print(f"Pulling raw Quality data for {len(tickers)} tickers...")
    
    ticker_meta = pd.read_csv(TICKER_FILE).set_index('symbol')
    
    for j, ticker in enumerate(tickers):
        try:
            asset_type = ticker_meta.loc[ticker, 'asset_type'].upper()
            direction = ticker_meta.loc[ticker, 'direction'].upper()
            
            # Pull Stock or Long ETF holding fundamentals
            if asset_type == 'STOCK':
                url = f'https://eodhd.com/api/fundamentals/{ticker}.US?api_token={API_KEY}'
                resp = requests.get(url, timeout=10)
                if resp.status_code == 200:
                    with open(os.path.join(RAW_QUALITY_DIR, f"{ticker}.json"), 'w', encoding='utf-8') as f:
                        f.write(resp.text)
            
            elif asset_type == 'ETF' and direction == 'LONG':
                # Check if local source holdings file exists to copy/cache
                holdings_file = f'./holdings/{ticker}_holdings.csv'
                if os.path.exists(holdings_file):
                    df_holdings = pd.read_csv(holdings_file)
                    df_holdings.to_csv(os.path.join(RAW_HOLDINGS_DIR, f"{ticker}_holdings.csv"), index=False)
                    
                    # Pull fundamentals for each holding
                    for _, row in df_holdings.iterrows():
                        h_raw_code = str(row['Code']).strip()
                        h_code = f"{h_raw_code}.US" if "." not in h_raw_code else h_raw_code
                        h_url = f'https://eodhd.com/api/fundamentals/{h_code}?api_token={API_KEY}'
                        h_resp = requests.get(h_url, timeout=10)
                        if h_resp.status_code == 200:
                            with open(os.path.join(RAW_QUALITY_DIR, f"{h_code}.json"), 'w', encoding='utf-8') as f:
                                f.write(h_resp.text)
                                
            print(f"Saved raw quality data for {ticker} ({j+1}/{len(tickers)})")
        except Exception as e:
            print(f"    ! Error pulling raw quality for {ticker}: {e}")


def pull_raw_data():
    """Part 1 Orchestrator: Pulls all raw data from sources and caches to disk."""
    os.makedirs(RAW_CACHE_DIR, exist_ok=True)
    csv_tickers = sorted(pd.read_csv(TICKER_FILE)['symbol'].unique().tolist())
    
    pull_raw_boc_data()
    pull_raw_momentum_data(csv_tickers)
    pull_raw_quality_data(csv_tickers)
    print("Part 1: Raw data pulling and disk caching complete.\n")


# ==========================================
# PART 2: PROCESSING & OUTPUT GENERATION
# ==========================================

def create_gic_dataarray(df: pd.DataFrame, max_date ) -> xr.DataArray:
    df = df.copy()
    df['growth_factor'] = (1 + .01 * df['gic']) ** (28 / 365)
    windows = [1, 6, 13, 26]
    bands = ['price_end'] + [f'dollar_ret_{w}p' for w in windows]
    symbols = ['GIC']
    dates = np.array(df.index)
    
    da = xr.DataArray(
        np.nan,
        coords={'band': bands, 'symbol': symbols, 'date': dates},
        dims=('band', 'symbol', 'date')
    )
    da.loc[dict(band='price_end')] = 1.0
    df.loc[df.index > max_date, 'growth_factor'] = np.nan
    for i, w in enumerate(windows, start=1):
        trailing_factor = (
            df['growth_factor']
            .rolling(window=w)
            .apply(lambda x: x.prod(), raw=True)
        )
        pct_change = trailing_factor - 1
        num_years = (w * 28) / 365.0
        annualized_return = ((trailing_factor ** (1 / num_years)) - 1) 
        da.loc[dict(band=f'dollar_ret_{w}p')] = annualized_return
        
    return da


def build_momentum_from_cache(tickers, target_dates, max_date = pd.to_datetime('2026-06-15')):
    """Processes cached momentum CSV files from disk."""
    print(f"Processing Momentum data from cache for {len(tickers)} tickers...")
    full_data = np.zeros((len(MOM_BANDS), len(tickers), len(target_dates)))
    
    for j, ticker in enumerate(tickers):
        file_path = os.path.join(RAW_MOMENTUM_DIR, f"{ticker}.csv")
        if not os.path.exists(file_path):
            print(f"    ! Cached file missing for {ticker}")
            continue
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                df = pd.read_csv(io.StringIO(f.read()), index_col='Date', parse_dates=True)
            
            if 'Adjusted_close' not in df.columns:
                print(f"    ! Adjusted data missing for {ticker}")
                continue
                
            series = df['Adjusted_close'].reindex(target_dates, method='ffill').ffill().bfill()
            series.loc[series.index > max_date] = np.nan
            ratios = series / series.shift(1)
            series = series.mask((ratios > 5.0) | (ratios < 0.2))
            
            full_data[0, j, :] = series.values.astype(float)
            
            for i, window in enumerate([1, 6, 13, 26], start=1):
                pct_change = series.pct_change(periods=window, fill_method=None).values
                #pct_change = series.pct_change(periods=window).values
                num_years = (window * 28) / 365.0
                annualized_return = (((pct_change + 1) ** (1 / num_years)) - 1) 
                expected_1yr_profit = series * annualized_return
                full_data[i, j, :] = expected_1yr_profit.values

        except Exception as e:
            print(f"    ! Error processing cached momentum for {ticker}: {e}")
            continue
        result = xr.DataArray(full_data, coords={'band': MOM_BANDS, 'symbol': tickers, 'date': target_dates}, dims=['band', 'symbol', 'date'])
        zzz=1

    return result


def build_quality_from_cache(tickers, target_dates, da_mom):
    """Processes cached quality JSON and holdings files from disk."""
    results = []
    print(f"Processing Quality data from cache for {len(tickers)} tickers...")
    ticker_meta = pd.read_csv(TICKER_FILE).set_index('symbol')
    
    for j, ticker in enumerate(tickers):
        asset_type = ticker_meta.loc[ticker, 'asset_type'].upper()
        direction = ticker_meta.loc[ticker, 'direction'].upper()
        data = np.zeros((4, len(target_dates)))

        if asset_type == 'ETF' and direction == 'SHORT':
            leverage = LEVERAGE_MAP.get(ticker, -1.0)
            p_etf = da_mom.sel(symbol=ticker, band='price_end')
            p_mkt = da_mom.sel(band='price_end').mean(dim='symbol')
            raw_proxy = (p_etf / p_mkt) * leverage
            
            for i, window in enumerate([1, 2, 4, 8]):
                data[i] = 4 * pd.Series(raw_proxy.values).rolling(window=window, min_periods=1).mean().values

        elif asset_type == 'STOCK':
            json_path = os.path.join(RAW_QUALITY_DIR, f"{ticker}.json")
            if os.path.exists(json_path):
                with open(json_path, 'r', encoding='utf-8') as f:
                    resp = json.load(f)
                earnings = resp.get('Earnings', {}).get('History', {})
                if earnings:
                    eps_df = pd.DataFrame.from_dict(earnings, orient='index')
                    if 'filing_date' in eps_df.columns:
                        effective_dates = pd.to_datetime(eps_df['filing_date']).fillna(pd.to_datetime(eps_df['date']) + pd.Timedelta(days=45))
                    else:
                        effective_dates = pd.to_datetime(eps_df['date']) + pd.Timedelta(days=45)
                    
                    eps_df['effective_date'] = effective_dates
                    eps_df['epsActual'] = pd.to_numeric(eps_df['epsActual'], errors='coerce').fillna(0)
                    eps_df = eps_df.sort_values('effective_date').dropna(subset=['effective_date'])
                    
                    target_df = pd.DataFrame({'date': target_dates}).sort_values('date')
                    merged = pd.merge_asof(
                        target_df, 
                        eps_df[['effective_date', 'epsActual']], 
                        left_on='date', 
                        right_on='effective_date', 
                        direction='backward'
                    )
                    merged['epsActual'] = merged['epsActual'].fillna(0)
                    aligned_eps = pd.Series(merged['epsActual'].values, index=target_df['date']).reindex(target_dates)
                    
                    for i, window in enumerate([1, 2, 4, 8]):
                        data[i] = 4 * aligned_eps.rolling(window=window, min_periods=1).mean().values

        elif asset_type == 'ETF' and direction == 'LONG':
            holdings_file = os.path.join(RAW_HOLDINGS_DIR, f"{ticker}_holdings.csv")
            if os.path.exists(holdings_file):
                df_holdings = pd.read_csv(holdings_file)
                weighted_eps_sum = pd.Series(0.0, index=target_dates)
                total_w = 0.0
                
                for _, row in df_holdings.iterrows():
                    h_raw_code = str(row['Code']).strip()
                    h_code = f"{h_raw_code}.US" if "." not in h_raw_code else h_raw_code
                    h_weight = float(row['Assets_%']) / 100.0
                    
                    h_json_path = os.path.join(RAW_QUALITY_DIR, f"{h_code}.json")
                    if os.path.exists(h_json_path):
                        with open(h_json_path, 'r', encoding='utf-8') as f:
                            h_resp = json.load(f)
                        h_hist = h_resp.get('Earnings', {}).get('History', {})
                        if h_hist:
                            h_df = pd.DataFrame.from_dict(h_hist, orient='index')
                            if 'filing_date' in h_df.columns:
                                h_effective_dates = pd.to_datetime(h_df['filing_date']).fillna(pd.to_datetime(h_df['date']) + pd.Timedelta(days=45))
                            else:
                                h_effective_dates = pd.to_datetime(h_df['date']) + pd.Timedelta(days=45)
                                
                            h_df['effective_date'] = h_effective_dates
                            h_df['epsActual'] = pd.to_numeric(h_df['epsActual'], errors='coerce').fillna(0)
                            h_df = h_df.sort_values('effective_date').dropna(subset=['effective_date'])
                            
                            target_df = pd.DataFrame({'date': target_dates}).sort_values('date')
                            merged = pd.merge_asof(
                                target_df, 
                                h_df[['effective_date', 'epsActual']], 
                                left_on='date', 
                                right_on='effective_date', 
                                direction='backward'
                            )
                            merged['epsActual'] = merged['epsActual'].fillna(0)
                            h_series = pd.Series(merged['epsActual'].values, index=target_df['date']).reindex(target_dates)
                            
                            weighted_eps_sum += (h_series * h_weight)
                            total_w += h_weight
                
                if total_w > 0:
                    final_eps = weighted_eps_sum / total_w
                    for i, window in enumerate([1, 2, 4, 8]):
                        data[i] = 4 * final_eps.rolling(window=window, min_periods=1).mean().values

        da = xr.DataArray(data[:, np.newaxis, :], 
                          coords={'band': QUAL_BANDS, 'symbol': [ticker], 'date': target_dates}, 
                          dims=['band', 'symbol', 'date'])
        results.append(da)

    return xr.concat(results, dim='symbol')


def get_gic_data_from_cache(target_dates, max_date):
    if not os.path.exists(RAW_BOC_PATH):
        print("Cached BoC data missing.")
        return None
        
    with open(RAW_BOC_PATH, 'r', encoding='utf-8') as f:
        lines = f.read().splitlines()
        
    data_start = next(i for i, line in enumerate(lines) if line.startswith('"date"'))
    df = pd.read_csv(io.StringIO("\n".join(lines[data_start:])))
    df.columns = [c.strip('"') for c in df.columns]
    df = df.rename(columns={'date': 'REF_DATE'})
    
    df_melted = df.melt(id_vars=['REF_DATE'], var_name='VECTOR', value_name='VALUE')
    df_melted['VECTOR'] = df_melted['VECTOR'].str.lower()
    df_melted['REF_DATE'] = pd.to_datetime(df_melted['REF_DATE'])
    
    pivot_df = df_melted.pivot(index='REF_DATE', columns='VECTOR', values='VALUE')
    pivot_df = pivot_df.rename(columns={
        'v80691310': 'Bank_Rate',
        'v80691339': 'GIC_1_Year'
    })
    
    processed_df = pivot_df.reindex(pivot_df.index.union(target_dates)).ffill().reindex(target_dates)
    processed_df['gic'] = ((processed_df['Bank_Rate'] + processed_df['GIC_1_Year']) / 2) - 0.35
    return create_gic_dataarray(processed_df, max_date)


def run_strict_alignment_check(da_mom, da_qual, csv_tickers):
    print("\n" + "="*60)
    print("--- SCRATCH REBUILD AUDIT ---")
    print(f"Target Tickers (CSV): {len(csv_tickers)}")
    print(f"Momentum Symbols:     {len(da_mom.symbol)}")
    print(f"Quality Symbols:      {len(da_qual.symbol)}")
    
    mom_zero_pct = (da_mom == 0).sum().values / da_mom.size * 100
    qual_zero_pct = (da_qual == 0).sum().values / da_qual.size * 100
    
    print(f"Momentum Fill Rate: {100 - mom_zero_pct:.2f}%")
    print(f"Quality Fill Rate:  {100 - qual_zero_pct:.2f}%")
    
    if np.array_equal(da_mom.symbol.values, da_qual.symbol.values):
        print("[PASS] Ticker alignment successful.")
    else:
        print("[FAIL] Ticker mismatch detected!")

    if np.array_equal(da_mom.date.values, da_qual.date.values):
        print(f"[PASS] Date alignment successful ({len(da_mom.date)} dates).")
    else:
        print("[FAIL] Date mismatch detected!")
    print("="*60 + "\n")


def build_simulation_outputs():
    """Part 2 Orchestrator: Reads from raw cache files, processes metrics, and creates final NetCDFs."""
    os.makedirs('simulation_data', exist_ok=True)
    csv_tickers = sorted(pd.read_csv(TICKER_FILE)['symbol'].unique().tolist())
    
    now_utc = pd.Timestamp.now(tz='UTC').tz_localize(None) 
    target_dates = pd.date_range(start=PIN_DATE, end=now_utc, freq=f'{DAYS_INTERVAL}D')

    # Build Momentum from cache
    
    # for d in range(30, len(target_dates)):
    #     max_date = target_dates[d]
    #     da_mom = build_momentum_from_cache(csv_tickers, target_dates, max_date)
    #     valid_dates = [target_dates[window:d+1] for window in [0,1,6,13,26]]
    #     num_nulls = 0
    #     for i in range(5):
    #         _list = valid_dates[i]
    #         num_nulls += da_mom.sel(date = _list).isel(band = i).to_pandas().isnull().values.sum()
    #     print(d, num_nulls)
    # da_mom.to_netcdf(MOMENTUM_PATH)

    # Build GIC from cache
    # for d in range(30, len(target_dates)):
    #     max_date = target_dates[d]
    #     da_gic = get_gic_data_from_cache(target_dates, max_date)
    #     valid_dates = [target_dates[window:d+1] for window in [0,1,6,13,26]]
    #     num_nulls = 0
    #     for i in range(5):
    #         _list = valid_dates[i]
    #         num_nulls += da_gic.sel(date = _list).isel(band = i).to_pandas().isnull().values.sum()
    #     print(d, num_nulls)
    # if da_gic is not None:
    #     da_gic.to_netcdf(GIC_PATH)

    # Build Quality from cache
    da_qual = build_quality_from_cache(csv_tickers, target_dates, da_mom)
    da_qual.to_netcdf(QUALITY_PATH)

    run_strict_alignment_check(da_mom, da_qual, csv_tickers)
    print("Part 2: Processing and output generation complete.")


def main():
    # --- PART 1: Pull raw data and store on disk ---
    # Comment this out if you already pulled data and want to run look-ahead tests repeatedly
    #pull_raw_data()
    
    # --- PART 2: Read disk data, process, and output final files ---
    build_simulation_outputs()


if __name__ == "__main__":
    main()