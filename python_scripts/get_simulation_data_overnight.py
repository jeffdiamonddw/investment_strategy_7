import os
import io
import json
import warnings
import requests
import numpy as np
import pandas as pd
import xarray as xr

from get_macro_data import get_macro_data
from get_macro_signals import get_macro_signals

# Silence internal warnings
warnings.filterwarnings("ignore", message=".*Timestamp.utcnow is deprecated.*")

# --- CONFIGURATION ---
API_KEY = '693327461e9541.04731237' 
TICKER_FILE = 'strategy/stocks_533.csv'
OUTPUT_DIR = "simulation_data_overnight"

MOMENTUM_PATH = os.path.join(OUTPUT_DIR, 'momentum.nc')
QUALITY_PATH = os.path.join(OUTPUT_DIR, 'quality.nc')
DIVIDEND_PATH = os.path.join(OUTPUT_DIR, 'dividend.parquet')
BIL_PATH = os.path.join(OUTPUT_DIR, 'bil_data.nc')

RAW_CACHE_DIR = 'raw_cache'
RAW_MOMENTUM_DIR = os.path.join(RAW_CACHE_DIR, 'momentum')
RAW_QUALITY_DIR = os.path.join(RAW_CACHE_DIR, 'quality')
RAW_HOLDINGS_DIR = os.path.join(RAW_CACHE_DIR, 'holdings')
RAW_FRED_PATH = os.path.join(RAW_CACHE_DIR, 'dtb3_data.csv')

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

def pull_raw_fred_and_bil_data(start_date='2005-01-01'):
    """Pulls U.S. 3-Month T-Bill Rate (DTB3) from FRED and saves to disk."""
    os.makedirs(RAW_CACHE_DIR, exist_ok=True)
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DTB3"
    
    try:
        print("Pulling raw US 3-Month T-Bill rate (DTB3) from FRED...")
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        with open(RAW_FRED_PATH, 'w', encoding='utf-8') as f:
            f.write(response.text)
        print(f"Saved raw FRED data to {RAW_FRED_PATH}")
    except Exception as e:
        print(f"Failed to pull raw FRED data: {e}")


def pull_raw_momentum_data(tickers):
    """
    Checks cached momentum CSV files. Determines if there are any new market days 
    between the cache's latest date and today (accounting for weekends/holidays). 
    If no new market days exist, only downloads for tickers completely missing from the cache. 
    Otherwise, appends new market data for existing cached tickers and fully downloads new ones.
    """
    os.makedirs(RAW_MOMENTUM_DIR, exist_ok=True)
    print(f"Pulling/updating raw Momentum data for {len(tickers)} tickers from EODHD...")
    
    now_utc = pd.Timestamp.now(tz='UTC').tz_localize(None)
    today_str = now_utc.strftime('%Y-%m-%d')

    # Step 1: Check if any cached file has a latest date older than the most recent potential market day.
    # We look at the maximum date across all existing cache files to determine if new market days could exist.
    max_cached_date = None
    existing_files_count = 0
    for ticker in tickers:
        file_path = os.path.join(RAW_MOMENTUM_DIR, f"{ticker}.csv")
        if os.path.exists(file_path):
            try:
                df_temp = pd.read_csv(file_path, index_col='Date', parse_dates=True)
                if not df_temp.empty:
                    existing_files_count += 1
                    d_max = df_temp.index.max()
                    if max_cached_date is None or d_max > max_cached_date:
                        max_cached_date = d_max
            except Exception:
                pass

    # Determine if there are any new market days to check against
    has_new_market_days = True
    if max_cached_date is not None:
        max_cached_dt = pd.to_datetime(max_cached_date).normalize()
        now_dt = now_utc.normalize()
        
        # Simple market day heuristic: if max cached date is today or later, 
        # or if the gap only covers weekends and today is before market close (or weekend itself), 
        # check if there's any weekday gap.
        # More robustly, let's generate business days between max_cached_dt + 1 day and now_dt.
        if max_cached_dt >= now_dt:
            has_new_market_days = False
        else:
            bus_days = pd.bdate_range(start=max_cached_dt + pd.Timedelta(days=1), end=now_dt)
            if len(bus_days) == 0:
                has_new_market_days = False

    if not has_new_market_days and existing_files_count >= len(tickers):
        print("    -> Cache is fully up-to-date with current market days. Skipping updates for existing tickers; checking only for missing tickers...")

    for j, ticker in enumerate(tickers):
        file_path = os.path.join(RAW_MOMENTUM_DIR, f"{ticker}.csv")
        file_exists = os.path.exists(file_path)

        # If cache is up to date and file already exists, skip it entirely
        if not has_new_market_days and file_exists:
            continue

        try:
            if file_exists:
                df_existing = pd.read_csv(file_path, index_col='Date', parse_dates=True)
                if not df_existing.empty:
                    last_date = df_existing.index.max().strftime('%Y-%m-%d')
                    
                    url = f'https://eodhd.com/api/eod/{ticker}.US?api_token={API_KEY}&fmt=csv&from={last_date}&to={today_str}'
                    response = requests.get(url, timeout=10)
                    if response.status_code == 200 and len(response.text.strip()) > 0:
                        df_new = pd.read_csv(io.StringIO(response.text), index_col='Date', parse_dates=True)
                        if not df_new.empty:
                            df_combined = pd.concat([df_existing, df_new])
                            df_combined = df_combined[~df_combined.index.duplicated(keep='last')]
                            df_combined = df_combined.sort_index()
                            
                            df_combined.to_csv(file_path)
                            print(f"Updated raw momentum for {ticker} with new market days ({j+1}/{len(tickers)})")
                            continue
                
                url = f'https://eodhd.com/api/eod/{ticker}.US?api_token={API_KEY}&fmt=csv&from={PIN_DATE}'
            else:
                # Full download for new tickers not in cache
                url = f'https://eodhd.com/api/eod/{ticker}.US?api_token={API_KEY}&fmt=csv&from={PIN_DATE}'
                print(f"Downloading new ticker missing from cache: {ticker} ({j+1}/{len(tickers)})")

            response = requests.get(url, timeout=10)
            if response.status_code != 200:
                print(f"    ! Failed to fetch {ticker} (Status: {response.status_code})")
                continue
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(response.text)
                
            print(f"Saved raw momentum for {ticker} ({j+1}/{len(tickers)})")
        except Exception as e:
            print(f"    ! Error pulling/updating raw momentum for {ticker}: {e}")

def pull_raw_quality_data(tickers):
    """
    Checks cached quality files. Determines if there are new market days using the 
    momentum cache status. If no new market days exist, only pulls/downloads fundamentals 
    and holdings files for tickers/components completely missing from the cache. Otherwise, 
    incrementally merges/updates earnings history and holdings for existing entries.
    """
    os.makedirs(RAW_QUALITY_DIR, exist_ok=True)
    os.makedirs(RAW_HOLDINGS_DIR, exist_ok=True)
    print(f"Pulling/updating raw Quality data for {len(tickers)} tickers...")
    
    ticker_meta = pd.read_csv(TICKER_FILE).set_index('symbol')
    
    # Determine if there are new market days using the same logic as momentum cache check
    now_utc = pd.Timestamp.now(tz='UTC').tz_localize(None)
    max_cached_date = None
    existing_momentum_files = 0
    for ticker in tickers:
        file_path = os.path.join(RAW_MOMENTUM_DIR, f"{ticker}.csv")
        if os.path.exists(file_path):
            try:
                df_temp = pd.read_csv(file_path, index_col='Date', parse_dates=True)
                if not df_temp.empty:
                    existing_momentum_files += 1
                    d_max = df_temp.index.max()
                    if max_cached_date is None or d_max > max_cached_date:
                        max_cached_date = d_max
            except Exception:
                pass

    has_new_market_days = True
    if max_cached_date is not None:
        max_cached_dt = pd.to_datetime(max_cached_date).normalize()
        now_dt = now_utc.normalize()
        if max_cached_dt >= now_dt:
            has_new_market_days = False
        else:
            bus_days = pd.bdate_range(start=max_cached_dt + pd.Timedelta(days=1), end=now_dt)
            if len(bus_days) == 0:
                has_new_market_days = False

    for j, ticker in enumerate(tickers):
        try:
            asset_type = ticker_meta.loc[ticker, 'asset_type'].upper()
            direction = ticker_meta.loc[ticker, 'direction'].upper()
            
            if asset_type == 'STOCK':
                json_path = os.path.join(RAW_QUALITY_DIR, f"{ticker}.json")
                file_exists = os.path.exists(json_path)
                
                # If cache is up to date and file exists, skip entirely
                if not has_new_market_days and file_exists:
                    continue
                
                if not file_exists:
                    print(f"Downloading new quality stock missing from cache: {ticker} ({j+1}/{len(tickers)})")

                url = f'https://eodhd.com/api/fundamentals/{ticker}.US?api_token={API_KEY}'
                resp = requests.get(url, timeout=10)
                
                if resp.status_code == 200:
                    new_data = resp.json()
                    if file_exists:
                        with open(json_path, 'r', encoding='utf-8') as f:
                            existing_data = json.load(f)
                        
                        existing_history = existing_data.get('Earnings', {}).get('History', {})
                        new_history = new_data.get('Earnings', {}).get('History', {})
                        if existing_history and new_history:
                            existing_history.update(new_history)
                            existing_data['Earnings']['History'] = existing_history
                        else:
                            existing_data = new_data
                            
                        with open(json_path, 'w', encoding='utf-8') as f:
                            json.dump(existing_data, f)
                    else:
                        with open(json_path, 'w', encoding='utf-8') as f:
                            f.write(resp.text)
            
            elif asset_type == 'ETF' and direction == 'LONG':
                holdings_file = f'./holdings/{ticker}_holdings.csv'
                target_holdings_path = os.path.join(RAW_HOLDINGS_DIR, f"{ticker}_holdings.csv")
                holdings_exists = os.path.exists(target_holdings_path)
                
                if os.path.exists(holdings_file):
                    df_holdings = pd.read_csv(holdings_file)
                    df_holdings.to_csv(target_holdings_path, index=False)
                    
                    for _, row in df_holdings.iterrows():
                        h_raw_code = str(row['Code']).strip()
                        h_code = f"{h_raw_code}.US" if "." not in h_raw_code else h_raw_code
                        h_json_path = os.path.join(RAW_QUALITY_DIR, f"{h_code}.json")
                        h_exists = os.path.exists(h_json_path)
                        
                        if not has_new_market_days and holdings_exists and h_exists:
                            continue
                        
                        if not h_exists:
                            print(f"Downloading new underlying holding quality file missing from cache: {h_code}")

                        h_url = f'https://eodhd.com/api/fundamentals/{h_code}?api_token={API_KEY}'
                        h_resp = requests.get(h_url, timeout=10)
                        
                        if h_resp.status_code == 200:
                            h_new_data = h_resp.json()
                            if h_exists:
                                with open(h_json_path, 'r', encoding='utf-8') as f:
                                    h_existing_data = json.load(f)
                                
                                h_existing_history = h_existing_data.get('Earnings', {}).get('History', {})
                                h_new_history = h_new_data.get('Earnings', {}).get('History', {})
                                if h_existing_history and h_new_history:
                                    h_existing_history.update(h_new_history)
                                    h_existing_data['Earnings']['History'] = h_existing_history
                                else:
                                    h_existing_data = h_new_data
                                    
                                with open(h_json_path, 'w', encoding='utf-8') as f:
                                    json.dump(h_existing_data, f)
                            else:
                                with open(h_json_path, 'w', encoding='utf-8') as f:
                                    f.write(h_resp.text)
                                    
            print(f"Checked/updated raw quality data for {ticker} ({j+1}/{len(tickers)})")
        except Exception as e:
            print(f"    ! Error pulling/updating raw quality for {ticker}: {e}")

def pull_raw_dividends_data(tickers, start_date, end_date):
    """
    Checks existing dividend cache. Determines if there are new market days relevant 
    to dividends after the latest cached date. If no new days exist, only pulls for 
    tickers completely missing from the cache. Otherwise, pulls only new data incrementally 
    for existing cached tickers from their respective latest cached dates onwards, and full range for new tickers.
    """
    print("Pulling/updating raw dividend data...")
    now_utc = pd.Timestamp.now(tz='UTC').tz_localize(None)
    today_str = now_utc.strftime('%Y-%m-%d')
    
    # Load existing cache if available
    df_existing = None
    max_cached_div_date = None
    if os.path.isfile(DIVIDEND_PATH):
        try:
            df_existing = pd.read_parquet(DIVIDEND_PATH)
            if not df_existing.empty and 'ex_dividend_date' in df_existing.columns:
                max_cached_div_date = df_existing['ex_dividend_date'].max()
        except Exception as e:
            print(f"    ! Error reading existing dividend cache: {e}")

    # Determine if there are new market days after the latest cached dividend date
    has_new_market_days = True
    if max_cached_div_date is not None and pd.notnull(max_cached_div_date):
        max_div_dt = pd.to_datetime(max_cached_div_date).normalize()
        now_dt = now_utc.normalize()
        if max_div_dt >= now_dt:
            has_new_market_days = False
        else:
            bus_days = pd.bdate_range(start=max_div_dt + pd.Timedelta(days=1), end=now_dt)
            if len(bus_days) == 0:
                has_new_market_days = False

    cached_tickers = set(df_existing['ticker'].unique()) if (df_existing is not None and not df_existing.empty) else set()

    if not has_new_market_days and len(cached_tickers) >= len(tickers):
        print("    -> Dividend cache is fully up-to-date. Skipping updates for existing tickers; checking only for missing tickers...")

    all_dividends = []
    
    for i, ticker in enumerate(tickers):
        ticker_upper = ticker.upper()
        ticker_in_cache = ticker_upper in cached_tickers

        # If cache is up to date and ticker is already cached, skip entirely
        if not has_new_market_days and ticker_in_cache:
            continue

        t_start = start_date
        if ticker_in_cache and df_existing is not None:
            ticker_subset = df_existing[df_existing['ticker'] == ticker_upper]
            if not ticker_subset.empty:
                max_cached_date = ticker_subset['ex_dividend_date'].max()
                if pd.notnull(max_cached_date):
                    t_start = (pd.to_datetime(max_cached_date) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        elif not ticker_in_cache:
            print(f"Downloading new dividend ticker missing from cache: {ticker_upper}")

        if pd.to_datetime(t_start) > pd.to_datetime(end_date):
            continue

        url = f"https://eodhd.com/api/div/{ticker}"
        params = {
            "api_token": API_KEY,
            "from": t_start,
            "to": end_date,
            "fmt": "json",
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list):
                    for div in data:
                        all_dividends.append({
                            "ticker": ticker_upper,
                            "ex_dividend_date": div.get("date"),
                            "payment_date": div.get("paymentDate"),
                            "amount": div.get("value"),
                        })
        except Exception as e:
            print(f"    ! Error pulling dividends for {ticker}: {e}")

    df_new = pd.DataFrame(all_dividends)
    if not df_new.empty:
        df_new["ex_dividend_date"] = pd.to_datetime(df_new["ex_dividend_date"], errors="coerce")
        df_new["payment_date"] = pd.to_datetime(df_new["payment_date"], errors="coerce")

    if df_existing is not None and not df_existing.empty:
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined = df_combined.drop_duplicates(subset=["ticker", "ex_dividend_date", "amount"])
    else:
        df_combined = df_new

    if not df_combined.empty:
        df_combined = df_combined.sort_values(by=["ex_dividend_date", "ticker"]).reset_index(drop=True)

    return df_combined


def pull_raw_data():
    """Part 1 Orchestrator: Pulls all raw data from sources and caches to disk incrementally."""
    os.makedirs(RAW_CACHE_DIR, exist_ok=True)
    csv_tickers = sorted(pd.read_csv(TICKER_FILE)['symbol'].unique().tolist())
    
    pull_raw_fred_and_bil_data()
    pull_raw_momentum_data(csv_tickers)
    pull_raw_quality_data(csv_tickers)

    now_utc = pd.Timestamp.now(tz='UTC').tz_localize(None)
    if os.path.isfile(DIVIDEND_PATH):
        df_dividends = pd.read_parquet(DIVIDEND_PATH)
        existing_tickers = df_dividends['ticker'].unique().tolist()
        new_tickers = sorted(list(set(csv_tickers).difference(existing_tickers)))
        start_date = df_dividends.ex_dividend_date.min().strftime('%Y-%m-%d')
        
        if now_utc > df_dividends.ex_dividend_date.max():
            df_add = pull_raw_dividends_data(csv_tickers, df_dividends.ex_dividend_date.max().strftime('%Y-%m-%d'), now_utc.strftime('%Y-%m-%d'))
            df_dividends = pd.concat([df_dividends, df_add]).drop_duplicates()
        if len(new_tickers) > 0:
            df_add = pull_raw_dividends_data(new_tickers, PIN_DATE, now_utc.strftime('%Y-%m-%d'))
            df_dividends = pd.concat([df_dividends, df_add]).drop_duplicates()
        df_dividends.to_parquet(DIVIDEND_PATH)
    else:
        df_dividends = pull_raw_dividends_data(csv_tickers, PIN_DATE, now_utc.strftime('%Y-%m-%d'))
        df_dividends.to_parquet(DIVIDEND_PATH)

    print("Part 1: Raw data pulling and disk caching complete.\n")


# ==========================================
# PART 2: PROCESSING & OUTPUT GENERATION
# ==========================================

def create_bil_dataarray(df: pd.DataFrame, max_date) -> xr.DataArray:
    df = df.copy()
    df['growth_factor'] = (1 + .01 * df['bil_rate']) ** (28 / 365)
    windows = [1, 6, 13, 26]
    bands = ['price_end'] + [f'dollar_ret_{w}p' for w in windows]
    symbols = ['BIL']
    dates = np.array(df.index)
    
    da = xr.DataArray(
        np.nan,
        coords={'band': bands, 'symbol': symbols, 'date': dates},
        dims=('band', 'symbol', 'date')
    )
    da.loc[dict(band='price_end')] = df.close
    df.loc[df.index > max_date, 'growth_factor'] = np.nan
    
    for i, w in enumerate(windows, start=1):
        trailing_factor = (
            df['growth_factor']
            .rolling(window=w)
            .apply(lambda x: x.prod(), raw=True)
        )
        annualized_return = ((trailing_factor ** (1 / (((w * 28) / 365.0)))) - 1) 
        da.loc[dict(band=f'dollar_ret_{w}p')] = annualized_return
        
    return da


def get_bil_data_from_cache(target_dates, max_date):
    if not os.path.exists(RAW_FRED_PATH):
        print("Cached FRED data missing.")
        return None
        
    df = pd.read_csv(RAW_FRED_PATH, index_col='observation_date', parse_dates=True)
    df = df.rename(columns={'DTB3': 'RATE'})
    df['RATE'] = pd.to_numeric(df['RATE'], errors='coerce')
    df = df.ffill().bfill()

    bil_csv = os.path.join(RAW_MOMENTUM_DIR, "BIL.csv")
    if os.path.exists(bil_csv):
        with open(bil_csv, 'r', encoding='utf-8') as f:
            df1 = pd.read_csv(io.StringIO(f.read()), index_col='Date', parse_dates=True)
        df['close'] = df1['Close']
    
    processed_df = df.reindex(df.index.union(target_dates)).ffill().reindex(target_dates)
    processed_df['bil_rate'] = processed_df['RATE'] - 0.07

    return create_bil_dataarray(processed_df, max_date)


def build_momentum_from_cache(tickers, target_dates, max_date):
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
                num_years = (window * 28) / 365.0
                annualized_return = (((pct_change + 1) ** (1 / num_years)) - 1) 
                expected_1yr_profit = series * annualized_return
                full_data[i, j, :] = expected_1yr_profit.values

        except Exception as e:
            print(f"    ! Error processing cached momentum for {ticker}: {e}")
            continue

    return xr.DataArray(full_data, coords={'band': MOM_BANDS, 'symbol': tickers, 'date': target_dates}, dims=['band', 'symbol', 'date'])


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
    """Part 2 Orchestrator: Dynamically targets dates backwards from the last market date spanning 2 years."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_tickers = sorted(pd.read_csv(TICKER_FILE)['symbol'].unique().tolist())
    
    # Read the latest available market date from a reference ticker to use as end date
    ref_ticker = csv_tickers[0]
    ref_file = os.path.join(RAW_MOMENTUM_DIR, f"{ref_ticker}.csv")
    if os.path.exists(ref_file):
        with open(ref_file, 'r', encoding='utf-8') as f:
            df_ref = pd.read_csv(io.StringIO(f.read()), index_col='Date', parse_dates=True)
        max_date = df_ref.index.max()
    else:
        max_date = pd.Timestamp.now(tz='UTC').tz_localize(None)

    # Work backwards 2 years (approx 26 intervals of 28 days) from the last market date
    num_periods = (365 * 3) // DAYS_INTERVAL
    target_dates = pd.date_range(end=max_date, periods=num_periods, freq=f'{DAYS_INTERVAL}D')
    target_dates = target_dates[target_dates >= pd.to_datetime(PIN_DATE)]

    da_mom = build_momentum_from_cache(csv_tickers, target_dates, max_date)
    da_mom.to_netcdf(MOMENTUM_PATH)

    da_qual = build_quality_from_cache(csv_tickers, target_dates, da_mom)
    da_qual.to_netcdf(QUALITY_PATH)

    da_bil = get_bil_data_from_cache(target_dates, max_date)
    if da_bil is not None:
        da_bil.to_netcdf(BIL_PATH)

    run_strict_alignment_check(da_mom, da_qual, csv_tickers)

    get_macro_data()
    get_macro_signals()

    print("Part 2: Processing and output generation complete.")


def main():
    # --- PART 1: Pull raw data and store on disk ---
    #pull_raw_data()
    
    # --- PART 2: Read disk data, process with dynamic backward-looking dates, and output final files ---
    build_simulation_outputs()


if __name__ == "__main__":
    main()