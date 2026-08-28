import argparse
import asyncio
import json
import os
import threading
import time
import pandas as pd
import requests
import websockets
from datetime import datetime
import numpy as np

PARQUET_FILE = "live_quotes_cache.parquet"
API_TOKEN = '693327461e9541.04731237'

# Thread-safe in-memory state dictionary
_CACHE_LOCK = threading.Lock()
_LIVE_DATA = {}

def log_with_time(message):
    """Utility to print messages with clear timestamps for log files."""
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)

def to_shares(val):
    """Safely converts a lot-size value to total shares (multiplies by 100)."""
    try:
        if val is not None:
            return float(val) * 100.0
    except (ValueError, TypeError):
        pass
    return None

def initialize_from_rest(tickers):
    """Step 1: Instantly populate baseline data using the REST endpoint.
    Includes multi-key fallback parsing and secondary live-endpoint fallback for stubborn tickers.
    """
    log_with_time("[REST] Fetching initial quote snapshot to populate baseline...")
    clean_tickers = [t.upper().replace(".US", "") for t in tickers]
    chunks = [clean_tickers[i:i + 34] for i in range(0, len(clean_tickers), 34)]
    
    initial_rows = []
    for chunk in chunks:
        formatted_tickers = ",".join([f"{t}.US" for t in chunk])
        url = "https://eodhd.com/api/us-quote-delayed"
        params = {"s": formatted_tickers, "api_token": API_TOKEN, "fmt": "json"}
        try:
            res = requests.get(url, params=params, timeout=5)
            if res.status_code == 200:
                data_json = res.json()
                # Handle both list structures and dictionary 'data' groupings
                data_dict = data_json.get("data", data_json) if isinstance(data_json, dict) else {}
                
                for symbol, details in data_dict.items():
                    if isinstance(details, dict):
                        sym = symbol.replace(".US", "")
                        
                        # Check multiple possible key variations returned by different EODHD tiers
                        last_p = (details.get("lastTradePrice") or 
                                  details.get("close") or 
                                  details.get("price") or 
                                  details.get("last"))
                                  
                        bid_p = details.get("bidPrice") or details.get("bid")
                        ask_p = details.get("askPrice") or details.get("ask")
                        vol = details.get("volume") or details.get("daily_volume")
                        ts = (details.get("lastTradeDateTime") or 
                              details.get("timestamp") or 
                              time.time())
                        
                        initial_rows.append({
                            "symbol": sym,
                            "bid_price": bid_p,
                            "bid_size": to_shares(details.get("bidSize")),
                            "ask_price": ask_p,
                            "ask_size": to_shares(details.get("askSize")),
                            "daily_volume": vol,
                            "last_price": last_p,
                            "timestamp": ts
                        })
        except Exception as e:
            log_with_time(f"[REST ERROR] {e}")
            
    df = pd.DataFrame(initial_rows)
    
    # Secondary Fallback: If any ticker still has null/missing last_price, query the 
    # individual /api/real-time/ endpoint as a backup source.
    if not df.empty and ("last_price" in df.columns):
        null_mask = df["last_price"].isna() | (df["last_price"] == "")
        if null_mask.any():
            stale_syms = df.loc[null_mask, "symbol"].tolist()
            log_with_time(f"[REST FALLBACK] Querying individual real-time endpoints for missing symbols: {stale_syms}")
            for sym in stale_syms:
                rt_url = f"https://eodhd.com/api/real-time/{sym}.US"
                rt_params = {"api_token": API_TOKEN, "fmt": "json"}
                try:
                    rt_res = requests.get(rt_url, params=rt_params, timeout=3)
                    if rt_res.status_code == 200:
                        rt_data = rt_res.json()
                        if isinstance(rt_data, dict) and rt_data.get("close"):
                            idx = df.index[df["symbol"] == sym][0]
                            df.at[idx, "last_price"] = rt_data.get("close")
                            df.at[idx, "daily_volume"] = rt_data.get("volume")
                            df.at[idx, "timestamp"] = rt_data.get("timestamp", time.time())
                            log_with_time(f"[REST FALLBACK SUCCESS] Recovered price for {sym}: {rt_data.get('close')}")
                except Exception as ex:
                    log_with_time(f"[REST FALLBACK ERROR for {sym}] {ex}")

    # Fallback structure for any completely missing symbols
    existing_symbols = set(df["symbol"]) if not df.empty else set()
    missing_rows = []
    for t in clean_tickers:
        if t not in existing_symbols:
            missing_rows.append({
                "symbol": t,
                "bid_price": None, "bid_size": None,
                "ask_price": None, "ask_size": None,
                "daily_volume": None, "last_price": None, "timestamp": None
            })
    if missing_rows:
        df = pd.concat([df, pd.DataFrame(missing_rows)], ignore_index=True)
            
    # Merge with existing Parquet cache if available to rescue stale gaps
    if os.path.exists(PARQUET_FILE):
        try:
            old_df = pd.read_parquet(PARQUET_FILE)
            old_map = old_df.set_index("symbol").to_dict(orient="index")
            
            for idx, row in df.iterrows():
                sym = row["symbol"]
                if sym in old_map:
                    for col in ["bid_price", "bid_size", "ask_price", "ask_size", "daily_volume", "last_price"]:
                        if pd.isna(row[col]) and not pd.isna(old_map[sym].get(col)):
                            df.at[idx, col] = old_map[sym][col]
                            log_with_time(f"[CACHE MERGE] Preserved cached value for {sym} ({col}: {old_map[sym][col]})")
        except Exception as ex:
            log_with_time(f"[CACHE MERGE ERROR] {ex}")

    with _CACHE_LOCK:
        for _, row in df.iterrows():
            _LIVE_DATA[row["symbol"]] = row.to_dict()
            
    save_cache_to_parquet()
    log_with_time("[REST] Initial baseline Parquet file created successfully.")
    return df


from concurrent.futures import ThreadPoolExecutor, as_completed

def _fetch_single_ticker_baseline(t):
    """Helper to fetch EOD baseline and live patch for a single ticker."""
    sym_us = f"{t}.US"
    row_data = {
        "symbol": t,
        "bid_price": None, "bid_size": None,
        "ask_price": None, "ask_size": None,
        "daily_volume": None,
        "last_price": None,
        "timestamp": None
    }
    
    # 1. Try EOD historical close baseline
    eod_url = f"https://eodhd.com/api/eod/{sym_us}"
    eod_params = {"api_token": API_TOKEN, "fmt": "json", "limit": 1}
    try:
        res = requests.get(eod_url, params=eod_params, timeout=3)
        if res.status_code == 200:
            eod_json = res.json()
            if isinstance(eod_json, list) and len(eod_json) > 0:
                latest = eod_json[-1]
                row_data["last_price"] = latest.get("close")
                row_data["daily_volume"] = latest.get("volume")
                if latest.get("date"):
                    row_data["timestamp"] = pd.Timestamp(latest.get("date")).timestamp()
            elif isinstance(eod_json, dict) and "close" in eod_json:
                row_data["last_price"] = eod_json.get("close")
                row_data["daily_volume"] = eod_json.get("volume")
                if eod_json.get("date"):
                    row_data["timestamp"] = pd.Timestamp(eod_json.get("date")).timestamp()
    except Exception:
        pass

    # 2. Check live/delayed snapshot to overlay intra-session updates if available
    live_url = f"https://eodhd.com/api/real-time/{sym_us}"
    live_params = {"api_token": API_TOKEN, "fmt": "json"}
    try:
        live_res = requests.get(live_url, params=live_params, timeout=2)
        if live_res.status_code == 200:
            live_data = live_res.json()
            if isinstance(live_data, dict):
                live_price = live_data.get("close") or live_data.get("price")
                if live_price is not None:
                    row_data["last_price"] = live_price
                    if live_data.get("volume") is not None:
                        row_data["daily_volume"] = live_data.get("volume")
                    if live_data.get("timestamp") is not None:
                        row_data["timestamp"] = live_data.get("timestamp")
    except Exception:
        pass

    return row_data


import pandas as pd
import requests

import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

def get_last_close(tickers, api_token='693327461e9541.04731237'):
    """Fetches closing prices using bulk download first, then uses a fast 
    concurrent thread pool to fill in any missing/null tickers.
    """
    clean_tickers = [t.upper().replace(".US", "") for t in tickers]
    prices_dict = {}

    # 1. Attempt Bulk Download
    bulk_url = "https://eodhd.com/api/eod-bulk-last-day/US"
    params = {"api_token": api_token, "fmt": "json"}
    
    try:
        response = requests.get(bulk_url, params=params, timeout=15)
        if response.status_code == 200:
            df_bulk = pd.DataFrame(response.json())
            if "code" in df_bulk.columns and "close" in df_bulk.columns:
                bulk_map = pd.Series(df_bulk["close"].values, index=df_bulk["code"]).to_dict()
                for t in clean_tickers:
                    if t in bulk_map:
                        prices_dict[t] = bulk_map[t]
    except Exception as e:
        print(f"[BULK WARNING] Bulk fetch failed ({e}), falling back to individual parallel requests.")

    # 2. Identify missing tickers (like your 11 stragglers)
    missing_tickers = [t for t in clean_tickers if t not in prices_dict]

    if missing_tickers:
        print(f"[FALLBACK] Fetching {len(missing_tickers)} missing tickers individually via threads...")
        
        def fetch_single(t):
            url = f"https://eodhd.com/api/eod/{t}.US"
            p = {"api_token": api_token, "fmt": "json", "limit": 1}
            try:
                res = requests.get(url, params=p, timeout=3)
                if res.status_code == 200:
                    data = res.json()
                    if isinstance(data, list) and len(data) > 0:
                        return t, data[-1].get("close")
                    elif isinstance(data, dict) and "close" in data:
                        return t, data.get("close")
            except Exception:
                pass
            return t, None

        with ThreadPoolExecutor(max_workers=max(1, len(missing_tickers))) as executor:
            future_to_ticker = {executor.submit(fetch_single, t): t for t in missing_tickers}
            for future in as_completed(future_to_ticker):
                t, close_val = future.result()
                if close_val is not None:
                    prices_dict[t] = close_val

    # 3. Build final Series aligned with original input order
    final_series = pd.Series([prices_dict.get(t) for t in clean_tickers], index=clean_tickers)
    return final_series



def get_recent_price(tickers):
    """Step 1: Rapidly populates baseline data using parallel thread execution 
    for robust EOD close fetching and live patching across 500+ tickers.
    """
    s_price = get_robust_last_closes(tickers)

    log_with_time("[REST] Fetching baseline closing prices in parallel...")
    clean_tickers = [t.upper().replace(".US", "") for t in tickers]
    
    initial_rows = []
    # Use a thread pool with up to 30 concurrent workers to speed up requests drastically
    max_workers = min(30, len(clean_tickers)) if clean_tickers else 1
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_ticker = {
            executor.submit(_fetch_single_ticker_baseline, t): t for t in clean_tickers
        }
        for future in as_completed(future_to_ticker):
            try:
                data = future.result()
                initial_rows.append(data)
            except Exception as e:
                t = future_to_ticker[future]
                log_with_time(f"[REST ERROR for {t}] {e}")

    df = pd.DataFrame(initial_rows)

    # Merge with existing Parquet cache if available to rescue any previously cached state
    if os.path.exists(PARQUET_FILE):
        try:
            old_df = pd.read_parquet(PARQUET_FILE)
            old_map = old_df.set_index("symbol").to_dict(orient="index")
            
            for idx, row in df.iterrows():
                sym = row["symbol"]
                if sym in old_map:
                    for col in ["bid_price", "bid_size", "ask_price", "ask_size", "daily_volume", "last_price"]:
                        if pd.isna(row[col]) and not pd.isna(old_map[sym].get(col)):
                            df.at[idx, col] = old_map[sym][col]
        except Exception as ex:
            log_with_time(f"[CACHE MERGE ERROR] {ex}")

    with _CACHE_LOCK:
        for _, row in df.iterrows():
            _LIVE_DATA[row["symbol"]] = row.to_dict()
            

    return df

def save_cache_to_parquet():
    """Flushes the current in-memory cache safely using an atomic write pattern."""
    with _CACHE_LOCK:
        df = pd.DataFrame(list(_LIVE_DATA.values()))
        
    temp_file = PARQUET_FILE + ".tmp"
    df.to_parquet(temp_file, index=False)
    os.replace(temp_file, PARQUET_FILE)  # Atomic rename prevents 0-byte reads

async def _websocket_listener(batch_id, tickers):
    """Step 2: Persistent WebSocket consumer using clean, un-suffixed tickers."""
    uri = f"wss://ws.eodhistoricaldata.com/ws/us-quote?api_token={API_TOKEN}"
    
    # CRITICAL FIX: WebSocket endpoint expects plain tickers without .US
    clean_symbols = [t.upper().replace(".US", "") for t in tickers]
    
    while True:
        try:
            async with websockets.connect(uri) as websocket:
                sub_message = {
                    "action": "subscribe",
                    "symbols": ",".join(clean_symbols)
                }
                await websocket.send(json.dumps(sub_message))
                log_with_time(f"[WS BATCH {batch_id}] Subscribed to {len(clean_symbols)} clean symbols. Listening...")
                
                async for message in websocket:
                    data = json.loads(message)
                    symbol = data.get("s")
                    if symbol:
                        clean_key = symbol.replace(".US", "")
                        
                        bp = data.get("bp") or data.get("bidPrice")
                        bs = to_shares(data.get("bs") or data.get("bidSize"))
                        ap = data.get("ap") or data.get("askPrice")
                        as_sz = to_shares(data.get("as") or data.get("askSize"))
                        ts = data.get("t") or data.get("timestamp") or time.time()
                        
                        with _CACHE_LOCK:
                            if clean_key in _LIVE_DATA:
                                _LIVE_DATA[clean_key]["bid_price"] = bp if bp is not None else _LIVE_DATA[clean_key]["bid_price"]
                                _LIVE_DATA[clean_key]["bid_size"] = bs if bs is not None else _LIVE_DATA[clean_key]["bid_size"]
                                _LIVE_DATA[clean_key]["ask_price"] = ap if ap is not None else _LIVE_DATA[clean_key]["ask_price"]
                                _LIVE_DATA[clean_key]["ask_size"] = as_sz if as_sz is not None else _LIVE_DATA[clean_key]["ask_size"]
                                _LIVE_DATA[clean_key]["timestamp"] = ts
                            else:
                                _LIVE_DATA[clean_key] = {
                                    "symbol": clean_key,
                                    "bid_price": bp,
                                    "bid_size": bs,
                                    "ask_price": ap,
                                    "ask_size": as_sz,
                                    "daily_volume": None,
                                    "last_price": None,
                                    "timestamp": ts
                                }
                                
                        # Persist to Parquet immediately on update
                        save_cache_to_parquet()
                        log_with_time(f"[WS BATCH {batch_id}] Updated & Saved -> Symbol: {clean_key} | Bid: {bp} ({bs} shs) | Ask: {ap} ({as_sz} shs)")
                        
        except Exception as e:
            log_with_time(f"[WS BATCH {batch_id}] Disconnected ({e}). Reconnecting in 5s...")
            await asyncio.sleep(5)

def start_background_daemon(tickers):
    """Splits tickers into chunks of 50 and boots the background loop threads."""
    clean_tickers = [t.upper().replace(".US", "") for t in tickers]
    chunks = [clean_tickers[i:i + 50] for i in range(0, len(clean_tickers), 50)]
    
    for idx, chunk in enumerate(chunks, start=1):
        def target(b_id=idx, ch=chunk):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(_websocket_listener(b_id, ch))
            
        t = threading.Thread(target=target, daemon=True)
        t.start()

# --- Main Daemon Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--s3_path', required=True)
    args = parser.parse_args()

    df_holdings = pd.read_parquet("{}/proposed_holdings.parquet".format(args.s3_path))
    df_holdings = df_holdings.loc[df_holdings.sum(1) > 0]
    df_holdings = df_holdings.loc[df_holdings.index != 'CASH']
    raw_symbols = list(np.array(df_holdings.index))

    log_with_time(f"[DAEMON] Starting live quote daemon for {len(raw_symbols)} symbols...")
    
    # 1. Establish immediate baseline so the Parquet file exists right away
    initialize_from_rest(raw_symbols)
    
    # 2. Boot up the background WebSockets to continuously patch the Parquet file
    start_background_daemon(raw_symbols)
    
    log_with_time(f"[DAEMON] Background daemon running. Parquet updates will write to '{PARQUET_FILE}'.")
    
    # Keep main thread alive
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        log_with_time("[DAEMON] Stopping daemon.")