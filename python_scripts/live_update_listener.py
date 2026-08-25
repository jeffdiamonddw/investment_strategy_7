import asyncio
import json
import os
import threading
import time
import pandas as pd
import requests
import websockets
from datetime import datetime

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
    """Step 1: Instantly populate baseline data using the REST endpoint."""
    log_with_time("[REST] Fetching initial quote snapshot to populate baseline...")
    clean_tickers = [t.upper().replace(".US", "") for t in tickers]
    chunks = [clean_tickers[i:i + 34] for i in range(0, len(clean_tickers), 34)]
    
    initial_rows = []
    for chunk in chunks:
        # REST requires .US suffix
        formatted_tickers = ",".join([f"{t}.US" for t in chunk])
        url = "https://eodhd.com/api/us-quote-delayed"
        params = {"s": formatted_tickers, "api_token": API_TOKEN, "fmt": "json"}
        try:
            res = requests.get(url, params=params, timeout=5)
            if res.status_code == 200:
                data_dict = res.json().get("data", {})
                for symbol, details in data_dict.items():
                    if isinstance(details, dict):
                        sym = symbol.replace(".US", "")
                        initial_rows.append({
                            "symbol": sym,
                            "bid_price": details.get("bidPrice"),
                            "bid_size": to_shares(details.get("bidSize")),
                            "ask_price": details.get("askPrice"),
                            "ask_size": to_shares(details.get("askSize")),
                            "daily_volume": details.get("volume"),
                            "last_price": details.get("lastTradePrice"),
                            "timestamp": details.get("lastTradeDateTime") or time.time()
                        })
        except Exception as e:
            log_with_time(f"[REST ERROR] {e}")
            
    # Fallback structure for any missing symbols
    existing_symbols = {r["symbol"] for r in initial_rows}
    for t in clean_tickers:
        if t not in existing_symbols:
            initial_rows.append({
                "symbol": t,
                "bid_price": None, "bid_size": None,
                "ask_price": None, "ask_size": None,
                "daily_volume": None, "last_price": None, "timestamp": None
            })
            
    df = pd.DataFrame(initial_rows)
    with _CACHE_LOCK:
        for _, row in df.iterrows():
            _LIVE_DATA[row["symbol"]] = row.to_dict()
            
    save_cache_to_parquet()
    log_with_time("[REST] Initial baseline Parquet file created successfully.")

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
    try:
        raw_symbols = pd.read_csv('strategy/multi_dim_stock_list.csv').symbol.tolist()
    except Exception:
        raw_symbols = ["AAPL", "MSFT", "XLG"]

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