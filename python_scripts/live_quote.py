import asyncio
import json
import threading
import time
import pandas as pd
import websockets

# Unified in-memory dictionary to hold the latest live quotes from all streams
_LIVE_QUOTE_CACHE = {}

async def _run_websocket_listener(batch_id, tickers, api_token):
    """Async background task that connects to EODHD and streams a specific batch of live quotes."""
    uri = f"wss://ws.eodhistoricaldata.com/ws/us-quote?api_token={api_token}"
    formatted_symbols = [t.upper() if ".US" in t.upper() else f"{t.upper()}.US" for t in tickers]
    
    print(f"[WS BATCH {batch_id}] Connecting for {len(formatted_symbols)} symbols...")
    
    while True:
        try:
            async with websockets.connect(uri) as websocket:
                sub_message = {
                    "action": "subscribe",
                    "symbols": ",".join(formatted_symbols)
                }
                await websocket.send(json.dumps(sub_message))
                print(f"[WS BATCH {batch_id}] Subscribed successfully.")
                
                async for message in websocket:
                    data = json.loads(message)
                    symbol = data.get("s")
                    if symbol:
                        clean_key = symbol.replace(".US", "")
                        _LIVE_QUOTE_CACHE[clean_key] = {
                            "symbol": clean_key,
                            "bid_price": data.get("bp") or data.get("bidPrice"),
                            "bid_size": data.get("bs") or data.get("bidSize"),
                            "ask_price": data.get("ap") or data.get("askPrice"),
                            "ask_size": data.get("as") or data.get("askSize"),
                            "timestamp": data.get("t") or data.get("timestamp")
                        }
        except Exception as e:
            print(f"[WS BATCH {batch_id}] Error: {type(e).__name__} - {e}. Reconnecting in 3s...")
            await asyncio.sleep(3)

def start_split_streams(tickers, api_token, chunk_size=50):
    """Splits tickers into chunks and spawns a background thread for each WebSocket stream."""
    # Ensure clean base symbols
    cleaned_tickers = [t.upper().replace(".US", "") for t in tickers]
    
    # Split into chunks of 50 to respect EODHD limits
    chunks = [cleaned_tickers[i:i + chunk_size] for i in range(0, len(cleaned_tickers), chunk_size)]
    
    for idx, chunk in enumerate(chunks, start=1):
        def target(b_id=idx, ch=chunk):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(_run_websocket_listener(b_id, ch, api_token))
            
        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        print(f"[SETUP] Spawned WebSocket thread {idx} for {len(chunk)} symbols.")

def get_quotes_with_retry(tickers, max_attempts=10, delay_seconds=3):
    """
    Polls the unified live cache every `delay_seconds` until at least one 
    non-null quote is returned or max_attempts is reached.
    """
    clean_tickers = [t.upper().replace(".US", "") for t in tickers]
    
    for attempt in range(1, max_attempts + 1):
        rows = []
        has_data = False
        
        print(f"\n[CACHE POLL] Attempt {attempt}/{max_attempts} checking unified cache...")
        for t in clean_tickers:
            if t in _LIVE_QUOTE_CACHE and _LIVE_QUOTE_CACHE[t].get("bid_price") is not None:
                rows.append(_LIVE_QUOTE_CACHE[t])
                has_data = True
            else:
                rows.append({
                    "symbol": t,
                    "bid_price": None,
                    "bid_size": None,
                    "ask_price": None,
                    "ask_size": None,
                    "timestamp": None
                })
                
        if has_data:
            print("[CACHE POLL] Data found in cache! Returning DataFrame.")
            return pd.DataFrame(rows)
            
        print(f"[CACHE POLL] Cache empty for targets. Waiting {delay_seconds}s...")
        time.sleep(delay_seconds)
        
    print("[CACHE POLL] Max attempts reached with empty cache. Returning current state.")
    return pd.DataFrame(rows)

# --- Main Execution ---
if __name__ == "__main__":
    API_TOKEN = '693327461e9541.04731237'
    
    # Load all tickers from your file
    try:
        raw_symbols = pd.read_csv('strategy/multi_dim_stock_list.csv').symbol.tolist()
    except Exception as e:
        print(f"[SETUP WARNING] Could not read CSV ({e}). Using sample list.")
        raw_symbols = ["AAPL", "MSFT", "XLG", "VTI", "VOO"]

    print(f"[SETUP] Loaded {len(raw_symbols)} total symbols to track.")
    
    # 1. Start split streams across multiple background threads (chunks of 50)
    start_split_streams(raw_symbols, API_TOKEN, chunk_size=50)
    
    # Give threads a moment to initialize connection and authenticate
    time.sleep(3)
    
    # 2. Query cache with retry loop
    print("Fetching live quotes across all streams...")
    df_live = get_quotes_with_retry(raw_symbols, max_attempts=6, delay_seconds=3)
    print("\n--- Final Output DataFrame ---")
    print(df_live)