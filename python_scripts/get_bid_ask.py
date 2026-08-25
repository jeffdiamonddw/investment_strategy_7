import pandas as pd
import requests
import time

API_KEY = '693327461e9541.04731237'

def get_eodhd_quotes(tickers, api_token = API_KEY):
    """
    Fetches real-time/delayed quotes including bid, ask, and sizes for a list of US tickers.
    
    Parameters:
    - tickers (list): List of ticker symbols (e.g., ['AAPL', 'MSFT', 'XLG'])
    - api_token (str): Your EODHD API key
    
    Returns:
    - pd.DataFrame: DataFrame containing columns for symbol, bid/ask prices, and sizes.
    """
    # Format tickers into a comma-separated string (e.g., "AAPL.US,MSFT.US,XLG.US")
    formatted_tickers = ",".join([t if "." in t else f"{t}.US" for t in tickers])
    
    url = "https://eodhd.com/api/us-quote-delayed"
    params = {
        "s": formatted_tickers,
        "api_token": api_token,
        "fmt": "json"
    }
    
    response = requests.get(url, params=params)
    
    if response.status_code != 200:
        raise Exception(f"EODHD API Error: {response.status_code} - {response.text}")
        
    res_json = response.json()
    data_dict = res_json.get("data", {})
    
    rows = []
    for symbol, details in data_dict.items():
        rows.append({
            "symbol": symbol.split(".")[0],
            "last_price": details.get("lastTradePrice"),
            "bid_price": details.get("bidPrice"),
            "bid_size": details.get("bidSize"),
            "ask_price": details.get("askPrice"),
            "ask_size": details.get("askSize"),
            "timestamp": details.get("lastTradeDateTime")
        })
        
    return pd.DataFrame(rows)

import asyncio
import json
import threading
import pandas as pd
import websockets

# Global store to keep the latest quotes fresh in memory
LIVE_QUOTE_CACHE = {}

async def _eodhd_ws_listener(tickers, api_token = API_KEY):
    """Background coroutine that maintains a persistent connection to EODHD."""
    uri = f"wss://ws.eodhistoricaldata.com/ws/us-quote?api_token={api_token}"
    formatted_symbols = [t.upper().replace(".US", "") for t in tickers]
    
    while True:
        try:
            async with websockets.connect(uri) as websocket:
                # Subscribe to symbols
                sub_message = {
                    "action": "subscribe",
                    "symbols": ",".join(formatted_symbols)
                }
                await websocket.send(json.dumps(sub_message))
                
                # Continuously listen for incoming ticks
                async for message in websocket:
                    data = json.loads(message)
                    symbol = data.get("s")
                    if symbol:
                        LIVE_QUOTE_CACHE[symbol] = {
                            "symbol": symbol,
                            "last_price": data.get("p") or data.get("lastTradePrice"),
                            "bid_price": data.get("bp") or data.get("bidPrice"),
                            "bid_size": data.get("bs") or data.get("bidSize"),
                            "ask_price": data.get("ap") or data.get("askPrice"),
                            "ask_size": data.get("as") or data.get("askSize"),
                            "timestamp": data.get("t") or data.get("timestamp")
                        }
        except Exception as e:
            # Auto-reconnect after a brief pause if connection drops
            await asyncio.sleep(5)

def start_live_feed_background(tickers, api_token = API_KEY):
    """Spawns the WebSocket listener in a background daemon thread."""
    def run_loop():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(_eodhd_ws_listener(tickers, api_token))
        
    t = threading.Thread(target=run_loop, daemon=True)
    t.start()

def get_cached_live_quotes_df(tickers):
    """Instantly pulls the latest quotes from the background cache into a DataFrame."""
    rows = []
    for t in tickers:
        clean_t = t.upper().replace(".US", "")
        if clean_t in LIVE_QUOTE_CACHE:
            rows.append(LIVE_QUOTE_CACHE[clean_t])
        else:
            rows.append({"symbol": clean_t, "last_price": None, "bid_price": None, "bid_size": None, "ask_price": None, "ask_size": None, "timestamp": None})
            
    return pd.DataFrame(rows)

#

tickers = pd.read_csv('strategy/multi_dim_stock_list.csv').symbol
#df = get_eodhd_quotes(tickers)
#df_live = asyncio.run(fetch_live_quotes_ws(tickers, timeout_seconds=10))


# --- How to use it in your app or script ---
# 1. Start the listener once when your application boots up:
start_live_feed_background(tickers)

# 2. Whenever your dashboard needs quotes, it reads the memory cache instantly:
t1 = time.time()
while True:
    df_quotes = get_cached_live_quotes_df(tickers)
    print(time.time() - t1, df_quotes.iloc[:, 1:-1].notnull().values.sum())
    time.sleep(3)




