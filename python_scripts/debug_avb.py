import json
import requests
from datetime import datetime

API_TOKEN = '693327461e9541.04731237'

def log_with_time(message):
    """Utility to print messages with clear timestamps."""
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)

def run_eodhd_diagnostic(ticker="AVB"):
    """Performs a deep diagnostic on the EODHD REST endpoint for a single ticker."""
    clean_ticker = ticker.upper().replace(".US", "")
    formatted_ticker = f"{clean_ticker}.US"
    
    url = "https://eodhd.com/api/us-quote-delayed"
    params = {
        "s": formatted_ticker,
        "api_token": API_TOKEN,
        "fmt": "json"
    }
    
    log_with_time(f"--- STARTING DIAGNOSTIC FOR: {formatted_ticker} ---")
    log_with_time(f"Request URL: {url}")
    log_with_time(f"Request Parameters: {params}")
    
    try:
        response = requests.get(url, params=params, timeout=10)
        
        # 1. HTTP Status & Headers Check
        log_with_time(f"HTTP Status Code: {response.status_code}")
        log_with_time(f"Response Headers: {dict(response.headers)}")
        
        if response.status_code != 200:
            log_with_time(f"[ERROR] Non-200 status code received: {response.text}")
            return
            
        # 2. Raw Content Check
        raw_text = response.text.strip()
        log_with_time(f"Raw Response Text Length: {len(raw_text)} characters")
        log_with_time(f"Raw Response Text Preview: {raw_text[:500]}")
        
        if not raw_text or raw_text == "null":
            log_with_time("[DIAGNOSTIC FINDING] The API returned an outright empty or null string response. This typically indicates an invalid token, a subscription limitation blocking this specific endpoint, or an unrecognized symbol format.")
            return

        # 3. JSON Parsing Check
        try:
            data = response.json()
        except json.JSONDecodeError as jde:
            log_with_time(f"[ERROR] Failed to parse response text as JSON: {jde}")
            return
            
        log_with_time(f"Parsed JSON Type: {type(data)}")
        log_with_time(f"Parsed JSON Content: {json.dumps(data, indent=2)}")
        
        # 4. Payload Structure Inspection
        # EODHD /us-quote-delayed usually nests items under a 'data' dictionary or returns a list.
        target_dict = data.get("data", data) if isinstance(data, dict) else data
        
        if not target_dict:
            log_with_time("[DIAGNOSTIC FINDING] The JSON parsed successfully, but the container dictionary is empty.")
            return
            
        # Check if our target symbol or its variation exists in the keys
        matching_key = None
        for key in target_dict.keys():
            if clean_ticker in key.upper():
                matching_key = key
                break
                
        if not matching_key:
            log_with_time(f"[DIAGNOSTIC FINDING] Symbol '{formatted_ticker}' was not found among the returned keys: {list(target_dict.keys())}")
            return
            
        details = target_dict[matching_key]
        log_with_time(f"Found Data Key: '{matching_key}' -> Details type: {type(details)}")
        
        if not isinstance(details, dict):
            log_with_time(f"[DIAGNOSTIC FINDING] Expected a dictionary of fields for '{matching_key}', but got: {details}")
            return
            
        # 5. Field Mapping Check
        log_with_time("--- FIELD EXTRACTION CHECK ---")
        fields_to_check = {
            "bidPrice": details.get("bidPrice"),
            "bidSize": details.get("bidSize"),
            "askPrice": details.get("askPrice"),
            "askSize": details.get("askSize"),
            "lastTradePrice": details.get("lastTradePrice"),
            "lastTradeDateTime": details.get("lastTradeDateTime"),
            "volume": details.get("volume")
        }
        
        for field_name, val in fields_to_check.items():
            log_with_time(f"  - {field_name}: {val} (Type: {type(val)})")
            
        if all(v is None for v in [fields_to_check["bidPrice"], fields_to_check["askPrice"], fields_to_check["lastTradePrice"]]):
            log_with_time("[DIAGNOSTIC FINDING] All price fields are returning None. Even though the endpoint responded, the exchange feed for this ticker may be blank or restricted on your current API subscription level.")
        else:
            log_with_time("[SUCCESS] Valid price data successfully parsed!")

    except Exception as e:
        log_with_time(f"[EXCEPTION] An error occurred during the request: {e}")

if __name__ == "__main__":
    run_eodhd_diagnostic("AVB")