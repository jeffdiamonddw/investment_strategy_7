from datetime import datetime, timedelta
import pandas as pd
import requests

# --- CONFIGURATION ---
API_KEY = "693327461e9541.04731237"  # Replace with your actual EODHD API token
START_DATE = "2005-09-05"
END_DATE = datetime.today().strftime("%Y-%m-%d")


def fetch_eodhd_data(ticker, start_date, end_date):
  """Fetches historical end-of-day data from EODHD API."""
  url = f"https://eodhd.com/api/eod/{ticker}"
  params = {
      "api_token": API_KEY,
      "from": start_date,
      "to": end_date,
      "fmt": "json",
  }
  response = requests.get(url, params=params)
  if response.status_code != 200:
    raise Exception(
        f"Error fetching data for {ticker}: {response.status_code} -"
        f" {response.text}"
    )

  data = response.json()
  df = pd.DataFrame(data)
  if "date" not in df.columns:
    raise Exception(f"Unexpected response format for {ticker}: {data}")

  df["date"] = pd.to_datetime(df["date"])
  df.set_index("date", inplace=True)
  return df[["adjusted_close"]]


# 1. Fetch data for VOO (ETF) and GSPC (S&P 500 Index)
print("Fetching VOO and S&P 500 data from EODHD...")
voo_df = fetch_eodhd_data("VOO.US", START_DATE, END_DATE)
gspc_df = fetch_eodhd_data("GSPC.INDX", START_DATE, END_DATE)

# Combine into a single dataframe
combined = pd.merge(
    gspc_df, voo_df, left_index=True, right_index=True, how="left"
)
combined.columns = ["GSPC", "VOO"]

# 2. Find VOO's first available trading date (Inception baseline)
# VOO launched in Sept 2010. We find the scaling factor using its first valid price relative to the index.
valid_voo = combined[combined["VOO"].notnull()].iloc[0]
inception_date = valid_voo.name
scale_factor = valid_voo["VOO"] / valid_voo["GSPC"]
print(f"VOO Inception Date found: {inception_date.strftime('%Y-%m-%d')}")
print(f"Baseline Scale Factor (VOO / S&P500): {scale_factor:.6f}")

# 3. Estimate VOO prices prior to inception using the S&P 500 index path
# Estimated VOO = S&P 500 Index * scale_factor
combined["Estimated_VOO"] = combined["GSPC"] * scale_factor
# For dates where real VOO exists, use real VOO; otherwise use the estimate
combined["Final_VOO_Series"] = combined["VOO"].combine_first(
    combined["Estimated_VOO"]
)

# 4. Generate every 28-day multiple starting from Sept 5, 2005
start_dt = datetime.strptime(START_DATE, "%Y-%m-%d")
end_dt = datetime.strptime(END_DATE, "%Y-%m-%d")

target_dates = []
curr_date = start_dt
while curr_date <= end_dt:
  target_dates.append(curr_date)
  curr_date += timedelta(days=28)

# Convert to pandas datetime index to find closest trading days
target_index = pd.to_datetime(target_dates)

# Reindex to target 28-day intervals using 'nearest' trading day matching
result_df = combined.reindex(target_index, method="nearest")

# Format output columns
output = pd.DataFrame(
    {
        "S&P500_Index": result_df["GSPC"],
        "Actual_VOO": result_df["VOO"],
        "Estimated_or_Actual_VOO": result_df["Final_VOO_Series"],
    }
)

# Display rows
print("\n--- Extracted & Estimated VOO Data (Every 28 Days) ---")
print(output.to_string())

# Optional: Save to CSV
output.to_csv("simulation_data/voo_backcasted_28days.csv")
print("\nSaved output successfully to 'voo_backcasted_28days.csv'")