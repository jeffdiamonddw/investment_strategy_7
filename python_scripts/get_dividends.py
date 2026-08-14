



import pandas as pd
import requests

API_KEY = '693327461e9541.04731237'


def get_dividends_df(tickers, start_date, end_date, api_token=API_KEY):
  """Fetches historical dividend data for a list of tickers from EODHD

  within a specified date range and returns a combined Pandas DataFrame.
  """
  all_dividends = []

  i = -1
  for ticker in tickers:
    i+=1
    print("{}/{}".format(i, len(tickers)))
    url = f"https://eodhd.com/api/div/{ticker}"
    params = {
        "api_token": api_token,
        "from": start_date,
        "to": end_date,
        "fmt": "json",
    }

    response = requests.get(url, params=params)
    if response.status_code == 200:
      data = response.json()
      if isinstance(data, list):
        for div in data:
          all_dividends.append({
              "ticker": ticker.upper(),
              "ex_dividend_date": div.get("date"),
              "payment_date": div.get("paymentDate"),
              "amount": div.get("value"),
          })

  df = pd.DataFrame(all_dividends)

  if not df.empty:
    df["ex_dividend_date"] = pd.to_datetime(
        df["ex_dividend_date"], errors="coerce"
    )
    df["payment_date"] = pd.to_datetime(df["payment_date"], errors="coerce")
    df = df.sort_values(by=["ex_dividend_date", "ticker"]).reset_index(drop=True)

  return df


if __name__ == "__main__":

  tickers = list(pd.read_csv('strategy/multi_dim_stock_list.csv').symbol)
  start_date = pd.to_datetime('2025-07-01')
  end_date = pd.Timestamp.now(tz='UTC').tz_localize(None)
  df_div = get_dividends_df(tickers, start_date, end_date)
  df_div.to_parquet('simulation_data/dividends.parquet')