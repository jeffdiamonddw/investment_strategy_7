import pandas as pd

import xarray as xr


def calculate_cash_dividends(holdings_df, dividends_df):
  """Calculates the total dividend cash payout per ticker on each payment date.

  Parameters:
  - holdings_df: DataFrame with date index, ticker columns, and share
  counts (post-trade).
  - dividends_df: DataFrame with columns ['ticker', 'ex_dividend_date',
  'payment_date', 'amount']
  """
  # Ensure date formats match for alignment
  holdings_df.index = pd.to_datetime(holdings_df.index)
  divs = dividends_df.copy()
  divs["ex_dividend_date"] = pd.to_datetime(divs["ex_dividend_date"])
  divs["payment_date"] = pd.to_datetime(divs["payment_date"])

  results = []

  for _, row in divs.iterrows():
    ticker = row["ticker"]
    ex_date = row["ex_dividend_date"]
    pay_date = row["payment_date"]
    amount = row["amount"]

    if ticker in holdings_df.columns:
      # Find the latest available holding on or before the ex-dividend date
      valid_dates = holdings_df.index[holdings_df.index <= ex_date]

      if not valid_dates.empty:
        closest_date = valid_dates[-1]
        shares = holdings_df.loc[closest_date, ticker]

        if pd.notna(shares) and shares > 0:
          total_dividend = shares * amount
          results.append({
              "payment_date": pay_date,
              "ticker": ticker,
              "total_dividends": total_dividend,
          })

  result_df = pd.DataFrame(results)

  if not result_df.empty:
    # Group by payment date and ticker in case of multiple distributions in a period
    result_df = (
        result_df.groupby(["payment_date", "ticker"], as_index=False)[
            "total_dividends"
        ]
        .sum()
        .sort_values(by=["payment_date", "ticker"])
        .reset_index(drop=True)
    )

  return result_df



if __name__ == '__main__':

  data_hist = xr.open_dataarray('sim_results/holdings_history.nc')
  df_div = pd.read_parquet('simulation_data/dividends.parquet')

  div_list = []
  for account in list(data_hist.account.values):
    div_list += [calculate_cash_dividends(data_hist.sel(account = account).to_pandas(), df_div).total_dividends.sum()]

  