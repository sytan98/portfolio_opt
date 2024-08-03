import argparse
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf
from os.path import exists
import config


yf.pdr_override()
pd.options.display.float_format = "{:.4%}".format


def download_data(csv_name: str, assets: list[str], start: str = "2015-01-01", end: str = "2024-04-14"):
    # Create file name
    final_csv_name = f"{csv_name}_{start}_{end}.csv"
    if not exists(final_csv_name):
        # Downloading data
        assets.sort()
        data = yf.download(assets, start=start, end=end)["Adj Close"]
        data.to_csv(final_csv_name)
    return final_csv_name


if __name__ == "__main__":
    # # Tickers of assets
    parser = argparse.ArgumentParser(prog="Rebalancer", description="Rebalancing script")
    parser.add_argument("-d", "--end_date", type=str)
    args = parser.parse_args()

    # Download data
    end_date_obj = datetime.strptime(args.end_date, "%Y-%m-%d")
    start_date_obj = end_date_obj - timedelta(days=10 * 365)

    file_name = download_data(
        "data/stock_data_for_backtesting",
        config.assets + ["CSPX.L"],
        start_date_obj.strftime("%Y-%m-%d"),
        args.end_date,
    )
    print(file_name)
    # download_data("market_prices.csv", ["SPY"])
