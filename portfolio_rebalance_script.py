from datetime import datetime, timedelta
import pandas as pd
import argparse
import seaborn as sb
import matplotlib.pyplot as plt

from download import download_data
from portfolio_opt_strat import historical_mvo_strategy, historical_mvo_min_vol_strategy
import config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Rebalancer", description="Rebalancing script")
    parser.add_argument("-d", "--end_date", type=str)
    args = parser.parse_args()

    # Download data
    end_date_obj = datetime.strptime(args.end_date, "%Y-%m-%d")
    start_date_obj = end_date_obj - timedelta(days=config.training_period)

    file_name = download_data("data/stock_data", config.assets, start_date_obj.strftime("%Y-%m-%d"), args.end_date)

    # Read in price data
    df = pd.read_csv(file_name, parse_dates=True, index_col="Date")
    df.dropna(inplace=True)

    returns_all = df.pct_change().dropna()
    training_instance = returns_all.iloc[-config.training_period :, :]
    latest_prices = df.iloc[-1]  # prices as of the day you are allocating
    print(latest_prices)
    print(f"Training period from {training_instance.index[0]} to {training_instance.index[-1]}")

    prev_weight = list(config.prev_weight.values())
    weights = historical_mvo_min_vol_strategy(
        training_instance, returns_data=True, frequency=252, prev_weight=prev_weight
    )

    keys = list(weights.keys())
    vals = [float(weights[k]) for k in keys]
    sb.barplot(x=keys, y=vals)
    plt.savefig(f"data/rebalanced_weights_{args.end_date}.png")
    for ticker, weight in weights.items():
        print(
            f"Ticker {ticker}: Weight {weight} Shares {round(weight * config.current_portfolio_val / latest_prices[ticker], 2)}"
        )
