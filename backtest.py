import pandas as pd
from datetime import datetime, timedelta


def backtest_rebalance(
    training_df: pd.DataFrame,
    test_df: pd.DataFrame,
    training_period: int,
    trading_horizon: int,
    starting_portfolio_val: float,
    strategy,
):
    total_df = pd.concat([training_df, test_df])
    strategy_returns = pd.Series(0, index=test_df.index, dtype=float)
    # Starting point with weights determined from training set
    weights = strategy(training_df.iloc[-training_period:, :], returns_data=True, frequency=252)

    current_portfolio_val = starting_portfolio_val
    # Weights modified by training window
    for i in range(0, len(test_df), trading_horizon):
        print(f"Rebalancing for {test_df.index[i]}")
        transaction_costs = 0
        if i != 0:
            training_instance = total_df.loc[test_df.index[i] - timedelta(days=training_period) : test_df.index[i], :]
            try:
                new_weights = strategy(
                    training_instance, returns_data=True, frequency=252, prev_weight=list(weights.values())
                )
                # Calculate transaction cost
                for ticker, prev_w in weights.items():
                    new_w = new_weights[ticker]
                    if abs(new_w - prev_w) != 0:
                        # IBKR USD Dominated stock commision for asia pacific (SG) accounts
                        # USD Denominated Tier 1 - 0.08% of Trade Value with minimum per order of USD 2.00
                        # https://www.interactivebrokers.com/en/pricing/commissions-stocks-asia-pacific.php?re=apac
                        transaction_costs += min(
                            abs(new_w - prev_w) * current_portfolio_val * 0.0008,
                            2,
                        )
                weights = new_weights
            except Exception as e:
                print(f"Not solvable, using previous weights. Error: {e}")
                pass

        # Calculate results of current trading horizon
        test_returns_instance = test_df.iloc[i : i + trading_horizon, :]
        for ticker, weight in weights.items():
            strategy_returns.iloc[i : i + trading_horizon] += test_returns_instance[ticker] * weight

        # Subtract transaction cost from returns on rebalancing day based on percentage changes
        strategy_returns.iloc[i] -= transaction_costs / current_portfolio_val
        print(f"Total transaction costs {transaction_costs}, in percentage {transaction_costs/current_portfolio_val}")

        # Update current_portfolio_val
        current_portfolio_val *= (1 + strategy_returns.iloc[i : i + trading_horizon]).prod()
        print(f"Current portfolio val {current_portfolio_val}")
    return strategy_returns
