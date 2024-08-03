from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import HRPOpt, risk_models
from pypfopt import expected_returns
from pypfopt import objective_functions
import pandas as pd

import config


def equal_weight_strategy(training_data: pd.DataFrame, returns_data: bool, frequency: int, prev_weight=None):
    return {ticker: 1 / len(training_data.columns) for ticker in list(training_data.columns)}


def historical_mvo_strategy(training_data: pd.DataFrame, returns_data: bool, frequency: int, prev_weight=None):
    # Calculate expected returns and sample covariance
    mu = expected_returns.ema_historical_return(training_data, returns_data=returns_data, frequency=frequency, span=50)
    S = risk_models.CovarianceShrinkage(training_data, returns_data=returns_data, frequency=frequency).ledoit_wolf()

    # Optimize for maximal Sharpe ratio
    ef = EfficientFrontier(mu, S)
    ef.add_objective(objective_functions.L2_reg, gamma=1)

    if prev_weight:
        ef.add_objective(objective_functions.transaction_cost, w_prev=prev_weight, k=0.0008)

    ef.add_sector_constraints(config.sector_mapper, config.sector_lower, config.sector_upper)
    ef.max_sharpe(config.estimated_daily_risk_free_rate)
    return ef.clean_weights()


def historical_mvo_min_vol_strategy(training_data: pd.DataFrame, returns_data: bool, frequency: int, prev_weight=None):
    # Calculate expected returns and sample covariance
    mu = expected_returns.ema_historical_return(training_data, returns_data=returns_data, frequency=frequency, span=50)
    S = risk_models.CovarianceShrinkage(training_data, returns_data=returns_data, frequency=frequency).ledoit_wolf()

    # Optimize for maximal Sharpe ratio
    ef = EfficientFrontier(mu, S)
    ef.add_objective(objective_functions.L2_reg, gamma=1)

    if prev_weight:
        ef.add_objective(objective_functions.transaction_cost, w_prev=prev_weight, k=0.0008)

    ef.add_sector_constraints(config.sector_mapper, config.sector_lower, config.sector_upper)
    ef.min_volatility()
    return ef.clean_weights()


def historical_mvo_expected_ret_strategy(
    training_data: pd.DataFrame, returns_data: bool, frequency: int, prev_weight=None
):
    mu = expected_returns.ema_historical_return(training_data, returns_data=returns_data, frequency=frequency, span=50)
    S = risk_models.CovarianceShrinkage(training_data, returns_data=returns_data, frequency=frequency).ledoit_wolf()

    ef = EfficientFrontier(mu, S)
    ef.add_objective(objective_functions.L2_reg, gamma=1)

    if prev_weight:
        ef.add_objective(objective_functions.transaction_cost, w_prev=prev_weight, k=0.0008)

    ef.add_sector_constraints(config.sector_mapper, config.sector_lower, config.sector_upper)
    ef.efficient_return(target_return=config.target_returns)
    return ef.clean_weights()


def historical_mvo_capm_strategy(training_data: pd.DataFrame, returns_data: bool, frequency: int, prev_weight=None):
    snp_returns = training_data["SPY"]
    training_data.drop(columns=["SPY"], inplace=True)
    mu = expected_returns.capm_return(
        training_data, market_prices=snp_returns, returns_data=returns_data, frequency=frequency
    )
    S = risk_models.CovarianceShrinkage(training_data, returns_data=returns_data, frequency=frequency).ledoit_wolf()

    ef = EfficientFrontier(mu, S)
    ef.add_objective(objective_functions.L2_reg, gamma=1)

    if prev_weight:
        ef.add_objective(objective_functions.transaction_cost, w_prev=prev_weight, k=0.0008)

    ef.add_sector_constraints(config.sector_mapper, config.sector_lower, config.sector_upper)
    ef.max_sharpe(config.estimated_daily_risk_free_rate)
    return ef.clean_weights()


def hrpopt_strategy(training_data: pd.DataFrame, returns_data: bool, frequency: int, prev_weight=None):
    if not returns_data:
        print("Not supported yet")
        raise Exception
    hrp = HRPOpt(training_data)
    hrp.optimize()
    cleaned_baseline_weights = hrp.clean_weights()
    return cleaned_baseline_weights
