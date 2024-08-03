assets = ["QQQ", "VBR", "VWRA.L", "DBC", "VNQ", "BNDX", "SPHY"]

sector_mapper = {
    "QQQ": "equities",
    "VBR": "equities",
    "VWRA.L": "equities",
    "DBC": "commodities",
    "VNQ": "reits",
    "BNDX": "bonds",
    "SPHY": "bonds",
}

prev_weight = {
    "BNDX": 0.05,
    "DBC": 0.006,
    "QQQ": 0.23,
    "SPHY": 0.03,
    "VBR": 0.13,
    "VNQ": 0.02,
    "VWRA.L": 0.14,
}

sector_lower = {"equities": 0.6, "bonds": 0.05, "commodities": 0.05, "reits": 0.05}
sector_upper = {"equities": 0.9, "bonds": 0.5, "commodities": 0.5, "reits": 0.5}

current_portfolio_val = 27620
estimated_daily_risk_free_rate = 0.05 / 252
target_returns = 0.03

training_period = 180
trading_horizon = 120
