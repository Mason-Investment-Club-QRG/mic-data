import pandas as pd
import numpy as np
import yfinance as yf
import statsmodels.api as sm
from datetime import datetime
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
FF_PATH = PROJECT_ROOT / "data" / "raw" / "factors" / "F-F_Research_Data_Factors.csv"

# # Defining static portfolio weights for MIC portfolio
# weights = {
#     "AMD": 0.0956,
#     "AXP": 0.1060,
#     "COST": 0.1314,
#     "CPNG": 0.0291,
#     "DUK": 0.0480,
#     "EHC": 0.0519,
#     "GE": 0.0632,
#     "GEHC": 0.0582,
#     "PM": 0.0720,
#     "QCOM": 0.0450,
#     "SPGI": 0.0576,
#     "TMUS": 0.0712,
#     "UNH": 0.0363,
#     "WCN": 0.0832,
#     "XYL": 0.0513,
# }

holdings = pd.read_csv("data/processed/holdings_latest.csv")
weights = holdings.set_index("ticker")["weight"]
tickers = weights.index.tolist()

assert abs(weights.sum() - 1) < 1e-6, "Weights must sum to 1"


# Download historical price data for the past 5 years
end_date = datetime.today()
start_date = end_date - pd.DateOffset(years=5)


raw = yf.download(
    tickers,
    start=start_date,
    end=end_date,
    interval="1mo",
    auto_adjust=True,
    progress=False,
)
if raw is None:
    raise RuntimeError("yfinance.download returned None")
prices = raw["Close"]

# Drop empty rows
prices = prices.dropna(how="all")

# Calculate monthly returns
asset_returns = prices.pct_change().dropna()

# Calculate static portfolio returns
portfolio_returns: pd.Series[float] = asset_returns.mul(weights, axis=1).sum(axis=1)  # type: ignore
portfolio_returns.name = "Portfolio_Return"

# Load and clean up Fama-French three-factor data
if not FF_PATH.exists():
    raise FileNotFoundError(f"Fama-French file not found at {FF_PATH}")

ff_raw = pd.read_csv(FF_PATH, skiprows=3)

ff_filtered = ff_raw[ff_raw.iloc[:, 0].astype(str).str.match(r"^\d{6}$")]


ff_filtered.rename(columns={ff_filtered.columns[0]: "Date"}, inplace=True)
ff_filtered["Date"] = pd.to_datetime(ff_filtered["Date"], format="%Y%m")
ff_filtered.set_index("Date", inplace=True)

ff_filtered = ff_filtered.apply(pd.to_numeric, errors="coerce").dropna()

ff = ff_filtered / 100
print(ff.head())
print(ff.tail())

# Merge portfolio returns with Fama-French data
data = pd.concat([portfolio_returns, ff], axis=1, join="inner")
data["Excess_Return"] = data["Portfolio_Return"] - data["RF"]
regression_data = data[["Excess_Return", "Mkt-RF", "SMB", "HML"]]

# Perform the Fama-French three-factor regression
X = sm.add_constant(regression_data[["Mkt-RF", "SMB", "HML"]])
y = regression_data["Excess_Return"]

model = sm.OLS(y, X).fit()
print(model.summary())
