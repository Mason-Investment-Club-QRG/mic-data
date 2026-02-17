from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd
import yfinance as yf


@dataclass(frozen=True)
class HoldingsLatestPaths:
    positions_latest_csv: Path = Path("data/processed/positions_latest.csv")
    holdings_latest_csv: Path = Path("data/processed/holdings_latest.csv")


def build_holdings_latest(
    paths: HoldingsLatestPaths = HoldingsLatestPaths(),
) -> pd.DataFrame:
    pos = pd.read_csv(paths.positions_latest_csv)

    # Expect at least: ticker, shares (and maybe as_of, name, sector)
    pos["ticker"] = pos["ticker"].astype(str).str.strip().str.upper()
    pos["shares"] = pd.to_numeric(pos["shares"], errors="raise")

    tickers = sorted(pos["ticker"].unique().tolist())

    # Fast, robust: pull latest adj close via 5d history and take last
    px = yf.download(
        tickers, period="5d", interval="1d", auto_adjust=True, progress=False
    )
    if px is None or px.empty:
        raise RuntimeError("yfinance returned no data for latest prices.")

    # Handle both single-ticker and multi-ticker shapes
    close = px["Close"]
    if isinstance(close, pd.Series):
        latest_prices = pd.Series({tickers[0]: float(close.dropna().iloc[-1])})
    else:
        latest_prices = close.dropna(how="all").iloc[-1].astype(float)

    out = pos[["ticker", "shares"]].copy()
    out["price"] = out["ticker"].map(latest_prices.to_dict())
    if out["price"].isna().any():
        missing = out.loc[out["price"].isna(), "ticker"].tolist()
        raise ValueError(f"Missing latest prices for: {missing}")

    out["value"] = out["shares"] * out["price"]
    total = float(out["value"].sum())
    if total <= 0:
        raise ValueError(
            "Total portfolio value is non-positive; cannot compute weights."
        )

    out["weight"] = out["value"] / total
    out.insert(0, "as_of", date.today().isoformat())

    paths.holdings_latest_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(paths.holdings_latest_csv, index=False)

    return out


if __name__ == "__main__":
    df = build_holdings_latest()
    print(df.sort_values("weight", ascending=False).head(10))
