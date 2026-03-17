from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from mic_data.market.prices_daily import latest_security_prices, load_security_returns_dataset


@dataclass(frozen=True)
class HoldingsLatestPaths:
    positions_latest_csv: Path = Path("data/processed/positions_latest.csv")
    holdings_latest_csv: Path = Path("data/processed/holdings_latest.csv")
    security_returns_path: Path = Path("data/processed/returns/security_returns_daily.parquet")
    max_business_day_lag: int = 3


def build_holdings_latest(
    paths: HoldingsLatestPaths = HoldingsLatestPaths(),
) -> pd.DataFrame:
    pos = pd.read_csv(paths.positions_latest_csv)

    # Expect at least: ticker, shares (and maybe as_of, name, sector)
    pos["ticker"] = pos["ticker"].astype(str).str.strip().str.upper()
    pos["shares"] = pd.to_numeric(pos["shares"], errors="raise")

    tickers = sorted(pos["ticker"].unique().tolist())

    security_returns = load_security_returns_dataset(paths.security_returns_path)
    latest_snapshot = latest_security_prices(
        security_returns,
        tickers=tickers,
        as_of_date=date.today(),
        max_business_day_lag=paths.max_business_day_lag,
    )

    out = pos[["ticker", "shares"]].copy()
    out["price"] = out["ticker"].map(latest_snapshot.prices.to_dict())
    if out["price"].isna().any():
        missing = out.loc[out["price"].isna(), "ticker"].tolist()
        raise ValueError(
            "Persisted security returns are incomplete for the latest trade date "
            f"{latest_snapshot.trade_date.date()}: missing prices for {missing}"
        )

    out["value"] = out["shares"] * out["price"]
    total = float(out["value"].sum())
    if total <= 0:
        raise ValueError(
            "Total portfolio value is non-positive; cannot compute weights."
        )

    out["weight"] = out["value"] / total
    out.insert(0, "as_of", latest_snapshot.trade_date.date().isoformat())

    paths.holdings_latest_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(paths.holdings_latest_csv, index=False)

    return out


if __name__ == "__main__":
    df = build_holdings_latest()
    print(df.sort_values("weight", ascending=False).head(10))
