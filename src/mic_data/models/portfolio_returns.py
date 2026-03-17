from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from mic_data.contracts.daily_returns_contracts import validate_portfolio_returns_daily
from mic_data.market.prices_daily import expected_latest_trade_date
from mic_data.models.constants import ModelFrequency
from mic_data.models.interfaces import PortfolioReturnSource


@dataclass(frozen=True)
class PersistedPortfolioReturnSource(PortfolioReturnSource):
    """Build monthly portfolio returns from persisted daily pipeline outputs."""

    portfolio_returns_path: Path = Path("data/processed/returns/portfolio_returns_daily.parquet")
    max_business_day_lag: int = 3

    def load_portfolio_returns(
        self,
        start_date: str,
        end_date: str,
        frequency: ModelFrequency = "M",
    ) -> pd.Series:
        if frequency != "M":
            raise ValueError("PersistedPortfolioReturnSource only supports monthly frequency 'M'.")

        daily = self._load_daily_returns()
        latest_trade_date = pd.Timestamp(daily["trade_date"].max()).normalize()
        stale_floor = expected_latest_trade_date(max_business_day_lag=self.max_business_day_lag)
        if latest_trade_date < stale_floor:
            raise RuntimeError(
                "Portfolio returns dataset is stale. "
                f"Latest trade_date={latest_trade_date.date()} accepted_floor={stale_floor.date()}."
            )

        start_ts = pd.Timestamp(start_date).normalize()
        end_ts = pd.Timestamp(end_date).normalize()
        if end_ts < start_ts:
            raise ValueError("end_date must be greater than or equal to start_date.")

        filtered = daily[
            (daily["trade_date"] >= start_ts) & (daily["trade_date"] <= end_ts)
        ].copy()
        if filtered.empty:
            raise ValueError(
                "Portfolio returns dataset has no observations in the requested window "
                f"[{start_date}, {end_date}]."
            )

        monthly = (
            (1.0 + filtered.set_index("trade_date")["portfolio_ret"])
            .resample("ME")
            .prod()
            .sub(1.0)
        )
        monthly.name = "portfolio_return"
        monthly = monthly.dropna()

        expected_months = pd.period_range(start_ts.to_period("M"), end_ts.to_period("M"), freq="M")
        observed_months = monthly.index.to_period("M")
        missing_months = [str(period) for period in expected_months if period not in observed_months]
        if missing_months:
            raise ValueError(
                "Portfolio returns dataset is incomplete for the requested monthly window. "
                f"Missing month(s): {missing_months}"
            )

        return monthly

    def _load_daily_returns(self) -> pd.DataFrame:
        path = self.portfolio_returns_path
        if not path.exists():
            raise FileNotFoundError(
                f"Portfolio returns file not found: {path}. "
                "Run 'python -m mic_data.market.build_portfolio_returns' first."
            )

        if path.suffix == ".parquet":
            frame = pd.read_parquet(path)
        elif path.suffix == ".csv":
            frame = pd.read_csv(path)
        else:
            raise ValueError(
                f"Unsupported portfolio returns file type for {path}. Expected .parquet or .csv."
            )

        return validate_portfolio_returns_daily(frame)
