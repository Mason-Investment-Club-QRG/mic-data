from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

import pandas as pd
import yfinance as yf

from mic_data.models.interfaces import PortfolioReturnSource


@dataclass(frozen=True)
class YFinancePortfolioReturnSource(PortfolioReturnSource):
    """Build portfolio returns from holdings weights and yfinance prices.

    Inputs:
      - holdings_path: CSV with at least ticker and weight columns.
      - yf_download: Optional injectable download function for tests.

    Returns:
      - pd.Series named portfolio_return indexed by month-end date.

    Raises:
      - FileNotFoundError if holdings CSV is missing.
      - ValueError for malformed holdings/weights or unsupported frequency.
      - RuntimeError when yfinance returns no price data.

    Notes on units:
      - Output series values are decimal returns (0.01 = 1%).
    """

    holdings_path: Path = Path("data/processed/holdings_latest.csv")
    yf_download: Callable[..., Any] | None = None

    def load_portfolio_returns(
        self,
        start_date: str,
        end_date: str,
        frequency: Literal["M"] = "M",
    ) -> pd.Series:
        """Load monthly portfolio returns for FF3 regression.

        Inputs:
          - start_date: Inclusive start date in YYYY-MM-DD format.
          - end_date: Inclusive end date in YYYY-MM-DD format.
          - frequency: Only "M" is supported.

        Returns:
          - pd.Series monthly portfolio returns (decimal) named portfolio_return.

        Raises:
          - FileNotFoundError when holdings file is missing.
          - ValueError for bad holdings schema/weights.
          - RuntimeError when market prices cannot be pulled.

        Notes on units:
          - Returns are decimal values.
        """
        if frequency != "M":
            raise ValueError("YFinancePortfolioReturnSource only supports monthly frequency 'M'.")

        weights = self._load_weights()
        tickers = weights.index.tolist()

        downloader = self.yf_download or yf.download
        raw = downloader(
            tickers,
            start=pd.Timestamp(start_date).strftime("%Y-%m-%d"),
            end=pd.Timestamp(end_date).strftime("%Y-%m-%d"),
            interval="1mo",
            auto_adjust=True,
            progress=False,
        )

        if raw is None or getattr(raw, "empty", True):
            raise RuntimeError("yfinance returned no data for portfolio return calculation.")

        close = raw["Close"]
        if isinstance(close, pd.Series):
            close = close.to_frame(name=tickers[0])

        close.index = pd.to_datetime(close.index)
        # Normalize to month-end labels so all sources align on the same calendar index.
        close.index = close.index.to_period("M").to_timestamp("M")
        close = close.groupby(level=0).last().sort_index()

        asset_returns = close.pct_change().dropna(how="all")
        asset_returns = asset_returns.reindex(columns=tickers)

        # Existing behavior: weighted sum across available ticker returns each month.
        portfolio_returns = asset_returns.mul(weights, axis=1).sum(axis=1, min_count=1)
        portfolio_returns.name = "portfolio_return"

        start_period = pd.Timestamp(start_date).to_period("M")
        end_period = pd.Timestamp(end_date).to_period("M")
        mask = (portfolio_returns.index.to_period("M") >= start_period) & (
            portfolio_returns.index.to_period("M") <= end_period
        )
        portfolio_returns = portfolio_returns.loc[mask].dropna()

        if portfolio_returns.empty:
            raise ValueError(
                "Portfolio return series is empty after alignment/filtering. "
                "Check holdings tickers and date range."
            )

        return portfolio_returns

    def _load_weights(self) -> pd.Series:
        if not self.holdings_path.exists():
            raise FileNotFoundError(f"Holdings file not found: {self.holdings_path}")

        holdings = pd.read_csv(self.holdings_path)
        required = {"ticker", "weight"}
        missing = required - set(holdings.columns)
        if missing:
            raise ValueError(
                f"Holdings file missing required columns: {sorted(missing)}. "
                f"Available columns: {list(holdings.columns)}"
            )

        holdings = holdings.copy()
        holdings["ticker"] = holdings["ticker"].astype(str).str.strip().str.upper()
        holdings["weight"] = pd.to_numeric(holdings["weight"], errors="raise")

        weights = holdings.set_index("ticker")["weight"]
        total = float(weights.sum())
        if total <= 0:
            raise ValueError("Holdings weights sum to a non-positive value.")

        if abs(total - 1.0) > 1e-3:
            raise ValueError(
                f"Holdings weights must sum to ~1.0 for FF3 regression, got {total:.6f}."
            )

        return weights
