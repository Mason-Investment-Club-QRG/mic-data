from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from mic_data.models.portfolio_returns import PersistedPortfolioReturnSource


class TestPersistedPortfolioReturnSource(unittest.TestCase):
    @staticmethod
    def _daily_frame() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "trade_date": [
                    "2026-01-29",
                    "2026-01-30",
                    "2026-02-26",
                    "2026-02-27",
                ],
                "portfolio_ret": [0.01, 0.02, -0.01, 0.03],
                "n_constituents": [2, 2, 2, 2],
                "gross_exposure": [1.0, 1.0, 1.0, 1.0],
                "method": [
                    "holdings_weighted_sum",
                    "holdings_weighted_sum",
                    "holdings_weighted_sum",
                    "holdings_weighted_sum",
                ],
                "load_ts_utc": [
                    "2026-01-30T22:00:00Z",
                    "2026-01-30T22:00:00Z",
                    "2026-02-27T22:00:00Z",
                    "2026-02-27T22:00:00Z",
                ],
            }
        )

    def test_load_portfolio_returns_compounds_monthly_from_persisted_daily_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "portfolio_returns_daily.csv"
            self._daily_frame().to_csv(path, index=False)
            source = PersistedPortfolioReturnSource(portfolio_returns_path=path)

            with patch(
                "mic_data.models.portfolio_returns.expected_latest_trade_date",
                return_value=pd.Timestamp("2026-02-20"),
            ):
                out = source.load_portfolio_returns("2026-01-01", "2026-02-28")

        self.assertEqual(len(out), 2)
        self.assertAlmostEqual(float(out.iloc[0]), (1.01 * 1.02) - 1.0, places=10)
        self.assertAlmostEqual(float(out.iloc[1]), (0.99 * 1.03) - 1.0, places=10)

    def test_load_portfolio_returns_rejects_missing_months(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "portfolio_returns_daily.csv"
            frame = self._daily_frame()[self._daily_frame()["trade_date"] != "2026-02-26"].copy()
            frame = frame[frame["trade_date"] != "2026-02-27"].copy()
            frame.to_csv(path, index=False)
            source = PersistedPortfolioReturnSource(portfolio_returns_path=path)

            with patch(
                "mic_data.models.portfolio_returns.expected_latest_trade_date",
                return_value=pd.Timestamp("2026-01-20"),
            ):
                with self.assertRaises(ValueError):
                    source.load_portfolio_returns("2026-01-01", "2026-02-28")

    def test_load_portfolio_returns_rejects_stale_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "portfolio_returns_daily.csv"
            self._daily_frame().to_csv(path, index=False)
            source = PersistedPortfolioReturnSource(portfolio_returns_path=path)

            with patch(
                "mic_data.models.portfolio_returns.expected_latest_trade_date",
                return_value=pd.Timestamp("2026-03-05"),
            ):
                with self.assertRaises(RuntimeError):
                    source.load_portfolio_returns("2026-01-01", "2026-02-28")


if __name__ == "__main__":
    unittest.main()
