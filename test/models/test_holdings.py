from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from mic_data.portfolio.holdings import HoldingsLatestPaths, build_holdings_latest


class TestBuildHoldingsLatest(unittest.TestCase):
    @staticmethod
    def _positions_frame() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "ticker": ["AAPL", "MSFT"],
                "shares": [10, 5],
            }
        )

    @staticmethod
    def _security_returns_frame() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "trade_date": ["2026-03-14", "2026-03-14", "2026-03-17", "2026-03-17"],
                "ticker": ["AAPL", "MSFT", "AAPL", "MSFT"],
                "permno": [14593, 10107, 14593, 10107],
                "ret": [0.01, 0.02, -0.01, 0.03],
                "prc": [200.0, 300.0, 198.0, 309.0],
                "vol": [1000.0, 2000.0, 1100.0, 2100.0],
                "shrout": [10000.0, 15000.0, 10000.0, 15000.0],
                "source": ["wrds_crsp", "wrds_crsp", "wrds_crsp", "wrds_crsp"],
                "load_ts_utc": [
                    "2026-03-14T22:00:00Z",
                    "2026-03-14T22:00:00Z",
                    "2026-03-17T22:00:00Z",
                    "2026-03-17T22:00:00Z",
                ],
            }
        )

    def test_build_holdings_latest_uses_persisted_security_returns(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            positions_path = root / "positions_latest.csv"
            holdings_path = root / "holdings_latest.csv"
            security_returns_path = root / "security_returns_daily.csv"

            self._positions_frame().to_csv(positions_path, index=False)
            self._security_returns_frame().to_csv(security_returns_path, index=False)

            out = build_holdings_latest(
                HoldingsLatestPaths(
                    positions_latest_csv=positions_path,
                    holdings_latest_csv=holdings_path,
                    security_returns_path=security_returns_path,
                    max_business_day_lag=10000,
                )
            )

        self.assertEqual(out["as_of"].iloc[0], "2026-03-17")
        self.assertEqual(out.set_index("ticker").loc["AAPL", "price"], 198.0)
        self.assertEqual(out.set_index("ticker").loc["MSFT", "price"], 309.0)
        self.assertAlmostEqual(float(out["weight"].sum()), 1.0, places=10)

    def test_build_holdings_latest_rejects_missing_latest_prices(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            positions_path = root / "positions_latest.csv"
            holdings_path = root / "holdings_latest.csv"
            security_returns_path = root / "security_returns_daily.csv"

            self._positions_frame().to_csv(positions_path, index=False)
            frame = self._security_returns_frame()
            frame = frame[~((frame["trade_date"] == "2026-03-17") & (frame["ticker"] == "MSFT"))].copy()
            frame.to_csv(security_returns_path, index=False)

            with self.assertRaises(ValueError):
                build_holdings_latest(
                    HoldingsLatestPaths(
                        positions_latest_csv=positions_path,
                        holdings_latest_csv=holdings_path,
                        security_returns_path=security_returns_path,
                        max_business_day_lag=10000,
                    )
                )


if __name__ == "__main__":
    unittest.main()
