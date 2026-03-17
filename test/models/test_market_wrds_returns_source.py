from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from mic_data.market.prices_daily import latest_security_prices
from mic_data.market.wrds_returns_source import WrdsConnection, WrdsCrspDailyReturnSource


class _FakeConn(WrdsConnection):
    def raw_sql(self, query: str, date_cols: list[str] | None = None) -> pd.DataFrame:
        _ = query
        _ = date_cols
        return pd.DataFrame(
            {
                "trade_date": ["2026-03-01", "2026-03-01", "2026-03-02"],
                "ticker": ["AAPL", "AAPL", "AAPL"],
                "permno": [14593, 14593, 14593],
                "ret": [0.01, 0.01, -0.02],
                "prc": [100.0, 100.0, 98.0],
                "vol": [1000.0, 1000.0, 1200.0],
                "shrout": [10000.0, 10000.0, 10000.0],
                "namedt": ["2020-01-01", "2020-01-01", "2020-01-01"],
                "nameenddt": ["2030-12-31", "2030-12-31", "2030-12-31"],
            }
        )

    def close(self) -> None:
        return None


class TestWrdsReturnsSource(unittest.TestCase):
    def test_build_query_targets_crsp_v2_tables(self) -> None:
        source = WrdsCrspDailyReturnSource(username="test")
        query = source._build_query(
            tickers=["AMD", "^GSPC"],
            start_date="2025-01-01",
            end_date="2025-12-31",
        )

        self.assertIn("crsp.stocknames_v2", query)
        self.assertIn("crsp.dsf_v2", query)
        self.assertIn("UPPER(TRIM(ticker)) AS match_ticker", query)
        self.assertIn("('GSPC')", query)

    def test_load_security_returns_normalizes_and_deduplicates(self) -> None:
        source = WrdsCrspDailyReturnSource(
            username="test",
            connection_factory=lambda wrds_username=None: _FakeConn(),
        )

        universe = pd.DataFrame(
            {
                "as_of_date": ["2026-03-03"],
                "ticker": ["AAPL"],
                "is_holding": [True],
                "is_watchlist": [False],
                "shares": [10.0],
                "name": ["Apple"],
                "sector": ["Tech"],
            }
        )

        out = source.load_security_returns(
            universe=universe,
            start_date="2026-03-01",
            end_date="2026-03-03",
        )

        self.assertEqual(len(out), 2)
        self.assertEqual(out["source"].iloc[0], "wrds_crsp")
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(out["trade_date"]))

    def test_latest_security_prices_rejects_stale_dataset(self) -> None:
        security_returns = pd.DataFrame(
            {
                "trade_date": ["2026-03-10", "2026-03-10"],
                "ticker": ["AAPL", "MSFT"],
                "permno": [14593, 10107],
                "ret": [0.01, 0.02],
                "prc": [100.0, 200.0],
                "vol": [1000.0, 2000.0],
                "shrout": [10000.0, 15000.0],
                "source": ["wrds_crsp", "wrds_crsp"],
                "load_ts_utc": [
                    pd.Timestamp("2026-03-10T23:00:00Z"),
                    pd.Timestamp("2026-03-10T23:00:00Z"),
                ],
            }
        )

        with self.assertRaises(RuntimeError):
            latest_security_prices(
                security_returns,
                as_of_date=pd.Timestamp("2026-03-17"),
                max_business_day_lag=3,
            )


if __name__ == "__main__":
    unittest.main()
