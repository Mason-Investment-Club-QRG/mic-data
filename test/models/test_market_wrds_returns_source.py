from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

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


if __name__ == "__main__":
    unittest.main()
