from __future__ import annotations

import sys
import unittest
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from mic_data.positions.universe_sync import merge_universe


class TestUniverseSync(unittest.TestCase):
    def test_merge_universe_combines_flags_and_preserves_shares(self) -> None:
        as_of = date(2026, 3, 3).isoformat()
        holdings = pd.DataFrame(
            {
                "as_of_date": [as_of],
                "ticker": ["AAPL"],
                "is_holding": [True],
                "is_watchlist": [False],
                "shares": [100.0],
                "name": ["Apple"],
                "sector": ["Tech"],
            }
        )
        watchlist = pd.DataFrame(
            {
                "as_of_date": [as_of, as_of],
                "ticker": ["AAPL", "MSFT"],
                "is_holding": [False, False],
                "is_watchlist": [True, True],
                "shares": [None, None],
                "name": ["Apple", "Microsoft"],
                "sector": ["Tech", "Tech"],
            }
        )

        out = merge_universe(holdings, watchlist)
        self.assertEqual(len(out), 2)

        aapl = out[out["ticker"] == "AAPL"].iloc[0]
        self.assertTrue(bool(aapl["is_holding"]))
        self.assertTrue(bool(aapl["is_watchlist"]))
        self.assertEqual(float(aapl["shares"]), 100.0)


if __name__ == "__main__":
    unittest.main()
