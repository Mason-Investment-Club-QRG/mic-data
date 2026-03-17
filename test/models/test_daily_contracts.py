from __future__ import annotations

import sys
import unittest
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from mic_data.contracts.daily_returns_contracts import (
    QaRow,
    qa_row_to_frame,
    validate_daily_qa,
    validate_portfolio_returns_daily,
    validate_security_returns_daily,
    validate_universe_daily,
)


class TestDailyContracts(unittest.TestCase):
    def test_validate_universe_daily_normalizes_and_enforces_key(self) -> None:
        raw = pd.DataFrame(
            {
                "as_of_date": ["2026-03-01", "2026-03-01"],
                "ticker": ["aapl", "msft"],
                "is_holding": [True, False],
                "is_watchlist": [False, True],
                "shares": [10, None],
                "name": ["Apple", "Microsoft"],
                "sector": ["Tech", "Tech"],
            }
        )

        out = validate_universe_daily(raw)
        self.assertEqual(list(out["ticker"]), ["AAPL", "MSFT"])

        dup = pd.concat([out, out.iloc[[0]]], ignore_index=True)
        with self.assertRaises(ValueError):
            validate_universe_daily(dup)

    def test_validate_security_returns_daily_rejects_duplicate_key(self) -> None:
        raw = pd.DataFrame(
            {
                "trade_date": ["2026-03-01", "2026-03-01"],
                "ticker": ["AAPL", "AAPL"],
                "permno": [14593, 14593],
                "ret": [0.01, 0.01],
                "prc": [100, 100],
                "vol": [10, 10],
                "shrout": [1000, 1000],
                "source": ["wrds_crsp", "wrds_crsp"],
                "load_ts_utc": ["2026-03-03T00:00:00Z", "2026-03-03T00:00:00Z"],
            }
        )
        with self.assertRaises(ValueError):
            validate_security_returns_daily(raw)

    def test_validate_portfolio_returns_daily_accepts_decimal_series(self) -> None:
        raw = pd.DataFrame(
            {
                "trade_date": ["2026-03-01", "2026-03-02"],
                "portfolio_ret": [0.001, -0.002],
                "n_constituents": [10, 9],
                "gross_exposure": [1.0, 1.0],
                "method": ["holdings_weighted_sum", "holdings_weighted_sum"],
                "load_ts_utc": ["2026-03-03T00:00:00Z", "2026-03-03T00:00:00Z"],
            }
        )
        out = validate_portfolio_returns_daily(raw)
        self.assertEqual(len(out), 2)

    def test_validate_daily_qa_and_row_builder(self) -> None:
        frame = qa_row_to_frame(
            QaRow(
                run_date=date(2026, 3, 3),
                stage="pull_wrds_returns",
                rows_universe=10,
                rows_returns=9,
                missing_permno=1,
                missing_ret=0,
                status="ok",
                error_message=None,
            )
        )
        out = validate_daily_qa(frame)
        self.assertEqual(int(out.iloc[0]["rows_universe"]), 10)


if __name__ == "__main__":
    unittest.main()
