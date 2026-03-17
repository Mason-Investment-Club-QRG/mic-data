from __future__ import annotations

import sys
import tempfile
import unittest
import warnings
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from mic_data.market.prices_daily import (
    DailyReturnsConfig,
    DailyReturnsPaths,
    QaRow,
    upsert_daily_qa_row,
)


class TestPricesDailyHelpers(unittest.TestCase):
    def test_upsert_daily_qa_row_replaces_existing_row_without_future_warning(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = DailyReturnsConfig(
                start_date="2025-01-01",
                end_date="2025-12-31",
                google_sheets_config_path=root / "google_sheets.yaml",
                positions_config_path=root / "positions.yaml",
                paths=DailyReturnsPaths(
                    universe_latest_path=root / "universe.parquet",
                    security_returns_path=root / "security_returns.parquet",
                    security_returns_csv_path=root / "security_returns.csv",
                    portfolio_returns_path=root / "portfolio_returns.parquet",
                    portfolio_returns_csv_path=root / "portfolio_returns.csv",
                    daily_qa_path=root / "daily_qa.parquet",
                    manifests_dir=root / "manifests",
                    locks_dir=root / "locks",
                    logs_dir=root / "logs",
                ),
            )

            first_row = QaRow(
                run_date=date(2026, 3, 17),
                stage="pull_wrds_returns",
                rows_universe=10,
                rows_returns=100,
                missing_permno=0,
                missing_ret=0,
                status="ok",
                error_message=None,
            )
            second_row = QaRow(
                run_date=date(2026, 3, 17),
                stage="pull_wrds_returns",
                rows_universe=10,
                rows_returns=95,
                missing_permno=1,
                missing_ret=0,
                status="warn",
                error_message="updated",
            )

            with warnings.catch_warnings():
                warnings.simplefilter("error", FutureWarning)
                upsert_daily_qa_row(config=config, qa_row=first_row)
                out = upsert_daily_qa_row(config=config, qa_row=second_row)

        self.assertEqual(len(out), 1)
        self.assertEqual(int(out.iloc[0]["rows_returns"]), 95)
        self.assertEqual(str(out.iloc[0]["status"]), "warn")


if __name__ == "__main__":
    unittest.main()
