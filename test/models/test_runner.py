from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from mic_data.models.constants import ModelFrequency
from mic_data.models.comparison import ComparisonThresholds
from mic_data.models.interfaces import FactorSource, PortfolioReturnSource
from mic_data.models.runner import FF3PipelineConfig, run_ff3_pipeline


class _StaticSource(FactorSource):
    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame

    def load_factors(
        self,
        start_date: str,
        end_date: str,
        frequency: ModelFrequency = "M",
    ) -> pd.DataFrame:
        _ = start_date
        _ = end_date
        _ = frequency
        return self.frame


class _FailingSource(FactorSource):
    def load_factors(
        self,
        start_date: str,
        end_date: str,
        frequency: ModelFrequency = "M",
    ) -> pd.DataFrame:
        _ = start_date
        _ = end_date
        _ = frequency
        raise RuntimeError("simulated WRDS outage")


class _PortfolioSource(PortfolioReturnSource):
    def __init__(self, series: pd.Series) -> None:
        self.series = series

    def load_portfolio_returns(
        self,
        start_date: str,
        end_date: str,
        frequency: ModelFrequency = "M",
    ) -> pd.Series:
        _ = start_date
        _ = end_date
        _ = frequency
        return self.series


class TestRunner(unittest.TestCase):
    @staticmethod
    def _mock_write_parquet(df: pd.DataFrame, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Keep a lightweight marker file so tests do not require pyarrow.
        path.write_text(df.to_csv(), encoding="utf-8")

    def test_runner_fallback_when_wrds_fails(self) -> None:
        idx = pd.date_range("2022-01-31", periods=12, freq="ME")
        static_factors = pd.DataFrame(
            {
                "mkt_rf": [0.01] * 12,
                "smb": [0.002] * 12,
                "hml": [0.001] * 12,
                "rf": [0.0005] * 12,
            },
            index=idx,
        )
        portfolio = pd.Series([0.012] * 12, index=idx, name="portfolio_return")

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = FF3PipelineConfig(
                start_date="2022-01-01",
                end_date="2022-12-31",
                processed_model_inputs_dir=root / "data/processed/model_inputs",
                validation_output_dir=root / "outputs/validation",
                logs_dir=root / "outputs/logs",
            )

            with patch("mic_data.models.runner._write_parquet", side_effect=self._mock_write_parquet):
                artifacts = run_ff3_pipeline(
                    config,
                    wrds_source=_FailingSource(),
                    static_source=_StaticSource(static_factors),
                    portfolio_source=_PortfolioSource(portfolio),
                )

            self.assertTrue(artifacts.used_fallback)
            self.assertFalse(artifacts.wrds_available)
            self.assertIsNone(artifacts.wrds_factors_path)
            self.assertTrue(artifacts.static_factors_path is not None)
            self.assertTrue(artifacts.factor_comparison_csv.exists())
            self.assertTrue(artifacts.regression_comparison_json.exists())
            self.assertTrue(artifacts.validation_summary_md.exists())
            self.assertTrue(artifacts.log_path.exists())

            payload = json.loads(artifacts.regression_comparison_json.read_text())
            self.assertTrue(payload["fallback_used"])
            self.assertFalse(payload["wrds_available"])

    def test_runner_comparison_passes_when_sources_are_close(self) -> None:
        idx = pd.date_range("2023-01-31", periods=18, freq="ME")
        wrds = pd.DataFrame(
            {
                "mkt_rf": [0.0100 + i * 0.0001 for i in range(18)],
                "smb": [0.0020 + i * 0.00005 for i in range(18)],
                "hml": [0.0010 + i * 0.00003 for i in range(18)],
                "rf": [0.0005 + i * 0.00001 for i in range(18)],
            },
            index=idx,
        )
        static = wrds.copy()
        static["mkt_rf"] = static["mkt_rf"] + 0.00002

        portfolio = 0.001 + 0.9 * wrds["mkt_rf"] + 0.2 * wrds["smb"] - 0.1 * wrds["hml"] + wrds["rf"]
        portfolio.name = "portfolio_return"

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = FF3PipelineConfig(
                start_date="2023-01-01",
                end_date="2024-06-30",
                processed_model_inputs_dir=root / "data/processed/model_inputs",
                validation_output_dir=root / "outputs/validation",
                logs_dir=root / "outputs/logs",
            )

            with patch("mic_data.models.runner._write_parquet", side_effect=self._mock_write_parquet):
                artifacts = run_ff3_pipeline(
                    config,
                    thresholds=ComparisonThresholds(),
                    wrds_source=_StaticSource(wrds),
                    static_source=_StaticSource(static),
                    portfolio_source=_PortfolioSource(portfolio),
                )

            payload = json.loads(artifacts.regression_comparison_json.read_text())
            self.assertTrue(payload["wrds_available"])
            self.assertIn("gate", payload)
            self.assertTrue(payload["gate"]["passed"])


if __name__ == "__main__":
    unittest.main()
