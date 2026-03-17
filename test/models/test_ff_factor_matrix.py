from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from mic_data.models.ff_factor_matrix import (
    PipelineReturnInputs,
    main,
    estimate_portfolio_ff3_loading,
    load_pipeline_return_inputs,
    run_ff3_factor_analysis,
)


class TestFFFactorMatrix(unittest.TestCase):
    @staticmethod
    def _build_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        idx = pd.date_range("2024-01-02", periods=140, freq="B")
        rng = np.random.default_rng(19)

        factors = pd.DataFrame(
            {
                "trade_date": idx,
                "mkt_rf": rng.normal(0.0004, 0.01, len(idx)),
                "smb": rng.normal(0.0001, 0.006, len(idx)),
                "hml": rng.normal(0.0002, 0.007, len(idx)),
                "rf": np.full(len(idx), 0.00005),
            }
        )

        aaa_excess = (
            0.0002
            + 1.20 * factors["mkt_rf"]
            + 0.35 * factors["smb"]
            - 0.25 * factors["hml"]
            + rng.normal(0.0, 0.001, len(idx))
        )
        bbb_excess = (
            -0.0001
            + 0.80 * factors["mkt_rf"]
            - 0.15 * factors["smb"]
            + 0.40 * factors["hml"]
            + rng.normal(0.0, 0.001, len(idx))
        )

        security_returns = pd.concat(
            [
                pd.DataFrame(
                    {
                        "trade_date": idx,
                        "ticker": "AAA",
                        "permno": 10001,
                        "ret": aaa_excess + factors["rf"],
                        "prc": 10.0,
                        "vol": 1000.0,
                        "shrout": 100.0,
                        "source": "wrds_crsp",
                        "load_ts_utc": pd.Timestamp("2026-03-03T00:00:00Z"),
                    }
                ),
                pd.DataFrame(
                    {
                        "trade_date": idx,
                        "ticker": "BBB",
                        "permno": 10002,
                        "ret": bbb_excess + factors["rf"],
                        "prc": 20.0,
                        "vol": 1500.0,
                        "shrout": 100.0,
                        "source": "wrds_crsp",
                        "load_ts_utc": pd.Timestamp("2026-03-03T00:00:00Z"),
                    }
                ),
            ],
            ignore_index=True,
        )

        portfolio_returns = pd.DataFrame(
            {
                "trade_date": idx,
                "portfolio_ret": 0.6 * (aaa_excess + factors["rf"]) + 0.4 * (bbb_excess + factors["rf"]),
                "n_constituents": 2,
                "gross_exposure": 1.0,
                "method": "holdings_weighted_sum",
                "load_ts_utc": pd.Timestamp("2026-03-03T00:00:00Z"),
            }
        )

        return security_returns, portfolio_returns, factors

    def test_run_ff3_factor_analysis_builds_reusable_outputs(self) -> None:
        security_returns, portfolio_returns, factors = self._build_inputs()

        result = run_ff3_factor_analysis(
            security_returns=security_returns,
            portfolio_returns=portfolio_returns,
            factors=factors,
            security_weights=pd.Series({"AAA": 0.6, "BBB": 0.4}),
            min_obs=100,
        )

        self.assertEqual(list(result.security_beta_matrix.columns), ["mkt_rf", "smb", "hml"])
        self.assertAlmostEqual(float(result.security_loadings.loc["AAA", "mkt_rf"]), 1.20, delta=0.08)
        self.assertAlmostEqual(float(result.security_loadings.loc["BBB", "hml"]), 0.40, delta=0.08)
        self.assertAlmostEqual(float(result.portfolio_return_exposure["mkt_rf"]), 1.04, delta=0.08)
        self.assertIsNotNone(result.portfolio_holdings_exposure)
        assert result.portfolio_holdings_exposure is not None
        self.assertAlmostEqual(float(result.portfolio_holdings_exposure["smb"]), 0.15, delta=0.08)
        self.assertEqual(result.security_factor_covariance.shape, (2, 2))
        self.assertIn("portfolio_exposure_comparison", result.tables)
        self.assertIn("limitations", result.metadata)
        self.assertGreater(float(result.portfolio_risk_summary["factor_share"]), 0.5)

    def test_load_pipeline_return_inputs_filters_dates_and_tickers(self) -> None:
        security_returns, portfolio_returns, factors = self._build_inputs()
        _ = factors

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            security_path = root / "security_returns_daily.csv"
            portfolio_path = root / "portfolio_returns_daily.csv"
            security_returns.to_csv(security_path, index=False)
            portfolio_returns.to_csv(portfolio_path, index=False)

            out = load_pipeline_return_inputs(
                security_returns_path=security_path,
                portfolio_returns_path=portfolio_path,
                start_date="2024-02-01",
                end_date="2024-03-15",
                tickers=["aaa"],
            )

        self.assertEqual(sorted(out.security_returns["ticker"].unique().tolist()), ["AAA"])
        self.assertGreaterEqual(out.security_returns["trade_date"].min(), pd.Timestamp("2024-02-01"))
        self.assertLessEqual(out.security_returns["trade_date"].max(), pd.Timestamp("2024-03-15"))
        self.assertGreaterEqual(out.portfolio_returns["trade_date"].min(), pd.Timestamp("2024-02-01"))
        self.assertLessEqual(out.portfolio_returns["trade_date"].max(), pd.Timestamp("2024-03-15"))

    def test_estimate_portfolio_ff3_loading_accepts_legacy_factor_columns(self) -> None:
        _, portfolio_returns, factors = self._build_inputs()
        legacy_factors = factors.rename(
            columns={"mkt_rf": "Mkt-RF", "rf": "RF", "trade_date": "date"}
        )

        result = estimate_portfolio_ff3_loading(
            portfolio_returns=portfolio_returns,
            factors=legacy_factors,
            min_obs=100,
        )

        self.assertIn("mkt_rf", result.index)
        self.assertIn("r2", result.index)

    def test_main_prints_json_summary(self) -> None:
        security_returns, portfolio_returns, factors = self._build_inputs()
        payload_buffer = io.StringIO()

        with patch(
            "mic_data.models.ff_factor_matrix.load_pipeline_return_inputs",
            return_value=PipelineReturnInputs(
                security_returns=security_returns,
                portfolio_returns=portfolio_returns,
            ),
        ), patch(
            "mic_data.models.ff_factor_matrix.load_ff3_factors_from_wrds",
            return_value=factors,
        ), redirect_stdout(payload_buffer):
            main(["--start-date", "2024-01-01", "--end-date", "2024-12-31", "--min-obs", "100"])

        payload = json.loads(payload_buffer.getvalue())
        self.assertIn("portfolio_return_exposure", payload)
        self.assertIn("metadata", payload)
        self.assertEqual(payload["metadata"]["modeled_security_count"], 2)


if __name__ == "__main__":
    unittest.main()
