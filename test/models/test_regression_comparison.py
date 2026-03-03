from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from mic_data.models.comparison import (
    ComparisonThresholds,
    compare_factor_inputs,
    compare_regression_results,
    evaluate_similarity,
)
from mic_data.models.regression import run_ff3_regression


class TestRegressionAndComparison(unittest.TestCase):
    def test_run_ff3_regression_recovers_coefficients(self) -> None:
        idx = pd.date_range("2020-01-31", periods=36, freq="ME")
        rng = np.random.default_rng(7)

        factors = pd.DataFrame(
            {
                "mkt_rf": rng.normal(0.005, 0.02, len(idx)),
                "smb": rng.normal(0.001, 0.01, len(idx)),
                "hml": rng.normal(0.0005, 0.01, len(idx)),
                "rf": rng.normal(0.002, 0.0005, len(idx)),
            },
            index=idx,
        )

        alpha = 0.0015
        beta_mkt = 1.1
        beta_smb = 0.25
        beta_hml = -0.35

        excess = (
            alpha
            + beta_mkt * factors["mkt_rf"]
            + beta_smb * factors["smb"]
            + beta_hml * factors["hml"]
        )
        portfolio_returns = excess + factors["rf"]

        result = run_ff3_regression(portfolio_returns, factors)

        self.assertAlmostEqual(result.alpha, alpha, places=8)
        self.assertAlmostEqual(result.beta_mkt, beta_mkt, places=8)
        self.assertAlmostEqual(result.beta_smb, beta_smb, places=8)
        self.assertAlmostEqual(result.beta_hml, beta_hml, places=8)
        self.assertGreater(result.r2, 0.999999)

    def test_similarity_gate_balanced_pass(self) -> None:
        idx = pd.date_range("2021-01-31", periods=24, freq="ME")
        rng = np.random.default_rng(11)

        wrds = pd.DataFrame(
            {
                "mkt_rf": rng.normal(0.004, 0.015, len(idx)),
                "smb": rng.normal(0.001, 0.01, len(idx)),
                "hml": rng.normal(0.001, 0.01, len(idx)),
                "rf": rng.normal(0.002, 0.0003, len(idx)),
            },
            index=idx,
        )

        static = wrds + rng.normal(0.0, 0.00002, wrds.shape)
        static = pd.DataFrame(static, index=idx, columns=wrds.columns)

        portfolio_wrds = 0.001 + 0.9 * wrds["mkt_rf"] + 0.2 * wrds["smb"] - 0.1 * wrds["hml"] + wrds["rf"]
        portfolio_static = 0.001 + 0.9 * static["mkt_rf"] + 0.2 * static["smb"] - 0.1 * static["hml"] + static["rf"]

        wrds_result = run_ff3_regression(portfolio_wrds, wrds)
        static_result = run_ff3_regression(portfolio_static, static)

        factor_metrics = compare_factor_inputs(wrds, static)
        deltas = compare_regression_results(wrds_result, static_result)
        gate = evaluate_similarity(factor_metrics, deltas, ComparisonThresholds())

        self.assertTrue(gate["passed"])
        self.assertEqual(gate["reason_codes"], [])

    def test_regression_handles_nullable_float_inputs(self) -> None:
        idx = pd.date_range("2022-01-31", periods=12, freq="ME")
        factors = pd.DataFrame(
            {
                "mkt_rf": [0.01] * 12,
                "smb": [0.002] * 12,
                "hml": [0.001] * 12,
                "rf": [0.0005] * 12,
            },
            index=idx,
        ).astype("Float64")
        portfolio = pd.Series(
            [0.0115] * 12,
            index=idx,
            dtype="Float64",
            name="portfolio_return",
        )

        result = run_ff3_regression(portfolio, factors)

        self.assertEqual(result.n_obs, 12)
        self.assertIsInstance(result.alpha, float)


if __name__ == "__main__":
    unittest.main()
