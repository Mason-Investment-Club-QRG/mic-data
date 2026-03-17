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

from mic_data.analytics.dashboard import (
    AnalyticsInputs,
    build_portfolio_analytics,
)
from mic_data.analytics.report import main


class TestAnalyticsReport(unittest.TestCase):
    @staticmethod
    def _build_inputs() -> tuple[AnalyticsInputs, pd.DataFrame]:
        idx = pd.date_range("2024-01-02", periods=160, freq="B")
        rng = np.random.default_rng(23)

        factors = pd.DataFrame(
            {
                "trade_date": idx,
                "mkt_rf": rng.normal(0.0004, 0.009, len(idx)),
                "smb": rng.normal(0.0001, 0.005, len(idx)),
                "hml": rng.normal(0.0002, 0.006, len(idx)),
                "rf": np.full(len(idx), 0.00005),
            }
        )

        spy_ret = factors["mkt_rf"] + factors["rf"] + rng.normal(0.0, 0.0007, len(idx))
        aaa_ret = (
            0.0002
            + 1.10 * factors["mkt_rf"]
            + 0.30 * factors["smb"]
            - 0.10 * factors["hml"]
            + factors["rf"]
            + rng.normal(0.0, 0.001, len(idx))
        )
        bbb_ret = (
            -0.0001
            + 0.75 * factors["mkt_rf"]
            - 0.05 * factors["smb"]
            + 0.25 * factors["hml"]
            + factors["rf"]
            + rng.normal(0.0, 0.001, len(idx))
        )

        security_returns = pd.concat(
            [
                pd.DataFrame(
                    {
                        "trade_date": idx,
                        "ticker": "AAA",
                        "permno": 10001,
                        "ret": aaa_ret,
                        "prc": 150.0,
                        "vol": 1000.0,
                        "shrout": 2_100_000.0,
                        "source": "wrds_crsp",
                        "load_ts_utc": pd.Timestamp("2026-03-17T00:00:00Z"),
                    }
                ),
                pd.DataFrame(
                    {
                        "trade_date": idx,
                        "ticker": "BBB",
                        "permno": 10002,
                        "ret": bbb_ret,
                        "prc": 80.0,
                        "vol": 900.0,
                        "shrout": 250_000.0,
                        "source": "wrds_crsp",
                        "load_ts_utc": pd.Timestamp("2026-03-17T00:00:00Z"),
                    }
                ),
                pd.DataFrame(
                    {
                        "trade_date": idx,
                        "ticker": "SPY",
                        "permno": 10003,
                        "ret": spy_ret,
                        "prc": 500.0,
                        "vol": 1500.0,
                        "shrout": 900_000.0,
                        "source": "wrds_crsp",
                        "load_ts_utc": pd.Timestamp("2026-03-17T00:00:00Z"),
                    }
                ),
            ],
            ignore_index=True,
        )

        portfolio_returns = pd.DataFrame(
            {
                "trade_date": idx,
                "portfolio_ret": 0.6 * aaa_ret + 0.4 * bbb_ret,
                "n_constituents": 2,
                "gross_exposure": 1.0,
                "method": "holdings_weighted_sum",
                "load_ts_utc": pd.Timestamp("2026-03-17T00:00:00Z"),
            }
        )

        universe = pd.DataFrame(
            {
                "as_of_date": pd.Timestamp("2026-03-17"),
                "ticker": ["AAA", "BBB", "SPY"],
                "is_holding": [True, True, False],
                "is_watchlist": [False, False, True],
                "shares": [1000.0, 400.0, np.nan],
                "name": ["Alpha Corp", "Beta Health", "SPDR S&P 500 ETF"],
                "sector": ["Technology", "Healthcare", "ETF"],
            }
        )

        return (
            AnalyticsInputs(
                universe=universe,
                security_returns=security_returns,
                portfolio_returns=portfolio_returns,
            ),
            factors,
        )

    def test_build_portfolio_analytics_produces_expected_outputs(self) -> None:
        inputs, factors = self._build_inputs()

        result = build_portfolio_analytics(
            inputs=inputs,
            factors=factors,
            benchmark_ticker="SPY",
            beta_frequency="weekly",
            ff3_min_obs=100,
            top_n_holdings=2,
        )

        self.assertIn("benchmark_ret", result.benchmark_comparison.columns)
        self.assertEqual(result.summary["metadata"]["benchmark_ticker"], "SPY")
        self.assertGreater(float(result.summary["sharpe"]["annualized_sharpe"]), 0.0)
        self.assertEqual(result.holdings_snapshot.iloc[0]["ticker"], "AAA")
        self.assertEqual(result.holdings_snapshot.iloc[0]["market_cap_bucket"], "Mega Cap")
        self.assertAlmostEqual(float(result.market_cap_mix["portfolio_weight"].sum()), 1.0, places=6)
        self.assertIn("portfolio_return_exposure", result.summary["ff3"])
        self.assertIn("mkt_rf", result.holdings_ff3_loadings.columns)
        self.assertIn("ff3_modeled", result.holdings_ff3_loadings.columns)
        self.assertTrue(result.holdings_ff3_loadings["ff3_modeled"].all())
        self.assertIn(
            "Historical portfolio composition changes are not yet modeled in this analytics layer.",
            result.summary["metadata"]["limitations"],
        )

    def test_build_portfolio_analytics_requires_persisted_benchmark(self) -> None:
        inputs, factors = self._build_inputs()
        missing_benchmark_inputs = AnalyticsInputs(
            universe=inputs.universe,
            security_returns=inputs.security_returns[inputs.security_returns["ticker"] != "SPY"].copy(),
            portfolio_returns=inputs.portfolio_returns,
        )

        with self.assertRaisesRegex(ValueError, "Add it to the Google Sheets Universe tab"):
            build_portfolio_analytics(
                inputs=missing_benchmark_inputs,
                factors=factors,
                benchmark_ticker="SPY",
                ff3_min_obs=100,
            )

    def test_main_writes_tables_charts_and_summary_json(self) -> None:
        inputs, factors = self._build_inputs()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            universe_path = root / "universe_latest.parquet"
            security_returns_path = root / "security_returns_daily.parquet"
            portfolio_returns_path = root / "portfolio_returns_daily.parquet"
            analytics_dir = root / "analytics"
            charts_dir = root / "charts"
            summary_json_path = root / "portfolio_dashboard_summary.json"
            config_path = root / "analytics.yaml"

            inputs.universe.to_parquet(universe_path, index=False)
            inputs.security_returns.to_parquet(security_returns_path, index=False)
            inputs.portfolio_returns.to_parquet(portfolio_returns_path, index=False)

            config_path.write_text(
                "\n".join(
                    [
                        "run:",
                        '  benchmark_ticker: "SPY"',
                        '  beta_frequency: "weekly"',
                        "  ff3_min_obs: 100",
                        "  top_n_holdings: 2",
                        "sources:",
                        f'  universe_path: "{universe_path}"',
                        f'  security_returns_path: "{security_returns_path}"',
                        f'  portfolio_returns_path: "{portfolio_returns_path}"',
                        "outputs:",
                        f'  analytics_dir: "{analytics_dir}"',
                        f'  charts_dir: "{charts_dir}"',
                        f'  summary_json_path: "{summary_json_path}"',
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            stdout_buffer = io.StringIO()
            with patch(
                "mic_data.analytics.report.load_ff3_factors_from_wrds",
                return_value=factors,
            ), redirect_stdout(stdout_buffer):
                main(["--config", str(config_path)])

            payload = json.loads(stdout_buffer.getvalue())
            self.assertEqual(payload["summary"]["metadata"]["benchmark_ticker"], "SPY")
            self.assertTrue((analytics_dir / "benchmark_comparison.parquet").exists())
            self.assertTrue((analytics_dir / "current_holdings_snapshot.csv").exists())
            self.assertTrue((analytics_dir / "ff3" / "portfolio_exposure_comparison.csv").exists())
            self.assertTrue((analytics_dir / "ff3" / "holdings_ff3_loadings.parquet").exists())
            self.assertTrue((charts_dir / "performance_vs_spy.svg").exists())
            self.assertTrue((charts_dir / "ff3_portfolio_exposure.svg").exists())
            self.assertTrue((charts_dir / "ff3_exposure_comparison.svg").exists())
            self.assertTrue((charts_dir / "ff3_factor_risk_contributions.svg").exists())
            self.assertTrue((charts_dir / "ff3_security_heatmap.svg").exists())
            self.assertTrue(summary_json_path.exists())
            self.assertIn("<svg", (charts_dir / "performance_vs_spy.svg").read_text(encoding="utf-8"))
            self.assertIn("ff3", payload["summary"])


if __name__ == "__main__":
    unittest.main()
