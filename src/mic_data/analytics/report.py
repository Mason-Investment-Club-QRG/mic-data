from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import yaml

from mic_data.analytics.charts import render_analytics_charts
from mic_data.analytics.dashboard import (
    DEFAULT_UNIVERSE_PATH,
    AnalyticsInputs,
    PortfolioAnalyticsResult,
    build_portfolio_analytics,
    load_analytics_inputs,
)
from mic_data.models.ff_factor_matrix import (
    DEFAULT_PORTFOLIO_RETURNS_PATH,
    DEFAULT_SECURITY_RETURNS_PATH,
    load_ff3_factors_from_wrds,
)
from mic_data.utils.idempotent_io import (
    atomic_write_csv,
    atomic_write_parquet,
    write_manifest_json,
)


@dataclass(frozen=True)
class AnalyticsReportConfig:
    """Runtime configuration for the portfolio analytics report."""

    start_date: str | None
    end_date: str | None
    universe_path: Path
    security_returns_path: Path
    portfolio_returns_path: Path
    benchmark_ticker: str
    beta_frequency: str
    ff3_min_obs: int
    trading_days_per_year: int
    top_n_holdings: int
    output_value_base: float
    analytics_dir: Path
    charts_dir: Path
    summary_json_path: Path
    wrds_username: str | None


def load_report_config(path: str | Path) -> AnalyticsReportConfig:
    """Load analytics report configuration from YAML."""

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Analytics config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    root = _require_mapping(raw, section_name="root")
    run = _require_mapping(root.get("run"), section_name="run")
    sources = _require_mapping(root.get("sources"), section_name="sources")
    outputs = _require_mapping(root.get("outputs"), section_name="outputs")

    return AnalyticsReportConfig(
        start_date=_optional_string(run, key="start_date"),
        end_date=_optional_string(run, key="end_date"),
        universe_path=Path(
            _optional_string(sources, key="universe_path") or str(DEFAULT_UNIVERSE_PATH)
        ),
        security_returns_path=Path(
            _optional_string(sources, key="security_returns_path")
            or str(DEFAULT_SECURITY_RETURNS_PATH)
        ),
        portfolio_returns_path=Path(
            _optional_string(sources, key="portfolio_returns_path")
            or str(DEFAULT_PORTFOLIO_RETURNS_PATH)
        ),
        benchmark_ticker=_optional_string(run, key="benchmark_ticker") or "SPY",
        beta_frequency=_optional_string(run, key="beta_frequency") or "weekly",
        ff3_min_obs=_require_int(run, key="ff3_min_obs", default=60),
        trading_days_per_year=_require_int(run, key="trading_days_per_year", default=252),
        top_n_holdings=_require_int(run, key="top_n_holdings", default=5),
        output_value_base=_require_float(run, key="output_value_base", default=100.0),
        analytics_dir=Path(
            _optional_string(outputs, key="analytics_dir") or "data/processed/analytics"
        ),
        charts_dir=Path(_optional_string(outputs, key="charts_dir") or "outputs/charts"),
        summary_json_path=Path(
            _optional_string(outputs, key="summary_json_path")
            or "outputs/analytics/portfolio_dashboard_summary.json"
        ),
        wrds_username=_optional_string(sources, key="wrds_username"),
    )


def run_analytics_report(config: AnalyticsReportConfig) -> dict[str, object]:
    """Execute the persisted-data portfolio analytics report and write artifacts."""

    inputs = load_analytics_inputs(
        universe_path=config.universe_path,
        security_returns_path=config.security_returns_path,
        portfolio_returns_path=config.portfolio_returns_path,
        start_date=config.start_date,
        end_date=config.end_date,
    )
    factor_start, factor_end = _resolve_factor_window(
        inputs=inputs,
        start_date=config.start_date,
        end_date=config.end_date,
    )
    factors = load_ff3_factors_from_wrds(
        start_date=factor_start,
        end_date=factor_end,
        username=config.wrds_username,
    )
    result = build_portfolio_analytics(
        inputs=inputs,
        factors=factors,
        benchmark_ticker=config.benchmark_ticker,
        beta_frequency=config.beta_frequency,
        ff3_min_obs=config.ff3_min_obs,
        trading_days_per_year=config.trading_days_per_year,
        top_n_holdings=config.top_n_holdings,
        output_value_base=config.output_value_base,
    )

    artifact_paths = _write_analytics_tables(result=result, analytics_dir=config.analytics_dir)
    chart_paths = render_analytics_charts(
        result=result,
        charts_dir=config.charts_dir,
        benchmark_ticker=config.benchmark_ticker,
        top_n_holdings=config.top_n_holdings,
    )
    payload = _summary_payload(
        config=config,
        result=result,
        artifact_paths=artifact_paths,
        chart_paths=chart_paths,
    )
    write_manifest_json(
        path=config.summary_json_path,
        payload=payload,
        mode="replace",
    )
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate portfolio benchmark/composition analytics and SVG charts."
    )
    parser.add_argument("--config", default="config/analytics.yaml", help="Analytics config YAML path.")
    parser.add_argument("--start-date", default=None, help="Inclusive start date (YYYY-MM-DD).")
    parser.add_argument("--end-date", default=None, help="Inclusive end date (YYYY-MM-DD).")
    parser.add_argument(
        "--benchmark-ticker",
        default=None,
        help="Benchmark ticker expected in persisted security returns.",
    )
    parser.add_argument(
        "--beta-frequency",
        choices=["daily", "weekly", "monthly"],
        default=None,
        help="Return frequency used for beta regression.",
    )
    parser.add_argument(
        "--ff3-min-obs",
        type=int,
        default=None,
        help="Minimum overlapping observations required for FF3 regressions.",
    )
    parser.add_argument(
        "--top-n-holdings",
        type=int,
        default=None,
        help="Number of holdings to show in the top-holdings chart.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional override for the summary JSON output path.",
    )
    parser.add_argument(
        "--wrds-username",
        default=None,
        help="Optional WRDS username override for factor pulls.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    config = load_report_config(args.config)
    config = _apply_overrides(
        config,
        start_date=args.start_date,
        end_date=args.end_date,
        benchmark_ticker=args.benchmark_ticker,
        beta_frequency=args.beta_frequency,
        ff3_min_obs=args.ff3_min_obs,
        top_n_holdings=args.top_n_holdings,
        output_json=args.output_json,
        wrds_username=args.wrds_username,
    )
    payload = run_analytics_report(config)
    print(json.dumps(payload, indent=2))


def _apply_overrides(
    config: AnalyticsReportConfig,
    *,
    start_date: str | None,
    end_date: str | None,
    benchmark_ticker: str | None,
    beta_frequency: str | None,
    ff3_min_obs: int | None,
    top_n_holdings: int | None,
    output_json: str | None,
    wrds_username: str | None,
) -> AnalyticsReportConfig:
    return AnalyticsReportConfig(
        start_date=start_date or config.start_date,
        end_date=end_date or config.end_date,
        universe_path=config.universe_path,
        security_returns_path=config.security_returns_path,
        portfolio_returns_path=config.portfolio_returns_path,
        benchmark_ticker=(benchmark_ticker or config.benchmark_ticker).upper(),
        beta_frequency=beta_frequency or config.beta_frequency,
        ff3_min_obs=ff3_min_obs or config.ff3_min_obs,
        trading_days_per_year=config.trading_days_per_year,
        top_n_holdings=top_n_holdings or config.top_n_holdings,
        output_value_base=config.output_value_base,
        analytics_dir=config.analytics_dir,
        charts_dir=config.charts_dir,
        summary_json_path=Path(output_json) if output_json else config.summary_json_path,
        wrds_username=wrds_username or config.wrds_username,
    )


def _write_analytics_tables(
    *,
    result: PortfolioAnalyticsResult,
    analytics_dir: Path,
) -> dict[str, str]:
    analytics_dir.mkdir(parents=True, exist_ok=True)

    benchmark_parquet = analytics_dir / "benchmark_comparison.parquet"
    benchmark_csv = analytics_dir / "benchmark_comparison.csv"
    beta_parquet = analytics_dir / "beta_regression.parquet"
    beta_csv = analytics_dir / "beta_regression.csv"
    holdings_parquet = analytics_dir / "current_holdings_snapshot.parquet"
    holdings_csv = analytics_dir / "current_holdings_snapshot.csv"
    market_cap_parquet = analytics_dir / "market_cap_mix.parquet"
    market_cap_csv = analytics_dir / "market_cap_mix.csv"
    ff3_dir = analytics_dir / "ff3"

    atomic_write_parquet(
        result.benchmark_comparison,
        path=benchmark_parquet,
        sort_by=["trade_date"],
        mode="replace",
    )
    atomic_write_csv(
        result.benchmark_comparison,
        path=benchmark_csv,
        sort_by=["trade_date"],
        mode="replace",
    )
    atomic_write_parquet(
        result.beta_regression,
        path=beta_parquet,
        sort_by=["trade_date"],
        mode="replace",
    )
    atomic_write_csv(
        result.beta_regression,
        path=beta_csv,
        sort_by=["trade_date"],
        mode="replace",
    )
    atomic_write_parquet(
        result.holdings_snapshot,
        path=holdings_parquet,
        sort_by=["weight_rank", "ticker"],
        mode="replace",
    )
    atomic_write_csv(
        result.holdings_snapshot,
        path=holdings_csv,
        sort_by=["weight_rank", "ticker"],
        mode="replace",
    )
    atomic_write_parquet(
        result.market_cap_mix,
        path=market_cap_parquet,
        sort_by=["bucket_order"],
        mode="replace",
    )
    atomic_write_csv(
        result.market_cap_mix,
        path=market_cap_csv,
        sort_by=["bucket_order"],
        mode="replace",
    )
    ff3_artifacts = _write_ff3_tables(result=result, ff3_dir=ff3_dir)

    return {
        "benchmark_comparison_parquet": str(benchmark_parquet),
        "benchmark_comparison_csv": str(benchmark_csv),
        "beta_regression_parquet": str(beta_parquet),
        "beta_regression_csv": str(beta_csv),
        "holdings_snapshot_parquet": str(holdings_parquet),
        "holdings_snapshot_csv": str(holdings_csv),
        "market_cap_mix_parquet": str(market_cap_parquet),
        "market_cap_mix_csv": str(market_cap_csv),
        "ff3": ff3_artifacts,
    }


def _write_ff3_tables(
    *,
    result: PortfolioAnalyticsResult,
    ff3_dir: Path,
) -> dict[str, str]:
    ff3_dir.mkdir(parents=True, exist_ok=True)

    sort_keys = {
        "security_loadings": ["ticker"],
        "security_beta_matrix": ["ticker"],
        "portfolio_exposure_comparison": ["exposure_method"],
        "portfolio_risk_summary": ["metric"],
        "portfolio_holdings_risk_summary": ["metric"],
        "factor_risk_contributions": ["factor"],
        "factor_covariance": ["factor"],
        "factor_correlation": ["factor"],
        "security_factor_covariance": ["ticker"],
        "security_factor_correlation": ["ticker"],
        "holdings_ff3_loadings": ["weight_rank", "ticker"],
    }

    artifacts: dict[str, str] = {}
    for table_name, frame in result.ff3_analysis.tables.items():
        preferred_sort = sort_keys.get(table_name)
        sort_by = _resolve_sort_columns(frame, preferred_sort)
        parquet_path = ff3_dir / f"{table_name}.parquet"
        csv_path = ff3_dir / f"{table_name}.csv"
        atomic_write_parquet(frame, path=parquet_path, sort_by=sort_by, mode="replace")
        atomic_write_csv(frame, path=csv_path, sort_by=sort_by, mode="replace")
        artifacts[f"{table_name}_parquet"] = str(parquet_path)
        artifacts[f"{table_name}_csv"] = str(csv_path)

    holdings_ff3_parquet = ff3_dir / "holdings_ff3_loadings.parquet"
    holdings_ff3_csv = ff3_dir / "holdings_ff3_loadings.csv"
    atomic_write_parquet(
        result.holdings_ff3_loadings,
        path=holdings_ff3_parquet,
        sort_by=sort_keys["holdings_ff3_loadings"],
        mode="replace",
    )
    atomic_write_csv(
        result.holdings_ff3_loadings,
        path=holdings_ff3_csv,
        sort_by=sort_keys["holdings_ff3_loadings"],
        mode="replace",
    )
    artifacts["holdings_ff3_loadings_parquet"] = str(holdings_ff3_parquet)
    artifacts["holdings_ff3_loadings_csv"] = str(holdings_ff3_csv)
    return artifacts


def _summary_payload(
    *,
    config: AnalyticsReportConfig,
    result: PortfolioAnalyticsResult,
    artifact_paths: dict[str, str],
    chart_paths: dict[str, Path],
) -> dict[str, object]:
    payload = {
        "config": {
            "start_date": config.start_date,
            "end_date": config.end_date,
            "benchmark_ticker": config.benchmark_ticker,
            "beta_frequency": config.beta_frequency,
            "ff3_min_obs": config.ff3_min_obs,
            "top_n_holdings": config.top_n_holdings,
            "trading_days_per_year": config.trading_days_per_year,
        },
        "summary": result.summary,
        "artifacts": artifact_paths,
        "charts": {key: str(path) for key, path in chart_paths.items()},
    }
    return _json_safe(payload)


def _resolve_factor_window(
    *,
    inputs: AnalyticsInputs,
    start_date: str | None,
    end_date: str | None,
) -> tuple[str, str]:
    if start_date is not None and end_date is not None:
        return start_date, end_date

    trade_dates = pd.to_datetime(inputs.portfolio_returns["trade_date"], errors="raise")
    return (
        start_date or trade_dates.min().strftime("%Y-%m-%d"),
        end_date or trade_dates.max().strftime("%Y-%m-%d"),
    )


def _resolve_sort_columns(
    frame: pd.DataFrame,
    preferred: list[str] | None,
) -> list[str]:
    if preferred:
        available = [column for column in preferred if column in frame.columns]
        if available:
            return available
    if len(frame.columns) == 0:
        raise ValueError("Cannot persist an empty-column dataframe.")
    return [str(frame.columns[0])]


def _require_mapping(value: Any, *, section_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"Section '{section_name}' must be a mapping.")
    return value


def _optional_string(mapping: dict[str, Any], *, key: str) -> str | None:
    value = mapping.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"Optional key '{key}' must be a string when provided.")
    stripped = value.strip()
    return stripped if stripped else None


def _require_int(mapping: dict[str, Any], *, key: str, default: int) -> int:
    value = mapping.get(key, default)
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Key '{key}' must be an integer.") from exc


def _require_float(mapping: dict[str, Any], *, key: str, default: float) -> float:
    value = mapping.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Key '{key}' must be a float.") from exc


def _json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_safe(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [_json_safe(inner) for inner in value]
    if isinstance(value, tuple):
        return [_json_safe(inner) for inner in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return value
    return value


__all__ = [
    "AnalyticsReportConfig",
    "build_parser",
    "load_report_config",
    "main",
    "run_analytics_report",
]


if __name__ == "__main__":
    main()
