from __future__ import annotations

import json
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import TypedDict

import pandas as pd
import yaml

from mic_data.models.comparison import (
    ComparisonThresholds,
    RegressionDeltas,
    compare_factor_inputs,
    compare_regression_results,
    evaluate_similarity,
)
from mic_data.models.constants import CANONICAL_FREQUENCY, ModelFrequency
from mic_data.models.factors.static_csv_source import StaticCsvFactorSource
from mic_data.models.factors.wrds_source import WrdsFactorSource
from mic_data.models.interfaces import FactorSource, PortfolioReturnSource
from mic_data.models.portfolio_returns import YFinancePortfolioReturnSource
from mic_data.models.regression import run_ff3_regression


DEFAULT_START_DATE = "2000-01-01"
DEFAULT_END_DATE = date.today().isoformat()
DEFAULT_ALLOW_FALLBACK = True


@dataclass(frozen=True)
class FF3PipelineConfig:
    """Configuration for WRDS-first FF3 pipeline execution.

    Inputs:
      - start_date/end_date: Inclusive analysis window in YYYY-MM-DD.
      - frequency: Currently monthly only ("M").
      - holdings_path: Path to holdings_latest CSV.
      - static_factor_path: Path to local Ken French factor CSV.
      - processed_model_inputs_dir: Destination for factor parquet outputs.
      - validation_output_dir: Destination for diagnostics/report artifacts.
      - logs_dir: Destination for JSONL run logs.
      - wrds_username: Optional username override for WRDS auth.
      - allow_fallback: Whether to continue with static factors if WRDS fails.

    Returns:
      - Immutable dataclass used by run_ff3_pipeline.

    Raises:
      - None.

    Notes on units:
      - Dates and returns are interpreted by downstream modules as monthly decimal returns.
    """

    start_date: str = DEFAULT_START_DATE
    end_date: str = DEFAULT_END_DATE
    frequency: ModelFrequency = CANONICAL_FREQUENCY
    holdings_path: Path = Path("data/processed/holdings_latest.csv")
    static_factor_path: Path = Path("data/raw/factors/F-F_Research_Data_Factors.csv")
    processed_model_inputs_dir: Path = Path("data/processed/model_inputs")
    validation_output_dir: Path = Path("outputs/validation")
    logs_dir: Path = Path("outputs/logs")
    wrds_username: str | None = None
    allow_fallback: bool = DEFAULT_ALLOW_FALLBACK


@dataclass(frozen=True)
class FF3PipelineArtifacts:
    """Artifact paths emitted by a single FF3 pipeline run."""

    log_path: Path
    static_factors_path: Path | None
    wrds_factors_path: Path | None
    factor_comparison_csv: Path
    regression_comparison_json: Path
    validation_summary_md: Path
    used_fallback: bool
    wrds_available: bool


class JsonlRunLogger:
    """Append-only JSONL logger for reproducible pipeline diagnostics."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        *,
        stage: str,
        source: str,
        status: str,
        rows: int | None = None,
        start: str | None = None,
        end: str | None = None,
        duration_ms: int | None = None,
        error_type: str | None = None,
        error_message: str | None = None,
        fallback_used: bool = False,
        extra: Mapping[str, object] | None = None,
    ) -> None:
        payload: dict[str, object] = {
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "stage": stage,
            "source": source,
            "status": status,
            "rows": rows,
            "start": start,
            "end": end,
            "duration_ms": duration_ms,
            "error_type": error_type,
            "error_message": error_message,
            "fallback_used": fallback_used,
        }
        if extra:
            payload.update(extra)

        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, default=str) + "\n")


@dataclass(frozen=True)
class _RunState:
    static_factors: pd.DataFrame | None
    wrds_factors: pd.DataFrame | None
    used_fallback: bool


class _FactorMetricsPayload(TypedDict):
    factor: str
    corr: float
    mad_bps: float
    rmse_bps: float
    n_obs: int


def _thresholds_payload(thresholds: ComparisonThresholds) -> dict[str, float]:
    return {
        "corr_min": thresholds.corr_min,
        "mad_max_bps": thresholds.mad_max_bps,
        "delta_beta_max": thresholds.delta_beta_max,
        "delta_alpha_max_bps": thresholds.delta_alpha_max_bps,
        "delta_r2_max": thresholds.delta_r2_max,
    }


def _factor_metrics_records(factor_metrics: pd.DataFrame) -> list[_FactorMetricsPayload]:
    records: list[_FactorMetricsPayload] = []
    for factor, row in factor_metrics.iterrows():
        records.append(
            {
                "factor": str(factor),
                "corr": float(row["corr"]),
                "mad_bps": float(row["mad_bps"]),
                "rmse_bps": float(row["rmse_bps"]),
                "n_obs": int(row["n_obs"]),
            }
        )
    return records


def _as_mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _get_string(mapping: Mapping[str, object], key: str, default: str) -> str:
    value = mapping.get(key)
    return value if isinstance(value, str) else default


def _get_optional_string(mapping: Mapping[str, object], key: str) -> str | None:
    value = mapping.get(key)
    return value if isinstance(value, str) else None


def _get_path(mapping: Mapping[str, object], key: str, default: Path) -> Path:
    value = mapping.get(key)
    return Path(value) if isinstance(value, str) else default


def _get_bool(mapping: Mapping[str, object], key: str, default: bool) -> bool:
    value = mapping.get(key)
    return value if isinstance(value, bool) else default


def _parse_frequency(raw_frequency: str) -> ModelFrequency:
    if raw_frequency != CANONICAL_FREQUENCY:
        raise ValueError(
            f"Unsupported frequency '{raw_frequency}'. Only '{CANONICAL_FREQUENCY}' is allowed."
        )
    return CANONICAL_FREQUENCY


def load_ff3_pipeline_config(path: str | Path) -> FF3PipelineConfig:
    """Load pipeline config from YAML.

    Inputs:
      - path: YAML path containing run/source/output/options blocks.

    Returns:
      - FF3PipelineConfig with defaults merged with file values.

    Raises:
      - FileNotFoundError when config file is missing.
      - ValueError when YAML root is not a mapping.

    Notes on units:
      - Date strings are passed through to downstream loaders unchanged.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"FF3 pipeline config not found: {p}")

    with p.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    if not isinstance(raw, dict):
        raise ValueError("FF3 config YAML root must be a mapping.")

    run = _as_mapping(raw.get("run", {}))
    sources = _as_mapping(raw.get("sources", {}))
    outputs = _as_mapping(raw.get("outputs", {}))
    options = _as_mapping(raw.get("options", {}))

    frequency = _parse_frequency(
        _get_string(run, "frequency", CANONICAL_FREQUENCY),
    )

    return FF3PipelineConfig(
        start_date=_get_string(run, "start_date", DEFAULT_START_DATE),
        end_date=_get_string(run, "end_date", DEFAULT_END_DATE),
        frequency=frequency,
        holdings_path=_get_path(
            sources,
            "holdings_path",
            Path("data/processed/holdings_latest.csv"),
        ),
        static_factor_path=_get_path(
            sources,
            "static_factor_path",
            Path("data/raw/factors/F-F_Research_Data_Factors.csv"),
        ),
        processed_model_inputs_dir=_get_path(
            outputs,
            "processed_model_inputs_dir",
            Path("data/processed/model_inputs"),
        ),
        validation_output_dir=_get_path(
            outputs,
            "validation_output_dir",
            Path("outputs/validation"),
        ),
        logs_dir=_get_path(outputs, "logs_dir", Path("outputs/logs")),
        wrds_username=_get_optional_string(sources, "wrds_username"),
        allow_fallback=_get_bool(
            options,
            "allow_fallback",
            DEFAULT_ALLOW_FALLBACK,
        ),
    )


def run_ff3_pipeline(
    config: FF3PipelineConfig,
    *,
    thresholds: ComparisonThresholds | None = None,
    wrds_source: FactorSource | None = None,
    static_source: FactorSource | None = None,
    portfolio_source: PortfolioReturnSource | None = None,
) -> FF3PipelineArtifacts:
    """Run WRDS-first FF3 pipeline with fallback and A/B validation.

    Inputs:
      - config: FF3 pipeline settings.
      - thresholds: Optional similarity thresholds for validation gates.
      - wrds_source/static_source/portfolio_source: Optional injectable sources for tests.

    Returns:
      - FF3PipelineArtifacts listing output locations and fallback status.

    Raises:
      - RuntimeError when both WRDS and static factor loads fail.
      - ValueError for invalid inputs propagated from source modules.

    Notes on units:
      - All return/factor values are decimal returns in monthly frequency.
    """
    if config.frequency != CANONICAL_FREQUENCY:
        raise ValueError("FF3 pipeline currently supports monthly frequency only ('M').")

    thresholds = thresholds or ComparisonThresholds()

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    log_path = config.logs_dir / f"ff3_pipeline_{stamp}.jsonl"
    logger = JsonlRunLogger(log_path)

    wrds_source = wrds_source or WrdsFactorSource(username=config.wrds_username)
    static_source = static_source or StaticCsvFactorSource(config.static_factor_path)
    portfolio_source = portfolio_source or YFinancePortfolioReturnSource(config.holdings_path)

    portfolio_returns = _timed_load_portfolio(
        source=portfolio_source,
        config=config,
        logger=logger,
    )

    run_state = _load_factor_sources(
        wrds_source=wrds_source,
        static_source=static_source,
        config=config,
        logger=logger,
    )

    if run_state.wrds_factors is None and run_state.static_factors is None:
        raise RuntimeError(
            "Both WRDS and static factor pulls failed. See JSONL run log for details."
        )

    config.processed_model_inputs_dir.mkdir(parents=True, exist_ok=True)
    config.validation_output_dir.mkdir(parents=True, exist_ok=True)

    static_factors_path: Path | None = None
    wrds_factors_path: Path | None = None

    if run_state.static_factors is not None:
        static_factors_path = config.processed_model_inputs_dir / "factors_static_m.parquet"
        _write_parquet(run_state.static_factors, static_factors_path)

    if run_state.wrds_factors is not None:
        wrds_factors_path = config.processed_model_inputs_dir / "factors_wrds_m.parquet"
        _write_parquet(run_state.wrds_factors, wrds_factors_path)

    wrds_regression = None
    static_regression = None
    if run_state.wrds_factors is not None:
        wrds_regression = run_ff3_regression(portfolio_returns, run_state.wrds_factors)
    if run_state.static_factors is not None:
        static_regression = run_ff3_regression(portfolio_returns, run_state.static_factors)

    factor_comparison_csv = config.validation_output_dir / "ff3_input_comparison.csv"
    regression_comparison_json = config.validation_output_dir / "ff3_regression_comparison.json"
    validation_summary_md = config.validation_output_dir / "ff3_validation_summary.md"

    summary_payload: dict[str, object] = {
        "config": {
            "start_date": config.start_date,
            "end_date": config.end_date,
            "frequency": config.frequency,
            "allow_fallback": config.allow_fallback,
        },
        "wrds_available": run_state.wrds_factors is not None,
        "fallback_used": run_state.used_fallback,
        "thresholds": _thresholds_payload(thresholds),
    }
    gate_passed = False

    if run_state.wrds_factors is not None and run_state.static_factors is not None:
        factor_metrics = compare_factor_inputs(run_state.wrds_factors, run_state.static_factors)
        factor_metrics.to_csv(factor_comparison_csv)

        assert wrds_regression is not None
        assert static_regression is not None
        regression_deltas: RegressionDeltas = compare_regression_results(
            wrds_regression,
            static_regression,
        )
        gate = evaluate_similarity(factor_metrics, regression_deltas, thresholds)
        gate_passed = gate["passed"]
        factor_records = _factor_metrics_records(factor_metrics)

        summary_payload.update(
            {
                "factor_metrics": factor_records,
                "regression_deltas": regression_deltas,
                "gate": gate,
                "wrds_regression": wrds_regression.to_dict(),
                "static_regression": static_regression.to_dict(),
            }
        )
    else:
        msg = (
            "A/B comparison skipped because one factor source is unavailable "
            "(WRDS failed and fallback path was used)."
        )
        pd.DataFrame([{"message": msg}]).to_csv(factor_comparison_csv, index=False)
        summary_payload.update(
            {
                "factor_metrics": [],
                "regression_deltas": {},
                "gate": {
                    "passed": False,
                    "reason_codes": ["comparison_skipped_missing_source"],
                    "thresholds": _thresholds_payload(thresholds),
                },
                "wrds_regression": wrds_regression.to_dict() if wrds_regression else None,
                "static_regression": (
                    static_regression.to_dict() if static_regression else None
                ),
            }
        )

    with regression_comparison_json.open("w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2, default=str)

    _write_validation_summary(
        path=validation_summary_md,
        summary_payload=summary_payload,
        static_factors_path=static_factors_path,
        wrds_factors_path=wrds_factors_path,
        factor_comparison_csv=factor_comparison_csv,
        regression_comparison_json=regression_comparison_json,
        log_path=log_path,
    )

    logger.log(
        stage="pipeline",
        source="ff3_runner",
        status="success",
        start=config.start_date,
        end=config.end_date,
        fallback_used=run_state.used_fallback,
        extra={
            "wrds_available": run_state.wrds_factors is not None,
            "validation_passed": gate_passed,
        },
    )

    return FF3PipelineArtifacts(
        log_path=log_path,
        static_factors_path=static_factors_path,
        wrds_factors_path=wrds_factors_path,
        factor_comparison_csv=factor_comparison_csv,
        regression_comparison_json=regression_comparison_json,
        validation_summary_md=validation_summary_md,
        used_fallback=run_state.used_fallback,
        wrds_available=run_state.wrds_factors is not None,
    )


def _timed_load_portfolio(
    *,
    source: PortfolioReturnSource,
    config: FF3PipelineConfig,
    logger: JsonlRunLogger,
) -> pd.Series:
    start = time.perf_counter()
    try:
        portfolio_returns = source.load_portfolio_returns(
            config.start_date,
            config.end_date,
            config.frequency,
        )
    except Exception as exc:
        logger.log(
            stage="portfolio_returns",
            source=source.__class__.__name__,
            status="error",
            start=config.start_date,
            end=config.end_date,
            duration_ms=int((time.perf_counter() - start) * 1000),
            error_type=type(exc).__name__,
            error_message=str(exc),
        )
        raise

    logger.log(
        stage="portfolio_returns",
        source=source.__class__.__name__,
        status="success",
        rows=len(portfolio_returns),
        start=config.start_date,
        end=config.end_date,
        duration_ms=int((time.perf_counter() - start) * 1000),
    )
    return portfolio_returns


def _load_factor_sources(
    *,
    wrds_source: FactorSource,
    static_source: FactorSource,
    config: FF3PipelineConfig,
    logger: JsonlRunLogger,
) -> _RunState:
    static_factors = None
    wrds_factors = None
    used_fallback = False

    start = time.perf_counter()
    try:
        static_factors = static_source.load_factors(
            config.start_date,
            config.end_date,
            config.frequency,
        )
        logger.log(
            stage="factors",
            source="static_csv",
            status="success",
            rows=len(static_factors),
            start=config.start_date,
            end=config.end_date,
            duration_ms=int((time.perf_counter() - start) * 1000),
        )
    except Exception as exc:
        logger.log(
            stage="factors",
            source="static_csv",
            status="error",
            start=config.start_date,
            end=config.end_date,
            duration_ms=int((time.perf_counter() - start) * 1000),
            error_type=type(exc).__name__,
            error_message=str(exc),
        )

    start = time.perf_counter()
    try:
        wrds_factors = wrds_source.load_factors(
            config.start_date,
            config.end_date,
            config.frequency,
        )
        logger.log(
            stage="factors",
            source="wrds",
            status="success",
            rows=len(wrds_factors),
            start=config.start_date,
            end=config.end_date,
            duration_ms=int((time.perf_counter() - start) * 1000),
        )
    except Exception as exc:
        used_fallback = True
        logger.log(
            stage="factors",
            source="wrds",
            status="error",
            start=config.start_date,
            end=config.end_date,
            duration_ms=int((time.perf_counter() - start) * 1000),
            error_type=type(exc).__name__,
            error_message=str(exc),
            fallback_used=config.allow_fallback,
        )
        if not config.allow_fallback:
            raise RuntimeError(
                "WRDS factor pull failed and allow_fallback is disabled."
            ) from exc

    return _RunState(
        static_factors=static_factors,
        wrds_factors=wrds_factors,
        used_fallback=used_fallback,
    )


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path, index=True)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to write parquet artifact at {path}. "
            "Install pyarrow or fastparquet for parquet support."
        ) from exc


def _write_validation_summary(
    *,
    path: Path,
    summary_payload: Mapping[str, object],
    static_factors_path: Path | None,
    wrds_factors_path: Path | None,
    factor_comparison_csv: Path,
    regression_comparison_json: Path,
    log_path: Path,
) -> None:
    gate_obj = summary_payload.get("gate", {})
    gate = gate_obj if isinstance(gate_obj, Mapping) else {}
    reason_codes_obj = gate.get("reason_codes", [])
    reason_codes_list = (
        [str(code) for code in reason_codes_obj]
        if isinstance(reason_codes_obj, list)
        else []
    )
    lines = [
        "# FF3 Validation Summary",
        "",
        f"- WRDS available: {summary_payload.get('wrds_available')}",
        f"- Fallback used: {summary_payload.get('fallback_used')}",
        f"- Validation passed: {gate.get('passed')}",
        f"- Reason codes: {', '.join(reason_codes_list) or 'none'}",
        "",
        "## Artifacts",
        f"- Static factors parquet: {static_factors_path}",
        f"- WRDS factors parquet: {wrds_factors_path}",
        f"- Factor comparison CSV: {factor_comparison_csv}",
        f"- Regression comparison JSON: {regression_comparison_json}",
        f"- Run log JSONL: {log_path}",
        "",
    ]

    factor_metrics_obj = summary_payload.get("factor_metrics")
    factor_metrics = (
        factor_metrics_obj
        if isinstance(factor_metrics_obj, list)
        else []
    )
    if factor_metrics:
        lines.extend(
            [
                "## Factor Metrics",
                "",
                "| factor | corr | mad_bps | rmse_bps | n_obs |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for row in factor_metrics:
            if not isinstance(row, Mapping):
                continue
            lines.append(
                "| {factor} | {corr:.6f} | {mad_bps:.4f} | {rmse_bps:.4f} | {n_obs} |".format(
                    factor=row.get("factor", "unknown"),
                    corr=float(row.get("corr", 0.0)),
                    mad_bps=float(row.get("mad_bps", 0.0)),
                    rmse_bps=float(row.get("rmse_bps", 0.0)),
                    n_obs=int(row.get("n_obs", 0)),
                )
            )
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
