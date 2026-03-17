# Purpose: Provide shared config/runtime helpers and optional run-all orchestration for the daily returns pipeline.

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

# Purpose: Make package imports work when this file is executed directly by path.
if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mic_data.contracts.daily_returns_contracts import DAILY_QA_KEY, QaRow, WriteMode, qa_row_to_frame, validate_daily_qa
from mic_data.utils.idempotent_io import atomic_write_parquet, write_manifest_json


@dataclass(frozen=True)
class DailyReturnsPaths:
    """File paths for all daily returns pipeline artifacts."""

    universe_latest_path: Path
    security_returns_path: Path
    security_returns_csv_path: Path
    portfolio_returns_path: Path
    portfolio_returns_csv_path: Path
    daily_qa_path: Path
    manifests_dir: Path
    locks_dir: Path
    logs_dir: Path


@dataclass(frozen=True)
class DailyReturnsConfig:
    """Runtime configuration for daily returns stages.

    Inputs:
      - start_date/end_date: Inclusive pull window in YYYY-MM-DD.
      - google_sheets_config_path: Centralized sheet IDs/tab names.
      - positions_config_path: Holdings/watchlist mapping config.
      - paths: Output and operational path settings.
      - wrds_username: Optional explicit WRDS username override.
      - if_exists: Idempotent write behavior for stage outputs.
      - dry_run: Validate and compute without mutating tracked outputs.

    Returns:
      - Immutable config consumed by stage CLIs and orchestrator.

    Raises:
      - None. Validation performed by loader.

    Notes on units:
      - Date ranges are calendar dates; financial units are handled per dataset contract.
    """

    start_date: str
    end_date: str
    google_sheets_config_path: Path
    positions_config_path: Path
    paths: DailyReturnsPaths
    wrds_username: str | None = None
    if_exists: WriteMode = "replace"
    dry_run: bool = False


@dataclass(frozen=True)
class DailyReturnsArtifacts:
    """Paths emitted by `run_daily_returns_pipeline`."""

    universe_path: Path
    security_returns_path: Path
    portfolio_returns_path: Path
    daily_qa_path: Path
    manifest_path: Path
    log_path: Path


# Purpose: Enforce mapping shape for YAML sections.
def _require_mapping(value: Any, *, section_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"Section '{section_name}' must be a mapping.")
    return value


# Purpose: Read required non-empty string values from YAML mappings.
def _require_string(mapping: dict[str, Any], *, key: str, section_name: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Section '{section_name}' requires non-empty string '{key}'.")
    return value.strip()


# Purpose: Read optional non-empty string values from YAML mappings.
def _optional_string(mapping: dict[str, Any], *, key: str) -> str | None:
    value = mapping.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"Optional key '{key}' must be a string when provided.")
    stripped = value.strip()
    return stripped if stripped else None


# Purpose: Validate and normalize write-mode strings from config or CLI.
def parse_write_mode(value: str) -> WriteMode:
    """Parse write mode literal.

    Inputs:
      - value: Candidate write-mode string.

    Returns:
      - Valid `WriteMode` literal.

    Raises:
      - ValueError if value is not one of replace/skip/error.

    Notes on units:
      - Control value only; no financial units.
    """

    if value not in ("replace", "skip", "error"):
        raise ValueError("if_exists must be one of: replace, skip, error")
    return value


# Purpose: Load daily returns YAML config consumed by all stage CLIs.
def load_daily_returns_config(path: str | Path) -> DailyReturnsConfig:
    """Load daily returns config from YAML.

    Inputs:
      - path: Config file path.

    Returns:
      - DailyReturnsConfig with validated required fields.

    Raises:
      - FileNotFoundError if file is missing.
      - ValueError if YAML schema is malformed.

    Notes on units:
      - Date strings are interpreted as inclusive calendar bounds.
    """

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Daily returns config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    root = _require_mapping(raw, section_name="root")
    run = _require_mapping(root.get("run"), section_name="run")
    sources = _require_mapping(root.get("sources"), section_name="sources")
    outputs = _require_mapping(root.get("outputs"), section_name="outputs")
    options = _require_mapping(root.get("options", {}), section_name="options")

    if_exists_raw = options.get("if_exists", "replace")
    if not isinstance(if_exists_raw, str):
        raise ValueError("options.if_exists must be a string.")

    dry_run_raw = options.get("dry_run", False)
    if not isinstance(dry_run_raw, bool):
        raise ValueError("options.dry_run must be boolean.")

    return DailyReturnsConfig(
        start_date=_require_string(run, key="start_date", section_name="run"),
        end_date=_require_string(run, key="end_date", section_name="run"),
        google_sheets_config_path=Path(
            _require_string(
                sources,
                key="google_sheets_config_path",
                section_name="sources",
            )
        ),
        positions_config_path=Path(
            _require_string(
                sources,
                key="positions_config_path",
                section_name="sources",
            )
        ),
        wrds_username=_optional_string(sources, key="wrds_username"),
        paths=DailyReturnsPaths(
            universe_latest_path=Path(
                _require_string(outputs, key="universe_latest_path", section_name="outputs")
            ),
            security_returns_path=Path(
                _require_string(outputs, key="security_returns_path", section_name="outputs")
            ),
            security_returns_csv_path=Path(
                _require_string(
                    outputs,
                    key="security_returns_csv_path",
                    section_name="outputs",
                )
            ),
            portfolio_returns_path=Path(
                _require_string(outputs, key="portfolio_returns_path", section_name="outputs")
            ),
            portfolio_returns_csv_path=Path(
                _require_string(
                    outputs,
                    key="portfolio_returns_csv_path",
                    section_name="outputs",
                )
            ),
            daily_qa_path=Path(_require_string(outputs, key="daily_qa_path", section_name="outputs")),
            manifests_dir=Path(_require_string(outputs, key="manifests_dir", section_name="outputs")),
            locks_dir=Path(_require_string(outputs, key="locks_dir", section_name="outputs")),
            logs_dir=Path(_require_string(outputs, key="logs_dir", section_name="outputs")),
        ),
        if_exists=parse_write_mode(if_exists_raw),
        dry_run=dry_run_raw,
    )


# Purpose: Return canonical run-date used for dated manifests and logs.
def pipeline_run_date() -> date:
    return date.today()


# Purpose: Resolve stage-specific lock path using configured lock directory.
def stage_lock_path(config: DailyReturnsConfig, *, stage_name: str) -> Path:
    return config.paths.locks_dir / f"{stage_name}.lock"


# Purpose: Resolve daily log file path in configured logs directory.
def log_path_for_date(config: DailyReturnsConfig, *, run_date: date) -> Path:
    return config.paths.logs_dir / f"daily_returns_{run_date.isoformat()}.jsonl"


# Purpose: Append structured stage logs for run observability.
def append_stage_log(
    *,
    config: DailyReturnsConfig,
    stage: str,
    status: str,
    row_count: int | None = None,
    detail: dict[str, object] | None = None,
) -> Path:
    """Append one JSONL stage log event.

    Inputs:
      - config: Daily pipeline config.
      - stage: Stage identifier.
      - status: Status label (ok/warn/error/skipped).
      - row_count: Optional row count metadata.
      - detail: Additional structured metadata.

    Returns:
      - Path to the log file where the event was appended.

    Raises:
      - OSError on write failures.

    Notes on units:
      - Row counts are integer metrics.
    """

    event = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "stage": stage,
        "status": status,
        "row_count": row_count,
    }
    if detail:
        event.update(detail)

    run_date = pipeline_run_date()
    path = log_path_for_date(config, run_date=run_date)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, default=str) + "\n")
    return path


# Purpose: Upsert a QA row into persistent daily QA parquet with contract validation.
def upsert_daily_qa_row(
    *,
    config: DailyReturnsConfig,
    qa_row: QaRow,
) -> pd.DataFrame:
    """Insert or replace one QA row by `(run_date, stage)` key.

    Inputs:
      - config: Daily pipeline config.
      - qa_row: Typed QA row.

    Returns:
      - Full validated QA dataframe after upsert.

    Raises:
      - ValueError for contract violations.
      - OSError for write failures when not in dry-run mode.

    Notes on units:
      - QA count columns are unitless integer metrics.
    """

    qa_path = config.paths.daily_qa_path
    if qa_path.exists():
        existing = pd.read_parquet(qa_path)
    else:
        existing = pd.DataFrame(columns=[
            "run_date",
            "stage",
            "rows_universe",
            "rows_returns",
            "missing_permno",
            "missing_ret",
            "status",
            "error_message",
        ])

    incoming = qa_row_to_frame(qa_row)
    if existing.empty:
        merged = incoming.copy()
    else:
        merged = pd.concat([existing, incoming], ignore_index=True)

    key_cols = list(DAILY_QA_KEY)
    merged = merged.sort_values(key_cols).drop_duplicates(key_cols, keep="last")
    validated = validate_daily_qa(merged)

    if not config.dry_run:
        atomic_write_parquet(
            validated,
            path=qa_path,
            sort_by=key_cols,
            mode="replace",
        )

    return validated


# Purpose: Write a run manifest that records dataset hashes and metadata for idempotency audits.
def write_daily_manifest(
    *,
    config: DailyReturnsConfig,
    as_of_date: date,
    datasets: dict[str, dict[str, object]],
) -> Path:
    """Persist daily manifest JSON payload.

    Inputs:
      - config: Daily pipeline config.
      - as_of_date: Business date for manifest partitioning.
      - datasets: Per-dataset metrics/hash payload.

    Returns:
      - Path to manifest file.

    Raises:
      - OSError/ValueError for write or serialization errors.

    Notes on units:
      - Manifest metadata is unitless.
    """

    manifest_path = config.paths.manifests_dir / f"daily_returns_{as_of_date.isoformat()}.json"
    existing_datasets: dict[str, object] = {}
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as f:
            existing_payload = json.load(f)
        if isinstance(existing_payload, dict):
            raw_datasets = existing_payload.get("datasets")
            if isinstance(raw_datasets, dict):
                existing_datasets = {str(k): v for k, v in raw_datasets.items()}

    merged_datasets: dict[str, object] = {**existing_datasets, **datasets}

    payload: dict[str, object] = {
        "as_of_date": as_of_date.isoformat(),
        "written_at_utc": datetime.now(UTC).isoformat(),
        "datasets": merged_datasets,
    }

    if not config.dry_run:
        write_manifest_json(path=manifest_path, payload=payload, mode="replace")

    return manifest_path


# Purpose: Execute all pipeline stages in order while preserving stage-level CLI autonomy.
def run_daily_returns_pipeline(config: DailyReturnsConfig) -> DailyReturnsArtifacts:
    """Run all daily returns stages sequentially.

    Inputs:
      - config: Pipeline config loaded from YAML or CLI overrides.

    Returns:
      - DailyReturnsArtifacts with output and tracking paths.

    Raises:
      - Propagates stage failures from individual stage modules.

    Notes on units:
      - Stage modules enforce per-dataset unit contracts.
    """

    from mic_data.market.build_portfolio_returns import run_build_portfolio_returns_stage
    from mic_data.market.pull_wrds_returns import run_pull_wrds_returns_stage
    from mic_data.positions.sync import run_positions_sync_stage
    from mic_data.positions.universe_sync import run_universe_sync_stage
    from mic_data.reporting.publish_sheets import run_publish_sheets_stage

    run_date = pipeline_run_date()

    run_positions_sync_stage(
        config_path=config.positions_config_path,
        as_of_date=run_date,
        if_exists=config.if_exists,
        dry_run=config.dry_run,
    )
    run_universe_sync_stage(
        config_path=config.positions_config_path,
        as_of_date=run_date,
        if_exists=config.if_exists,
        dry_run=config.dry_run,
    )
    run_pull_wrds_returns_stage(config=config)
    run_build_portfolio_returns_stage(config=config)
    run_publish_sheets_stage(config=config, as_of_date=run_date, skip_unchanged=False)

    manifest_path = config.paths.manifests_dir / f"daily_returns_{run_date.isoformat()}.json"
    log_path = log_path_for_date(config, run_date=run_date)

    return DailyReturnsArtifacts(
        universe_path=config.paths.universe_latest_path,
        security_returns_path=config.paths.security_returns_path,
        portfolio_returns_path=config.paths.portfolio_returns_path,
        daily_qa_path=config.paths.daily_qa_path,
        manifest_path=manifest_path,
        log_path=log_path,
    )


# Purpose: Parse run-all CLI arguments for optional wrapper orchestration.
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Daily returns orchestration wrapper.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_all = subparsers.add_parser("run-all", help="Run all daily pipeline stages in order")
    run_all.add_argument("--config", default="config/returns_daily.yaml", help="Daily config path")
    run_all.add_argument(
        "--start-date",
        default=None,
        help="Override pull window start date (YYYY-MM-DD)",
    )
    run_all.add_argument(
        "--end-date",
        default=None,
        help="Override pull window end date (YYYY-MM-DD)",
    )
    run_all.add_argument(
        "--if-exists",
        choices=["replace", "skip", "error"],
        default=None,
        help="Override output write mode",
    )
    run_all.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate stages without mutating outputs",
    )

    return parser.parse_args()


# Purpose: Apply CLI overrides onto loaded config while keeping immutable dataclass usage.
def _with_cli_overrides(config: DailyReturnsConfig, *, args: argparse.Namespace) -> DailyReturnsConfig:
    if_exists = parse_write_mode(args.if_exists) if args.if_exists else config.if_exists
    dry_run = bool(args.dry_run) if bool(args.dry_run) else config.dry_run
    start_date = args.start_date if isinstance(args.start_date, str) and args.start_date else config.start_date
    end_date = args.end_date if isinstance(args.end_date, str) and args.end_date else config.end_date

    return DailyReturnsConfig(
        start_date=start_date,
        end_date=end_date,
        google_sheets_config_path=config.google_sheets_config_path,
        positions_config_path=config.positions_config_path,
        paths=config.paths,
        wrds_username=config.wrds_username,
        if_exists=if_exists,
        dry_run=dry_run,
    )


# Purpose: CLI entrypoint for optional run-all wrapper command.
def main() -> None:
    """Execute the orchestration CLI.

    Inputs:
      - CLI args selecting subcommand and config overrides.

    Returns:
      - None.

    Raises:
      - Propagates stage/config errors.

    Notes on units:
      - Units are enforced in stage modules.
    """

    args = _parse_args()
    if args.command != "run-all":
        raise ValueError(f"Unsupported command: {args.command}")

    config = load_daily_returns_config(args.config)
    config = _with_cli_overrides(config, args=args)
    print(f"Effective date window: start_date={config.start_date} end_date={config.end_date}")
    artifacts = run_daily_returns_pipeline(config)

    print("Daily returns pipeline complete")
    print(f"universe_path={artifacts.universe_path}")
    print(f"security_returns_path={artifacts.security_returns_path}")
    print(f"portfolio_returns_path={artifacts.portfolio_returns_path}")
    print(f"daily_qa_path={artifacts.daily_qa_path}")
    print(f"manifest_path={artifacts.manifest_path}")
    print(f"log_path={artifacts.log_path}")


if __name__ == "__main__":
    main()
