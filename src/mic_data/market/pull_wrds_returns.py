# Purpose: Pull WRDS CRSP daily security returns for the canonical universe and persist contract-validated outputs.

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import pandas as pd

# Purpose: Make package imports work when this file is executed directly by path.
if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mic_data.contracts.daily_returns_contracts import QaRow, WriteMode, validate_universe_daily
from mic_data.market.prices_daily import (
    DailyReturnsConfig,
    append_stage_log,
    load_daily_returns_config,
    parse_write_mode,
    pipeline_run_date,
    stage_lock_path,
    upsert_daily_qa_row,
    write_daily_manifest,
)
from mic_data.market.wrds_returns_source import WrdsCrspDailyReturnSource
from mic_data.utils.idempotent_io import atomic_write_csv, atomic_write_parquet, stage_lock


# Purpose: Apply optional CLI overrides to immutable daily config.
def _apply_overrides(
    config: DailyReturnsConfig,
    *,
    start_date: str | None,
    end_date: str | None,
    if_exists: str | None,
    dry_run: bool,
) -> DailyReturnsConfig:
    resolved_if_exists = parse_write_mode(if_exists) if if_exists else config.if_exists
    resolved_dry_run = True if dry_run else config.dry_run

    return DailyReturnsConfig(
        start_date=start_date or config.start_date,
        end_date=end_date or config.end_date,
        google_sheets_config_path=config.google_sheets_config_path,
        positions_config_path=config.positions_config_path,
        paths=config.paths,
        wrds_username=config.wrds_username,
        if_exists=resolved_if_exists,
        dry_run=resolved_dry_run,
    )


# Purpose: Load and validate the canonical universe dataset required for WRDS pulls.
def _load_universe(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Universe dataset not found: {path}. Run 'python -m mic_data.positions.universe_sync' first."
        )
    universe = pd.read_parquet(path)
    return validate_universe_daily(universe)


# Purpose: Execute WRDS daily returns pull stage with idempotent writes and QA/log updates.
def run_pull_wrds_returns_stage(
    *,
    config: DailyReturnsConfig,
) -> pd.DataFrame:
    """Run WRDS pull stage.

    Inputs:
      - config: Daily pipeline config.

    Returns:
      - Contract-validated security returns dataframe.

    Raises:
      - RuntimeError/ValueError for WRDS, contract, or write failures.

    Notes on units:
      - `ret` is decimal total return from CRSP.
    """

    stage_name = "pull_wrds_returns"
    lock_path = stage_lock_path(config, stage_name=stage_name)
    run_date = pipeline_run_date()
    print(
        "Running pull stage with "
        f"start_date={config.start_date} end_date={config.end_date}"
    )

    try:
        with stage_lock(lock_path):
            universe = _load_universe(config.paths.universe_latest_path)

            source = WrdsCrspDailyReturnSource(username=config.wrds_username)
            security_returns = source.load_security_returns(
                universe=universe,
                start_date=config.start_date,
                end_date=config.end_date,
            )

            security_write = None
            security_csv_write = None
            if not config.dry_run:
                security_write = atomic_write_parquet(
                    security_returns,
                    path=config.paths.security_returns_path,
                    sort_by=["trade_date", "ticker", "permno"],
                    mode=config.if_exists,
                )
                security_csv_write = atomic_write_csv(
                    security_returns,
                    path=config.paths.security_returns_csv_path,
                    sort_by=["trade_date", "ticker", "permno"],
                    mode=config.if_exists,
                )

            universe_tickers = set(universe["ticker"].astype(str).tolist())
            returned_tickers = set(security_returns["ticker"].astype(str).tolist())
            missing_permno_count = len(universe_tickers - returned_tickers)
            missing_ret_count = int(security_returns["ret"].isna().sum())

            qa_row = QaRow(
                run_date=run_date,
                stage=stage_name,
                rows_universe=len(universe),
                rows_returns=len(security_returns),
                missing_permno=missing_permno_count,
                missing_ret=missing_ret_count,
                status="ok",
                error_message=None,
            )
            upsert_daily_qa_row(config=config, qa_row=qa_row)

            dataset_payload: dict[str, dict[str, object]] = {
                "security_returns_daily": {
                    "path": str(config.paths.security_returns_path),
                    "rows": len(security_returns),
                    "hash": security_write.content_hash if security_write else "dry_run",
                    "csv_hash": security_csv_write.content_hash if security_csv_write else "dry_run",
                }
            }
            write_daily_manifest(config=config, as_of_date=run_date, datasets=dataset_payload)

            append_stage_log(
                config=config,
                stage=stage_name,
                status="ok",
                row_count=len(security_returns),
                detail={
                    "missing_permno": missing_permno_count,
                    "missing_ret": missing_ret_count,
                    "start_date": config.start_date,
                    "end_date": config.end_date,
                    "dry_run": config.dry_run,
                },
            )
            return security_returns
    except Exception as exc:
        append_stage_log(
            config=config,
            stage=stage_name,
            status="error",
            row_count=None,
            detail={"error_type": type(exc).__name__, "error_message": str(exc)},
        )
        qa_row = QaRow(
            run_date=run_date,
            stage=stage_name,
            rows_universe=0,
            rows_returns=0,
            missing_permno=0,
            missing_ret=0,
            status="error",
            error_message=str(exc),
        )
        upsert_daily_qa_row(config=config, qa_row=qa_row)
        raise


# Purpose: Parse CLI args for WRDS pull stage.
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pull WRDS CRSP daily security returns.")
    parser.add_argument("--config", default="config/returns_daily.yaml", help="Daily returns config")
    parser.add_argument("--start-date", default=None, help="Override start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", default=None, help="Override end date (YYYY-MM-DD)")
    parser.add_argument(
        "--if-exists",
        choices=["replace", "skip", "error"],
        default=None,
        help="Write behavior for stage outputs",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate pull without writes")
    return parser.parse_args()


# Purpose: CLI entrypoint for WRDS pull stage.
def main() -> None:
    """Run WRDS pull CLI.

    Inputs:
      - CLI args from `_parse_args`.

    Returns:
      - None.

    Raises:
      - Propagates config and stage execution errors.

    Notes on units:
      - Returned `ret` values are decimals.
    """

    args = _parse_args()
    config = load_daily_returns_config(args.config)
    config = _apply_overrides(
        config,
        start_date=args.start_date,
        end_date=args.end_date,
        if_exists=args.if_exists,
        dry_run=bool(args.dry_run),
    )

    df = run_pull_wrds_returns_stage(config=config)
    print(f"WRDS pull complete rows={len(df)}")
    print(f"output_parquet={config.paths.security_returns_path}")
    print(f"output_csv={config.paths.security_returns_csv_path}")


if __name__ == "__main__":
    main()
