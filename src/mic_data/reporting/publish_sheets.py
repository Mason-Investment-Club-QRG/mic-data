# Purpose: Publish validated daily datasets to configured Google Sheets tabs with optional hash-based skip behavior.

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

import gspread
import pandas as pd
from google.oauth2.service_account import Credentials

# Purpose: Make package imports work when this file is executed directly by path.
if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mic_data.config.google_sheets import GoogleSheetsConfig, load_google_sheets_config
from mic_data.config.secrets import require_path_env
from mic_data.contracts.daily_returns_contracts import (
    QaRow,
    validate_daily_qa,
    validate_portfolio_returns_daily,
    validate_security_returns_daily,
)
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
from mic_data.utils.idempotent_io import canonicalize_frame, dataframe_hash, stage_lock, write_manifest_json


# Purpose: Apply CLI overrides onto immutable daily config for publish-only runs.
def _apply_overrides(
    config: DailyReturnsConfig,
    *,
    if_exists: str | None,
    dry_run: bool,
) -> DailyReturnsConfig:
    resolved_if_exists = parse_write_mode(if_exists) if if_exists else config.if_exists
    resolved_dry_run = True if dry_run else config.dry_run

    return DailyReturnsConfig(
        start_date=config.start_date,
        end_date=config.end_date,
        google_sheets_config_path=config.google_sheets_config_path,
        positions_config_path=config.positions_config_path,
        paths=config.paths,
        wrds_username=config.wrds_username,
        if_exists=resolved_if_exists,
        dry_run=resolved_dry_run,
    )


# Purpose: Build authenticated Google Sheets client and expose service-account principal for error guidance.
def _build_client(sheets_cfg: GoogleSheetsConfig) -> tuple[gspread.Client, str]:
    creds_path = str(require_path_env(sheets_cfg.credentials_env_var))

    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    creds = Credentials.from_service_account_file(creds_path, scopes=scopes)
    principal = str(getattr(creds, "service_account_email", "unknown-service-account"))
    return gspread.authorize(creds), principal


# Purpose: Ensure a worksheet exists, creating it if missing so stage runs stay idempotent.
def _ensure_worksheet(spreadsheet: gspread.Spreadsheet, *, title: str) -> gspread.Worksheet:
    try:
        return spreadsheet.worksheet(title)
    except gspread.WorksheetNotFound:
        return spreadsheet.add_worksheet(title=title, rows=1000, cols=20)


# Purpose: Convert dataframe values into sheet-friendly scalar strings while preserving headers.
def _sheet_rows(df: pd.DataFrame) -> list[list[str]]:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = pd.to_datetime(out[col], utc=False).dt.strftime("%Y-%m-%d %H:%M:%S")
        else:
            out[col] = out[col].astype("string").fillna("")

    header = [str(c) for c in out.columns]
    rows = out.astype(str).values.tolist()
    return [header, *rows]


# Purpose: Replace entire worksheet contents with header + full dataset rows.
def _replace_worksheet(
    worksheet: gspread.Worksheet,
    *,
    data: pd.DataFrame,
    clear_before_write: bool,
    max_rows: int,
) -> None:
    if len(data) > max_rows:
        raise ValueError(
            "Dataset row count "
            f"{len(data)} exceeds google_sheets.options.max_rows={max_rows}. "
            "Increase `google_sheets.options.max_rows` in config/google_sheets.yaml "
            "or narrow the date window in config/returns_daily.yaml."
        )

    rows = _sheet_rows(data)
    if clear_before_write:
        worksheet.clear()

    end_col = len(rows[0])
    end_row = len(rows)
    range_label = f"A1:{gspread.utils.rowcol_to_a1(end_row, end_col)}"
    worksheet.update(range_label, rows, value_input_option="RAW")


# Purpose: Load validated stage outputs that are expected to be published.
def _load_publish_frames(config: DailyReturnsConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    security_path = config.paths.security_returns_path
    portfolio_path = config.paths.portfolio_returns_path
    qa_path = config.paths.daily_qa_path

    if not security_path.exists():
        raise FileNotFoundError(
            f"Missing security returns dataset: {security_path}. Run pull stage first."
        )
    if not portfolio_path.exists():
        raise FileNotFoundError(
            f"Missing portfolio returns dataset: {portfolio_path}. Run build stage first."
        )
    if not qa_path.exists():
        raise FileNotFoundError(
            f"Missing daily QA dataset: {qa_path}. Run prior stages first."
        )

    security = validate_security_returns_daily(pd.read_parquet(security_path))
    portfolio = validate_portfolio_returns_daily(pd.read_parquet(portfolio_path))
    qa = validate_daily_qa(pd.read_parquet(qa_path))
    return security, portfolio, qa


# Purpose: Resolve persistent hash-state file used by skip-unchanged publish behavior.
def _publish_state_path(config: DailyReturnsConfig) -> Path:
    return config.paths.manifests_dir / "sheets_publish_state.json"


# Purpose: Load previous publish hashes so unchanged tabs can be skipped safely.
def _load_publish_state(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, dict):
        return {}

    return {str(k): str(v) for k, v in payload.items()}


# Purpose: Save current publish hashes to support future skip-unchanged runs.
def _save_publish_state(path: Path, *, state: dict[str, str], dry_run: bool) -> None:
    if dry_run:
        return

    write_manifest_json(path=path, payload=state, mode="replace")


# Purpose: Compute deterministic dataset hash for publish idempotency decisions.
def _dataset_hash(df: pd.DataFrame, *, sort_by: list[str]) -> str:
    normalized = canonicalize_frame(df, sort_by=sort_by)
    return dataframe_hash(normalized)


# Purpose: Execute publish stage with optional unchanged-hash skipping and QA/log updates.
def run_publish_sheets_stage(
    *,
    config: DailyReturnsConfig,
    as_of_date: date,
    skip_unchanged: bool,
) -> dict[str, bool]:
    """Publish validated outputs to Google Sheets tabs.

    Inputs:
      - config: Daily pipeline config.
      - as_of_date: Run date used for QA/manifests.
      - skip_unchanged: Skip tab update when hash matches previous publish state.

    Returns:
      - Mapping of tab names to whether data was written.

    Raises:
      - RuntimeError/ValueError for Google API, validation, or row-limit failures.

    Notes on units:
      - Financial values remain decimals in sheet output.
    """

    stage_name = "publish_sheets"
    lock_path = stage_lock_path(config, stage_name=stage_name)

    try:
        with stage_lock(lock_path):
            sheets_cfg = load_google_sheets_config(config.google_sheets_config_path)
            security, portfolio, qa = _load_publish_frames(config)

            qa_for_day = qa[qa["run_date"] == as_of_date].copy()

            hashes = {
                sheets_cfg.outputs.security_returns_tab: _dataset_hash(
                    security,
                    sort_by=["trade_date", "ticker", "permno"],
                ),
                sheets_cfg.outputs.portfolio_returns_tab: _dataset_hash(
                    portfolio,
                    sort_by=["trade_date"],
                ),
                sheets_cfg.outputs.qa_tab: _dataset_hash(
                    qa_for_day,
                    sort_by=["run_date", "stage"],
                ),
            }

            state_path = _publish_state_path(config)
            previous_state = _load_publish_state(state_path)
            writes: dict[str, bool] = {tab: True for tab in hashes}

            if skip_unchanged:
                for tab_name, digest in hashes.items():
                    if previous_state.get(tab_name) == digest:
                        writes[tab_name] = False

            if not config.dry_run and any(writes.values()):
                client, principal = _build_client(sheets_cfg)
                spreadsheet = client.open_by_key(sheets_cfg.sheet_id)
                try:
                    if writes[sheets_cfg.outputs.security_returns_tab]:
                        ws = _ensure_worksheet(
                            spreadsheet,
                            title=sheets_cfg.outputs.security_returns_tab,
                        )
                        _replace_worksheet(
                            ws,
                            data=security,
                            clear_before_write=sheets_cfg.options.clear_before_write,
                            max_rows=sheets_cfg.options.max_rows,
                        )

                    if writes[sheets_cfg.outputs.portfolio_returns_tab]:
                        ws = _ensure_worksheet(
                            spreadsheet,
                            title=sheets_cfg.outputs.portfolio_returns_tab,
                        )
                        _replace_worksheet(
                            ws,
                            data=portfolio,
                            clear_before_write=sheets_cfg.options.clear_before_write,
                            max_rows=sheets_cfg.options.max_rows,
                        )

                    if writes[sheets_cfg.outputs.qa_tab]:
                        ws = _ensure_worksheet(spreadsheet, title=sheets_cfg.outputs.qa_tab)
                        _replace_worksheet(
                            ws,
                            data=qa_for_day,
                            clear_before_write=sheets_cfg.options.clear_before_write,
                            max_rows=sheets_cfg.options.max_rows,
                        )
                except gspread.exceptions.APIError as exc:
                    status_code = getattr(getattr(exc, "response", None), "status_code", None)
                    if status_code == 403:
                        raise RuntimeError(
                            "Google Sheets publish failed with 403 permission denied. "
                            f"Share sheet '{sheets_cfg.sheet_id}' with service account "
                            f"'{principal}' as Editor. "
                            "Also verify the credentials file in GOOGLE_APPLICATION_CREDENTIALS "
                            "matches that service account."
                        ) from exc
                    raise

            current_state = previous_state | hashes
            _save_publish_state(state_path, state=current_state, dry_run=config.dry_run)

            qa_row = QaRow(
                run_date=as_of_date,
                stage=stage_name,
                rows_universe=0,
                rows_returns=len(security),
                missing_permno=0,
                missing_ret=int(security["ret"].isna().sum()),
                status="ok",
                error_message=None,
            )
            upsert_daily_qa_row(config=config, qa_row=qa_row)

            write_daily_manifest(
                config=config,
                as_of_date=as_of_date,
                datasets={
                    "publish_sheets": {
                        "writes": writes,
                        "hashes": hashes,
                        "skip_unchanged": skip_unchanged,
                        "dry_run": config.dry_run,
                    }
                },
            )

            append_stage_log(
                config=config,
                stage=stage_name,
                status="ok",
                row_count=len(security),
                detail={
                    "writes": writes,
                    "skip_unchanged": skip_unchanged,
                    "dry_run": config.dry_run,
                },
            )
            return writes
    except Exception as exc:
        append_stage_log(
            config=config,
            stage=stage_name,
            status="error",
            row_count=None,
            detail={"error_type": type(exc).__name__, "error_message": str(exc)},
        )
        qa_row = QaRow(
            run_date=as_of_date,
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


# Purpose: Parse CLI args for publish stage.
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish daily pipeline outputs to Google Sheets.")
    parser.add_argument("--config", default="config/returns_daily.yaml", help="Daily returns config")
    parser.add_argument(
        "--as-of-date",
        default=None,
        help="Run date for QA filtering (YYYY-MM-DD, defaults to today)",
    )
    parser.add_argument(
        "--if-exists",
        choices=["replace", "skip", "error"],
        default=None,
        help="Override write mode used by metadata outputs",
    )
    parser.add_argument(
        "--skip-unchanged",
        action="store_true",
        help="Skip tab writes where dataset hash is unchanged from previous publish",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate publish payloads without API writes")
    return parser.parse_args()


# Purpose: CLI entrypoint for publish stage execution.
def main() -> None:
    """Run publish stage CLI.

    Inputs:
      - CLI args from `_parse_args`.

    Returns:
      - None.

    Raises:
      - Propagates config and publish-stage execution errors.

    Notes on units:
      - Published returns remain decimal values.
    """

    args = _parse_args()
    config = load_daily_returns_config(args.config)
    config = _apply_overrides(
        config,
        if_exists=args.if_exists,
        dry_run=bool(args.dry_run),
    )

    run_date = date.fromisoformat(args.as_of_date) if args.as_of_date else pipeline_run_date()
    writes = run_publish_sheets_stage(
        config=config,
        as_of_date=run_date,
        skip_unchanged=bool(args.skip_unchanged),
    )

    print("Sheets publish complete")
    print(f"writes={writes}")


if __name__ == "__main__":
    main()
