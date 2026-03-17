# Purpose: Pull holdings rows from Google Sheets, normalize them, and write idempotent position snapshots.

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import gspread
import pandas as pd
import yaml
from google.oauth2.service_account import Credentials

# Purpose: Make package imports work when this file is executed directly by path.
if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mic_data.config.google_sheets import GoogleSheetsConfig, load_google_sheets_config
from mic_data.config.secrets import require_path_env
from mic_data.contracts.daily_returns_contracts import WriteMode
from mic_data.utils.idempotent_io import atomic_write_csv, stage_lock


@dataclass(frozen=True)
class HoldingsMapping:
    """Column mappings for holdings tab ingestion."""

    ticker: str
    shares: str
    name: str | None
    sector: str | None


@dataclass(frozen=True)
class PositionsOutputs:
    """Output destinations for holdings snapshots."""

    processed_path: Path
    raw_dir: Path


@dataclass(frozen=True)
class PositionsSyncConfig:
    """Configuration for holdings sync stage.

    Inputs:
      - google_sheets_config_path: Path to unified Google Sheets config.
      - mapping: Column mapping for holdings tab.
      - outputs: Processed/latest and raw snapshot paths.
      - include_as_of: Whether to include as_of_date column in outputs.

    Returns:
      - Immutable config object for stage execution.

    Raises:
      - None. Validation occurs in loader and execution.

    Notes on units:
      - Shares remain raw counts in output files.
    """

    google_sheets_config_path: Path
    holdings_mapping: HoldingsMapping
    outputs: PositionsOutputs
    include_as_of: bool = True


# Purpose: Validate mapping sections in YAML and return concrete dictionary objects.
def _require_mapping(value: Any, *, section_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"Section '{section_name}' must be a mapping.")
    return value


# Purpose: Read required non-empty string fields from config mappings.
def _require_string(mapping: dict[str, Any], *, key: str, section_name: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Section '{section_name}' requires non-empty string '{key}'.")
    return value.strip()


# Purpose: Read optional string field and normalize blank values to None.
def _optional_string(mapping: dict[str, Any], *, key: str) -> str | None:
    value = mapping.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"Optional key '{key}' must be a string when provided.")
    stripped = value.strip()
    return stripped if stripped else None


# Purpose: Load local stage config that references centralized Google Sheets settings.
def load_config(path: str | Path) -> PositionsSyncConfig:
    """Load positions sync YAML config.

    Inputs:
      - path: YAML path containing mapping/outputs/options sections.

    Returns:
      - PositionsSyncConfig for holdings sync.

    Raises:
      - FileNotFoundError if config path does not exist.
      - ValueError if required sections/keys are malformed.

    Notes on units:
      - Shares are treated as counts.
    """

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Positions config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    root = _require_mapping(raw, section_name="root")
    mapping_root = _require_mapping(root.get("mapping"), section_name="mapping")
    mapping_section = _require_mapping(
        mapping_root.get("holdings"),
        section_name="mapping.holdings",
    )
    outputs_section = _require_mapping(root.get("outputs"), section_name="outputs")
    options_section = _require_mapping(root.get("options", {}), section_name="options")

    google_path = root.get("google_sheets_config_path", "config/google_sheets.yaml")
    if not isinstance(google_path, str) or not google_path.strip():
        raise ValueError("google_sheets_config_path must be a non-empty string.")

    include_as_of_raw = options_section.get("include_as_of", True)
    if not isinstance(include_as_of_raw, bool):
        raise ValueError("options.include_as_of must be boolean.")

    return PositionsSyncConfig(
        google_sheets_config_path=Path(google_path),
        holdings_mapping=HoldingsMapping(
            ticker=_require_string(
                mapping_section,
                key="ticker",
                section_name="mapping.holdings",
            ),
            shares=_require_string(
                mapping_section,
                key="shares",
                section_name="mapping.holdings",
            ),
            name=_optional_string(mapping_section, key="name"),
            sector=_optional_string(mapping_section, key="sector"),
        ),
        outputs=PositionsOutputs(
            processed_path=Path(
                _require_string(
                    outputs_section, key="processed_path", section_name="outputs"
                )
            ),
            raw_dir=Path(
                _require_string(outputs_section, key="raw_dir", section_name="outputs")
            ),
        ),
        include_as_of=include_as_of_raw,
    )


# Purpose: Resolve credentials path from environment configured in the central Google Sheets config.
def _get_credentials_path(sheets_cfg: GoogleSheetsConfig) -> str:
    env_var = sheets_cfg.credentials_env_var
    return str(require_path_env(env_var))


# Purpose: Build authenticated gspread client from service-account credentials.
def _build_client(sheets_cfg: GoogleSheetsConfig) -> gspread.Client:
    creds_path = _get_credentials_path(sheets_cfg)
    scopes = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
    creds = Credentials.from_service_account_file(creds_path, scopes=scopes)
    return gspread.authorize(creds)


# Purpose: Pull all values from the configured holdings tab in Google Sheets.
def fetch_holdings_values(sheets_cfg: GoogleSheetsConfig) -> list[list[str]]:
    """Fetch holdings tab values from Google Sheets.

    Inputs:
      - sheets_cfg: Centralized sheets config with tab names and sheet ID.

    Returns:
      - Matrix of string values with header row first.

    Raises:
      - RuntimeError for credential issues.
      - gspread exceptions for sheet access failures.

    Notes on units:
      - Raw values are textual; units are interpreted downstream.
    """

    client = _build_client(sheets_cfg)
    spreadsheet = client.open_by_key(sheets_cfg.sheet_id)
    worksheet = spreadsheet.worksheet(sheets_cfg.inputs.holdings_tab)
    return worksheet.get_all_values()


# Purpose: Convert raw sheet values into a dataframe using first row as header.
def values_to_df(values: list[list[str]]) -> pd.DataFrame:
    """Transform sheet matrix into dataframe.

    Inputs:
      - values: Raw worksheet matrix.

    Returns:
      - Dataframe where first row is used as column headers.

    Raises:
      - ValueError when worksheet is empty.

    Notes on units:
      - String values are left untyped at this stage.
    """

    if not values or len(values) < 2:
        raise ValueError(
            "Holdings tab appears empty (requires header + at least one row)."
        )

    header = [str(h).strip() for h in values[0]]
    rows = values[1:]
    return pd.DataFrame(rows, columns=header)


# Purpose: Normalize holdings rows to canonical schema expected by downstream stages.
def canonicalize_positions(
    df_raw: pd.DataFrame,
    *,
    holdings_mapping: HoldingsMapping,
    as_of_date: date | None,
) -> pd.DataFrame:
    """Map and clean holdings data to canonical positions schema.

    Inputs:
      - df_raw: Raw holdings dataframe from Google Sheets.
      - mapping: Column mapping for ticker/shares plus optional metadata.
      - as_of_date: Optional snapshot date for provenance.

    Returns:
      - Clean dataframe with columns: as_of_date, ticker, shares, name, sector.

    Raises:
      - ValueError for missing mapped columns or invalid numeric shares.

    Notes on units:
      - `shares` is numeric share count.
    """

    required_sheet_cols = [holdings_mapping.ticker, holdings_mapping.shares]
    if holdings_mapping.name is not None:
        required_sheet_cols.append(holdings_mapping.name)
    if holdings_mapping.sector is not None:
        required_sheet_cols.append(holdings_mapping.sector)

    missing_cols = [c for c in required_sheet_cols if c not in df_raw.columns]
    if missing_cols:
        raise ValueError(
            "Holdings tab missing expected columns from mapping: "
            f"{missing_cols}. Available columns: {list(df_raw.columns)}"
        )

    out = pd.DataFrame()
    if as_of_date is not None:
        out["as_of_date"] = pd.Series(
            [as_of_date.isoformat()] * len(df_raw), dtype="string"
        )
    out["ticker"] = (
        df_raw[holdings_mapping.ticker].astype("string").str.strip().str.upper()
    )

    shares_text = (
        df_raw[holdings_mapping.shares]
        .astype("string")
        .str.replace(",", "", regex=False)
        .str.strip()
    )
    out["shares"] = pd.to_numeric(shares_text, errors="raise").astype("float64")

    out["name"] = (
        df_raw[holdings_mapping.name].astype("string").str.strip()
        if holdings_mapping.name is not None
        else pd.Series([pd.NA] * len(df_raw), dtype="string")
    )
    out["sector"] = (
        df_raw[holdings_mapping.sector].astype("string").str.strip()
        if holdings_mapping.sector is not None
        else pd.Series([pd.NA] * len(df_raw), dtype="string")
    )

    out = out[out["ticker"].notna() & (out["ticker"].str.len() > 0)].copy()
    return out


# Purpose: Validate holdings rows for duplicate tickers and invalid share values.
def validate_positions(df: pd.DataFrame) -> None:
    """Validate canonical positions dataframe.

    Inputs:
      - df: Canonical holdings dataframe.

    Returns:
      - None.

    Raises:
      - ValueError for empty dataset, duplicate tickers, or negative shares.

    Notes on units:
      - Shares are validated as non-negative counts.
    """

    if df.empty:
        raise ValueError("Positions dataframe is empty after normalization.")

    duplicates = df["ticker"][df["ticker"].duplicated()].unique().tolist()
    if duplicates:
        raise ValueError(f"Duplicate holdings tickers found: {duplicates}")

    if (df["shares"] < 0).any():
        bad_rows = df.loc[df["shares"] < 0, ["ticker", "shares"]].to_dict("records")
        raise ValueError(f"Negative share counts found: {bad_rows}")


# Purpose: Persist raw and processed holdings files with deterministic sorting and write-mode control.
def write_outputs(
    df: pd.DataFrame,
    *,
    outputs: PositionsOutputs,
    as_of_date: date,
    if_exists: WriteMode,
    dry_run: bool,
) -> tuple[Path, Path, bool]:
    """Write holdings artifacts for raw snapshot and latest processed outputs.

    Inputs:
      - df: Validated holdings dataframe.
      - outputs: Target file paths.
      - as_of_date: Snapshot date used in raw filename.
      - if_exists: Existing-file behavior.
      - dry_run: When true, skip file writes.

    Returns:
      - Tuple `(raw_path, processed_path, wrote_anything)`.

    Raises:
      - OSError/FileExistsError for write mode conflicts or filesystem failures.

    Notes on units:
      - Values are persisted without unit transformation.
    """

    raw_path = outputs.raw_dir / f"positions_{as_of_date.isoformat()}.csv"
    processed_path = outputs.processed_path

    if dry_run:
        return raw_path, processed_path, False

    atomic_write_csv(df, path=raw_path, sort_by=["ticker"], mode=if_exists)
    atomic_write_csv(df, path=processed_path, sort_by=["ticker"], mode=if_exists)
    return raw_path, processed_path, True


# Purpose: Execute the holdings sync stage end-to-end with lock protection.
def run_positions_sync_stage(
    *,
    config_path: str | Path = "config/positions.yaml",
    as_of_date: date | None = None,
    if_exists: WriteMode = "replace",
    dry_run: bool = False,
) -> pd.DataFrame:
    """Run holdings sync pipeline stage.

    Inputs:
      - config_path: Positions stage config path.
      - as_of_date: Optional snapshot date override.
      - if_exists: Existing-file write behavior.
      - dry_run: Validate and print without file writes.

    Returns:
      - Canonical holdings dataframe.

    Raises:
      - RuntimeError/ValueError for config, sheet, validation, or lock failures.

    Notes on units:
      - Shares remain as numeric counts.
    """

    cfg = load_config(config_path)
    sheets_cfg = load_google_sheets_config(cfg.google_sheets_config_path)
    snapshot_date = as_of_date or date.today()

    lock_path = Path("outputs/locks") / "positions_sync.lock"
    with stage_lock(lock_path):
        values = fetch_holdings_values(sheets_cfg)
        df_raw = values_to_df(values)
        as_of = snapshot_date if cfg.include_as_of else None
        canonical = canonicalize_positions(
            df_raw,
            holdings_mapping=cfg.holdings_mapping,
            as_of_date=as_of,
        )
        validate_positions(canonical)

        raw_path, processed_path, wrote = write_outputs(
            canonical,
            outputs=cfg.outputs,
            as_of_date=snapshot_date,
            if_exists=if_exists,
            dry_run=dry_run,
        )

    mode_label = "DRY_RUN" if dry_run else ("WROTE" if wrote else "SKIPPED")
    print(
        f"[{mode_label}] holdings_rows={len(canonical)} raw_path={raw_path} processed_path={processed_path}"
    )
    return canonical


# Purpose: Parse CLI arguments for isolated stage execution and idempotent behaviors.
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sync holdings tab from Google Sheets."
    )
    parser.add_argument(
        "--config", default="config/positions.yaml", help="Positions config path"
    )
    parser.add_argument(
        "--as-of-date",
        default=None,
        help="Snapshot date in YYYY-MM-DD (defaults to today)",
    )
    parser.add_argument(
        "--if-exists",
        choices=["replace", "skip", "error"],
        default="replace",
        help="Write behavior when destination files already exist",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print row counts without writing files",
    )
    return parser.parse_args()


# Purpose: CLI entrypoint for holdings sync stage.
def main() -> None:
    """Run holdings sync CLI.

    Inputs:
      - CLI args from `_parse_args`.

    Returns:
      - None.

    Raises:
      - ValueError for invalid argument formats.
      - Propagates stage execution errors.

    Notes on units:
      - Shares are preserved as numeric counts.
    """

    args = _parse_args()
    parsed_date = date.fromisoformat(args.as_of_date) if args.as_of_date else None
    run_positions_sync_stage(
        config_path=args.config,
        as_of_date=parsed_date,
        if_exists=args.if_exists,
        dry_run=bool(args.dry_run),
    )


if __name__ == "__main__":
    main()
