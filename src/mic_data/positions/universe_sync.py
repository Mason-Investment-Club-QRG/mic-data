# Purpose: Build a canonical holdings+watchlist universe dataset from Google Sheets with idempotent writes.

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
from mic_data.contracts.daily_returns_contracts import (
    UNIVERSE_COLUMNS,
    WriteMode,
    validate_universe_daily,
)
from mic_data.utils.idempotent_io import atomic_write_parquet, stage_lock


@dataclass(frozen=True)
class HoldingsMapping:
    """Column mapping for holdings tab inputs."""

    ticker: str
    shares: str
    name: str | None
    sector: str | None


@dataclass(frozen=True)
class WatchlistMapping:
    """Column mapping for watchlist tab inputs."""

    ticker: str
    name: str | None
    sector: str | None


@dataclass(frozen=True)
class UniverseOutputs:
    """Output destinations for universe datasets."""

    universe_latest_path: Path
    universe_snapshot_dir: Path


@dataclass(frozen=True)
class UniverseSyncConfig:
    """Configuration for universe sync stage.

    Inputs:
      - google_sheets_config_path: Path to centralized sheet ID/tab config.
      - holdings_mapping: Holdings tab column mapping.
      - watchlist_mapping: Watchlist tab column mapping.
      - outputs: Universe output destinations.

    Returns:
      - Immutable config used by `run_universe_sync_stage`.

    Raises:
      - None. Validation occurs in loader and stage execution.

    Notes on units:
      - Shares remain as count units.
    """

    google_sheets_config_path: Path
    holdings_mapping: HoldingsMapping
    watchlist_mapping: WatchlistMapping
    outputs: UniverseOutputs


# Purpose: Require dictionary-like structure for typed config extraction.
def _require_mapping(value: Any, *, section_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"Section '{section_name}' must be a mapping.")
    return value


# Purpose: Read required non-empty string values from config mappings.
def _require_string(mapping: dict[str, Any], *, key: str, section_name: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Section '{section_name}' requires non-empty string '{key}'.")
    return value.strip()


# Purpose: Read optional non-empty string values from config mappings.
def _optional_string(mapping: dict[str, Any], *, key: str) -> str | None:
    value = mapping.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"Optional key '{key}' must be a string when provided.")
    stripped = value.strip()
    return stripped if stripped else None


# Purpose: Load stage config for universe sync from positions YAML.
def load_config(path: str | Path) -> UniverseSyncConfig:
    """Load universe sync configuration from YAML.

    Inputs:
      - path: Positions config path with mapping and output blocks.

    Returns:
      - UniverseSyncConfig containing mappings and output paths.

    Raises:
      - FileNotFoundError if config path does not exist.
      - ValueError for malformed or incomplete config sections.

    Notes on units:
      - Shares are expected as count units in holdings mapping.
    """

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Universe sync config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    root = _require_mapping(raw, section_name="root")
    mapping_root = _require_mapping(root.get("mapping"), section_name="mapping")
    holdings_mapping = _require_mapping(
        mapping_root.get("holdings"),
        section_name="mapping.holdings",
    )
    watchlist_mapping = _require_mapping(
        mapping_root.get("watchlist"),
        section_name="mapping.watchlist",
    )
    outputs = _require_mapping(root.get("outputs"), section_name="outputs")

    google_path = root.get("google_sheets_config_path", "config/google_sheets.yaml")
    if not isinstance(google_path, str) or not google_path.strip():
        raise ValueError("google_sheets_config_path must be a non-empty string.")

    return UniverseSyncConfig(
        google_sheets_config_path=Path(google_path),
        holdings_mapping=HoldingsMapping(
            ticker=_require_string(
                holdings_mapping,
                key="ticker",
                section_name="mapping.holdings",
            ),
            shares=_require_string(
                holdings_mapping,
                key="shares",
                section_name="mapping.holdings",
            ),
            name=_optional_string(holdings_mapping, key="name"),
            sector=_optional_string(holdings_mapping, key="sector"),
        ),
        watchlist_mapping=WatchlistMapping(
            ticker=_require_string(
                watchlist_mapping,
                key="ticker",
                section_name="mapping.watchlist",
            ),
            name=_optional_string(watchlist_mapping, key="name"),
            sector=_optional_string(watchlist_mapping, key="sector"),
        ),
        outputs=UniverseOutputs(
            universe_latest_path=Path(
                _require_string(outputs, key="universe_latest_path", section_name="outputs")
            ),
            universe_snapshot_dir=Path(
                _require_string(outputs, key="universe_snapshot_dir", section_name="outputs")
            ),
        ),
    )


# Purpose: Resolve credentials path from the centrally configured environment variable.
def _get_credentials_path(sheets_cfg: GoogleSheetsConfig) -> str:
    return str(require_path_env(sheets_cfg.credentials_env_var))


# Purpose: Build a Google Sheets client using service-account credentials.
def _build_client(sheets_cfg: GoogleSheetsConfig) -> gspread.Client:
    creds_path = _get_credentials_path(sheets_cfg)
    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    creds = Credentials.from_service_account_file(creds_path, scopes=scopes)
    return gspread.authorize(creds)


# Purpose: Fetch full worksheet data for a specific tab.
def _fetch_values(sheets_cfg: GoogleSheetsConfig, tab_name: str) -> list[list[str]]:
    client = _build_client(sheets_cfg)
    spreadsheet = client.open_by_key(sheets_cfg.sheet_id)
    worksheet = spreadsheet.worksheet(tab_name)
    return worksheet.get_all_values()


# Purpose: Convert worksheet cell matrix into dataframe with first row as headers.
def _values_to_df(values: list[list[str]], *, tab_name: str) -> pd.DataFrame:
    if not values or len(values) < 2:
        raise ValueError(f"Tab '{tab_name}' is empty (requires header and data rows).")

    header = [str(h).strip() for h in values[0]]
    return pd.DataFrame(values[1:], columns=header)


# Purpose: Canonicalize holdings rows into the `universe_daily` schema columns.
def _canonicalize_holdings(
    df_raw: pd.DataFrame,
    *,
    mapping: HoldingsMapping,
    as_of_date: date,
) -> pd.DataFrame:
    required = [mapping.ticker, mapping.shares]
    if mapping.name is not None:
        required.append(mapping.name)
    if mapping.sector is not None:
        required.append(mapping.sector)

    missing = [c for c in required if c not in df_raw.columns]
    if missing:
        raise ValueError(f"Holdings tab is missing mapped columns: {missing}")

    out = pd.DataFrame(
        {
            "as_of_date": pd.Series([as_of_date.isoformat()] * len(df_raw), dtype="string"),
            "ticker": df_raw[mapping.ticker].astype("string").str.strip().str.upper(),
            "is_holding": pd.Series([True] * len(df_raw), dtype="bool"),
            "is_watchlist": pd.Series([False] * len(df_raw), dtype="bool"),
            "shares": pd.to_numeric(
                df_raw[mapping.shares].astype("string").str.replace(",", "", regex=False).str.strip(),
                errors="raise",
            ).astype("float64"),
            "name": (
                df_raw[mapping.name].astype("string").str.strip()
                if mapping.name is not None
                else pd.Series([pd.NA] * len(df_raw), dtype="string")
            ),
            "sector": (
                df_raw[mapping.sector].astype("string").str.strip()
                if mapping.sector is not None
                else pd.Series([pd.NA] * len(df_raw), dtype="string")
            ),
        }
    )
    out = out[out["ticker"].notna() & (out["ticker"].str.len() > 0)].copy()
    return out


# Purpose: Canonicalize watchlist rows into the same `universe_daily` schema.
def _canonicalize_watchlist(
    df_raw: pd.DataFrame,
    *,
    mapping: WatchlistMapping,
    as_of_date: date,
) -> pd.DataFrame:
    required = [mapping.ticker]
    if mapping.name is not None:
        required.append(mapping.name)
    if mapping.sector is not None:
        required.append(mapping.sector)

    missing = [c for c in required if c not in df_raw.columns]
    if missing:
        raise ValueError(f"Watchlist tab is missing mapped columns: {missing}")

    out = pd.DataFrame(
        {
            "as_of_date": pd.Series([as_of_date.isoformat()] * len(df_raw), dtype="string"),
            "ticker": df_raw[mapping.ticker].astype("string").str.strip().str.upper(),
            "is_holding": pd.Series([False] * len(df_raw), dtype="bool"),
            "is_watchlist": pd.Series([True] * len(df_raw), dtype="bool"),
            "shares": pd.Series([float("nan")] * len(df_raw), dtype="float64"),
            "name": (
                df_raw[mapping.name].astype("string").str.strip()
                if mapping.name is not None
                else pd.Series([pd.NA] * len(df_raw), dtype="string")
            ),
            "sector": (
                df_raw[mapping.sector].astype("string").str.strip()
                if mapping.sector is not None
                else pd.Series([pd.NA] * len(df_raw), dtype="string")
            ),
        }
    )
    out = out[out["ticker"].notna() & (out["ticker"].str.len() > 0)].copy()
    return out


# Purpose: Merge holdings and watchlist rows while preserving provenance flags and metadata.
def merge_universe(holdings: pd.DataFrame, watchlist: pd.DataFrame) -> pd.DataFrame:
    """Merge holdings and watchlist into one canonical universe dataframe.

    Inputs:
      - holdings: Canonical holdings dataframe.
      - watchlist: Canonical watchlist dataframe.

    Returns:
      - Merged dataframe with one row per `(as_of_date, ticker)`.

    Raises:
      - ValueError if required fields are malformed.

    Notes on units:
      - Shares remain counts from holdings rows when available.
    """

    holdings_norm = holdings.copy()
    watchlist_norm = watchlist.copy()

    for frame in (holdings_norm, watchlist_norm):
        frame["as_of_date"] = frame["as_of_date"].astype("string")
        frame["ticker"] = frame["ticker"].astype("string")
        frame["is_holding"] = frame["is_holding"].astype(bool)
        frame["is_watchlist"] = frame["is_watchlist"].astype(bool)
        frame["shares"] = pd.to_numeric(frame["shares"], errors="coerce").astype("float64")
        frame["name"] = frame["name"].astype("string")
        frame["sector"] = frame["sector"].astype("string")

    combined = pd.concat([holdings_norm, watchlist_norm], ignore_index=True)

    # Combine duplicate ticker rows by preserving any true flag and preferring non-null metadata.
    grouped = combined.groupby(["as_of_date", "ticker"], as_index=False).agg(
        is_holding=("is_holding", "max"),
        is_watchlist=("is_watchlist", "max"),
        shares=("shares", "max"),
        name=("name", "first"),
        sector=("sector", "first"),
    )

    grouped = grouped[list(UNIVERSE_COLUMNS)]
    return validate_universe_daily(grouped)


# Purpose: Persist latest and dated universe snapshots in an idempotent and deterministic form.
def write_outputs(
    df: pd.DataFrame,
    *,
    outputs: UniverseOutputs,
    as_of_date: date,
    if_exists: WriteMode,
    dry_run: bool,
) -> tuple[Path, Path, bool]:
    """Write universe artifacts.

    Inputs:
      - df: Contract-validated universe dataframe.
      - outputs: Output path configuration.
      - as_of_date: Snapshot date used in snapshot filename.
      - if_exists: Existing-file mode.
      - dry_run: Skip writing when true.

    Returns:
      - `(latest_path, snapshot_path, wrote_anything)`.

    Raises:
      - OSError/FileExistsError for write failures or mode conflicts.

    Notes on units:
      - Shares are persisted unchanged.
    """

    latest_path = outputs.universe_latest_path
    snapshot_path = outputs.universe_snapshot_dir / f"universe_{as_of_date.isoformat()}.parquet"

    if dry_run:
        return latest_path, snapshot_path, False

    atomic_write_parquet(
        df,
        path=latest_path,
        sort_by=["as_of_date", "ticker"],
        mode=if_exists,
    )
    atomic_write_parquet(
        df,
        path=snapshot_path,
        sort_by=["as_of_date", "ticker"],
        mode=if_exists,
    )
    return latest_path, snapshot_path, True


# Purpose: Execute universe sync stage with locking and contract validation.
def run_universe_sync_stage(
    *,
    config_path: str | Path = "config/positions.yaml",
    as_of_date: date | None = None,
    if_exists: WriteMode = "replace",
    dry_run: bool = False,
) -> pd.DataFrame:
    """Run holdings+watchlist merge stage.

    Inputs:
      - config_path: Positions config file path.
      - as_of_date: Optional snapshot date override.
      - if_exists: Existing-file write behavior.
      - dry_run: Validate and print without writes.

    Returns:
      - Contract-validated `universe_daily` dataframe.

    Raises:
      - RuntimeError/ValueError for lock, fetch, or validation issues.

    Notes on units:
      - Shares are count units when present.
    """

    cfg = load_config(config_path)
    sheets_cfg = load_google_sheets_config(cfg.google_sheets_config_path)
    snapshot_date = as_of_date or date.today()

    lock_path = Path("outputs/locks") / "universe_sync.lock"
    with stage_lock(lock_path):
        holdings_values = _fetch_values(sheets_cfg, sheets_cfg.inputs.holdings_tab)
        watchlist_values = _fetch_values(sheets_cfg, sheets_cfg.inputs.watchlist_tab)

        holdings_raw = _values_to_df(holdings_values, tab_name=sheets_cfg.inputs.holdings_tab)
        watchlist_raw = _values_to_df(
            watchlist_values,
            tab_name=sheets_cfg.inputs.watchlist_tab,
        )

        holdings_df = _canonicalize_holdings(
            holdings_raw,
            mapping=cfg.holdings_mapping,
            as_of_date=snapshot_date,
        )
        watchlist_df = _canonicalize_watchlist(
            watchlist_raw,
            mapping=cfg.watchlist_mapping,
            as_of_date=snapshot_date,
        )

        universe = merge_universe(holdings_df, watchlist_df)
        latest_path, snapshot_path, wrote = write_outputs(
            universe,
            outputs=cfg.outputs,
            as_of_date=snapshot_date,
            if_exists=if_exists,
            dry_run=dry_run,
        )

    mode_label = "DRY_RUN" if dry_run else ("WROTE" if wrote else "SKIPPED")
    print(
        f"[{mode_label}] universe_rows={len(universe)} latest_path={latest_path} snapshot_path={snapshot_path}"
    )
    return universe


# Purpose: Parse CLI arguments for universe sync stage execution.
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build holdings+watchlist universe dataset.")
    parser.add_argument("--config", default="config/positions.yaml", help="Positions config path")
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
        help="Validate and print rows without writing output files",
    )
    return parser.parse_args()


# Purpose: CLI entrypoint for universe sync stage.
def main() -> None:
    """Run universe sync from CLI.

    Inputs:
      - CLI args from `_parse_args`.

    Returns:
      - None.

    Raises:
      - ValueError for invalid date format.
      - Propagates stage execution errors.

    Notes on units:
      - Shares remain counts.
    """

    args = _parse_args()
    parsed_date = date.fromisoformat(args.as_of_date) if args.as_of_date else None
    run_universe_sync_stage(
        config_path=args.config,
        as_of_date=parsed_date,
        if_exists=args.if_exists,
        dry_run=bool(args.dry_run),
    )


if __name__ == "__main__":
    main()
