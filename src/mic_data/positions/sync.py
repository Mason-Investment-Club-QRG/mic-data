from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

import gspread
import pandas as pd
import yaml
from google.oauth2.service_account import Credentials


# =========================
# Config models
# =========================


@dataclass(frozen=True)
class SheetsConfig:
    sheet_id: str
    tab_name: str


@dataclass(frozen=True)
class PositionsSyncConfig:
    sheets: SheetsConfig
    mapping: Dict[str, str]
    processed_path: Path
    raw_dir: Path
    include_as_of: bool = True


# =========================
# Load config
# =========================


def load_config(path: str | Path) -> PositionsSyncConfig:
    """
    Expected config/positions.yaml shape:

    google_sheets:
      sheet_id: "..."
      tab_name: "Holdings"

    mapping:
      ticker: "TICKER"
      shares: "SHARES"
      name: "STOCK"      # optional
      sector: "SECTOR"   # optional

    outputs:
      processed_path: "data/processed/positions_latest.csv"
      raw_dir: "data/raw/positions"   # <-- recommended (see note below)

    options:
      include_as_of: true
    """
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    sheets = SheetsConfig(
        sheet_id=cfg["google_sheets"]["sheet_id"],
        tab_name=cfg["google_sheets"]["tab_name"],
    )

    processed_path = Path(cfg["outputs"]["processed_path"])
    raw_dir = Path(cfg["outputs"]["raw_dir"])

    include_as_of = bool(cfg.get("options", {}).get("include_as_of", True))

    return PositionsSyncConfig(
        sheets=sheets,
        mapping=cfg["mapping"],
        processed_path=processed_path,
        raw_dir=raw_dir,
        include_as_of=include_as_of,
    )


# =========================
# Google Sheets fetch
# =========================


def _get_creds_path() -> str:
    creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if not creds_path:
        raise RuntimeError(
            "GOOGLE_APPLICATION_CREDENTIALS is not set.\n"
            "Example:\n"
            '  export GOOGLE_APPLICATION_CREDENTIALS="secrets/<key>.json"\n'
            "Or put it in scripts/env.sh and source it."
        )
    return creds_path


def fetch_sheet_values(sheets: SheetsConfig) -> List[List[str]]:
    creds_path = _get_creds_path()
    scopes = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
    creds = Credentials.from_service_account_file(creds_path, scopes=scopes)

    gc = gspread.authorize(creds)
    sh = gc.open_by_key(sheets.sheet_id)
    ws = sh.worksheet(sheets.tab_name)
    return ws.get_all_values()


# =========================
# Transform + validate
# =========================


def values_to_df(values: List[List[str]]) -> pd.DataFrame:
    if not values or len(values) < 2:
        raise ValueError("Sheet appears empty (no header + rows).")
    header = [h.strip() for h in values[0]]
    rows = values[1:]
    return pd.DataFrame(rows, columns=header)


def canonicalize_positions(
    df_raw: pd.DataFrame, mapping: Dict[str, str], *, as_of: Optional[str]
) -> pd.DataFrame:
    """
    Canonical positions schema (minimum):
      - ticker: str (upper)
      - shares: float
    Optional:
      - name: str
      - sector: str
      - as_of: YYYY-MM-DD
    """
    required = ["ticker", "shares"]
    missing = [k for k in required if k not in mapping]
    if missing:
        raise ValueError(f"Config mapping missing required keys: {missing}")

    # Ensure referenced sheet columns exist
    referenced_cols = [mapping[k] for k in required if k in mapping]
    for opt in ("name", "sector"):
        if opt in mapping:
            referenced_cols.append(mapping[opt])

    missing_cols = [c for c in referenced_cols if c not in df_raw.columns]
    if missing_cols:
        raise ValueError(
            "Sheet is missing expected column(s) referenced in config mapping: "
            f"{missing_cols}. Available columns: {list(df_raw.columns)}"
        )

    out = pd.DataFrame()
    out["ticker"] = df_raw[mapping["ticker"]]
    out["shares"] = df_raw[mapping["shares"]]

    if "name" in mapping and mapping["name"] in df_raw.columns:
        out["name"] = df_raw[mapping["name"]]
    if "sector" in mapping and mapping["sector"] in df_raw.columns:
        out["sector"] = df_raw[mapping["sector"]]

    # Clean
    out["ticker"] = out["ticker"].astype(str).str.strip().str.upper()
    out["shares"] = (
        out["shares"].astype(str).str.replace(",", "", regex=False).str.strip()
    )
    out["shares"] = pd.to_numeric(out["shares"], errors="raise")

    # Drop empty trailing rows
    out = out[out["ticker"] != ""].copy()

    # Add as_of if requested
    if as_of is not None:
        out.insert(0, "as_of", as_of)

    return out


def validate_positions(df: pd.DataFrame) -> None:
    if df.empty:
        raise ValueError("Positions dataset is empty after cleaning.")

    if "ticker" not in df.columns or "shares" not in df.columns:
        raise ValueError("Positions dataset must include 'ticker' and 'shares'.")

    if df["ticker"].isna().any() or (df["ticker"].astype(str).str.len() == 0).any():
        raise ValueError("Found empty ticker(s) after cleaning.")

    dupes = df["ticker"][df["ticker"].duplicated()].unique().tolist()
    if dupes:
        raise ValueError(f"Duplicate tickers found: {dupes}")

    if (df["shares"] < 0).any():
        bad = df[df["shares"] < 0][["ticker", "shares"]].to_dict("records")
        raise ValueError(f"Negative shares found: {bad}")


# =========================
# Write outputs
# =========================


def write_outputs(df: pd.DataFrame, *, processed_path: Path, raw_dir: Path) -> None:
    """
    Writes:
      - raw snapshot: <raw_dir>/positions_YYYY-MM-DD.csv
      - processed latest: <processed_path>
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_path.parent.mkdir(parents=True, exist_ok=True)

    stamp = date.today().isoformat()
    raw_path = raw_dir / f"positions_{stamp}.csv"

    df.to_csv(raw_path, index=False)
    df.to_csv(processed_path, index=False)

    print(f"Wrote raw snapshot: {raw_path}")
    print(f"Wrote processed latest: {processed_path}")


# =========================
# Entry point
# =========================


def main() -> None:
    cfg = load_config("config/positions.yaml")

    values = fetch_sheet_values(cfg.sheets)
    df_raw = values_to_df(values)

    as_of = date.today().isoformat() if cfg.include_as_of else None
    df = canonicalize_positions(df_raw, cfg.mapping, as_of=as_of)
    validate_positions(df)

    write_outputs(df, processed_path=cfg.processed_path, raw_dir=cfg.raw_dir)


if __name__ == "__main__":
    main()
