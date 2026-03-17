# Purpose: Define explicit schemas and validators for all daily returns pipeline datasets.

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Final, Literal, TypeAlias

import pandas as pd

WriteMode: TypeAlias = Literal["replace", "skip", "error"]

UNIVERSE_COLUMNS: Final[tuple[str, ...]] = (
    "as_of_date",
    "ticker",
    "is_holding",
    "is_watchlist",
    "shares",
    "name",
    "sector",
)
UNIVERSE_KEY: Final[tuple[str, str]] = ("as_of_date", "ticker")

SECURITY_RETURNS_COLUMNS: Final[tuple[str, ...]] = (
    "trade_date",
    "ticker",
    "permno",
    "ret",
    "prc",
    "vol",
    "shrout",
    "source",
    "load_ts_utc",
)
SECURITY_RETURNS_KEY: Final[tuple[str, str]] = ("trade_date", "permno")

PORTFOLIO_RETURNS_COLUMNS: Final[tuple[str, ...]] = (
    "trade_date",
    "portfolio_ret",
    "n_constituents",
    "gross_exposure",
    "method",
    "load_ts_utc",
)
PORTFOLIO_RETURNS_KEY: Final[tuple[str, ...]] = ("trade_date",)

DAILY_QA_COLUMNS: Final[tuple[str, ...]] = (
    "run_date",
    "stage",
    "rows_universe",
    "rows_returns",
    "missing_permno",
    "missing_ret",
    "status",
    "error_message",
)
DAILY_QA_KEY: Final[tuple[str, str]] = ("run_date", "stage")


@dataclass(frozen=True)
class QaRow:
    """Structured daily QA row aligned with `daily_qa` contract."""

    run_date: date
    stage: str
    rows_universe: int
    rows_returns: int
    missing_permno: int
    missing_ret: int
    status: str
    error_message: str | None


# Purpose: Ensure required columns exist before dataset-specific coercion and validation.
def require_columns(df: pd.DataFrame, *, required: tuple[str, ...], dataset_name: str) -> None:
    """Validate required column presence.

    Inputs:
      - df: Dataframe under validation.
      - required: Required column tuple.
      - dataset_name: Human-readable dataset identifier for errors.

    Returns:
      - None.

    Raises:
      - ValueError if one or more columns are missing.

    Notes on units:
      - Structure validation only; does not inspect financial units.
    """

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{dataset_name} missing required columns: {missing}")


# Purpose: Enforce natural-key uniqueness for idempotent datasets.
def require_unique_key(df: pd.DataFrame, *, key: tuple[str, ...], dataset_name: str) -> None:
    """Validate that contract key columns form unique rows.

    Inputs:
      - df: Dataframe under validation.
      - key: Natural-key column tuple.
      - dataset_name: Human-readable dataset identifier for errors.

    Returns:
      - None.

    Raises:
      - ValueError if duplicate key rows are detected.

    Notes on units:
      - Key checks are unitless integrity constraints.
    """

    duplicated = df.duplicated(list(key), keep=False)
    if bool(duplicated.any()):
        sample = df.loc[duplicated, list(key)].head(10).to_dict("records")
        raise ValueError(f"{dataset_name} has duplicate key rows: {sample}")


# Purpose: Normalize universe records to canonical dtypes and value conventions.
def coerce_universe_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce `universe_daily` columns to canonical types.

    Inputs:
      - df: Raw dataframe from holdings/watchlist sources.

    Returns:
      - Normalized dataframe with canonical schema and dtypes.

    Raises:
      - ValueError when required fields cannot be coerced.

    Notes on units:
      - `shares` remains raw share count.
    """

    require_columns(df, required=UNIVERSE_COLUMNS, dataset_name="universe_daily")

    out = df.copy()
    out["as_of_date"] = pd.to_datetime(out["as_of_date"], errors="raise").dt.normalize()
    out["ticker"] = out["ticker"].astype("string").str.strip().str.upper()
    out["is_holding"] = out["is_holding"].astype(bool)
    out["is_watchlist"] = out["is_watchlist"].astype(bool)
    out["shares"] = pd.to_numeric(out["shares"], errors="coerce").astype("float64")
    out["name"] = out["name"].astype("string")
    out["sector"] = out["sector"].astype("string")

    return out[list(UNIVERSE_COLUMNS)]


# Purpose: Validate business rules for canonical universe rows.
def validate_universe_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Validate the `universe_daily` dataset contract.

    Inputs:
      - df: Candidate universe dataframe.

    Returns:
      - Canonicalized dataframe.

    Raises:
      - ValueError for missing tickers, invalid rows, or duplicate keys.

    Notes on units:
      - No conversions beyond dtype normalization.
    """

    out = coerce_universe_daily(df)
    if out["ticker"].isna().any() or (out["ticker"].str.len() == 0).any():
        raise ValueError("universe_daily contains empty ticker values")

    if (~(out["is_holding"] | out["is_watchlist"])) .any():
        raise ValueError("Each universe_daily row must be holding or watchlist (or both)")

    require_unique_key(out, key=UNIVERSE_KEY, dataset_name="universe_daily")
    return out


# Purpose: Normalize WRDS security return rows to canonical schema and dtypes.
def coerce_security_returns_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce `security_returns_daily` columns to canonical types.

    Inputs:
      - df: Raw WRDS returns dataframe.

    Returns:
      - Normalized security returns dataframe.

    Raises:
      - ValueError for missing columns or coercion issues.

    Notes on units:
      - `ret` is decimal total return (0.01 = 1%).
    """

    require_columns(
        df,
        required=SECURITY_RETURNS_COLUMNS,
        dataset_name="security_returns_daily",
    )

    out = df.copy()
    out["trade_date"] = pd.to_datetime(out["trade_date"], errors="raise").dt.normalize()
    out["ticker"] = out["ticker"].astype("string").str.strip().str.upper()
    out["permno"] = pd.to_numeric(out["permno"], errors="raise").astype("int64")

    for col in ("ret", "prc", "vol", "shrout"):
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")

    out["source"] = out["source"].astype("string")
    out["load_ts_utc"] = pd.to_datetime(out["load_ts_utc"], errors="raise", utc=True)

    return out[list(SECURITY_RETURNS_COLUMNS)]


# Purpose: Validate canonical WRDS security returns including unique-key and source semantics.
def validate_security_returns_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Validate the `security_returns_daily` dataset contract.

    Inputs:
      - df: Candidate security returns dataframe.

    Returns:
      - Canonicalized dataframe.

    Raises:
      - ValueError for empty ticker/source values or duplicate keys.

    Notes on units:
      - `ret` remains decimal return from CRSP.
    """

    out = coerce_security_returns_daily(df)

    if out["ticker"].isna().any() or (out["ticker"].str.len() == 0).any():
        raise ValueError("security_returns_daily contains empty ticker values")

    if out["source"].isna().any() or (out["source"].str.len() == 0).any():
        raise ValueError("security_returns_daily contains empty source values")

    require_unique_key(
        out,
        key=SECURITY_RETURNS_KEY,
        dataset_name="security_returns_daily",
    )
    return out


# Purpose: Normalize aggregated portfolio return rows to canonical schema and dtypes.
def coerce_portfolio_returns_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce `portfolio_returns_daily` columns to canonical types.

    Inputs:
      - df: Aggregated portfolio returns dataframe.

    Returns:
      - Normalized portfolio returns dataframe.

    Raises:
      - ValueError for missing columns or coercion failures.

    Notes on units:
      - `portfolio_ret` is decimal return (0.01 = 1%).
    """

    require_columns(
        df,
        required=PORTFOLIO_RETURNS_COLUMNS,
        dataset_name="portfolio_returns_daily",
    )

    out = df.copy()
    out["trade_date"] = pd.to_datetime(out["trade_date"], errors="raise").dt.normalize()
    out["portfolio_ret"] = pd.to_numeric(out["portfolio_ret"], errors="coerce").astype(
        "float64"
    )
    out["n_constituents"] = pd.to_numeric(
        out["n_constituents"], errors="raise"
    ).astype("int64")
    out["gross_exposure"] = pd.to_numeric(
        out["gross_exposure"], errors="raise"
    ).astype("float64")
    out["method"] = out["method"].astype("string")
    out["load_ts_utc"] = pd.to_datetime(out["load_ts_utc"], errors="raise", utc=True)

    return out[list(PORTFOLIO_RETURNS_COLUMNS)]


# Purpose: Validate canonical portfolio returns for key uniqueness and non-empty method labels.
def validate_portfolio_returns_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Validate the `portfolio_returns_daily` dataset contract.

    Inputs:
      - df: Candidate portfolio returns dataframe.

    Returns:
      - Canonicalized dataframe.

    Raises:
      - ValueError for invalid method values or duplicate keys.

    Notes on units:
      - Financial return values remain decimals.
    """

    out = coerce_portfolio_returns_daily(df)

    if out["method"].isna().any() or (out["method"].str.len() == 0).any():
        raise ValueError("portfolio_returns_daily contains empty method values")

    require_unique_key(
        out,
        key=PORTFOLIO_RETURNS_KEY,
        dataset_name="portfolio_returns_daily",
    )
    return out


# Purpose: Convert a typed QA row into one-row dataframe compliant with QA contract columns.
def qa_row_to_frame(row: QaRow) -> pd.DataFrame:
    """Build a single QA dataframe row from `QaRow`.

    Inputs:
      - row: Typed QA row dataclass.

    Returns:
      - One-row dataframe with `daily_qa` schema.

    Raises:
      - None.

    Notes on units:
      - QA counts are integer row statistics.
    """

    return pd.DataFrame(
        [
            {
                "run_date": row.run_date,
                "stage": row.stage,
                "rows_universe": row.rows_universe,
                "rows_returns": row.rows_returns,
                "missing_permno": row.missing_permno,
                "missing_ret": row.missing_ret,
                "status": row.status,
                "error_message": row.error_message,
            }
        ]
    )


# Purpose: Normalize QA records for deterministic storage and downstream publishing.
def coerce_daily_qa(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce `daily_qa` columns to canonical types.

    Inputs:
      - df: QA dataframe.

    Returns:
      - Normalized QA dataframe.

    Raises:
      - ValueError for missing columns or coercion errors.

    Notes on units:
      - QA counts are unitless row metrics.
    """

    require_columns(df, required=DAILY_QA_COLUMNS, dataset_name="daily_qa")

    out = df.copy()
    out["run_date"] = pd.to_datetime(out["run_date"], errors="raise").dt.date
    out["stage"] = out["stage"].astype("string")

    for col in ("rows_universe", "rows_returns", "missing_permno", "missing_ret"):
        out[col] = pd.to_numeric(out[col], errors="raise").astype("int64")

    out["status"] = out["status"].astype("string")
    out["error_message"] = out["error_message"].astype("string")

    return out[list(DAILY_QA_COLUMNS)]


# Purpose: Validate QA contract rules and unique key behavior.
def validate_daily_qa(df: pd.DataFrame) -> pd.DataFrame:
    """Validate `daily_qa` dataset schema and key uniqueness.

    Inputs:
      - df: Candidate QA dataframe.

    Returns:
      - Canonicalized QA dataframe.

    Raises:
      - ValueError if status/stage fields are empty or keys duplicate.

    Notes on units:
      - QA metrics remain integer counts.
    """

    out = coerce_daily_qa(df)
    if out["stage"].isna().any() or (out["stage"].str.len() == 0).any():
        raise ValueError("daily_qa contains empty stage values")

    if out["status"].isna().any() or (out["status"].str.len() == 0).any():
        raise ValueError("daily_qa contains empty status values")

    require_unique_key(out, key=DAILY_QA_KEY, dataset_name="daily_qa")
    return out


# Purpose: Build consistent UTC timestamp used in load metadata columns.
def utc_now() -> datetime:
    """Return current UTC timestamp for load metadata.

    Inputs:
      - None.

    Returns:
      - timezone-aware UTC datetime.

    Raises:
      - None.

    Notes on units:
      - Timestamp metadata only.
    """

    return datetime.now(UTC)
