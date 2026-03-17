# Purpose: Centralize all Google Sheets identifiers, tab names, and output tab settings.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class GoogleSheetsInputs:
    """Input tab names for upstream club-maintained data."""

    holdings_tab: str
    watchlist_tab: str


@dataclass(frozen=True)
class GoogleSheetsOutputs:
    """Output tab names for pipeline-published datasets."""

    security_returns_tab: str
    portfolio_returns_tab: str
    qa_tab: str


@dataclass(frozen=True)
class GoogleSheetsOptions:
    """Behavior options used by publisher and sync stages."""

    clear_before_write: bool = True
    max_rows: int = 50000


@dataclass(frozen=True)
class GoogleSheetsConfig:
    """Full Google Sheets configuration contract.

    Inputs:
      - sheet_id: Spreadsheet identifier.
      - credentials_env_var: Environment variable containing service-account JSON path.
      - inputs: Input tab names for holdings and watchlist.
      - outputs: Output tab names for published datasets.
      - options: Publisher behavior flags.

    Returns:
      - Immutable config object consumed by all sheet-integrated modules.

    Raises:
      - None. Validation is handled by `load_google_sheets_config`.

    Notes on units:
      - Tab names and IDs are unitless identifiers.
    """

    sheet_id: str
    credentials_env_var: str
    inputs: GoogleSheetsInputs
    outputs: GoogleSheetsOutputs
    options: GoogleSheetsOptions


# Purpose: Validate mapping keys for required configuration sections.
def _require_mapping(value: Any, *, section_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"Section '{section_name}' must be a mapping.")
    return value


# Purpose: Read required string fields with clear failures for malformed YAML.
def _require_string(mapping: dict[str, Any], *, key: str, section_name: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Section '{section_name}' requires non-empty string '{key}'.")
    return value.strip()


# Purpose: Read optional integer fields while enforcing non-negative bounds.
def _optional_non_negative_int(
    mapping: dict[str, Any],
    *,
    key: str,
    section_name: str,
    default: int,
) -> int:
    raw = mapping.get(key, default)
    if not isinstance(raw, int) or raw < 0:
        raise ValueError(
            f"Section '{section_name}' requires non-negative integer '{key}'."
        )
    return raw


# Purpose: Read optional boolean fields while preserving explicit defaults.
def _optional_bool(
    mapping: dict[str, Any],
    *,
    key: str,
    section_name: str,
    default: bool,
) -> bool:
    raw = mapping.get(key, default)
    if not isinstance(raw, bool):
        raise ValueError(f"Section '{section_name}' requires boolean '{key}'.")
    return raw


# Purpose: Load and validate the canonical Google Sheets config file for all modules.
def load_google_sheets_config(path: str | Path) -> GoogleSheetsConfig:
    """Load Google Sheets settings from YAML.

    Inputs:
      - path: YAML path for unified sheet identifiers and tab names.

    Returns:
      - GoogleSheetsConfig with validated required sections.

    Raises:
      - FileNotFoundError if `path` does not exist.
      - ValueError if YAML structure is invalid or required keys are missing.

    Notes on units:
      - All values are identifiers/flags and do not carry financial units.
    """

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Google Sheets config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    root = _require_mapping(raw, section_name="root")

    sheets = _require_mapping(root.get("google_sheets"), section_name="google_sheets")
    inputs = _require_mapping(sheets.get("inputs"), section_name="google_sheets.inputs")
    outputs = _require_mapping(
        sheets.get("outputs"), section_name="google_sheets.outputs"
    )
    options = _require_mapping(
        sheets.get("options", {}), section_name="google_sheets.options"
    )

    return GoogleSheetsConfig(
        sheet_id=_require_string(sheets, key="sheet_id", section_name="google_sheets"),
        credentials_env_var=_require_string(
            sheets,
            key="credentials_env_var",
            section_name="google_sheets",
        ),
        inputs=GoogleSheetsInputs(
            holdings_tab=_require_string(
                inputs,
                key="holdings_tab",
                section_name="google_sheets.inputs",
            ),
            watchlist_tab=_require_string(
                inputs,
                key="watchlist_tab",
                section_name="google_sheets.inputs",
            ),
        ),
        outputs=GoogleSheetsOutputs(
            security_returns_tab=_require_string(
                outputs,
                key="security_returns_tab",
                section_name="google_sheets.outputs",
            ),
            portfolio_returns_tab=_require_string(
                outputs,
                key="portfolio_returns_tab",
                section_name="google_sheets.outputs",
            ),
            qa_tab=_require_string(
                outputs,
                key="qa_tab",
                section_name="google_sheets.outputs",
            ),
        ),
        options=GoogleSheetsOptions(
            clear_before_write=_optional_bool(
                options,
                key="clear_before_write",
                section_name="google_sheets.options",
                default=True,
            ),
            max_rows=_optional_non_negative_int(
                options,
                key="max_rows",
                section_name="google_sheets.options",
                default=50000,
            ),
        ),
    )
