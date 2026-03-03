from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from mic_data.models.constants import FACTOR_COLUMNS, ModelFrequency
from mic_data.models.interfaces import FactorSource


@dataclass(frozen=True)
class StaticCsvFactorSource(FactorSource):
    """Load FF3 factors from the local Ken French CSV file.

    Inputs:
      - csv_path: Path to F-F_Research_Data_Factors.csv.

    Returns:
      - Factor dataframe indexed by month-end date with columns:
        mkt_rf, smb, hml, rf.

    Raises:
      - FileNotFoundError if the CSV path does not exist.
      - ValueError if required columns are missing or rows are invalid.

    Notes on units:
      - File values are percentages and are converted to decimal returns.
    """

    csv_path: Path

    def load_factors(
        self,
        start_date: str,
        end_date: str,
        frequency: ModelFrequency = "M",
    ) -> pd.DataFrame:
        """Load and normalize static FF3 factors.

        Inputs:
          - start_date: Inclusive start date in YYYY-MM-DD format.
          - end_date: Inclusive end date in YYYY-MM-DD format.
          - frequency: Only "M" is supported.

        Returns:
          - pd.DataFrame indexed by month-end date with canonical factor columns.

        Raises:
          - FileNotFoundError when csv_path is missing.
          - ValueError for unsupported frequency or malformed file content.

        Notes on units:
          - Output values are decimals (0.01 = 1%).
        """
        if frequency != "M":
            raise ValueError(
                "StaticCsvFactorSource only supports monthly frequency 'M'."
            )

        if not self.csv_path.exists():
            raise FileNotFoundError(f"Static factor file not found: {self.csv_path}")

        raw = pd.read_csv(self.csv_path, skiprows=3)
        raw.columns = [str(c).strip() for c in raw.columns]
        if raw.empty:
            raise ValueError("Static factor file has no data rows.")

        # The first column stores YYYYMM values. Footer/comment rows are filtered out.
        date_col = raw.columns[0]
        raw = raw[raw[date_col].astype(str).str.match(r"^\d{6}$")].copy()
        if raw.empty:
            raise ValueError("Static factor file contains no valid YYYYMM rows.")

        raw = raw.rename(columns={date_col: "date"})

        # Normalize legacy column names from Ken French CSV to internal schema.
        rename_map = {
            "Mkt-RF": "mkt_rf",
            "SMB": "smb",
            "HML": "hml",
            "RF": "rf",
            "mktrf": "mkt_rf",
        }
        raw = raw.rename(columns=rename_map)

        missing = [c for c in FACTOR_COLUMNS if c not in raw.columns]
        if missing:
            raise ValueError(
                f"Static factor file missing required columns: {missing}. "
                f"Available columns: {list(raw.columns)}"
            )

        raw["date"] = pd.to_datetime(raw["date"], format="%Y%m") + pd.offsets.MonthEnd(
            0
        )

        for col in FACTOR_COLUMNS:
            raw[col] = pd.to_numeric(raw[col], errors="coerce") / 100.0

        out = raw[["date", *FACTOR_COLUMNS]].dropna().set_index("date").sort_index()

        start_period = pd.Timestamp(start_date).to_period("M")
        end_period = pd.Timestamp(end_date).to_period("M")
        mask = (out.index.to_period("M") >= start_period) & (
            out.index.to_period("M") <= end_period
        )
        out = out.loc[mask]

        if out.empty:
            raise ValueError(
                "No static factor observations in requested date window "
                f"[{start_date}, {end_date}]."
            )

        return out
