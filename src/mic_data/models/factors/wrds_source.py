from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, Literal

import pandas as pd

from mic_data.models.constants import FACTOR_COLUMNS
from mic_data.models.interfaces import FactorSource


@dataclass(frozen=True)
class WrdsFactorSource(FactorSource):
    """Load FF3 factors from WRDS as the primary source.

    Inputs:
      - username: Optional WRDS username override. If omitted, WRDS_USERNAME env var is used.
      - connection_factory: Optional injectable connection builder for tests.

    Returns:
      - Factor dataframe indexed by month-end date with canonical columns:
        mkt_rf, smb, hml, rf.

    Raises:
      - RuntimeError for WRDS auth/query failures.
      - ValueError for unsupported frequency or malformed source columns.

    Notes on units:
      - WRDS ff.factors_daily values are consumed directly as decimal returns.
    """

    username: str | None = None
    connection_factory: Callable[..., Any] | None = None

    def load_factors(
        self,
        start_date: str,
        end_date: str,
        frequency: Literal["M"] = "M",
    ) -> pd.DataFrame:
        """Pull WRDS factors and aggregate to monthly frequency.

        Inputs:
          - start_date: Inclusive start date in YYYY-MM-DD format.
          - end_date: Inclusive end date in YYYY-MM-DD format.
          - frequency: Only "M" is supported in the FF3 pipeline.

        Returns:
          - pd.DataFrame indexed by month-end date with mkt_rf, smb, hml, rf.

        Raises:
          - RuntimeError when WRDS connection/query fails.
          - ValueError for unsupported frequency or invalid result schema.

        Notes on units:
          - Output values are monthly decimal returns.
        """
        if frequency != "M":
            raise ValueError("WrdsFactorSource only supports monthly frequency 'M'.")

        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date)

        query = f"""
            SELECT date, mktrf, smb, hml, rf
            FROM ff.factors_daily
            WHERE date BETWEEN '{start_ts.strftime("%Y-%m-%d")}'
              AND '{end_ts.strftime("%Y-%m-%d")}'
            ORDER BY date
        """

        conn = None
        try:
            conn = self._connect()
            daily = conn.raw_sql(query, date_cols=["date"])
        except Exception as exc:  # pragma: no cover - exact exception class depends on wrds lib
            raise RuntimeError(f"WRDS factor pull failed: {exc}") from exc
        finally:
            if conn is not None and hasattr(conn, "close"):
                conn.close()

        if daily.empty:
            raise ValueError(
                "WRDS returned no ff.factors_daily rows for the requested window."
            )

        daily.columns = [str(c).strip().lower() for c in daily.columns]
        daily = daily.rename(columns={"mktrf": "mkt_rf"})
        missing = [c for c in ["date", *FACTOR_COLUMNS] if c not in daily.columns]
        if missing:
            raise ValueError(
                f"WRDS factor pull missing columns: {missing}. "
                f"Available columns: {list(daily.columns)}"
            )

        daily["date"] = pd.to_datetime(daily["date"])
        for col in FACTOR_COLUMNS:
            daily[col] = pd.to_numeric(daily[col], errors="coerce").astype("float64")

        daily = daily[["date", *FACTOR_COLUMNS]].dropna().set_index("date").sort_index()
        if daily.empty:
            raise ValueError("WRDS factor pull has no usable rows after numeric cleaning.")

        # Compound daily returns to month-end to match FF3 monthly regression convention.
        monthly = (1.0 + daily[FACTOR_COLUMNS]).resample("ME").prod() - 1.0

        start_period = start_ts.to_period("M")
        end_period = end_ts.to_period("M")
        mask = (monthly.index.to_period("M") >= start_period) & (
            monthly.index.to_period("M") <= end_period
        )
        monthly = monthly.loc[mask]

        if monthly.empty:
            raise ValueError(
                "WRDS factor pull contains no monthly observations after aggregation."
            )

        return monthly

    def _connect(self) -> Any:
        username = self.username or os.getenv("WRDS_USERNAME")
        factory = self.connection_factory
        if factory is not None:
            return factory(wrds_username=username)

        try:
            import wrds  # Imported lazily so environments without wrds can still run static paths.
        except Exception as exc:
            raise RuntimeError(
                "wrds package is not available. Install it and configure WRDS_USERNAME/.pgpass."
            ) from exc

        try:
            return wrds.Connection(wrds_username=username)
        except Exception as exc:  # pragma: no cover - depends on external auth/runtime
            raise RuntimeError(
                "Unable to establish WRDS connection. Ensure WRDS_USERNAME and .pgpass "
                "(or PGPASSWORD) are configured for non-interactive authentication."
            ) from exc
