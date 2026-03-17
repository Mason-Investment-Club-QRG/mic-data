# Purpose: Pull WRDS/CRSP daily return data and normalize it to the canonical security_returns contract.

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, cast

import pandas as pd

from mic_data.config.secrets import wrds_password, wrds_username
from mic_data.contracts.daily_returns_contracts import (
    validate_security_returns_daily,
    validate_universe_daily,
)
from mic_data.market.interfaces import DailyReturnSource


class WrdsConnection(Protocol):
    """Minimal WRDS connection protocol required by this source."""

    def raw_sql(self, query: str, date_cols: list[str] | None = None) -> pd.DataFrame:
        ...

    def close(self) -> None:
        ...


class WrdsConnectionFactory(Protocol):
    """Factory protocol for dependency-injected WRDS connections."""

    def __call__(self, *, wrds_username: str | None = None) -> WrdsConnection:
        ...


@dataclass(frozen=True)
class WrdsCrspDailyReturnSource(DailyReturnSource):
    """Load daily returns from WRDS CRSP DSF.

    Inputs:
      - username: Optional WRDS username override.
      - connection_factory: Optional injectable connection factory for tests.

    Returns:
      - Dataframe conforming to `security_returns_daily` contract.

    Raises:
      - RuntimeError for WRDS auth/query failures.
      - ValueError for malformed inputs or empty required universe.

    Notes on units:
      - `ret` is CRSP decimal total return.
    """

    username: str | None = None
    connection_factory: WrdsConnectionFactory | None = None

    # Purpose: Pull WRDS daily returns for tickers in the validated universe and date range.
    def load_security_returns(
        self,
        *,
        universe: pd.DataFrame,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Fetch and normalize WRDS CRSP daily security returns.

        Inputs:
          - universe: Candidate `universe_daily` dataframe.
          - start_date: Inclusive date range start (YYYY-MM-DD).
          - end_date: Inclusive date range end (YYYY-MM-DD).

        Returns:
          - Contract-validated security returns dataframe.

        Raises:
          - ValueError for invalid universe/date range.
          - RuntimeError for WRDS connectivity/query failures.

        Notes on units:
          - `ret` stays as decimal return from CRSP.
        """

        universe_clean = validate_universe_daily(universe)
        tickers = sorted(
            {
                str(t)
                for t in universe_clean["ticker"].dropna().astype(str).tolist()
                if str(t).strip()
            }
        )
        if not tickers:
            raise ValueError("Universe contains no tickers to query from WRDS.")

        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date)
        if end_ts < start_ts:
            raise ValueError("end_date must be greater than or equal to start_date.")

        query = self._build_query(
            tickers=tickers,
            start_date=start_ts.strftime("%Y-%m-%d"),
            end_date=end_ts.strftime("%Y-%m-%d"),
        )

        conn: WrdsConnection | None = None
        try:
            conn = self._connect()
            raw = conn.raw_sql(query, date_cols=["trade_date", "namedt", "nameenddt"])
        except Exception as exc:  # pragma: no cover - external auth/runtime variance
            raise RuntimeError(f"WRDS CRSP pull failed: {exc}") from exc
        finally:
            if conn is not None:
                conn.close()

        if raw.empty:
            raise RuntimeError("WRDS returned no CRSP DSF rows for requested universe/date window.")

        normalized = self._normalize_rows(raw)
        return validate_security_returns_daily(normalized)

    # Purpose: Construct SQL query joining CRSP DSF and STOCKNAMES by date-effective ticker mapping.
    def _build_query(self, *, tickers: list[str], start_date: str, end_date: str) -> str:
        ticker_list = ", ".join([f"'{self._escape_sql_literal(t)}'" for t in tickers])
        return f"""
            SELECT
                d.date AS trade_date,
                s.ticker,
                d.permno,
                d.ret,
                d.prc,
                d.vol,
                d.shrout,
                s.namedt,
                s.nameenddt
            FROM crsp.dsf AS d
            INNER JOIN crsp.stocknames AS s
                ON d.permno = s.permno
               AND d.date BETWEEN s.namedt AND s.nameenddt
            WHERE s.ticker IN ({ticker_list})
              AND d.date BETWEEN '{start_date}' AND '{end_date}'
            ORDER BY d.date, d.permno
        """

    # Purpose: Escape single quotes in SQL literals to avoid malformed ticker predicates.
    def _escape_sql_literal(self, value: str) -> str:
        return value.replace("'", "''")

    # Purpose: Convert raw WRDS query output into canonical contract columns and metadata.
    def _normalize_rows(self, raw: pd.DataFrame) -> pd.DataFrame:
        load_ts = pd.Timestamp.now(tz="UTC")

        frame = raw.copy()
        frame.columns = [str(c).strip().lower() for c in frame.columns]

        if "trade_date" not in frame.columns:
            raise ValueError("WRDS output missing required column 'trade_date'.")

        frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="raise")
        frame["ticker"] = frame["ticker"].astype("string").str.strip().str.upper()

        for col in ("permno", "ret", "prc", "vol", "shrout"):
            frame[col] = pd.to_numeric(frame[col], errors="coerce")

        # CRSP can emit overlapping stocknames rows; drop duplicates on natural security key.
        frame = frame.sort_values(["trade_date", "permno", "ticker"]).drop_duplicates(
            subset=["trade_date", "permno"],
            keep="first",
        )

        frame["source"] = "wrds_crsp"
        frame["load_ts_utc"] = load_ts

        return frame[
            [
                "trade_date",
                "ticker",
                "permno",
                "ret",
                "prc",
                "vol",
                "shrout",
                "source",
                "load_ts_utc",
            ]
        ]

    # Purpose: Build a WRDS connection from either injected factory or runtime environment config.
    def _connect(self) -> WrdsConnection:
        username = wrds_username(self.username)
        factory = self.connection_factory
        if factory is not None:
            return factory(wrds_username=username)

        try:
            import wrds
        except Exception as exc:
            raise RuntimeError("wrds package is unavailable in this environment.") from exc

        try:
            conn = wrds.Connection(
                wrds_username=username,
                wrds_password=wrds_password(),
            )
            return cast(WrdsConnection, conn)
        except Exception as exc:  # pragma: no cover - external auth/runtime variance
            raise RuntimeError(
                "Unable to establish WRDS connection. Check WRDS_USERNAME and WRDS_PASSWORD/.pgpass."
            ) from exc
