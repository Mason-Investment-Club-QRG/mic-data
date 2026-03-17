# Purpose: Define abstract contracts for daily universe and daily return sources.

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class UniverseSource(ABC):
    """Contract for loading canonical daily universe rows."""

    # Purpose: Provide contract method signature for retrieving daily universe data.
    @abstractmethod
    def load_universe(self, *, as_of_date: str) -> pd.DataFrame:
        """Load universe rows for the specified date.

        Inputs:
          - as_of_date: Snapshot date in YYYY-MM-DD format.

        Returns:
          - Dataframe conforming to `universe_daily` contract.

        Raises:
          - ValueError for malformed date or contract mismatches.
          - RuntimeError for upstream source failures.

        Notes on units:
          - Shares are count units when present.
        """

        raise NotImplementedError


class DailyReturnSource(ABC):
    """Contract for loading canonical daily security returns."""

    # Purpose: Provide contract method signature for retrieving security-level daily returns.
    @abstractmethod
    def load_security_returns(
        self,
        *,
        universe: pd.DataFrame,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Load security returns for a universe and date range.

        Inputs:
          - universe: Contract-validated `universe_daily` dataframe.
          - start_date: Inclusive range start YYYY-MM-DD.
          - end_date: Inclusive range end YYYY-MM-DD.

        Returns:
          - Dataframe conforming to `security_returns_daily` contract.

        Raises:
          - ValueError for invalid range or bad universe inputs.
          - RuntimeError for external source connectivity/query issues.

        Notes on units:
          - `ret` must be decimal return values.
        """

        raise NotImplementedError
