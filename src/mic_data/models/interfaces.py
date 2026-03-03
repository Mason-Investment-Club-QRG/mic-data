from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from mic_data.models.constants import ModelFrequency


class FactorSource(ABC):
    """Contract for loading Fama-French factor data.

    Inputs:
      - start_date: Inclusive start date in YYYY-MM-DD format.
      - end_date: Inclusive end date in YYYY-MM-DD format.
      - frequency: Sampling frequency. Current pipeline supports monthly ("M").

    Returns:
      - pd.DataFrame indexed by date with columns: mkt_rf, smb, hml, rf.

    Raises:
      - ValueError for invalid frequency/date windows or missing columns.
      - RuntimeError for source connectivity failures.

    Notes on units:
      - Returned factor values must be decimal returns (e.g., 0.01 == 1%).
    """

    @abstractmethod
    def load_factors(
        self,
        start_date: str,
        end_date: str,
        frequency: ModelFrequency = "M",
    ) -> pd.DataFrame:
        raise NotImplementedError


class PortfolioReturnSource(ABC):
    """Contract for loading portfolio return series for FF3 regressions.

    Inputs:
      - start_date: Inclusive start date in YYYY-MM-DD format.
      - end_date: Inclusive end date in YYYY-MM-DD format.
      - frequency: Sampling frequency. Current pipeline supports monthly ("M").

    Returns:
      - pd.Series indexed by date named "portfolio_return".

    Raises:
      - ValueError for invalid frequency/date windows or malformed inputs.
      - RuntimeError for upstream market data access issues.

    Notes on units:
      - Returned values must be decimal returns (e.g., 0.01 == 1%).
    """

    @abstractmethod
    def load_portfolio_returns(
        self,
        start_date: str,
        end_date: str,
        frequency: ModelFrequency = "M",
    ) -> pd.Series:
        raise NotImplementedError
