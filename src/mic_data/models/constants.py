from __future__ import annotations

from typing import Final, Literal, TypeAlias

# Canonical model sampling frequency used across all FF3 modules.
ModelFrequency: TypeAlias = Literal["M"]

# Canonical factor schema used across all FF3 data sources.
FACTOR_COLUMNS: Final[tuple[str, str, str, str]] = ("mkt_rf", "smb", "hml", "rf")
CANONICAL_FREQUENCY: Final[ModelFrequency] = "M"
