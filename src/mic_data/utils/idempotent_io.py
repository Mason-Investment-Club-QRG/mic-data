# Purpose: Provide idempotent file writes, deterministic hashing, lock files, and manifest persistence.

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal

import pandas as pd

WriteMode = Literal["replace", "skip", "error"]


@dataclass(frozen=True)
class WriteResult:
    """Outcome metadata for an idempotent write operation."""

    path: Path
    wrote: bool
    content_hash: str
    row_count: int


# Purpose: Normalize frame ordering and index handling before deterministic hashing or writing.
def canonicalize_frame(df: pd.DataFrame, *, sort_by: list[str]) -> pd.DataFrame:
    """Return deterministic row/column ordering for idempotent writes.

    Inputs:
      - df: Dataframe to normalize.
      - sort_by: Ordered columns used for deterministic sorting.

    Returns:
      - Copy of `df` with stable sorting and reset integer index.

    Raises:
      - ValueError if any sort column is missing.

    Notes on units:
      - Does not alter financial units.
    """

    missing = [c for c in sort_by if c not in df.columns]
    if missing:
        raise ValueError(f"Missing sort column(s): {missing}")

    normalized = df.copy()
    normalized = normalized.sort_values(sort_by, kind="mergesort").reset_index(drop=True)
    return normalized


# Purpose: Compute a stable SHA256 digest from a canonical dataframe representation.
def dataframe_hash(df: pd.DataFrame) -> str:
    """Compute deterministic hash for a dataframe payload.

    Inputs:
      - df: Canonicalized dataframe.

    Returns:
      - Hex SHA256 digest of CSV bytes with consistent NA rendering.

    Raises:
      - None.

    Notes on units:
      - Hash is unitless metadata.
    """

    payload = df.to_csv(index=False, na_rep="", lineterminator="\n")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# Purpose: Enforce write-mode semantics before mutating output files.
def should_write(path: Path, *, mode: WriteMode) -> bool:
    """Decide whether a write should proceed based on existing file and mode.

    Inputs:
      - path: Destination path.
      - mode: `replace`, `skip`, or `error`.

    Returns:
      - True if writer should emit file content.

    Raises:
      - FileExistsError when `mode=error` and path already exists.
      - ValueError for unsupported mode.

    Notes on units:
      - Control-flow only; no units.
    """

    if mode not in ("replace", "skip", "error"):
        raise ValueError(f"Unsupported write mode: {mode}")

    if not path.exists():
        return True

    if mode == "replace":
        return True
    if mode == "skip":
        return False

    raise FileExistsError(f"File already exists and mode='error': {path}")


# Purpose: Write a dataframe atomically as parquet using temporary-file replacement.
def atomic_write_parquet(
    df: pd.DataFrame,
    *,
    path: Path,
    sort_by: list[str],
    mode: WriteMode,
) -> WriteResult:
    """Idempotently persist a dataframe to parquet with deterministic ordering.

    Inputs:
      - df: Dataset to persist.
      - path: Destination parquet path.
      - sort_by: Stable sort key columns.
      - mode: Existing-file behavior.

    Returns:
      - WriteResult containing hash and row count.

    Raises:
      - FileExistsError when mode is `error` and destination exists.
      - ValueError if sort columns are invalid.
      - OSError on filesystem write failures.

    Notes on units:
      - Values are preserved exactly; no unit conversion.
    """

    normalized = canonicalize_frame(df, sort_by=sort_by)
    digest = dataframe_hash(normalized)

    if not should_write(path, mode=mode):
        return WriteResult(path=path, wrote=False, content_hash=digest, row_count=len(normalized))

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        suffix=".parquet",
        delete=False,
        dir=path.parent,
    ) as tmp:
        tmp_path = Path(tmp.name)

    try:
        normalized.to_parquet(tmp_path, index=False)
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)

    return WriteResult(path=path, wrote=True, content_hash=digest, row_count=len(normalized))


# Purpose: Write a dataframe atomically as CSV for easy human-readable consumption.
def atomic_write_csv(
    df: pd.DataFrame,
    *,
    path: Path,
    sort_by: list[str],
    mode: WriteMode,
) -> WriteResult:
    """Idempotently persist a dataframe to CSV with deterministic ordering.

    Inputs:
      - df: Dataset to persist.
      - path: Destination CSV path.
      - sort_by: Stable sort key columns.
      - mode: Existing-file behavior.

    Returns:
      - WriteResult containing hash and row count.

    Raises:
      - FileExistsError when mode is `error` and destination exists.
      - ValueError if sort columns are invalid.
      - OSError on filesystem write failures.

    Notes on units:
      - Values are preserved exactly; no unit conversion.
    """

    normalized = canonicalize_frame(df, sort_by=sort_by)
    digest = dataframe_hash(normalized)

    if not should_write(path, mode=mode):
        return WriteResult(path=path, wrote=False, content_hash=digest, row_count=len(normalized))

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".csv",
        delete=False,
        dir=path.parent,
        encoding="utf-8",
    ) as tmp:
        tmp_path = Path(tmp.name)
        normalized.to_csv(tmp_path, index=False)

    try:
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)

    return WriteResult(path=path, wrote=True, content_hash=digest, row_count=len(normalized))


# Purpose: Persist run metadata manifests with atomic replacement semantics.
def write_manifest_json(
    *,
    path: Path,
    payload: dict[str, object],
    mode: WriteMode,
) -> bool:
    """Write a JSON manifest atomically.

    Inputs:
      - path: Destination manifest path.
      - payload: Serializable metadata payload.
      - mode: Existing-file behavior.

    Returns:
      - True if file was written; False if skipped.

    Raises:
      - FileExistsError when mode is `error` and destination exists.
      - OSError/ValueError on serialization or write issues.

    Notes on units:
      - Manifest values are metadata.
    """

    if not should_write(path, mode=mode):
        return False

    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2, sort_keys=True, default=str)

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".json",
        delete=False,
        dir=path.parent,
        encoding="utf-8",
    ) as tmp:
        tmp_path = Path(tmp.name)
        tmp.write(serialized)

    try:
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)

    return True


# Purpose: Acquire an exclusive stage lock to prevent concurrent duplicate runs.
@contextmanager
def stage_lock(lock_path: Path) -> Iterator[None]:
    """Acquire and release an exclusive lock file for a pipeline stage.

    Inputs:
      - lock_path: Lock file path under outputs/locks.

    Returns:
      - Context manager yielding once lock acquisition succeeds.

    Raises:
      - RuntimeError if lock already exists.
      - OSError for filesystem failures.

    Notes on units:
      - Locking metadata has no financial units.
    """

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(f"Stage lock already exists: {lock_path}") from exc

    try:
        os.write(fd, str(os.getpid()).encode("utf-8"))
        os.close(fd)
        yield
    finally:
        lock_path.unlink(missing_ok=True)
