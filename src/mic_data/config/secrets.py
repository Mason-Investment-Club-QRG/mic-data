# Purpose: Centralize environment-secret loading and retrieval for Google, WRDS, and other credentials.

from __future__ import annotations

import os
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[3]

_ENV_LOADED = False


# Purpose: Load `.env` exactly once so all modules share one secret source.
def ensure_env_loaded() -> None:
    """Load `.env` file from project root/parents into process environment.

    Inputs:
      - None.

    Returns:
      - None.

    Raises:
      - None. Missing `.env` is allowed so CI can rely on injected env vars.

    Notes on units:
      - Environment values are strings/paths, not financial quantities.
    """

    global _ENV_LOADED
    if _ENV_LOADED:
        return

    dotenv_path = find_dotenv(usecwd=True)
    if dotenv_path:
        load_dotenv(dotenv_path=dotenv_path, override=False)

    _ENV_LOADED = True


# Purpose: Resolve a path-valued secret from environment and normalize relative paths.
def require_path_env(var_name: str) -> Path:
    """Read required path from environment and normalize to absolute path.

    Inputs:
      - var_name: Environment variable name expected to contain a file path.

    Returns:
      - Absolute path to the requested file.

    Raises:
      - RuntimeError if env var is missing.
      - FileNotFoundError if the resolved file path does not exist.

    Notes on units:
      - Path metadata only.
    """

    ensure_env_loaded()
    raw_value = os.getenv(var_name)
    if not raw_value:
        raise RuntimeError(
            f"{var_name} is not set. Set it in one place: your repo `.env` file.\n"
            f"Example:\n"
            f"  {var_name}=secrets/your_service_account.json\n"
            f"Then rerun the command."
        )

    path = Path(raw_value)
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()

    if not path.exists():
        raise FileNotFoundError(
            f"{var_name} points to a missing file: {path}. "
            "Update `.env` to the correct credential path."
        )

    return path


# Purpose: Read optional secret values from environment after ensuring `.env` has been loaded.
def optional_env(var_name: str) -> str | None:
    """Read optional environment variable.

    Inputs:
      - var_name: Environment variable key.

    Returns:
      - String value if present and non-empty, otherwise None.

    Raises:
      - None.

    Notes on units:
      - Secret metadata only.
    """

    ensure_env_loaded()
    value = os.getenv(var_name)
    if value is None:
        return None
    stripped = value.strip()
    return stripped if stripped else None


# Purpose: Resolve WRDS username with explicit override priority over environment.
def wrds_username(explicit_username: str | None = None) -> str | None:
    """Resolve WRDS username.

    Inputs:
      - explicit_username: Optional explicit value from function args/config.

    Returns:
      - Username string if available; otherwise None.

    Raises:
      - None.

    Notes on units:
      - Identifier metadata only.
    """

    if explicit_username and explicit_username.strip():
        return explicit_username.strip()
    return optional_env("WRDS_USERNAME")


# Purpose: Resolve WRDS password when provided via environment for non-interactive auth.
def wrds_password() -> str | None:
    """Read optional WRDS password from environment.

    Inputs:
      - None.

    Returns:
      - WRDS password string if set; otherwise None.

    Raises:
      - None.

    Notes on units:
      - Secret metadata only.
    """

    return optional_env("WRDS_PASSWORD")
