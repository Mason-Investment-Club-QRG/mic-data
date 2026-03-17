# Purpose: Resolve external API credentials through centralized secret-loading helpers.

from __future__ import annotations

from mic_data.config.secrets import optional_env

# Purpose: Read optional FRED API key from environment/.env for macro data consumers.
FRED_API_KEY = optional_env("FRED_API_KEY")

if not FRED_API_KEY:
    raise RuntimeError(
        "FRED_API_KEY is not set. Define it once in your `.env` file."
    )
