# MIC Data Pipeline

A durable data pipeline for Mason Investment Club

**Goals:**
- Read **holdings + universe** from Google Sheets (or a local CSV fallback)
- Pull **daily prices** (Yahoo Finance)
- Pull **macro series** (FRED)
- Produce **clean tables** in `data/cleaned/` and small **dashboard-ready exports** in `outputs/`


## Repo layout

- `config/`
  - `universe.csv` — tickers to track + roles (HOLDING / WATCHLIST / BENCHMARK)
  - `settings.yaml` — non-secret settings (date ranges, sheet tab name, etc.)
- `src/mic_data/` — pipeline code
- `data/` — raw + cleaned datasets (not committed)
- `outputs/` — exported CSVs for Sheets / BI (not committed)
- `docs/` — data contracts + onboarding

## Setup
See `docs/onbboarding.md` if more details are needed.
### Quickstart (local)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
Create .env (do not commit):
echo "FRED_API_KEY=your_key_here" > .env
Then run (once implemented):
# TODO: add runner/CLI entrypoint
```

## Expected outputs
### Cleaned datasets (internal)
* `data/cleaned/prices_daily.parquet`
* `data/cleaned/holdings.parquet`
* `data/cleaned/universe.parquet`

### Exports (for dashboards)
* `outputs/portfolio_summary.csv`
* `outputs/positions_current.csv`

Sheets should **import** exports (e.g., `IMPORTDATA`) rather than recompute core logic.

### Secrets

API keys go in environment variables (`.env`), never in `settings.yaml` or committed files.

## Packages
* `pandas`
* `pyyaml`
* `python-dotenv`
* `yfinance` (prices)
* `fredapi` (macro, via FRED)
