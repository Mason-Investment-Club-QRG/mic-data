# MIC Data Pipeline

Data engineering pipeline for Mason Investment Club with a WRDS-first modeling stack and a WRDS-only daily returns stack.

## What It Does
- Syncs holdings input from Google Sheets.
- Builds a canonical holdings + watchlist universe dataset.
- Pulls daily security returns from WRDS/CRSP (`ret` as decimal total return).
- Aggregates daily portfolio returns from holdings weights.
- Publishes curated daily outputs back to Google Sheets.
- Runs FF3 monthly modeling with WRDS factors (static-file fallback remains available in FF3 module).

## Pipeline Architecture
### Daily Returns (stage-based, idempotent)
1. `positions.sync`
2. `positions.universe_sync`
3. `market.pull_wrds_returns`
4. `market.build_portfolio_returns`
5. `reporting.publish_sheets`

Optional wrapper:
- `market.prices_daily run-all` (convenience only)

### Monthly FF3 (existing model path)
1. `positions.sync`
2. `portfolio.holdings`
3. `models.fama_french_3`

## Key Config Files
- `config/google_sheets.yaml`
  - Single source of truth for sheet ID, input tabs, and output tabs.
- `config/positions.yaml`
  - Holdings/watchlist column mappings and universe output locations.
- `config/returns_daily.yaml`
  - Daily returns date window, source paths, output paths, and idempotency defaults.
- `config/ff3_pipeline.yaml`
  - FF3 monthly model settings.

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Credentials
Use one local file for all secrets:

```bash
cp .env.example .env
```

Then edit `.env` once:

```bash
GOOGLE_APPLICATION_CREDENTIALS=secrets/your_service_account.json
WRDS_USERNAME=your_wrds_username
WRDS_PASSWORD=your_wrds_password
FRED_API_KEY=your_fred_key
```

The pipeline auto-loads `.env`, so you do not need to export each variable manually.

If you want shell exports for the current terminal session:
```bash
set -a; source .env; set +a
```

## Run Daily Stages (recommended)
```bash
PYTHONPATH=src python -m mic_data.positions.sync --config config/positions.yaml --if-exists replace
PYTHONPATH=src python -m mic_data.positions.universe_sync --config config/positions.yaml --if-exists replace
PYTHONPATH=src python -m mic_data.market.pull_wrds_returns --config config/returns_daily.yaml --if-exists replace
PYTHONPATH=src python -m mic_data.market.build_portfolio_returns --config config/returns_daily.yaml --if-exists replace
PYTHONPATH=src python -m mic_data.reporting.publish_sheets --config config/returns_daily.yaml --skip-unchanged
```

### Optional wrapper
```bash
PYTHONPATH=src python -m mic_data.market.prices_daily run-all --config config/returns_daily.yaml
```

Force a one-off window without editing YAML:
```bash
PYTHONPATH=src python -m mic_data.market.prices_daily run-all --config config/returns_daily.yaml --start-date 2026-02-28 --end-date 2026-03-03
```

## Idempotency Rules
- Natural keys:
  - `universe_daily`: `(as_of_date, ticker)`
  - `security_returns_daily`: `(trade_date, permno)`
  - `portfolio_returns_daily`: `(trade_date)`
  - `daily_qa`: `(run_date, stage)`
- Atomic writes: temp file then rename.
- Stable ordering before write for deterministic hashes.
- Write mode control: `--if-exists replace|skip|error`.
- Optional no-mutation validation: `--dry-run`.
- Stage locks: `outputs/locks/<stage>.lock`.
- Manifest output: `outputs/manifests/daily_returns_<date>.json`.
- Google Sheet publish row cap: `config/google_sheets.yaml -> google_sheets.options.max_rows` (default `50000`).

## Outputs
### Daily
- `data/processed/universe_latest.parquet`
- `data/processed/returns/security_returns_daily.parquet`
- `data/processed/returns/security_returns_daily.csv`
- `data/processed/returns/portfolio_returns_daily.parquet`
- `data/processed/returns/portfolio_returns_daily.csv`
- `data/processed/returns/daily_qa.parquet`
- `outputs/manifests/daily_returns_<date>.json`
- `outputs/logs/daily_returns_<date>.jsonl`

### Monthly FF3
- `data/processed/model_inputs/factors_wrds_m.parquet`
- `data/processed/model_inputs/factors_static_m.parquet`
- `outputs/validation/ff3_input_comparison.csv`
- `outputs/validation/ff3_regression_comparison.json`
- `outputs/validation/ff3_validation_summary.md`

## Tests
```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s test/models -p 'test_*.py'
```

## GitHub Automation
Workflow file:
- `.github/workflows/daily_returns.yml`

Required GitHub Secrets:
- `WRDS_USERNAME`
- `WRDS_PASSWORD`
- `GOOGLE_SERVICE_ACCOUNT_JSON`

Automation behavior:
- Scheduled weekdays in a UTC window and ET-gated to the 5:00 PM ET run target.
- Stage-based execution with fail-fast behavior.
- Uploads logs/manifests as artifacts.

## Important Note on WRDS Return Field
Daily security returns use CRSP `ret`.
- This is a decimal total return field.
- Example: `0.01` means `+1.00%` for that day.
