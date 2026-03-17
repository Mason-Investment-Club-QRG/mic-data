# MIC Data Pipeline

Data engineering pipeline for Mason Investment Club with a WRDS-first modeling stack and a WRDS-only daily returns stack.

## What It Does
- Syncs holdings input from Google Sheets.
- Builds a canonical holdings + watchlist universe dataset.
- Pulls daily security returns from WRDS/CRSP (`ret` as decimal total return).
- Aggregates daily portfolio returns from holdings weights.
- Publishes curated daily outputs back to Google Sheets.
- Runs FF3 analysis from persisted daily pipeline outputs plus WRDS factor data.
- Builds benchmark/composition analytics tables and SVG charts from persisted outputs.
- Extends the analytics report with FF3 exposure, risk, and holdings-level factor charts.

## Pipeline Architecture
### Daily Returns (stage-based, idempotent)
1. `positions.sync`
2. `positions.universe_sync`
3. `market.pull_wrds_returns`
4. `market.build_portfolio_returns`
5. `reporting.publish_sheets`

Optional wrapper:
- `market.prices_daily run-all` (convenience only)

### Monthly FF3 Analysis
1. Run the daily returns stages so `security_returns_daily` and `portfolio_returns_daily` exist.
2. Run `mic_data.models.ff_factor_matrix` against those persisted datasets.

Example:
```bash
PYTHONPATH=src .venv/bin/python -m mic_data.models.ff_factor_matrix \
  --start-date 2025-01-01 \
  --end-date 2025-12-31 \
  --output-json outputs/validation/ff3_summary_2025.json
```

### Portfolio Analytics
1. Add `SPY` to the Google Sheets `Universe` tab so it is included in `security_returns_daily`.
2. Run the daily returns stages so the persisted datasets are current.
3. Run `mic_data.analytics.report` against those persisted datasets.
4. Review both the benchmark/composition charts and the FF3 model charts written to `outputs/charts/`.

Example:
```bash
PYTHONPATH=src .venv/bin/python -m mic_data.analytics.report \
  --config config/analytics.yaml \
  --start-date 2025-01-01 \
  --end-date 2025-12-31
```

## Key Config Files
- `config/google_sheets.yaml`
  - Single source of truth for sheet ID, input tabs, and output tabs.
- `config/positions.yaml`
  - Holdings/watchlist column mappings and universe output locations.
- `config/returns_daily.yaml`
  - Daily returns date window, source paths, output paths, and idempotency defaults.
- `config/analytics.yaml`
  - Benchmark ticker, beta frequency, analytics output paths, and chart settings.

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Fastest Way To Run
From the repo root, use the wrapper script:

```bash
./scripts/mic help
```

Common commands:

```bash
./scripts/mic daily --config config/returns_daily.yaml
./scripts/mic analytics --config config/analytics.yaml --start-date 2025-01-01 --end-date 2025-12-31
./scripts/mic ff3 --start-date 2025-01-01 --end-date 2025-12-31 --output-json outputs/validation/ff3_summary_2025.json
./scripts/mic module mic_data.positions.sync --config config/positions.yaml --if-exists replace
./scripts/mic test
```

This wrapper automatically uses `.venv/bin/python` when available and sets `PYTHONPATH=src` for you.

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

Run portfolio analytics after the daily refresh:
```bash
PYTHONPATH=src .venv/bin/python -m mic_data.analytics.report \
  --config config/analytics.yaml \
  --start-date 2025-01-01 \
  --end-date 2025-12-31
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

### Analytics
- `data/processed/analytics/benchmark_comparison.parquet`
- `data/processed/analytics/current_holdings_snapshot.parquet`
- `data/processed/analytics/market_cap_mix.parquet`
- `data/processed/analytics/beta_regression.parquet`
- `data/processed/analytics/ff3/security_loadings.parquet`
- `data/processed/analytics/ff3/portfolio_exposure_comparison.parquet`
- `data/processed/analytics/ff3/factor_risk_contributions.parquet`
- `data/processed/analytics/ff3/holdings_ff3_loadings.parquet`
- `outputs/analytics/portfolio_dashboard_summary.json`
- `outputs/charts/performance_vs_spy.svg`
- `outputs/charts/beta_vs_spy.svg`
- `outputs/charts/top_holdings.svg`
- `outputs/charts/market_cap_mix.svg`
- `outputs/charts/sharpe_ratio.svg`
- `outputs/charts/ff3_portfolio_exposure.svg`
- `outputs/charts/ff3_exposure_comparison.svg`
- `outputs/charts/ff3_factor_risk_contributions.svg`
- `outputs/charts/ff3_security_heatmap.svg`

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

## Important Note on Portfolio Analytics
- Performance, beta, and Sharpe analytics use the pipeline's holdings-weighted proxy return series.
- That means the current holdings snapshot is applied backward across the requested history.
- Historical portfolio composition changes are not yet modeled.
- The FF3 charts inherit the same limitation: return-based FF3 results are estimated from that proxy series, and holdings-based FF3 results use the current holdings weights.
