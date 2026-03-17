# Onboarding

## What this repo does
- Syncs holdings and watchlist inputs from Google Sheets.
- Builds a canonical daily universe dataset.
- Pulls daily returns from WRDS/CRSP.
- Aggregates daily portfolio returns.
- Publishes daily outputs to Google Sheets.
- Runs monthly FF3 model analysis.
- Builds portfolio analytics tables and SVG charts from persisted outputs.

## Quickstart

### 0) Prereqs
- Python 3.11+ recommended
- WRDS account access
- Google service-account key with spreadsheet access

### 1) Clone + create virtual environment
```bash
git clone https://github.com/Mason-Investment-Club-QRG/mic-data.git
cd mic-data
python3 -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
```

### 3) Configure credentials
Create one local secrets file:

```bash
cp .env.example .env
```

Edit `.env`:

```bash
GOOGLE_APPLICATION_CREDENTIALS=secrets/your_service_account.json
WRDS_USERNAME=your_wrds_username
WRDS_PASSWORD=your_wrds_password
```

Optional: load `.env` into current shell.
```bash
set -a; source .env; set +a
```

### 4) Validate config files
- `config/google_sheets.yaml`
- `config/positions.yaml`
- `config/returns_daily.yaml`
- `config/analytics.yaml`

### 5) Run daily stages
```bash
PYTHONPATH=src python -m mic_data.positions.sync --config config/positions.yaml --if-exists replace
PYTHONPATH=src python -m mic_data.positions.universe_sync --config config/positions.yaml --if-exists replace
PYTHONPATH=src python -m mic_data.market.pull_wrds_returns --config config/returns_daily.yaml --if-exists replace
PYTHONPATH=src python -m mic_data.market.build_portfolio_returns --config config/returns_daily.yaml --if-exists replace
PYTHONPATH=src python -m mic_data.reporting.publish_sheets --config config/returns_daily.yaml --skip-unchanged
```

### 6) Optional run-all wrapper
```bash
PYTHONPATH=src python -m mic_data.market.prices_daily run-all --config config/returns_daily.yaml
```

Monthly FF3 analysis should read the persisted daily outputs from
`mic_data.models.ff_factor_matrix` rather than pulling a second price source.

Example:
```bash
PYTHONPATH=src .venv/bin/python -m mic_data.models.ff_factor_matrix \
  --start-date 2025-01-01 \
  --end-date 2025-12-31
```

Portfolio analytics should also read the persisted daily outputs. Add `SPY` to the
Google Sheets `Universe` tab first so the benchmark is present in
`security_returns_daily`.

Example:
```bash
PYTHONPATH=src .venv/bin/python -m mic_data.analytics.report \
  --config config/analytics.yaml \
  --start-date 2025-01-01 \
  --end-date 2025-12-31
```

### 7) Run tests
```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s test/models -p 'test_*.py'
```

## Analytics Limitation
- Portfolio performance, beta, and Sharpe in the analytics layer are based on the
  pipeline's `holdings_weighted_sum` proxy series.
- The analytics outputs do not yet reflect historical portfolio composition changes.
