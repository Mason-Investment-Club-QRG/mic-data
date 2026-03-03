# MIC Data Pipeline

Portfolio data pipeline for Mason Investment Club with WRDS-first FF3 modeling.

## What It Does
- Syncs current positions from Google Sheets into canonical CSV snapshots.
- Builds latest holdings weights from market prices.
- Runs FF3 regressions using WRDS factor pulls as the primary source.
- Falls back to static Ken French factors when WRDS fails.
- Produces A/B validation artifacts comparing WRDS vs static inputs.

## Pipeline Flow
1. `positions.sync` -> `data/processed/positions_latest.csv`
2. `portfolio.holdings` -> `data/processed/holdings_latest.csv`
3. `models.fama_french_3` -> factor/model artifacts + validation report

## Repo Layout
- `config/positions.yaml`: Google Sheet mapping and output paths.
- `config/ff3_pipeline.yaml`: FF3 run dates, paths, and fallback options.
- `src/mic_data/positions/`: positions sync logic.
- `src/mic_data/portfolio/`: holdings + weights logic.
- `src/mic_data/models/`: FF3 modular pipeline (sources, regression, comparison, runner).
- `data/`: generated data artifacts (ignored in git).
- `outputs/`: logs and validation exports (ignored in git).

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Credentials
- Google Sheets:
```bash
source scripts/env.sh
```
or set `GOOGLE_APPLICATION_CREDENTIALS` manually.

- WRDS (recommended non-interactive):
```bash
export WRDS_USERNAME="your_wrds_username"
```
On first WRDS use, create `.pgpass` when prompted.

## Run
```bash
PYTHONPATH=src python -m mic_data.positions.sync
PYTHONPATH=src python -m mic_data.portfolio.holdings
PYTHONPATH=src python -m mic_data.models.fama_french_3
```

Optional strict mode (no fallback if WRDS fails):
```bash
PYTHONPATH=src python -m mic_data.models.fama_french_3 --no-fallback
```

## Outputs
### Model Inputs
- `data/processed/model_inputs/factors_wrds_m.parquet`
- `data/processed/model_inputs/factors_static_m.parquet`

### Validation
- `outputs/validation/ff3_input_comparison.csv`
- `outputs/validation/ff3_regression_comparison.json`
- `outputs/validation/ff3_validation_summary.md`
- `outputs/logs/ff3_pipeline_<timestamp>.jsonl`

## Tests
```bash
.venv/bin/python -m unittest discover -s test/models -p 'test_*.py'
```

## Notes
- All model returns/factors are treated as decimal returns.
- Keep secrets out of git (`.env`, service-account JSON, WRDS credentials).
- See `docs/onboarding.md` for additional project onboarding details.
