# MIC Data Pipeline

A durable data pipeline for Mason Investment Club

**Goals:**
- Read **positions/holdings** from Google Sheets
- Pull market data for portfolio analytics
- Run FF3 regression with WRDS as primary factor source
- Produce validation artifacts comparing WRDS vs static factor inputs


## Repo layout

- `config/`
  - `positions.yaml` — Google Sheet -> canonical positions mapping
  - `ff3_pipeline.yaml` — FF3 run settings (dates, source paths, outputs)
- `src/mic_data/` — pipeline code
- `data/` — raw + cleaned datasets (not committed)
- `outputs/` — exported CSVs for Sheets / BI (not committed)
- `docs/` — data contracts + onboarding

## Setup
See `docs/onboarding.md` if more details are needed.
### Quickstart (local)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create `.env` (do not commit) and export sheet credentials as needed:

```bash
echo "FRED_API_KEY=your_key_here" > .env
source scripts/env.sh
```

### FF3 pipeline run

```bash
# Optional: export WRDS username for non-interactive auth
export WRDS_USERNAME="your_wrds_username"

# Uses config/ff3_pipeline.yaml if present
PYTHONPATH=src python -m mic_data.models.fama_french_3
```

## Expected outputs
### FF3 model inputs
* `data/processed/model_inputs/factors_wrds_m.parquet`
* `data/processed/model_inputs/factors_static_m.parquet`

### Validation artifacts
* `outputs/validation/ff3_input_comparison.csv`
* `outputs/validation/ff3_regression_comparison.json`
* `outputs/validation/ff3_validation_summary.md`
* `outputs/logs/ff3_pipeline_<timestamp>.jsonl`

### Secrets

API keys and credentials go in environment variables or secret files, never in committed config.

## Packages
* `pandas`
* `pyyaml`
* `python-dotenv`
* `yfinance` (prices)
* `fredapi` (macro, via FRED)
* `statsmodels` (regression)
* `wrds` (factor pulls)
* `pyarrow` (parquet artifacts)
