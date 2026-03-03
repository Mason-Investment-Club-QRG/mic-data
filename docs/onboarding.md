# Onboarding

## What this repo does
- Syncs current positions from Google Sheets
- Builds latest holdings weights from market prices
- Runs FF3 regression with WRDS-first factor pulls and static fallback
- Writes validation artifacts comparing WRDS vs static factor inputs

## Quickstart

### 0) Prereqs
- Python 3.11+ recommended
- macOS/Linux terminal (Windows works too)

### 1) Clone + create a virtual environment
```bash
git clone https://github.com/Mason-Investment-Club-QRG/mic-data.git
cd mic-data

python3 -m venv .venv
source .venv/bin/activate
````
### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Set secrets (do NOT commit)

Create a `.env` file in the repo root and export service-account credentials:

```bash
cat > .env <<'EOF'
FRED_API_KEY=your_key_here
EOF

source scripts/env.sh
```

### 4) Configure data + model settings

Edit:

* `config/positions.yaml`
* `config/ff3_pipeline.yaml`

### 5) Run the pipeline

```bash
PYTHONPATH=src python -m mic_data.positions.sync
PYTHONPATH=src python -m mic_data.portfolio.holdings
PYTHONPATH=src python -m mic_data.models.fama_french_3
```
