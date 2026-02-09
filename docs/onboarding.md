# Onboarding

## What this repo does
- Produces clean tables for prices, holdings, macro data
- Exports small CSV summaries for dashboards (Google Sheets / Excel / BI)

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

Create a `.env` file in the repo root:

```bash
cat > .env <<'EOF'
FRED_API_KEY=your_key_here
EOF
```

### 4) Configure tickers + settings

Edit:

* `config/universe.csv`
* `config/settings.yaml`

### 5) Run the pipeline

Pipeline not implemented yet.
```bash
```
