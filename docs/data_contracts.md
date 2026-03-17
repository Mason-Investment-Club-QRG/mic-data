# Daily Returns Data Contracts

This document defines canonical schemas for the WRDS daily returns pipeline.

## Units Convention
- Returns are decimal values (`0.01 == 1%`).
- Shares are raw counts.
- Timestamps are UTC where explicitly marked (`*_ts_utc`).

## `universe_daily`
Purpose: one row per tracked ticker per as-of date, combining holdings and watchlist.

Columns:
- `as_of_date` (`datetime64[ns]`, non-null)
- `ticker` (`string`, non-null, uppercase)
- `is_holding` (`bool`, non-null)
- `is_watchlist` (`bool`, non-null)
- `shares` (`float64`, nullable)
- `name` (`string`, nullable)
- `sector` (`string`, nullable)

Natural key:
- (`as_of_date`, `ticker`)

Rules:
- At least one of `is_holding` or `is_watchlist` must be `true`.

## `security_returns_daily`
Purpose: WRDS/CRSP daily security returns and market fields for tracked universe tickers.

Columns:
- `trade_date` (`datetime64[ns]`, non-null)
- `ticker` (`string`, non-null)
- `permno` (`int64`, non-null)
- `ret` (`float64`, nullable)
- `prc` (`float64`, nullable)
- `vol` (`float64`, nullable)
- `shrout` (`float64`, nullable)
- `source` (`string`, non-null; expected `wrds_crsp`)
- `load_ts_utc` (`datetime64[ns, UTC]`, non-null)

Natural key:
- (`trade_date`, `permno`)

Rules:
- `ret` is CRSP total return decimal field.

## `portfolio_returns_daily`
Purpose: daily club portfolio return series from holdings-weighted aggregation.

Columns:
- `trade_date` (`datetime64[ns]`, non-null)
- `portfolio_ret` (`float64`, nullable)
- `n_constituents` (`int64`, non-null)
- `gross_exposure` (`float64`, non-null)
- `method` (`string`, non-null; expected `holdings_weighted_sum`)
- `load_ts_utc` (`datetime64[ns, UTC]`, non-null)

Natural key:
- (`trade_date`)

Rules:
- `portfolio_ret` remains decimal return.

## `daily_qa`
Purpose: stage-level operational QA and error visibility.

Columns:
- `run_date` (`date`, non-null)
- `stage` (`string`, non-null)
- `rows_universe` (`int64`, non-null)
- `rows_returns` (`int64`, non-null)
- `missing_permno` (`int64`, non-null)
- `missing_ret` (`int64`, non-null)
- `status` (`string`, non-null)
- `error_message` (`string`, nullable)

Natural key:
- (`run_date`, `stage`)

Status values:
- `ok`
- `error`
- `warn` (reserved for future use)

## Idempotency Contract
- Deterministic sort before write.
- Atomic file write (temp + replace).
- Stage locks in `outputs/locks`.
- Manifest file in `outputs/manifests/daily_returns_<date>.json` with dataset hashes.
- Re-runs should produce stable hashes for unchanged inputs.
