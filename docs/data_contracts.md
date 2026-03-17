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

## `benchmark_comparison`
Purpose: aligned portfolio proxy and benchmark return series with cumulative value and drawdown fields.

Columns:
- `trade_date` (`datetime64[ns]`, non-null)
- `portfolio_ret` (`float64`, nullable)
- `benchmark_ret` (`float64`, nullable)
- `active_ret` (`float64`, nullable)
- `portfolio_value` (`float64`, non-null; base 100 curve)
- `benchmark_value` (`float64`, non-null; base 100 curve)
- `portfolio_drawdown` (`float64`, nullable)
- `benchmark_drawdown` (`float64`, nullable)
- `n_constituents` (`int64`, non-null)
- `gross_exposure` (`float64`, non-null)
- `method` (`string`, non-null)

## `current_holdings_snapshot`
Purpose: latest holdings composition with current weights from persisted prices and CRSP size fields.

Columns:
- `as_of_date` (`datetime64[ns]`, non-null)
- `latest_trade_date` (`datetime64[ns]`, non-null)
- `ticker` (`string`, non-null)
- `name` (`string`, nullable)
- `sector` (`string`, nullable)
- `shares` (`float64`, non-null)
- `latest_price` (`float64`, non-null)
- `latest_market_cap_usd` (`float64`, non-null)
- `market_cap_bucket` (`string`, non-null)
- `position_value` (`float64`, non-null)
- `portfolio_weight` (`float64`, non-null)
- `portfolio_weight_pct` (`float64`, non-null)
- `weight_rank` (`int64`, non-null)

## `market_cap_mix`
Purpose: market-cap bucket aggregation of the current holdings snapshot.

Columns:
- `market_cap_bucket` (`string`, non-null)
- `bucket_order` (`int64`, non-null)
- `constituent_count` (`int64`, non-null)
- `portfolio_weight` (`float64`, non-null)
- `portfolio_weight_pct` (`float64`, non-null)
- `position_value` (`float64`, non-null)

## `beta_regression`
Purpose: sampled benchmark and portfolio proxy return pairs used for the beta regression chart.

Columns:
- `trade_date` (`datetime64[ns]`, non-null)
- `portfolio_ret` (`float64`, nullable)
- `benchmark_ret` (`float64`, nullable)
- `fitted_portfolio_ret` (`float64`, nullable)

## `ff3/security_loadings`
Purpose: per-security FF3 regression outputs for all modeled securities in the analytics window.

Columns:
- `ticker` (`string`, non-null)
- `alpha` (`float64`, nullable)
- `mkt_rf` (`float64`, nullable)
- `smb` (`float64`, nullable)
- `hml` (`float64`, nullable)
- `r2` (`float64`, nullable)
- `n_obs` (`int64`, nullable)
- `residual_var` (`float64`, nullable)
- `explained_var` (`float64`, nullable)
- `explained_var_ratio` (`float64`, nullable)

## `ff3/portfolio_exposure_comparison`
Purpose: return-based and holdings-based FF3 portfolio exposures used in comparison charts.

Columns:
- `exposure_method` (`string`, non-null)
- `alpha` (`float64`, nullable)
- `mkt_rf` (`float64`, nullable)
- `smb` (`float64`, nullable)
- `hml` (`float64`, nullable)

## `ff3/factor_risk_contributions`
Purpose: FF3 variance contribution by factor for the portfolio proxy return series.

Columns:
- `factor` (`string`, non-null)
- `variance_contribution` (`float64`, nullable)

## `ff3/holdings_ff3_loadings`
Purpose: current holdings snapshot enriched with FF3 loadings for the holdings heatmap.

Columns:
- `weight_rank` (`int64`, non-null)
- `ticker` (`string`, non-null)
- `name` (`string`, nullable)
- `portfolio_weight` (`float64`, non-null)
- `portfolio_weight_pct` (`float64`, non-null)
- `ff3_modeled` (`bool`, non-null)
- `ff3_excluded_reason` (`string`, nullable)
- `alpha` (`float64`, nullable)
- `mkt_rf` (`float64`, nullable)
- `smb` (`float64`, nullable)
- `hml` (`float64`, nullable)
- `r2` (`float64`, nullable)
- `n_obs` (`int64`, nullable)
- `explained_var_ratio` (`float64`, nullable)

## Idempotency Contract
- Deterministic sort before write.
- Atomic file write (temp + replace).
- Stage locks in `outputs/locks`.
- Manifest file in `outputs/manifests/daily_returns_<date>.json` with dataset hashes.
- Re-runs should produce stable hashes for unchanged inputs.
