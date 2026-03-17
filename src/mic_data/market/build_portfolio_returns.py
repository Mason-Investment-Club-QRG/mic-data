# Purpose: Build daily portfolio returns from WRDS security returns and holdings share counts.

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Purpose: Make package imports work when this file is executed directly by path.
if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mic_data.contracts.daily_returns_contracts import (
    PORTFOLIO_RETURNS_COLUMNS,
    QaRow,
    validate_portfolio_returns_daily,
    validate_security_returns_daily,
    validate_universe_daily,
)
from mic_data.market.prices_daily import (
    DailyReturnsConfig,
    append_stage_log,
    load_daily_returns_config,
    parse_write_mode,
    pipeline_run_date,
    stage_lock_path,
    upsert_daily_qa_row,
    write_daily_manifest,
)
from mic_data.utils.idempotent_io import atomic_write_csv, atomic_write_parquet, stage_lock


# Purpose: Apply CLI overrides to immutable stage configuration.
def _apply_overrides(
    config: DailyReturnsConfig,
    *,
    start_date: str | None,
    end_date: str | None,
    if_exists: str | None,
    dry_run: bool,
) -> DailyReturnsConfig:
    resolved_if_exists = parse_write_mode(if_exists) if if_exists else config.if_exists
    resolved_dry_run = True if dry_run else config.dry_run

    return DailyReturnsConfig(
        start_date=start_date or config.start_date,
        end_date=end_date or config.end_date,
        google_sheets_config_path=config.google_sheets_config_path,
        positions_config_path=config.positions_config_path,
        paths=config.paths,
        wrds_username=config.wrds_username,
        if_exists=resolved_if_exists,
        dry_run=resolved_dry_run,
    )


# Purpose: Load and validate persisted universe dataset used for holdings and ticker coverage.
def _load_universe(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Universe dataset not found: {path}. Run 'python -m mic_data.positions.universe_sync' first."
        )
    return validate_universe_daily(pd.read_parquet(path))


# Purpose: Load and validate persisted security returns dataset required for aggregation.
def _load_security_returns(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Security returns dataset not found: {path}. Run 'python -m mic_data.market.pull_wrds_returns' first."
        )
    return validate_security_returns_daily(pd.read_parquet(path))


# Purpose: Build normalized portfolio weights from holdings shares and latest available WRDS prices.
def _build_weights(holdings: pd.DataFrame, security_returns: pd.DataFrame) -> pd.DataFrame:
    latest_price = (
        security_returns.sort_values(["ticker", "trade_date"])
        .groupby("ticker", as_index=False)
        .tail(1)[["ticker", "prc"]]
        .copy()
    )
    latest_price["price_abs"] = latest_price["prc"].abs()

    weights = holdings.merge(latest_price[["ticker", "price_abs"]], on="ticker", how="left")
    weights["position_value"] = weights["shares"] * weights["price_abs"]
    weights = weights.dropna(subset=["position_value"])

    total_value = float(weights["position_value"].sum())
    if total_value <= 0:
        raise ValueError("Holdings position value is non-positive; cannot build portfolio weights.")

    weights["weight"] = weights["position_value"] / total_value
    return weights[["ticker", "weight"]]


# Purpose: Aggregate security-level returns into daily portfolio returns using fixed holdings weights.
def _aggregate_portfolio_returns(
    *,
    security_returns: pd.DataFrame,
    weights: pd.DataFrame,
) -> pd.DataFrame:
    merged = security_returns.merge(weights, on="ticker", how="inner")
    merged = merged.copy()
    merged["weighted_ret"] = merged["ret"] * merged["weight"]

    grouped = merged.groupby("trade_date", as_index=False)
    out = grouped.agg(
        portfolio_ret=("weighted_ret", lambda s: s.sum(min_count=1)),
        n_constituents=("ret", "count"),
    )

    gross_exposure = float(weights["weight"].abs().sum())
    out["gross_exposure"] = gross_exposure
    out["method"] = "holdings_weighted_sum"
    out["load_ts_utc"] = pd.Timestamp.now(tz="UTC")
    out = out[list(PORTFOLIO_RETURNS_COLUMNS)]

    return validate_portfolio_returns_daily(out)


# Purpose: Execute portfolio return aggregation stage with QA/log/manifest outputs.
def run_build_portfolio_returns_stage(
    *,
    config: DailyReturnsConfig,
) -> pd.DataFrame:
    """Run portfolio return aggregation stage.

    Inputs:
      - config: Daily pipeline config.

    Returns:
      - Contract-validated portfolio returns dataframe.

    Raises:
      - RuntimeError/ValueError for missing inputs, validation, or write failures.

    Notes on units:
      - `portfolio_ret` is decimal return.
    """

    stage_name = "build_portfolio_returns"
    run_date = pipeline_run_date()
    lock_path = stage_lock_path(config, stage_name=stage_name)

    try:
        with stage_lock(lock_path):
            universe = _load_universe(config.paths.universe_latest_path)
            security_returns = _load_security_returns(config.paths.security_returns_path)

            holdings = universe[universe["is_holding"]].copy()
            if holdings.empty:
                raise ValueError("No holdings rows found in universe; cannot compute portfolio returns.")

            weights = _build_weights(holdings, security_returns)
            portfolio_returns = _aggregate_portfolio_returns(
                security_returns=security_returns,
                weights=weights,
            )

            parquet_write = None
            csv_write = None
            if not config.dry_run:
                parquet_write = atomic_write_parquet(
                    portfolio_returns,
                    path=config.paths.portfolio_returns_path,
                    sort_by=["trade_date"],
                    mode=config.if_exists,
                )
                csv_write = atomic_write_csv(
                    portfolio_returns,
                    path=config.paths.portfolio_returns_csv_path,
                    sort_by=["trade_date"],
                    mode=config.if_exists,
                )

            holdings_tickers = set(holdings["ticker"].astype(str).tolist())
            return_tickers = set(security_returns["ticker"].astype(str).tolist())
            missing_permno_count = len(holdings_tickers - return_tickers)
            missing_ret_count = int(security_returns["ret"].isna().sum())

            qa_row = QaRow(
                run_date=run_date,
                stage=stage_name,
                rows_universe=len(holdings),
                rows_returns=len(portfolio_returns),
                missing_permno=missing_permno_count,
                missing_ret=missing_ret_count,
                status="ok",
                error_message=None,
            )
            upsert_daily_qa_row(config=config, qa_row=qa_row)

            dataset_payload: dict[str, dict[str, object]] = {
                "portfolio_returns_daily": {
                    "path": str(config.paths.portfolio_returns_path),
                    "rows": len(portfolio_returns),
                    "hash": parquet_write.content_hash if parquet_write else "dry_run",
                    "csv_hash": csv_write.content_hash if csv_write else "dry_run",
                }
            }
            write_daily_manifest(config=config, as_of_date=run_date, datasets=dataset_payload)

            append_stage_log(
                config=config,
                stage=stage_name,
                status="ok",
                row_count=len(portfolio_returns),
                detail={
                    "holdings_count": len(holdings),
                    "dry_run": config.dry_run,
                },
            )
            return portfolio_returns
    except Exception as exc:
        append_stage_log(
            config=config,
            stage=stage_name,
            status="error",
            row_count=None,
            detail={"error_type": type(exc).__name__, "error_message": str(exc)},
        )
        qa_row = QaRow(
            run_date=run_date,
            stage=stage_name,
            rows_universe=0,
            rows_returns=0,
            missing_permno=0,
            missing_ret=0,
            status="error",
            error_message=str(exc),
        )
        upsert_daily_qa_row(config=config, qa_row=qa_row)
        raise


# Purpose: Parse CLI arguments for stage execution.
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build daily portfolio returns from WRDS security returns.")
    parser.add_argument("--config", default="config/returns_daily.yaml", help="Daily returns config")
    parser.add_argument("--start-date", default=None, help="Override start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", default=None, help="Override end date (YYYY-MM-DD)")
    parser.add_argument(
        "--if-exists",
        choices=["replace", "skip", "error"],
        default=None,
        help="Write behavior for stage outputs",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate aggregation without writes")
    return parser.parse_args()


# Purpose: CLI entrypoint for portfolio return build stage.
def main() -> None:
    """Run portfolio returns stage CLI.

    Inputs:
      - CLI args from `_parse_args`.

    Returns:
      - None.

    Raises:
      - Propagates stage execution errors.

    Notes on units:
      - `portfolio_ret` output values are decimals.
    """

    args = _parse_args()
    config = load_daily_returns_config(args.config)
    config = _apply_overrides(
        config,
        start_date=args.start_date,
        end_date=args.end_date,
        if_exists=args.if_exists,
        dry_run=bool(args.dry_run),
    )

    frame = run_build_portfolio_returns_stage(config=config)
    print(f"Portfolio returns build complete rows={len(frame)}")
    print(f"output_parquet={config.paths.portfolio_returns_path}")
    print(f"output_csv={config.paths.portfolio_returns_csv_path}")


if __name__ == "__main__":
    main()
