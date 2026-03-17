from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from mic_data.contracts.daily_returns_contracts import (
    validate_portfolio_returns_daily,
    validate_security_returns_daily,
    validate_universe_daily,
)
from mic_data.models.ff_factor_matrix import (
    DEFAULT_PORTFOLIO_RETURNS_PATH,
    DEFAULT_SECURITY_RETURNS_PATH,
    FF3AnalysisResult,
    analysis_summary_payload,
    load_pipeline_return_inputs,
    normalize_ff3_factors,
    run_ff3_factor_analysis,
)


DEFAULT_UNIVERSE_PATH = Path("data/processed/universe_latest.parquet")
CAP_BUCKET_ORDER = ("Mega Cap", "Large Cap", "Mid Cap", "Small Cap", "Micro Cap")


@dataclass(frozen=True)
class AnalyticsInputs:
    """Canonical persisted datasets consumed by the portfolio analytics layer."""

    universe: pd.DataFrame
    security_returns: pd.DataFrame
    portfolio_returns: pd.DataFrame


@dataclass(frozen=True)
class PortfolioAnalyticsResult:
    """Reusable analytics outputs for charts, exports, and downstream publishing."""

    benchmark_comparison: pd.DataFrame
    beta_regression: pd.DataFrame
    holdings_snapshot: pd.DataFrame
    holdings_ff3_loadings: pd.DataFrame
    market_cap_mix: pd.DataFrame
    ff3_analysis: FF3AnalysisResult
    summary: dict[str, object]


def load_analytics_inputs(
    *,
    universe_path: str | Path = DEFAULT_UNIVERSE_PATH,
    security_returns_path: str | Path = DEFAULT_SECURITY_RETURNS_PATH,
    portfolio_returns_path: str | Path = DEFAULT_PORTFOLIO_RETURNS_PATH,
    start_date: str | None = None,
    end_date: str | None = None,
) -> AnalyticsInputs:
    """Load canonical persisted datasets required for portfolio analytics."""

    universe = validate_universe_daily(_read_tabular(Path(universe_path)))
    inputs = load_pipeline_return_inputs(
        security_returns_path=security_returns_path,
        portfolio_returns_path=portfolio_returns_path,
        start_date=start_date,
        end_date=end_date,
    )
    return AnalyticsInputs(
        universe=universe.reset_index(drop=True),
        security_returns=inputs.security_returns.reset_index(drop=True),
        portfolio_returns=inputs.portfolio_returns.reset_index(drop=True),
    )


def build_portfolio_analytics(
    *,
    inputs: AnalyticsInputs,
    factors: pd.DataFrame,
    benchmark_ticker: str = "SPY",
    beta_frequency: str = "weekly",
    ff3_min_obs: int = 60,
    trading_days_per_year: int = 252,
    top_n_holdings: int = 5,
    output_value_base: float = 100.0,
) -> PortfolioAnalyticsResult:
    """Build portfolio analytics tables and summary metadata from persisted inputs."""

    universe = validate_universe_daily(inputs.universe)
    security_returns = validate_security_returns_daily(inputs.security_returns)
    portfolio_returns = validate_portfolio_returns_daily(inputs.portfolio_returns)
    factor_frame = normalize_ff3_factors(factors)

    benchmark_ticker_clean = str(benchmark_ticker).strip().upper()
    comparison = _build_benchmark_comparison(
        portfolio_returns=portfolio_returns,
        security_returns=security_returns,
        benchmark_ticker=benchmark_ticker_clean,
        output_value_base=output_value_base,
    )
    beta_regression, beta_summary = _build_beta_regression(
        comparison=comparison,
        frequency=beta_frequency,
    )
    sharpe_summary = _build_sharpe_summary(
        portfolio_returns=portfolio_returns,
        factors=factor_frame,
        trading_days_per_year=trading_days_per_year,
    )
    holdings_snapshot = _build_current_holdings_snapshot(
        universe=universe,
        security_returns=security_returns,
    )
    ff3_analysis = run_ff3_factor_analysis(
        security_returns=security_returns,
        portfolio_returns=portfolio_returns,
        factors=factor_frame,
        security_weights=holdings_snapshot.set_index("ticker")["portfolio_weight"],
        min_obs=ff3_min_obs,
    )
    holdings_ff3_loadings = _build_holdings_ff3_loadings(
        holdings_snapshot=holdings_snapshot,
        ff3_analysis=ff3_analysis,
    )
    market_cap_mix = _build_market_cap_mix(holdings_snapshot)
    performance_summary = _build_performance_summary(
        comparison=comparison,
        trading_days_per_year=trading_days_per_year,
    )

    top_holding = holdings_snapshot.iloc[0]
    portfolio_methods = sorted(
        {str(value) for value in portfolio_returns["method"].dropna().astype(str).tolist()}
    )
    limitations = _build_limitations(portfolio_methods=portfolio_methods)

    summary: dict[str, object] = {
        "metadata": {
            "benchmark_ticker": benchmark_ticker_clean,
            "analysis_start_date": _date_string(comparison["trade_date"].min()),
            "analysis_end_date": _date_string(comparison["trade_date"].max()),
            "benchmark_observations": int(len(comparison)),
            "beta_frequency": beta_summary["frequency"],
            "top_n_holdings": int(top_n_holdings),
            "portfolio_method": portfolio_methods[0] if portfolio_methods else None,
            "limitations": limitations,
        },
        "performance": performance_summary,
        "beta": beta_summary,
        "sharpe": sharpe_summary,
        "holdings": {
            "as_of_date": _date_string(holdings_snapshot["as_of_date"].max()),
            "latest_trade_date": _date_string(holdings_snapshot["latest_trade_date"].max()),
            "holding_count": int(len(holdings_snapshot)),
            "top_holding_ticker": str(top_holding["ticker"]),
            "top_holding_weight": float(top_holding["portfolio_weight"]),
            "total_position_value": float(holdings_snapshot["position_value"].sum()),
        },
        "ff3": _build_ff3_summary(
            ff3_analysis=ff3_analysis,
            holdings_ff3_loadings=holdings_ff3_loadings,
            ff3_min_obs=ff3_min_obs,
        ),
    }

    return PortfolioAnalyticsResult(
        benchmark_comparison=comparison,
        beta_regression=beta_regression,
        holdings_snapshot=holdings_snapshot,
        holdings_ff3_loadings=holdings_ff3_loadings,
        market_cap_mix=market_cap_mix,
        ff3_analysis=ff3_analysis,
        summary=summary,
    )


def _read_tabular(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)

    raise ValueError(f"Unsupported file type for {path}. Expected .parquet or .csv.")


def _build_benchmark_comparison(
    *,
    portfolio_returns: pd.DataFrame,
    security_returns: pd.DataFrame,
    benchmark_ticker: str,
    output_value_base: float,
) -> pd.DataFrame:
    benchmark = _load_benchmark_returns(
        security_returns=security_returns,
        benchmark_ticker=benchmark_ticker,
    )
    out = portfolio_returns.merge(benchmark, on="trade_date", how="inner")
    out = out.dropna(subset=["portfolio_ret", "benchmark_ret"]).copy()
    if out.empty:
        raise ValueError(
            "Portfolio and benchmark returns have no overlapping non-null trade dates."
        )

    out = out.sort_values("trade_date").reset_index(drop=True)
    portfolio_growth = _growth_curve(out["portfolio_ret"])
    benchmark_growth = _growth_curve(out["benchmark_ret"])
    out["active_ret"] = out["portfolio_ret"] - out["benchmark_ret"]
    out["portfolio_value"] = output_value_base * portfolio_growth
    out["benchmark_value"] = output_value_base * benchmark_growth
    out["portfolio_drawdown"] = _drawdown_curve(portfolio_growth)
    out["benchmark_drawdown"] = _drawdown_curve(benchmark_growth)

    return out[
        [
            "trade_date",
            "portfolio_ret",
            "benchmark_ret",
            "active_ret",
            "portfolio_value",
            "benchmark_value",
            "portfolio_drawdown",
            "benchmark_drawdown",
            "n_constituents",
            "gross_exposure",
            "method",
        ]
    ]


def _load_benchmark_returns(
    *,
    security_returns: pd.DataFrame,
    benchmark_ticker: str,
) -> pd.DataFrame:
    benchmark = security_returns[
        security_returns["ticker"].astype("string").str.upper() == benchmark_ticker
    ][["trade_date", "ticker", "ret"]].copy()
    if benchmark.empty:
        raise ValueError(
            f"Benchmark ticker '{benchmark_ticker}' is not present in "
            "security_returns_daily. Add it to the Google Sheets Universe tab and "
            "rerun the daily pipeline before running analytics."
        )

    duplicated = benchmark.duplicated(["trade_date"], keep=False)
    if bool(duplicated.any()):
        sample = benchmark.loc[duplicated, ["trade_date", "ticker"]].head(10).to_dict("records")
        raise ValueError(
            "Benchmark returns contain multiple rows for the same trade_date. "
            f"Sample: {sample}"
        )

    benchmark = benchmark.rename(columns={"ret": "benchmark_ret"})
    return benchmark[["trade_date", "benchmark_ret"]].sort_values("trade_date").reset_index(
        drop=True
    )


def _build_beta_regression(
    *,
    comparison: pd.DataFrame,
    frequency: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    frequency_key = str(frequency).strip().lower()
    returns = comparison[["trade_date", "portfolio_ret", "benchmark_ret"]].copy()
    returns = returns.set_index("trade_date").sort_index()
    sampled = _resample_returns(returns, frequency=frequency_key).dropna()

    if len(sampled) < 3:
        raise ValueError(
            "Need at least 3 overlapping observations to estimate portfolio beta."
        )

    X = sm.add_constant(sampled["benchmark_ret"], has_constant="add")
    y = sampled["portfolio_ret"]
    model = sm.OLS(y, X).fit()

    sampled = sampled.copy()
    sampled["fitted_portfolio_ret"] = model.predict(X)
    sampled = sampled.reset_index()

    summary = {
        "beta": float(model.params["benchmark_ret"]),
        "alpha_per_period": float(model.params["const"]),
        "r_squared": float(model.rsquared),
        "frequency": frequency_key,
        "observations": int(model.nobs),
    }
    return sampled, summary


def _resample_returns(frame: pd.DataFrame, *, frequency: str) -> pd.DataFrame:
    if frequency == "daily":
        return frame.copy()
    if frequency == "weekly":
        return frame.resample("W-FRI").agg(_compound_returns)
    if frequency == "monthly":
        return frame.resample("M").agg(_compound_returns)

    raise ValueError("beta_frequency must be one of: daily, weekly, monthly.")


def _compound_returns(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return float("nan")
    return float((1.0 + clean).prod() - 1.0)


def _build_sharpe_summary(
    *,
    portfolio_returns: pd.DataFrame,
    factors: pd.DataFrame,
    trading_days_per_year: int,
) -> dict[str, object]:
    factor_frame = normalize_ff3_factors(factors)
    merged = (
        portfolio_returns.set_index("trade_date")[["portfolio_ret"]]
        .join(factor_frame[["rf"]], how="inner")
        .dropna()
    )
    if len(merged) < 2:
        raise ValueError(
            "Need at least 2 overlapping portfolio/risk-free observations to compute Sharpe."
        )

    excess = merged["portfolio_ret"] - merged["rf"]
    excess_std = float(excess.std(ddof=1))
    sharpe = (
        float(excess.mean()) / excess_std * float(np.sqrt(trading_days_per_year))
        if excess_std > 0
        else float("nan")
    )

    return {
        "annualized_sharpe": sharpe,
        "mean_daily_excess_return": float(excess.mean()),
        "daily_excess_volatility": excess_std,
        "observations": int(len(excess)),
    }


def _build_current_holdings_snapshot(
    *,
    universe: pd.DataFrame,
    security_returns: pd.DataFrame,
) -> pd.DataFrame:
    holdings = universe[universe["is_holding"]].copy()
    if holdings.empty:
        raise ValueError("Universe contains no holdings rows for analytics.")

    latest_market = (
        security_returns.sort_values(["ticker", "trade_date"])
        .groupby("ticker", as_index=False)
        .tail(1)[["ticker", "trade_date", "prc", "shrout"]]
        .rename(columns={"trade_date": "latest_trade_date"})
    )

    out = holdings.merge(latest_market, on="ticker", how="left")
    missing_prices = sorted(
        out.loc[out["prc"].isna(), "ticker"].astype(str).unique().tolist()
    )
    if missing_prices:
        raise ValueError(
            "Latest persisted prices are missing for holding tickers: "
            f"{missing_prices}. Rerun the daily pipeline and inspect security_returns_daily."
        )

    missing_shrout = sorted(
        out.loc[out["shrout"].isna(), "ticker"].astype(str).unique().tolist()
    )
    if missing_shrout:
        raise ValueError(
            "Latest persisted share-outstanding values are missing for holding tickers: "
            f"{missing_shrout}. Market-cap mix cannot be computed safely."
        )

    out["latest_price"] = pd.to_numeric(out["prc"], errors="coerce").abs()
    out["latest_market_cap_usd"] = out["latest_price"] * pd.to_numeric(
        out["shrout"], errors="coerce"
    ) * 1000.0
    out["position_value"] = pd.to_numeric(out["shares"], errors="coerce") * out["latest_price"]

    if (out["position_value"] <= 0).any():
        sample = (
            out.loc[out["position_value"] <= 0, ["ticker", "shares", "latest_price"]]
            .head(10)
            .to_dict("records")
        )
        raise ValueError(
            "Holdings snapshot produced non-positive position values. "
            f"Sample: {sample}"
        )

    total_value = float(out["position_value"].sum())
    if total_value <= 0:
        raise ValueError("Holdings position value total is non-positive.")

    out["portfolio_weight"] = out["position_value"] / total_value
    out["portfolio_weight_pct"] = out["portfolio_weight"] * 100.0
    out["market_cap_bucket"] = out["latest_market_cap_usd"].map(_market_cap_bucket)
    out = out.sort_values(["portfolio_weight", "ticker"], ascending=[False, True]).reset_index(
        drop=True
    )
    out["weight_rank"] = np.arange(1, len(out) + 1, dtype="int64")

    return out[
        [
            "as_of_date",
            "latest_trade_date",
            "ticker",
            "name",
            "sector",
            "shares",
            "latest_price",
            "latest_market_cap_usd",
            "market_cap_bucket",
            "position_value",
            "portfolio_weight",
            "portfolio_weight_pct",
            "weight_rank",
        ]
    ]


def _market_cap_bucket(market_cap_usd: float) -> str:
    if pd.isna(market_cap_usd):
        raise ValueError("Cannot bucket null market cap values.")
    if market_cap_usd >= 200_000_000_000:
        return "Mega Cap"
    if market_cap_usd >= 10_000_000_000:
        return "Large Cap"
    if market_cap_usd >= 2_000_000_000:
        return "Mid Cap"
    if market_cap_usd >= 300_000_000:
        return "Small Cap"
    return "Micro Cap"


def _build_market_cap_mix(holdings_snapshot: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        holdings_snapshot.groupby("market_cap_bucket", as_index=False)
        .agg(
            constituent_count=("ticker", "size"),
            portfolio_weight=("portfolio_weight", "sum"),
            position_value=("position_value", "sum"),
        )
        .copy()
    )
    grouped["portfolio_weight_pct"] = grouped["portfolio_weight"] * 100.0
    grouped["bucket_order"] = grouped["market_cap_bucket"].map(
        {bucket: idx for idx, bucket in enumerate(CAP_BUCKET_ORDER, start=1)}
    )

    missing_buckets = [
        bucket for bucket in CAP_BUCKET_ORDER if bucket not in set(grouped["market_cap_bucket"])
    ]
    if missing_buckets:
        grouped = pd.concat(
            [
                grouped,
                pd.DataFrame(
                    {
                        "market_cap_bucket": missing_buckets,
                        "constituent_count": 0,
                        "portfolio_weight": 0.0,
                        "position_value": 0.0,
                        "portfolio_weight_pct": 0.0,
                        "bucket_order": [
                            CAP_BUCKET_ORDER.index(bucket) + 1 for bucket in missing_buckets
                        ],
                    }
                ),
            ],
            ignore_index=True,
        )

    grouped = grouped.sort_values("bucket_order").reset_index(drop=True)
    return grouped[
        [
            "market_cap_bucket",
            "bucket_order",
            "constituent_count",
            "portfolio_weight",
            "portfolio_weight_pct",
            "position_value",
        ]
    ]


def _build_holdings_ff3_loadings(
    *,
    holdings_snapshot: pd.DataFrame,
    ff3_analysis: FF3AnalysisResult,
) -> pd.DataFrame:
    security_loadings = ff3_analysis.security_loadings.reset_index().copy()
    out = holdings_snapshot.merge(security_loadings, on="ticker", how="left")
    out["ff3_modeled"] = out["mkt_rf"].notna()
    out["ff3_excluded_reason"] = pd.Series(pd.NA, index=out.index, dtype="string")
    out.loc[~out["ff3_modeled"], "ff3_excluded_reason"] = (
        "insufficient_overlapping_history_for_ff3"
    )

    return out[
        [
            "weight_rank",
            "ticker",
            "name",
            "portfolio_weight",
            "portfolio_weight_pct",
            "ff3_modeled",
            "ff3_excluded_reason",
            "alpha",
            "mkt_rf",
            "smb",
            "hml",
            "r2",
            "n_obs",
            "explained_var_ratio",
        ]
    ].sort_values(["weight_rank", "ticker"]).reset_index(drop=True)


def _build_performance_summary(
    *,
    comparison: pd.DataFrame,
    trading_days_per_year: int,
) -> dict[str, object]:
    portfolio_metrics = _return_series_summary(
        returns=comparison["portfolio_ret"],
        trading_days_per_year=trading_days_per_year,
    )
    benchmark_metrics = _return_series_summary(
        returns=comparison["benchmark_ret"],
        trading_days_per_year=trading_days_per_year,
    )
    active_returns = comparison["active_ret"]
    tracking_error = float(active_returns.std(ddof=1)) * float(np.sqrt(trading_days_per_year))
    active_return = (
        float((1.0 + active_returns).prod()) ** (trading_days_per_year / len(active_returns)) - 1.0
        if len(active_returns) > 0
        else float("nan")
    )

    return {
        "portfolio_total_return": portfolio_metrics["total_return"],
        "benchmark_total_return": benchmark_metrics["total_return"],
        "portfolio_annualized_return": portfolio_metrics["annualized_return"],
        "benchmark_annualized_return": benchmark_metrics["annualized_return"],
        "portfolio_annualized_volatility": portfolio_metrics["annualized_volatility"],
        "benchmark_annualized_volatility": benchmark_metrics["annualized_volatility"],
        "portfolio_max_drawdown": portfolio_metrics["max_drawdown"],
        "benchmark_max_drawdown": benchmark_metrics["max_drawdown"],
        "tracking_error": tracking_error,
        "active_annualized_return": active_return,
        "information_ratio": (
            active_return / tracking_error if tracking_error > 0 else float("nan")
        ),
    }


def _build_ff3_summary(
    *,
    ff3_analysis: FF3AnalysisResult,
    holdings_ff3_loadings: pd.DataFrame,
    ff3_min_obs: int,
) -> dict[str, object]:
    payload = analysis_summary_payload(ff3_analysis)
    current_holdings_skipped = (
        holdings_ff3_loadings.loc[~holdings_ff3_loadings["ff3_modeled"], "ticker"]
        .astype(str)
        .tolist()
    )
    payload["current_holdings_modeled_count"] = int(holdings_ff3_loadings["ff3_modeled"].sum())
    payload["current_holdings_skipped_count"] = int((~holdings_ff3_loadings["ff3_modeled"]).sum())
    payload["current_holdings_skipped"] = current_holdings_skipped
    payload["ff3_min_obs"] = int(ff3_min_obs)
    return payload


def _return_series_summary(
    *,
    returns: pd.Series,
    trading_days_per_year: int,
) -> dict[str, float]:
    clean = pd.to_numeric(returns, errors="coerce").dropna()
    if clean.empty:
        raise ValueError("Cannot summarize an empty return series.")

    growth = _growth_curve(clean)
    total_return = float(growth.iloc[-1] - 1.0)
    annualized_return = float(growth.iloc[-1] ** (trading_days_per_year / len(clean)) - 1.0)
    annualized_volatility = float(clean.std(ddof=1)) * float(np.sqrt(trading_days_per_year))
    max_drawdown = float(_drawdown_curve(growth).min())

    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "max_drawdown": max_drawdown,
    }


def _growth_curve(returns: pd.Series) -> pd.Series:
    clean = pd.to_numeric(returns, errors="coerce").fillna(0.0).astype("float64")
    return (1.0 + clean).cumprod()


def _drawdown_curve(growth: pd.Series) -> pd.Series:
    normalized = pd.to_numeric(growth, errors="coerce").fillna(1.0).astype("float64")
    baseline = pd.Series([1.0], dtype="float64")
    running_peak = pd.concat([baseline, normalized.reset_index(drop=True)], ignore_index=True).cummax()
    running_peak = running_peak.iloc[1:].reset_index(drop=True)
    drawdown = normalized.reset_index(drop=True) / running_peak - 1.0
    drawdown.index = growth.index
    return drawdown


def _build_limitations(*, portfolio_methods: list[str]) -> list[str]:
    limitations: list[str] = []
    if "holdings_weighted_sum" in portfolio_methods:
        limitations.append(
            "Portfolio performance, beta, and Sharpe use the pipeline's holdings-weighted "
            "proxy series built from the current holdings snapshot."
        )
    limitations.append(
        "Historical portfolio composition changes are not yet modeled in this analytics layer."
    )
    return limitations


def _date_string(value: pd.Timestamp | object) -> str | None:
    if value is None or pd.isna(value):
        return None
    ts = pd.Timestamp(value)
    return ts.strftime("%Y-%m-%d")
