from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence, cast

import numpy as np
import pandas as pd
import statsmodels.api as sm

from mic_data.config.secrets import wrds_username
from mic_data.contracts.daily_returns_contracts import (
    validate_portfolio_returns_daily,
    validate_security_returns_daily,
)
from mic_data.models.constants import FACTOR_COLUMNS


DEFAULT_SECURITY_RETURNS_PATH = Path("data/processed/returns/security_returns_daily.parquet")
DEFAULT_PORTFOLIO_RETURNS_PATH = Path("data/processed/returns/portfolio_returns_daily.parquet")
FACTOR_BETA_COLUMNS = ("mkt_rf", "smb", "hml")
_FACTOR_RENAME_MAP = {
    "Mkt-RF": "mkt_rf",
    "SMB": "smb",
    "HML": "hml",
    "RF": "rf",
    "mktrf": "mkt_rf",
    "date": "trade_date",
}


class WrdsConnection(Protocol):
    def raw_sql(self, query: str, date_cols: list[str] | None = None) -> pd.DataFrame:
        ...

    def close(self) -> None:
        ...


class WrdsConnectionFactory(Protocol):
    def __call__(self, *, wrds_username: str | None = None) -> WrdsConnection:
        ...


@dataclass(frozen=True)
class PipelineReturnInputs:
    """Canonical return inputs consumed by FF3 analytics."""

    security_returns: pd.DataFrame
    portfolio_returns: pd.DataFrame


@dataclass(frozen=True)
class FF3AnalysisResult:
    """Reusable FF3 analytics outputs for reports, plots, or sheet publishing."""

    security_loadings: pd.DataFrame
    security_beta_matrix: pd.DataFrame
    portfolio_return_exposure: pd.Series
    portfolio_holdings_exposure: pd.Series | None
    factor_covariance: pd.DataFrame
    factor_correlation: pd.DataFrame
    security_factor_covariance: pd.DataFrame
    security_factor_correlation: pd.DataFrame
    factor_risk_contributions: pd.Series
    portfolio_risk_summary: pd.Series
    portfolio_holdings_risk_summary: pd.Series | None
    tables: dict[str, pd.DataFrame]
    metadata: dict[str, object]


@dataclass(frozen=True)
class _FF3Fit:
    alpha: float
    betas: pd.Series
    r2: float
    n_obs: int
    residual_var: float
    explained_var: float
    start_date: pd.Timestamp
    end_date: pd.Timestamp

    def to_series(self) -> pd.Series:
        total_var = self.explained_var + self.residual_var
        explained_var_ratio = (
            self.explained_var / total_var if total_var > 0 else float("nan")
        )
        return pd.Series(
            {
                "alpha": self.alpha,
                "mkt_rf": float(self.betas["mkt_rf"]),
                "smb": float(self.betas["smb"]),
                "hml": float(self.betas["hml"]),
                "r2": self.r2,
                "n_obs": self.n_obs,
                "residual_var": self.residual_var,
                "explained_var": self.explained_var,
                "explained_var_ratio": explained_var_ratio,
            }
        )


def load_pipeline_return_inputs(
    *,
    security_returns_path: str | Path = DEFAULT_SECURITY_RETURNS_PATH,
    portfolio_returns_path: str | Path = DEFAULT_PORTFOLIO_RETURNS_PATH,
    start_date: str | None = None,
    end_date: str | None = None,
    tickers: Sequence[str] | None = None,
) -> PipelineReturnInputs:
    """Load canonical return outputs from the daily pipeline."""

    security_returns = validate_security_returns_daily(
        _read_tabular(Path(security_returns_path))
    )
    portfolio_returns = validate_portfolio_returns_daily(
        _read_tabular(Path(portfolio_returns_path))
    )

    security_returns = _filter_date_window(
        security_returns,
        date_col="trade_date",
        start_date=start_date,
        end_date=end_date,
    )
    portfolio_returns = _filter_date_window(
        portfolio_returns,
        date_col="trade_date",
        start_date=start_date,
        end_date=end_date,
    )

    if tickers:
        ticker_set = {str(ticker).strip().upper() for ticker in tickers}
        security_returns = security_returns[
            security_returns["ticker"].isin(ticker_set)
        ].copy()

    return PipelineReturnInputs(
        security_returns=security_returns.reset_index(drop=True),
        portfolio_returns=portfolio_returns.reset_index(drop=True),
    )


def load_ff3_factors_from_wrds(
    *,
    start_date: str,
    end_date: str,
    username: str | None = None,
    connection_factory: WrdsConnectionFactory | None = None,
) -> pd.DataFrame:
    """Fetch daily FF3 factors from WRDS and return canonical decimal columns."""

    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    query = f"""
        SELECT date, mktrf, smb, hml, rf
        FROM ff.factors_daily
        WHERE date BETWEEN '{start_ts.strftime("%Y-%m-%d")}'
          AND '{end_ts.strftime("%Y-%m-%d")}'
        ORDER BY date
    """

    conn: WrdsConnection | None = None
    try:
        conn = _build_wrds_connection(
            username=username,
            connection_factory=connection_factory,
        )
        raw = conn.raw_sql(query, date_cols=["date"])
    finally:
        if conn is not None and hasattr(conn, "close"):
            conn.close()

    if raw.empty:
        raise ValueError("WRDS returned no FF3 factor rows for the requested window.")

    return normalize_ff3_factors(raw)


def normalize_ff3_factors(factors: pd.DataFrame) -> pd.DataFrame:
    """Normalize factor columns to canonical daily FF3 schema."""

    out = factors.copy().rename(columns=_FACTOR_RENAME_MAP)

    if "trade_date" in out.columns:
        trade_date = _normalize_dates(out["trade_date"])
        out = out.drop(columns=["trade_date"])
        out.index = trade_date
    else:
        out.index = _normalize_dates(pd.Index(out.index))

    out.index.name = "trade_date"

    missing = [column for column in FACTOR_COLUMNS if column not in out.columns]
    if missing:
        raise ValueError(
            f"Factor data missing required columns: {missing}. "
            f"Available columns: {list(out.columns)}"
        )

    out = out[list(FACTOR_COLUMNS)].copy()
    for column in FACTOR_COLUMNS:
        out[column] = pd.to_numeric(out[column], errors="coerce").astype("float64")

    out = out.sort_index().dropna()
    if out.index.has_duplicates:
        raise ValueError("Factor data contains duplicate trade_date rows.")
    if out.empty:
        raise ValueError("Factor data is empty after normalization.")

    return out


def estimate_security_ff3_loadings(
    security_returns: pd.DataFrame,
    factors: pd.DataFrame,
    *,
    min_obs: int = 60,
) -> tuple[pd.DataFrame, list[str]]:
    """Estimate per-security FF3 loadings from canonical security return data."""

    validated = validate_security_returns_daily(security_returns)
    _require_unique_ticker_dates(validated)

    factor_frame = normalize_ff3_factors(factors)
    security_matrix = (
        validated.pivot(index="trade_date", columns="ticker", values="ret").sort_index()
    )

    modeled_rows: list[pd.Series] = []
    skipped: list[str] = []
    for ticker in security_matrix.columns:
        try:
            fit = _fit_ff3_model(
                security_matrix[ticker].rename(str(ticker)),
                factor_frame,
                min_obs=min_obs,
            )
        except ValueError:
            skipped.append(str(ticker))
            continue

        row = fit.to_series()
        row.name = str(ticker)
        modeled_rows.append(row)

    if not modeled_rows:
        raise ValueError(
            "No securities had enough overlapping observations to estimate FF3 loadings."
        )

    security_loadings = pd.DataFrame(modeled_rows)
    security_loadings.index.name = "ticker"
    security_loadings = security_loadings.sort_index()

    return security_loadings, skipped


def estimate_portfolio_ff3_loading(
    portfolio_returns: pd.DataFrame,
    factors: pd.DataFrame,
    *,
    min_obs: int = 60,
) -> pd.Series:
    """Estimate FF3 exposure from canonical portfolio return data."""

    validated = validate_portfolio_returns_daily(portfolio_returns)
    portfolio_series = (
        validated.set_index("trade_date")["portfolio_ret"].rename("portfolio_return")
    )
    fit = _fit_ff3_model(portfolio_series, normalize_ff3_factors(factors), min_obs=min_obs)
    return fit.to_series()


def run_ff3_factor_analysis(
    *,
    security_returns: pd.DataFrame,
    portfolio_returns: pd.DataFrame,
    factors: pd.DataFrame,
    security_weights: pd.Series | None = None,
    min_obs: int = 60,
) -> FF3AnalysisResult:
    """Run downstream FF3 analytics on canonical pipeline outputs."""

    factor_frame = normalize_ff3_factors(factors)
    security_loadings, skipped_tickers = estimate_security_ff3_loadings(
        security_returns,
        factor_frame,
        min_obs=min_obs,
    )
    portfolio_return_exposure = estimate_portfolio_ff3_loading(
        portfolio_returns,
        factor_frame,
        min_obs=min_obs,
    )

    security_beta_matrix = security_loadings[list(FACTOR_BETA_COLUMNS)].copy()
    factor_covariance = factor_frame[list(FACTOR_BETA_COLUMNS)].cov()
    factor_correlation = factor_frame[list(FACTOR_BETA_COLUMNS)].corr()
    security_factor_covariance = _build_security_factor_covariance(
        security_loadings,
        factor_covariance,
    )
    security_factor_correlation = _covariance_to_correlation(
        security_factor_covariance
    )

    portfolio_risk_summary, factor_risk_contributions = _summarize_portfolio_risk(
        exposure=portfolio_return_exposure,
        factor_covariance=factor_covariance,
        residual_var=float(portfolio_return_exposure["residual_var"]),
    )

    portfolio_holdings_exposure: pd.Series | None = None
    portfolio_holdings_risk_summary: pd.Series | None = None
    exposure_rows = [
        portfolio_return_exposure[["alpha", *FACTOR_BETA_COLUMNS]].rename(
            "return_based"
        )
    ]

    weight_metadata: dict[str, object] = {"weights_provided": security_weights is not None}
    if security_weights is not None:
        aligned_weights, weight_metadata = _prepare_security_weights(
            security_weights,
            security_beta_matrix.index,
        )
        portfolio_holdings_exposure = pd.Series(
            {
                "alpha": float(aligned_weights @ security_loadings["alpha"]),
                "mkt_rf": float(aligned_weights @ security_loadings["mkt_rf"]),
                "smb": float(aligned_weights @ security_loadings["smb"]),
                "hml": float(aligned_weights @ security_loadings["hml"]),
            },
            name="holdings_based",
        )
        portfolio_holdings_risk_summary = _summarize_holdings_based_risk(
            weights=aligned_weights,
            security_loadings=security_loadings,
            factor_covariance=factor_covariance,
            security_factor_covariance=security_factor_covariance,
        )
        exposure_rows.append(portfolio_holdings_exposure)

    exposure_comparison = pd.DataFrame(exposure_rows)
    tables = {
        "security_loadings": _to_table(security_loadings, index_name="ticker"),
        "security_beta_matrix": _to_table(security_beta_matrix, index_name="ticker"),
        "portfolio_exposure_comparison": _to_table(
            exposure_comparison,
            index_name="exposure_method",
        ),
        "portfolio_risk_summary": _series_to_table(
            portfolio_risk_summary,
            index_name="metric",
            value_name="value",
        ),
        "factor_risk_contributions": _series_to_table(
            factor_risk_contributions,
            index_name="factor",
            value_name="variance_contribution",
        ),
        "factor_covariance": _to_table(factor_covariance, index_name="factor"),
        "factor_correlation": _to_table(factor_correlation, index_name="factor"),
        "security_factor_covariance": _to_table(
            security_factor_covariance,
            index_name="ticker",
        ),
        "security_factor_correlation": _to_table(
            security_factor_correlation,
            index_name="ticker",
        ),
    }
    if portfolio_holdings_risk_summary is not None:
        tables["portfolio_holdings_risk_summary"] = _series_to_table(
            portfolio_holdings_risk_summary,
            index_name="metric",
            value_name="value",
        )

    validated_portfolio = validate_portfolio_returns_daily(portfolio_returns)
    portfolio_methods = sorted(
        {str(value) for value in validated_portfolio["method"].dropna().astype(str)}
    )
    limitations = _build_limitations(
        portfolio_methods=portfolio_methods,
        skipped_tickers=skipped_tickers,
        min_obs=min_obs,
        security_weights=security_weights,
    )

    metadata: dict[str, object] = {
        "analysis_start_date": str(factor_frame.index.min().date()),
        "analysis_end_date": str(factor_frame.index.max().date()),
        "factor_observations": int(len(factor_frame)),
        "modeled_security_count": int(len(security_loadings)),
        "skipped_tickers": skipped_tickers,
        "portfolio_methods": portfolio_methods,
        "min_obs": min_obs,
        "limitations": limitations,
    }
    metadata.update(weight_metadata)

    return FF3AnalysisResult(
        security_loadings=security_loadings,
        security_beta_matrix=security_beta_matrix,
        portfolio_return_exposure=portfolio_return_exposure,
        portfolio_holdings_exposure=portfolio_holdings_exposure,
        factor_covariance=factor_covariance,
        factor_correlation=factor_correlation,
        security_factor_covariance=security_factor_covariance,
        security_factor_correlation=security_factor_correlation,
        factor_risk_contributions=factor_risk_contributions,
        portfolio_risk_summary=portfolio_risk_summary,
        portfolio_holdings_risk_summary=portfolio_holdings_risk_summary,
        tables=tables,
        metadata=metadata,
    )


def _read_tabular(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)

    raise ValueError(f"Unsupported file type for {path}. Expected .parquet or .csv.")


def _filter_date_window(
    frame: pd.DataFrame,
    *,
    date_col: str,
    start_date: str | None,
    end_date: str | None,
) -> pd.DataFrame:
    out = frame.copy()
    if start_date is not None:
        out = out[out[date_col] >= pd.Timestamp(start_date)].copy()
    if end_date is not None:
        out = out[out[date_col] <= pd.Timestamp(end_date)].copy()
    return out


def _build_wrds_connection(
    *,
    username: str | None,
    connection_factory: WrdsConnectionFactory | None,
) -> WrdsConnection:
    resolved_username = wrds_username(username)
    if connection_factory is not None:
        return connection_factory(wrds_username=resolved_username)

    try:
        import wrds
    except Exception as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "wrds package is not available. Install it and configure WRDS access."
        ) from exc

    try:
        return cast(WrdsConnection, wrds.Connection(wrds_username=resolved_username))
    except Exception as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "Unable to establish a WRDS connection. Configure WRDS_USERNAME and "
            "non-interactive authentication first."
        ) from exc


def _normalize_dates(values: pd.Series | pd.Index) -> pd.DatetimeIndex:
    out = pd.to_datetime(values, errors="raise", utc=True)
    return pd.DatetimeIndex(out).tz_convert(None).normalize()


def _require_unique_ticker_dates(security_returns: pd.DataFrame) -> None:
    duplicated = security_returns.duplicated(["trade_date", "ticker"], keep=False)
    if bool(duplicated.any()):
        sample = security_returns.loc[
            duplicated,
            ["trade_date", "ticker", "permno"],
        ].head(10)
        raise ValueError(
            "security_returns contains multiple rows for the same trade_date/ticker. "
            f"Sample: {sample.to_dict('records')}"
        )


def _fit_ff3_model(
    returns: pd.Series,
    factors: pd.DataFrame,
    *,
    min_obs: int,
) -> _FF3Fit:
    returns_numeric = pd.to_numeric(returns.copy(), errors="coerce").astype("float64")
    returns_numeric.index = _normalize_dates(pd.Index(returns_numeric.index))
    returns_numeric.name = returns.name or "return"

    merged = pd.concat(
        [returns_numeric.rename("asset_return"), factors[list(FACTOR_COLUMNS)]],
        axis=1,
        join="inner",
    ).dropna()
    if len(merged) < min_obs:
        raise ValueError(
            f"Need at least {min_obs} overlapping observations, found {len(merged)}."
        )

    merged["excess_return"] = merged["asset_return"] - merged["rf"]
    X = sm.add_constant(merged[list(FACTOR_BETA_COLUMNS)], has_constant="add")
    y = merged["excess_return"]
    model = sm.OLS(y, X).fit()

    betas = pd.Series(
        {
            "mkt_rf": float(model.params["mkt_rf"]),
            "smb": float(model.params["smb"]),
            "hml": float(model.params["hml"]),
        }
    )
    explained_var = float(model.fittedvalues.var(ddof=1))

    return _FF3Fit(
        alpha=float(model.params["const"]),
        betas=betas,
        r2=float(model.rsquared),
        n_obs=int(model.nobs),
        residual_var=float(model.resid.var(ddof=1)),
        explained_var=explained_var,
        start_date=pd.Timestamp(merged.index.min()),
        end_date=pd.Timestamp(merged.index.max()),
    )


def _build_security_factor_covariance(
    security_loadings: pd.DataFrame,
    factor_covariance: pd.DataFrame,
) -> pd.DataFrame:
    beta_matrix = security_loadings[list(FACTOR_BETA_COLUMNS)]
    residual_var = security_loadings["residual_var"]

    cov = (
        beta_matrix.to_numpy() @ factor_covariance.to_numpy() @ beta_matrix.to_numpy().T
        + np.diag(residual_var.to_numpy())
    )
    return pd.DataFrame(
        cov,
        index=beta_matrix.index,
        columns=beta_matrix.index,
    )


def _summarize_portfolio_risk(
    *,
    exposure: pd.Series,
    factor_covariance: pd.DataFrame,
    residual_var: float,
) -> tuple[pd.Series, pd.Series]:
    beta_vector = exposure[list(FACTOR_BETA_COLUMNS)].to_numpy(dtype="float64")
    sigma_f = factor_covariance.to_numpy(dtype="float64")

    factor_variance = float(beta_vector @ sigma_f @ beta_vector.T)
    idiosyncratic_variance = float(residual_var)
    total_variance = factor_variance + idiosyncratic_variance
    factor_share = factor_variance / total_variance if total_variance > 0 else float("nan")
    idiosyncratic_share = (
        idiosyncratic_variance / total_variance if total_variance > 0 else float("nan")
    )

    contributions = beta_vector * (sigma_f @ beta_vector)
    factor_risk_contributions = pd.Series(
        contributions,
        index=FACTOR_BETA_COLUMNS,
        name="variance_contribution",
    )

    summary = pd.Series(
        {
            "factor_variance": factor_variance,
            "idiosyncratic_variance": idiosyncratic_variance,
            "total_variance": total_variance,
            "volatility": float(np.sqrt(total_variance)) if total_variance >= 0 else float("nan"),
            "factor_share": factor_share,
            "idiosyncratic_share": idiosyncratic_share,
            "r2": float(exposure["r2"]),
        },
        name="portfolio_return_based",
    )
    return summary, factor_risk_contributions


def _summarize_holdings_based_risk(
    *,
    weights: pd.Series,
    security_loadings: pd.DataFrame,
    factor_covariance: pd.DataFrame,
    security_factor_covariance: pd.DataFrame,
) -> pd.Series:
    aligned_weights = weights.reindex(security_factor_covariance.index).fillna(0.0)
    exposure = pd.Series(
        {
            "mkt_rf": float(aligned_weights @ security_loadings["mkt_rf"]),
            "smb": float(aligned_weights @ security_loadings["smb"]),
            "hml": float(aligned_weights @ security_loadings["hml"]),
        }
    )
    factor_variance = float(
        exposure[list(FACTOR_BETA_COLUMNS)].to_numpy()
        @ factor_covariance.to_numpy()
        @ exposure[list(FACTOR_BETA_COLUMNS)].to_numpy().T
    )
    total_variance = float(
        aligned_weights.to_numpy()
        @ security_factor_covariance.to_numpy()
        @ aligned_weights.to_numpy().T
    )
    idiosyncratic_variance = total_variance - factor_variance

    return pd.Series(
        {
            "factor_variance": factor_variance,
            "idiosyncratic_variance": idiosyncratic_variance,
            "total_variance": total_variance,
            "volatility": float(np.sqrt(total_variance)) if total_variance >= 0 else float("nan"),
            "factor_share": factor_variance / total_variance if total_variance > 0 else float("nan"),
            "idiosyncratic_share": (
                idiosyncratic_variance / total_variance if total_variance > 0 else float("nan")
            ),
            "modeled_weight_count": float(len(aligned_weights)),
        },
        name="portfolio_holdings_based",
    )


def _prepare_security_weights(
    weights: pd.Series,
    modeled_tickers: pd.Index,
) -> tuple[pd.Series, dict[str, object]]:
    out = pd.to_numeric(weights.copy(), errors="raise").astype("float64")
    out.index = out.index.astype(str).str.strip().str.upper()
    out = out[~out.index.duplicated(keep="last")]

    aligned = out.reindex(modeled_tickers).dropna()
    if aligned.empty:
        raise ValueError("No provided weights overlap with modeled security loadings.")

    total_weight = float(aligned.sum())
    if total_weight == 0:
        raise ValueError("Provided weights sum to zero after aligning to modeled securities.")

    normalized = aligned / total_weight
    metadata = {
        "weights_provided": True,
        "input_weight_count": int(len(out)),
        "modeled_weight_count": int(len(aligned)),
        "input_weight_sum": float(out.sum()),
        "modeled_weight_sum": total_weight,
    }
    return normalized, metadata


def _covariance_to_correlation(covariance: pd.DataFrame) -> pd.DataFrame:
    values = covariance.to_numpy(dtype="float64")
    std = np.sqrt(np.diag(values))
    denom = np.outer(std, std)
    corr = np.divide(
        values,
        denom,
        out=np.full_like(values, np.nan),
        where=denom > 0,
    )
    return pd.DataFrame(corr, index=covariance.index, columns=covariance.columns)


def _to_table(frame: pd.DataFrame, *, index_name: str) -> pd.DataFrame:
    out = frame.copy()
    out.index.name = index_name
    return out.reset_index()


def _series_to_table(
    values: pd.Series,
    *,
    index_name: str,
    value_name: str,
) -> pd.DataFrame:
    out = values.rename(value_name).to_frame()
    out.index.name = index_name
    return out.reset_index()


def _build_limitations(
    *,
    portfolio_methods: list[str],
    skipped_tickers: list[str],
    min_obs: int,
    security_weights: pd.Series | None,
) -> list[str]:
    limitations: list[str] = []
    if "holdings_weighted_sum" in portfolio_methods:
        limitations.append(
            "portfolio_ret is the pipeline's holdings-weighted proxy series, not a full "
            "transaction-history portfolio return."
        )
    if skipped_tickers:
        limitations.append(
            f"{len(skipped_tickers)} tickers were excluded for having fewer than {min_obs} "
            "overlapping return/factor observations."
        )
    if security_weights is None:
        limitations.append(
            "Holdings-based portfolio exposure was not computed because no security weight "
            "snapshot was provided to the analytics module."
        )
    return limitations


__all__ = [
    "DEFAULT_PORTFOLIO_RETURNS_PATH",
    "DEFAULT_SECURITY_RETURNS_PATH",
    "FACTOR_BETA_COLUMNS",
    "FF3AnalysisResult",
    "PipelineReturnInputs",
    "estimate_portfolio_ff3_loading",
    "estimate_security_ff3_loadings",
    "load_ff3_factors_from_wrds",
    "load_pipeline_return_inputs",
    "normalize_ff3_factors",
    "run_ff3_factor_analysis",
]
