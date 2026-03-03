from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd
import statsmodels.api as sm

from mic_data.models.constants import FACTOR_COLUMNS


@dataclass(frozen=True)
class FF3RegressionResult:
    """Structured FF3 regression output for consistent downstream comparisons."""

    alpha: float
    beta_mkt: float
    beta_smb: float
    beta_hml: float
    r2: float
    n_obs: int
    t_stats: dict[str, float]

    def to_dict(self) -> dict[str, float | int | dict[str, float]]:
        return asdict(self)


def run_ff3_regression(
    portfolio_returns: pd.Series,
    factors: pd.DataFrame,
) -> FF3RegressionResult:
    """Run FF3 OLS regression with canonical factor schema.

    Inputs:
      - portfolio_returns: pd.Series of decimal returns named portfolio_return.
      - factors: pd.DataFrame with decimal-return columns mkt_rf, smb, hml, rf.

    Returns:
      - FF3RegressionResult containing coefficients, R^2, observation count, and t-stats.

    Raises:
      - ValueError if required columns are missing or merged frame is empty.

    Notes on units:
      - All returns must be decimal values (0.01 = 1%).
      - Alpha is a per-period intercept in the same units as returns.
    """
    required = set(FACTOR_COLUMNS)
    missing = required - set(factors.columns)
    if missing:
        raise ValueError(f"Factors missing required columns: {sorted(missing)}")

    # Statsmodels requires NumPy-native numeric dtypes; pandas nullable dtypes can
    # otherwise arrive as object arrays and raise a conversion error.
    series = pd.to_numeric(portfolio_returns.copy(), errors="coerce").astype("float64")
    series.name = "portfolio_return"
    factors_numeric = (
        factors[list(FACTOR_COLUMNS)]
        .apply(lambda col: pd.to_numeric(col, errors="coerce"))
        .astype("float64")
    )

    merged = pd.concat([series, factors_numeric], axis=1, join="inner").dropna()
    if merged.empty:
        raise ValueError("No overlapping observations between portfolio returns and factors.")

    merged["excess_return"] = merged["portfolio_return"] - merged["rf"]

    X = sm.add_constant(merged[["mkt_rf", "smb", "hml"]], has_constant="add")
    y = merged["excess_return"]

    model = sm.OLS(y, X).fit()

    return FF3RegressionResult(
        alpha=float(model.params.get("const", float("nan"))),
        beta_mkt=float(model.params.get("mkt_rf", float("nan"))),
        beta_smb=float(model.params.get("smb", float("nan"))),
        beta_hml=float(model.params.get("hml", float("nan"))),
        r2=float(model.rsquared),
        n_obs=int(model.nobs),
        t_stats={
            "alpha": float(model.tvalues.get("const", float("nan"))),
            "beta_mkt": float(model.tvalues.get("mkt_rf", float("nan"))),
            "beta_smb": float(model.tvalues.get("smb", float("nan"))),
            "beta_hml": float(model.tvalues.get("hml", float("nan"))),
        },
    )
