from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from mic_data.models.constants import FACTOR_COLUMNS
from mic_data.models.regression import FF3RegressionResult


@dataclass(frozen=True)
class ComparisonThresholds:
    """Thresholds for WRDS-vs-static FF3 similarity checks.

    Notes on units:
      - mad_max_bps and delta_alpha_max_bps are in basis points.
      - All coefficient thresholds are in decimal-return units.
    """

    corr_min: float = 0.98
    mad_max_bps: float = 5.0
    delta_beta_max: float = 0.05
    delta_alpha_max_bps: float = 10.0
    delta_r2_max: float = 0.03


def compare_factor_inputs(wrds_factors: pd.DataFrame, static_factors: pd.DataFrame) -> pd.DataFrame:
    """Compute side-by-side diagnostics for WRDS and static factor inputs.

    Inputs:
      - wrds_factors: Monthly factor dataframe with canonical columns.
      - static_factors: Monthly factor dataframe with canonical columns.

    Returns:
      - pd.DataFrame indexed by factor with columns corr, mad_bps, rmse_bps, n_obs.

    Raises:
      - ValueError when no overlapping observations are available.

    Notes on units:
      - Inputs must be decimal returns.
      - MAD/RMSE are reported in basis points for readability.
    """
    wrds = wrds_factors[FACTOR_COLUMNS].copy()
    static = static_factors[FACTOR_COLUMNS].copy()

    overlap = wrds.join(static, how="inner", lsuffix="_wrds", rsuffix="_static").dropna()
    if overlap.empty:
        raise ValueError("No overlapping rows between WRDS and static factor inputs.")

    rows: list[dict[str, Any]] = []
    for col in FACTOR_COLUMNS:
        wrds_col = overlap[f"{col}_wrds"]
        static_col = overlap[f"{col}_static"]
        diff = wrds_col - static_col

        rows.append(
            {
                "factor": col,
                "corr": float(wrds_col.corr(static_col)),
                "mad_bps": float(diff.abs().mean() * 10000.0),
                "rmse_bps": float((diff.pow(2).mean() ** 0.5) * 10000.0),
                "n_obs": int(len(diff)),
            }
        )

    return pd.DataFrame(rows).set_index("factor")


def compare_regression_results(
    wrds_result: FF3RegressionResult,
    static_result: FF3RegressionResult,
) -> dict[str, float | int]:
    """Compute deltas between WRDS-driven and static-driven FF3 regressions.

    Inputs:
      - wrds_result: FF3 regression result from WRDS factor inputs.
      - static_result: FF3 regression result from static factor inputs.

    Returns:
      - Dict with signed deltas and absolute delta helper fields.

    Raises:
      - None.

    Notes on units:
      - Alpha deltas are also provided in basis points for gating/reporting.
    """
    delta_alpha = wrds_result.alpha - static_result.alpha
    delta_beta_mkt = wrds_result.beta_mkt - static_result.beta_mkt
    delta_beta_smb = wrds_result.beta_smb - static_result.beta_smb
    delta_beta_hml = wrds_result.beta_hml - static_result.beta_hml
    delta_r2 = wrds_result.r2 - static_result.r2

    return {
        "delta_alpha": float(delta_alpha),
        "delta_alpha_bps": float(delta_alpha * 10000.0),
        "delta_beta_mkt": float(delta_beta_mkt),
        "delta_beta_smb": float(delta_beta_smb),
        "delta_beta_hml": float(delta_beta_hml),
        "delta_r2": float(delta_r2),
        "delta_n_obs": int(wrds_result.n_obs - static_result.n_obs),
        "abs_delta_alpha_bps": float(abs(delta_alpha) * 10000.0),
        "abs_delta_beta_mkt": float(abs(delta_beta_mkt)),
        "abs_delta_beta_smb": float(abs(delta_beta_smb)),
        "abs_delta_beta_hml": float(abs(delta_beta_hml)),
        "abs_delta_r2": float(abs(delta_r2)),
    }


def evaluate_similarity(
    factor_metrics: pd.DataFrame,
    regression_deltas: dict[str, float | int],
    thresholds: ComparisonThresholds,
) -> dict[str, Any]:
    """Evaluate balanced pass/fail criteria for WRDS-vs-static comparability.

    Inputs:
      - factor_metrics: output from compare_factor_inputs.
      - regression_deltas: output from compare_regression_results.
      - thresholds: gating threshold configuration.

    Returns:
      - Dict with pass flag, reason codes, and threshold snapshot.

    Raises:
      - ValueError if factor_metrics is empty.

    Notes on units:
      - Basis-point thresholds are applied to *_bps fields only.
    """
    if factor_metrics.empty:
        raise ValueError("Factor metrics are empty; cannot evaluate similarity gates.")

    reasons: list[str] = []

    corr_values = factor_metrics["corr"].fillna(float("-inf"))
    bad_corr = factor_metrics[corr_values < thresholds.corr_min]
    if not bad_corr.empty:
        reasons.extend([f"corr_below_threshold:{idx}" for idx in bad_corr.index])

    bad_mad = factor_metrics[factor_metrics["mad_bps"] > thresholds.mad_max_bps]
    if not bad_mad.empty:
        reasons.extend([f"mad_above_threshold:{idx}" for idx in bad_mad.index])

    if float(regression_deltas["abs_delta_beta_mkt"]) > thresholds.delta_beta_max:
        reasons.append("delta_beta_mkt_above_threshold")
    if float(regression_deltas["abs_delta_beta_smb"]) > thresholds.delta_beta_max:
        reasons.append("delta_beta_smb_above_threshold")
    if float(regression_deltas["abs_delta_beta_hml"]) > thresholds.delta_beta_max:
        reasons.append("delta_beta_hml_above_threshold")
    if float(regression_deltas["abs_delta_alpha_bps"]) > thresholds.delta_alpha_max_bps:
        reasons.append("delta_alpha_above_threshold")
    if float(regression_deltas["abs_delta_r2"]) > thresholds.delta_r2_max:
        reasons.append("delta_r2_above_threshold")

    return {
        "passed": len(reasons) == 0,
        "reason_codes": reasons,
        "thresholds": asdict(thresholds),
    }
