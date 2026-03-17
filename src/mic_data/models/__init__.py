from __future__ import annotations

from importlib import import_module

__all__ = [
    "FF3AnalysisResult",
    "PipelineReturnInputs",
    "analysis_summary_payload",
    "build_parser",
    "estimate_portfolio_ff3_loading",
    "estimate_security_ff3_loadings",
    "load_ff3_factors_from_wrds",
    "load_pipeline_return_inputs",
    "main",
    "normalize_ff3_factors",
    "run_ff3_factor_analysis",
]


def __getattr__(name: str) -> object:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module("mic_data.models.ff_factor_matrix")
    return getattr(module, name)
