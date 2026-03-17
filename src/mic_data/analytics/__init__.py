from __future__ import annotations

from importlib import import_module

_EXPORT_TO_MODULE = {
    "AnalyticsInputs": "mic_data.analytics.dashboard",
    "PortfolioAnalyticsResult": "mic_data.analytics.dashboard",
    "build_portfolio_analytics": "mic_data.analytics.dashboard",
    "load_analytics_inputs": "mic_data.analytics.dashboard",
    "AnalyticsReportConfig": "mic_data.analytics.report",
    "build_parser": "mic_data.analytics.report",
    "main": "mic_data.analytics.report",
    "run_analytics_report": "mic_data.analytics.report",
}

__all__ = list(_EXPORT_TO_MODULE)


def __getattr__(name: str) -> object:
    module_name = _EXPORT_TO_MODULE.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(module_name)
    return getattr(module, name)
