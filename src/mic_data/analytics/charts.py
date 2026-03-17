from __future__ import annotations

import html
import math
import tempfile
from pathlib import Path

import pandas as pd

from mic_data.analytics.dashboard import PortfolioAnalyticsResult


CHART_WIDTH = 1200
CHART_HEIGHT = 760
BACKGROUND = "#f7f2e8"
PANEL = "#fffdf7"
TEXT = "#201a17"
MUTED = "#6f665d"
GRID = "#d9cfbf"
PORTFOLIO = "#134e5e"
BENCHMARK = "#c06c45"
ACCENT = "#a8842c"
BAR = "#5f8b4c"
NOTE = (
    "Proxy series note: portfolio returns use the current holdings snapshot applied backward; "
    "historical composition changes are not yet modeled."
)


def render_analytics_charts(
    *,
    result: PortfolioAnalyticsResult,
    charts_dir: Path,
    benchmark_ticker: str,
    top_n_holdings: int,
) -> dict[str, Path]:
    """Write SVG chart artifacts for the portfolio analytics report."""

    charts_dir.mkdir(parents=True, exist_ok=True)
    benchmark_slug = str(benchmark_ticker).strip().lower()

    chart_paths = {
        "performance_chart": charts_dir / f"performance_vs_{benchmark_slug}.svg",
        "beta_chart": charts_dir / f"beta_vs_{benchmark_slug}.svg",
        "top_holdings_chart": charts_dir / "top_holdings.svg",
        "market_cap_mix_chart": charts_dir / "market_cap_mix.svg",
        "sharpe_card": charts_dir / "sharpe_ratio.svg",
    }

    _atomic_write_text(
        chart_paths["performance_chart"],
        _render_performance_chart(
            comparison=result.benchmark_comparison,
            benchmark_ticker=benchmark_ticker,
        ),
    )
    _atomic_write_text(
        chart_paths["beta_chart"],
        _render_beta_chart(
            beta_regression=result.beta_regression,
            benchmark_ticker=benchmark_ticker,
            frequency=str(result.summary["beta"]["frequency"]),
        ),
    )
    _atomic_write_text(
        chart_paths["top_holdings_chart"],
        _render_top_holdings_chart(
            holdings_snapshot=result.holdings_snapshot,
            top_n_holdings=top_n_holdings,
        ),
    )
    _atomic_write_text(
        chart_paths["market_cap_mix_chart"],
        _render_market_cap_mix_chart(result.market_cap_mix),
    )
    _atomic_write_text(
        chart_paths["sharpe_card"],
        _render_sharpe_card(
            sharpe_summary=result.summary["sharpe"],
            benchmark_ticker=benchmark_ticker,
        ),
    )
    return chart_paths


def _render_performance_chart(
    *,
    comparison: pd.DataFrame,
    benchmark_ticker: str,
) -> str:
    frame = comparison.copy().sort_values("trade_date")
    plot_left = 110.0
    plot_top = 110.0
    plot_width = 960.0
    plot_height = 470.0
    bottom = plot_top + plot_height

    x_points = _scale_dates(frame["trade_date"], plot_left, plot_left + plot_width)
    all_values = pd.concat([frame["portfolio_value"], frame["benchmark_value"]], ignore_index=True)
    y_min = min(100.0, float(all_values.min()))
    y_max = float(all_values.max())
    y_points_port = _scale_numeric(frame["portfolio_value"], y_min, y_max, bottom, plot_top)
    y_points_bench = _scale_numeric(frame["benchmark_value"], y_min, y_max, bottom, plot_top)

    portfolio_path = _line_path(list(zip(x_points, y_points_port)))
    benchmark_path = _line_path(list(zip(x_points, y_points_bench)))

    elements = [_svg_root()]
    elements.append(_panel_rect(40, 40, CHART_WIDTH - 80, CHART_HEIGHT - 80))
    elements.append(
        _text(72, 86, "Portfolio Proxy vs Benchmark Growth of $100", size=32, weight="700")
    )
    elements.append(
        _text(
            72,
            120,
            f"Benchmark: {benchmark_ticker} from persisted WRDS returns",
            size=17,
            fill=MUTED,
        )
    )
    elements.extend(_axes(x0=plot_left, y0=plot_top, width=plot_width, height=plot_height))
    elements.extend(_y_grid(y_min=y_min, y_max=y_max, plot_left=plot_left, plot_top=plot_top, plot_width=plot_width, plot_height=plot_height, currency=True))
    elements.extend(_date_ticks(frame["trade_date"], plot_left=plot_left, plot_width=plot_width, y=bottom + 34))
    elements.append(
        f'<path d="{portfolio_path}" fill="none" stroke="{PORTFOLIO}" stroke-width="4" stroke-linecap="round" />'
    )
    elements.append(
        f'<path d="{benchmark_path}" fill="none" stroke="{BENCHMARK}" stroke-width="4" stroke-linecap="round" />'
    )

    end_x = float(x_points.iloc[-1])
    end_y_port = float(y_points_port.iloc[-1])
    end_y_bench = float(y_points_bench.iloc[-1])
    for label, fill, end_x, end_y, value in (
        ("Portfolio", PORTFOLIO, end_x, end_y_port, frame["portfolio_value"].iloc[-1]),
        (benchmark_ticker, BENCHMARK, end_x, end_y_bench, frame["benchmark_value"].iloc[-1]),
    ):
        elements.append(
            f'<circle cx="{end_x:.2f}" cy="{end_y:.2f}" r="6" fill="{fill}" />'
        )
        elements.append(
            _text(
                min(end_x + 14, CHART_WIDTH - 150),
                end_y - 8,
                f"{label} ${float(value):.2f}",
                size=16,
                weight="700",
                fill=fill,
            )
        )

    legend_x = 820.0
    legend_y = 126.0
    elements.append(_legend_swatch(legend_x, legend_y, PORTFOLIO, "Portfolio"))
    elements.append(_legend_swatch(legend_x + 170, legend_y, BENCHMARK, benchmark_ticker))
    elements.append(_text(72, 670, NOTE, size=15, fill=MUTED))
    elements.append("</svg>")
    return "".join(elements)


def _render_beta_chart(
    *,
    beta_regression: pd.DataFrame,
    benchmark_ticker: str,
    frequency: str,
) -> str:
    frame = beta_regression.copy().sort_values("trade_date")
    x = frame["benchmark_ret"] * 100.0
    y = frame["portfolio_ret"] * 100.0
    fitted = frame["fitted_portfolio_ret"] * 100.0

    limit = max(5.0, float(max(x.abs().max(), y.abs().max()) * 1.15))
    plot_left = 170.0
    plot_top = 120.0
    plot_size = 460.0

    x_points = _scale_numeric(x, -limit, limit, plot_left, plot_left + plot_size)
    y_points = _scale_numeric(y, -limit, limit, plot_top + plot_size, plot_top)
    fitted_x = _scale_numeric(x, -limit, limit, plot_left, plot_left + plot_size)
    fitted_y = _scale_numeric(fitted, -limit, limit, plot_top + plot_size, plot_top)

    beta = _safe_float(frame["portfolio_ret"].cov(frame["benchmark_ret"]) / frame["benchmark_ret"].var())
    alpha = _safe_float(frame["portfolio_ret"].mean() - beta * frame["benchmark_ret"].mean())
    r_squared = _safe_float(
        1.0
        - ((frame["portfolio_ret"] - frame["fitted_portfolio_ret"]) ** 2).sum()
        / ((frame["portfolio_ret"] - frame["portfolio_ret"].mean()) ** 2).sum()
    )

    elements = [_svg_root()]
    elements.append(_panel_rect(40, 40, CHART_WIDTH - 80, CHART_HEIGHT - 80))
    elements.append(
        _text(72, 86, f"Portfolio Beta vs {benchmark_ticker}", size=32, weight="700")
    )
    elements.append(
        _text(
            72,
            120,
            f"Regression on {frequency} compounded returns against {benchmark_ticker}",
            size=17,
            fill=MUTED,
        )
    )
    elements.extend(_axes(x0=plot_left, y0=plot_top, width=plot_size, height=plot_size))

    zero_x = _scale_numeric(pd.Series([0.0]), -limit, limit, plot_left, plot_left + plot_size).iloc[0]
    zero_y = _scale_numeric(pd.Series([0.0]), -limit, limit, plot_top + plot_size, plot_top).iloc[0]
    elements.append(
        f'<line x1="{zero_x:.2f}" y1="{plot_top:.2f}" x2="{zero_x:.2f}" y2="{plot_top + plot_size:.2f}" stroke="{GRID}" stroke-width="1.5" />'
    )
    elements.append(
        f'<line x1="{plot_left:.2f}" y1="{zero_y:.2f}" x2="{plot_left + plot_size:.2f}" y2="{zero_y:.2f}" stroke="{GRID}" stroke-width="1.5" />'
    )

    for x_value, y_value in zip(x_points, y_points, strict=False):
        elements.append(
            f'<circle cx="{float(x_value):.2f}" cy="{float(y_value):.2f}" r="4.2" fill="{PORTFOLIO}" fill-opacity="0.48" />'
        )

    elements.append(
        f'<path d="{_line_path(sorted(zip(fitted_x, fitted_y), key=lambda point: point[0]))}" '
        f'fill="none" stroke="{ACCENT}" stroke-width="4" stroke-linecap="round" />'
    )

    elements.extend(
        _numeric_ticks(
            minimum=-limit,
            maximum=limit,
            plot_left=plot_left,
            plot_top=plot_top,
            plot_width=plot_size,
            plot_height=plot_size,
            suffix="%",
        )
    )
    elements.append(_text(plot_left + plot_size / 2 - 70, 635, f"{benchmark_ticker} return", size=17))
    elements.append(_text(85, plot_top + plot_size / 2, "Portfolio return", size=17, rotate=-90))
    elements.append(
        _info_box(
            760,
            170,
            300,
            170,
            [
                f"Beta: {_format_number(beta, 2)}",
                f"Alpha / period: {_format_pct(alpha)}",
                f"R-squared: {_format_number(r_squared, 2)}",
                f"Observations: {len(frame)}",
            ],
        )
    )
    elements.append(_text(72, 690, NOTE, size=15, fill=MUTED))
    elements.append("</svg>")
    return "".join(elements)


def _render_top_holdings_chart(
    *,
    holdings_snapshot: pd.DataFrame,
    top_n_holdings: int,
) -> str:
    top = holdings_snapshot.nsmallest(top_n_holdings, "weight_rank").sort_values("portfolio_weight")
    plot_left = 270.0
    plot_top = 150.0
    plot_width = 760.0
    row_height = 82.0

    max_weight = max(float(top["portfolio_weight_pct"].max()), 1.0)
    elements = [_svg_root()]
    elements.append(_panel_rect(40, 40, CHART_WIDTH - 80, CHART_HEIGHT - 80))
    elements.append(_text(72, 86, f"Top {top_n_holdings} Holdings", size=32, weight="700"))
    elements.append(
        _text(
            72,
            120,
            "Current holdings snapshot weighted by latest persisted WRDS prices",
            size=17,
            fill=MUTED,
        )
    )

    for idx, (_, row) in enumerate(top.iterrows(), start=0):
        y = plot_top + idx * row_height
        bar_width = (float(row["portfolio_weight_pct"]) / max_weight) * plot_width
        bar_color = PORTFOLIO if idx % 2 == 0 else BAR
        elements.append(_text(90, y + 34, str(row["ticker"]), size=24, weight="700"))
        name = str(row["name"]) if pd.notna(row["name"]) else ""
        elements.append(_text(90, y + 60, name, size=15, fill=MUTED))
        elements.append(
            f'<rect x="{plot_left:.2f}" y="{y:.2f}" width="{bar_width:.2f}" height="46" '
            f'rx="12" fill="{bar_color}" />'
        )
        elements.append(
            _text(
                plot_left + min(bar_width + 18, plot_width + 30),
                y + 31,
                _format_pct(float(row["portfolio_weight"])),
                size=19,
                weight="700",
                fill=TEXT,
            )
        )

    elements.append(
        _text(
            72,
            692,
            "Weights reflect latest holdings shares times latest persisted prices, not historical trade-by-trade allocations.",
            size=15,
            fill=MUTED,
        )
    )
    elements.append("</svg>")
    return "".join(elements)


def _render_market_cap_mix_chart(market_cap_mix: pd.DataFrame) -> str:
    frame = market_cap_mix.copy().sort_values("bucket_order")
    plot_left = 110.0
    plot_top = 140.0
    plot_width = 950.0
    plot_height = 420.0
    bar_width = 120.0
    gap = 55.0
    bottom = plot_top + plot_height
    max_pct = max(5.0, float(frame["portfolio_weight_pct"].max()) * 1.15)

    elements = [_svg_root()]
    elements.append(_panel_rect(40, 40, CHART_WIDTH - 80, CHART_HEIGHT - 80))
    elements.append(_text(72, 86, "Market-Cap Mix", size=32, weight="700"))
    elements.append(
        _text(
            72,
            120,
            "Portfolio weight by latest CRSP market-cap bucket using abs(prc) * shrout * 1000",
            size=17,
            fill=MUTED,
        )
    )
    elements.extend(_axes(x0=plot_left, y0=plot_top, width=plot_width, height=plot_height))
    elements.extend(_y_grid(y_min=0.0, y_max=max_pct, plot_left=plot_left, plot_top=plot_top, plot_width=plot_width, plot_height=plot_height, currency=False))

    for idx, (_, row) in enumerate(frame.iterrows(), start=0):
        x = plot_left + 55.0 + idx * (bar_width + gap)
        height = 0.0 if max_pct == 0 else (float(row["portfolio_weight_pct"]) / max_pct) * plot_height
        y = bottom - height
        fill = PORTFOLIO if idx % 2 == 0 else BENCHMARK
        elements.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width:.2f}" height="{height:.2f}" '
            f'rx="14" fill="{fill}" />'
        )
        elements.append(
            _text(x + bar_width / 2 - 24, y - 10, _format_pct(float(row["portfolio_weight"])), size=18, weight="700")
        )
        elements.append(
            _text(x + 2, bottom + 34, str(row["market_cap_bucket"]), size=17, weight="700")
        )
        elements.append(
            _text(
                x + 26,
                bottom + 58,
                f"n={int(row['constituent_count'])}",
                size=15,
                fill=MUTED,
            )
        )

    elements.append(
        _text(
            72,
            690,
            "Bucket weights use current holdings and latest persisted CRSP size fields; they are not a history of past portfolio composition.",
            size=15,
            fill=MUTED,
        )
    )
    elements.append("</svg>")
    return "".join(elements)


def _render_sharpe_card(
    *,
    sharpe_summary: object,
    benchmark_ticker: str,
) -> str:
    if not isinstance(sharpe_summary, dict):
        raise ValueError("Sharpe summary must be a mapping.")

    sharpe = _safe_float(sharpe_summary.get("annualized_sharpe"))
    mean_excess = _safe_float(sharpe_summary.get("mean_daily_excess_return"))
    excess_vol = _safe_float(sharpe_summary.get("daily_excess_volatility"))
    observations = int(sharpe_summary.get("observations", 0))

    elements = [_svg_root()]
    elements.append(_panel_rect(140, 95, CHART_WIDTH - 280, CHART_HEIGHT - 190))
    elements.append(_text(200, 185, "Annualized Sharpe Ratio", size=32, weight="700"))
    elements.append(
        _text(
            200,
            220,
            f"Daily portfolio proxy excess return over WRDS risk-free rate; benchmark context: {benchmark_ticker}",
            size=17,
            fill=MUTED,
        )
    )
    elements.append(
        _text(
            580,
            390,
            _format_number(sharpe, 2),
            size=128,
            weight="700",
            fill=PORTFOLIO,
            anchor="middle",
        )
    )
    elements.append(_info_box(250, 450, 660, 120, [
        f"Mean daily excess return: {_format_pct(mean_excess)}",
        f"Daily excess volatility: {_format_pct(excess_vol)}",
        f"Observations: {observations}",
    ]))
    elements.append(_text(200, 620, NOTE, size=15, fill=MUTED))
    elements.append("</svg>")
    return "".join(elements)


def _svg_root() -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{CHART_WIDTH}" height="{CHART_HEIGHT}" '
        f'viewBox="0 0 {CHART_WIDTH} {CHART_HEIGHT}" role="img">'
        f'<rect width="{CHART_WIDTH}" height="{CHART_HEIGHT}" fill="{BACKGROUND}" />'
    )


def _panel_rect(x: float, y: float, width: float, height: float) -> str:
    return (
        f'<rect x="{x:.2f}" y="{y:.2f}" width="{width:.2f}" height="{height:.2f}" '
        f'rx="28" fill="{PANEL}" stroke="{GRID}" stroke-width="2" />'
    )


def _text(
    x: float,
    y: float,
    value: object,
    *,
    size: int,
    weight: str = "400",
    fill: str = TEXT,
    anchor: str = "start",
    rotate: int | None = None,
) -> str:
    transform = ""
    if rotate is not None:
        transform = f' transform="rotate({rotate} {x:.2f} {y:.2f})"'
    safe = html.escape("" if value is None else str(value))
    return (
        f'<text x="{x:.2f}" y="{y:.2f}" font-family="Georgia, serif" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}"{transform}>{safe}</text>'
    )


def _axes(*, x0: float, y0: float, width: float, height: float) -> list[str]:
    return [
        f'<line x1="{x0:.2f}" y1="{y0 + height:.2f}" x2="{x0 + width:.2f}" y2="{y0 + height:.2f}" stroke="{TEXT}" stroke-width="2" />',
        f'<line x1="{x0:.2f}" y1="{y0:.2f}" x2="{x0:.2f}" y2="{y0 + height:.2f}" stroke="{TEXT}" stroke-width="2" />',
    ]


def _y_grid(
    *,
    y_min: float,
    y_max: float,
    plot_left: float,
    plot_top: float,
    plot_width: float,
    plot_height: float,
    currency: bool,
) -> list[str]:
    steps = 5
    elements: list[str] = []
    if math.isclose(y_min, y_max):
        y_max = y_min + 1.0

    for idx in range(steps + 1):
        value = y_min + ((y_max - y_min) * idx / steps)
        y = plot_top + plot_height - (plot_height * idx / steps)
        elements.append(
            f'<line x1="{plot_left:.2f}" y1="{y:.2f}" x2="{plot_left + plot_width:.2f}" y2="{y:.2f}" stroke="{GRID}" stroke-width="1.2" stroke-dasharray="4 8" />'
        )
        label = f"${value:,.0f}" if currency else f"{value:.0f}%"
        elements.append(_text(plot_left - 20, y + 5, label, size=15, fill=MUTED, anchor="end"))
    return elements


def _date_ticks(
    dates: pd.Series,
    *,
    plot_left: float,
    plot_width: float,
    y: float,
) -> list[str]:
    unique_dates = pd.to_datetime(dates).sort_values().reset_index(drop=True)
    tick_indices = sorted(set([0, len(unique_dates) // 3, (2 * len(unique_dates)) // 3, len(unique_dates) - 1]))
    elements: list[str] = []
    start = float(unique_dates.min().toordinal())
    end = float(unique_dates.max().toordinal())
    for idx in tick_indices:
        tick_date = unique_dates.iloc[idx]
        if math.isclose(start, end):
            x = plot_left + (plot_width / 2.0)
        else:
            x = plot_left + ((tick_date.toordinal() - start) / (end - start)) * plot_width
        elements.append(_text(float(x), y, tick_date.strftime("%Y-%m"), size=15, fill=MUTED, anchor="middle"))
    return elements


def _numeric_ticks(
    *,
    minimum: float,
    maximum: float,
    plot_left: float,
    plot_top: float,
    plot_width: float,
    plot_height: float,
    suffix: str,
) -> list[str]:
    elements: list[str] = []
    steps = 4
    for idx in range(steps + 1):
        value = minimum + ((maximum - minimum) * idx / steps)
        x = plot_left + plot_width * idx / steps
        y = plot_top + plot_height - plot_height * idx / steps
        elements.append(_text(x, plot_top + plot_height + 28, f"{value:.1f}{suffix}", size=15, fill=MUTED, anchor="middle"))
        elements.append(_text(plot_left - 18, y + 5, f"{value:.1f}{suffix}", size=15, fill=MUTED, anchor="end"))
    return elements


def _legend_swatch(x: float, y: float, fill: str, label: str) -> str:
    return (
        f'<rect x="{x:.2f}" y="{y - 11:.2f}" width="30" height="8" rx="4" fill="{fill}" />'
        + _text(x + 42, y, label, size=16, weight="700")
    )


def _line_path(points: list[tuple[float, float]]) -> str:
    if not points:
        raise ValueError("Cannot render a line with zero points.")
    head_x, head_y = points[0]
    commands = [f"M {head_x:.2f} {head_y:.2f}"]
    commands.extend([f"L {x:.2f} {y:.2f}" for x, y in points[1:]])
    return " ".join(commands)


def _scale_dates(values: pd.Series, start_px: float, end_px: float) -> pd.Series:
    series = pd.to_datetime(values, errors="raise")
    start = float(series.min().toordinal())
    end = float(series.max().toordinal())
    ordinals = series.map(pd.Timestamp.toordinal).astype("float64")
    if math.isclose(start, end):
        return pd.Series([start_px + ((end_px - start_px) / 2.0)] * len(series), index=series.index)
    return start_px + ((ordinals - start) / (end - start)) * (end_px - start_px)


def _scale_numeric(
    values: pd.Series,
    data_min: float,
    data_max: float,
    pixel_min: float,
    pixel_max: float,
) -> pd.Series:
    series = pd.to_numeric(values, errors="coerce").astype("float64")
    if math.isclose(data_min, data_max):
        return pd.Series([pixel_min + ((pixel_max - pixel_min) / 2.0)] * len(series), index=series.index)
    return pixel_min + ((series - data_min) / (data_max - data_min)) * (pixel_max - pixel_min)


def _info_box(x: float, y: float, width: float, height: float, lines: list[str]) -> str:
    line_elements = [
        _text(x + 28, y + 38 + idx * 32, line, size=18, fill=TEXT)
        for idx, line in enumerate(lines)
    ]
    return (
        f'<rect x="{x:.2f}" y="{y:.2f}" width="{width:.2f}" height="{height:.2f}" rx="20" '
        f'fill="{BACKGROUND}" stroke="{GRID}" stroke-width="1.5" />'
        + "".join(line_elements)
    )


def _format_pct(value: float) -> str:
    if not math.isfinite(value):
        return "n/a"
    return f"{value * 100:.2f}%"


def _format_number(value: float, digits: int) -> str:
    if not math.isfinite(value):
        return "n/a"
    return f"{value:.{digits}f}"


def _safe_float(value: object) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return numeric


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=path.suffix,
        delete=False,
        dir=path.parent,
        encoding="utf-8",
    ) as tmp:
        tmp_path = Path(tmp.name)
        tmp.write(content)

    try:
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
