from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path

_MATPLOTLIB_CACHE_DIR = Path(tempfile.gettempdir()) / "mic_data_matplotlib"
_MATPLOTLIB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MATPLOTLIB_CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(_MATPLOTLIB_CACHE_DIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import FuncFormatter

from mic_data.analytics.dashboard import PortfolioAnalyticsResult
from mic_data.analytics.theme import CLUB_COLORS, build_theme_rc, cap_mix_palette, holdings_palette


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
    _apply_theme()

    chart_paths = {
        "performance_chart": charts_dir / f"performance_vs_{benchmark_slug}.svg",
        "beta_chart": charts_dir / f"beta_vs_{benchmark_slug}.svg",
        "top_holdings_chart": charts_dir / "top_holdings.svg",
        "market_cap_mix_chart": charts_dir / "market_cap_mix.svg",
        "sharpe_card": charts_dir / "sharpe_ratio.svg",
    }

    _save_svg_figure(
        chart_paths["performance_chart"],
        _plot_performance_chart(
            comparison=result.benchmark_comparison,
            benchmark_ticker=benchmark_ticker,
        ),
    )
    _save_svg_figure(
        chart_paths["beta_chart"],
        _plot_beta_chart(
            beta_regression=result.beta_regression,
            benchmark_ticker=benchmark_ticker,
            beta_summary=result.summary["beta"],
        ),
    )
    _save_svg_figure(
        chart_paths["top_holdings_chart"],
        _plot_top_holdings_chart(
            holdings_snapshot=result.holdings_snapshot,
            top_n_holdings=top_n_holdings,
        ),
    )
    _save_svg_figure(
        chart_paths["market_cap_mix_chart"],
        _plot_market_cap_mix_chart(result.market_cap_mix),
    )
    _save_svg_figure(
        chart_paths["sharpe_card"],
        _plot_sharpe_card(result.summary["sharpe"]),
    )
    return chart_paths


def _apply_theme() -> None:
    sns.set_theme(
        context="talk",
        style="whitegrid",
        palette=[CLUB_COLORS.green, CLUB_COLORS.gold, CLUB_COLORS.green_soft],
        rc=build_theme_rc(),
    )


def _plot_performance_chart(
    *,
    comparison: pd.DataFrame,
    benchmark_ticker: str,
) -> plt.Figure:
    frame = comparison.copy().sort_values("trade_date")
    fig, ax = plt.subplots(figsize=(11.5, 7.0))
    _style_axes(ax)

    ax.axhline(100.0, color=CLUB_COLORS.silver_deep, linewidth=1.3, linestyle=(0, (4, 4)))
    ax.plot(
        frame["trade_date"],
        frame["portfolio_value"],
        color=CLUB_COLORS.green,
        linewidth=3.4,
        label="Portfolio",
    )
    ax.plot(
        frame["trade_date"],
        frame["benchmark_value"],
        color=CLUB_COLORS.gold,
        linewidth=3.0,
        label=benchmark_ticker,
    )
    ax.fill_between(
        frame["trade_date"],
        frame["portfolio_value"],
        100.0,
        color=CLUB_COLORS.green_soft,
        alpha=0.10,
    )
    ax.fill_between(
        frame["trade_date"],
        frame["benchmark_value"],
        100.0,
        color=CLUB_COLORS.gold_soft,
        alpha=0.09,
    )

    last_row = frame.iloc[-1]
    for label, color, y_value in (
        ("Portfolio", CLUB_COLORS.green, float(last_row["portfolio_value"])),
        (benchmark_ticker, CLUB_COLORS.gold, float(last_row["benchmark_value"])),
    ):
        ax.scatter(last_row["trade_date"], y_value, s=90, color=color, edgecolor=CLUB_COLORS.white, zorder=5)
        ax.annotate(
            f"{label} ${y_value:,.2f}",
            xy=(last_row["trade_date"], y_value),
            xytext=(12, 0 if label == "Portfolio" else -18),
            textcoords="offset points",
            fontsize=12,
            fontweight="bold",
            color=color,
            va="center",
        )

    ax.set_ylabel("Value of $100")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"${value:,.0f}"))
    locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

    ax.legend(
        frameon=True,
        fancybox=True,
        framealpha=1.0,
        facecolor=CLUB_COLORS.white,
        edgecolor=CLUB_COLORS.silver_deep,
        loc="upper left",
    )
    _set_titles(
        fig,
        ax,
        title="Portfolio Proxy vs Benchmark Growth of $100",
        subtitle=f"Benchmark sourced from persisted WRDS returns: {benchmark_ticker}",
    )
    _add_note(fig, NOTE)
    fig.tight_layout(rect=(0, 0.06, 1, 0.92))
    return fig


def _plot_beta_chart(
    *,
    beta_regression: pd.DataFrame,
    benchmark_ticker: str,
    beta_summary: object,
) -> plt.Figure:
    if not isinstance(beta_summary, dict):
        raise ValueError("Beta summary must be a mapping.")

    frame = beta_regression.copy().sort_values("trade_date")
    frame["benchmark_pct"] = frame["benchmark_ret"] * 100.0
    frame["portfolio_pct"] = frame["portfolio_ret"] * 100.0
    frame["fitted_pct"] = frame["fitted_portfolio_ret"] * 100.0

    fig, ax = plt.subplots(figsize=(10.0, 8.2))
    _style_axes(ax)

    sns.regplot(
        data=frame,
        x="benchmark_pct",
        y="portfolio_pct",
        ci=None,
        truncate=False,
        scatter_kws={
            "s": 52,
            "alpha": 0.55,
            "color": CLUB_COLORS.green,
            "edgecolor": CLUB_COLORS.white,
            "linewidths": 0.7,
        },
        line_kws={"color": CLUB_COLORS.gold, "linewidth": 3.2},
        ax=ax,
    )

    ax.axhline(0.0, color=CLUB_COLORS.silver_deep, linewidth=1.2)
    ax.axvline(0.0, color=CLUB_COLORS.silver_deep, linewidth=1.2)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0f}%"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0f}%"))
    ax.set_xlabel(f"{benchmark_ticker} return")
    ax.set_ylabel("Portfolio return")

    stats_text = "\n".join(
        [
            f"Beta: {_format_number(beta_summary.get('beta'), 2)}",
            f"Alpha / period: {_format_percent(beta_summary.get('alpha_per_period'))}",
            f"R-squared: {_format_number(beta_summary.get('r_squared'), 2)}",
            f"Observations: {int(beta_summary.get('observations', 0))}",
            f"Frequency: {str(beta_summary.get('frequency', 'n/a')).title()}",
        ]
    )
    ax.text(
        0.98,
        0.05,
        stats_text,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=12,
        bbox={
            "boxstyle": "round,pad=0.5",
            "facecolor": CLUB_COLORS.white,
            "edgecolor": CLUB_COLORS.silver_deep,
            "linewidth": 1.0,
        },
    )

    _set_titles(
        fig,
        ax,
        title=f"Portfolio Beta vs {benchmark_ticker}",
        subtitle=(
            f"Regression on {str(beta_summary.get('frequency', 'weekly')).title()} "
            f"compounded returns"
        ),
    )
    _add_note(fig, NOTE)
    fig.tight_layout(rect=(0, 0.06, 1, 0.92))
    return fig


def _plot_top_holdings_chart(
    *,
    holdings_snapshot: pd.DataFrame,
    top_n_holdings: int,
) -> plt.Figure:
    top = (
        holdings_snapshot.nsmallest(top_n_holdings, "weight_rank")
        .sort_values("portfolio_weight_pct", ascending=True)
        .copy()
    )
    top["label"] = top.apply(
        lambda row: (
            f"{row['ticker']}  |  {row['name']}"
            if pd.notna(row["name"]) and str(row["name"]).strip()
            else str(row["ticker"])
        ),
        axis=1,
    )

    fig, ax = plt.subplots(figsize=(11.2, 7.0))
    _style_axes(ax, grid_axis="x")

    palette = holdings_palette(len(top))
    sns.barplot(
        data=top,
        x="portfolio_weight_pct",
        y="label",
        hue="label",
        orient="h",
        palette=palette,
        legend=False,
        ax=ax,
    )

    ax.set_xlabel("Portfolio weight")
    ax.set_ylabel("")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0f}%"))
    for patch, (_, row) in zip(ax.patches, top.iterrows(), strict=False):
        ax.text(
            patch.get_width() + 0.35,
            patch.get_y() + patch.get_height() / 2.0,
            f"{float(row['portfolio_weight_pct']):.1f}%",
            va="center",
            fontsize=12,
            fontweight="bold",
            color=CLUB_COLORS.ink,
        )

    _set_titles(
        fig,
        ax,
        title=f"Top {top_n_holdings} Holdings",
        subtitle="Current holdings snapshot weighted by latest persisted WRDS prices",
    )
    _add_note(
        fig,
        "Weights reflect latest holdings shares times latest persisted prices, not historical trade-by-trade allocations.",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.92))
    return fig


def _plot_market_cap_mix_chart(market_cap_mix: pd.DataFrame) -> plt.Figure:
    frame = market_cap_mix.copy().sort_values("bucket_order")
    fig, ax = plt.subplots(figsize=(10.8, 7.0))
    _style_axes(ax, grid_axis="y")

    sns.barplot(
        data=frame,
        x="market_cap_bucket",
        y="portfolio_weight_pct",
        hue="market_cap_bucket",
        palette=cap_mix_palette(len(frame)),
        legend=False,
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Portfolio weight")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0f}%"))

    for patch, (_, row) in zip(ax.patches, frame.iterrows(), strict=False):
        ax.text(
            patch.get_x() + patch.get_width() / 2.0,
            patch.get_height() + 0.6,
            f"{float(row['portfolio_weight_pct']):.1f}%\n n={int(row['constituent_count'])}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            color=CLUB_COLORS.ink,
        )

    _set_titles(
        fig,
        ax,
        title="Market-Cap Mix",
        subtitle="Current holdings weighted by latest CRSP market-cap bucket",
    )
    _add_note(
        fig,
        "Bucket weights use current holdings and latest persisted CRSP size fields; they are not a history of past portfolio composition.",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.92))
    return fig


def _plot_sharpe_card(sharpe_summary: object) -> plt.Figure:
    if not isinstance(sharpe_summary, dict):
        raise ValueError("Sharpe summary must be a mapping.")

    sharpe = _safe_float(sharpe_summary.get("annualized_sharpe"))
    mean_excess = _safe_float(sharpe_summary.get("mean_daily_excess_return"))
    daily_vol = _safe_float(sharpe_summary.get("daily_excess_volatility"))
    observations = int(sharpe_summary.get("observations", 0))

    fig, ax = plt.subplots(figsize=(10.4, 6.2))
    fig.patch.set_facecolor(CLUB_COLORS.canvas)
    ax.axis("off")

    card = FancyBboxPatch(
        (0.06, 0.12),
        0.88,
        0.74,
        transform=ax.transAxes,
        boxstyle="round,pad=0.02,rounding_size=20",
        linewidth=1.3,
        edgecolor=CLUB_COLORS.silver_deep,
        facecolor=CLUB_COLORS.white,
    )
    ax.add_patch(card)

    ax.text(
        0.12,
        0.80,
        "Annualized Sharpe Ratio",
        transform=ax.transAxes,
        fontsize=24,
        fontweight="bold",
        color=CLUB_COLORS.ink,
    )
    ax.text(
        0.12,
        0.73,
        "Daily portfolio proxy excess return over WRDS risk-free rate",
        transform=ax.transAxes,
        fontsize=13,
        color=CLUB_COLORS.muted,
    )
    ax.text(
        0.50,
        0.50,
        _format_number(sharpe, 2),
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=78,
        fontweight="bold",
        color=CLUB_COLORS.green,
    )
    stats = [
        f"Mean daily excess return: {_format_percent(mean_excess)}",
        f"Daily excess volatility: {_format_percent(daily_vol)}",
        f"Observations: {observations}",
    ]
    for idx, line in enumerate(stats):
        ax.text(
            0.12,
            0.30 - idx * 0.07,
            line,
            transform=ax.transAxes,
            fontsize=14,
            color=CLUB_COLORS.ink,
        )

    _add_note(fig, NOTE)
    fig.tight_layout(rect=(0, 0.05, 1, 0.98))
    return fig


def _style_axes(ax: plt.Axes, *, grid_axis: str = "both") -> None:
    ax.set_facecolor(CLUB_COLORS.white)
    ax.grid(True, axis=grid_axis, color=CLUB_COLORS.silver, linewidth=1.0, alpha=0.85)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(CLUB_COLORS.silver_deep)
    ax.spines["bottom"].set_color(CLUB_COLORS.silver_deep)
    ax.tick_params(colors=CLUB_COLORS.ink)
    ax.xaxis.label.set_color(CLUB_COLORS.ink)
    ax.yaxis.label.set_color(CLUB_COLORS.ink)


def _set_titles(fig: plt.Figure, ax: plt.Axes, *, title: str, subtitle: str) -> None:
    fig.suptitle(
        title,
        x=0.06,
        y=0.97,
        ha="left",
        fontsize=24,
        fontweight="bold",
        color=CLUB_COLORS.ink,
    )
    ax.set_title(
        subtitle,
        loc="left",
        fontsize=13,
        color=CLUB_COLORS.muted,
        pad=14,
    )


def _add_note(fig: plt.Figure, note: str) -> None:
    fig.text(
        0.06,
        0.025,
        note,
        ha="left",
        va="bottom",
        fontsize=11.5,
        color=CLUB_COLORS.muted,
    )


def _save_svg_figure(path: Path, fig: plt.Figure) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w+b",
        suffix=path.suffix,
        delete=False,
        dir=path.parent,
    ) as tmp:
        tmp_path = Path(tmp.name)

    try:
        fig.savefig(
            tmp_path,
            format="svg",
            bbox_inches="tight",
            facecolor=fig.get_facecolor(),
        )
        tmp_path.replace(path)
    finally:
        plt.close(fig)
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def _safe_float(value: object) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return numeric


def _format_number(value: object, digits: int) -> str:
    numeric = _safe_float(value)
    if not math.isfinite(numeric):
        return "n/a"
    return f"{numeric:.{digits}f}"


def _format_percent(value: object) -> str:
    numeric = _safe_float(value)
    if not math.isfinite(numeric):
        return "n/a"
    return f"{numeric * 100:.2f}%"
