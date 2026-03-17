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
from mic_data.models.ff_factor_matrix import FACTOR_BETA_COLUMNS


NOTE = (
    "Proxy series note: portfolio returns use the current holdings snapshot applied backward; "
    "historical composition changes are not yet modeled."
)
FF3_NOTE = (
    "FF3 note: return-based exposure comes from the portfolio proxy return series; "
    "holdings-based exposure uses the current holdings weights."
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
        "ff3_portfolio_exposure_chart": charts_dir / "ff3_portfolio_exposure.svg",
        "ff3_exposure_comparison_chart": charts_dir / "ff3_exposure_comparison.svg",
        "ff3_factor_risk_chart": charts_dir / "ff3_factor_risk_contributions.svg",
        "ff3_security_heatmap_chart": charts_dir / "ff3_security_heatmap.svg",
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
    _save_svg_figure(
        chart_paths["ff3_portfolio_exposure_chart"],
        _plot_ff3_portfolio_exposure_chart(result),
    )
    _save_svg_figure(
        chart_paths["ff3_exposure_comparison_chart"],
        _plot_ff3_exposure_comparison_chart(result),
    )
    _save_svg_figure(
        chart_paths["ff3_factor_risk_chart"],
        _plot_ff3_factor_risk_chart(result),
    )
    _save_svg_figure(
        chart_paths["ff3_security_heatmap_chart"],
        _plot_ff3_security_heatmap_chart(result),
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


def _plot_ff3_portfolio_exposure_chart(result: PortfolioAnalyticsResult) -> plt.Figure:
    exposure = result.ff3_analysis.portfolio_return_exposure[list(FACTOR_BETA_COLUMNS)].rename(
        index=_factor_label
    )
    exposure_frame = exposure.reset_index()
    exposure_frame.columns = ["factor", "exposure"]

    fig, ax = plt.subplots(figsize=(10.6, 7.0))
    _style_axes(ax, grid_axis="y")

    sns.barplot(
        data=exposure_frame,
        x="factor",
        y="exposure",
        hue="factor",
        palette=cap_mix_palette(len(exposure_frame)),
        legend=False,
        ax=ax,
    )
    ax.axhline(0.0, color=CLUB_COLORS.silver_deep, linewidth=1.2)
    ax.set_xlabel("")
    ax.set_ylabel("Estimated beta")

    for patch, (_, row) in zip(ax.patches, exposure_frame.iterrows(), strict=False):
        value = float(row["exposure"])
        ax.text(
            patch.get_x() + patch.get_width() / 2.0,
            value + (0.03 if value >= 0 else -0.06),
            f"{value:.2f}",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=12,
            fontweight="bold",
            color=CLUB_COLORS.ink,
        )

    ax.text(
        0.98,
        0.05,
        "\n".join(
            [
                f"Alpha / day: {_format_percent(result.ff3_analysis.portfolio_return_exposure['alpha'])}",
                f"R-squared: {_format_number(result.ff3_analysis.portfolio_return_exposure['r2'], 2)}",
                f"Obs: {int(result.ff3_analysis.portfolio_return_exposure['n_obs'])}",
            ]
        ),
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
        title="Portfolio FF3 Return-Based Exposure",
        subtitle="Mkt-RF, SMB, and HML exposures estimated from the portfolio proxy return series",
    )
    _add_note(fig, _ff3_note(result))
    fig.tight_layout(rect=(0, 0.06, 1, 0.92))
    return fig


def _plot_ff3_exposure_comparison_chart(result: PortfolioAnalyticsResult) -> plt.Figure:
    comparison = result.ff3_analysis.tables["portfolio_exposure_comparison"].copy()
    plot_frame = comparison.melt(
        id_vars=["exposure_method"],
        value_vars=list(FACTOR_BETA_COLUMNS),
        var_name="factor",
        value_name="exposure",
    )
    plot_frame["factor"] = plot_frame["factor"].map(_factor_label)
    plot_frame["exposure_method"] = plot_frame["exposure_method"].map(
        {
            "return_based": "Return-based",
            "holdings_based": "Holdings-based",
        }
    )

    fig, ax = plt.subplots(figsize=(11.0, 7.0))
    _style_axes(ax, grid_axis="y")

    sns.barplot(
        data=plot_frame,
        x="factor",
        y="exposure",
        hue="exposure_method",
        palette=[CLUB_COLORS.green, CLUB_COLORS.gold],
        ax=ax,
    )
    ax.axhline(0.0, color=CLUB_COLORS.silver_deep, linewidth=1.2)
    ax.set_xlabel("")
    ax.set_ylabel("Estimated beta")
    ax.legend(
        title="Exposure source",
        frameon=True,
        fancybox=True,
        facecolor=CLUB_COLORS.white,
        edgecolor=CLUB_COLORS.silver_deep,
        loc="upper right",
    )

    alpha_text = [
        f"Return-based alpha / day: {_format_percent(result.ff3_analysis.portfolio_return_exposure['alpha'])}"
    ]
    if result.ff3_analysis.portfolio_holdings_exposure is not None:
        alpha_text.append(
            "Holdings-based alpha / day: "
            f"{_format_percent(result.ff3_analysis.portfolio_holdings_exposure['alpha'])}"
        )
    ax.text(
        0.02,
        0.05,
        "\n".join(alpha_text),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=12,
        bbox={
            "boxstyle": "round,pad=0.45",
            "facecolor": CLUB_COLORS.white,
            "edgecolor": CLUB_COLORS.silver_deep,
            "linewidth": 1.0,
        },
    )

    _set_titles(
        fig,
        ax,
        title="Return-Based vs Holdings-Based FF3 Exposure",
        subtitle="Current holdings weights are compared against exposures estimated from the portfolio proxy return series",
    )
    _add_note(fig, _ff3_note(result))
    fig.tight_layout(rect=(0, 0.06, 1, 0.92))
    return fig


def _plot_ff3_factor_risk_chart(result: PortfolioAnalyticsResult) -> plt.Figure:
    contributions = result.ff3_analysis.factor_risk_contributions.copy()
    total_variance = float(result.ff3_analysis.portfolio_risk_summary["total_variance"])
    plot_frame = (
        contributions.rename("variance_contribution")
        .rename(index=_factor_label)
        .reset_index()
        .rename(columns={"index": "factor"})
    )
    plot_frame["share_of_total_variance"] = (
        plot_frame["variance_contribution"] / total_variance if total_variance > 0 else 0.0
    )
    plot_frame["share_pct"] = plot_frame["share_of_total_variance"] * 100.0
    plot_frame["bar_color"] = plot_frame["share_pct"].apply(
        lambda value: CLUB_COLORS.green if value >= 0 else CLUB_COLORS.gold_dark
    )

    fig, ax = plt.subplots(figsize=(10.8, 7.0))
    _style_axes(ax, grid_axis="y")

    ax.bar(
        plot_frame["factor"],
        plot_frame["share_pct"],
        color=plot_frame["bar_color"],
        width=0.62,
    )
    ax.axhline(0.0, color=CLUB_COLORS.silver_deep, linewidth=1.2)
    ax.set_xlabel("")
    ax.set_ylabel("Share of total variance")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0f}%"))

    for idx, row in plot_frame.iterrows():
        value = float(row["share_pct"])
        ax.text(
            idx,
            value + (0.7 if value >= 0 else -1.2),
            f"{value:.1f}%",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=12,
            fontweight="bold",
            color=CLUB_COLORS.ink,
        )

    risk = result.ff3_analysis.portfolio_risk_summary
    ax.text(
        0.98,
        0.05,
        "\n".join(
            [
                f"Factor share: {_format_percent(risk['factor_share'])}",
                f"Idiosyncratic share: {_format_percent(risk['idiosyncratic_share'])}",
                f"Volatility: {_format_percent(risk['volatility'])}",
                f"R-squared: {_format_number(risk['r2'], 2)}",
            ]
        ),
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
        title="FF3 Factor Risk Contributions",
        subtitle="Each bar shows the factor's contribution to total portfolio variance",
    )
    _add_note(fig, _ff3_note(result))
    fig.tight_layout(rect=(0, 0.06, 1, 0.92))
    return fig


def _plot_ff3_security_heatmap_chart(result: PortfolioAnalyticsResult) -> plt.Figure:
    modeled = result.holdings_ff3_loadings[result.holdings_ff3_loadings["ff3_modeled"]].copy()
    if modeled.empty:
        raise ValueError("No current holdings have FF3 loadings available for the heatmap.")

    modeled["row_label"] = modeled.apply(
        lambda row: f"{row['ticker']}  ({float(row['portfolio_weight_pct']):.1f}%)",
        axis=1,
    )
    heatmap_frame = modeled.set_index("row_label")[list(FACTOR_BETA_COLUMNS)].rename(
        columns=_factor_label
    )
    max_abs = float(heatmap_frame.abs().to_numpy().max())
    vmax = max(0.5, min(2.0, max_abs))

    fig_height = max(6.8, 0.55 * len(heatmap_frame) + 2.2)
    fig, ax = plt.subplots(figsize=(9.2, fig_height))
    fig.patch.set_facecolor(CLUB_COLORS.canvas)
    ax.set_facecolor(CLUB_COLORS.white)

    cmap = sns.diverging_palette(
        h_neg=160,
        h_pos=42,
        s=85,
        l=45,
        as_cmap=True,
    )
    sns.heatmap(
        heatmap_frame,
        cmap=cmap,
        center=0.0,
        vmin=-vmax,
        vmax=vmax,
        annot=True,
        fmt=".2f",
        linewidths=0.8,
        linecolor=CLUB_COLORS.silver,
        cbar_kws={"label": "Estimated beta"},
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", rotation=0)

    skipped_count = int(result.summary["ff3"]["current_holdings_skipped_count"])
    subtitle = "Current holdings only, ordered by latest portfolio weight"
    if skipped_count > 0:
        subtitle += f" | {skipped_count} holding(s) omitted for insufficient FF3 history"

    _set_titles(
        fig,
        ax,
        title="Current Holdings FF3 Beta Heatmap",
        subtitle=subtitle,
    )
    _add_note(fig, _ff3_note(result))
    fig.tight_layout(rect=(0, 0.06, 1, 0.92))
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


def _factor_label(value: str) -> str:
    mapping = {
        "mkt_rf": "Mkt-RF",
        "smb": "SMB",
        "hml": "HML",
    }
    return mapping.get(str(value), str(value))


def _ff3_note(result: PortfolioAnalyticsResult) -> str:
    note = FF3_NOTE
    skipped = int(result.summary["ff3"]["current_holdings_skipped_count"])
    if skipped > 0:
        note += f" {skipped} current holding(s) were omitted for insufficient overlapping FF3 history."
    return note
