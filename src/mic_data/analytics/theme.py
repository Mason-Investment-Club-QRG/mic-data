from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ClubPalette:
    green: str
    green_soft: str
    gold: str
    gold_soft: str
    gold_dark: str
    silver: str
    silver_deep: str
    white: str
    ink: str
    muted: str
    canvas: str


CLUB_COLORS = ClubPalette(
    green="#004E38",
    green_soft="#3A7A69",
    gold="#B79257",
    gold_soft="#D4C29F",
    gold_dark="#866F45",
    silver="#D8DCDB",
    silver_deep="#9EA8A4",
    white="#FFFFFF",
    ink="#173229",
    muted="#52655F",
    canvas="#F6F4EE",
)


def build_theme_rc() -> dict[str, object]:
    return {
        "figure.facecolor": CLUB_COLORS.canvas,
        "axes.facecolor": CLUB_COLORS.white,
        "axes.edgecolor": CLUB_COLORS.silver_deep,
        "axes.labelcolor": CLUB_COLORS.ink,
        "axes.titlecolor": CLUB_COLORS.ink,
        "grid.color": CLUB_COLORS.silver,
        "grid.alpha": 0.85,
        "text.color": CLUB_COLORS.ink,
        "xtick.color": CLUB_COLORS.ink,
        "ytick.color": CLUB_COLORS.ink,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.family": "DejaVu Sans",
        "axes.titlepad": 12.0,
        "legend.borderpad": 0.6,
    }


def holdings_palette(n_colors: int) -> list[str]:
    palette = [
        CLUB_COLORS.green,
        CLUB_COLORS.green_soft,
        CLUB_COLORS.gold,
        CLUB_COLORS.gold_dark,
        CLUB_COLORS.silver_deep,
    ]
    return palette[:n_colors] if n_colors <= len(palette) else palette + [CLUB_COLORS.gold_soft] * (n_colors - len(palette))


def cap_mix_palette(n_colors: int) -> list[str]:
    palette = [
        CLUB_COLORS.green,
        CLUB_COLORS.gold,
        CLUB_COLORS.green_soft,
        CLUB_COLORS.gold_dark,
        CLUB_COLORS.silver_deep,
    ]
    return palette[:n_colors] if n_colors <= len(palette) else palette + [CLUB_COLORS.gold_soft] * (n_colors - len(palette))
