#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_PATH = PROJECT_DIR / "abl_type_weather_results" / "abl_type_weather_merged.csv"
OUT_PATH = SCRIPT_DIR / "monthly_ablh_distribution_color_violin.png"

MPL_DIR = SCRIPT_DIR / ".mplconfig"
MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(MPL_DIR)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


MONTH_ORDER = [11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9]
MONTH_LABELS = ["Nov", "Dec", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep"]
CONDITION_ORDER = ["Clear", "Cloudy", "Fog/Mist"]
CONDITION_LABELS = {
    "Clear": "Clear",
    "Cloudy": "Cloudy",
    "Fog/Mist": "Fog",
}

COLORS = {
    "violin": "#f7ada8",
    "violin_edge": "#ff7368",
    "box": "#ffffff",
    "box_edge": "#ff7368",
    "median": "#ff7368",
    "grid": "#c8cdd3",
    "Clear": "#f28f89",
    "Cloudy": "#a8c95a",
    "Fog/Mist": "#23c0d3",
    "All": "#b784ef",
}

MARKERS = {
    "Clear": "o",
    "Cloudy": "s",
    "Fog/Mist": "^",
    "All": "D",
}


def set_style() -> None:
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 10.5
    plt.rcParams["axes.titlesize"] = 10.5
    plt.rcParams["axes.labelsize"] = 10.5
    plt.rcParams["xtick.labelsize"] = 10.5
    plt.rcParams["ytick.labelsize"] = 10.5
    plt.rcParams["legend.fontsize"] = 10.5


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, usecols=["month", "ablh_m", "weather_condition"])
    df = df.dropna(subset=["month", "ablh_m", "weather_condition"]).copy()
    df["month"] = df["month"].astype(int)
    df = df[df["month"].isin(MONTH_ORDER)]
    df = df[df["weather_condition"].isin(CONDITION_ORDER)]
    df = df[np.isfinite(df["ablh_m"])]
    return df


def compute_monthly_means(df: pd.DataFrame) -> dict[str, pd.Series]:
    monthly_means: dict[str, pd.Series] = {}
    monthly_means["All"] = df.groupby("month")["ablh_m"].mean().reindex(MONTH_ORDER)
    for condition in CONDITION_ORDER:
        monthly_means[condition] = (
            df.loc[df["weather_condition"] == condition].groupby("month")["ablh_m"].mean().reindex(MONTH_ORDER)
        )
    return monthly_means


def build_monthly_distributions(df: pd.DataFrame) -> list[np.ndarray]:
    return [df.loc[df["month"] == month, "ablh_m"].to_numpy() for month in MONTH_ORDER]


def plot_monthly_distribution(df: pd.DataFrame) -> None:
    set_style()
    monthly_means = compute_monthly_means(df)
    monthly_distributions = build_monthly_distributions(df)
    positions = np.arange(1, len(MONTH_ORDER) + 1)

    fig_width_cm = 18.0
    fig_height_cm = 9.5
    left_margin = 0.075
    right_margin = 0.985
    bottom_margin = 0.17
    top_margin = 0.90
    axes_width_cm = fig_width_cm * (right_margin - left_margin)
    category_spacing_cm = axes_width_cm / len(MONTH_ORDER)
    box_width_data = 0.5 / category_spacing_cm

    fig, ax = plt.subplots(figsize=(fig_width_cm / 2.54, fig_height_cm / 2.54))

    violin = ax.violinplot(
        monthly_distributions,
        positions=positions,
        widths=0.90,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for body in violin["bodies"]:
        body.set_facecolor(COLORS["violin"])
        body.set_edgecolor(COLORS["violin_edge"])
        body.set_linewidth(1.0)
        body.set_alpha(0.82)

    ax.boxplot(
        monthly_distributions,
        positions=positions,
        widths=box_width_data,
        patch_artist=True,
        showfliers=False,
        boxprops=dict(facecolor=COLORS["box"], edgecolor=COLORS["box_edge"], linewidth=1.2, alpha=0.92),
        whiskerprops=dict(color=COLORS["box_edge"], linewidth=1.0),
        capprops=dict(color=COLORS["box_edge"], linewidth=1.0),
        medianprops=dict(color=COLORS["median"], linewidth=1.6),
    )

    for condition in CONDITION_ORDER:
        ax.scatter(
            positions,
            monthly_means[condition].to_numpy(),
            color=COLORS[condition],
            marker=MARKERS[condition],
            s=40,
            linewidths=0.7,
            edgecolors="white",
            label=CONDITION_LABELS[condition],
            zorder=4,
        )

    ax.scatter(
        positions,
        monthly_means["All"].to_numpy(),
        color=COLORS["All"],
        marker=MARKERS["All"],
        s=42,
        linewidths=0.7,
        edgecolors="white",
        label="All",
        zorder=4,
    )

    ax.set_xlim(0.5, len(MONTH_ORDER) + 0.5)
    ax.set_ylim(30, 700)
    ax.set_xticks(positions)
    ax.set_xticklabels(MONTH_LABELS, rotation=0, ha="center")
    ax.set_yticks(np.arange(100, 701, 100))
    ax.set_ylabel("ABLH (m)")

    ax.grid(axis="y", linestyle=(0, (4, 2)), linewidth=0.7, color=COLORS["grid"], alpha=0.95)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend = ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.065),
        ncol=4,
        frameon=False,
        columnspacing=1.2,
        handletextpad=0.6,
    )

    fig.subplots_adjust(left=left_margin, right=right_margin, bottom=bottom_margin, top=top_margin)
    fig.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    df = load_data()
    plot_monthly_distribution(df)
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
