#!/usr/bin/env python3
"""
Create a compact two-panel summary figure for LLJ influence on ABLH.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from plot_font_utils import get_available_serif_font


CONDITION_NAMES = {
    0: "Clear",
    1: "Cloudy",
    2: "Fog/Mist",
}

NON_LLJ_COLOR = "#BFC5CC"
LLJ_COLOR = "#D95319"  # MATLAB-like orange
ACCENT_COLOR = "#D95319"
TEXT_COLOR = "#222222"


def _set_style() -> None:
    serif_font = get_available_serif_font()
    plt.rcParams.update(
        {
            "figure.dpi": 180,
            "savefig.dpi": 300,
            "font.size": 10.5,
            "font.family": serif_font,
            "axes.linewidth": 0.8,
            "axes.edgecolor": "#333333",
            "axes.labelcolor": TEXT_COLOR,
            "axes.titlesize": 10.5,
            "axes.labelsize": 10.5,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "legend.fontsize": 10.5,
            "axes.grid": True,
            "grid.color": "#C8C8C8",
            "grid.alpha": 0.30,
            "grid.linewidth": 0.45,
        }
    )


def _panel_label(ax: plt.Axes, label: str, title: str, subtitle: str | None = None) -> None:
    ax.text(
        0.01,
        1.035,
        f"{label} {title}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10.5,
        fontweight="normal",
        color=TEXT_COLOR,
    )
    if subtitle:
        ax.text(
            0.01,
            0.965,
            subtitle,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10.5,
            color=TEXT_COLOR,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.78, pad=1.0),
        )


def _despine(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=3, width=0.7, colors=TEXT_COLOR)


def _safe_corr(x: pd.Series, y: pd.Series) -> float:
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3:
        return np.nan
    return float(np.corrcoef(x[valid], y[valid])[0, 1])


def _panel_core_speed_vs_blh(ax: plt.Axes, fig: plt.Figure, df: pd.DataFrame) -> matplotlib.colorbar.Colorbar:
    llj = df.loc[df["llj_flag"], ["llj_core_speed_mps", "ablh_m"]].dropna().copy()
    x = llj["llj_core_speed_mps"].to_numpy(dtype=float)
    y = llj["ablh_m"].to_numpy(dtype=float)

    xbins = np.arange(5.0, 33.5, 0.75)
    ybins = np.arange(0.0, 1300.0, 35.0)
    counts, _, _, mesh = ax.hist2d(
        x,
        y,
        bins=[xbins, ybins],
        cmap="parula" if "parula" in plt.colormaps() else "turbo",
        cmin=1,
    )

    if np.isfinite(x).sum() >= 3:
        slope, intercept = np.polyfit(x, y, 1)
        xfit = np.linspace(np.nanpercentile(x, 1), np.nanpercentile(x, 99), 120)
        yfit = slope * xfit + intercept
        ax.plot(xfit, yfit, color="#111111", linewidth=1.4)

    llj["speed_bin"] = pd.cut(llj["llj_core_speed_mps"], bins=np.arange(5.0, max(17.5, np.nanmax(x) + 1.0), 1.5), include_lowest=True)
    grouped = llj.groupby("speed_bin", observed=False)["ablh_m"]
    centers, means, stds = [], [], []
    for interval, values in grouped:
        vals = values.dropna()
        if len(vals) < 20:
            continue
        centers.append(interval.mid)
        means.append(vals.mean())
        stds.append(vals.std())
    if centers:
        ax.errorbar(
            centers,
            means,
            yerr=np.array(stds) * 0.25,
            fmt="o-",
            color=ACCENT_COLOR,
            linewidth=1.4,
            markersize=3.0,
            capsize=2.2,
            elinewidth=0.8,
            label="Binned mean",
        )

    corr = _safe_corr(llj["llj_core_speed_mps"], llj["ablh_m"])
    _panel_label(ax, "(a)", "LLJ core speed and ABLH", f"Pearson R = {corr:.2f}")
    ax.set_xlabel("LLJ core speed (m s$^{-1}$)")
    ax.set_ylabel("ABLH (m)")
    ax.set_xlim(5, 32.5)
    ax.set_ylim(0.0, 1300.0)
    ax.set_yticks(np.arange(0.0, 1301.0, 200.0))
    ax.legend(loc="lower right", frameon=False)
    _despine(ax)

    cbar = fig.colorbar(mesh, ax=ax, pad=0.010, fraction=0.032)
    cbar.ax.tick_params(labelsize=10.5, length=2.5, width=0.6)
    return cbar


def _panel_blh_boxplot(ax: plt.Axes, df: pd.DataFrame) -> None:
    centers = np.arange(3)
    width = 0.30
    for offset, flag, color in [
        (-width / 1.35, False, NON_LLJ_COLOR),
        (width / 1.35, True, LLJ_COLOR),
    ]:
        data = []
        positions = []
        for idx in range(3):
            subset = df.loc[(df["condition_label"] == idx) & (df["llj_flag"] == flag), "ablh_m"].dropna()
            if subset.empty:
                continue
            data.append(subset.values)
            positions.append(centers[idx] + offset)
        if data:
            ax.boxplot(
                data,
                positions=positions,
                widths=width,
                patch_artist=True,
                showfliers=False,
                boxprops=dict(facecolor=color, alpha=0.68, edgecolor=color, linewidth=0.9),
                medianprops=dict(color="#111111", linewidth=1.2),
                whiskerprops=dict(color=color, linewidth=0.9),
                capprops=dict(color=color, linewidth=0.9),
            )

    means = df.groupby(["condition_label", "llj_flag"])["ablh_m"].mean()
    texts = []
    for idx, display_name in enumerate(["clear", "cloud", "fog"]):
        non_llj = means.get((idx, False), np.nan)
        llj = means.get((idx, True), np.nan)
        if np.isfinite(non_llj) and np.isfinite(llj):
            texts.append(f"{display_name}: +{llj - non_llj:.0f} m")

    _panel_label(ax, "(b)", "ABLH by weather regime")
    ax.set_xticks(centers)
    ax.set_xticklabels(["clear", "cloud", "fog"])
    ax.set_ylabel("")
    ax.set_ylim(0.0, 1300.0)
    ax.set_yticks(np.arange(0.0, 1301.0, 200.0))
    ax.tick_params(labelleft=True)
    handles = [
        plt.Line2D([0], [0], color=NON_LLJ_COLOR, lw=6, alpha=0.8, label="Non-LLJ"),
        plt.Line2D([0], [0], color=LLJ_COLOR, lw=6, alpha=0.8, label="LLJ"),
    ]
    ax.legend(handles=handles, loc="upper right", frameon=False)
    ax.text(
        0.03,
        0.95,
        "LLJ - non-LLJ\n" + "\n".join(texts),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10.5,
        color=TEXT_COLOR,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.78, pad=1.0),
    )
    _despine(ax)


def build_figure(df: pd.DataFrame, output_path: Path) -> None:
    _set_style()
    fig = plt.figure(figsize=(18.0 / 2.54, 7.0 / 2.54))
    gs = fig.add_gridspec(
        1,
        2,
        width_ratios=[1.16, 1.0],
        left=0.10,
        right=0.985,
        bottom=0.212,
        top=0.89,
        wspace=0.22,
    )

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    cbar = _panel_core_speed_vs_blh(ax1, fig, df)
    _panel_blh_boxplot(ax2, df)

    # Pull the full left panel footprint 0.5 cm to the left by shrinking
    # the right extent of ax1 and shifting the colorbar with it.
    shift_left_cm = 0.5
    shift_left_frac = shift_left_cm / 18.0
    ax1_pos = ax1.get_position()
    ax1.set_position([ax1_pos.x0, ax1_pos.y0, ax1_pos.width - shift_left_frac, ax1_pos.height])
    cbar_pos = cbar.ax.get_position()
    cbar.ax.set_position([cbar_pos.x0 - shift_left_frac, cbar_pos.y0, cbar_pos.width, cbar_pos.height])

    fig.savefig(output_path, dpi=300)
    fig.savefig(output_path.with_suffix(".svg"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a two-panel LLJ-ABLH summary figure.")
    parser.add_argument(
        "--input-csv",
        default="./llj_analysis_results/llj_analysis_merged.csv",
        help="Merged LLJ analysis CSV.",
    )
    parser.add_argument(
        "--output",
        default="./llj_analysis_results/llj_final_conclusion_figure.png",
        help="Output PNG path.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    required = {"llj_flag", "ablh_m", "llj_core_speed_mps", "condition_label"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(f"Input CSV is missing required columns: {missing}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    build_figure(df, output_path)
    print(f"Saved figure to: {output_path}")


if __name__ == "__main__":
    main()
