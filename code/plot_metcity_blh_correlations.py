#!/usr/bin/env python3
"""
Create a manuscript-style figure for Met City controls on ABLH.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TEXT_COLOR = "#222222"
LINE_COLOR = "#111111"
MEAN_COLOR = "#D95F02"
SCATTER_COLOR = "#4C78A8"
BAR_COLOR = "#6F93B7"

PANELS = [
    {
        "column": "bulk_tau",
        "title": "(a)",
        "xlabel": "Wind stress (Pa)",
        "logx": True,
        "xlim": (1e-2, 0.45),
        "subtitle": "Strongest linear relation",
    },
    {
        "column": "ustar_10m",
        "title": "(b)",
        "xlabel": "Friction velocity (m s$^{-1}$)",
        "logx": False,
        "xlim": (0.0, 0.75),
        "subtitle": "Mechanical mixing strength",
    },
    {
        "column": "epsilon_10m",
        "title": "(c)",
        "xlabel": "$\\epsilon$ (m$^2$ s$^{-3}$)",
        "logx": True,
        "xlim": (1e-6, 0.15),
        "subtitle": "Nonlinear turbulent control",
    },
    {
        "column": "down_long_hemisp",
        "title": "(d)",
        "xlabel": "Downwelling longwave (W m$^{-2}$)",
        "logx": False,
        "xlim": (120, 320),
        "subtitle": "Radiative influence",
    },
    {
        "column": "surface_air_temp_diff_c",
        "title": "(e)",
        "xlabel": "$T_s - T_{2m}$ (°C)",
        "logx": False,
        "xlim": (-3.2, 1.0),
        "subtitle": "Surface thermal forcing",
    },
    {
        "column": "Hl",
        "title": "(f)",
        "xlabel": "Latent heat flux (W m$^{-2}$)",
        "logx": False,
        "xlim": (-4.0, 11.0),
        "subtitle": "Moisture-related surface exchange",
    },
]


def _set_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 180,
            "savefig.dpi": 300,
            "font.size": 8,
            "font.family": "DejaVu Sans",
            "axes.linewidth": 0.8,
            "axes.edgecolor": "#333333",
            "axes.labelcolor": TEXT_COLOR,
            "axes.labelsize": 8,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "axes.grid": True,
            "grid.color": "#C8C8C8",
            "grid.alpha": 0.30,
            "grid.linewidth": 0.45,
            "legend.fontsize": 7.2,
        }
    )


def _safe_corr(x: pd.Series, y: pd.Series, method: str) -> float:
    pair = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(pair) < 20:
        return np.nan
    return float(pair["x"].corr(pair["y"], method=method))


def _despine(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=3, width=0.7, colors=TEXT_COLOR)


def _plot_relationship(ax: plt.Axes, df: pd.DataFrame, cfg: dict[str, object], *, show_ylabel: bool) -> None:
    column = cfg["column"]
    full_pair = df[[column, "ablh_m"]].dropna().copy()
    pair = full_pair.copy()

    if "clip_quantiles" in cfg:
        q_low, q_high = cfg["clip_quantiles"]
        low = pair[column].quantile(q_low)
        high = pair[column].quantile(q_high)
        pair = pair.loc[pair[column].between(low, high)]

    pair = pair.sort_values(column)
    x = pair[column].to_numpy(dtype=float)
    y = pair["ablh_m"].to_numpy(dtype=float)

    if cfg.get("logx"):
        valid = x > 0
        x = x[valid]
        y = y[valid]
        pair = pair.loc[valid]

    rng = np.random.default_rng(42)
    sample_n = min(5000, len(pair))
    if len(pair) > sample_n:
        idx = rng.choice(len(pair), sample_n, replace=False)
        xs = x[idx]
        ys = y[idx]
    else:
        xs = x
        ys = y

    ax.scatter(
        xs,
        ys,
        s=5.0,
        facecolors=SCATTER_COLOR,
        alpha=0.10,
        edgecolors="#ffffff",
        linewidths=0.12,
        rasterized=True,
    )

    if cfg.get("fit_only"):
        if len(pair) >= 20:
            xfit = pair[column].to_numpy(dtype=float)
            yfit = pair["ablh_m"].to_numpy(dtype=float)
            coef = np.polyfit(xfit, yfit, 1)
            xx = np.linspace(cfg["xlim"][0], cfg["xlim"][1], 120)
            yy = coef[0] * xx + coef[1]
            ax.plot(xx, yy, color=MEAN_COLOR, linewidth=1.8)
    else:
        bins = np.geomspace(cfg["xlim"][0], cfg["xlim"][1], 13) if cfg.get("logx") else np.linspace(cfg["xlim"][0], cfg["xlim"][1], 13)
        pair["bin"] = pd.cut(pair[column], bins=bins, include_lowest=True)
        grouped = pair.groupby("bin", observed=False)["ablh_m"]
        centers, means, p25, p75 = [], [], [], []
        for interval, values in grouped:
            vals = values.dropna()
            if len(vals) < 40:
                continue
            center = np.sqrt(max(interval.left, 1e-12) * interval.right) if cfg.get("logx") else interval.mid
            centers.append(center)
            means.append(vals.mean())
            p25.append(vals.quantile(0.25))
            p75.append(vals.quantile(0.75))

        if centers:
            ax.plot(centers, means, color=MEAN_COLOR, linewidth=1.6, marker="o", markersize=3.3)
            ax.fill_between(centers, p25, p75, color=MEAN_COLOR, alpha=0.18, linewidth=0)

    pearson_r = _safe_corr(full_pair[column], full_pair["ablh_m"], "pearson")
    pearson_r = _safe_corr(full_pair[column], full_pair["ablh_m"], "pearson")
    spearman_r = _safe_corr(full_pair[column], full_pair["ablh_m"], "spearman")

    ax.text(
        0.02,
        1.03,
        str(cfg["title"]),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.0,
        fontweight="bold",
        color=TEXT_COLOR,
    )
    ax.text(
        0.02,
        0.97,
        f"{cfg['subtitle']}\nPearson r = {pearson_r:.2f}; Spearman r = {spearman_r:.2f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.4,
        color=TEXT_COLOR,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=1.0),
    )

    ax.set_xlabel(cfg["xlabel"])
    ax.set_ylabel("ABLH (m)" if show_ylabel else "")
    ax.set_ylim(0, 780)
    ax.set_xlim(*cfg["xlim"])
    if cfg.get("logx"):
        ax.set_xscale("log")
    _despine(ax)


def build_figure(merged_csv: Path, output: Path) -> None:
    _set_style()
    df = pd.read_csv(merged_csv)

    fig = plt.figure(figsize=(8.2, 5.4))
    gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.28)

    axes = [fig.add_subplot(gs[i // 3, i % 3]) for i in range(6)]
    for idx, (ax, cfg) in enumerate(zip(axes, PANELS)):
        _plot_relationship(ax, df, cfg, show_ylabel=(idx % 3 == 0))

    fig.savefig(output, bbox_inches="tight", dpi=300)
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Met City controls on ABLH.")
    parser.add_argument(
        "--merged-csv",
        default="metcity_blh_correlation_results/metcity_blh_merged.csv",
    )
    parser.add_argument(
        "--summary-csv",
        default="metcity_blh_correlation_results/metcity_blh_correlation_summary.csv",
    )
    parser.add_argument(
        "--output",
        default="metcity_blh_correlation_results/metcity_blh_correlation_figure.png",
    )
    args = parser.parse_args()

    build_figure(Path(args.merged_csv), Path(args.output))
    print(f"Saved figure to: {args.output}")


if __name__ == "__main__":
    main()
