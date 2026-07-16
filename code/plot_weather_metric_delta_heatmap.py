#!/usr/bin/env python3
"""Plot a single heatmap for multi-metric deltas across five configurations and three weather classes."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
OUT_DIR = PROJECT_DIR / "pbl_results_fixed_v3"
MPL_DIR = OUT_DIR / ".mplconfig"
MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SUMMARY_CSV = OUT_DIR / "summary_weather_metrics.csv"
DELTA_CSV = OUT_DIR / "summary_weather_metric_deltas.csv"
PNG_OUT = OUT_DIR / "summary_weather_metric_deltas_heatmap.png"
SVG_OUT = OUT_DIR / "summary_weather_metric_deltas_heatmap.svg"

CONFIG_ORDER = ["No AERI", "No Microwave", "No Wind", "No Ceilometer"]
WEATHER_ORDER = ["Clear", "Cloudy", "Fog-Mist"]
METRIC_COLS = ["delta_rmse", "delta_mae", "delta_bias", "delta_r2"]
METRIC_LABELS = ["ΔRMSE (m)", "ΔMAE (m)", "ΔBias (m)", "ΔR²"]


def load_and_prepare() -> pd.DataFrame:
    df = pd.read_csv(SUMMARY_CSV)
    if "condition" not in df.columns or "weather" not in df.columns:
        raise ValueError(f"Unexpected summary schema in {SUMMARY_CSV}")

    full = df[df["condition"] == "Full"].set_index("weather")
    rows = []
    for cond in CONFIG_ORDER:
        sub = df[df["condition"] == cond]
        for weather in WEATHER_ORDER:
            row = sub[sub["weather"] == weather]
            if row.empty:
                continue
            r = row.iloc[0].to_dict()
            base = full.loc[weather]
            r["delta_rmse"] = float(r["rmse"] - base["rmse"])
            r["delta_mae"] = float(r["mae"] - base["mae"])
            r["delta_bias"] = float(r["bias"] - base["bias"])
            r["delta_r2"] = float(r["r2"] - base["r2"])
            r["row_label"] = f"{cond}\n{weather}"
            rows.append(r)

    out = pd.DataFrame(rows)
    out = out[
        [
            "condition",
            "weather",
            "row_label",
            "samples",
            "rmse",
            "mae",
            "bias",
            "r2",
            "mean_actual",
            "mean_pred",
            *METRIC_COLS,
        ]
    ]
    out.to_csv(DELTA_CSV, index=False)
    return out


def build_color_matrix(values: np.ndarray) -> np.ndarray:
    """Scale each metric column independently to preserve visual contrast."""
    scaled = np.zeros_like(values, dtype=float)
    for j in range(values.shape[1]):
        col = values[:, j].astype(float)
        max_abs = np.nanmax(np.abs(col))
        if not np.isfinite(max_abs) or max_abs == 0:
            scaled[:, j] = 0.0
        else:
            scaled[:, j] = col / max_abs
    return scaled


def main() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 170,
            "font.size": 10.5,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "font.family": "DejaVu Sans",
        }
    )

    df = load_and_prepare()
    raw = df[METRIC_COLS].to_numpy(dtype=float)
    color_vals = build_color_matrix(raw)

    fig, ax = plt.subplots(figsize=(9.2, 6.8))
    im = ax.imshow(color_vals, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

    ax.set_xticks(np.arange(len(METRIC_LABELS)))
    ax.set_xticklabels(METRIC_LABELS)
    ax.set_yticks(np.arange(len(df)))
    ax.set_yticklabels(df["row_label"].tolist())
    ax.set_xlabel("")
    ax.set_ylabel("")

    # Cell boundaries.
    ax.set_xticks(np.arange(-0.5, len(METRIC_LABELS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(df), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.15)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Group separators after each configuration block.
    for sep in [2.5, 5.5, 8.5]:
        ax.hlines(sep, -0.5, len(METRIC_LABELS) - 0.5, colors="#7a7a7a", linewidth=1.0)

    # Annotate each cell with the raw delta value.
    for i in range(raw.shape[0]):
        for j in range(raw.shape[1]):
            v = raw[i, j]
            txt = f"{v:+.3f}" if METRIC_COLS[j] == "delta_r2" else f"{v:+.1f}"
            txt_color = "white" if abs(color_vals[i, j]) > 0.55 else "#1f1f1f"
            ax.text(j, i, txt, ha="center", va="center", fontsize=10, fontweight="bold", color=txt_color)

    for spine in ax.spines.values():
        spine.set_visible(False)

    cbar = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.05)
    cbar.set_label("Normalized delta within each metric column")

    fig.tight_layout()
    fig.savefig(PNG_OUT, bbox_inches="tight")
    fig.savefig(SVG_OUT, bbox_inches="tight")
    print(f"Saved: {PNG_OUT}")
    print(f"Saved: {SVG_OUT}")
    print(f"Saved: {DELTA_CSV}")


if __name__ == "__main__":
    main()
