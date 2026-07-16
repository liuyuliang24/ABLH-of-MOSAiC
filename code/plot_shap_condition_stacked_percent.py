#!/usr/bin/env python3
from __future__ import annotations

import os
import re
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
MPL_DIR = SCRIPT_DIR / ".mplconfig"
MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(MPL_DIR)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


TXT_PATH = SCRIPT_DIR / "shap_group_importance.txt"
CSV_PATH = SCRIPT_DIR / "shap_group_importance.csv"
OUT_PATH = SCRIPT_DIR / "shap_condition_stacked_percent.png"

CONDITION_ORDER = ["Overall", "Clear", "Cloudy", "Fog-Mist"]
CONDITION_LABELS = {
    "Overall": "All",
    "Clear": "Clear",
    "Cloudy": "Cloudy",
    "Fog-Mist": "Fog-Mist",
}

FAMILY_COLORS = {
    "Wind": "#f28f89",
    "AERI": "#a8c95a",
    "MW-G/Win": "#23c0d3",
    "MW-K/V": "#87d6dc",
    "Cloud-Base/Status": "#c7926b",
    "Ceilometer": "#b784ef",
    "Time": "#d88ad8",
    "Other": "#b7b7b7",
}

LEGEND_LABELS = {
    "Wind": "Wind",
    "AERI": "AERI",
    "MW-G/Win": "MW-G/Win",
    "MW-K/V": "MW-K/V",
    "Cloud-Base/Status": "Cloud",
    "Ceilometer": "Ceilo",
    "Time": "Time",
    "Other": "Other",
}

PLOT_ORDER = [
    "Wind",
    "AERI",
    "MW-G/Win",
    "MW-K/V",
    "Cloud-Base/Status",
    "Ceilometer",
    "Other",
    "Time",
]


def load_detailed_dataframe() -> pd.DataFrame:
    if CSV_PATH.exists():
        df = pd.read_csv(CSV_PATH)
        return df

    pattern_header = re.compile(r"^\[(?P<condition>[^\]]+)\]\s+samples=(?P<samples>\d+)\s*$")
    pattern_row = re.compile(
        r"^(?P<group>[a-zA-Z0-9_]+)\s+\((?P<source>[^)]+)\):\s+importance=(?P<importance>[0-9.]+),\s+"
        r"ratio=(?P<ratio>[0-9.]+)%,\s+features=(?P<features>\d+)\s*$"
    )

    rows: list[dict[str, object]] = []
    condition = None
    sample_count = None
    with TXT_PATH.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            m_header = pattern_header.match(line)
            if m_header:
                condition = m_header.group("condition")
                sample_count = int(m_header.group("samples"))
                continue
            m_row = pattern_row.match(line)
            if m_row and condition is not None and sample_count is not None:
                rows.append(
                    {
                        "condition": condition,
                        "group": m_row.group("group"),
                        "source_family": m_row.group("source"),
                        "importance": float(m_row.group("importance")),
                        "importance_ratio_percent": float(m_row.group("ratio")),
                        "feature_count": int(m_row.group("features")),
                        "sample_count": sample_count,
                    }
                )

    if not rows:
        raise RuntimeError(f"Failed to parse SHAP text file: {TXT_PATH}")
    return pd.DataFrame(rows)


def normalize_conditions(series: pd.Series) -> pd.Series:
    return series.replace(
        {
            "All": "Overall",
            "Fog/Mist": "Fog-Mist",
            "Foggy-Mist": "Fog-Mist",
        }
    )


def map_source_family(source_family: str) -> str:
    mapping = {
        "Wind": "Wind",
        "AERI": "AERI",
        "MiRAC-P": "MW-G/Win",
        "HATPRO": "MW-K/V",
        "CloudBase/Status": "Cloud-Base/Status",
        "Ceilometer": "Ceilometer",
        "Time": "Time",
        "Other": "Other",
    }
    return mapping.get(source_family, source_family)


def load_family_dataframe() -> pd.DataFrame:
    df = load_detailed_dataframe()
    df["condition"] = normalize_conditions(df["condition"])
    df["family"] = df["source_family"].map(map_source_family)
    grouped = df.groupby(["condition", "family"], as_index=False)["importance_ratio_percent"].sum()
    return grouped


def get_family_order(df: pd.DataFrame) -> list[str]:
    ordered = [family for family in PLOT_ORDER if family in df["family"].unique()]
    for family in df["family"].tolist():
        if family not in ordered:
            ordered.append(family)
    return ordered


def reorder_for_row_major(items: list, ncol: int) -> list:
    n_items = len(items)
    nrow = (n_items + ncol - 1) // ncol
    reordered = []
    for col in range(ncol):
        for row in range(nrow):
            idx = row * ncol + col
            if idx < n_items:
                reordered.append(items[idx])
    return reordered


def plot_stacked_percent(df: pd.DataFrame) -> None:
    plot_df = df[df["condition"].isin(CONDITION_ORDER)].copy()
    plot_df = plot_df.groupby(["condition", "family"], as_index=False)["importance_ratio_percent"].sum()
    family_order = get_family_order(plot_df)
    family_order = [family for family in family_order if family in plot_df["family"].unique()]

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 8
    plt.rcParams["axes.labelsize"] = 8
    plt.rcParams["axes.titlesize"] = 8
    plt.rcParams["legend.fontsize"] = 7
    plt.rcParams["xtick.labelsize"] = 7.5
    plt.rcParams["ytick.labelsize"] = 8

    fig_width_in = 9.0 / 2.54
    fig_height_in = (8.0 * 2.0 / 3.0) / 2.54
    fig, ax = plt.subplots(figsize=(fig_width_in, fig_height_in))
    bar_height = 0.155
    bar_gap = 0.17
    y_step = bar_height + bar_gap
    y_positions = [idx * y_step for idx in range(len(CONDITION_ORDER))]
    y_map = {condition: y for condition, y in zip(CONDITION_ORDER, y_positions)}

    for condition in CONDITION_ORDER:
        cond_df = plot_df[plot_df["condition"] == condition].copy()
        cond_df["family_order"] = cond_df["family"].map({name: idx for idx, name in enumerate(family_order)})
        cond_df = cond_df.sort_values("family_order")
        left = 0.0
        for _, row in cond_df.iterrows():
            width = float(row["importance_ratio_percent"])
            if width <= 0:
                continue
            ax.barh(
                y_map[condition],
                width,
                left=left,
                height=bar_height,
                color=FAMILY_COLORS.get(row["family"], "#999999"),
                edgecolor="none",
                linewidth=0.0,
            )
            left += width

    ax.set_xlim(0, 100)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_xlabel("Importance percentage (%)")
    ax.set_yticks([y_map[c] for c in CONDITION_ORDER])
    ax.set_yticklabels([CONDITION_LABELS[c] for c in CONDITION_ORDER])
    ax.set_ylim(-0.12, y_positions[-1] + 0.12)
    ax.invert_yaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend_families = [family for family in family_order if family in plot_df["family"].unique()]
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=FAMILY_COLORS.get(family, "#999999"), edgecolor="none")
        for family in legend_families
    ]
    labels = [LEGEND_LABELS.get(family, family) for family in legend_families]
    legend_ncol = 4
    handles = reorder_for_row_major(handles, legend_ncol)
    labels = reorder_for_row_major(labels, legend_ncol)
    ax.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.24),
        ncol=legend_ncol,
        frameon=False,
        columnspacing=0.9,
        handlelength=1.2,
        handletextpad=0.4,
    )

    fig.subplots_adjust(left=0.23, right=0.98, top=0.97, bottom=0.36)
    fig.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    df = load_family_dataframe()
    plot_stacked_percent(df)
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
