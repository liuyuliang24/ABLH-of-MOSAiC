#!/usr/bin/env python3
"""
Classify ABL type from sounding-derived equivalent potential temperature
difference and summarize its distribution under different weather conditions.

Rules:
- theta_e(100 m) - theta_e(50 m) > 0.2 K: SBL
- theta_e(100 m) - theta_e(50 m) < -0.2 K: CBL
- otherwise: NBL

Cloudy cases are further refined by cloud-BLH coupling state:
- Coupled
- CBH above BLH
- BLH above CBH
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
OUT_DIR = SCRIPT_DIR
MPL_DIR = OUT_DIR / ".mplconfig"
MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_DIR))

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from plot_font_utils import get_available_serif_font


FEATURES_CSV = Path("/media/lyl/DATA11/THz_band/THz_data/MOSAiC/ppp/matched_all_data_10min_with_aeri-1.csv")
LABELS_CSV = Path("/media/lyl/DATA11/THz_band/THz_data/MOSAiC/ppp/mosaic_refined_ablh_aeri-1.csv")

HEIGHT_START_M = 10.0
HEIGHT_STEP_M = 20.0
COUPLED_THRESHOLD_M = 100.0

ABL_ORDER = ["SBL", "NBL", "CBL"]
ABL_COLORS = {
    "SBL": "#006BFF",
    "NBL": "#FFB000",
    "CBL": "#FF2D2D",
}
WEATHER_ORDER = ["Clear", "Cloudy", "Fog/Mist"]
WEATHER_COLORS = {
    "Clear": "#355070",
    "Cloudy": "#c1121f",
    "Fog/Mist": "#2a9d8f",
}
REFINED_CONDITION_ORDER = [
    "Clear",
    "Cloud coupled",
    "CBH above BLH",
    "CBH below BLH",
    "Cloud no valid CBH",
    "Fog/Mist",
]
REFINED_CONDITION_SHORT_LABELS = {
    "Clear": "CL",
    "Fog/Mist": "Fog",
    "Cloud coupled": "CC",
    "CBH above BLH": "CAB",
    "CBH below BLH": "CBB",
    "Cloud no valid CBH": "NC",
}
SEASON_ORDER = ["Fall", "Winter", "Spring", "Summer"]
SEASON_DISPLAY_LABELS = {
    "Fall": "Autumn",
    "Winter": "Winter",
    "Spring": "Spring",
    "Summer": "Summer",
}
COUPLING_ORDER = ["Coupled", "CBH above BLH", "BLH above CBH"]
COUPLING_COLORS = {
    "Coupled": "#2a9d8f",
    "CBH above BLH": "#e76f51",
    "BLH above CBH": "#4361ee",
}


def _set_style() -> None:
    serif_font = get_available_serif_font()
    plt.rcParams["figure.dpi"] = 140
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.22
    plt.rcParams["font.family"] = serif_font
    plt.rcParams["font.size"] = 10


def _sorted_profile_columns(columns: list[str], prefix: str) -> list[str]:
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)$")
    matched = []
    for col in columns:
        m = pattern.match(col)
        if m:
            matched.append((int(m.group(1)), col))
    matched.sort(key=lambda item: item[0])
    return [col for _, col in matched]


def build_height_grid(n_levels: int, start_m: float = HEIGHT_START_M, step_m: float = HEIGHT_STEP_M) -> np.ndarray:
    return start_m + np.arange(n_levels, dtype=float) * step_m


def _group_weather(status: float) -> str:
    if not np.isfinite(status):
        return "Unknown"
    status = int(status)
    if status == 0:
        return "Clear"
    if 1 <= status <= 3:
        return "Cloudy"
    return "Fog/Mist"


def _month_to_season(month: int) -> str:
    if month in (12, 1, 2):
        return "Winter"
    if month in (3, 4, 5):
        return "Spring"
    if month in (6, 7, 8):
        return "Summer"
    return "Fall"


def load_dataset() -> pd.DataFrame:
    header = pd.read_csv(FEATURES_CSV, nrows=0)
    temp_cols = _sorted_profile_columns(header.columns.tolist(), "temp_")
    sh_cols = _sorted_profile_columns(header.columns.tolist(), "sh_")
    pres_cols = _sorted_profile_columns(header.columns.tolist(), "bar_pres_")
    n_levels = min(len(temp_cols), len(sh_cols), len(pres_cols))
    if n_levels == 0:
        raise RuntimeError("No temperature/specific-humidity/pressure profile columns found.")

    usecols = ["target_10min", "detection_status", "first_cbh"]
    usecols += temp_cols[:n_levels] + sh_cols[:n_levels] + pres_cols[:n_levels]
    feat = pd.read_csv(FEATURES_CSV, usecols=usecols)
    lab = pd.read_csv(LABELS_CSV, usecols=["time", "ablh_m"])

    feat["time"] = pd.to_datetime(feat["target_10min"], errors="coerce")
    lab["time"] = pd.to_datetime(lab["time"], errors="coerce")

    merged = feat.merge(lab, on="time", how="inner")
    merged["first_cbh"] = pd.to_numeric(merged["first_cbh"], errors="coerce")
    merged.loc[(merged["first_cbh"] <= 0.0) | (merged["first_cbh"] > 10000.0), "first_cbh"] = np.nan
    merged["weather_condition"] = merged["detection_status"].apply(_group_weather)
    merged["month"] = merged["time"].dt.month
    merged["season"] = merged["month"].apply(_month_to_season)
    merged["cbh_valid"] = merged["first_cbh"].notna()
    merged["cbh_to_blh_delta_m"] = merged["first_cbh"] - merged["ablh_m"]

    merged = add_thetae_abl_type(merged, temp_cols[:n_levels], sh_cols[:n_levels], pres_cols[:n_levels])
    merged = classify_coupling(merged)
    merged = add_refined_condition(merged)
    return merged


def specific_humidity_to_vapor_pressure(q: np.ndarray, p_pa: np.ndarray) -> np.ndarray:
    eps = 0.622
    return q * p_pa / (eps + (1.0 - eps) * q)


def dewpoint_from_vapor_pressure(e_pa: np.ndarray) -> np.ndarray:
    e_hpa = e_pa / 100.0
    safe = np.where(e_hpa > 0.0, e_hpa, np.nan)
    ln_ratio = np.log(safe / 6.112)
    td_c = 243.5 * ln_ratio / (17.67 - ln_ratio)
    return td_c + 273.15


def equivalent_potential_temperature(temp_k: np.ndarray, p_pa: np.ndarray, q: np.ndarray) -> np.ndarray:
    p_hpa = p_pa / 100.0
    mixing_ratio = q / np.clip(1.0 - q, 1e-8, None)
    e_pa = specific_humidity_to_vapor_pressure(q, p_pa)
    td_k = dewpoint_from_vapor_pressure(e_pa)
    valid = np.isfinite(temp_k) & np.isfinite(p_hpa) & np.isfinite(q) & np.isfinite(td_k) & (p_hpa > 0.0) & (q >= 0.0)

    theta_e = np.full_like(temp_k, np.nan, dtype=float)
    if not np.any(valid):
        return theta_e

    t = temp_k[valid]
    p = p_hpa[valid]
    r = mixing_ratio[valid]
    td = td_k[valid]
    tl = 1.0 / (1.0 / (td - 56.0) + np.log(np.clip(t / td, 1e-8, None)) / 800.0) + 56.0
    exponent = (3.376 / np.clip(tl, 1e-8, None) - 0.00254) * r * 1000.0 * (1.0 + 0.81 * r)
    theta_e[valid] = t * np.power(1000.0 / p, 0.2854 * (1.0 - 0.28 * r)) * np.exp(exponent)
    return theta_e


def interpolate_profile(profile: np.ndarray, heights: np.ndarray, target_height: float) -> np.ndarray:
    out = np.full(profile.shape[0], np.nan, dtype=float)
    for i in range(profile.shape[0]):
        row = profile[i]
        valid = np.isfinite(row) & np.isfinite(heights)
        if valid.sum() < 2:
            continue
        x = heights[valid]
        y = row[valid]
        if target_height < x.min() or target_height > x.max():
            continue
        out[i] = np.interp(target_height, x, y)
    return out


def add_thetae_abl_type(df: pd.DataFrame, temp_cols: list[str], sh_cols: list[str], pres_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    heights = build_height_grid(len(temp_cols))

    temp_c = out[temp_cols].to_numpy(dtype=float)
    sh = out[sh_cols].to_numpy(dtype=float)
    pres = out[pres_cols].to_numpy(dtype=float)

    temp_k = temp_c + 273.15
    pres[(pres <= 1000.0) | (pres >= 110000.0)] = np.nan
    sh[(sh < 0.0) | (sh >= 0.1)] = np.nan

    theta_e = equivalent_potential_temperature(temp_k, pres, sh)
    thetae_50m = interpolate_profile(theta_e, heights, 50.0)
    thetae_100m = interpolate_profile(theta_e, heights, 100.0)
    diff = thetae_100m - thetae_50m

    out["thetae_50m_k"] = thetae_50m
    out["thetae_100m_k"] = thetae_100m
    out["thetae_diff_100m_50m_k"] = diff
    out["abl_type"] = np.where(
        diff > 0.2,
        "SBL",
        np.where(diff < -0.2, "CBL", "NBL"),
    )
    out.loc[~np.isfinite(diff), "abl_type"] = np.nan
    return out


def classify_coupling(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    coupling = np.full(len(out), np.nan, dtype=object)
    valid = out["cbh_valid"].to_numpy(dtype=bool)
    delta = out["cbh_to_blh_delta_m"].to_numpy(dtype=float)
    coupling[valid & (np.abs(delta) <= COUPLED_THRESHOLD_M)] = "Coupled"
    coupling[valid & (delta > COUPLED_THRESHOLD_M)] = "CBH above BLH"
    coupling[valid & (delta < -COUPLED_THRESHOLD_M)] = "BLH above CBH"
    out["coupling_category"] = pd.Categorical(coupling, categories=COUPLING_ORDER, ordered=True)
    return out


def add_refined_condition(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    refined = np.full(len(out), "Unknown", dtype=object)
    weather = out["weather_condition"].astype(str).to_numpy()
    cbh_valid = out["cbh_valid"].to_numpy(dtype=bool)
    delta = out["cbh_to_blh_delta_m"].to_numpy(dtype=float)

    refined[weather == "Clear"] = "Clear"
    refined[weather == "Fog/Mist"] = "Fog/Mist"

    cloudy = weather == "Cloudy"
    refined[cloudy & ~cbh_valid] = "Cloud no valid CBH"
    refined[cloudy & cbh_valid & (np.abs(delta) <= COUPLED_THRESHOLD_M)] = "Cloud coupled"
    refined[cloudy & cbh_valid & (delta > COUPLED_THRESHOLD_M)] = "CBH above BLH"
    refined[cloudy & cbh_valid & (delta < -COUPLED_THRESHOLD_M)] = "CBH below BLH"

    out["refined_condition"] = pd.Categorical(refined, categories=REFINED_CONDITION_ORDER, ordered=True)
    return out


def summarize_weather(df: pd.DataFrame) -> pd.DataFrame:
    subset = df.loc[df["weather_condition"].isin(WEATHER_ORDER) & df["abl_type"].isin(ABL_ORDER)].copy()
    count = (
        subset.groupby(["weather_condition", "abl_type"], observed=False)
        .size()
        .rename("n_samples")
        .reset_index()
    )
    total = count.groupby("weather_condition", observed=False)["n_samples"].transform("sum")
    count["fraction_percent"] = count["n_samples"] / total * 100.0
    return count


def summarize_refined_condition(df: pd.DataFrame) -> pd.DataFrame:
    subset = df.loc[df["refined_condition"].isin(REFINED_CONDITION_ORDER) & df["abl_type"].isin(ABL_ORDER)].copy()
    count = (
        subset.groupby(["refined_condition", "abl_type"], observed=False)
        .size()
        .rename("n_samples")
        .reset_index()
    )
    total = count.groupby("refined_condition", observed=False)["n_samples"].transform("sum")
    count["fraction_percent"] = np.where(total > 0, count["n_samples"] / total * 100.0, 0.0)
    return count


def summarize_cloudy_refined(df: pd.DataFrame) -> pd.DataFrame:
    subset = df.loc[
        (df["weather_condition"] == "Cloudy")
        & df["abl_type"].isin(ABL_ORDER)
        & df["coupling_category"].isin(COUPLING_ORDER)
    ].copy()
    count = (
        subset.groupby(["coupling_category", "abl_type"], observed=False)
        .size()
        .rename("n_samples")
        .reset_index()
    )
    total = count.groupby("coupling_category", observed=False)["n_samples"].transform("sum")
    count["fraction_percent"] = count["n_samples"] / total * 100.0
    return count


def summarize_weather_abl_grid(df: pd.DataFrame) -> pd.DataFrame:
    subset = df.loc[
        df["weather_condition"].isin(WEATHER_ORDER)
        & df["abl_type"].isin(ABL_ORDER)
        & df["season"].isin(SEASON_ORDER)
    ].copy()
    total_valid = len(subset)

    rows = []
    category_id = 1
    for weather in WEATHER_ORDER:
        for abl_type in ABL_ORDER:
            cell = subset.loc[(subset["weather_condition"] == weather) & (subset["abl_type"] == abl_type)]
            cell_total = len(cell)
            annual_percent = cell_total / max(total_valid, 1) * 100.0
            for season in SEASON_ORDER:
                season_n = int((cell["season"] == season).sum())
                rows.append(
                    {
                        "category_id": category_id,
                        "weather_condition": weather,
                        "abl_type": abl_type,
                        "season": season,
                        "total_n": int(cell_total),
                        "season_n": season_n,
                        "annual_percent_of_all_valid": annual_percent,
                        "season_percent_within_category": season_n / max(cell_total, 1) * 100.0,
                    }
                )
            category_id += 1
    return pd.DataFrame(rows)


def summarize_refined_condition_abl_grid(df: pd.DataFrame) -> pd.DataFrame:
    subset = df.loc[
        df["refined_condition"].isin(REFINED_CONDITION_ORDER)
        & df["abl_type"].isin(ABL_ORDER)
        & df["season"].isin(SEASON_ORDER)
    ].copy()
    total_valid = len(subset)

    rows = []
    category_id = 1
    for condition in REFINED_CONDITION_ORDER:
        for abl_type in ABL_ORDER:
            cell = subset.loc[(subset["refined_condition"] == condition) & (subset["abl_type"] == abl_type)]
            cell_total = len(cell)
            annual_percent = cell_total / max(total_valid, 1) * 100.0
            for season in SEASON_ORDER:
                season_n = int((cell["season"] == season).sum())
                rows.append(
                    {
                        "category_id": category_id,
                        "refined_condition": condition,
                        "abl_type": abl_type,
                        "season": season,
                        "total_n": int(cell_total),
                        "season_n": season_n,
                        "annual_percent_of_all_valid": annual_percent,
                        "season_percent_within_category": season_n / max(cell_total, 1) * 100.0,
                    }
                )
            category_id += 1
    return pd.DataFrame(rows)


def write_overview(df: pd.DataFrame, output_path: Path) -> None:
    total = len(df)
    valid_type = df["abl_type"].notna().sum()
    cloudy = (df["weather_condition"] == "Cloudy").sum()
    cloudy_valid_cbh = ((df["weather_condition"] == "Cloudy") & df["cbh_valid"]).sum()

    lines = [
        f"Total merged samples: {total}",
        f"Samples with valid thetaE-difference ABL type: {valid_type} ({valid_type / max(total, 1) * 100.0:.1f}%)",
        f"Cloudy samples: {cloudy}",
        f"Cloudy samples with valid CBH: {cloudy_valid_cbh} ({cloudy_valid_cbh / max(cloudy, 1) * 100.0:.1f}%)",
        "",
        "ABL type rule:",
        "SBL: thetaE(100 m) - thetaE(50 m) > 0.2 K",
        "CBL: thetaE(100 m) - thetaE(50 m) < -0.2 K",
        "NBL: otherwise",
        "",
        "Grid figure definition:",
        "Rows are weather conditions and columns are ABL types.",
        "Annual panel: percent of all valid samples in each weather-condition/ABL-type category.",
        "Seasonal panels: for all samples in a given category, the percent that occurred in each season.",
        "",
        "Refined cloudy-condition definition:",
        f"Cloud coupled: abs(CBH - BLH) <= {COUPLED_THRESHOLD_M:.0f} m",
        f"CBH above BLH: CBH - BLH > {COUPLED_THRESHOLD_M:.0f} m",
        f"CBH below BLH: CBH - BLH < -{COUPLED_THRESHOLD_M:.0f} m",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_weather_distribution(summary_df: pd.DataFrame, output_path: Path) -> None:
    _set_style()
    pivot = (
        summary_df.pivot(index="weather_condition", columns="abl_type", values="fraction_percent")
        .reindex(index=WEATHER_ORDER, columns=ABL_ORDER)
        .fillna(0.0)
    )

    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    bottom = np.zeros(len(pivot), dtype=float)
    x = np.arange(len(pivot))
    for abl in ABL_ORDER:
        values = pivot[abl].to_numpy(dtype=float)
        ax.bar(x, values, bottom=bottom, color=ABL_COLORS[abl], width=0.65, label=abl)
        for i, (b, v) in enumerate(zip(bottom, values)):
            if v >= 5.0:
                ax.text(i, b + v / 2.0, f"{v:.1f}%", ha="center", va="center", color="white", fontsize=9)
        bottom += values

    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index.tolist())
    ax.set_ylabel("Fraction (%)")
    ax.set_ylim(0, 100)
    ax.set_title("ABL Type Distribution Under Different Weather Conditions")
    ax.legend(loc="upper right", ncol=3)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_weather_counts(summary_df: pd.DataFrame, output_path: Path) -> None:
    _set_style()
    pivot = (
        summary_df.pivot(index="weather_condition", columns="abl_type", values="n_samples")
        .reindex(index=WEATHER_ORDER, columns=ABL_ORDER)
        .fillna(0.0)
    )

    fig, ax = plt.subplots(figsize=(8.8, 5.8))
    x = np.arange(len(pivot))
    width = 0.22
    offsets = [-width, 0.0, width]
    for offset, abl in zip(offsets, ABL_ORDER):
        values = pivot[abl].to_numpy(dtype=float)
        bars = ax.bar(x + offset, values, width=width, color=ABL_COLORS[abl], label=abl)
        for bar, value in zip(bars, values):
            if value > 0:
                ax.text(bar.get_x() + bar.get_width() / 2.0, value, f"{int(value)}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index.tolist())
    ax.set_ylabel("Sample count")
    ax.set_title("ABL Type Counts Under Different Weather Conditions")
    ax.legend(loc="upper right", ncol=3)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_refined_condition_distribution(summary_df: pd.DataFrame, output_path: Path) -> None:
    _set_style()
    pivot = (
        summary_df.pivot(index="refined_condition", columns="abl_type", values="fraction_percent")
        .reindex(index=REFINED_CONDITION_ORDER, columns=ABL_ORDER)
        .fillna(0.0)
    )
    pivot = pivot.loc[pivot.sum(axis=1) > 0]

    fig, ax = plt.subplots(figsize=(10.2, 6.2))
    bottom = np.zeros(len(pivot), dtype=float)
    x = np.arange(len(pivot))
    for abl in ABL_ORDER:
        values = pivot[abl].to_numpy(dtype=float)
        ax.bar(x, values, bottom=bottom, color=ABL_COLORS[abl], width=0.66, label=abl)
        for i, (b, v) in enumerate(zip(bottom, values)):
            if v >= 5.0:
                color = "white" if abl in ("SBL", "CBL") else "#1f1f1f"
                ax.text(i, b + v / 2.0, f"{v:.1f}%", ha="center", va="center", color=color, fontsize=8)
        bottom += values

    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index.tolist(), rotation=18, ha="right")
    ax.set_ylabel("Fraction within refined condition (%)")
    ax.set_ylim(0, 100)
    ax.set_title("ABL Type Distribution Under Refined Weather/Cloud Conditions")
    ax.legend(loc="upper right", ncol=3)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_cloudy_refined(summary_df: pd.DataFrame, output_path: Path) -> None:
    _set_style()
    pivot = (
        summary_df.pivot(index="coupling_category", columns="abl_type", values="fraction_percent")
        .reindex(index=COUPLING_ORDER, columns=ABL_ORDER)
        .fillna(0.0)
    )

    fig, ax = plt.subplots(figsize=(9.0, 5.8))
    bottom = np.zeros(len(pivot), dtype=float)
    x = np.arange(len(pivot))
    for abl in ABL_ORDER:
        values = pivot[abl].to_numpy(dtype=float)
        ax.bar(x, values, bottom=bottom, color=ABL_COLORS[abl], width=0.65, label=abl)
        for i, (b, v) in enumerate(zip(bottom, values)):
            if v >= 6.0:
                ax.text(i, b + v / 2.0, f"{v:.1f}%", ha="center", va="center", color="white", fontsize=9)
        bottom += values

    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index.tolist(), rotation=10)
    ax.set_ylabel("Fraction (%)")
    ax.set_ylim(0, 100)
    ax.set_title("ABL Type Distribution Within Cloudy Cases, Refined by Coupling State")
    ax.legend(loc="upper right", ncol=3)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_cloudy_refined_counts(summary_df: pd.DataFrame, output_path: Path) -> None:
    _set_style()
    pivot = (
        summary_df.pivot(index="coupling_category", columns="abl_type", values="n_samples")
        .reindex(index=COUPLING_ORDER, columns=ABL_ORDER)
        .fillna(0.0)
    )

    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    x = np.arange(len(pivot))
    width = 0.22
    offsets = [-width, 0.0, width]
    for offset, abl in zip(offsets, ABL_ORDER):
        values = pivot[abl].to_numpy(dtype=float)
        bars = ax.bar(x + offset, values, width=width, color=ABL_COLORS[abl], label=abl)
        for bar, value in zip(bars, values):
            if value > 0:
                ax.text(bar.get_x() + bar.get_width() / 2.0, value, f"{int(value)}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index.tolist(), rotation=10)
    ax.set_ylabel("Sample count")
    ax.set_title("ABL Type Counts Within Cloudy Cases, Refined by Coupling State")
    ax.legend(loc="upper right", ncol=3)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _format_percent(value: float) -> str:
    if not np.isfinite(value):
        return "NA"
    if abs(value - round(value)) < 0.05:
        return f"{value:.0f}%"
    return f"{value:.1f}%"


def _grid_value(summary_df: pd.DataFrame, weather: str, abl_type: str, column: str, season: str | None = None) -> float:
    mask = (summary_df["weather_condition"] == weather) & (summary_df["abl_type"] == abl_type)
    if season is not None:
        mask &= summary_df["season"] == season
    vals = summary_df.loc[mask, column]
    if vals.empty:
        return 0.0
    return float(vals.iloc[0])


def _condition_grid_value(
    summary_df: pd.DataFrame,
    condition_column: str,
    condition: str,
    abl_type: str,
    column: str,
    season: str | None = None,
) -> float:
    mask = (summary_df[condition_column] == condition) & (summary_df["abl_type"] == abl_type)
    if season is not None:
        mask &= summary_df["season"] == season
    vals = summary_df.loc[mask, column]
    if vals.empty:
        return 0.0
    return float(vals.iloc[0])


def _draw_weather_abl_grid(
    ax: plt.Axes,
    summary_df: pd.DataFrame,
    *,
    title: str,
    value_column: str,
    season: str | None,
    norm: matplotlib.colors.Normalize,
    cmap: matplotlib.colors.Colormap,
) -> None:
    n_rows = len(WEATHER_ORDER)
    n_cols = len(ABL_ORDER)
    ax.set_xlim(0, n_cols)
    ax.set_ylim(n_rows, 0)
    ax.set_aspect("auto")
    ax.set_title(title, fontsize=12, pad=4)
    ax.set_xticks(np.arange(n_cols) + 0.5)
    ax.set_xticklabels(ABL_ORDER, fontsize=8)
    ax.set_yticks(np.arange(n_rows) + 0.5)
    ax.set_yticklabels(WEATHER_ORDER, fontsize=8)
    ax.tick_params(length=0)

    for spine in ax.spines.values():
        spine.set_visible(False)

    category_id = 1
    for row, weather in enumerate(WEATHER_ORDER):
        for col, abl_type in enumerate(ABL_ORDER):
            value = _grid_value(summary_df, weather, abl_type, value_column, season)
            facecolor = cmap(norm(value))
            edgecolor = ABL_COLORS[abl_type]
            rect = plt.Rectangle((col, row), 1, 1, facecolor=facecolor, edgecolor=edgecolor, linewidth=2.0)
            ax.add_patch(rect)

            text_color = "white" if norm(value) > 0.62 else "#1f1f1f"
            ax.text(col + 0.04, row + 0.14, str(category_id), ha="left", va="center", fontsize=7, color=text_color)
            ax.text(col + 0.96, row + 0.14, abl_type, ha="right", va="center", fontsize=7, color=text_color)
            ax.text(col + 0.50, row + 0.43, weather, ha="center", va="center", fontsize=7, color=text_color)
            ax.text(
                col + 0.50,
                row + 0.72,
                _format_percent(value),
                ha="center",
                va="center",
                fontsize=13,
                color=text_color,
                bbox=dict(facecolor=(1, 1, 1, 0.42), edgecolor="none", pad=0.8),
            )
            category_id += 1


def _draw_condition_abl_grid(
    ax: plt.Axes,
    summary_df: pd.DataFrame,
    *,
    title: str,
    panel_label: str | None = None,
    condition_column: str,
    condition_order: list[str],
    value_column: str,
    season: str | None,
    norm: matplotlib.colors.Normalize,
    cmap: matplotlib.colors.Colormap,
    show_ylabels: bool = True,
) -> None:
    base_fontsize = 10.5
    n_rows = len(ABL_ORDER)
    n_cols = len(condition_order)
    ax.set_xlim(0, n_cols)
    ax.set_ylim(n_rows, 0)
    ax.set_aspect("auto")
    ax.set_anchor("C")
    ax.set_title(SEASON_DISPLAY_LABELS.get(title, title), fontsize=base_fontsize, pad=4, fontweight="normal")
    if panel_label:
        ax.text(
            -0.085,
            1.055,
            panel_label,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=base_fontsize,
            fontweight="normal",
        )
    ax.set_xticks(np.arange(n_cols) + 0.5)
    ax.set_xticklabels([])
    ax.set_yticks(np.arange(n_rows) + 0.5)
    if show_ylabels:
        ax.set_yticklabels(ABL_ORDER, fontsize=base_fontsize, fontweight="normal")
        ax.tick_params(labelleft=True)
    else:
        ax.set_yticklabels(["", "", ""])
        ax.tick_params(labelleft=False)
    ax.tick_params(length=0)

    for spine in ax.spines.values():
        spine.set_visible(False)

    for col, condition in enumerate(condition_order):
        for row, abl_type in enumerate(ABL_ORDER):
            value = _condition_grid_value(summary_df, condition_column, condition, abl_type, value_column, season)
            facecolor = cmap(norm(value))
            edgecolor = ABL_COLORS[abl_type]
            rect = plt.Rectangle((col, row), 1, 1, facecolor=facecolor, edgecolor=edgecolor, linewidth=1.8)
            ax.add_patch(rect)

            text_color = "white" if norm(value) > 0.62 else "#1f1f1f"
            ax.text(
                col + 0.96,
                row + 0.26,
                REFINED_CONDITION_SHORT_LABELS.get(condition, condition),
                ha="right",
                va="center",
                fontsize=base_fontsize,
                color=text_color,
                fontweight="normal",
            )
            ax.text(
                col + 0.50,
                row + 0.76,
                _format_percent(value),
                ha="center",
                va="center",
                fontsize=base_fontsize,
                color=text_color,
                fontweight="normal",
                bbox=dict(facecolor=(1, 1, 1, 0.42), edgecolor="none", pad=0.6),
            )


def plot_weather_abl_grid(summary_df: pd.DataFrame, output_path: Path) -> None:
    _set_style()
    plt.rcParams["axes.grid"] = False
    cmap = plt.get_cmap("Greys")
    annual_max = max(1.0, float(summary_df["annual_percent_of_all_valid"].max()))
    season_max = max(1.0, float(summary_df["season_percent_within_category"].max()))
    annual_norm = matplotlib.colors.Normalize(vmin=0.0, vmax=np.ceil(annual_max / 5.0) * 5.0)
    season_norm = matplotlib.colors.Normalize(vmin=0.0, vmax=np.ceil(season_max / 5.0) * 5.0)

    fig = plt.figure(figsize=(10.5, 10.0))
    gs = fig.add_gridspec(
        nrows=4,
        ncols=4,
        height_ratios=[1.0, 0.09, 1.0, 1.0],
        width_ratios=[1.0, 1.0, 0.14, 0.86],
        hspace=0.48,
        wspace=0.28,
    )

    annual_ax = fig.add_subplot(gs[0, 0:2])
    _draw_weather_abl_grid(
        annual_ax,
        summary_df,
        title="Annual",
        value_column="annual_percent_of_all_valid",
        season=None,
        norm=annual_norm,
        cmap=cmap,
    )

    legend_ax = fig.add_subplot(gs[0, 3])
    legend_ax.axis("off")
    legend_ax.text(0.0, 0.96, "ABL Type Key", fontsize=10, fontweight="bold", ha="left", va="top")
    for i, abl_type in enumerate(ABL_ORDER):
        y = 0.76 - i * 0.18
        legend_ax.plot([0.0, 0.28], [y, y], color=ABL_COLORS[abl_type], linewidth=2.4)
        legend_ax.text(0.34, y, abl_type, fontsize=9, va="center", ha="left")
    legend_ax.set_xlim(0, 1)
    legend_ax.set_ylim(0, 1)

    annual_cax = fig.add_subplot(gs[1, 0:2])
    annual_sm = plt.cm.ScalarMappable(norm=annual_norm, cmap=cmap)
    annual_cb = fig.colorbar(annual_sm, cax=annual_cax, orientation="horizontal")
    annual_cb.set_label("Percent of all valid cases", fontsize=9)
    annual_cax.xaxis.set_label_position("top")

    season_axes = [
        fig.add_subplot(gs[2, 0]),
        fig.add_subplot(gs[2, 1]),
        fig.add_subplot(gs[3, 0]),
        fig.add_subplot(gs[3, 1]),
    ]
    for ax, season in zip(season_axes, SEASON_ORDER):
        _draw_weather_abl_grid(
            ax,
            summary_df,
            title=season,
            value_column="season_percent_within_category",
            season=season,
            norm=season_norm,
            cmap=cmap,
        )

    season_cax = fig.add_subplot(gs[3, 3])
    season_sm = plt.cm.ScalarMappable(norm=season_norm, cmap=cmap)
    season_cb = fig.colorbar(season_sm, cax=season_cax, orientation="vertical")
    season_cb.set_label("Percent within weather-ABL category", fontsize=9)

    fig.suptitle("Weather-Condition ABL-Type Frequency Grid", fontsize=15, y=0.985)
    fig.text(
        0.02,
        0.012,
        "Cell number identifies the weather-condition/ABL-type category. "
        "Border color indicates ABL type. Annual values are fractions of all valid samples; "
        "seasonal values are fractions within each category.",
        fontsize=9,
        ha="left",
        va="bottom",
    )
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_refined_condition_abl_grid(summary_df: pd.DataFrame, output_path: Path) -> None:
    _set_style()
    plt.rcParams["axes.grid"] = False
    plot_order = [
        condition
        for condition in REFINED_CONDITION_ORDER
        if summary_df.loc[summary_df["refined_condition"] == condition, "total_n"].sum() > 0
    ]
    cmap = plt.get_cmap("Greys")
    annual_max = max(1.0, float(summary_df["annual_percent_of_all_valid"].max()))
    season_max = max(1.0, float(summary_df["season_percent_within_category"].max()))
    annual_norm = matplotlib.colors.Normalize(vmin=0.0, vmax=np.ceil(annual_max / 5.0) * 5.0)
    season_norm = matplotlib.colors.Normalize(vmin=0.0, vmax=np.ceil(season_max / 5.0) * 5.0)

    fig_width_in = 18.0 / 2.54
    fig_height_in = 5.80
    fig = plt.figure(figsize=(fig_width_in, fig_height_in))

    # Chinese "5号" typography is approximately 10.5 pt.
    top_fontsize = 10.5
    left_edge = 1.0 / 18.0
    right_edge = 0.92

    annual_left = left_edge
    annual_width = 0.65
    annual_height = 0.225
    annual_ax = fig.add_axes([annual_left, 0.715, annual_width, annual_height])
    _draw_condition_abl_grid(
        annual_ax,
        summary_df,
        title="Annual",
        panel_label="(a)",
        condition_column="refined_condition",
        condition_order=plot_order,
        value_column="annual_percent_of_all_valid",
        season=None,
        norm=annual_norm,
        cmap=cmap,
    )
    annual_ax.set_title("Annual", fontsize=top_fontsize, pad=6, fontweight="normal")
    annual_ax.tick_params(axis="y", labelsize=top_fontsize)

    legend_ax = fig.add_axes([0.75, 0.715, 0.20, annual_height])
    legend_ax.axis("off")
    legend_ax.text(0.0, 0.92, "ABL Type Key", fontsize=top_fontsize, fontweight="normal", ha="left", va="top")
    aligned_line_end = 0.12 + 1.0 / (2.54 * fig_width_in * 0.20)
    legend_y_positions = [0.68, 0.43, 0.18]
    for abl_type, y in zip(ABL_ORDER, legend_y_positions):
        legend_ax.plot([0.0, aligned_line_end], [y, y], color=ABL_COLORS[abl_type], linewidth=2.0)
        legend_ax.text(
            max(0.16, aligned_line_end + 0.04),
            y,
            abl_type,
            fontsize=top_fontsize,
            fontweight="normal",
            va="center",
            ha="left",
        )
    legend_ax.set_xlim(0, 1)
    legend_ax.set_ylim(0, 1)

    annual_cax = fig.add_axes([annual_left, 0.635, annual_width, 0.024])
    annual_sm = plt.cm.ScalarMappable(norm=annual_norm, cmap=cmap)
    annual_cb = fig.colorbar(annual_sm, cax=annual_cax, orientation="horizontal")
    annual_cb.set_label("Percent of all valid cases", fontsize=top_fontsize, labelpad=1.5)
    annual_cb.ax.tick_params(labelsize=top_fontsize, length=2.5, pad=1.5)

    sep = matplotlib.lines.Line2D(
        [left_edge, 0.96],
        [0.575, 0.575],
        transform=fig.transFigure,
        color="#9a9a9a",
        linewidth=0.9,
        linestyle="-",
    )
    fig.add_artist(sep)

    season_width = 0.38
    season_height = 0.205
    season_axes = [
        fig.add_axes([left_edge, 0.335, season_width, season_height]),
        fig.add_axes([0.54, 0.335, season_width, season_height]),
        fig.add_axes([left_edge, 0.075, season_width, season_height]),
        fig.add_axes([0.54, 0.075, season_width, season_height]),
    ]
    for idx, (ax, season) in enumerate(zip(season_axes, SEASON_ORDER)):
        _draw_condition_abl_grid(
            ax,
            summary_df,
            title=season,
            panel_label=f"({chr(ord('a') + idx + 1)})",
            condition_column="refined_condition",
            condition_order=plot_order,
            value_column="season_percent_within_category",
            season=season,
            norm=season_norm,
            cmap=cmap,
            show_ylabels=(idx % 2 == 0),
        )

    fig.savefig(output_path, dpi=300)

    original_size = fig.get_size_inches().copy()
    svg_path = output_path.with_name(f"{output_path.stem}_14cm.svg")
    fig.savefig(svg_path, format="svg")
    fig.set_size_inches(original_size)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_dataset()

    weather_summary = summarize_weather(df)
    refined_condition_summary = summarize_refined_condition(df)
    cloudy_refined_summary = summarize_cloudy_refined(df)
    weather_abl_grid_summary = summarize_weather_abl_grid(df)
    refined_condition_abl_grid_summary = summarize_refined_condition_abl_grid(df)

    df.to_csv(OUT_DIR / "abl_type_weather_merged.csv", index=False)
    weather_summary.to_csv(OUT_DIR / "weather_abl_type_summary.csv", index=False)
    refined_condition_summary.to_csv(OUT_DIR / "refined_condition_abl_type_summary.csv", index=False)
    cloudy_refined_summary.to_csv(OUT_DIR / "cloudy_coupling_abl_type_summary.csv", index=False)
    weather_abl_grid_summary.to_csv(OUT_DIR / "weather_abl_grid_summary.csv", index=False)
    refined_condition_abl_grid_summary.to_csv(OUT_DIR / "refined_condition_abl_grid_summary.csv", index=False)
    write_overview(df, OUT_DIR / "overview.txt")

    plot_weather_distribution(weather_summary, OUT_DIR / "weather_abl_type_fraction.png")
    plot_weather_counts(weather_summary, OUT_DIR / "weather_abl_type_counts.png")
    plot_refined_condition_distribution(refined_condition_summary, OUT_DIR / "refined_condition_abl_type_fraction.png")
    plot_cloudy_refined(cloudy_refined_summary, OUT_DIR / "cloudy_coupling_abl_type_fraction.png")
    plot_cloudy_refined_counts(cloudy_refined_summary, OUT_DIR / "cloudy_coupling_abl_type_counts.png")
    plot_weather_abl_grid(weather_abl_grid_summary, OUT_DIR / "weather_abl_grid.png")
    plot_refined_condition_abl_grid(
        refined_condition_abl_grid_summary,
        OUT_DIR / "refined_condition_abl_grid.png",
    )

    print("ABL type weather analysis written to:", OUT_DIR)


if __name__ == "__main__":
    main()
