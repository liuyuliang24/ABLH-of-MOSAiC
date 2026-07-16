#!/usr/bin/env python3
from __future__ import annotations

import math
import os
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent
MPL_DIR = OUT_DIR / ".mplconfig"
MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT_DIR = OUT_DIR.parent
DATA_CSV = ROOT_DIR / "llj_analysis_results" / "llj_analysis_merged.csv"
SELECTED_CSV = OUT_DIR / "selected_abl_type_soundings.csv"
COMBINED_FIG = OUT_DIR / "selected_abl_type_sonde_examples.png"

HEIGHT_START_M = 10.0
HEIGHT_STEP_M = 20.0
MIN_PLOT_HEIGHT_M = 30.0
MAX_PLOT_HEIGHT_M = 1000.0
BASE_FONTSIZE = 10.5
FIG_WIDTH_CM = 18.0

ABL_ORDER = ["CBL", "NBL", "SBL"]
CLASS_COLORS = {
    "CBL": "#d62828",
    "NBL": "#f4a261",
    "SBL": "#1d4ed8",
}
TOP_XLABELS = {
    "temp": "Virtual Potential Temp. (K)",
    "wind": "Wind Direction (°)",
}
BOTTOM_XLABELS = {
    "temp": "Temperature (°C)",
    "wind": "Wind Speed (m/s)",
    "humidity": "Specific Humidity (kg/kg)",
}


def sorted_profile_columns(columns: list[str], prefix: str) -> list[str]:
    matched: list[tuple[int, str]] = []
    for col in columns:
        if col.startswith(prefix):
            try:
                matched.append((int(col.split("_")[-1]), col))
            except ValueError:
                continue
    matched.sort(key=lambda item: item[0])
    return [col for _, col in matched]


def build_height_grid(n_levels: int) -> np.ndarray:
    return HEIGHT_START_M + np.arange(n_levels, dtype=float) * HEIGHT_STEP_M


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


def virtual_potential_temperature(temp_k: np.ndarray, p_pa: np.ndarray, q: np.ndarray) -> np.ndarray:
    p_hpa = p_pa / 100.0
    theta_v = np.full_like(temp_k, np.nan, dtype=float)
    valid = np.isfinite(temp_k) & np.isfinite(p_hpa) & np.isfinite(q) & (p_hpa > 0.0) & (q >= 0.0)
    if not np.any(valid):
        return theta_v
    theta_v[valid] = temp_k[valid] * (1.0 + 0.61 * q[valid]) * np.power(1000.0 / p_hpa[valid], 0.2854)
    return theta_v


def interpolate_row(profile: np.ndarray, heights: np.ndarray, target_height: float) -> float:
    valid = np.isfinite(profile) & np.isfinite(heights)
    if valid.sum() < 2:
        return float("nan")
    x = heights[valid]
    y = profile[valid]
    if target_height < x.min() or target_height > x.max():
        return float("nan")
    return float(np.interp(target_height, x, y))


def weather_group(status: float) -> str:
    if not np.isfinite(status):
        return "Unknown"
    status = int(status)
    if status == 0:
        return "Clear"
    if 1 <= status <= 3:
        return "Cloudy"
    return "Fog/Mist"


def wind_direction_from_uv(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    direction = (270.0 - np.degrees(np.arctan2(v, u))) % 360.0
    direction[~(np.isfinite(u) & np.isfinite(v))] = np.nan
    return direction


def break_wrapped_direction(direction: np.ndarray) -> np.ndarray:
    out = direction.astype(float).copy()
    for i in range(1, len(out)):
        if np.isfinite(out[i]) and np.isfinite(out[i - 1]) and abs(out[i] - out[i - 1]) > 180.0:
            out[i] = np.nan
    return out


def read_dataset() -> tuple[pd.DataFrame, list[str], list[str], list[str], list[str], list[str], np.ndarray]:
    header = pd.read_csv(DATA_CSV, nrows=0)
    cols = header.columns.tolist()
    temp_cols = sorted_profile_columns(cols, "temp_")
    sh_cols = sorted_profile_columns(cols, "sh_")
    pres_cols = sorted_profile_columns(cols, "bar_pres_")
    u_cols = sorted_profile_columns(cols, "u_wind_")
    v_cols = sorted_profile_columns(cols, "v_wind_")

    usecols = [
        "sonde_time",
        "time",
        "ablh_m",
        "detection_status",
        "first_cbh",
        *temp_cols,
        *sh_cols,
        *pres_cols,
        *u_cols,
        *v_cols,
    ]
    df = pd.read_csv(DATA_CSV, usecols=usecols)
    df["sonde_time"] = pd.to_datetime(df["sonde_time"], errors="coerce")
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    heights = build_height_grid(len(temp_cols))
    return df, temp_cols, sh_cols, pres_cols, u_cols, v_cols, heights


def classify_abl_types(
    df: pd.DataFrame,
    temp_cols: list[str],
    sh_cols: list[str],
    pres_cols: list[str],
    heights: np.ndarray,
) -> pd.DataFrame:
    out = df.copy()
    temp_c = out[temp_cols].to_numpy(dtype=float)
    temp_k = temp_c + 273.15
    sh = out[sh_cols].to_numpy(dtype=float)
    pres = out[pres_cols].to_numpy(dtype=float)

    pres[(pres <= 1000.0) | (pres >= 110000.0)] = np.nan
    sh[(sh < 0.0) | (sh >= 0.1)] = np.nan

    theta_e = equivalent_potential_temperature(temp_k, pres, sh)
    theta_v = virtual_potential_temperature(temp_k, pres, sh)

    t50 = np.array([interpolate_row(theta_e[i], heights, 50.0) for i in range(len(out))], dtype=float)
    t100 = np.array([interpolate_row(theta_e[i], heights, 100.0) for i in range(len(out))], dtype=float)
    diff = t100 - t50

    out["thetae_diff_100m_50m_k"] = diff
    out["abl_type"] = np.where(diff > 0.2, "SBL", np.where(diff < -0.2, "CBL", "NBL"))
    out.loc[~np.isfinite(diff), "abl_type"] = np.nan
    out["weather_condition"] = out["detection_status"].apply(weather_group)
    out.attrs["theta_v"] = theta_v
    return out


def choose_examples(
    df: pd.DataFrame,
    theta_v: np.ndarray,
    temp_cols: list[str],
    sh_cols: list[str],
    pres_cols: list[str],
    u_cols: list[str],
    v_cols: list[str],
    heights: np.ndarray,
) -> pd.DataFrame:
    plot_mask = (heights >= MIN_PLOT_HEIGHT_M) & (heights <= MAX_PLOT_HEIGHT_M)
    low_temp_cols = [col for col, keep in zip(temp_cols, plot_mask) if keep]
    low_sh_cols = [col for col, keep in zip(sh_cols, plot_mask) if keep]
    low_pres_cols = [col for col, keep in zip(pres_cols, plot_mask) if keep]
    low_u_cols = [col for col, keep in zip(u_cols, plot_mask) if keep]
    low_v_cols = [col for col, keep in zip(v_cols, plot_mask) if keep]

    valid = df.loc[df["abl_type"].isin(ABL_ORDER) & df["ablh_m"].notna()].copy()
    valid["temp_complete"] = np.sum(np.isfinite(valid[low_temp_cols].to_numpy(dtype=float)), axis=1)
    valid["sh_complete"] = np.sum(np.isfinite(valid[low_sh_cols].to_numpy(dtype=float)), axis=1)
    valid["pres_complete"] = np.sum(np.isfinite(valid[low_pres_cols].to_numpy(dtype=float)), axis=1)
    valid["u_complete"] = np.sum(np.isfinite(valid[low_u_cols].to_numpy(dtype=float)), axis=1)
    valid["v_complete"] = np.sum(np.isfinite(valid[low_v_cols].to_numpy(dtype=float)), axis=1)
    valid["theta_v_complete"] = np.sum(np.isfinite(theta_v[valid.index.to_numpy()][:, plot_mask]), axis=1)

    required_levels = int(plot_mask.sum() * 0.9)
    valid = valid.loc[
        (valid["temp_complete"] >= required_levels)
        & (valid["sh_complete"] >= required_levels)
        & (valid["pres_complete"] >= required_levels)
        & (valid["u_complete"] >= required_levels)
        & (valid["v_complete"] >= required_levels)
        & (valid["theta_v_complete"] >= required_levels)
    ].copy()

    selected_rows: list[pd.Series] = []
    for abl_type in ABL_ORDER:
        subset = valid.loc[valid["abl_type"] == abl_type].copy()
        if subset.empty:
            raise RuntimeError(f"No valid profiles found for {abl_type}.")

        clear_subset = subset.loc[subset["detection_status"] == 0].copy()
        pool = clear_subset if not clear_subset.empty else subset
        target_ablh = float(pool["ablh_m"].median())
        target_theta = float(pool["thetae_diff_100m_50m_k"].median())

        pool["selection_score"] = (
            (pool["ablh_m"] - target_ablh).abs()
            + 15.0 * (pool["thetae_diff_100m_50m_k"] - target_theta).abs()
        )
        pool = pool.sort_values(["selection_score", "sonde_time"], ascending=[True, True])
        selected_rows.append(pool.iloc[0])

    selected = pd.DataFrame(selected_rows).reset_index(drop=True)
    return selected


def axis_limits(values: np.ndarray, min_pad: float, default: tuple[float, float]) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return default
    lower = float(finite.min() - min_pad)
    upper = float(finite.max() + min_pad)
    if math.isclose(lower, upper):
        upper = lower + min_pad * 2.0
    return lower, upper


def draw_profile_row(
    axes: tuple[plt.Axes, plt.Axes, plt.Axes],
    row: pd.Series,
    theta_v_profile: np.ndarray,
    heights: np.ndarray,
    *,
    show_legend: bool,
    show_top_labels: bool,
    show_bottom_labels: bool,
    show_ylabels: bool,
    panel_labels: tuple[str | None, str | None, str | None] = (None, None, None),
) -> None:
    ax_temp, ax_wind, ax_q = axes
    plot_mask = (heights >= MIN_PLOT_HEIGHT_M) & (heights <= MAX_PLOT_HEIGHT_M)
    z = heights[plot_mask]

    temp_c = row.filter(regex=r"^temp_").to_numpy(dtype=float)[plot_mask]
    q = row.filter(regex=r"^sh_").to_numpy(dtype=float)[plot_mask]
    p = row.filter(regex=r"^bar_pres_").to_numpy(dtype=float)[plot_mask]
    u = row.filter(regex=r"^u_wind_").to_numpy(dtype=float)[plot_mask]
    v = row.filter(regex=r"^v_wind_").to_numpy(dtype=float)[plot_mask]

    p[(p <= 1000.0) | (p >= 110000.0)] = np.nan
    q[(q < 0.0) | (q >= 0.1)] = np.nan
    theta_v = theta_v_profile[plot_mask]

    wind_speed = np.sqrt(u ** 2 + v ** 2)
    wind_dir = break_wrapped_direction(wind_direction_from_uv(u, v))

    ax_temp_top = ax_temp.twiny()
    ax_wind_top = ax_wind.twiny()

    temp_line, = ax_temp.plot(temp_c, z, color="#1d4ed8", linewidth=1.6, label="Temperature (°C)")
    theta_v_line, = ax_temp_top.plot(theta_v, z, color="#dc2626", linewidth=1.4, label="Virtual Potential Temp. (K)")
    speed_line, = ax_wind.plot(wind_speed, z, color="#15803d", linewidth=1.6, label="Wind Speed (m/s)")
    dir_line, = ax_wind_top.plot(wind_dir, z, color="#d946ef", linewidth=1.2, label="Wind Direction (°)")
    q_line, = ax_q.plot(q, z, color="#06b6d4", linewidth=1.6, label="Specific Humidity (kg/kg)")

    for axis in (ax_temp, ax_wind, ax_q, ax_temp_top, ax_wind_top):
        axis.grid(False)

    ablh = float(row["ablh_m"])
    for ax in (ax_temp, ax_wind, ax_q):
        ax.axhline(ablh, color="black", linestyle=":", linewidth=1.1)
        ax.set_ylim(MIN_PLOT_HEIGHT_M, MAX_PLOT_HEIGHT_M)
        ax.set_yticks([30.0, 250.0, 500.0, 750.0, 1000.0])
        ax.tick_params(axis="y", labelleft=show_ylabels, left=True)
        for label in ax.get_yticklabels():
            label.set_visible(show_ylabels)

    ax_temp.set_xlim(*axis_limits(temp_c, 0.8, (-30.0, 0.0)))
    ax_temp_top.set_xlim(*axis_limits(theta_v, 0.8, (250.0, 300.0)))
    ax_wind.set_xlim(*axis_limits(wind_speed, 0.7, (0.0, 20.0)))
    ax_wind_top.set_xlim(0.0, 360.0)
    ax_q.set_xlim(*axis_limits(q, 0.00008, (0.0, 0.0025)))

    if show_ylabels:
        ax_temp.set_ylabel("Height (m)")
    else:
        ax_temp.set_ylabel("")
    ax_temp.tick_params(axis="x", colors="#1d4ed8", labelbottom=True, bottom=True)
    for label in ax_temp.get_xticklabels():
        label.set_visible(True)
    if show_bottom_labels:
        ax_temp.set_xlabel(BOTTOM_XLABELS["temp"], color="#1d4ed8")
    else:
        ax_temp.set_xlabel("")

    ax_temp_top.tick_params(axis="x", colors="#dc2626", labeltop=True, top=True)
    for label in ax_temp_top.get_xticklabels():
        label.set_visible(True)
    if show_top_labels:
        ax_temp_top.set_xlabel(TOP_XLABELS["temp"], color="#dc2626")
    else:
        ax_temp_top.set_xlabel("")

    ax_wind.tick_params(axis="x", colors="#15803d", labelbottom=True, bottom=True)
    for label in ax_wind.get_xticklabels():
        label.set_visible(True)
    if show_bottom_labels:
        ax_wind.set_xlabel(BOTTOM_XLABELS["wind"], color="#15803d")
    else:
        ax_wind.set_xlabel("")

    ax_wind_top.tick_params(axis="x", colors="#d946ef", labeltop=True, top=True)
    for label in ax_wind_top.get_xticklabels():
        label.set_visible(True)
    if show_top_labels:
        ax_wind_top.set_xlabel(TOP_XLABELS["wind"], color="#d946ef")
    else:
        ax_wind_top.set_xlabel("")
    ax_wind_top.set_xticks(np.arange(0.0, 361.0, 60.0))

    ax_q.tick_params(axis="x", colors="#0891b2", labelbottom=True, bottom=True)
    for label in ax_q.get_xticklabels():
        label.set_visible(True)
    if show_bottom_labels:
        ax_q.set_xlabel(BOTTOM_XLABELS["humidity"], color="#0891b2")
    else:
        ax_q.set_xlabel("")

    row_title = (
        f"{row['abl_type']} | Sonde {pd.Timestamp(row['sonde_time']).strftime('%Y-%m-%d %H:%M:%S')} "
        f"(ABLH = {row['ablh_m']:.0f} m)"
    )
    ax_wind.set_title(row_title, fontsize=BASE_FONTSIZE, pad=5, color=CLASS_COLORS[str(row["abl_type"])])

    temp_panel_label, wind_panel_label, q_panel_label = panel_labels
    if temp_panel_label:
        ax_temp.text(
            0.03, 0.97, temp_panel_label, transform=ax_temp.transAxes,
            ha="left", va="top", fontsize=BASE_FONTSIZE, color="black"
        )
    if wind_panel_label:
        ax_wind.text(
            0.03, 0.97, wind_panel_label, transform=ax_wind.transAxes,
            ha="left", va="top", fontsize=BASE_FONTSIZE, color="black"
        )
    if q_panel_label:
        ax_q.text(
            0.03, 0.97, q_panel_label, transform=ax_q.transAxes,
            ha="left", va="top", fontsize=BASE_FONTSIZE, color="black"
        )

    if show_legend:
        handles = [temp_line, theta_v_line, speed_line, dir_line, q_line]
        labels = [h.get_label() for h in handles]
        ablh_handle = plt.Line2D([], [], color="black", linestyle=":", linewidth=1.1, label=f"ABLH = {ablh:.0f} m")
        handles.append(ablh_handle)
        labels.append(ablh_handle.get_label())
        ax_temp.legend(handles, labels, loc="upper left", fontsize=BASE_FONTSIZE, frameon=False)


def plot_single_example(
    row: pd.Series,
    theta_v_profile: np.ndarray,
    heights: np.ndarray,
    output_path: Path,
) -> None:
    fig_width_in = FIG_WIDTH_CM / 2.54
    fig_height_in = fig_width_in * (5.2 / 14.5)
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(fig_width_in, fig_height_in), sharey=True)
    draw_profile_row(
        (axes[0], axes[1], axes[2]),
        row,
        theta_v_profile,
        heights,
        show_legend=True,
        show_top_labels=True,
        show_bottom_labels=True,
        show_ylabels=True,
    )
    fig.subplots_adjust(left=0.12, right=0.985, top=0.78, bottom=0.18, wspace=0.32)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_combined_examples(selected: pd.DataFrame, theta_v: np.ndarray, heights: np.ndarray) -> None:
    fig_width_in = FIG_WIDTH_CM / 2.54
    fig_height_in = fig_width_in * (14.8 / 14.5)
    fig, axes = plt.subplots(nrows=len(selected), ncols=3, figsize=(fig_width_in, fig_height_in), sharey=True)
    for idx, row in selected.iterrows():
        base = idx * 3
        draw_profile_row(
            (axes[idx, 0], axes[idx, 1], axes[idx, 2]),
            row,
            theta_v[int(row.name)],
            heights,
            show_legend=False,
            show_top_labels=(idx == 0),
            show_bottom_labels=(idx == len(selected) - 1),
            show_ylabels=True,
            panel_labels=(
                f"({chr(ord('a') + base)})",
                f"({chr(ord('a') + base + 1)})",
                f"({chr(ord('a') + base + 2)})",
            ),
        )
        for col in (1, 2):
            axes[idx, col].tick_params(axis="y", labelleft=False)
    fig.subplots_adjust(left=0.11, right=0.985, top=0.91, bottom=0.08, hspace=0.54, wspace=0.24)
    fig.savefig(COMBINED_FIG, dpi=220)
    plt.close(fig)


def main() -> None:
    plt.rcParams["figure.dpi"] = 140
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = BASE_FONTSIZE
    plt.rcParams["axes.labelsize"] = BASE_FONTSIZE
    plt.rcParams["axes.titlesize"] = BASE_FONTSIZE
    plt.rcParams["axes.grid"] = False
    plt.rcParams["xtick.labelsize"] = BASE_FONTSIZE
    plt.rcParams["ytick.labelsize"] = BASE_FONTSIZE
    plt.rcParams["legend.fontsize"] = BASE_FONTSIZE

    df, temp_cols, sh_cols, pres_cols, u_cols, v_cols, heights = read_dataset()
    classified = classify_abl_types(df, temp_cols, sh_cols, pres_cols, heights)
    theta_v = classified.attrs["theta_v"]

    selected = choose_examples(classified, theta_v, temp_cols, sh_cols, pres_cols, u_cols, v_cols, heights)
    selected = selected.assign(
        thetae_diff_100m_50m_k=selected["thetae_diff_100m_50m_k"].astype(float),
        ablh_m=selected["ablh_m"].astype(float),
    )
    selected = selected[
        [
            "abl_type",
            "sonde_time",
            "time",
            "ablh_m",
            "thetae_diff_100m_50m_k",
            "weather_condition",
            "detection_status",
            "first_cbh",
            "selection_score",
        ]
    ].copy()
    selected.to_csv(SELECTED_CSV, index=False)

    classified_indexed = classified.reset_index()
    merged = selected.merge(
        classified_indexed[["index", "sonde_time", "abl_type"]],
        on=["sonde_time", "abl_type"],
        how="left",
        validate="one_to_one",
    )
    if merged["index"].isna().any():
        raise RuntimeError("Failed to map selected rows back to profile indices.")

    selected_with_index = selected.copy()
    selected_with_index["profile_index"] = merged["index"].astype(int).to_numpy()

    plot_rows = classified.iloc[selected_with_index["profile_index"].to_numpy()].copy().reset_index(drop=True)
    plot_combined_examples(plot_rows, theta_v[selected_with_index["profile_index"].to_numpy()], heights)

    print("Selected examples written to:", SELECTED_CSV)
    print("Combined figure written to:", COMBINED_FIG)


if __name__ == "__main__":
    main()
