from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import rcParams


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

MATCHED_DEFAULT = PROJECT_DIR / "matched_all_data_10min_with_aeri.csv"
PBL_DEFAULT = PROJECT_DIR / "train_val_predictions.csv"

TIME_COLUMN_MATCHED = "target_10min"
TIME_COLUMN_PBL = "time"
PBL_COLUMNS = ["predicted_pbl_height", "actual_pbl_height"]
FIRST_CBH_COLUMN = "first_cbh"

TEMP_COLUMNS = [f"temp_{i}" for i in range(1, 101)]
SH_COLUMNS = [f"sh_{i}" for i in range(1, 101)]
U_COLUMNS = [f"u_wind_{i}" for i in range(1, 101)]
V_COLUMNS = [f"v_wind_{i}" for i in range(1, 101)]
PRESSURE_COLUMNS = [f"bar_pres_{i}" for i in range(1, 101)]

HEIGHTS_M = np.arange(0.01, 2.00, 0.02)[:100] * 1000.0
EPSILON = 0.622
CP_AIR = 1004.0
LV = 2.5e6
FONT_SIZE_PT = 10.5

rcParams.update(
    {
        "font.family": "Times New Roman",
        "font.size": FONT_SIZE_PT,
        "font.weight": "normal",
        "axes.titlesize": FONT_SIZE_PT,
        "axes.titleweight": "normal",
        "axes.labelsize": FONT_SIZE_PT,
        "axes.labelweight": "normal",
        "xtick.labelsize": FONT_SIZE_PT,
        "ytick.labelsize": FONT_SIZE_PT,
        "legend.fontsize": FONT_SIZE_PT,
        "figure.titlesize": FONT_SIZE_PT,
        "figure.titleweight": "normal",
        "mathtext.fontset": "stix",
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a MOSAiC-style multi-panel profile overview figure from local CSV files.",
    )
    parser.add_argument("--matched-csv", type=Path, default=MATCHED_DEFAULT)
    parser.add_argument("--pbl-csv", type=Path, default=PBL_DEFAULT)
    parser.add_argument("--start", default="2020-05-01")
    parser.add_argument("--end", default="2020-05-15")
    parser.add_argument("--freq", default="10min")
    parser.add_argument("--max-height-km", type=float, default=1.5)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output PNG path. Defaults to a file in the script directory.",
    )
    return parser.parse_args()


def parse_window(start_str: str, end_str: str, freq: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp(start_str)
    end = pd.Timestamp(end_str)
    freq_offset = pd.tseries.frequencies.to_offset(freq)

    if " " not in end_str and "T" not in end_str:
        end = end + pd.Timedelta(days=1) - freq_offset

    if end < start:
        raise ValueError("`end` must be later than or equal to `start`.")

    return start, end


def load_windowed_data(
    matched_csv: Path,
    pbl_csv: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
    freq: str,
) -> tuple[pd.DatetimeIndex, pd.DataFrame, pd.DataFrame]:
    matched_usecols = [TIME_COLUMN_MATCHED, FIRST_CBH_COLUMN, *TEMP_COLUMNS, *SH_COLUMNS, *U_COLUMNS, *V_COLUMNS, *PRESSURE_COLUMNS]
    matched_df = pd.read_csv(matched_csv, usecols=matched_usecols)
    matched_df[TIME_COLUMN_MATCHED] = pd.to_datetime(matched_df[TIME_COLUMN_MATCHED], errors="coerce")
    matched_df = matched_df.dropna(subset=[TIME_COLUMN_MATCHED])
    matched_df = matched_df.loc[matched_df[TIME_COLUMN_MATCHED].between(start, end)].copy()
    matched_df = matched_df.groupby(TIME_COLUMN_MATCHED, as_index=True).mean(numeric_only=True)
    matched_rows = len(matched_df)

    pbl_usecols = [TIME_COLUMN_PBL, *PBL_COLUMNS]
    pbl_df = pd.read_csv(pbl_csv, usecols=pbl_usecols)
    pbl_df[TIME_COLUMN_PBL] = pd.to_datetime(pbl_df[TIME_COLUMN_PBL], errors="coerce")
    pbl_df = pbl_df.dropna(subset=[TIME_COLUMN_PBL])
    pbl_df = pbl_df.loc[pbl_df[TIME_COLUMN_PBL].between(start, end)].copy()
    pbl_df = pbl_df.groupby(TIME_COLUMN_PBL, as_index=True).mean(numeric_only=True)
    pbl_rows = len(pbl_df)

    if matched_rows == 0 or pbl_rows == 0:
        raise ValueError("No data available in the requested time window.")

    time_index = pd.date_range(start=start, end=end, freq=freq)
    matched_df = matched_df.reindex(time_index)
    pbl_df = pbl_df.reindex(time_index)

    return time_index, matched_df, pbl_df


def compute_relative_humidity(temp_c: np.ndarray, q: np.ndarray, pressure_pa: np.ndarray) -> np.ndarray:
    vapor_pressure = (q * pressure_pa) / (EPSILON + (1.0 - EPSILON) * q)
    saturation_pressure = 611.2 * np.exp((17.67 * temp_c) / (temp_c + 243.5))
    rh = 100.0 * vapor_pressure / np.clip(saturation_pressure, 1e-6, None)
    return np.clip(rh, 0.0, 100.0)


def compute_theta_e_anomaly(temp_c: np.ndarray, q: np.ndarray, pressure_pa: np.ndarray) -> np.ndarray:
    temp_k = temp_c + 273.15
    theta = temp_k * np.power(100000.0 / np.clip(pressure_pa, 1.0, None), 0.286)
    theta_e = theta * np.exp((LV * q) / (CP_AIR * np.clip(temp_k, 1.0, None)))
    return theta_e - np.nanmean(theta_e, axis=0, keepdims=True)


def smooth_series(values: np.ndarray, window: int) -> np.ndarray:
    series = pd.Series(values)
    return series.rolling(window=window, center=True, min_periods=1).median().to_numpy()


def prepare_fields(
    matched_df: pd.DataFrame,
    pbl_df: pd.DataFrame,
    max_height_km: float,
) -> dict[str, np.ndarray]:
    height_mask = (HEIGHTS_M / 1000.0) <= max_height_km + 1e-12
    heights_km = HEIGHTS_M[height_mask] / 1000.0

    temp_c = matched_df[TEMP_COLUMNS].to_numpy(dtype=float)[:, height_mask]
    q = matched_df[SH_COLUMNS].to_numpy(dtype=float)[:, height_mask]
    pressure_pa = matched_df[PRESSURE_COLUMNS].to_numpy(dtype=float)[:, height_mask]
    u_wind = matched_df[U_COLUMNS].to_numpy(dtype=float)[:, height_mask]
    v_wind = matched_df[V_COLUMNS].to_numpy(dtype=float)[:, height_mask]

    theta_e_anom = compute_theta_e_anomaly(temp_c, q, pressure_pa)
    wind_speed = np.sqrt(u_wind**2 + v_wind**2)
    relative_humidity = compute_relative_humidity(temp_c, q, pressure_pa)

    surface_pressure_hpa = matched_df[PRESSURE_COLUMNS[0]].to_numpy(dtype=float) / 100.0
    surface_temp_c = matched_df[TEMP_COLUMNS[:3]].mean(axis=1, skipna=True).to_numpy(dtype=float)
    surface_temp_mean_c = (
        pd.Series(surface_temp_c)
        .rolling(window=144, center=True, min_periods=12)
        .mean()
        .to_numpy()
    )

    predicted_pbl_height_km = smooth_series(
        pbl_df["predicted_pbl_height"].to_numpy(dtype=float) / 1000.0,
        window=3,
    )
    actual_pbl_height_km = smooth_series(
        pbl_df["actual_pbl_height"].to_numpy(dtype=float) / 1000.0,
        window=3,
    )
    first_cbh_km = smooth_series(matched_df[FIRST_CBH_COLUMN].to_numpy(dtype=float) / 1000.0, window=3)

    max_height = heights_km.max()
    predicted_pbl_height_km = np.where(
        (predicted_pbl_height_km >= 0.0) & (predicted_pbl_height_km <= max_height),
        predicted_pbl_height_km,
        np.nan,
    )
    actual_pbl_height_km = np.where(
        (actual_pbl_height_km >= 0.0) & (actual_pbl_height_km <= max_height),
        actual_pbl_height_km,
        np.nan,
    )
    first_cbh_km = np.where((first_cbh_km >= 0.0) & (first_cbh_km <= max_height), first_cbh_km, np.nan)

    return {
        "heights_km": heights_km,
        "theta_e_anom": theta_e_anom,
        "wind_speed": wind_speed,
        "relative_humidity": relative_humidity,
        "surface_pressure_hpa": surface_pressure_hpa,
        "surface_temp_c": surface_temp_c,
        "surface_temp_mean_c": surface_temp_mean_c,
        "predicted_pbl_height_km": predicted_pbl_height_km,
        "actual_pbl_height_km": actual_pbl_height_km,
        "first_cbh_km": first_cbh_km,
    }


def build_time_edges(time_index: pd.DatetimeIndex) -> np.ndarray:
    time_num = mdates.date2num(time_index.to_pydatetime())
    if len(time_num) == 1:
        half_step = 1.0 / 24.0 / 6.0
        return np.array([time_num[0] - half_step, time_num[0] + half_step])

    step = np.median(np.diff(time_num))
    edges = np.empty(len(time_num) + 1, dtype=float)
    edges[1:-1] = time_num[:-1] + np.diff(time_num) / 2.0
    edges[0] = time_num[0] - step / 2.0
    edges[-1] = time_num[-1] + step / 2.0
    return edges


def build_height_edges(heights_km: np.ndarray) -> np.ndarray:
    if len(heights_km) == 1:
        return np.array([heights_km[0] - 0.01, heights_km[0] + 0.01])

    step = np.median(np.diff(heights_km))
    edges = np.empty(len(heights_km) + 1, dtype=float)
    edges[1:-1] = heights_km[:-1] + np.diff(heights_km) / 2.0
    edges[0] = heights_km[0] - step / 2.0
    edges[-1] = heights_km[-1] + step / 2.0
    return edges


def compute_period_boundaries(start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    if start.normalize() == pd.Timestamp("2020-05-01") and end.normalize() == pd.Timestamp("2020-05-15"):
        return [pd.Timestamp("2020-05-05"), pd.Timestamp("2020-05-11")]

    total = end - start
    return [start + total / 3.0, start + 2.0 * total / 3.0]


def add_period_annotations(ax: plt.Axes, start: pd.Timestamp, end: pd.Timestamp, boundaries: list[pd.Timestamp]) -> None:
    segments = [start, *boundaries, end]
    for idx, label in enumerate(["P1", "P2", "P3"]):
        midpoint = segments[idx] + (segments[idx + 1] - segments[idx]) / 2.0
        ax.text(
            midpoint,
            1.01,
            label,
            ha="center",
            va="bottom",
            transform=ax.get_xaxis_transform(),
            fontsize=FONT_SIZE_PT,
        )


def overlay_lines(
    ax: plt.Axes,
    time_index: pd.DatetimeIndex,
    _actual_pbl_height_km: np.ndarray,
    predicted_pbl_height_km: np.ndarray,
    first_cbh_km: np.ndarray,
) -> None:
    ax.plot(time_index, first_cbh_km, color="0.6", linewidth=1.6, zorder=4)
    ax.plot(
        time_index,
        predicted_pbl_height_km,
        color="white",
        linewidth=1.25,
        linestyle="--",
        zorder=6,
    )


def plot_overview(
    time_index: pd.DatetimeIndex,
    fields: dict[str, np.ndarray],
    start: pd.Timestamp,
    end: pd.Timestamp,
    output_path: Path,
) -> None:
    theta_cmap = plt.get_cmap("turbo").copy()
    wind_cmap = LinearSegmentedColormap.from_list(
        "white_red",
        ["#ffffff", "#fee0d2", "#fcbba1", "#fb6a4a", "#de2d26", "#67000d"],
    )
    rh_cmap = plt.get_cmap("YlGn").copy()

    for cmap in (theta_cmap, wind_cmap, rh_cmap):
        cmap.set_bad(color="white")

    time_edges = build_time_edges(time_index)
    height_edges = build_height_edges(fields["heights_km"])
    boundaries = compute_period_boundaries(start, end)

    fig = plt.figure(figsize=(8.2, 10.1))
    gs = fig.add_gridspec(
        nrows=5,
        ncols=2,
        width_ratios=[38, 1],
        height_ratios=[2.35, 2.35, 2.35, 1.45, 1.45],
        hspace=0.06,
        wspace=0.10,
    )

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[1, 0], sharex=ax_a, sharey=ax_a)
    ax_c = fig.add_subplot(gs[2, 0], sharex=ax_a, sharey=ax_a)
    ax_d = fig.add_subplot(gs[3, 0], sharex=ax_a)
    ax_e = fig.add_subplot(gs[4, 0], sharex=ax_a)

    cax_a = fig.add_subplot(gs[0, 1])
    cax_b = fig.add_subplot(gs[1, 1])
    cax_c = fig.add_subplot(gs[2, 1])
    fig.add_subplot(gs[3, 1]).axis("off")
    fig.add_subplot(gs[4, 1]).axis("off")

    mesh_a = ax_a.pcolormesh(
        time_edges,
        height_edges,
        np.ma.masked_invalid(fields["theta_e_anom"].T),
        cmap=theta_cmap,
        shading="auto",
        vmin=-12,
        vmax=7,
    )
    mesh_b = ax_b.pcolormesh(
        time_edges,
        height_edges,
        np.ma.masked_invalid(fields["wind_speed"].T),
        cmap=wind_cmap,
        shading="auto",
        vmin=0,
        vmax=20,
    )
    mesh_c = ax_c.pcolormesh(
        time_edges,
        height_edges,
        np.ma.masked_invalid(fields["relative_humidity"].T),
        cmap=rh_cmap,
        shading="auto",
        vmin=40,
        vmax=100,
    )

    overlay_lines(
        ax_a,
        time_index,
        fields["actual_pbl_height_km"],
        fields["predicted_pbl_height_km"],
        fields["first_cbh_km"],
    )
    overlay_lines(
        ax_b,
        time_index,
        fields["actual_pbl_height_km"],
        fields["predicted_pbl_height_km"],
        fields["first_cbh_km"],
    )
    overlay_lines(
        ax_c,
        time_index,
        fields["actual_pbl_height_km"],
        fields["predicted_pbl_height_km"],
        fields["first_cbh_km"],
    )

    cbar_a = fig.colorbar(mesh_a, cax=cax_a)
    cbar_b = fig.colorbar(mesh_b, cax=cax_b)
    cbar_c = fig.colorbar(mesh_c, cax=cax_c)

    cbar_a.set_label(r"$\theta_e$ anomaly ($^\circ$C)", fontsize=12)
    cbar_b.set_label(r"wind speed (m s$^{-1}$)", fontsize=12)
    cbar_c.set_label("relative humidity (%)", fontsize=12)

    ax_d.plot(time_index, fields["surface_pressure_hpa"], color="0.45", linewidth=1.7)
    ax_d.axhline(1000.0, color="0.75", linestyle=":", linewidth=1.0)

    ax_e.plot(time_index, fields["surface_temp_c"], color="#ff375f", linewidth=1.7, label="T")
    ax_e.plot(
        time_index,
        fields["surface_temp_mean_c"],
        color="#2d5bff",
        linewidth=1.8,
        linestyle="--",
        label=r"$T_{mean}$",
    )

    for ax in (ax_a, ax_b, ax_c):
        ax.set_ylim(0, fields["heights_km"].max())
        ax.set_ylabel("Height (km)", fontsize=12)

    ax_d.set_ylabel("PS (hPa)", fontsize=12)
    ax_e.set_ylabel(r"T ($^\circ$C)", fontsize=12)
    ax_e.set_xlabel("Date", fontsize=12)

    ax_e.legend(loc="lower right", frameon=False, fontsize=11)

    for idx, ax in enumerate([ax_a, ax_b, ax_c, ax_d, ax_e]):
        ax.text(
            0.01,
            0.96,
            f"({chr(97 + idx)})",
            transform=ax.transAxes,
            fontsize=FONT_SIZE_PT,
            fontweight="normal",
            zorder=1000,
            clip_on=False,
            bbox=dict(facecolor="white", edgecolor="none", pad=0.2, alpha=0.85),
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlim(start, end)

        for boundary in boundaries:
            ax.axvline(boundary, color="black", linewidth=1.6)
        ax.axvline(end, color="black", linewidth=1.2, linestyle="--", alpha=0.9)

    add_period_annotations(ax_a, start, end, boundaries)

    ax_a.set_yticks(np.arange(0, fields["heights_km"].max() + 0.001, 0.25))
    ax_d.yaxis.set_major_locator(plt.MaxNLocator(4))
    ax_e.yaxis.set_major_locator(plt.MaxNLocator(4))

    day_locator = mdates.DayLocator(interval=2)
    date_formatter = mdates.DateFormatter("%m-%d")
    for ax in (ax_a, ax_b, ax_c, ax_d, ax_e):
        ax.xaxis.set_major_locator(day_locator)
        ax.xaxis.set_major_formatter(date_formatter)
        ax.tick_params(axis="both", labelsize=FONT_SIZE_PT)
    for ax in (ax_a, ax_b, ax_c, ax_d):
        ax.tick_params(labelbottom=False)

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def default_output_path(start: pd.Timestamp, end: pd.Timestamp) -> Path:
    return SCRIPT_DIR / f"mosaic_profile_overview_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}.png"


def main() -> None:
    args = parse_args()
    start, end = parse_window(args.start, args.end, args.freq)
    time_index, matched_df, pbl_df = load_windowed_data(
        matched_csv=args.matched_csv,
        pbl_csv=args.pbl_csv,
        start=start,
        end=end,
        freq=args.freq,
    )
    fields = prepare_fields(
        matched_df=matched_df,
        pbl_df=pbl_df,
        max_height_km=args.max_height_km,
    )
    output_path = args.output if args.output is not None else default_output_path(start, end)
    plot_overview(
        time_index=time_index,
        fields=fields,
        start=start,
        end=end,
        output_path=output_path,
    )
    print(f"Saved figure to: {output_path}")


if __name__ == "__main__":
    main()
