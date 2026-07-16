from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator


TRAIN_FILE = Path("train_val_predictions.csv")
MATCHED_FILE = Path("matched_all_data_10min_with_aeri-1.csv")
OUTPUT_FIGURE = Path("seasonal_diurnal_pbl_surface_temp_2hour.png")
OUTPUT_SUMMARY = Path("seasonal_diurnal_pbl_surface_temp_2hour_summary.csv")

TRAIN_TIME_COLUMN = "time"
TRAIN_SET_COLUMN = "dataset"
TRAIN_SET_VALUE = "train"
ACTUAL_PBL_COLUMN = "actual_pbl_height"
PREDICTED_PBL_COLUMN = "predicted_pbl_height"

MATCHED_TIME_COLUMN = "target_10min"
SURFACE_TEMP_COLUMNS = ["temp_1", "temp_2", "temp_3"]

SEASON_ORDER = ["Autumn", "Winter", "Spring", "Summer"]
SEASON_MAP = {
    12: "Winter",
    1: "Winter",
    2: "Winter",
    3: "Spring",
    4: "Spring",
    5: "Spring",
    6: "Summer",
    7: "Summer",
    8: "Summer",
    9: "Autumn",
    10: "Autumn",
    11: "Autumn",
}
MATLAB_COLORS = {
    "actual_pbl": "#0072BD",
    "predicted_pbl": "#D95319",
    "surface_temp": "#EDB120",
}
SEASON_AXIS_LIMITS = {
    "Autumn": {"pbl": (160, 240), "temp": (-15, -10)},
    "Winter": {"pbl": (120, 180), "temp": (-27, -23)},
    "Spring": {"pbl": (190, 300), "temp": (-22, -17)},
    "Summer": {"pbl": (120, 180), "temp": (-1, 1)},
}
BIN_HOURS = 2
FONT_SIZE_PT = 10.5


def apply_acp_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": FONT_SIZE_PT,
            "font.weight": "normal",
            "axes.labelsize": FONT_SIZE_PT,
            "axes.labelweight": "normal",
            "axes.titlesize": FONT_SIZE_PT,
            "axes.titleweight": "normal",
            "axes.linewidth": 0.8,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 3.5,
            "ytick.major.size": 3.5,
            "legend.fontsize": FONT_SIZE_PT,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "mathtext.fontset": "stix",
        }
    )


def load_train_data() -> pd.DataFrame:
    df = pd.read_csv(
        TRAIN_FILE,
        usecols=[
            TRAIN_TIME_COLUMN,
            TRAIN_SET_COLUMN,
            ACTUAL_PBL_COLUMN,
            PREDICTED_PBL_COLUMN,
        ],
    )
    df = df[df[TRAIN_SET_COLUMN] == TRAIN_SET_VALUE].copy()
    df[TRAIN_TIME_COLUMN] = pd.to_datetime(df[TRAIN_TIME_COLUMN], errors="coerce")
    df = df.dropna(subset=[TRAIN_TIME_COLUMN, ACTUAL_PBL_COLUMN, PREDICTED_PBL_COLUMN])
    return df.sort_values(TRAIN_TIME_COLUMN)


def load_matched_surface_temp() -> pd.DataFrame:
    df = pd.read_csv(MATCHED_FILE, usecols=[MATCHED_TIME_COLUMN, *SURFACE_TEMP_COLUMNS])
    df[MATCHED_TIME_COLUMN] = pd.to_datetime(df[MATCHED_TIME_COLUMN], errors="coerce")
    df = df.dropna(subset=[MATCHED_TIME_COLUMN, *SURFACE_TEMP_COLUMNS]).copy()
    df["surface_temp_3layer_mean"] = df[SURFACE_TEMP_COLUMNS].mean(axis=1)

    # A few timestamps appear twice in the matched table, so aggregate before merge.
    return (
        df.groupby(MATCHED_TIME_COLUMN, as_index=False)["surface_temp_3layer_mean"]
        .mean()
        .sort_values(MATCHED_TIME_COLUMN)
    )


def build_diurnal_summary() -> pd.DataFrame:
    train_df = load_train_data()
    matched_df = load_matched_surface_temp()

    merged = train_df.merge(
        matched_df,
        left_on=TRAIN_TIME_COLUMN,
        right_on=MATCHED_TIME_COLUMN,
        how="left",
    )
    merged = merged.dropna(subset=["surface_temp_3layer_mean"]).copy()

    merged["season"] = merged[TRAIN_TIME_COLUMN].dt.month.map(SEASON_MAP)
    merged["bin_start_hour"] = (merged[TRAIN_TIME_COLUMN].dt.hour // BIN_HOURS) * BIN_HOURS
    merged["bin_end_hour"] = merged["bin_start_hour"] + BIN_HOURS
    merged["time_bin"] = (
        merged["bin_start_hour"].astype(str).str.zfill(2)
        + ":00-"
        + merged["bin_end_hour"].astype(str).str.zfill(2)
        + ":00"
    )
    merged["hour_float"] = merged["bin_start_hour"] + BIN_HOURS / 2

    summary = (
        merged.groupby(["season", "bin_start_hour", "time_bin", "hour_float"], as_index=False)
        .agg(
            actual_pbl_mean=(ACTUAL_PBL_COLUMN, "mean"),
            predicted_pbl_mean=(PREDICTED_PBL_COLUMN, "mean"),
            surface_temp_mean=("surface_temp_3layer_mean", "mean"),
            sample_count=(ACTUAL_PBL_COLUMN, "size"),
        )
        .sort_values(["season", "bin_start_hour"])
    )

    summary["season"] = pd.Categorical(summary["season"], categories=SEASON_ORDER, ordered=True)
    return summary.sort_values(["season", "bin_start_hour"]).reset_index(drop=True)


def plot_summary(summary: pd.DataFrame) -> None:
    apply_acp_style()
    fig, axes = plt.subplots(2, 2, figsize=(14 / 2.54, 8 / 2.54), sharex=True)
    axes = axes.flatten()
    panel_labels = ["(a)", "(b)", "(c)", "(d)"]
    legend_handles = None
    legend_labels = None

    for ax, season, panel_label in zip(axes, SEASON_ORDER, panel_labels):
        season_df = summary[summary["season"] == season]
        if season_df.empty:
            ax.set_visible(False)
            continue

        ax.plot(
            season_df["hour_float"],
            season_df["actual_pbl_mean"],
            color=MATLAB_COLORS["actual_pbl"],
            linewidth=1.4,
            marker="o",
            markersize=3.6,
            markeredgewidth=0.6,
            label="Actual PBLH",
        )
        ax.plot(
            season_df["hour_float"],
            season_df["predicted_pbl_mean"],
            color=MATLAB_COLORS["predicted_pbl"],
            linewidth=1.4,
            linestyle="--",
            marker="s",
            markersize=3.4,
            markeredgewidth=0.6,
            label="Predicted PBLH",
        )
        ax.text(
            0.03,
            0.93,
            f"{panel_label} {season}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=FONT_SIZE_PT,
            fontweight="normal",
        )
        if season in {"Autumn", "Spring"}:
            ax.set_ylabel("ABLH (m)")
        else:
            ax.set_ylabel("")
        ax.set_ylim(*SEASON_AXIS_LIMITS[season]["pbl"])
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.tick_params(axis="y", colors=MATLAB_COLORS["actual_pbl"])
        ax.yaxis.label.set_color(MATLAB_COLORS["actual_pbl"])

        ax2 = ax.twinx()
        ax2.plot(
            season_df["hour_float"],
            season_df["surface_temp_mean"],
            color=MATLAB_COLORS["surface_temp"],
            linewidth=1.4,
            marker="^",
            markersize=3.8,
            markeredgewidth=0.6,
            label="Mean Surface Temp",
        )
        if season in {"Winter", "Summer"}:
            ax2.set_ylabel("Mean Surface Temp", color=MATLAB_COLORS["surface_temp"])
        else:
            ax2.set_ylabel("")
        ax2.set_ylim(*SEASON_AXIS_LIMITS[season]["temp"])
        ax2.yaxis.set_major_locator(MaxNLocator(5))
        ax2.tick_params(axis="y", colors=MATLAB_COLORS["surface_temp"])
        ax2.yaxis.label.set_color(MATLAB_COLORS["surface_temp"])

        ax.spines["left"].set_color(MATLAB_COLORS["actual_pbl"])
        ax.spines["bottom"].set_color("0.25")
        ax.spines["top"].set_color("0.25")
        ax.spines["right"].set_color("0.25")
        ax2.spines["right"].set_color(MATLAB_COLORS["surface_temp"])
        ax2.spines["top"].set_color("0.25")
        ax2.spines["left"].set_color("0.25")
        ax2.spines["bottom"].set_color("0.25")

        if legend_handles is None:
            handles1, labels1 = ax.get_legend_handles_labels()
            handles2, labels2 = ax2.get_legend_handles_labels()
            legend_handles = handles1 + handles2
            legend_labels = labels1 + labels2

    for ax in axes[-2:]:
        ax.set_xlabel("")

    tick_positions = [0, 6, 12, 18]
    tick_labels = ["0", "6", "12", "18"]
    for ax in axes:
        if ax.get_visible():
            ax.set_xlim(0, 24)
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels)

    if legend_handles is not None:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            ncol=3,
            frameon=False,
            bbox_to_anchor=(0.5, 0.01),
        )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(OUTPUT_FIGURE)


def main() -> None:
    summary = build_diurnal_summary()
    summary.to_csv(OUTPUT_SUMMARY, index=False)
    plot_summary(summary)
    print(f"Saved figure: {OUTPUT_FIGURE}")
    print(f"Saved summary: {OUTPUT_SUMMARY}")


if __name__ == "__main__":
    main()
