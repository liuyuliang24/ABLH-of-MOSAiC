import argparse
import os
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_DATA_HOME", "/tmp")

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import numpy as np
from matplotlib.collections import LineCollection
from netCDF4 import Dataset

try:
    import cartopy
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
except ModuleNotFoundError as exc:
    raise SystemExit(
        "cartopy is required to draw the polar map. Install it with "
        "`pip install cartopy` or `conda install -c conda-forge cartopy`."
    ) from exc

cartopy.config["data_dir"] = "/tmp/cartopy"


DEFAULT_DATA_DIR = Path("/media/lyl/DATA11/THz_band/THz_data/ARM/262064/mossondewnpnM1.b1")
DEFAULT_OUTPUT = Path("mosaic_ship_track_75N.svg")
FIGURE_SIZE_CM = 7.0


plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 6.5,
    "axes.titlesize": 7.5,
    "axes.labelsize": 6.5,
    "xtick.labelsize": 6.0,
    "ytick.labelsize": 6.0,
    "legend.fontsize": 6.0,
    "svg.fonttype": "none",
})


def normalize_longitude(lon: float) -> float:
    """Convert longitude to [-180, 180)."""
    return ((lon + 180.0) % 360.0) - 180.0


def format_longitude_label(lon: float) -> str:
    lon = normalize_longitude(lon)
    if np.isclose(lon, 0.0):
        return "0°"
    if np.isclose(abs(lon), 180.0):
        return "180°"
    suffix = "E" if lon > 0 else "W"
    return f"{abs(int(round(lon)))}°{suffix}"


def parse_launch_time_from_name(cdf_path: Path) -> datetime:
    parts = cdf_path.stem.split(".")
    return datetime.strptime("".join(parts[-2:]), "%Y%m%d%H%M%S")


def extract_ship_track(data_dir: Path, min_latitude: float = 75.0):
    """
    Extract the ship position from each sounding file using the first valid lat/lon.
    """
    rows = []
    for cdf_path in sorted(data_dir.glob("*.cdf")):
        with Dataset(cdf_path) as ds:
            lat = np.asarray(ds.variables["lat"][:], dtype=float)
            lon = np.asarray(ds.variables["lon"][:], dtype=float)
            valid = np.isfinite(lat) & np.isfinite(lon)
            if not valid.any():
                continue

            first_idx = np.flatnonzero(valid)[0]
            launch_lat = float(lat[first_idx])
            launch_lon = normalize_longitude(float(lon[first_idx]))
            if launch_lat < min_latitude:
                continue

            launch_time = parse_launch_time_from_name(cdf_path)

            rows.append((launch_time, launch_lat, launch_lon, cdf_path.name))

    if not rows:
        raise ValueError(f"No launch positions found above {min_latitude}°N in {data_dir}")

    rows.sort(key=lambda item: item[0])
    times = [row[0] for row in rows]
    lats = np.array([row[1] for row in rows], dtype=float)
    lons = np.array([row[2] for row in rows], dtype=float)
    names = [row[3] for row in rows]
    return times, lats, lons, names


def build_polar_axes(min_latitude: float = 75.0,
                     dmeridian: float = 30.0,
                     dparallel: float = 5.0):
    figure_size_in = FIGURE_SIZE_CM / 2.54
    fig = plt.figure(figsize=(figure_size_in, figure_size_in))
    ax = fig.add_axes([0.08, 0.18, 0.84, 0.74], projection=ccrs.NorthPolarStereo())

    ax.set_extent([-180, 180, min_latitude, 90], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor="#d9d9d9", edgecolor="none", zorder=0)
    ax.coastlines(linewidth=0.6, zorder=1)

    meridians = np.arange(-180, 180 + dmeridian, dmeridian)
    parallels = np.arange(min_latitude, 90 + dparallel, dparallel)
    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        xlocs=meridians,
        ylocs=parallels,
        linestyle="--",
        linewidth=0.65,
        color="0.35",
        alpha=0.6,
    )
    gl.n_steps = 180

    theta = np.linspace(0, 2 * np.pi, 361)
    center, radius = np.array([0.5, 0.5]), 0.5
    circle = mpath.Path(np.column_stack([np.sin(theta), np.cos(theta)]) * radius + center)
    ax.set_boundary(circle, transform=ax.transAxes)
    ax.spines["geo"].set_visible(False)

    # Cover any clipped stroke artifacts outside the circular frame.
    cleanup_ring = mpatches.Circle(
        center,
        radius,
        transform=ax.transAxes,
        facecolor="none",
        edgecolor="white",
        linewidth=2.4,
        zorder=5.2,
        clip_on=False,
    )
    outline_ring = mpatches.Circle(
        center,
        radius,
        transform=ax.transAxes,
        facecolor="none",
        edgecolor="0.35",
        linewidth=0.6,
        zorder=5.3,
        clip_on=False,
    )
    ax.add_patch(cleanup_ring)
    ax.add_patch(outline_ring)

    # Place longitude labels just outside the circular frame.
    label_latitude = min_latitude - 0.65
    for lon in meridians:
        if np.isclose(abs(lon), 180.0):
            va, ha = "bottom", "center"
        elif np.isclose(lon, 0.0):
            va, ha = "top", "center"
        elif 0.0 < lon < 180.0:
            va, ha = "center", "left"
        else:
            va, ha = "center", "right"

        x, y = ax.projection.transform_point(lon, label_latitude, ccrs.Geodetic())
        ax.text(
            x,
            y,
            format_longitude_label(lon),
            color="#1f5c42",
            fontsize=5.8,
            va=va,
            ha=ha,
            clip_on=False,
            zorder=6,
        )

    latitude_label_lon = 180.0
    for lat in parallels:
        if lat <= min_latitude or lat >= 90.0:
            continue
        x, y = ax.projection.transform_point(latitude_label_lon, lat, ccrs.Geodetic())
        ax.text(
            x,
            y,
            f"{int(round(lat))}°N",
            color="#8e2b2b",
            fontsize=5.8,
            va="center",
            ha="right",
            clip_on=False,
            zorder=6,
        )

    return fig, ax


def add_time_colored_track(ax, lons: np.ndarray, lats: np.ndarray, times):
    time_numbers = mdates.date2num(times)
    projected = ax.projection.transform_points(ccrs.PlateCarree(), lons, lats)
    xy = projected[:, :2]

    segments = np.stack([xy[:-1], xy[1:]], axis=1)
    segment_values = 0.5 * (time_numbers[:-1] + time_numbers[1:])

    ax.plot(
        lons,
        lats,
        color="0.75",
        linewidth=0.8,
        transform=ccrs.PlateCarree(),
        zorder=2,
    )

    collection = LineCollection(
        segments,
        cmap="viridis",
        norm=plt.Normalize(time_numbers.min(), time_numbers.max()),
        linewidth=1.6,
        capstyle="round",
        joinstyle="round",
        zorder=3,
    )
    collection.set_array(segment_values)
    ax.add_collection(collection)

    scatter = ax.scatter(
        lons,
        lats,
        c=time_numbers,
        cmap="viridis",
        s=6,
        linewidth=0.0,
        transform=ccrs.PlateCarree(),
        zorder=4,
    )
    return collection, scatter, time_numbers


def add_time_colorbar(fig, artist, times):
    cax = fig.add_axes([0.22, 0.08, 0.56, 0.025])
    colorbar = fig.colorbar(artist, cax=cax, orientation="horizontal")
    tick_times = [times[0], times[len(times) // 2], times[-1]]
    tick_values = mdates.date2num(tick_times)
    colorbar.set_ticks(tick_values)
    colorbar.set_ticklabels([item.strftime("%Y-%m") for item in tick_times])
    colorbar.outline.set_linewidth(0.45)
    colorbar.ax.tick_params(length=2, pad=1)


def plot_ship_track(data_dir: Path,
                    output_path: Path,
                    min_latitude: float = 75.0,
                    dmeridian: float = 30.0,
                    dparallel: float = 5.0):
    times, lats, lons, _ = extract_ship_track(data_dir, min_latitude=min_latitude)
    fig, ax = build_polar_axes(
        min_latitude=min_latitude,
        dmeridian=dmeridian,
        dparallel=dparallel,
    )

    track_artist, _, _ = add_time_colored_track(ax, lons, lats, times)
    ax.scatter(
        lons[0],
        lats[0],
        s=22,
        color="#1f77b4",
        edgecolor="white",
        linewidth=0.5,
        transform=ccrs.PlateCarree(),
        zorder=5,
    )
    ax.scatter(
        lons[-1],
        lats[-1],
        s=22,
        color="#111111",
        edgecolor="white",
        linewidth=0.5,
        transform=ccrs.PlateCarree(),
        zorder=5,
    )
    add_time_colorbar(fig, track_artist, times)

    if getattr(fig.canvas, "manager", None) is not None:
        fig.canvas.manager.set_window_title("北极科考船航迹图")
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    print(f"Saved figure to: {output_path.resolve()}")
    print(f"Track points used: {len(times)}")
    print(f"Latitude range: {lats.min():.2f} to {lats.max():.2f} °N")
    print(f"Longitude range: {lons.min():.2f} to {lons.max():.2f} °")


def parse_args():
    parser = argparse.ArgumentParser(description="Plot Arctic ship track from MOSAiC sounding files.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Directory containing MOSAiC .cdf files.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output image path.")
    parser.add_argument("--min-latitude", type=float, default=75.0, help="Minimum latitude to display and retain.")
    parser.add_argument("--dmeridian", type=float, default=30.0, help="Longitude grid interval in degrees.")
    parser.add_argument("--dparallel", type=float, default=5.0, help="Latitude grid interval in degrees.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    plot_ship_track(
        data_dir=args.data_dir,
        output_path=args.output,
        min_latitude=args.min_latitude,
        dmeridian=args.dmeridian,
        dparallel=args.dparallel,
    )
