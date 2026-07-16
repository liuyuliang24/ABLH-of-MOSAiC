#!/usr/bin/env python3
"""Draw a compact schematic diagram for the PBLH neural-network retrieval model."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
OUT_DIR = PROJECT_DIR / "model_schematic_results"
MPL_DIR = OUT_DIR / ".mplconfig"
MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUT_DIR.mkdir(exist_ok=True)

COLORS = {
    "input": "#CFE3F2",
    "prep": "#DDECD2",
    "encoder": "#F9E7B2",
    "fusion": "#F3D4CD",
    "head": "#D9D2EE",
    "loss": "#DDE5EA",
    "edge": "#35424F",
    "text": "#1F2933",
    "muted": "#5B6673",
}

FONT_FAMILY = "Times New Roman"
FONT_SIZE_PT = 10.5
FIG_WIDTH_CM = 18.0
FIG_HEIGHT_CM = 14.0
FIG_WIDTH_IN = FIG_WIDTH_CM / 2.54
FIG_HEIGHT_IN = FIG_HEIGHT_CM / 2.54
X_MAX = FIG_WIDTH_CM
Y_MAX = FIG_HEIGHT_CM


def box(ax, x, y, w, h, text, face, fontsize=FONT_SIZE_PT, lw=1.1, bold=False):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.025,rounding_size=0.10",
        linewidth=lw,
        edgecolor="#536170",
        facecolor=face,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=COLORS["text"],
        fontweight="bold" if bold else "normal",
        linespacing=1.04,
    )
    return patch


def arrow(ax, xy1, xy2, rad=0.0, lw=1.05, ms=10):
    patch = FancyArrowPatch(
        xy1,
        xy2,
        arrowstyle="-|>",
        mutation_scale=ms,
        linewidth=lw,
        color=COLORS["edge"],
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=2,
        shrinkB=2,
    )
    ax.add_patch(patch)
    return patch


def polyline(ax, points, lw=1.05):
    xs, ys = zip(*points)
    ax.plot(xs, ys, color=COLORS["edge"], lw=lw, solid_capstyle="round")


def poly_arrow(ax, points, lw=1.05, ms=10):
    if len(points) < 2:
        return
    if len(points) > 2:
        polyline(ax, points[:-1], lw=lw)
    arrow(ax, points[-2], points[-1], lw=lw, ms=ms)


def main() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "font.family": FONT_FAMILY,
            "font.size": FONT_SIZE_PT,
            "axes.linewidth": 0,
            "savefig.facecolor": "white",
        }
    )

    fig = plt.figure(figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, X_MAX)
    ax.set_ylim(0, Y_MAX)
    ax.set_aspect("equal")
    ax.axis("off")

    # ------------------------------------------------------------------
    # Inputs and encoded features
    # ------------------------------------------------------------------
    left_margin = 1.0
    right_margin = 1.0
    input_x = left_margin
    input_w = 3.55
    mid_gap = 0.40
    prep_x = input_x + input_w + mid_gap
    prep_w = 2.55
    enc_x = prep_x + prep_w + mid_gap
    enc_w = 2.55
    top_box_h = 1.18
    row_gap = 0.34
    top_row_y = 12.20
    row_y = [top_row_y - idx * (top_box_h + row_gap) for idx in range(5)]
    y_centers = [y + top_box_h / 2 for y in row_y]

    bus_x = enc_x + enc_w + 0.50
    encoded_w = 4.20
    encoded_x = X_MAX - right_margin - encoded_w
    encoded_y = row_y[-1]
    encoded_h = (row_y[0] + top_box_h) - row_y[-1]

    inputs = [
        "HATPRO +\nMiRAC-P",
        "AERI",
        "Ceilometer",
        "u/v wind",
        "others",
    ]
    preps = [
        "Quality\ncontrol",
        "Radiance\nscaling",
        "clip + log",
        "Wind\ndescriptors",
        "Condition\nlabel",
    ]
    encoders = [
        "Conv1D\n+ GELU",
        "GELU",
        "Conv1D\n+ GELU",
        "Conv1D\n+ GELU",
        "Linear\n+ GELU",
    ]

    for y, inp, prep, enc in zip(row_y, inputs, preps, encoders):
        box(ax, input_x, y, input_w, top_box_h, inp, COLORS["input"])
        box(ax, prep_x, y, prep_w, top_box_h, prep, COLORS["prep"])
        box(ax, enc_x, y, enc_w, top_box_h, enc, COLORS["encoder"])
        arrow(ax, (input_x + input_w, y + top_box_h / 2), (prep_x, y + top_box_h / 2), lw=0.90, ms=8)
        arrow(ax, (prep_x + prep_w, y + top_box_h / 2), (enc_x, y + top_box_h / 2), lw=0.90, ms=8)
        polyline(ax, [(enc_x + enc_w, y + top_box_h / 2), (bus_x, y + top_box_h / 2)], lw=0.95)

    polyline(ax, [(bus_x, y_centers[-1]), (bus_x, y_centers[0])], lw=0.95)
    mid_y = y_centers[2]
    arrow(ax, (bus_x, mid_y), (encoded_x, mid_y), lw=0.95, ms=9)

    box(
        ax,
        encoded_x,
        encoded_y,
        encoded_w,
        encoded_h,
        "Encoded features\n"
        "HATPRO / MiRAC-P: 48\n"
        "AERI: 40\n"
        "Ceilo: 48+32+32\n"
        "Wind: 48\n"
        "Physics: 40+16+16",
        "#F7F7F7",
        bold=True,
    )

    # ------------------------------------------------------------------
    # Fusion trunk and prediction heads
    # ------------------------------------------------------------------
    trunk_y = 3.55
    trunk_h = 1.18
    trunk_w = 2.88
    trunk_gap = 0.40
    trunk_boxes = [
        (1.00, "Fusion\n256 + GELU"),
        (1.00 + (trunk_w + trunk_gap), "Weather FiLM\nTanh"),
        (1.00 + 2 * (trunk_w + trunk_gap), "Residual\nBlock x2"),
        (1.00 + 3 * (trunk_w + trunk_gap), "Latent\n96 + GELU"),
        (1.00 + 4 * (trunk_w + trunk_gap), "Experts\n3 heads"),
    ]
    for x, text in trunk_boxes:
        box(ax, x, trunk_y, trunk_w, trunk_h, text, COLORS["fusion"])
    for (x1, _), (x2, _) in zip(trunk_boxes[:-1], trunk_boxes[1:]):
        arrow(ax, (x1 + trunk_w, trunk_y + trunk_h / 2), (x2, trunk_y + trunk_h / 2), lw=0.98, ms=9)

    fusion_x = trunk_boxes[0][0]
    fusion_cx = fusion_x + trunk_w / 2
    encoded_cx = encoded_x + encoded_w / 2
    connector_y = 4.95
    poly_arrow(
        ax,
        [
            (encoded_cx, encoded_y),
            (encoded_cx, connector_y),
            (fusion_cx, connector_y),
            (fusion_cx, trunk_y + trunk_h),
        ],
        lw=0.95,
        ms=8,
    )

    head_y = 1.30
    head_h = 1.18
    train_x, train_w = 1.00, 5.20
    final_x, final_w = 8.10, 3.10
    output_x, output_w = trunk_boxes[-1][0], trunk_w

    box(ax, output_x, head_y, output_w, head_h, "Correction\nheads", COLORS["head"])
    box(ax, final_x, head_y, final_w, head_h, "Final ABLH\n(m)", COLORS["head"], bold=True)
    box(ax, train_x, head_y, train_w, head_h, "Loss\nHuber + MSE", COLORS["loss"])

    expert_cx = trunk_boxes[-1][0] + trunk_w / 2
    arrow(ax, (expert_cx, trunk_y), (expert_cx, head_y + head_h), rad=0.0, lw=0.95, ms=8)
    arrow(ax, (output_x, head_y + head_h / 2), (final_x + final_w, head_y + head_h / 2), rad=0.0, lw=0.95, ms=8)
    arrow(ax, (final_x, head_y + head_h / 2), (train_x + train_w, head_y + head_h / 2), rad=0.0, lw=0.95, ms=8)

    for suffix in ("png", "svg"):
        out_path = OUT_DIR / f"pbl_model_schematic_compact.{suffix}"
        fig.savefig(out_path, dpi=300)
        print(f"Saved: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
