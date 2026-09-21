#!/usr/bin/env python3
"""Compare the material-A and material-B constitutive sample domains.

Only frozen strain coordinates and the archived A fit/validation split are
read. Display subsets are deterministic and never used for model selection.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
FIGURES = HERE / "figures"

COLORS = {"fit": "#255C82", "validation": "#D17A25", "test": "#794FA0"}
SHOW = {"fit": 240, "validation": 110, "test": 110}


def material_a():
    with np.load(PROJECT / "03_data/data.npz", allow_pickle=False) as data:
        training = data["E_train"].copy()
    with np.load(PROJECT / "06_pann/enrichment_results/flex_icnn_learn32/split.npz",
                 allow_pickle=False) as split:
        fit = training[split["fit"]]
        validation = training[split["validation"]]
    with np.load(PROJECT / "02_sampling/eval_sets.npz", allow_pickle=False) as heldout:
        test = heldout["test"].copy()
    with np.load(PROJECT / "02_sampling/train_grid.npz", allow_pickle=False) as grid:
        lower, upper = grid["blo"].copy(), grid["bhi"].copy()
        assert np.array_equal(training, grid["E"][1:])
    assert (len(fit), len(validation), len(test)) == (4208, 742, 400)
    return {"fit": fit, "validation": validation, "test": test}, lower, upper


def material_b():
    with np.load(PROJECT / "07_material_b/results/data_protocol_design_v1.npz",
                 allow_pickle=False) as design:
        sets = {role: design[f"E_{role}"].copy()
                for role in ("fit", "validation", "test")}
        lower, upper = design["lower"].copy(), design["upper"].copy()
    assert tuple(len(sets[key]) for key in ("fit", "validation", "test")) == (4200, 512, 512)
    return sets, lower, upper


def subset(points, count, seed):
    rng = np.random.default_rng(seed)
    return points[rng.choice(len(points), min(count, len(points)), replace=False)]


def draw_box(ax, lower, upper):
    corners = [(x, y, z) for x in (lower[0], upper[0])
               for y in (lower[1], upper[1]) for z in (lower[2], upper[2])]
    for i, a in enumerate(corners):
        for b in corners[i + 1:]:
            if sum(a[k] != b[k] for k in range(3)) == 1:
                ax.plot([a[0] * 100, b[0] * 100],
                        [a[1] * 100, b[1] * 100],
                        [a[2] * 100, b[2] * 100],
                        color="#555D64", linewidth=0.7, alpha=0.65)


def draw_panel(ax, sets, lower, upper, seed, label):
    # The same deterministic display rule is used for both materials.
    clouds, colors, sizes = [], [], []
    for role, size in (("fit", 7), ("validation", 19), ("test", 19)):
        points = subset(sets[role], SHOW[role], seed + list(SHOW).index(role)) * 100
        clouds.append(points)
        colors.extend([COLORS[role]] * len(points))
        sizes.extend([size] * len(points))
    points = np.concatenate(clouds)
    # One collection sorts all three roles together by viewing depth.
    ax.scatter(points[:, 0], points[:, 1], points[:, 2],
               s=sizes, c=colors, alpha=0.82, marker="o",
               depthshade=False, linewidths=0, rasterized=True)
    draw_box(ax, lower, upper)
    ax.text2D(0.11, 0.91, label, transform=ax.transAxes,
              fontsize=10, fontweight="bold")
    ax.set(xlim=(lower[0] * 100, upper[0] * 100),
           ylim=(lower[1] * 100, upper[1] * 100),
           zlim=(lower[2] * 100, upper[2] * 100))
    for axis, lo, hi in ((ax.xaxis, lower[0], upper[0]),
                         (ax.yaxis, lower[1], upper[1]),
                         (ax.zaxis, lower[2], upper[2])):
        ticks = sorted(set([lo * 100, hi * 100] + ([0] if lo < 0 < hi else [])))
        if axis is ax.yaxis:
            # Its lower-end label collides with the x-axis upper-end label.
            # Both exact box limits remain stated in the text.
            ticks = ticks[1:]
        axis.set_ticks(ticks)
        axis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.1f}"))
    ax.set_xlabel(r"$E_{11}$ [\%]", labelpad=-3)
    ax.set_ylabel(r"$E_{22}$ [\%]", labelpad=-3)
    ax.set_zlabel(r"$2E_{12}$ [\%]", labelpad=-3)
    ax.tick_params(labelsize=7, pad=-2)
    ax.set_box_aspect((1.0, 0.9, 0.85))
    ax.view_init(elev=21, azim=-57)
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor("white")
        axis.pane.set_edgecolor("white")


def main():
    plt.rcParams.update({"text.usetex": True,
                         "text.latex.preamble": r"\usepackage{lmodern}",
                         "font.family": "serif", "font.size": 9})
    FIGURES.mkdir(exist_ok=True)
    fig = plt.figure(figsize=(7.3, 3.75))
    for position, load, seed, label in ((121, material_a, 10, "Material A"),
                                       (122, material_b, 20, "Material B")):
        ax = fig.add_subplot(position, projection="3d")
        draw_panel(ax, *load(), seed, label)
        if position == 121:
            # A's 3D z label is otherwise overpainted by the neighboring axes.
            ax.set_zlabel("")
    fig.text(0.485, 0.62, r"$2E_{12}$ [\%]", rotation=90,
             ha="center", va="center")
    fig.subplots_adjust(left=0.00, right=0.95, bottom=0.13, top=1.04, wspace=0.04)
    legend = [Line2D([0], [0], marker="o", linestyle="none", color=COLORS["fit"],
                     markersize=5, label="Fit"),
              Line2D([0], [0], marker="o", linestyle="none", color=COLORS["validation"],
                     markersize=5, label="Validation"),
              Line2D([0], [0], marker="o", linestyle="none", color=COLORS["test"],
                     markersize=5, label="Test")]
    fig.legend(handles=legend, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.015))
    pdf = FIGURES / "constitutive_domains.pdf"
    fig.savefig(pdf, facecolor="white")
    plt.close(fig)
    subprocess.run(["pdftoppm", "-png", "-r", "220", "-singlefile",
                    str(pdf), str(FIGURES / "constitutive_domains")], check=True)

    # A compact, label-free rendering of Material A for the method overview.
    # It deliberately reuses Figure 7's data, display subset, colors, viewing
    # angle, and wireframe rather than introducing a second sampling graphic.
    sets, lower, upper = material_a()
    fig = plt.figure(figsize=(2.0, 1.8))
    ax = fig.add_subplot(111, projection="3d")
    clouds, colors, sizes = [], [], []
    for role, size in (("fit", 7), ("validation", 19), ("test", 19)):
        points = subset(sets[role], SHOW[role],
                        10 + list(SHOW).index(role)) * 100
        clouds.append(points)
        colors.extend([COLORS[role]] * len(points))
        sizes.extend([size] * len(points))
    points = np.concatenate(clouds)
    ax.scatter(points[:, 0], points[:, 1], points[:, 2],
               s=sizes, c=colors, alpha=0.82, marker="o",
               depthshade=False, linewidths=0, rasterized=True)
    draw_box(ax, lower, upper)
    ax.set(xlim=(lower[0] * 100, upper[0] * 100),
           ylim=(lower[1] * 100, upper[1] * 100),
           zlim=(lower[2] * 100, upper[2] * 100))
    ax.set_box_aspect((1.0, 0.9, 0.85))
    ax.view_init(elev=21, azim=-57)
    ax.set_axis_off()
    fig.subplots_adjust(0, 0, 1, 1)
    overview_pdf = FIGURES / "sampling_material_a_overview.pdf"
    fig.savefig(overview_pdf, transparent=True, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    subprocess.run(["pdftoppm", "-png", "-r", "220", "-singlefile",
                    str(overview_pdf),
                    str(FIGURES / "sampling_material_a_overview")], check=True)


if __name__ == "__main__":
    main()
