#!/usr/bin/env python3
"""Draw the material-A and material-B cells in one manuscript figure style.

Only geometry and meshes are read. All figure lettering is rendered by LaTeX;
experiment descriptions and mesh counts belong in the manuscript text.
"""

from __future__ import annotations

import json
import subprocess
import sys
from math import cos, pi, radians, sin, sqrt
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from matplotlib.patches import Ellipse, Rectangle


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
FIGURES = HERE / "figures"
sys.path.insert(0, str(PROJECT))
import config as material_a  # noqa: E402


def load_a_mesh():
    nodes, triangles = {}, []
    section = None
    for line in (PROJECT / "03_data/rve_mesh.mdpa").read_text().splitlines():
        if line.startswith("Begin Nodes"):
            section = "nodes"
            continue
        if line.startswith("Begin Geometries Triangle2D6"):
            section = "triangles"
            continue
        if line.startswith("Begin Elements"):
            section = "elements"
            continue
        if line.startswith("End "):
            section = None
            continue
        fields = line.split()
        if section == "nodes" and len(fields) == 4:
            nodes[int(fields[0])] = [float(value) for value in fields[1:3]]
        elif section == "triangles" and len(fields) == 7:
            triangles.append([int(value) for value in fields[1:7]])
        elif section == "elements" and len(fields) >= 8:
            triangles.append([int(value) for value in fields[2:8]])
    ids = list(nodes)
    index = {node_id: i for i, node_id in enumerate(ids)}
    xy = np.asarray([nodes[node_id] for node_id in ids])
    connectivity = np.asarray([[index[node_id] for node_id in row] for row in triangles])
    assert len(connectivity) == 1546
    return xy, connectivity


def load_b_mesh():
    path = PROJECT / "07_material_b/preflight_reference_v1/reference.npz"
    with np.load(path, allow_pickle=False) as mesh:
        xy = mesh["xy"].copy()
        connectivity = mesh["triangles"].copy()
    assert len(connectivity) == 4621
    return xy, connectivity


def cavities_a():
    a, b = material_a.ellipse_semi_axes()
    return [(0.0, 0.0, a, b, material_a.ELLIPSE_ANGLE_DEG)]


def cavities_b():
    spec = json.loads((PROJECT / "07_material_b/pilot_spec.json").read_text())
    assert spec["cell_side"] == material_a.CELL_SIDE == 2.0
    assert abs(sum(item["area_over_cell"] for item in spec["cavities"]) - 0.2) < 1e-12
    cavities = []
    for item in spec["cavities"]:
        area = spec["cell_side"] ** 2 * item["area_over_cell"]
        minor = sqrt(area / (pi * item["aspect"]))
        major = item["aspect"] * minor
        x, y = 2 * np.asarray(item["center_over_L"])
        cavities.append((x, y, major, minor, item["angle_deg"]))
    return cavities


def draw_cell(xy, connectivity, cavities, output):
    side = material_a.CELL_SIDE
    fig, (geometry_ax, mesh_ax) = plt.subplots(1, 2, figsize=(7.0, 3.35))
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.17, top=0.98, wspace=0.21)
    triangles = mtri.Triangulation(xy[:, 0] / side, xy[:, 1] / side, connectivity[:, :3])
    mesh_ax.triplot(triangles, color="#4F6A80", linewidth=0.21, alpha=0.8)
    geometry_ax.add_patch(Rectangle((-0.5, -0.5), 1, 1,
                                    facecolor="#DCE8F1", edgecolor="#334155",
                                    linewidth=0.9))
    for x, y, major, minor, angle in cavities:
        cx, cy = x / side, y / side
        geometry_ax.add_patch(Ellipse((cx, cy),
                                      2 * major / side, 2 * minor / side,
                                      angle=angle, facecolor="white",
                                      edgecolor="#334155", linewidth=1.1))
        theta = radians(angle)
        ux, uy = cos(theta), sin(theta)
        reach = 0.72 * major / side
        geometry_ax.plot([cx - reach * ux, cx + reach * ux],
                         [cy - reach * uy, cy + reach * uy],
                         "--", color="#B85A3C", linewidth=0.8)
        geometry_ax.text(cx - 0.05 * uy, cy + 0.05 * ux,
                         rf"${angle:g}^\circ$", ha="center", va="center",
                         color="#A3482C", fontsize=8,
                         bbox={"facecolor": "white", "edgecolor": "none",
                               "alpha": 0.9, "pad": 0.2})
    for ax in (mesh_ax, geometry_ax):
        ax.set(xlim=(-0.54, 0.54), ylim=(-0.54, 0.54), aspect="equal")
    mesh_ax.set_xticks([-0.5, 0, 0.5], [r"$-\frac12$", "$0$", r"$\frac12$"])
    mesh_ax.set_yticks([-0.5, 0, 0.5], [r"$-\frac12$", "$0$", r"$\frac12$"])
    mesh_ax.set_xlabel(r"$X_1/\ell$")
    mesh_ax.set_ylabel(r"$X_2/\ell$")
    geometry_ax.axis("off")
    pdf = FIGURES / f"{output}.pdf"
    fig.savefig(pdf, facecolor="white")
    plt.close(fig)
    subprocess.run(["pdftoppm", "-png", "-r", "220", "-singlefile",
                    str(pdf), str(FIGURES / output)], check=True)


def main():
    plt.rcParams.update({"text.usetex": True,
                         "text.latex.preamble": r"\usepackage{lmodern}",
                         "font.family": "serif", "font.size": 9,
                         "axes.labelsize": 9, "xtick.labelsize": 8,
                         "ytick.labelsize": 8})
    FIGURES.mkdir(exist_ok=True)
    draw_cell(*load_a_mesh(), cavities_a(), "rve_geometry")
    draw_cell(*load_b_mesh(), cavities_b(), "rve_geometry_b")


if __name__ == "__main__":
    main()
