#!/usr/bin/env python3
"""Export the actual HPROM-ANN residual and stress supports as vector EPS files."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection, PolyCollection


HERE = Path(__file__).resolve().parent
RVE_DIR = HERE.parent
MESH_FILE = RVE_DIR / "rve_geometry.mdpa"
ECM_FILE = RVE_DIR / (
    "stage_12_hprom_ann_mawecm_res_eps_sig_phase1to40_phase2to10_sum990_ann/"
    "ecm_weights_all.npz"
)


def load_mesh(path):
    nodes = {}
    triangles = []
    section = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line == "Begin Nodes":
            section = "nodes"
            continue
        if line == "End Nodes":
            section = None
            continue
        if line.startswith("Begin Geometries Triangle2D6"):
            section = "triangles"
            continue
        if line == "End Geometries":
            section = None
            continue
        if not line or line.startswith("//"):
            continue
        fields = line.split()
        if section == "nodes" and len(fields) >= 4:
            nodes[int(fields[0])] = np.array([float(fields[1]), float(fields[2])])
        elif section == "triangles" and len(fields) >= 7:
            triangles.append(tuple(int(value) for value in fields[1:4]))
    return nodes, triangles


def export_support(nodes, triangles, selected, color, output, show_gauss_points=False):
    selected = set(int(value) for value in np.asarray(selected).reshape(-1))
    polygons = [np.vstack([nodes[node_id] for node_id in tri]) for tri in triangles]
    edges = [np.vstack([poly, poly[0]]) for poly in polygons]
    selected_polygons = [polygons[index] for index in sorted(selected)]

    fig, ax = plt.subplots(figsize=(4.0, 4.0))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.add_collection(LineCollection(edges, colors="#D5DADF", linewidths=0.28, zorder=1))
    ax.add_collection(
        PolyCollection(
            selected_polygons,
            facecolors=color,
            edgecolors="#23303D",
            linewidths=0.75,
            zorder=3,
        )
    )

    if show_gauss_points:
        barycentric = np.array(
            [
                [2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0],
                [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0],
                [1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0],
            ]
        )
        gauss_points = np.vstack([barycentric @ poly for poly in selected_polygons])
        ax.scatter(
            gauss_points[:, 0],
            gauss_points[:, 1],
            s=5.0,
            c="#17202A",
            marker="o",
            linewidths=0.0,
            zorder=4,
        )

    ax.set_xlim(-1.015, 1.015)
    ax.set_ylim(-1.015, 1.015)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.savefig(output, format="eps", bbox_inches="tight", pad_inches=0.01, facecolor="white")
    plt.close(fig)


def main():
    nodes, triangles = load_mesh(MESH_FILE)
    ecm = np.load(ECM_FILE, allow_pickle=True)
    export_support(
        nodes,
        triangles,
        ecm["Z_res"],
        "#E76F2E",
        HERE / "residual_support_Zres.eps",
    )
    export_support(
        nodes,
        triangles,
        ecm["Z_sig"],
        "#2474C6",
        HERE / "stress_support_Zsigma.eps",
        show_gauss_points=True,
    )


if __name__ == "__main__":
    main()
