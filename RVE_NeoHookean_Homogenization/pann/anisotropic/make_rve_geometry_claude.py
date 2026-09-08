#!/usr/bin/env python3
"""RVE geometry, meshed: Fig. 2 of the paper. Regenerated from the
project's own mesh file (core/rve_geometry.mdpa) so the mesh-line color
matches Fig. 17's Cook reference-mesh figure (#7f9fbf) exactly, instead
of matplotlib's default gray -- same convention used throughout the
paper's other mesh figures (make_cook_reference_mesh_claude.py,
make_cook_deformed_field_claude.py, make_cook_vonmises_field_claude.py).

Parses the .mdpa directly (plain text, no Kratos import needed for a
static plot): node coordinates from "Begin Nodes", Tri6 connectivity
from "Begin Geometries Triangle2D6", and the Dirichlet boundary node
set from the "dirichlet" SubModelPart -- the same three pieces of
information core/fom_solver_rve.py itself reads from this file to build
the actual FE model.
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.tri as mtri

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
})

HERE = Path(__file__).resolve().parent
MDPA_PATH = HERE.parent.parent / "core" / "rve_geometry.mdpa"

MESH_COLOR = "#7f9fbf"  # same blue as make_cook_reference_mesh_claude.py
DIRICHLET_COLOR = "#d62728"


def parse_mdpa(path: Path):
    text = path.read_text()

    nodes_block = re.search(r"Begin Nodes\n(.*?)\nEnd Nodes", text, re.S).group(1)
    node_ids, coords = [], []
    for line in nodes_block.splitlines():
        parts = line.split()
        if not parts:
            continue
        node_ids.append(int(parts[0]))
        coords.append((float(parts[1]), float(parts[2])))
    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
    coords = np.asarray(coords, dtype=float)

    geom_block = re.search(r"Begin Geometries Triangle2D6.*?\n(.*?)\nEnd Geometries", text, re.S).group(1)
    tris = []
    for line in geom_block.splitlines():
        parts = line.split()
        if not parts:
            continue
        node_ids_of_tri = [id_to_idx[int(p)] for p in parts[1:7]]
        tris.append(node_ids_of_tri)
    tris = np.asarray(tris, dtype=int)

    dirichlet_block = re.search(
        r"Begin SubModelPart dirichlet.*?Begin SubModelPartNodes\n(.*?)\n\s*End SubModelPartNodes",
        text, re.S,
    ).group(1)
    dirichlet_ids = [int(x) for x in dirichlet_block.split()]
    dirichlet_idx = np.array([id_to_idx[nid] for nid in dirichlet_ids], dtype=int)

    return coords, tris, dirichlet_idx


def tri6_to_tri3(tris6: np.ndarray) -> np.ndarray:
    n0, n1, n2, n3, n4, n5 = (tris6[:, i] for i in range(6))
    return np.concatenate([
        np.stack([n0, n3, n5], axis=1),
        np.stack([n3, n1, n4], axis=1),
        np.stack([n5, n4, n2], axis=1),
        np.stack([n3, n4, n5], axis=1),
    ], axis=0)


def main() -> None:
    coords, tris, dirichlet_idx = parse_mdpa(MDPA_PATH)
    print(f"parsed {coords.shape[0]} nodes, {tris.shape[0]} Tri6 elements, "
          f"{dirichlet_idx.size} Dirichlet nodes")

    triang_fine = mtri.Triangulation(coords[:, 0], coords[:, 1], tri6_to_tri3(tris))

    fig, ax = plt.subplots(figsize=(5.4, 5.4))
    ax.triplot(triang_fine, color=MESH_COLOR, linewidth=0.35)
    ax.plot(coords[dirichlet_idx, 0], coords[dirichlet_idx, 1], "o",
            color=DIRICHLET_COLOR, markersize=3.6, zorder=3)
    ax.set_aspect("equal")
    ax.axis("off")

    fig.savefig(HERE / "rve_geometry_claude.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("wrote rve_geometry_claude.png")


if __name__ == "__main__":
    main()
