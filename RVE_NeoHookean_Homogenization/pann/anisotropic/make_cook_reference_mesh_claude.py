#!/usr/bin/env python3
"""Cook's membrane: reference (undeformed) mesh, standalone figure, with
a zoomed inset showing a handful of individual six-node quadratic
triangles (Tri6) explicitly -- the coarse cross-hatched look of the
full mesh at this zoom level is exactly two Tri6 sharing a diagonal per
coarse cell, not a rendering artifact. Real data from
Cook.gid/cook_results_certified_claude.npz (any tier has the same mesh).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.tri as mtri

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}\usepackage{bm}",
})

HERE = Path(__file__).resolve().parent
COOK_DIR = HERE.parent.parent / "Cook.gid"


def tri6_to_tri3(tris6: np.ndarray) -> np.ndarray:
    n0, n1, n2, n3, n4, n5 = (tris6[:, i] for i in range(6))
    return np.concatenate([
        np.stack([n0, n3, n5], axis=1),
        np.stack([n3, n1, n4], axis=1),
        np.stack([n5, n4, n2], axis=1),
        np.stack([n3, n4, n5], axis=1),
    ], axis=0)


def main() -> None:
    d = np.load(COOK_DIR / "cook_results_certified_claude.npz")
    coords, tris = d["coords"], d["tris"]
    tri3 = tri6_to_tri3(tris)
    triang = mtri.Triangulation(coords[:, 0], coords[:, 1], tri3)

    fig, (ax, ax_zoom) = plt.subplots(1, 2, figsize=(11.0, 5.2), gridspec_kw={"width_ratios": [1.5, 1.0]})

    ax.triplot(triang, color="#7f9fbf", linewidth=0.4)
    ax.set_title("Cook's membrane: reference (undeformed) mesh\n"
                  "512 six-node quadratic triangles (Tri6), 1\\,536 Gauss points", fontsize=12)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$X$ [m]")
    ax.set_ylabel(r"$Y$ [m]")

    zoom_x, zoom_y = (0.0, 9.0), (0.0, 11.25)
    ax.add_patch(plt.Rectangle((zoom_x[0], zoom_y[0]), zoom_x[1] - zoom_x[0], zoom_y[1] - zoom_y[0],
                                fill=False, edgecolor="#d62728", linewidth=1.3))

    ax_zoom.triplot(triang, color="#7f9fbf", linewidth=0.9)
    node_mask = (coords[:, 0] >= zoom_x[0]) & (coords[:, 0] <= zoom_x[1]) & \
                (coords[:, 1] >= zoom_y[0]) & (coords[:, 1] <= zoom_y[1])
    ax_zoom.plot(coords[node_mask, 0], coords[node_mask, 1], "o", color="#d62728", markersize=3.2, zorder=3)
    ax_zoom.set_xlim(zoom_x)
    ax_zoom.set_ylim(zoom_y)
    ax_zoom.set_aspect("equal")
    ax_zoom.set_title("Zoom: individual Tri6 elements\n(corner + mid-side nodes marked)", fontsize=11)
    ax_zoom.set_xlabel(r"$X$ [m]")
    for spine in ax_zoom.spines.values():
        spine.set_edgecolor("#d62728")
        spine.set_linewidth(1.3)

    fig.tight_layout()
    fig.savefig(HERE / "cook_reference_mesh_claude.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote cook_reference_mesh_claude.pdf")


if __name__ == "__main__":
    main()
