#!/usr/bin/env python3
"""Cook's membrane: reference (undeformed) mesh, standalone figure, with
a zoomed inset showing a handful of individual six-node quadratic
triangles (Tri6) explicitly. The main panel plots only the CORNER-node
triangulation (tris[:, :3]) -- each Tri6's own mid-side node would
otherwise add a visible line at its own edge midpoint, making an nx=8
mesh look like nx=16 in a naive plot of the full 6-node connectivity
(confirmed directly: 9 corner nodes along the left edge, 8 gaps, but 17
nodes total once the 8 mid-side nodes are included). The zoom inset
uses the full 6-node connectivity deliberately, to show those same
mid-side nodes explicitly -- that is the one place the finer,
cross-hatched look is real, not a rendering artifact. Real data from
fe2_extension/cook_results_certified_nx8_claude.npz (any tier has the
same mesh): the nx=ny=8 resolution used throughout the paper's Cook's
membrane section, not this project's finer, more usual density.
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
COOK_DIR = HERE.parent.parent / "fe2_extension"


def tri6_to_tri3(tris6: np.ndarray) -> np.ndarray:
    n0, n1, n2, n3, n4, n5 = (tris6[:, i] for i in range(6))
    return np.concatenate([
        np.stack([n0, n3, n5], axis=1),
        np.stack([n3, n1, n4], axis=1),
        np.stack([n5, n4, n2], axis=1),
        np.stack([n3, n4, n5], axis=1),
    ], axis=0)


def main() -> None:
    d = np.load(COOK_DIR / "cook_results_certified_nx8_claude.npz")
    coords, tris = d["coords"], d["tris"]
    triang_coarse = mtri.Triangulation(coords[:, 0], coords[:, 1], tris[:, :3])
    triang_fine = mtri.Triangulation(coords[:, 0], coords[:, 1], tri6_to_tri3(tris))

    fig, (ax, ax_zoom) = plt.subplots(1, 2, figsize=(11.0, 5.2), gridspec_kw={"width_ratios": [1.5, 1.0]})

    ax.triplot(triang_coarse, color="#7f9fbf", linewidth=0.4)
    ax.set_title("Cook's membrane: reference (undeformed) mesh\n"
                  "128 six-node quadratic triangles (Tri6), 384 Gauss points", fontsize=12)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$X$ [m]")
    ax.set_ylabel(r"$Y$ [m]")

    zoom_x, zoom_y = (0.0, 18.0), (0.0, 22.5)
    ax.add_patch(plt.Rectangle((zoom_x[0], zoom_y[0]), zoom_x[1] - zoom_x[0], zoom_y[1] - zoom_y[0],
                                fill=False, edgecolor="#d62728", linewidth=1.3))

    ax_zoom.triplot(triang_fine, color="#7f9fbf", linewidth=0.9)
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
