#!/usr/bin/env python3
"""Cook's membrane downstream demonstration: deformed configuration,
colored by displacement magnitude, for all four PANN tiers. Clean 2x2
grid (all four panels the same shape/size), read left-to-right/
top-to-bottom in the paper's own tier order (regression tier 1, free
tier 2, polyconvex ICNN tier 3a, polyconvex ICKAN tier 3b) -- uncertified
tiers on top, certified tiers on bottom. The undeformed reference mesh
is a separate figure (make_cook_reference_mesh_claude.py). Colors match
every other tier comparison in the paper (make_full_comparison_claude.py).
Real data from Cook.gid/cook_results_*_claude.npz (run_cook_pann_claude.py).
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

TIER_ORDER = ("regression", "free", "certified", "ickan")
TITLES = {
    "regression": "Regression (tier 1), stalled at 5\\% of load",
    "free": "Free hyperelastic (tier 2), stalled at 5\\% of load",
    "certified": "Polyconvex ICNN (tier 3a), full converged load",
    "ickan": "Polyconvex ICKAN (tier 3b), full converged load",
}


def tri6_to_tri3(tris6: np.ndarray) -> np.ndarray:
    """Subdivide each 6-node quadratic triangle into 4 linear sub-triangles
    for flat/smooth matplotlib rendering: (0,3,5),(3,1,4),(5,4,2),(3,4,5)."""
    n0, n1, n2, n3, n4, n5 = (tris6[:, i] for i in range(6))
    return np.concatenate([
        np.stack([n0, n3, n5], axis=1),
        np.stack([n3, n1, n4], axis=1),
        np.stack([n5, n4, n2], axis=1),
        np.stack([n3, n4, n5], axis=1),
    ], axis=0)


def load(which: str):
    d = np.load(COOK_DIR / f"cook_results_{which}_claude.npz")
    return d["coords"], d["tris"], d["u_nodal"]


def main() -> None:
    fields = []
    vmax = 0.0
    for which in TIER_ORDER:
        coords, tris, u_nodal = load(which)
        mag = np.linalg.norm(u_nodal, axis=1)
        vmax = max(vmax, float(mag.max()))
        fields.append((coords, tris, u_nodal, mag))

    fig, axes = plt.subplots(2, 2, figsize=(10.4, 10.0))

    for ax, which, (coords, tris, u_nodal, mag) in zip(axes.flat, TIER_ORDER, fields):
        deformed = coords + u_nodal
        tri3 = tri6_to_tri3(tris)
        triang = mtri.Triangulation(deformed[:, 0], deformed[:, 1], tri3)
        tpc = ax.tripcolor(triang, mag, shading="gouraud", cmap="viridis", vmin=0.0, vmax=vmax)
        ax.triplot(triang, color="white", linewidth=0.05, alpha=0.4)
        ax.set_title(TITLES[which], fontsize=11.2)
        ax.set_aspect("equal")
        ax.set_xlabel(r"$x$ [m]")
        ax.set_ylabel(r"$y$ [m]")
        cbar = fig.colorbar(tpc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(r"$\|\bm u\|$ [m]", fontsize=9.5)

    fig.suptitle("Cook's membrane: real deformed states, all four PANN tiers",
                 fontsize=13.5)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(HERE / "cook_deformed_field_claude.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote cook_deformed_field_claude.pdf")


if __name__ == "__main__":
    main()
