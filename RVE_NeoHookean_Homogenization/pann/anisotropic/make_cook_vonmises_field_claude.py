#!/usr/bin/env python3
"""Cook's membrane downstream demonstration: equivalent (von Mises)
stress field, all four PANN tiers, each at its own reported state
(certified cores at full converged load; free/regression at their
stalled 5% state). Same clean 2x2 grid, tier order, and color
convention as make_cook_deformed_field_claude.py. Real data from
Cook.gid/cook_results_*_claude.npz (run_cook_pann_claude.py).
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
    n0, n1, n2, n3, n4, n5 = (tris6[:, i] for i in range(6))
    return np.concatenate([
        np.stack([n0, n3, n5], axis=1),
        np.stack([n3, n1, n4], axis=1),
        np.stack([n5, n4, n2], axis=1),
        np.stack([n3, n4, n5], axis=1),
    ], axis=0)


def von_mises(stress: np.ndarray) -> np.ndarray:
    sxx, syy, sxy = stress[:, 0], stress[:, 1], stress[:, 2]
    return np.sqrt(np.maximum(sxx * sxx - sxx * syy + syy * syy + 3.0 * sxy * sxy, 0.0))


def load(which: str):
    d = np.load(COOK_DIR / f"cook_results_{which}_claude.npz")
    return d["coords"], d["tris"], d["u_nodal"], d["s_gp"]


def main() -> None:
    fields = []
    vmax = 0.0
    for which in TIER_ORDER:
        coords, tris, u_nodal, s_gp = load(which)
        vm_gp = von_mises(s_gp)
        # One value per element (mean of its 3 Gauss points), repeated 4x for
        # the 4 sub-triangles of tri6_to_tri3 -- flat, element-constant shading,
        # since stress is a Gauss-point/element quantity, not naturally nodal
        # (gouraud-smoothing it onto nodes first produced a checkerboard
        # artifact from the underlying diagonal Tri6 split).
        elem_mean_mpa = vm_gp.reshape(-1, 3).mean(axis=1) / 1.0e6
        vm_flat = np.tile(elem_mean_mpa, 4)
        vmax = max(vmax, float(elem_mean_mpa.max()))
        fields.append((coords, tris, u_nodal, vm_flat))

    fig, axes = plt.subplots(2, 2, figsize=(10.4, 10.0))

    for ax, which, (coords, tris, u_nodal, vm_flat) in zip(axes.flat, TIER_ORDER, fields):
        deformed = coords + u_nodal
        tri3 = tri6_to_tri3(tris)
        triang = mtri.Triangulation(deformed[:, 0], deformed[:, 1], tri3)
        tpc = ax.tripcolor(triang, facecolors=vm_flat, cmap="jet", vmin=0.0, vmax=vmax)
        ax.triplot(triang, color="white", linewidth=0.05, alpha=0.3)
        ax.set_title(TITLES[which], fontsize=11.2)
        ax.set_aspect("equal")
        ax.set_xlabel(r"$x$ [m]")
        ax.set_ylabel(r"$y$ [m]")
        cbar = fig.colorbar(tpc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(r"$\sigma_{\rm eq}$ [MPa]", fontsize=9.5)

    fig.suptitle("Cook's membrane: equivalent (von Mises) stress field, all four PANN tiers",
                 fontsize=13.0)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(HERE / "cook_vonmises_field_claude.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote cook_vonmises_field_claude.pdf")


if __name__ == "__main__":
    main()
