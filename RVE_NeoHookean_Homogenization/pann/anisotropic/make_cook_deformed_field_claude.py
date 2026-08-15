#!/usr/bin/env python3
"""Cook's membrane, genuine FE^2 demonstration: deformed configuration,
colored by displacement magnitude, for every row of Table 7 (the true,
non-reduced FE^2 solve first, centered, then the six trained/built laws
in two columns below it). All seven panels the same size (a 4x4
GridSpec: the true-FOM panel spans the middle two columns of row 0,
each other panel spans two columns of its own row). The undeformed
reference mesh is a separate figure (make_cook_reference_mesh_claude.py).
Real data from fe2_extension/cook_results_*_claude.npz
(run_cook_hprom_ann_claude.py, nx=ny=8), the same files and resolution
Table 7 itself reports.

Data-provenance note: regression/free use cook_results_pann_{regression,
free}_claude.npz, NOT cook_results_{regression,free}_nx8_claude.npz --
the latter was a stale, single-load-step (5%) diagnostic leftover
mislabeled with the canonical filename (already deleted once found; see
compare_true_fom_speedup_accuracy_claude.py's own docstring for the full
story). Confirm len(load_per_step) == 20 for any new file used here.
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

ROW_ORDER = ("fom_nested_full", "regression", "free", "certified", "ickan",
             "hprom_iterative_f64_consistent", "dhprom_f64_consistent")
FILENAMES = {
    "fom_nested_full": "cook_results_fom_nested_full_claude.npz",
    "regression": "cook_results_pann_regression_claude.npz",
    "free": "cook_results_pann_free_claude.npz",
    "certified": "cook_results_icnn_w5_final_claude.npz",
    "ickan": "cook_results_ickan_w5_final_claude.npz",
    "hprom_iterative_f64_consistent": "cook_results_hprom_iterative_f64_consistent_claude.npz",
    "dhprom_f64_consistent": "cook_results_dhprom_f64_consistent_claude.npz",
}
TITLES = {
    "fom_nested_full": "FOM-FE$^2$",
    "regression": "Regression (tier 1), Newton stalled every step",
    "free": "Free hyperelastic (tier 2), Newton stalled every step",
    "certified": "Polyconvex ICNN (tier 3a)",
    "ickan": "Polyconvex ICKAN (tier 3b)",
    "hprom_iterative_f64_consistent": "HPROM--ANN-FE$^2$",
    "dhprom_f64_consistent": "D-HPROM--ANN-FE$^2$",
}
# GridSpec(row, col-slice) for each row -- true FOM centered on its own
# row, the rest two-per-row in the remaining three rows, consistent
# reading order everywhere in the paper: FOM, tier 1, 2, 3a, 3b, then
# the two ROM/HPROM-FE^2 rows.
GRID = {
    "fom_nested_full": (0, slice(1, 3)),
    "regression": (1, slice(0, 2)),
    "free": (1, slice(2, 4)),
    "certified": (2, slice(0, 2)),
    "ickan": (2, slice(2, 4)),
    "hprom_iterative_f64_consistent": (3, slice(0, 2)),
    "dhprom_f64_consistent": (3, slice(2, 4)),
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
    path = COOK_DIR / FILENAMES[which]
    if not path.exists():
        return None
    d = np.load(path)
    if "load_per_step" in d:
        assert len(d["load_per_step"]) == 20, f"{which}: {path.name} has {len(d['load_per_step'])} steps, not 20"
    return d["coords"], d["tris"], d["u_nodal"]


def main() -> None:
    fields = {}
    vmax = 0.0
    for which in ROW_ORDER:
        loaded = load(which)
        if loaded is None:
            print(f"  [skip] {which}: {FILENAMES[which]} not yet available")
            continue
        coords, tris, u_nodal = loaded
        mag = np.linalg.norm(u_nodal, axis=1)
        vmax = max(vmax, float(mag.max()))
        fields[which] = (coords, tris, u_nodal, mag)

    fig = plt.figure(figsize=(9.6, 13.0))
    gs = fig.add_gridspec(4, 4)

    for which in ROW_ORDER:
        row, cols = GRID[which]
        ax = fig.add_subplot(gs[row, cols])
        if which not in fields:
            ax.axis("off")
            ax.set_title(f"{TITLES[which]}\n(pending)", fontsize=11.2, color="0.5")
            continue
        coords, tris, u_nodal, mag = fields[which]
        deformed = coords + u_nodal
        # Fine (mid-side-node-included) triangulation for smooth Gouraud
        # shading; a SEPARATE coarse (corner-only) one for the outline --
        # triplot-ing the fine one draws a line at every mid-side node too,
        # making the true nx=8 mesh look like nx=16 (same issue fixed in
        # make_cook_reference_mesh_claude.py; confirmed there directly
        # against the raw node coordinates before trusting the fix).
        triang_fine = mtri.Triangulation(deformed[:, 0], deformed[:, 1], tri6_to_tri3(tris))
        triang_coarse = mtri.Triangulation(deformed[:, 0], deformed[:, 1], tris[:, :3])
        tpc = ax.tripcolor(triang_fine, mag, shading="gouraud", cmap="viridis", vmin=0.0, vmax=vmax)
        ax.triplot(triang_coarse, color="white", linewidth=0.3, alpha=0.5)
        ax.set_title(TITLES[which], fontsize=11.2)
        ax.set_aspect("equal")
        ax.set_xlabel(r"$x$ [m]")
        ax.set_ylabel(r"$y$ [m]")
        cbar = fig.colorbar(tpc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(r"$\|\bm u\|$ [m]", fontsize=9.5)

    fig.suptitle("Cook's membrane: real deformed states, every row of Table 7",
                 fontsize=13.5)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(HERE / "cook_deformed_field_claude.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote cook_deformed_field_claude.pdf")


if __name__ == "__main__":
    main()
