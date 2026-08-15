#!/usr/bin/env python3
"""Cook's membrane, genuine FE^2 demonstration: equivalent (von Mises)
stress field, for every row of Table 7 (the true, non-reduced FE^2
solve first, centered, then the six trained/built laws in two columns
below it), each at its own reported state. Same 4x4-GridSpec layout,
row order, and color convention as make_cook_deformed_field_claude.py.
Real data from fe2_extension/cook_results_*_claude.npz
(run_cook_hprom_ann_claude.py, nx=ny=8), the same files and resolution
Table 7 itself reports.

Data-provenance note: regression/free use cook_results_pann_{regression,
free}_claude.npz, NOT cook_results_{regression,free}_nx8_claude.npz --
see make_cook_deformed_field_claude.py's own docstring for why.
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
    path = COOK_DIR / FILENAMES[which]
    if not path.exists():
        return None
    d = np.load(path)
    if "load_per_step" in d:
        assert len(d["load_per_step"]) == 20, f"{which}: {path.name} has {len(d['load_per_step'])} steps, not 20"
    return d["coords"], d["tris"], d["u_nodal"], d["s_gp"]


def main() -> None:
    fields = {}
    vmax = 0.0
    for which in ROW_ORDER:
        loaded = load(which)
        if loaded is None:
            print(f"  [skip] {which}: {FILENAMES[which]} not yet available")
            continue
        coords, tris, u_nodal, s_gp = loaded
        vm_gp = von_mises(s_gp)
        # One value per element (mean of its 3 Gauss points), repeated 4x for
        # the 4 sub-triangles of tri6_to_tri3 -- flat, element-constant shading,
        # since stress is a Gauss-point/element quantity, not naturally nodal
        # (gouraud-smoothing it onto nodes first produced a checkerboard
        # artifact from the underlying diagonal Tri6 split).
        elem_mean_mpa = vm_gp.reshape(-1, 3).mean(axis=1) / 1.0e6
        vm_flat = np.tile(elem_mean_mpa, 4)
        vmax = max(vmax, float(elem_mean_mpa.max()))
        fields[which] = (coords, tris, u_nodal, vm_flat)

    fig = plt.figure(figsize=(9.6, 13.0))
    gs = fig.add_gridspec(4, 4)

    for which in ROW_ORDER:
        row, cols = GRID[which]
        ax = fig.add_subplot(gs[row, cols])
        if which not in fields:
            ax.axis("off")
            ax.set_title(f"{TITLES[which]}\n(pending)", fontsize=11.2, color="0.5")
            continue
        coords, tris, u_nodal, vm_flat = fields[which]
        deformed = coords + u_nodal
        # Fine (mid-side-node-included) triangulation for the flat
        # per-sub-triangle stress coloring; a SEPARATE coarse (corner-only)
        # one for the outline -- see make_cook_deformed_field_claude.py's
        # own comment for why (triplot-ing the fine mesh makes nx=8 look
        # like nx=16).
        triang_fine = mtri.Triangulation(deformed[:, 0], deformed[:, 1], tri6_to_tri3(tris))
        triang_coarse = mtri.Triangulation(deformed[:, 0], deformed[:, 1], tris[:, :3])
        tpc = ax.tripcolor(triang_fine, facecolors=vm_flat, cmap="jet", vmin=0.0, vmax=vmax)
        ax.triplot(triang_coarse, color="white", linewidth=0.3, alpha=0.5)
        ax.set_title(TITLES[which], fontsize=11.2)
        ax.set_aspect("equal")
        ax.set_xlabel(r"$x$ [m]")
        ax.set_ylabel(r"$y$ [m]")
        cbar = fig.colorbar(tpc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(r"$\sigma_{\rm eq}$ [MPa]", fontsize=9.5)

    fig.suptitle("Cook's membrane: equivalent (von Mises) stress field, every row of Table 7",
                 fontsize=13.0)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(HERE / "cook_vonmises_field_claude.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote cook_vonmises_field_claude.pdf")


if __name__ == "__main__":
    main()
