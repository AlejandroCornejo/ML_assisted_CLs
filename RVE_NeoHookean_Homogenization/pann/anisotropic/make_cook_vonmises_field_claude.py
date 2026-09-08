#!/usr/bin/env python3
"""Cook's membrane, genuine FE^2 demonstration: equivalent (von Mises)
stress field, for every row of Table 7, in this paper's canonical model
order (FOM-FE^2, tiers 1/2/3a/3b, Linear-HPROM, HPROM--ANN,
D-HPROM--ANN), each at its own reported state. Same 2x4-GridSpec layout,
row order, and color convention as make_cook_deformed_field_claude.py.
Element-mean von Mises stress is recovered to nodes by plain averaging
over every element sharing each node, then Gouraud-shaded for a smooth,
continuous field -- the standard FE post-processing convention (matches
e.g. Hernandez et al.'s own damage-contour figures). Real data from
fe2_extension/cook_results_*_claude.npz (run_cook_hprom_ann_claude.py,
nx=ny=8), the same files and resolution Table 7 itself reports.

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

ROW_ORDER = ("fom_nested_consistent_parallel", "pann_regression", "pann_free", "pann_certified_w5",
             "pann_ickan_w5", "linear_hprom_parallel_continuation", "hprom_ann_parallel_continuation",
             "dhprom_ann_parallel")
FILENAMES = {which: f"cook_results_{which}_claude.npz" for which in ROW_ORDER}
TITLES = {
    "fom_nested_consistent_parallel": "FOM-FE$^2$",
    "pann_regression": "Regression (tier 1), Newton stalled every step",
    "pann_free": "Free hyperelastic (tier 2), Newton stalled every step",
    "pann_certified_w5": "Polyconvex ICNN (tier 3a)",
    "pann_ickan_w5": "Polyconvex ICKAN (tier 3b)",
    "hprom_ann_parallel_continuation": "HPROM--ANN-FE$^2$",
    "dhprom_ann_parallel": "D-HPROM--ANN-FE$^2$",
    "linear_hprom_parallel_continuation": "Linear-HPROM-FE$^2$",
}
GRID = {
    "fom_nested_consistent_parallel": (0, 0),
    "pann_regression": (0, 1),
    "pann_free": (0, 2),
    "pann_certified_w5": (0, 3),
    "pann_ickan_w5": (1, 0),
    "linear_hprom_parallel_continuation": (1, 1),
    "hprom_ann_parallel_continuation": (1, 2),
    "dhprom_ann_parallel": (1, 3),
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


def recover_to_nodes(elem_values: np.ndarray, tris: np.ndarray, n_nodes: int) -> np.ndarray:
    """Simple nodal patch recovery: each node's value is the plain average
    of elem_values over every Tri6 element containing it (as any of its 6
    local nodes, corner or mid-side), not just the sub-triangles of a
    single element. This is what avoids the checkerboard artifact an
    earlier, naive per-sub-triangle interpolation produced: that artifact
    came from smoothing within one Tri6's own diagonal split, not from
    smoothing itself -- averaging across the full node-sharing patch, as
    any standard FE post-processor does, gives a clean, continuous field
    (matches the convention in Hernandez et al.'s own ECM papers, e.g.
    hernandez2026hyperreduction.pdf Fig. 22(a)'s damage contour)."""
    node_sum = np.zeros(n_nodes)
    node_count = np.zeros(n_nodes)
    for e in range(tris.shape[0]):
        for local_node in tris[e]:
            node_sum[local_node] += elem_values[e]
            node_count[local_node] += 1
    return node_sum / np.maximum(node_count, 1)


def load(which: str):
    path = COOK_DIR / FILENAMES[which]
    if not path.exists():
        return None
    d = np.load(path)
    if "load_per_step" in d and len(d["load_per_step"]) != 20:
        print(f"  [skip] {which}: {path.name} is stale/incomplete ({len(d['load_per_step'])} steps, not 20)")
        return None
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
        elem_mean_mpa = vm_gp.reshape(-1, 3).mean(axis=1) / 1.0e6  # (n_elem,)
        node_vm_mpa = recover_to_nodes(elem_mean_mpa, tris, coords.shape[0])
        vmax = max(vmax, float(node_vm_mpa.max()))
        fields[which] = (coords, tris, u_nodal, node_vm_mpa)

    fig = plt.figure(figsize=(16.4, 8.4))
    gs = fig.add_gridspec(2, 4)

    for which in ROW_ORDER:
        row, cols = GRID[which]
        ax = fig.add_subplot(gs[row, cols])
        if which not in fields:
            ax.axis("off")
            ax.set_title(f"{TITLES[which]}\n(pending)", fontsize=11.2, color="0.5")
            continue
        coords, tris, u_nodal, node_vm_mpa = fields[which]
        deformed = coords + u_nodal
        # Fine (mid-side-node-included) triangulation for smooth Gouraud
        # shading of the recovered nodal field; a SEPARATE coarse
        # (corner-only) one for the outline -- see
        # make_cook_deformed_field_claude.py's own comment for why
        # (triplot-ing the fine mesh makes nx=8 look like nx=16).
        triang_fine = mtri.Triangulation(deformed[:, 0], deformed[:, 1], tri6_to_tri3(tris))
        triang_coarse = mtri.Triangulation(deformed[:, 0], deformed[:, 1], tris[:, :3])
        tpc = ax.tripcolor(triang_fine, node_vm_mpa, shading="gouraud", cmap="jet", vmin=0.0, vmax=vmax)
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
