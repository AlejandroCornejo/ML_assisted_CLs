#!/usr/bin/env python3
"""Cruciform specimen, genuine FE^2 demonstration: equivalent (von Mises)
stress field, for the six models carried forward to this second geometry
(tiers 1-2, pure regression and free hyperelastic, are excluded here --
disqualified by the certificate violations of Sections
sec:c1-regression/sec:non-polyconvex and shown to fail outright at Cook's
membrane; see make_cook_vonmises_field_claude.py for a figure that still
includes them). 2x3-GridSpec layout, same nodal patch-recovery convention
(plain average over every Tri6 element sharing each node) and color
convention as before. Real data from
fe2_extension/cruciform_results_*_claude.npz."""
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
FE2_DIR = HERE.parent.parent / "fe2_extension"

ROW_ORDER = ("fom_nested_consistent_parallel", "pann_certified", "pann_ickan",
             "linear_hprom_parallel_continuation", "hprom_ann_parallel_continuation",
             "dhprom_ann_parallel")
FILENAMES = {which: f"cruciform_results_{which}_claude.npz" for which in ROW_ORDER}
TITLES = {
    "fom_nested_consistent_parallel": "FOM-FE$^2$",
    "pann_certified": "Polyconvex ICNN (tier 3a)",
    "pann_ickan": "Polyconvex ICKAN (tier 3b)",
    "linear_hprom_parallel_continuation": "Linear-HPROM-FE$^2$",
    "hprom_ann_parallel_continuation": "HPROM--ANN-FE$^2$",
    "dhprom_ann_parallel": "D-HPROM--ANN-FE$^2$",
}
GRID = {
    "fom_nested_consistent_parallel": (0, 0),
    "pann_certified": (0, 1),
    "pann_ickan": (0, 2),
    "linear_hprom_parallel_continuation": (1, 0),
    "hprom_ann_parallel_continuation": (1, 1),
    "dhprom_ann_parallel": (1, 2),
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
    node_sum = np.zeros(n_nodes)
    node_count = np.zeros(n_nodes)
    for e in range(tris.shape[0]):
        for local_node in tris[e]:
            node_sum[local_node] += elem_values[e]
            node_count[local_node] += 1
    return node_sum / np.maximum(node_count, 1)


def load(which: str):
    path = FE2_DIR / FILENAMES[which]
    if not path.exists():
        return None
    d = np.load(path)
    if len(d["iters_per_step"]) != 20 or not bool(d["fully_converged"]):
        print(f"  [skip] {which}: {path.name} is stale/incomplete "
              f"({len(d['iters_per_step'])} steps, fully_converged={bool(d['fully_converged'])})")
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
        elem_mean_mpa = vm_gp.reshape(-1, 3).mean(axis=1) / 1.0e6
        node_vm_mpa = recover_to_nodes(elem_mean_mpa, tris, coords.shape[0])
        vmax = max(vmax, float(node_vm_mpa.max()))
        fields[which] = (coords, tris, u_nodal, node_vm_mpa)

    fig = plt.figure(figsize=(12.6, 8.4))
    gs = fig.add_gridspec(2, 3)

    for which in ROW_ORDER:
        row, cols = GRID[which]
        ax = fig.add_subplot(gs[row, cols])
        if which not in fields:
            ax.axis("off")
            ax.set_title(f"{TITLES[which]}\n(pending)", fontsize=11.2, color="0.5")
            continue
        coords, tris, u_nodal, node_vm_mpa = fields[which]
        deformed = coords + u_nodal
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

    fig.suptitle("Cruciform specimen: equivalent (von Mises) stress field, six surviving models", fontsize=13.0)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(HERE / "cruciform_vonmises_field_claude.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote cruciform_vonmises_field_claude.pdf")


if __name__ == "__main__":
    main()
