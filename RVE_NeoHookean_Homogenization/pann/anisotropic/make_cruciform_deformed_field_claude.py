#!/usr/bin/env python3
"""Cruciform specimen, genuine FE^2 demonstration: deformed configuration,
colored by displacement magnitude, for the six models carried forward to
this second geometry (tiers 1-2, pure regression and free hyperelastic,
are excluded here -- disqualified by the certificate violations of
Sections sec:c1-regression/sec:non-polyconvex and shown to fail outright
at Cook's membrane; see make_cook_deformed_field_claude.py for a figure
that still includes them). 2x3 GridSpec, same tri6->tri3 subdivision for
smooth Gouraud shading, same canonical model order. Real data from
fe2_extension/cruciform_results_*_claude.npz (run_cruciform_fe2_claude.py,
n_body=6, n_arm_len=4, delta=1.2, 20 steps), the mesh used throughout this
project's Cruciform demonstration."""
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


def load(which: str):
    path = FE2_DIR / FILENAMES[which]
    if not path.exists():
        return None
    d = np.load(path)
    if len(d["iters_per_step"]) != 20 or not bool(d["fully_converged"]):
        print(f"  [skip] {which}: {path.name} is stale/incomplete "
              f"({len(d['iters_per_step'])} steps, fully_converged={bool(d['fully_converged'])})")
        return None
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

    fig = plt.figure(figsize=(12.6, 8.4))
    gs = fig.add_gridspec(2, 3)

    for which in ROW_ORDER:
        row, cols = GRID[which]
        ax = fig.add_subplot(gs[row, cols])
        if which not in fields:
            ax.axis("off")
            ax.set_title(f"{TITLES[which]}\n(pending)", fontsize=11.2, color="0.5")
            continue
        coords, tris, u_nodal, mag = fields[which]
        deformed = coords + u_nodal
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

    fig.suptitle("Cruciform specimen: real deformed states, six surviving models", fontsize=13.5)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(HERE / "cruciform_deformed_field_claude.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote cruciform_deformed_field_claude.pdf")


if __name__ == "__main__":
    main()
