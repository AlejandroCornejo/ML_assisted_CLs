#!/usr/bin/env python3
"""Regenerate ecm_stress_support_claude.png with the CORRECT Z_sig support
for the reaction-force-targeted stress rule actually deployed in
HPROM--ANN/D-HPROM--ANN this paper reports (fe2_extension/fit_final_
10point_rule_claude.py's own Z_support, pruned from a boundary-restricted
37-point classic-ECM rule). The previous version of this figure showed
the ORIGINAL, pre-reaction-force MAW--ECM rule (naive Gauss-point-stress
volume-average target), which is not boundary-restricted and is no
longer what any row in this paper's Cook or test-trajectory tables uses.

Style matches the original figure directly (square RVE domain with a
centered circular hole, light full mesh, filled triangles for the
selected support), confirmed against the original render before this
script was written.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
FE2_DIR = ROOT / "fe2_extension"
CORE_DIR = ROOT / "core"
for p in (str(CORE_DIR), str(FE2_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

KRATOS_PATH = "/home/kratos/Kratos_Eigen_Check/bin/Release"
if KRATOS_PATH not in sys.path:
    sys.path.append(KRATOS_PATH)

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}\usepackage{bm}",
})

Z_SUPPORT = np.array([931, 869, 948, 821, 921, 939, 926, 946, 686, 911], dtype=np.int64)
N_ELEM = 990


def main() -> None:
    from reaction_force_ecm_target_claude import DirectStressGenerator

    gen = DirectStressGenerator()
    assembler = gen.assembler
    mp = gen.mp

    coords = np.array([[node.X0, node.Y0] for node in mp.Nodes], dtype=float)
    node_id_to_idx = {int(node.Id): i for i, node in enumerate(mp.Nodes)}
    tris6 = assembler.connectivity  # (n_elem, 6) 0-based local node indices into mp.Nodes order
    gen.close()

    assert tris6.shape[0] == N_ELEM, f"expected {N_ELEM} elements, got {tris6.shape[0]}"
    tris3 = tris6[:, :3]

    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    triang_full = mtri.Triangulation(coords[:, 0], coords[:, 1], tris3)
    ax.triplot(triang_full, color="0.75", linewidth=0.25, zorder=1)

    is_selected = np.zeros(N_ELEM, dtype=bool)
    is_selected[Z_SUPPORT] = True
    for e in np.flatnonzero(is_selected):
        pts = coords[tris3[e]]
        ax.fill(pts[:, 0], pts[:, 1], color="tab:blue", zorder=3)

    ax.set_aspect("equal")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_title(r"$\mathcal Z_\sigma$: " + f"{Z_SUPPORT.size}/{N_ELEM} elements "
                 f"({100.0 * Z_SUPPORT.size / N_ELEM:.1f}\\%)")
    ax.text(
        0.03, 0.03,
        f"Selected: {Z_SUPPORT.size}/{N_ELEM} ({100.0*Z_SUPPORT.size/N_ELEM:.1f}\\%)\n"
        r"reaction-force target, held-out error $0.56\%$",
        transform=ax.transAxes, fontsize=9.5, va="bottom", ha="left",
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.5"),
    )

    fig.tight_layout()
    out_path = HERE / "ecm_stress_support_claude.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
