#!/usr/bin/env python3
"""Cruciform specimen: reference (undeformed) mesh, standalone figure.

Uses n_body=6, n_arm_len=4 (fe2_extension/run_cruciform_fe2_claude.py's
own default resolution) -- NOT n_body=9, which this project's own
build_cruciform_mesh_claude.py builds as a genuinely non-conforming mesh
whenever arm_width_fraction != 1/3 (the default used everywhere in this
project is 2/3): the function's own alignment assertion only guards the
1/3 case, so at n_body=9 with the actual 2/3 arm width, the arm's own
corner-node grid lands exactly on the body's mid-side nodes (out of
phase by half a cell) instead of its corners, breaking Tri6 edge
compatibility across the arm/body interface even though the raw node
coordinates still coincide and get merged. Confirmed directly: at
n_body=9 the merged mesh's total open/boundary edge length is 176 m
against a true analytic perimeter of 112 m (a real, disconnected-arm
defect); n_body=6, 12, 18, 24 all give exactly 112 m (conforming). Only
a single mesh is shown here, so no n_body value needs to be called out.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.tri as mtri

HERE = Path(__file__).resolve().parent
FE2_DIR = HERE.parent.parent / "fe2_extension"
if str(FE2_DIR) not in sys.path:
    sys.path.insert(0, str(FE2_DIR))

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}\usepackage{bm}",
})


def main() -> None:
    from build_cruciform_mesh_claude import build_cruciform_mesh

    coords, tris, tip_nodes, center_node = build_cruciform_mesh(
        n_body=6, n_arm_len=4, arm_width_fraction=2.0 / 3.0)
    triang = mtri.Triangulation(coords[:, 0], coords[:, 1], tris[:, :3])

    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    ax.triplot(triang, color="0.25", linewidth=0.5)
    ax.set_title(f"Cruciform specimen reference mesh ({tris.shape[0]} Tri6 elements)")
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x$ [m]")
    ax.set_ylabel(r"$y$ [m]")

    fig.tight_layout()
    out_path = HERE / "cruciform_reference_mesh_claude.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path} ({tris.shape[0]} Tri6 elements, {coords.shape[0]} nodes)")


if __name__ == "__main__":
    main()
