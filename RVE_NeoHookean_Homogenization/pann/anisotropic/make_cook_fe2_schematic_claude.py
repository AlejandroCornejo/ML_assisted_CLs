#!/usr/bin/env python3
"""Schematic with two clearly separated parts. Top: what genuine FE^2
would look like (a microscale RVE boundary-value problem solved at
every macroscopic Gauss point, exchanging an imposed macro strain for
a homogenized macro stress) -- explicitly marked as NOT performed in
this paper. Bottom: what Section 6.6 actually does instead -- the
trained PANN evaluated directly as the macroscopic material law, a
plain function call, no RVE anywhere. The two are kept visually and
spatially distinct so neither can be mistaken for the other. Left-top
panel is the real, whole Cook's-membrane mesh (not a generic macro
shape); right-top panel is this project's own RVE mesh, parsed
directly from core/rve_geometry.mdpa (no Dirichlet-boundary
highlighting -- that belongs to Section 2, not here).
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.tri as mtri
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}\usepackage{bm}",
})

HERE = Path(__file__).resolve().parent
COOK_DIR = HERE.parent.parent / "Cook.gid"
RVE_MDPA = HERE.parent.parent / "core" / "rve_geometry.mdpa"


def parse_rve_mesh(mdpa_path: Path):
    text = mdpa_path.read_text()
    nodes_block = re.search(r"Begin Nodes(.*?)End Nodes", text, re.S).group(1)
    node_ids, coords = [], []
    for line in nodes_block.strip().splitlines():
        parts = line.split()
        node_ids.append(int(parts[0]))
        coords.append((float(parts[1]), float(parts[2])))
    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
    coords = np.array(coords)

    elems_block = re.search(r"Begin Geometries Triangle2D6.*?\n(.*?)End Geometries", text, re.S).group(1)
    tris = []
    for line in elems_block.strip().splitlines():
        parts = [int(p) for p in line.split()]
        tris.append([id_to_idx[n] for n in parts[1:7]])
    return coords, np.array(tris)


def tri6_to_tri3(tris6: np.ndarray) -> np.ndarray:
    n0, n1, n2, n3, n4, n5 = (tris6[:, i] for i in range(6))
    return np.concatenate([
        np.stack([n0, n3, n5], axis=1), np.stack([n3, n1, n4], axis=1),
        np.stack([n5, n4, n2], axis=1), np.stack([n3, n4, n5], axis=1),
    ], axis=0)


def gauss_point_of(coords, tris, elem_idx):
    elem = tris[elem_idx]
    corners = coords[elem[:3]]
    bary = np.array([1 / 6, 1 / 6, 2 / 3])
    return bary @ corners, corners.mean(axis=0)


def box(ax_or_fig, xy, w, h, text, facecolor, edgecolor, fontsize=10.5, transform=None):
    b = FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.012,rounding_size=0.015",
                        facecolor=facecolor, edgecolor=edgecolor, linewidth=1.4,
                        transform=transform, zorder=4)
    ax_or_fig.add_artist(b)
    cx, cy = xy[0] + w / 2, xy[1] + h / 2
    ax_or_fig.text(cx, cy, text, ha="center", va="center", fontsize=fontsize,
                    transform=transform, zorder=5)


def main() -> None:
    fig = plt.figure(figsize=(11.0, 9.6))
    gs = fig.add_gridspec(1, 2, left=0.06, right=0.97, top=0.90, bottom=0.55, wspace=0.28)
    ax_cook = fig.add_subplot(gs[0, 0])
    ax_rve = fig.add_subplot(gs[0, 1])

    d = np.load(COOK_DIR / "cook_results_certified_claude.npz")
    cook_coords, cook_tris = d["coords"], d["tris"]
    gauss_pt_cook, _ = gauss_point_of(cook_coords, cook_tris, 240)

    tri3_cook = tri6_to_tri3(cook_tris)
    triang_cook = mtri.Triangulation(cook_coords[:, 0], cook_coords[:, 1], tri3_cook)
    ax_cook.triplot(triang_cook, color="#7f9fbf", linewidth=0.35)
    ax_cook.plot(*gauss_pt_cook, "o", color="#d62728", markersize=9, zorder=5)
    ax_cook.annotate("a Gauss point", xy=gauss_pt_cook, xytext=(gauss_pt_cook[0] - 6, gauss_pt_cook[1] + 11),
                      fontsize=9.5, color="#d62728",
                      arrowprops=dict(arrowstyle="-", color="#d62728", lw=0.8), bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=1.5))
    ax_cook.set_aspect("equal")
    ax_cook.set_xticks([]); ax_cook.set_yticks([])
    ax_cook.set_title("Cook's membrane, in full\n(real mesh, Section~6.6)", fontsize=11.5)

    rve_coords, rve_tris = parse_rve_mesh(RVE_MDPA)
    gauss_pt_rve, _ = gauss_point_of(rve_coords, rve_tris, 100)
    tri3_rve = tri6_to_tri3(rve_tris)
    triang_rve = mtri.Triangulation(rve_coords[:, 0], rve_coords[:, 1], tri3_rve)
    ax_rve.triplot(triang_rve, color="#7f9fbf", linewidth=0.35)
    ax_rve.plot(*gauss_pt_rve, "o", color="#d62728", markersize=8, zorder=5)
    ax_rve.annotate("a Gauss point", xy=gauss_pt_rve, xytext=(gauss_pt_rve[0] + 0.55, gauss_pt_rve[1] + 0.55),
                     fontsize=9.5, color="#d62728",
                     arrowprops=dict(arrowstyle="-", color="#d62728", lw=0.8), bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=1.5))
    ax_rve.set_xlim(-1.15, 1.15)
    ax_rve.set_ylim(-1.15, 1.15)
    ax_rve.set_aspect("equal")
    ax_rve.set_xticks([]); ax_rve.set_yticks([])
    ax_rve.set_title("This project's own RVE", fontsize=11.5)

    # Middle band: the FE^2 exchange, explicitly marked as not performed here.
    fig.add_artist(Rectangle((0.06, 0.355), 0.91, 0.185, transform=fig.transFigure,
                              facecolor="#fdeaea", edgecolor="#b02a2a", linewidth=1.3, zorder=1))
    fig.text(0.515, 0.518, r"\textbf{Genuine FE\textsuperscript{2}} -- \textbf{not performed in this paper}",
              color="#b02a2a", fontsize=12, ha="center")
    y_top, y_bot = 0.455, 0.400
    fig.add_artist(FancyArrowPatch((0.37, y_top), (0.63, y_top), transform=fig.transFigure,
                                    connectionstyle="arc3,rad=-0.3", arrowstyle="-|>", mutation_scale=16,
                                    color="black", linewidth=1.6))
    fig.add_artist(FancyArrowPatch((0.63, y_bot), (0.37, y_bot), transform=fig.transFigure,
                                    connectionstyle="arc3,rad=-0.3", arrowstyle="-|>", mutation_scale=16,
                                    color="black", linewidth=1.6))
    fig.text(0.5, y_top + 0.028, r"imposed $\bm\varepsilon$", color="black", fontsize=10.5, ha="center")
    fig.text(0.5, y_bot - 0.028, r"homogenized $\bm S$", color="black", fontsize=10.5, ha="center")

    # Bottom band: what Section 6.6 actually does.
    fig.add_artist(Rectangle((0.06, 0.045), 0.91, 0.27, transform=fig.transFigure,
                              facecolor="#eaf7ea", edgecolor="#2a7a2a", linewidth=1.3, zorder=1))
    fig.text(0.515, 0.275, r"\textbf{What Section~6.6 actually does}", color="#2a7a2a", fontsize=12, ha="center")

    by = 0.12
    box(fig, (0.10, by), 0.20, 0.10, "Cook's Gauss point\n(strain $\\bm\\varepsilon$)",
        "#ffffff", "#2a7a2a", fontsize=10, transform=fig.transFigure)
    box(fig, (0.40, by), 0.24, 0.10, "trained PANN\n(one forward pass)",
        "#ffffff", "#2a7a2a", fontsize=10, transform=fig.transFigure)
    box(fig, (0.74, by), 0.20, 0.10, "stress $\\bm S$\nand tangent",
        "#ffffff", "#2a7a2a", fontsize=10, transform=fig.transFigure)
    fig.add_artist(FancyArrowPatch((0.305, by + 0.05), (0.395, by + 0.05), transform=fig.transFigure,
                                    arrowstyle="-|>", mutation_scale=16, color="#2a7a2a", linewidth=1.6))
    fig.add_artist(FancyArrowPatch((0.645, by + 0.05), (0.735, by + 0.05), transform=fig.transFigure,
                                    arrowstyle="-|>", mutation_scale=16, color="#2a7a2a", linewidth=1.6))
    fig.text(0.515, by - 0.035, "no RVE solved, no microscale problem, anywhere in this row",
              color="#2a7a2a", fontsize=9.5, ha="center", style="italic")

    fig.savefig(HERE / "cook_fe2_schematic_claude.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote cook_fe2_schematic_claude.pdf")


if __name__ == "__main__":
    main()
