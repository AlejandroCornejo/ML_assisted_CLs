#!/usr/bin/env python3
"""Schematic of the HPROM-ANN decoder network N: R^n -> R^{n-bar}, matching
this project's actual trained architecture (verified against
prom/ann/stage_7_ann_model_ls/manifold_ann_metadata.npz: n_primary=3,
n_secondary=36, hidden_layers=[128,128,128,128]). Drawn with a manageable
number of representative nodes per layer (real widths given as labels),
in the paper's own LaTeX font via matplotlib's usetex backend.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}\usepackage{bm}",
})

HERE = Path(__file__).resolve().parent

N_IN = 3
N_HIDDEN_SHOWN = 6
N_HIDDEN_LAYERS_SHOWN = 2
N_OUT_SHOWN = 8

LAYER_X = [0.0, 1.7, 3.4, 5.1]
IN_COLOR = "#d62728"
HIDDEN_COLOR = "#555555"
OUT_COLOR = "#1f77b4"


def layer_ys(n, height=3.4):
    if n == 1:
        return np.array([0.0])
    return np.linspace(height / 2, -height / 2, n)


def main() -> None:
    fig, ax = plt.subplots(figsize=(9.6, 4.4))

    ys_in = layer_ys(N_IN)
    ys_h1 = layer_ys(N_HIDDEN_SHOWN)
    ys_h2 = layer_ys(N_HIDDEN_SHOWN)
    ys_out = layer_ys(N_OUT_SHOWN, height=3.9)

    layers = [
        (LAYER_X[0], ys_in, IN_COLOR, "open"),
        (LAYER_X[1], ys_h1, HIDDEN_COLOR, "filled"),
        (LAYER_X[2], ys_h2, HIDDEN_COLOR, "filled"),
        (LAYER_X[3], ys_out, OUT_COLOR, "open"),
    ]

    # edges (draw first, behind nodes), a representative subset for clarity
    rng = np.random.default_rng(0)
    for (x0, ys0, _, _), (x1, ys1, _, _) in zip(layers[:-1], layers[1:]):
        for y0 in ys0:
            targets = ys1 if len(ys1) <= 8 else rng.choice(ys1, size=8, replace=False)
            for y1 in targets:
                ax.plot([x0, x1], [y0, y1], color="#b0b0b0", linewidth=0.5, zorder=1, alpha=0.7)

    node_kwargs_open = dict(s=110, facecolors="white", zorder=3, linewidths=1.4)
    node_kwargs_filled = dict(s=70, zorder=3, linewidths=0.8, edgecolors="white")

    ax.scatter([LAYER_X[0]] * N_IN, ys_in, edgecolors=IN_COLOR, **node_kwargs_open)
    ax.scatter([LAYER_X[1]] * N_HIDDEN_SHOWN, ys_h1, color=HIDDEN_COLOR, **node_kwargs_filled)
    ax.scatter([LAYER_X[2]] * N_HIDDEN_SHOWN, ys_h2, color=HIDDEN_COLOR, **node_kwargs_filled)
    ax.scatter([LAYER_X[3]] * N_OUT_SHOWN, ys_out, edgecolors=OUT_COLOR, **node_kwargs_open)

    for i, y in enumerate(ys_in):
        ax.text(LAYER_X[0] - 0.35, y, rf"$q_{{{i+1}}}$", ha="right", va="center", fontsize=11, color=IN_COLOR)
    ax.text(LAYER_X[3] + 0.32, ys_out[0], r"$\bar q_1$", ha="left", va="center", fontsize=10.5, color=OUT_COLOR)
    ax.text(LAYER_X[3] + 0.32, ys_out[-1], rf"$\bar q_{{\bar n}}$", ha="left", va="center", fontsize=10.5, color=OUT_COLOR)

    ax.text(LAYER_X[0], 2.55, r"$\bm q\in\mathbb R^{n}$" "\n" r"$(n=3)$", ha="center", va="bottom", fontsize=11, color=IN_COLOR)
    ax.text(LAYER_X[3], 2.55, r"$\bar{\bm q}\in\mathbb R^{\bar n}$" "\n" r"$(\bar n=36)$", ha="center", va="bottom", fontsize=11, color=OUT_COLOR)
    ax.text((LAYER_X[1] + LAYER_X[2]) / 2, 2.65, "4 hidden layers\n(128 units each, SiLU)", ha="center", va="bottom", fontsize=10.5, color=HIDDEN_COLOR)

    ax.text((LAYER_X[1] + LAYER_X[2]) / 2, -2.85, r"$\mathcal N:\mathbb R^{n}\to\mathbb R^{\bar n}$", ha="center", fontsize=13)

    ax.set_xlim(-1.6, LAYER_X[3] + 1.6)
    ax.set_ylim(-3.3, 4.3)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(HERE / "decoder_network_claude.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("wrote decoder_network_claude.png")


if __name__ == "__main__":
    main()
