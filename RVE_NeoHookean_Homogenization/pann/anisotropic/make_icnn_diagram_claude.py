#!/usr/bin/env python3
"""End-to-end diagram of the polyconvex ICNN core used in this paper:
deformation gradient -> the 15-dimensional feature map z_pc(F) of
Eq. (eq:features) -> the input-convex network Phi_theta (widths
15-64-64-32-1, Softplus, every weight matrix reparameterized
non-negative, with the characteristic ICNN skip connections feeding the
raw features into every layer) -> the energy W_pc(F) of Eq. (eq:pc-energy)
-> the stress via automatic differentiation, Eq. (eq:stress). All
equations/numbers are taken directly from the paper (widths from
\\IcnnWidths = 64,64,32).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}\usepackage{bm}",
})

HERE = Path(__file__).resolve().parent

BOX_EDGE = "#1f5fa8"
BOX_FACE = "white"
ARROW_COLOR = "#1f5fa8"
IN_COLOR = "#d62728"
HIDDEN_COLOR = "#555555"
OUT_COLOR = "#1f77b4"
SKIP_COLOR = "#d62728"


def rounded_box(ax, xy, w, h, **kwargs):
    x, y = xy
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.0,rounding_size=0.12",
        linewidth=1.6, edgecolor=BOX_EDGE, facecolor=BOX_FACE,
        zorder=2, **kwargs,
    )
    ax.add_patch(box)
    return box


def title_text(ax, xy, w, h, text, fontsize=10.8):
    x, y = xy
    ax.text(x, y + h / 2 - 0.16, text, ha="center", va="top",
             fontsize=fontsize, fontweight="bold", zorder=3)


def eq_text(ax, x, y, text, fontsize=9.2, **kwargs):
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, zorder=3, **kwargs)


def arrow(ax, p0, p1, **kwargs):
    a = FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=14, linewidth=1.6,
        color=ARROW_COLOR, shrinkA=2, shrinkB=2, zorder=2.9, **kwargs,
    )
    ax.add_patch(a)


def layer_ys(n, height):
    if n == 1:
        return np.array([0.0])
    return np.linspace(height / 2, -height / 2, n)


def draw_icnn_network(ax, x0, y0, w, h):
    """Draws the ICNN node diagram (15-64-64-32-1) with skip connections
    from the feature layer into every subsequent layer, inside a box of
    width w, height h centered at (x0, y0)."""
    n_in_shown, n_h1_shown, n_h2_shown, n_h3_shown, n_out = 7, 7, 7, 6, 1
    xs = np.linspace(x0 - w / 2 + 0.30, x0 + w / 2 - 0.22, 5)
    ys_in = layer_ys(n_in_shown, 1.55) + y0 - 0.15
    ys_h1 = layer_ys(n_h1_shown, 1.55) + y0 - 0.15
    ys_h2 = layer_ys(n_h2_shown, 1.55) + y0 - 0.15
    ys_h3 = layer_ys(n_h3_shown, 1.30) + y0 - 0.15
    ys_out = np.array([y0 - 0.15])

    layers = [xs[0], xs[1], xs[2], xs[3], xs[4]]
    ys_all = [ys_in, ys_h1, ys_h2, ys_h3, ys_out]
    colors = [IN_COLOR, HIDDEN_COLOR, HIDDEN_COLOR, HIDDEN_COLOR, OUT_COLOR]

    rng = np.random.default_rng(3)
    # standard forward ("y-path") edges, consecutive layers
    for xa, ya, xb, yb in zip(layers[:-1], ys_all[:-1], layers[1:], ys_all[1:]):
        for y0_ in ya:
            targets = yb if len(yb) <= 7 else rng.choice(yb, size=7, replace=False)
            for y1_ in targets:
                ax.plot([xa, xb], [y0_, y1_], color="#b0b0b0", linewidth=0.45, zorder=2.2, alpha=0.75)

    # nodes
    for x, ys, c, filled in zip(layers, ys_all, colors,
                                 [False, True, True, True, False]):
        if filled:
            ax.scatter([x] * len(ys), ys, s=42, color=c, zorder=3, linewidths=0.6, edgecolors="white")
        else:
            ax.scatter([x] * len(ys), ys, s=68, facecolors="white", edgecolors=c, zorder=3, linewidths=1.2)

    # ICNN skip connections: raw features feed every subsequent layer directly
    skip_top = max(ys_in) + 0.55
    for xb, yb in zip(layers[2:], ys_all[2:]):
        ax.plot([xb, xb], [skip_top, max(yb) + 0.06],
                color=SKIP_COLOR, lw=1.1, alpha=0.9, zorder=2.6)
    ax.plot([layers[0], layers[0]], [max(ys_in) + 0.06, skip_top],
            color=SKIP_COLOR, lw=1.1, alpha=0.9, zorder=2.6)
    ax.plot([layers[0], layers[4]], [skip_top, skip_top],
            color=SKIP_COLOR, lw=1.1, alpha=0.9, zorder=2.6)

    ax.text(x0, skip_top + 0.20,
            r"$\bm z_{\rm pc}$ additionally skip-feeds every \emph{later} layer (layer 1 gets it via the plain path)",
            ha="center", fontsize=6.8, color=SKIP_COLOR)

    # layer width labels
    labels = [r"$15$", r"$64$", r"$64$", r"$32$", r"$1$"]
    for x, lab in zip(layers, labels):
        ax.text(x, y0 - 1.15, lab, ha="center", fontsize=8.6)
    ax.text(x0, y0 - 1.42, r"Softplus, all weights $\geq0$", ha="center", fontsize=7.6, color=HIDDEN_COLOR)


def main() -> None:
    fig, ax = plt.subplots(figsize=(16.6, 5.4))

    c1 = (0.0, 0.0)
    c2 = (3.35, 0.0)
    c3 = (7.85, 0.0)
    c4 = (12.65, 0.0)
    c5 = (16.20, 0.0)

    w1, h1 = 2.7, 3.5
    w2, h2 = 3.0, 3.9
    w3, h3 = 5.1, 3.9
    w4, h4 = 3.55, 3.9
    w5, h5 = 2.7, 3.5

    rounded_box(ax, c1, w1, h1)
    rounded_box(ax, c2, w2, h2)
    rounded_box(ax, c3, w3, h3)
    rounded_box(ax, c4, w4, h4)
    rounded_box(ax, c5, w5, h5)

    # --- Box 1: input ---
    title_text(ax, c1, w1, h1, "input:\ndeformation gradient")
    eq_text(ax, c1[0], c1[1] + 0.60, r"$\bm F$")
    eq_text(ax, c1[0], c1[1] + 0.10, r"$\bm C=\bm F^T\bm F,\ \ J=\det\bm F$", fontsize=8.8)
    eq_text(ax, c1[0], c1[1] - 0.45, r"$\bm E=\tfrac12(\bm C-\bm I)$", fontsize=8.8)
    eq_text(ax, c1[0], c1[1] - 1.15, r"$\bm\varepsilon=(E_{11},E_{22},\gamma_{12})$", fontsize=8.6)

    # --- Box 2: 15 features ---
    title_text(ax, c2, w2, h2, r"15 polyconvex features $\bm z_{\rm pc}(\bm F)$")
    eq_text(ax, c2[0], c2[1] + 1.10, r"$I_1$", fontsize=8.8)
    for i, k in enumerate([1, 2, 3]):
        y = c2[1] + 0.68 - i * 0.62
        eq_text(ax, c2[0], y,
                rf"$Q_{{{k},2}}^F,\,Q_{{{k},3}}^F,\,Q_{{{k},2}}^H,\,Q_{{{k},3}}^H$",
                fontsize=8.3)
    eq_text(ax, c2[0], c2[1] - 1.10, r"$J,\ J^2$", fontsize=8.8)
    eq_text(ax, c2[0], c2[1] - 1.55, r"$\bm z_{\rm pc}\in\mathbb R^{15}$", fontsize=8.6, style="italic")

    # --- Box 3: ICNN architecture ---
    title_text(ax, c3, w3, h3, r"input-convex network $\Phi_\theta:\mathbb R^{15}\to\mathbb R$")
    draw_icnn_network(ax, c3[0], c3[1] + 0.15, w3 - 0.5, h3 - 1.3)

    # --- Box 4: output / energy ---
    title_text(ax, c4, w4, h4, "output: polyconvex energy")
    eq_text(ax, c4[0], c4[1] + 1.15, r"$\Phi_\theta(\bm z_{\rm pc})\in\mathbb R$", fontsize=9.2)
    eq_text(ax, c4[0], c4[1] + 0.55,
            r"$W_{\rm pc}(\bm F)=\Phi_\theta(\bm z_{\rm pc})$", fontsize=8.6)
    eq_text(ax, c4[0], c4[1] + 0.12,
            r"$-\,\Phi_\theta(\bm z_{\rm pc}(\bm I))$", fontsize=8.6)
    eq_text(ax, c4[0], c4[1] - 0.31,
            r"$-\,r\log J+\dfrac\beta2(J-1)^2$", fontsize=8.6)
    eq_text(ax, c4[0], c4[1] - 1.05,
            r"stress-free at $\bm F=\bm I$", fontsize=8.2, style="italic")
    eq_text(ax, c4[0], c4[1] - 1.40,
            r"for \emph{any} trained $\Phi_\theta$", fontsize=8.2, style="italic")

    # --- Box 5: derivative / stress ---
    title_text(ax, c5, w5, h5, "derivative:\nstress")
    eq_text(ax, c5[0], c5[1] + 0.55,
            r"$\bm\sigma=\dfrac{\partial W_{\rm pc}}{\partial\bm\varepsilon}$", fontsize=9.4)
    eq_text(ax, c5[0], c5[1] - 0.15,
            r"$=(S_{11},S_{22},S_{12})^T$", fontsize=8.8)
    eq_text(ax, c5[0], c5[1] - 0.85,
            "automatic\ndifferentiation", fontsize=8.4, style="italic")
    eq_text(ax, c5[0], c5[1] - 1.40,
            "no separate stress network", fontsize=7.8, color=HIDDEN_COLOR)

    # --- arrows ---
    arrow(ax, (c1[0] + w1 / 2, c1[1]), (c2[0] - w2 / 2, c2[1]))
    arrow(ax, (c2[0] + w2 / 2, c2[1]), (c3[0] - w3 / 2, c3[1]))
    arrow(ax, (c3[0] + w3 / 2, c3[1]), (c4[0] - w4 / 2, c4[1]))
    arrow(ax, (c4[0] + w4 / 2, c4[1]), (c5[0] - w5 / 2, c5[1]))

    ax.set_xlim(-1.6, 17.75)
    ax.set_ylim(-3.1, 3.1)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(HERE / "icnn_diagram_claude.png", dpi=230, bbox_inches="tight")
    plt.close(fig)
    print("wrote icnn_diagram_claude.png")


if __name__ == "__main__":
    main()
