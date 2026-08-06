#!/usr/bin/env python3
"""End-to-end HPROM-ANN / D-HPROM-ANN workflow diagram, mirroring the layout
of the WCCM2026 presentation slide (macro input -> affine/structured
initialization -> {HPROM-ANN Galerkin correction, D-HPROM-ANN direct state}
-> PROM-ANN decoder state map -> hyper-reduced homogenization), but with
every equation and mini-figure drawn from what this paper actually defines
(Eqs. for u_d(X), the decoder, the ECM residual/stress cubature rules) and
the real supporting figures already generated for this RVE (ecm_residual_
support_claude.png, ecm_stress_support_claude.png, a cropped displacement-
field thumbnail, and the structured-coordinate domain plot).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}\usepackage{bm}",
})

HERE = Path(__file__).resolve().parent
SCRATCH_DIR = Path(
    "/tmp/claude-1000/-home-kratos-ML-assisted-CLs-clean/"
    "4a3da423-8dfc-40e5-a7fa-8fa40d3c8b2f/scratchpad"
)
SCRATCH_DISPLACEMENT_THUMB = SCRATCH_DIR / "displacement_thumb.png"
SCRATCH_QM_STRUCTURED_THUMB = SCRATCH_DIR / "qm_structured_thumb.png"
SCRATCH_ECM_RES_THUMB = SCRATCH_DIR / "ecm_res_clean.png"
SCRATCH_ECM_SIG_THUMB = SCRATCH_DIR / "ecm_sig_clean.png"

BOX_EDGE = "#1f5fa8"
BOX_FACE = "white"
ARROW_COLOR = "#1f5fa8"


def rounded_box(ax, xy, w, h, **kwargs):
    x, y = xy
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle=f"round,pad=0.0,rounding_size=0.12",
        linewidth=1.6, edgecolor=BOX_EDGE, facecolor=BOX_FACE,
        zorder=2, **kwargs,
    )
    ax.add_patch(box)
    return box


def title_text(ax, xy, w, h, text, fontsize=10.5):
    x, y = xy
    ax.text(x, y + h / 2 - 0.16, text, ha="center", va="top",
             fontsize=fontsize, fontweight="bold", zorder=3, wrap=True)


def eq_text(ax, x, y, text, fontsize=9.2, **kwargs):
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, zorder=3, **kwargs)


def embed_image(ax, xy, path, zoom):
    img = plt.imread(str(path))
    im = OffsetImage(img, zoom=zoom)
    ab = AnnotationBbox(im, xy, frameon=False, zorder=3)
    ax.add_artist(ab)


def arrow(ax, p0, p1, **kwargs):
    a = FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=14, linewidth=1.6,
        color=ARROW_COLOR, shrinkA=2, shrinkB=2, zorder=2.9, **kwargs,
    )
    ax.add_patch(a)


def main() -> None:
    fig, ax = plt.subplots(figsize=(15.0, 7.2))

    # box centers
    c1 = (0.0, 0.0)
    c2 = (3.15, 0.0)
    c3a = (6.30, 2.15)
    c3b = (6.30, -2.15)
    c4 = (9.55, 0.0)
    c5 = (12.85, 0.0)

    w1, h1 = 2.55, 3.1
    w2, h2 = 2.55, 3.5
    w3, h3 = 2.75, 3.3
    w4, h4 = 2.75, 3.6
    w5, h5 = 3.05, 3.95

    rounded_box(ax, c1, w1, h1)
    rounded_box(ax, c2, w2, h2)
    rounded_box(ax, c3a, w3, h3)
    rounded_box(ax, c3b, w3, h3)
    rounded_box(ax, c4, w4, h4)
    rounded_box(ax, c5, w5, h5)

    # --- Box 1: macro input ---
    title_text(ax, c1, w1, h1, "macro input")
    eq_text(ax, c1[0], c1[1] + 0.75, r"$\bm\varepsilon=(E_{11},E_{22},\gamma_{12})$")
    eq_text(ax, c1[0], c1[1] + 0.20, r"$\downarrow$", fontsize=11)
    eq_text(ax, c1[0], c1[1] - 0.30, r"$\bm F=\bm U=\sqrt{2\bm E(\bm\varepsilon)+\bm I}$")
    eq_text(ax, c1[0], c1[1] - 0.80, r"$\downarrow$", fontsize=11)
    eq_text(ax, c1[0], c1[1] - 1.30, r"$\bm u_d(\bm X)=(\bm F-\bm I)\bm X$")

    # --- Box 2: structured-coordinate / affine initialization ---
    title_text(ax, c2, w2, h2, "structured-coordinate\ninitialization")
    eq_text(ax, c2[0], c2[1] + 1.05,
            r"$\bm q(\bm\varepsilon)$: least-squares", fontsize=9.0)
    eq_text(ax, c2[0], c2[1] + 0.78,
            r"structured primary coordinate", fontsize=9.0)
    eq_text(ax, c2[0], c2[1] + 0.50,
            r"(rel.\ err.\ $6.9\times10^{-4}$)", fontsize=8.6)
    qm_thumb = SCRATCH_QM_STRUCTURED_THUMB if SCRATCH_QM_STRUCTURED_THUMB.exists() else HERE / "domain_qm_structured_claude.png"
    embed_image(ax, (c2[0], c2[1] - 0.75), qm_thumb, zoom=0.062)

    # --- Box 3a: HPROM-ANN Galerkin correction ---
    title_text(ax, c3a, w3, h3, "HPROM--ANN:\nGalerkin correction")
    eq_text(ax, c3a[0], c3a[1] + 0.62,
            r"$\displaystyle\sum_{g\in\mathcal Z}\!\big(\mathbf V^T\bm r_g\big)(\bm u_f;\bm\varepsilon)\,\omega_g=\bm 0$",
            fontsize=8.6)
    eq_text(ax, c3a[0], c3a[1] + 0.10, r"Newton--corrected to convergence", fontsize=8.2)
    res_thumb = SCRATCH_ECM_RES_THUMB if SCRATCH_ECM_RES_THUMB.exists() else HERE / "ecm_residual_support_claude.png"
    embed_image(ax, (c3a[0], c3a[1] - 0.85), res_thumb, zoom=0.0484)

    # --- Box 3b: D-HPROM-ANN direct state ---
    title_text(ax, c3b, w3, h3, "D--HPROM--ANN:\ndirect state")
    eq_text(ax, c3b[0], c3b[1] + 0.75, r"$\bm q=\bm q(\bm\varepsilon)$", fontsize=9.6)
    eq_text(ax, c3b[0], c3b[1] + 0.28, r"zero Newton iterations", fontsize=8.4)
    eq_text(ax, c3b[0], c3b[1] - 0.05, r"($\mathcal N$'s output used as-is)", fontsize=8.4)
    eq_text(ax, c3b[0], c3b[1] - 1.15, r"skip residual correction", fontsize=8.6, style="italic")

    # --- Box 4: PROM-ANN decoder / state map ---
    title_text(ax, c4, w4, h4, "PROM--ANN decoder:\nstate map")
    eq_text(ax, c4[0], c4[1] + 1.15,
            r"$\bm u_f\approx \bm u_f^{\rm aff}(\bm\varepsilon)+\mathbf V\,\bm q+\bar{\mathbf V}\,\mathcal N(\bm q)$",
            fontsize=8.8)
    thumb = SCRATCH_DISPLACEMENT_THUMB if SCRATCH_DISPLACEMENT_THUMB.exists() else HERE / "rve_geometry_claude.png"
    embed_image(ax, (c4[0], c4[1] - 0.55), thumb, zoom=0.145)

    # --- Box 5: hyper-reduced homogenization ---
    title_text(ax, c5, w5, h5, "hyper-reduced\nhomogenization")
    eq_text(ax, c5[0], c5[1] + 0.95,
            r"$\displaystyle\bm S_{\rm macro}(\bm\varepsilon)\approx\frac{1}{|\Omega_0|}\!\!\sum_{g\in\mathcal Z_\sigma}\!\!\bm S_g(\bm u_f;\bm\varepsilon)\,\omega_{\sigma,g}$",
            fontsize=8.6)
    sig_thumb = SCRATCH_ECM_SIG_THUMB if SCRATCH_ECM_SIG_THUMB.exists() else HERE / "ecm_stress_support_claude.png"
    embed_image(ax, (c5[0], c5[1] - 1.05), sig_thumb, zoom=0.0497)

    # --- arrows ---
    arrow(ax, (c1[0] + w1 / 2, c1[1]), (c2[0] - w2 / 2, c2[1]))
    arrow(ax, (c2[0] + w2 / 2, c2[1] + 0.55), (c3a[0] - w3 / 2, c3a[1] - 0.35))
    arrow(ax, (c2[0] + w2 / 2, c2[1] - 0.55), (c3b[0] - w3 / 2, c3b[1] + 0.35))
    arrow(ax, (c3a[0] + w3 / 2, c3a[1] - 0.35), (c4[0] - w4 / 2, c4[1] + 0.55))
    arrow(ax, (c3b[0] + w3 / 2, c3b[1] + 0.35), (c4[0] - w4 / 2, c4[1] - 0.55))
    arrow(ax, (c4[0] + w4 / 2, c4[1]), (c5[0] - w5 / 2, c5[1]))

    ax.set_xlim(-1.6, 14.6)
    ax.set_ylim(-4.2, 4.2)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(HERE / "workflow_diagram_claude.png", dpi=230, bbox_inches="tight")
    plt.close(fig)
    print("wrote workflow_diagram_claude.png")


if __name__ == "__main__":
    main()
