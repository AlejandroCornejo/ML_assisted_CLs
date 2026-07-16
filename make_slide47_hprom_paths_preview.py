#!/usr/bin/env python3
"""Static visual draft for the WCCM 2026 HPROM-ANN bridge slide."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import image as mpimg
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parent
SUPPORT_DIR = ROOT / (
    "RVE_homogenization_NeoHookean_using_Kratos/"
    "stage_12_hprom_ann_mawecm_res_eps_sig_phase1to40_phase2to10_sum990_ann"
)
OUT = ROOT / "slide47_hprom_ann_paths_preview.png"

BLUE = "#1976D2"
ORANGE = "#D96B28"
INK = "#17202A"
MUTED = "#6C757D"
LIGHT_BLUE = "#EAF3FB"
LIGHT_ORANGE = "#FDF0E8"


def arrow(ax, start, end, color=INK, lw=2.6, rad=0.0, zorder=4):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="Simple,head_length=10,head_width=10,tail_width=2.2",
            connectionstyle=f"arc3,rad={rad}",
            color=color,
            linewidth=0,
            mutation_scale=1.0,
            zorder=zorder,
        )
    )


def box(ax, xy, width, height, title, body, edge=INK, face="white", title_color=INK):
    x, y = xy
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.012,rounding_size=0.015",
            facecolor=face,
            edgecolor=edge,
            linewidth=1.8,
            zorder=2,
        )
    )
    ax.text(x + width / 2, y + height * 0.69, title, ha="center", va="center",
            fontsize=18, fontweight="bold", color=title_color, zorder=3)
    ax.text(x + width / 2, y + height * 0.33, body, ha="center", va="center",
            fontsize=15, color=INK, zorder=3, linespacing=1.28)


def support_panel(fig, xywh, filename, title, color):
    ax = fig.add_axes(xywh)
    image = mpimg.imread(SUPPORT_DIR / filename)
    # Existing diagnostic images are cropped to retain the RVE mesh and supports only.
    ax.imshow(image[190:1270, 210:1670])
    ax.set_axis_off()
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(2.3)
        spine.set_edgecolor(color)
    ax.text(0.5, 1.06, title, transform=ax.transAxes, ha="center", va="bottom",
            fontsize=17, fontweight="bold", color=color)
    return ax


def main():
    fig = plt.figure(figsize=(16, 9), dpi=160, facecolor="white")
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()

    ax.text(0.5, 0.955, "Online hyper-reduced RVE response", ha="center", va="center",
            fontsize=30, fontweight="bold", color=INK)
    ax.plot((0.045, 0.955), (0.915, 0.915), color=BLUE, linewidth=2.3)

    box(ax, (0.045, 0.755), 0.16, 0.105,
        r"macroscale input", r"$\overline{\mathbf{F}}$", edge=BLUE, face=LIGHT_BLUE, title_color=BLUE)
    box(ax, (0.260, 0.755), 0.19, 0.105,
        r"Green--Lagrange strain", r"$\overline{\mathbf{E}}=\frac{1}{2}(\overline{\mathbf{F}}^T\overline{\mathbf{F}}-\mathbf{I})$" + "\n" + r"$\mu=(E_{xx},E_{yy},G_{xy})$",
        edge=BLUE, face=LIGHT_BLUE, title_color=BLUE)
    box(ax, (0.505, 0.755), 0.17, 0.105,
        r"primary coordinates", r"$\mathbf{q}^0(\mu)$", edge=BLUE, face=LIGHT_BLUE, title_color=BLUE)
    box(ax, (0.730, 0.735), 0.225, 0.145,
        r"PROM--ANN RVE state", r"$\widetilde{\mathbf{u}}=\mathbf{u}_{\rm aff}+\mathbf{V}\mathbf{A}\mathbf{q}$" + "\n" + r"$\quad+\overline{\mathbf{V}}\mathcal{N}_\theta(\mathbf{q})$",
        edge=INK, face="white", title_color=INK)

    arrow(ax, (0.205, 0.807), (0.260, 0.807), BLUE)
    arrow(ax, (0.450, 0.807), (0.505, 0.807), BLUE)
    arrow(ax, (0.675, 0.807), (0.730, 0.807), BLUE)

    ax.text(0.51, 0.685, "two online paths", ha="center", va="center", fontsize=16,
            fontweight="bold", color=MUTED)
    arrow(ax, (0.842, 0.735), (0.842, 0.610), INK)

    box(ax, (0.070, 0.500), 0.320, 0.120,
        "HPROM-ANN", r"correct $\mathbf{q}$ on the reduced manifold" + "\n" + r"$\mathbf{J}(\mathbf{q})^T\mathbf{r}_{\rm HR}(\mathbf{q})\approx\mathbf{0}$",
        edge=ORANGE, face=LIGHT_ORANGE, title_color=ORANGE)
    box(ax, (0.600, 0.500), 0.320, 0.120,
        "D-HPROM-ANN", r"accept $\mathbf{q}^0$ directly" + "\n" + "no equilibrium correction",
        edge=BLUE, face=LIGHT_BLUE, title_color=BLUE)
    arrow(ax, (0.825, 0.610), (0.760, 0.620), BLUE)
    arrow(ax, (0.805, 0.610), (0.390, 0.555), ORANGE, rad=0.12)

    support_panel(
        fig,
        (0.075, 0.180, 0.220, 0.205),
        "Z_union_selected_elements_Z_res.png",
        r"residual support  $\mathcal{Z}_{\rm res}$",
        ORANGE,
    )
    support_panel(
        fig,
        (0.415, 0.180, 0.220, 0.205),
        "Z_union_selected_elements_Z_sig.png",
        r"stress support  $\mathcal{Z}_{\sigma}$",
        BLUE,
    )

    arrow(ax, (0.230, 0.500), (0.190, 0.390), ORANGE)
    arrow(ax, (0.600, 0.500), (0.535, 0.390), BLUE, rad=0.12)
    arrow(ax, (0.295, 0.280), (0.415, 0.280), ORANGE)
    arrow(ax, (0.190, 0.450), (0.188, 0.515), ORANGE, lw=2.0, rad=0.55)

    box(ax, (0.730, 0.220), 0.225, 0.125,
        r"macroscopic response", r"$\overline{\mathbf{S}}=(\overline{S}_{xx},\overline{S}_{yy},\overline{S}_{xy})$" + "\n" + "weighted RVE average",
        edge=BLUE, face=LIGHT_BLUE, title_color=BLUE)
    arrow(ax, (0.635, 0.280), (0.730, 0.280), BLUE)

    ax.text(0.185, 0.105,
            r"$\mathbf{r}_{\rm HR}=\sum_{e\in\mathcal{Z}_{\rm res}} w_e^{\rm res}\,\mathbf{r}_e$" + "    (10 / 990 elements)",
            ha="center", va="center", fontsize=13.5, color=INK)
    ax.text(0.525, 0.105,
            r"$\overline{\mathbf{S}}_{\rm HR}\approx A_0^{-1}\sum_{e\in\mathcal{Z}_{\sigma}} w_e^{\sigma} A_e\langle\mathbf{S}_e\rangle$" + "    (10 / 990 elements)",
            ha="center", va="center", fontsize=13.0, color=INK)
    ax.text(0.5, 0.045,
            "Only HPROM-ANN evaluates the residual support.  Both routes use the stress support for homogenization.",
            ha="center", va="center", fontsize=14.5, color=INK)

    fig.savefig(OUT, dpi=160, facecolor="white")
    print(OUT)


if __name__ == "__main__":
    main()
