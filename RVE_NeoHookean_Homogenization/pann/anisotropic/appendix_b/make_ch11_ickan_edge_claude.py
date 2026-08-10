#!/usr/bin/env python3
"""Chapter 11 (Super Appendix B): a single real learned ICKAN edge
(spline), evaluated from the actual trained checkpoint (not a stylized
illustration), confirming visually that it is simultaneously convex and
non-decreasing across its whole domain -- the same two properties an
ICNN's Softplus neuron has (Chapter 9), reached by a completely
different mechanism (a doubly-cumulative-sum reparameterization of
B-spline control points instead of softplus-reparameterized weights).
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}\usepackage{bm}",
})

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, "/home/kratos/ICKANs")


def main() -> None:
    from anisotropic_pann_model_ickan_claude import load_anisotropic_polyconvex_ickan
    from ickan.spline import coef2curve

    model, strain_scale, energy_scale, ckpt = load_anisotropic_polyconvex_ickan(
        REPO_ROOT / "checkpoints" / "PANN_anisotropic_polyconvex_ickan_final_claude.pt",
        torch.device("cpu"),
    )
    model.eval()
    layer0 = model.base_icnn.core.act_fun[0]

    x = torch.linspace(0.0, 1.3, 400, dtype=torch.float64).unsqueeze(1).repeat(1, 15)
    y_eval, coef_convex = coef2curve(x, layer0.grid, layer0.coef, k=layer0.k)
    edge = y_eval[:, 0, 0].detach().numpy()
    xs = x[:, 0].detach().numpy()

    control_points = coef_convex[0, 0, :].detach().numpy()
    grid = layer0.grid[0].detach().numpy()
    k = layer0.k
    # B-spline control points live on the interior knot sequence for this order.
    n_ctrl = len(control_points)
    ctrl_x = np.linspace(grid[k], grid[-k - 1], n_ctrl)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))

    axes[0].plot(xs, edge, color="#1f5fa8", linewidth=2.2, label=r"curva $\phi(x)$ evaluada")
    axes[0].scatter(ctrl_x, control_points, color="#d62728", s=28, zorder=5,
                     label="puntos de control (reparametrizados)")
    axes[0].set_title(r"Un borde real de la red ICKAN entrenada" "\n" r"(entrada 0, neurona 0 de la capa 1)")
    axes[0].set_xlabel(r"$x$ (entrada, normalizada)")
    axes[0].set_ylabel(r"$\phi(x)$")
    axes[0].legend(loc="upper left", fontsize=8.5, framealpha=0.9)

    diffs = np.diff(edge)
    axes[1].plot(xs[:-1], diffs, color="#2ca02c", linewidth=2.0)
    axes[1].axhline(0, color="#999999", linewidth=1.0, linestyle=":")
    axes[1].set_title(r"Pendiente local $\Delta\phi$" "\n" r"(no-negativa en todo el dominio: no-decreciente)")
    axes[1].set_xlabel(r"$x$")
    axes[1].set_ylabel(r"$\phi(x_{i+1})-\phi(x_i)$")

    fig.suptitle(r"Un ICKAN construye convexidad con puntos de control, no con pesos",
                 fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(HERE / "ch11_ickan_edge_claude.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("wrote ch11_ickan_edge_claude.png")
    print("control points:", control_points)
    print("min first-diff (monotone):", diffs.min())
    print("min second-diff (convex):", np.diff(diffs).min())


if __name__ == "__main__":
    main()
