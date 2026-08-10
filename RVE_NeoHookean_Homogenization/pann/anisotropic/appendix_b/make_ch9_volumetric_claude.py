#!/usr/bin/env python3
"""New Chapter 9 (Super Appendix B): the volumetric barrier
h(J) = -r*log(J) + (beta/2)*(J-1)^2, evaluated with the real trained
model's r and beta. Two panels: a zoom near J=1 showing the collapse
barrier (J->0+) and the log-dominated, *decreasing* branch for J>1;
and a wide-range log-x view showing where the quadratic term actually
overtakes the logarithm (far beyond any J this RVE ever sees).
"""
from __future__ import annotations

import json
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
METRICS = HERE.parent / "results" / "polyconvex_final_claude_metrics.json"

_cert = json.loads(METRICS.read_text(encoding="utf-8"))["guarantee_audit"]["analytic_polyconvex_certificate"]
R = _cert["barrier_coefficient"]
BETA = _cert["quadratic_volumetric_coefficient"]


def h(J):
    return -R * np.log(J) + 0.5 * BETA * (J - 1.0) ** 2


def log_part(J):
    return -R * np.log(J)


def quad_part(J):
    return 0.5 * BETA * (J - 1.0) ** 2


def find_crossover() -> float:
    """Root of h'(J) = -R/J + BETA*(J-1) = 0 for J > 1, by bisection."""
    def hprime(J):
        return -R / J + BETA * (J - 1.0)

    lo, hi = 1.0, 1.0
    while hprime(hi) < 0.0:
        hi *= 2.0
    while hi / lo > 1.0 + 1e-12:
        mid = 0.5 * (lo + hi)
        if hprime(mid) < 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def main() -> None:
    j_star = find_crossover()
    fig, (ax_zoom, ax_wide) = plt.subplots(1, 2, figsize=(11.0, 4.6))

    J_zoom = np.linspace(0.05, 6.0, 500)
    ax_zoom.plot(J_zoom, h(J_zoom), color="#1f5fa8", linewidth=2.2, label=r"$h(J)$ total")
    ax_zoom.plot(J_zoom, log_part(J_zoom), color="#d62728", linewidth=1.6, linestyle="--",
                 label=r"$-r\log J$ solo")
    ax_zoom.axvline(1.0, color="#999999", linewidth=0.8, linestyle=":")
    ax_zoom.set_ylim(-2, 5)
    ax_zoom.set_xlabel(r"$J$")
    ax_zoom.set_ylabel(r"$h(J)$")
    ax_zoom.set_title(r"Cerca de $J=1$: la barrera de colapso" "\n" r"($J\to0^+$ diverge)")
    ax_zoom.legend(loc="upper right", fontsize=8.5, framealpha=0.9)

    J_wide = np.logspace(0, np.log10(j_star) + 0.3, 800)
    ax_wide.plot(J_wide, h(J_wide), color="#1f5fa8", linewidth=2.2, label=r"$h(J)$ total")
    ax_wide.plot(J_wide, log_part(J_wide), color="#d62728", linewidth=1.6, linestyle="--",
                 label=r"$-r\log J$ solo (decrece sin l\'imite)")
    ax_wide.plot(J_wide, quad_part(J_wide), color="#2ca02c", linewidth=1.6, linestyle="--",
                 label=r"$\tfrac\beta2(J-1)^2$ solo")
    ax_wide.axvline(j_star, color="#999999", linewidth=1.0, linestyle=":")
    ax_wide.text(j_star, 0, rf"$J\approx{j_star:.0f}$", rotation=90,
                 va="bottom", ha="right", fontsize=8.5, color="#555555")
    ax_wide.set_xscale("log")
    ax_wide.set_xlabel(r"$J$ (escala log)")
    ax_wide.set_ylabel(r"$h(J)$")
    ax_wide.set_title(r"Rango amplio: d\'onde el cuadr\'atico" "\n" r"realmente le gana al logaritmo")
    ax_wide.legend(loc="upper left", fontsize=8.5, framealpha=0.9)

    fig.suptitle(rf"La barrera volum\'etrica real: $r={R:.4f}$, $\beta={BETA:.4e}$",
                 fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(HERE / "ch9_volumetric_claude.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("wrote ch9_volumetric_claude.png")
    print(f"J* = {j_star}")
    print("turning check, h'(J*)=", -R / j_star + BETA * (j_star - 1))


if __name__ == "__main__":
    main()
