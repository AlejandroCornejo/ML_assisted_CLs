#!/usr/bin/env python3
"""Isolates the volumetric barrier h(J) = -r*log(J) + (beta/2)*(J-1)^2 on
its own, with no ICNN/feature terms and no coupling to a stretching path.
This is the figure Section 5.4 (C6) actually needs: the existing
polyconvex_volumetric_claude.pdf plots the *total* trained energy along
F = diag(J, 1), where growth for large J is dominated by the sixth-power
structural-stretch terms (Proposition 4), not by h(J) itself. Isolating
h(J) shows the honest shape: -r*log(J) alone decreases without bound as
J -> infinity, and only the quadratic term eventually reverses that,
at a crossover J* found by solving h'(J) = 0 for J > 1.
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
METRICS = HERE / "results" / "polyconvex_final_claude_metrics.json"


def find_crossover(r: float, beta: float) -> float:
    """Root of h'(J) = -r/J + beta*(J-1) = 0 for J > 1, by bisection."""
    def hprime(J: float) -> float:
        return -r / J + beta * (J - 1.0)

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
    metrics = json.loads(METRICS.read_text(encoding="utf-8"))
    cert = metrics["guarantee_audit"]["analytic_polyconvex_certificate"]
    r = cert["barrier_coefficient"]
    beta = cert["quadratic_volumetric_coefficient"]
    j_star = find_crossover(r, beta)

    def h(J):
        return -r * np.log(J) + 0.5 * beta * (J - 1.0) ** 2

    def log_part(J):
        return -r * np.log(J)

    def quad_part(J):
        return 0.5 * beta * (J - 1.0) ** 2

    fig, (ax_zoom, ax_wide) = plt.subplots(1, 2, figsize=(9.3, 3.6))

    J_zoom = np.linspace(0.05, 6.0, 500)
    ax_zoom.plot(J_zoom, h(J_zoom), color="#1f5fa8", linewidth=2.0, label=r"$h(J)$")
    ax_zoom.plot(J_zoom, log_part(J_zoom), color="#d62728", linewidth=1.4, linestyle="--",
                 label=r"$-r\log J$ alone")
    ax_zoom.axvline(1.0, color="#999999", linewidth=0.7, linestyle=":")
    ax_zoom.set_ylim(-2, 5)
    ax_zoom.set_xlabel(r"$J$")
    ax_zoom.set_ylabel(r"$h(J)$")
    ax_zoom.set_title(r"Collapse barrier ($J\to0^+$)", fontsize=10.5)
    ax_zoom.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax_zoom.grid(alpha=0.25)

    J_wide = np.logspace(0, np.log10(j_star) + 0.3, 800)
    ax_wide.plot(J_wide, h(J_wide), color="#1f5fa8", linewidth=2.0, label=r"$h(J)$")
    ax_wide.plot(J_wide, log_part(J_wide), color="#d62728", linewidth=1.4, linestyle="--",
                 label=r"$-r\log J$ alone")
    ax_wide.plot(J_wide, quad_part(J_wide), color="#2ca02c", linewidth=1.4, linestyle="--",
                 label=r"$\tfrac\beta2(J-1)^2$ alone")
    ax_wide.axvline(j_star, color="#999999", linewidth=0.9, linestyle=":")
    ax_wide.set_xscale("log")
    ax_wide.set_xlabel(r"$J$ (log scale)")
    ax_wide.set_ylabel(r"$h(J)$")
    ax_wide.set_title(rf"Crossover at $J^\ast\approx{j_star:.0f}$", fontsize=10.5)
    ax_wide.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax_wide.grid(alpha=0.25, which="both")

    fig.tight_layout()
    fig.savefig(HERE / "volumetric_isolated_claude.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote volumetric_isolated_claude.pdf, r={r}, beta={beta}, J*={j_star}")


if __name__ == "__main__":
    main()
