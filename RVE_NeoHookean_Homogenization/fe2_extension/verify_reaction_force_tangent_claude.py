#!/usr/bin/env python3
"""Verify the new analytic d(hom_sig)/dE (reaction_force_hom_tangent_claude.
py's reaction_force_hom_sig_and_jacobian, wired into both
DHpromAnnDirectLawFloat64.evaluate_with_tangent and
HpromAnnIterativeLawFloat64.evaluate_with_tangent) against a finite
difference of the already-verified value path (evaluate(), confirmed this
session to reproduce Table 6 at 0.51%/0.59% -- see
evaluate_table6_new_hrom_meshes_claude.py). Standard cross-check for a
hand-derived tangent: if analytic and FD disagree, something in the
Jacobian chain (d(sens)/dE curvature term or d(f_int)/dE product-rule
term) is wrong.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for sub in ("core", "prom/ann", "hprom/ann"):
    p = str(ROOT / sub)
    if p not in sys.path:
        sys.path.insert(0, p)

from dhprom_ann_direct_law_float64_claude import DHpromAnnDirectLawFloat64  # noqa: E402
from hprom_ann_iterative_law_float64_claude import HpromAnnIterativeLawFloat64  # noqa: E402

DHPROMANN_DIR = HERE / "maw_dynamic_reaction_force_dhpromann"
HPROMANN_DIR = HERE / "maw_dynamic_reaction_force_hpromann"

TEST_STATES = [
    np.array([0.0, 0.0, 0.0]),
    np.array([0.02, 0.0, 0.0]),
    np.array([0.03, -0.02, 0.015]),
    np.array([-0.025, 0.03, -0.01]),
    np.array([0.05, 0.04, 0.03]),
]

FAR_STATES = [
    np.array([0.6, 0.6, 0.6]),
    np.array([0.5, 0.5, -0.5]),
    np.array([0.6, -0.4, 0.45]),
]


def fd_jacobian_dhpromann(law, E, h=1.0e-5):
    J = np.zeros((3, 3), dtype=float)
    for k in range(3):
        Ep, Em = E.copy(), E.copy()
        Ep[k] += h
        Em[k] -= h
        _, sp = law.evaluate(Ep)
        _, sm = law.evaluate(Em)
        J[:, k] = (sp - sm) / (2.0 * h)
    return J


def fd_jacobian_hpromann(law, E, h=1.0e-5):
    J = np.zeros((3, 3), dtype=float)
    for k in range(3):
        Ep, Em = E.copy(), E.copy()
        Ep[k] += h
        Em[k] -= h
        _, sp, _, _, _ = law.evaluate(Ep)
        _, sm, _, _, _ = law.evaluate(Em)
        J[:, k] = (sp - sm) / (2.0 * h)
    return J


def rel_err(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    den = max(np.linalg.norm(b), 1.0e-30)
    return float(np.linalg.norm(a - b) / den)


def main():
    print("=== D-HPROM-ANN analytic vs FD tangent ===")
    law_d = DHpromAnnDirectLawFloat64(hprom_ann_dir=str(DHPROMANN_DIR))
    for E in TEST_STATES:
        hom_eps, hom_sig, dEps_dE, dSig_dE = law_d.evaluate_with_tangent(E)
        J_fd = fd_jacobian_dhpromann(law_d, E)
        err = rel_err(dSig_dE, J_fd)
        print(f"  E={E}: |analytic|={np.linalg.norm(dSig_dE):.4e}, |FD|={np.linalg.norm(J_fd):.4e}, "
              f"rel_err={err:.3e}")

    print("\n=== D-HPROM-ANN analytic vs FD tangent, far outside training range ===")
    for E in FAR_STATES:
        hom_eps, hom_sig, dEps_dE, dSig_dE = law_d.evaluate_with_tangent(E)
        J_fd = fd_jacobian_dhpromann(law_d, E)
        err = rel_err(dSig_dE, J_fd)
        print(f"  E={E} (|E|={np.linalg.norm(E):.3f}): |analytic|={np.linalg.norm(dSig_dE):.4e}, "
              f"|FD|={np.linalg.norm(J_fd):.4e}, rel_err={err:.3e}")

    print("\n=== HPROM-ANN analytic vs FD tangent ===")
    law_h = HpromAnnIterativeLawFloat64(hprom_ann_dir=str(HPROMANN_DIR))
    for E in TEST_STATES:
        hom_eps, hom_sig, q_p, n_it, conv, dEps_dE, dSig_dE, hom_eps_chk, hom_sig_chk = \
            law_h.evaluate_with_tangent(E, q_prev=None, step_index=1)
        J_fd = fd_jacobian_hpromann(law_h, E)
        err = rel_err(dSig_dE, J_fd)
        value_consistency = rel_err(hom_sig_chk, hom_sig)
        print(f"  E={E}: |analytic|={np.linalg.norm(dSig_dE):.4e}, |FD|={np.linalg.norm(J_fd):.4e}, "
              f"rel_err={err:.3e}, value_check_vs_hom_sig={value_consistency:.3e}")


if __name__ == "__main__":
    main()
