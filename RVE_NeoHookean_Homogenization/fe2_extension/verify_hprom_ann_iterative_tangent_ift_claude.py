#!/usr/bin/env python3
"""Decisive, isolated check of the ONE genuinely new and risky piece of
HpromAnnIterativeLaw.evaluate_with_tangent: dq_p/dE via the implicit
function theorem through the inner loop's OWN fixed point q_p*(E), where
q_p*(E) satisfies r_r(q_p*(E), E) = 0 (not an explicit formula like Stage
1's mu-affine q_p(E)).

Ground truth here is NOT another analytic derivation -- it is the inner
loop itself: run _evaluate_impl (the already-verified, unmodified Newton
loop) to convergence at E+h*e_k and E-h*e_k for each macro-strain
component k, and central-difference the returned q_p directly. This
sidesteps every other moving part (decoder, homogenization weights,
geometry) entirely -- it only tests whether dqp_dE_at's IFT formula
predicts how the CONVERGED REDUCED COORDINATE itself moves when E moves,
which is exactly what the implicit function theorem is a claim about.
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

from hprom_ann_iterative_law_claude import HpromAnnIterativeLaw


def main():
    print("[verify-ift] building HpromAnnIterativeLaw ...")
    law = HpromAnnIterativeLaw()

    test_states = [
        np.array([0.05, 0.0, 0.0]),
        np.array([0.3, -0.1, 0.05]),
        np.array([0.8, 0.4, -0.05]),
        np.array([1.2, 0.6, 0.06]),
    ]

    all_ok = True
    for E in test_states:
        hom_eps, hom_sig, q_p0, n_it, conv = law._evaluate_impl(E)
        print(f"\nE={E}: q_p0={q_p0}, n_it={n_it}, converged={conv}")

        u_aff_free = law._affine(E, law.x_free, law.y_free, law.is_x_free)
        disp_base = np.zeros(law.n_total_dof, dtype=float)
        disp_base[law.dir_dofs] = law._affine(E, law.x_dir, law.y_dir, law.is_x_dir)
        dq_p_dE_analytic, J_manifold, K_r, w_res_iter = law.dqp_dE_at(q_p0, E, disp_base, u_aff_free)
        print(f"    dq_p/dE (analytic, IFT) =\n{dq_p_dE_analytic}")
        print(f"    cond(K_r)={np.linalg.cond(K_r):.3e}")

        dq_p_dE_fd = np.zeros((law.n_primary, 3), dtype=float)
        for h in (1.0e-3, 1.0e-4):
            for k in range(3):
                Ep, Em = E.copy(), E.copy()
                Ep[k] += h
                Em[k] -= h
                _, _, q_p_p, _, conv_p = law._evaluate_impl(Ep)
                _, _, q_p_m, _, conv_m = law._evaluate_impl(Em)
                if not (conv_p and conv_m):
                    print(f"    WARNING: inner loop did not converge at Ep/Em for k={k}, h={h}")
                dq_p_dE_fd[:, k] = (q_p_p - q_p_m) / (2 * h)
            rel_err = np.linalg.norm(dq_p_dE_analytic - dq_p_dE_fd) / max(np.linalg.norm(dq_p_dE_fd), 1e-30)
            ok = rel_err < 5e-2
            all_ok = all_ok and ok
            print(f"    h={h:.0e}: dq_p/dE (FD)=\n{dq_p_dE_fd}\n"
                  f"    rel_err={rel_err:.3e} [{'ok' if ok else 'FAIL'}]")

    print(f"\n[verify-ift] {'PASS' if all_ok else 'FAIL'}")


if __name__ == "__main__":
    main()
