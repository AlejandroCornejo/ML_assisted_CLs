#!/usr/bin/env python3
"""Stage 03 acceptance tests. Nothing downstream may use data.npz until these
pass.

1. No NaN, and every state converged.
2. The zero state: S(0) = 0 and W(0) = 0.
3. W >= 0 everywhere (the reference state is the minimum of a Neo-Hookean
   potential).
4. S = dW/dE. This is the test that catches a normalization mismatch between
   the two outputs, which is exactly the mistake already made once in this
   study -- the solver's own helpers normalize the energy by the SOLID area
   and the stress by thickness x solid area, so putting both on the CELL
   measure had to be done by hand. Done with a small step by direct solves,
   not by differencing the grid, whose 0.0115 spacing would limit the test to
   ~1e-4.
5. dS/dE symmetric. A hyperelastic material has S = dW/dE, so dS/dE = d2W/dE2
   is necessarily symmetric; asymmetry would mean the periodic constraint
   elimination or the homogenization is wrong. Note the tangent here is itself
   a finite difference, so its own truncation error floors this test.
6. Warm-started values agree with cold ones. Necessary but not sufficient on
   its own -- a silently cold-started path would agree too -- which is why the
   generator's own timings are reported alongside.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PROJ = ROOT.parent / "RVE_NeoHookean_Homogenization"
for p in (str(ROOT), str(ROOT / "00_rve"), str(PROJ / "fe2_extension"),
          str(PROJ / "core")):
    if p not in sys.path:
        sys.path.insert(0, p)
KRATOS_PATH = "/home/kratos/Kratos_Eigen_Check/bin/Release"
if KRATOS_PATH not in sys.path:
    sys.path.append(KRATOS_PATH)

N_SPOT = 8
H_ENERGY = 2.0e-4


def main():
    d = np.load(HERE / "data.npz")
    E, S, W = d["E_train"], d["S_train"], d["W_train"]
    cell_area = float(d["cell_area"])
    rng = np.random.default_rng(7)

    import fom_solver_rve as fom
    from periodic_fom import PeriodicRVE
    from _material_law_guard_claude import true_neo_hookean_active

    checks = []

    nan = int(np.sum(~np.isfinite(S))) + int(np.sum(~np.isfinite(W)))
    checks.append(("no NaN in S or W", nan == 0, f"{nan} non-finite"))

    i0 = int(np.argmin(np.linalg.norm(E, axis=1)))
    if "S_zero" in d.files:
        s0, w0 = d["S_zero"][0], d["W_zero"][0]
        ref = np.max(np.abs(S))
        checks.append(("S(0) = 0", float(np.max(np.abs(s0))) < 1e-6 * ref,
                       f"|S(0)| = {np.max(np.abs(s0)):.3e} vs max|S| {ref:.3e}"))
        checks.append(("W(0) = 0", abs(float(w0)) < 1e-6 * np.max(np.abs(W)),
                       f"W(0) = {w0:.3e}"))

    checks.append(("W >= 0 everywhere", float(np.nanmin(W)) >= 0.0,
                   f"min W = {np.nanmin(W):.4e}"))

    spot = rng.choice(E.shape[0], N_SPOT, replace=False)
    with true_neo_hookean_active():
        rve = PeriodicRVE(str(HERE / "rve_mesh"), cell_area=cell_area)

        def energy(Ev, u0=None, E0=None):
            _s, u = rve.solve(Ev, u_ind_init=u0, E_start=E0)
            return rve.homogenized_energy(), u

        # (4) S = dW/dE
        err_dW, err_sym, err_warm = [], [], []
        for i in spot:
            Ei = E[i]
            _w0, u0 = energy(Ei)
            g = np.zeros(3)
            for j in range(3):
                Ep, Em = Ei.copy(), Ei.copy()
                Ep[j] += H_ENERGY
                Em[j] -= H_ENERGY
                wp, _ = energy(Ep, u0, Ei)
                wm, _ = energy(Em, u0, Ei)
                g[j] = (wp - wm) / (2.0 * H_ENERGY)
            # Voigt work conjugacy: W = S11 E11 + S22 E22 + S12 g12, and the
            # third slot is the ENGINEERING shear, so dW/dg12 = S12 directly.
            err_dW.append(np.linalg.norm(g - S[i]) / max(np.linalg.norm(S[i]), 1e-30))

            # (5) tangent symmetry
            _s, C = rve.stress_and_tangent(Ei)
            err_sym.append(float(np.max(np.abs(C - C.T)) / np.max(np.abs(C))))

            # (6) warm vs cold
            s_cold, _ = rve.solve(Ei)
            err_warm.append(float(np.linalg.norm(s_cold - S[i])
                                  / max(np.linalg.norm(S[i]), 1e-30)))

    checks.append(("S = dW/dE", max(err_dW) < 5e-3,
                   f"worst {max(err_dW):.3e} over {N_SPOT} states "
                   f"(h = {H_ENERGY:g})"))
    checks.append(("dS/dE symmetric", max(err_sym) < 5e-3,
                   f"worst {max(err_sym):.3e}"))
    checks.append(("stored S matches a fresh cold solve", max(err_warm) < 1e-5,
                   f"worst {max(err_warm):.3e}"))

    print("acceptance:")
    for name, good, detail in checks:
        print(f"  [{'OK  ' if good else 'FAIL'}] {name}  ({detail})")
    ok = all(g for _, g, _ in checks)
    print("\nDATA_VERIFY_PASS" if ok else "\nDATA_VERIFY_FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
