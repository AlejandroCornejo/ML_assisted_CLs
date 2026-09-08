#!/usr/bin/env python3
"""Gate 4: the finite-strain identities the earlier gates could not see.

Gates 1-3 were all at small strain or against small-strain references (gate 3
compared the nonlinear tangent at E = 1e-5 against the linear C0). That let a
finite-strain KINEMATICS bug through: the periodic jump was being written with
the Green-Lagrange strain in place of F - I, which agree only to first order,
so the error was O(E^2) and vanished identically at the point every gate
tested.

Two identities are checked here, both of which hold for ANY hyperelastic
material solved to equilibrium under periodic boundary conditions, and both of
which FAIL if the kinematics are wrong:

  S = dW/dE     Hill-Mandel. Requires the homogenized stress and energy to be
                on the same measure and the micro problem to be the true
                equilibrium of the potential at that E.
  dS/dE = dS/dE^T   A potential's second derivative is symmetric. Asymmetry
                means the discrete problem is not the gradient of anything.

Both are checked at STRAINS THE STUDY ACTUALLY USES, not near zero.
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
for p in (str(ROOT), str(HERE), str(PROJ / "fe2_extension"), str(PROJ / "core")):
    if p not in sys.path:
        sys.path.insert(0, p)
KRATOS_PATH = "/home/kratos/Kratos_Eigen_Check/bin/Release"
if KRATOS_PATH not in sys.path:
    sys.path.append(KRATOS_PATH)

H_ENERGY = 2.0e-4

# Spread over the sampled box, including its corners in shear and transverse.
STATES = np.array([
    [0.05000, -0.02300, -0.01600],
    [0.10000, -0.04500, -0.03200],
    [0.15000, -0.06500, -0.04800],
    [0.19000, -0.09000, -0.16000],
    [0.19000, -0.09000, +0.10000],
    [0.05000, -0.09000, +0.10000],
])


def main():
    from periodic_fom import PeriodicRVE, _build
    from _material_law_guard_claude import true_neo_hookean_active
    import fom_solver_rve as fom

    base, geom = _build(tag="finite_gate")
    print(f"mesh {geom['n_elements']} elements, cell area {geom['block_area']}")

    with true_neo_hookean_active():
        rve = PeriodicRVE(base, cell_area=geom["block_area"])

        def energy(Ev, u0=None, E0=None):
            _s, u = rve.solve(Ev, u_ind_init=u0, E_start=E0)
            return rve.homogenized_energy(), u

        print(f"\n{'E':>34} | {'|S-dW/dE|/|S|':>13} | {'asym dS/dE':>11}")
        e_dw, e_sym = [], []
        for E in STATES:
            S, u0 = rve.solve(E)
            _w, _ = energy(E)
            g = np.zeros(3)
            for j in range(3):
                Ep, Em = E.copy(), E.copy()
                Ep[j] += H_ENERGY
                Em[j] -= H_ENERGY
                wp, _ = energy(Ep, u0, E)
                wm, _ = energy(Em, u0, E)
                g[j] = (wp - wm) / (2.0 * H_ENERGY)
            r1 = float(np.linalg.norm(g - S) / max(np.linalg.norm(S), 1e-30))
            _s, C = rve.stress_and_tangent(E)
            r2 = float(np.max(np.abs(C - C.T)) / np.max(np.abs(C)))
            e_dw.append(r1)
            e_sym.append(r2)
            print(f"{np.array2string(E, precision=5):>34} | {r1:13.3e} | {r2:11.3e}")

    checks = [("S = dW/dE at finite strain", max(e_dw) < 5e-3, f"worst {max(e_dw):.3e}"),
              ("dS/dE symmetric at finite strain", max(e_sym) < 5e-3,
               f"worst {max(e_sym):.3e}")]
    print("\nacceptance:")
    for name, good, detail in checks:
        print(f"  [{'OK  ' if good else 'FAIL'}] {name}  ({detail})")
    ok = all(g for _, g, _ in checks)
    print("\nGATE4_PASS" if ok else "\nGATE4_FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
