#!/usr/bin/env python3
"""Step 1 of the hierarchy: the LINEAR PROM. Galerkin projection on the
39-mode POD basis, full mesh, no hyperreduction and no network.

    u_ind = Phi q,     u = T Phi q + g(E),     (T Phi)^T f_int(u) = 0

39 unknowns instead of 6320. This has to be measured BEFORE the ECM is added,
otherwise the hyperreduction error cannot be separated from the projection
error -- one number with two causes.

What it establishes:

  * the cost of Galerkin projection alone, against the FOM
  * a reference the ECM must not degrade
  * a comparison point for the nonlinear-manifold route, which on the same
    basis solves 3 unknowns instead of 39 but needs a trained closure

Note the projection error here is NOT the same as the POD projection error
already measured (6.7e-07 on test states). That one projects the KNOWN FOM
solution onto the basis. This one solves the reduced equations, so its answer
is the Galerkin solution, which differs from the projection of the exact
solution.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import argparse
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PROJ = ROOT.parent / "RVE_NeoHookean_Homogenization"
for p in (str(ROOT), str(HERE), str(ROOT / "00_rve"), str(ROOT / "04_training"),
          str(PROJ / "fe2_extension"), str(PROJ / "core")):
    if p not in sys.path:
        sys.path.insert(0, p)
KRATOS_PATH = "/home/kratos/Kratos_Eigen_Check/bin/Release"
if KRATOS_PATH not in sys.path:
    sys.path.append(KRATOS_PATH)

NEWTON_TOL = 1.0e-11
NEWTON_MAX_IT = 40
SUBSTEPS_PER_UNIT_STRAIN = 200.0    # the FOM's own density, and it is needed.
# An earlier 60.0 was chosen on the reasoning that a reduced problem with 39
# unknowns instead of 6320 tolerates a coarser ramp. That reasoning is wrong:
# Newton's basin of attraction depends on the NONLINEARITY OF THE RESIDUAL,
# which is the same physics regardless of how many unknowns parameterize it.
# Measured cost of the mistake -- 11 of 200 evaluated states came out with
# errors between 6.8e-01 and 2.5e+01, two of them diverging outright, and
# every one of them dropped to between 1.9e-07 and 3.2e-04 at 200/unit, with
# 600/unit giving bit-identical results (so 200 is converged, not merely
# better). All 11 sat at strongly negative E22, the most nonlinear direction.


class LinearPROM:
    def __init__(self, rve, Phi):
        self.rve = rve
        self.Phi = np.ascontiguousarray(Phi)
        self.TPhi = np.asarray((rve.T @ self.Phi))     # (n_dof, r)
        self.r = self.Phi.shape[1]

    def solve(self, E, q_init=None, E_start=None, return_iters=False):
        rve = self.rve
        E = np.asarray(E, dtype=float).reshape(3)
        E0 = np.zeros(3) if E_start is None else np.asarray(E_start, float).reshape(3)
        n_sub = max(1, int(np.ceil(SUBSTEPS_PER_UNIT_STRAIN * np.linalg.norm(E - E0))))
        q = np.zeros(self.r) if q_init is None else q_init.copy()
        total_it = 0
        for k in range(1, n_sub + 1):
            Et = E0 + (E - E0) * (k / n_sub)
            g = rve._g(Et)
            for it in range(NEWTON_MAX_IT):
                u = self.TPhi @ q + g
                K, R = rve.assembler.Assemble(u)
                rr = self.TPhi.T @ R
                Kr = self.TPhi.T @ (K @ self.TPhi)
                dq = np.linalg.solve(Kr, rr)
                q = q + dq
                total_it += 1
                nrm = np.linalg.norm(dq) / max(np.linalg.norm(q), 1e-30)
                if nrm < NEWTON_TOL:
                    break
            else:
                raise RuntimeError(f"PROM Newton failed at substep {k}/{n_sub}, "
                                   f"E={Et}, rel dq {nrm:.3e}")
        # final state and its homogenized stress
        rve.assembler.Assemble(self.TPhi @ q + rve._g(E))
        S = rve.homogenized_stress(E)
        if return_iters:
            return S, q, total_it
        return S, q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-eval", type=int, default=100)
    a = ap.parse_args()

    b = np.load(ROOT / "04_training" / "decoder_basis_B_r39.npz")
    d = np.load(ROOT / "03_data" / "data.npz")
    Phi = b["Phi_ROM"]
    print(f"basis {Phi.shape[0]} x {Phi.shape[1]}")

    from periodic_fom import PeriodicRVE
    from _material_law_guard_claude import true_neo_hookean_active

    rng = np.random.default_rng(3)
    with true_neo_hookean_active():
        rve = PeriodicRVE(str(ROOT / "03_data" / "rve_mesh"),
                          cell_area=float(d["cell_area"]))
        prom = LinearPROM(rve, Phi)
        out = {}
        print(f"\n{'set':>6} {'n':>4} {'fail':>5} | {'median':>11} {'p90':>11} "
              f"{'max':>11} | {'its/solve':>9} {'s/solve':>8}")
        for nm in ("test", "probe"):
            E_s, S_s = d[f"E_{nm}"], d[f"S_{nm}"]
            ok = np.isfinite(S_s).all(axis=1)
            E_s, S_s = E_s[ok], S_s[ok]
            idx = rng.choice(E_s.shape[0], min(a.n_eval, E_s.shape[0]), replace=False)
            errs, iters, nfail = [], [], 0
            t0 = time.perf_counter()
            for i in idx:
                try:
                    S, _q, nit = prom.solve(E_s[i], return_iters=True)
                except RuntimeError:
                    nfail += 1
                    continue
                errs.append(np.linalg.norm(S - S_s[i]) / np.linalg.norm(S_s[i]))
                iters.append(nit)
            dt = time.perf_counter() - t0
            errs = np.array(errs)
            out[nm] = errs
            print(f"{nm:>6} {len(idx):4d} {nfail:5d} | {np.median(errs):11.4e} "
                  f"{np.percentile(errs, 90):11.4e} {errs.max():11.4e} | "
                  f"{np.mean(iters):9.1f} {dt / max(len(idx), 1):8.3f}", flush=True)

    np.savez_compressed(HERE / "linear_prom_errors.npz",
                        err_test=out["test"], err_probe=out["probe"])
    print("\nLINEAR_PROM_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
