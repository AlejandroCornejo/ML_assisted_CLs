#!/usr/bin/env python3
"""HPROM with the residual ECM rule only, swept over the integrand-basis
truncation, against the PROM.

WHY THE POINT COUNT IS WHAT IT IS. The ECM needs at least one point per
integrand basis vector, so the support size is set by that basis's RANK, not
by the ECM. Measured rank against truncation tolerance:

    svd_tol   rank   % of mesh
      1e-02     79       5.1
      3e-03    105       6.8
      1e-03    134       8.7
      1e-04    234      15.1
      1e-06    610      39.5

An initial svd_tol of 1e-06 was chosen by analogy with the displacement POD
tolerance and gave 610 points, i.e. 39% of the mesh and almost no reduction.
The analogy was wrong: the displacement field is smooth and 39-dimensional,
whereas the ELEMENT-WISE residual contributions vary element by element and
are genuinely high-rank. The slow decay is a property of the problem.

HOW THE STRESS IS COMPUTED HERE, and why it matters. The hyperreduced
assembler carries the RESIDUAL rule's weights. Asking it for the homogenized
stress would apply weights fitted for one integrand to another -- precisely
the previous project's S == 0 bug. So the solve uses the hyperreduced
assembler and the stress is then evaluated with the FULL assembler at the
converged displacement. That isolates what the cubature does to the SOLUTION,
uncontaminated by the absence of a stress rule, which is the next step.

Reference, PROM under identical conditions: displacement 1.90e-06, stress
4.77e-07 relative Frobenius, 0 failures, 4.99x over the FOM.
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

N_EVAL = 40
TOL = 1.0e-10
SVD_TOLS = (1e-2, 3e-3, 1e-3)

import periodic_fom as pf      # noqa: E402
import linear_prom as lp       # noqa: E402
pf.NEWTON_TOL = TOL
lp.NEWTON_TOL = TOL
lp.SUBSTEPS_PER_UNIT_STRAIN = pf.SUBSTEPS_PER_UNIT_STRAIN


class HPROM:
    """Reduced solve on a weighted element subset; stress from the full mesh."""

    def __init__(self, rve, Phi, z, w):
        import fom_solver_rve as fom
        self.rve = rve
        self.Phi = np.ascontiguousarray(Phi)
        self.TPhi = np.asarray(rve.T @ self.Phi)
        self.r = self.Phi.shape[1]
        elems = list(rve._mp.Elements)
        sub = [elems[int(i)] for i in z]
        self.asm = fom.VectorizedAssembler(
            rve._mp, rve.n_dof, rve._eq_map, elements=sub,
            element_scales=np.asarray(w, dtype=float),
            log_label="HyperReducedAssembler")

    def solve(self, E, E_start=None):
        rve = self.rve
        E = np.asarray(E, dtype=float).reshape(3)
        E0 = np.zeros(3) if E_start is None else np.asarray(E_start, float).reshape(3)
        n_sub = max(1, int(np.ceil(lp.SUBSTEPS_PER_UNIT_STRAIN
                                   * np.linalg.norm(E - E0))))
        q = np.zeros(self.r)
        for k in range(1, n_sub + 1):
            Et = E0 + (E - E0) * (k / n_sub)
            g = rve._g(Et)
            for _it in range(lp.NEWTON_MAX_IT):
                K, R = self.asm.Assemble(self.TPhi @ q + g)
                dq = np.linalg.solve(self.TPhi.T @ (K @ self.TPhi), self.TPhi.T @ R)
                q = q + dq
                if np.linalg.norm(dq) / max(np.linalg.norm(q), 1e-30) < lp.NEWTON_TOL:
                    break
            else:
                raise RuntimeError(f"HPROM Newton failed at substep {k}/{n_sub}")
        # Stress from the FULL assembler: the hyperreduced one carries the
        # RESIDUAL rule's weights and must not be asked for a different integral.
        rve.assembler.Assemble(self.TPhi @ q + rve._g(E))
        return rve.homogenized_stress(E), q


def rank_for(sv, tol):
    tail = np.cumsum(sv[::-1] ** 2)[::-1]
    total = np.sum(sv ** 2)
    for i in range(1, sv.size + 1):
        if np.sqrt((tail[i] if i < sv.size else 0.0) / total) <= tol:
            return i
    return sv.size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-eval", type=int, default=N_EVAL)
    a_ = ap.parse_args()

    b = np.load(ROOT / "04_training" / "decoder_basis_B_r39.npz")
    d = np.load(ROOT / "03_data" / "data.npz")
    rule = np.load(HERE / "ecm_residual_rule.npz")
    Phi, sv = b["Phi_ROM"], rule["sv"]

    from periodic_fom import PeriodicRVE
    from linear_prom import LinearPROM
    from _material_law_guard_claude import true_neo_hookean_active
    from empirical_cubature_method import EmpiricalCubatureMethod
    from ecm_residual import build_integrand_basis

    E_all, S_all = d["E_test"], d["S_test"]
    ok = np.isfinite(S_all).all(axis=1)
    E_all = E_all[ok]
    idx = np.random.default_rng(3).choice(E_all.shape[0], a_.n_eval, replace=False)
    U_train, E_train = d["U_train"], d["E_train"]
    okt = np.isfinite(U_train).all(axis=1)
    U_train, E_train = U_train[okt], E_train[okt]

    with true_neo_hookean_active():
        rve = PeriodicRVE(str(ROOT / "03_data" / "rve_mesh"),
                          cell_area=float(d["cell_area"]))
        prom = LinearPROM(rve, Phi)

        # PROM reference, same states, same run
        print("PROM reference on these states:", flush=True)
        Sp_ref, Up_ref, tp = [], [], []
        for i in idx:
            t0 = time.perf_counter()
            S, q = prom.solve(E_all[i])
            tp.append(time.perf_counter() - t0)
            Sp_ref.append(S)
            Up_ref.append(Phi @ q)
        Sp_ref, Up_ref, tp = np.array(Sp_ref), np.array(Up_ref), np.array(tp)
        print(f"  {len(idx)} states, {tp.sum():.1f}s total", flush=True)

        Ubasis_full, _ = build_integrand_basis(
            rve, prom.TPhi, U_train, E_train, int(rule["n_snap"]),
            1e-12, verbose=False)

        print(f"\n{'svd_tol':>8} {'rank':>5} {'elems':>6} {'%mesh':>6} | "
              f"{'disp err':>11} {'stress err':>11} | {'HPROM s':>8} "
              f"{'vs PROM':>8} {'vs FOM':>8}")
        rows = []
        for st in SVD_TOLS:
            r = rank_for(sv, st)
            ecm = EmpiricalCubatureMethod(ECM_tolerance=1e-6, Filter_tolerance=0.0,
                                          Plotting=False)
            ecm.SetUp(np.ascontiguousarray(Ubasis_full[:, :r]),
                      constrain_sum_of_weights=True)
            ecm.Run()
            z = np.asarray(ecm.z, dtype=np.int64).ravel()
            w = np.asarray(ecm.w, dtype=float).ravel()
            hp = HPROM(rve, Phi, z, w)
            Sh, Uh, th, nfail = [], [], [], 0
            for j, i in enumerate(idx):
                try:
                    t0 = time.perf_counter()
                    S, q = hp.solve(E_all[i])
                    th.append(time.perf_counter() - t0)
                except RuntimeError:
                    nfail += 1
                    continue
                Sh.append(S)
                Uh.append(Phi @ q)
            Sh, Uh, th = np.array(Sh), np.array(Uh), np.array(th)
            n = len(Sh)
            eu = np.linalg.norm(Uh - Up_ref[:n]) / np.linalg.norm(Up_ref[:n])
            es = np.linalg.norm(Sh - Sp_ref[:n]) / np.linalg.norm(Sp_ref[:n])
            print(f"{st:8.0e} {r:5d} {z.size:6d} {100*z.size/1546:5.1f}% | "
                  f"{eu:11.4e} {es:11.4e} | {th.sum():8.1f} "
                  f"{tp.sum()/th.sum():7.2f}x {4.99*tp.sum()/th.sum():7.2f}x"
                  + (f"  ({nfail} fail)" if nfail else ""), flush=True)
            rows.append((st, r, z.size, eu, es, th.sum()))
        np.savez_compressed(HERE / "hprom_residual_sweep.npz",
                            rows=np.array(rows, dtype=float), t_prom=tp.sum())
    print("\nHPROM_RESIDUAL_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
