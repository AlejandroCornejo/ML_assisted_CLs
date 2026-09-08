#!/usr/bin/env python3
"""Step 3: the SECOND ECM rule, for the homogenized stress, and the complete
HPROM using both rules.

THE STRESS INTEGRAND is different from the residual's, and smaller:

    <P> = (1 / (t A0)) sum_e sum_g (F S)_eg w_detJ

which is 4 components, P not being symmetric -- the same n_cond = 4 Hernandez
declares for his homogenized PK1 output. Against the projected residual's 39
components, so a much lower-rank object is expected.

WHY A SECOND RULE AT ALL. The residual rule's weights reproduce
(T Phi)^T f_int. Applying them to <P> applies weights fitted for one integral
to another, which is precisely the previous project's S == 0 failure. The two
rules share nothing but the mesh.

COST OF SEPARATION, which is the objection to it: the stress assembly happens
ONCE per solve, not once per Newton iteration, so the added cost is
|z_stress| elements per solve against |z_residual| elements per iteration.
With ~160 Newton iterations per solve that is a percent-level overhead, which
is what makes the separated design nearly free -- the argument for it being
consistency with the MAW-ECM route, which is separated.

ACCEPTANCE. Both rules together, measured against the PROM on the same states
in the same run. The residual rule alone already gave a stress error of
1.07e-03 (80 elements) to 1.52e-04 (135 elements) with the stress computed on
the FULL mesh; adding the stress rule must not degrade that much, and its own
contribution is the difference.
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
RES_SVD_TOL = 1.0e-3        # 135 elements, the best of the residual sweep
SIG_SVD_TOLS = (1e-3, 1e-4, 1e-5)

import periodic_fom as pf      # noqa: E402
import linear_prom as lp       # noqa: E402
pf.NEWTON_TOL = TOL
lp.NEWTON_TOL = TOL
lp.SUBSTEPS_PER_UNIT_STRAIN = pf.SUBSTEPS_PER_UNIT_STRAIN


def build_stress_integrand(rve, U_train, E_train, n_snap, verbose=True):
    """Per-element contribution to <P>, 4 components, over a snapshot subset."""
    a = rve.assembler
    ne = a.n_elems
    step = max(1, U_train.shape[0] // n_snap)
    sel = np.arange(0, U_train.shape[0], step)[:n_snap]
    C = np.empty((ne, sel.size * 4))
    t0 = time.perf_counter()
    for k, j in enumerate(sel):
        a.Assemble(rve.T @ U_train[j] + rve._g(E_train[j]))
        Sv = a._S_voigt
        St = np.zeros(Sv.shape[:2] + (2, 2))
        St[..., 0, 0] = Sv[..., 0]
        St[..., 1, 1] = Sv[..., 1]
        St[..., 0, 1] = Sv[..., 2]
        St[..., 1, 0] = Sv[..., 2]
        P = np.matmul(a._F, St)                       # (ne, ng, 2, 2)
        c = np.einsum("eg,egij->eij", a.w_detJ, P).reshape(ne, 4) / rve.denom
        C[:, 4 * k:4 * (k + 1)] = c
    if verbose:
        print(f"  stress integrand matrix {C.shape} in "
              f"{time.perf_counter() - t0:.1f}s", flush=True)
    G = C @ C.T
    w, V = np.linalg.eigh(G)
    order = np.argsort(w)[::-1]
    sv = np.sqrt(np.clip(w[order], 0.0, None))
    return np.ascontiguousarray(V[:, order]), sv


def rank_for(sv, tol):
    tail = np.cumsum(sv[::-1] ** 2)[::-1]
    total = np.sum(sv ** 2)
    for i in range(1, sv.size + 1):
        if np.sqrt((tail[i] if i < sv.size else 0.0) / total) <= tol:
            return i
    return sv.size


def run_ecm(basis, tol=1e-6):
    from empirical_cubature_method import EmpiricalCubatureMethod
    ecm = EmpiricalCubatureMethod(ECM_tolerance=tol, Filter_tolerance=0.0,
                                  Plotting=False)
    ecm.SetUp(basis, constrain_sum_of_weights=True)
    ecm.Run()
    return (np.asarray(ecm.z, dtype=np.int64).ravel(),
            np.asarray(ecm.w, dtype=float).ravel())


def make_assembler(rve, z, w, label):
    import fom_solver_rve as fom
    elems = list(rve._mp.Elements)
    return fom.VectorizedAssembler(
        rve._mp, rve.n_dof, rve._eq_map,
        elements=[elems[int(i)] for i in z],
        element_scales=np.asarray(w, dtype=float), log_label=label)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-eval", type=int, default=N_EVAL)
    a_ = ap.parse_args()

    b = np.load(ROOT / "04_training" / "decoder_basis_B_r39.npz")
    d = np.load(ROOT / "03_data" / "data.npz")
    rule = np.load(HERE / "ecm_residual_rule.npz")
    Phi = b["Phi_ROM"]

    from periodic_fom import PeriodicRVE
    from linear_prom import LinearPROM
    from _material_law_guard_claude import true_neo_hookean_active
    from ecm_residual import build_integrand_basis

    E_all, S_all = d["E_test"], d["S_test"]
    ok = np.isfinite(S_all).all(axis=1)
    E_all = E_all[ok]
    idx = np.random.default_rng(3).choice(E_all.shape[0], a_.n_eval, replace=False)
    U_train, E_train = d["U_train"], d["E_train"]
    okt = np.isfinite(U_train).all(axis=1)
    U_train, E_train = U_train[okt], E_train[okt]
    n_snap = int(rule["n_snap"])

    with true_neo_hookean_active():
        rve = PeriodicRVE(str(ROOT / "03_data" / "rve_mesh"),
                          cell_area=float(d["cell_area"]))
        prom = LinearPROM(rve, Phi)

        print("PROM reference:", flush=True)
        Sp, tp = [], []
        Q_ref = []
        for i in idx:
            t0 = time.perf_counter()
            S, q = prom.solve(E_all[i])
            tp.append(time.perf_counter() - t0)
            Sp.append(S)
            Q_ref.append(q)
        Sp, tp = np.array(Sp), np.array(tp)
        print(f"  {len(idx)} states, {tp.sum():.1f}s", flush=True)

        print("\nresidual rule:", flush=True)
        Ures, sv_res = build_integrand_basis(rve, prom.TPhi, U_train, E_train,
                                             n_snap, 1e-12, verbose=False)
        r_res = rank_for(sv_res, RES_SVD_TOL)
        z_res, w_res = run_ecm(np.ascontiguousarray(Ures[:, :r_res]))
        asm_res = make_assembler(rve, z_res, w_res, "res")
        print(f"  rank {r_res} -> {z_res.size} elements "
              f"({100 * z_res.size / 1546:.1f}%)", flush=True)

        print("\nstress rule:", flush=True)
        Usig, sv_sig = build_stress_integrand(rve, U_train, E_train, n_snap)
        print(f"  {'sig_tol':>8} {'rank':>5} {'elems':>6} {'%mesh':>6}")
        for st in SIG_SVD_TOLS:
            print(f"  {st:8.0e} {rank_for(sv_sig, st):5d}", flush=True)

        print(f"\n{'sig_tol':>8} {'elems':>6} | {'stress err':>11} | "
              f"{'HPROM s':>8} {'vs PROM':>8} {'vs FOM':>8}")
        rows = []
        for st in SIG_SVD_TOLS:
            r_sig = rank_for(sv_sig, st)
            z_sig, w_sig = run_ecm(np.ascontiguousarray(Usig[:, :r_sig]))
            asm_sig = make_assembler(rve, z_sig, w_sig, "sig")
            Sh, th, nfail = [], [], 0
            for i in idx:
                E = E_all[i]
                try:
                    t0 = time.perf_counter()
                    n_sub = max(1, int(np.ceil(lp.SUBSTEPS_PER_UNIT_STRAIN
                                               * np.linalg.norm(E))))
                    q = np.zeros(Phi.shape[1])
                    for k in range(1, n_sub + 1):
                        Et = E * (k / n_sub)
                        g = rve._g(Et)
                        for _ in range(lp.NEWTON_MAX_IT):
                            K, R = asm_res.Assemble(prom.TPhi @ q + g)
                            dq = np.linalg.solve(prom.TPhi.T @ (K @ prom.TPhi),
                                                 prom.TPhi.T @ R)
                            q = q + dq
                            if (np.linalg.norm(dq)
                                    / max(np.linalg.norm(q), 1e-30) < lp.NEWTON_TOL):
                                break
                        else:
                            raise RuntimeError("HPROM Newton failed")
                    asm_sig.Assemble(prom.TPhi @ q + rve._g(E))
                    S = rve.homogenized_stress(E, assembler=asm_sig)
                    th.append(time.perf_counter() - t0)
                except RuntimeError:
                    nfail += 1
                    continue
                Sh.append(S)
            Sh, th = np.array(Sh), np.array(th)
            n = len(Sh)
            es = np.linalg.norm(Sh - Sp[:n]) / np.linalg.norm(Sp[:n])
            print(f"{st:8.0e} {z_sig.size:6d} | {es:11.4e} | {th.sum():8.1f} "
                  f"{tp.sum() / th.sum():7.2f}x {4.99 * tp.sum() / th.sum():7.2f}x"
                  + (f"  ({nfail} fail)" if nfail else ""), flush=True)
            rows.append((st, r_sig, z_sig.size, es, th.sum()))
        np.savez_compressed(HERE / "hprom_full_sweep.npz",
                            rows=np.array(rows, dtype=float),
                            z_res=z_res, w_res=w_res, sv_sig=sv_sig,
                            t_prom=tp.sum())
    print("\nHPROM_FULL_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
