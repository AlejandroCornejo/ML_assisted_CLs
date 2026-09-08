#!/usr/bin/env python3
"""HPROM on genuinely REDUCED meshes for both rules, against the PROM.

Replaces the earlier version that kept the full model part and passed an
element subset. That was correct but carried avoidable cost: the global sparse
system was still 6480 x 6480. Here each rule gets its own reduced mesh, so the
residual system is a few hundred dofs and the stress mesh a few hundred more.

Correctness rests on one identity, checked below rather than assumed: the
projected residual assembled on the reduced mesh must equal the one assembled
from the same element subset on the full mesh,

    (T Phi)_red^T R_red  ==  sum_{e in z} (T Phi)_e^T f_int,e

to roundoff. If the reduced-to-full dof map by node id were wrong, this is
what would catch it -- and nothing else would, since both routes would still
produce plausible numbers.
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

import periodic_fom as pf      # noqa: E402
import linear_prom as lp       # noqa: E402
pf.NEWTON_TOL = TOL
lp.NEWTON_TOL = TOL
lp.SUBSTEPS_PER_UNIT_STRAIN = pf.SUBSTEPS_PER_UNIT_STRAIN


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-eval", type=int, default=N_EVAL)
    a_ = ap.parse_args()

    b = np.load(ROOT / "04_training" / "decoder_basis_B_r39.npz")
    d = np.load(ROOT / "03_data" / "data.npz")
    sup = np.load(HERE / "ecm_supports.npz")
    sw = np.load(HERE / "hprom_full_sweep.npz")
    Phi = b["Phi_ROM"]

    # NO REMAPPING. build_ecm_supports.py guarantees z_* and w_* are aligned
    # with each other and self-checks that the sorted and ECM-order pairings
    # agree, so the arrays are used directly.
    #
    # The remapping this replaces was written to fix an alignment bug and
    # introduced a different one: it paired `z_sig_ecm_order` (ECM order) with
    # `w_sig` (already sorted). Both the bug and its "fix" reported 1.2503e-01,
    # identical to five digits, which is what revealed the fix had not taken
    # effect. The lesson is to make the stored data self-consistent and consume
    # it directly rather than to re-derive an ordering at each use site.
    z_res = np.asarray(sup["z_res"], dtype=np.int64)
    w_res = np.asarray(sup["w_res"], dtype=float)
    z_sig = np.asarray(sup["z_sig"], dtype=np.int64)
    w_sig = np.asarray(sup["w_sig"], dtype=float)

    # Cross-check the pairing against the ECM-order arrays, which catches any
    # index/weight mismatch no matter which pair a consumer picks.
    for nm in ("res", "sig"):
        d1 = dict(zip(np.asarray(sup[f"z_{nm}_ecm_order"]).tolist(),
                      np.asarray(sup[f"w_{nm}_ecm_order"]).tolist()))
        d2 = dict(zip(np.asarray(sup[f"z_{nm}"]).tolist(),
                      np.asarray(sup[f"w_{nm}"]).tolist()))
        if d1 != d2:
            raise RuntimeError(f"{nm} index/weight pairing differs between the "
                               f"two stored orders")
    print(f"weight pairing cross-check OK  "
          f"(res {z_res.size}, sig {z_sig.size} elements)", flush=True)

    from periodic_fom import PeriodicRVE
    from linear_prom import LinearPROM
    from _material_law_guard_claude import true_neo_hookean_active
    from reduced_mesh import ReducedAssembly
    import fom_solver_rve as fom

    E_all, S_all = d["E_test"], d["S_test"]
    ok = np.isfinite(S_all).all(axis=1)
    E_all = E_all[ok]
    idx = np.random.default_rng(3).choice(E_all.shape[0], a_.n_eval, replace=False)
    full_mdpa = str(ROOT / "03_data" / "rve_mesh") + ".mdpa"

    with true_neo_hookean_active():
        rve = PeriodicRVE(str(ROOT / "03_data" / "rve_mesh"),
                          cell_area=float(d["cell_area"]))
        prom = LinearPROM(rve, Phi)

        red_res = ReducedAssembly(full_mdpa, HERE / "mesh_res", z_res, w_res,
                                  rve, prom.TPhi)
        red_sig = ReducedAssembly(full_mdpa, HERE / "mesh_sig", z_sig, w_sig,
                                  rve, prom.TPhi)
        print(f"residual mesh: {red_res.n_elements} elements, "
              f"{red_res.n_nodes} nodes, {red_res.n_dof} dofs "
              f"(full: 1546 / 3240 / {rve.n_dof})")
        print(f"stress   mesh: {red_sig.n_elements} elements, "
              f"{red_sig.n_nodes} nodes, {red_sig.n_dof} dofs", flush=True)

        # --- consistency of the reduced-to-full dof map ---
        Etest = E_all[idx[0]]
        g = rve._g(Etest)
        q0 = np.zeros(Phi.shape[1])
        _K, R_red = red_res.assemble(q0, g)
        r_red = red_res.TPhi.T @ R_red
        elems = list(rve._mp.Elements)
        sub = fom.VectorizedAssembler(
            rve._mp, rve.n_dof, rve._eq_map,
            elements=[elems[int(i)] for i in z_res],
            element_scales=w_res, log_label="subset")
        _K2, R_sub = sub.Assemble(prom.TPhi @ q0 + g)
        r_sub = prom.TPhi.T @ R_sub
        rel = np.linalg.norm(r_red - r_sub) / max(np.linalg.norm(r_sub), 1e-300)
        print(f"\nreduced vs subset projected residual: {rel:.3e}  "
              f"{'OK' if rel < 1e-12 else 'FAIL'}", flush=True)
        if rel >= 1e-12:
            print("dof map is wrong; aborting")
            return 1

        # --- consistency of the STRESS rule, the check whose absence let a
        # weight misalignment through ---
        red_sig.assemble(q0, g)
        S_red = rve.homogenized_stress(Etest, assembler=red_sig.asm)
        sub_sig = fom.VectorizedAssembler(
            rve._mp, rve.n_dof, rve._eq_map,
            elements=[elems[int(i)] for i in z_sig],
            element_scales=w_sig, log_label="subset_sig")
        sub_sig.Assemble(prom.TPhi @ q0 + g)
        S_sub = rve.homogenized_stress(Etest, assembler=sub_sig)
        rel_s = np.linalg.norm(S_red - S_sub) / max(np.linalg.norm(S_sub), 1e-300)
        print(f"reduced vs subset homogenized stress:  {rel_s:.3e}  "
              f"{'OK' if rel_s < 1e-12 else 'FAIL'}", flush=True)
        if rel_s >= 1e-12:
            print("stress rule mapping is wrong; aborting")
            return 1

        # --- PROM reference ---
        Sp, tp = [], []
        for i in idx:
            t0 = time.perf_counter()
            S, _q = prom.solve(E_all[i])
            tp.append(time.perf_counter() - t0)
            Sp.append(S)
        Sp, tp = np.array(Sp), np.array(tp)

        # --- reduced HPROM ---
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
                    gk = rve._g(Et)
                    for _ in range(lp.NEWTON_MAX_IT):
                        K, R = red_res.assemble(q, gk)
                        dq = np.linalg.solve(red_res.TPhi.T @ (K @ red_res.TPhi),
                                             red_res.TPhi.T @ R)
                        q = q + dq
                        if (np.linalg.norm(dq)
                                / max(np.linalg.norm(q), 1e-30) < lp.NEWTON_TOL):
                            break
                    else:
                        raise RuntimeError("HPROM Newton failed")
                red_sig.assemble(q, rve._g(E))
                S = rve.homogenized_stress(E, assembler=red_sig.asm)
                th.append(time.perf_counter() - t0)
            except RuntimeError:
                nfail += 1
                continue
            Sh.append(S)
        Sh, th = np.array(Sh), np.array(th)
        n = len(Sh)
        es = np.linalg.norm(Sh - Sp[:n]) / np.linalg.norm(Sp[:n])

    # PER-STATE table, restored. It was dropped from this script and a global
    # Frobenius norm of 1.25e-01 then hid the fact that typical states were
    # fine at ~1e-04 while a few blew up -- the exact failure mode the
    # diagnostic table exists to expose, and which cost two wrong hypotheses
    # (weight misalignment, dof mapping) before a direct attribution settled it.
    es_i = np.linalg.norm(Sh - Sp[:n], axis=1) / np.linalg.norm(Sp[:n], axis=1)
    order = np.argsort(es_i)[::-1]
    print(f"\nworst 6 states by stress error:")
    for k in order[:6]:
        print(f"  E = [{E_all[idx[k]][0]:+.5f} {E_all[idx[k]][1]:+.5f} "
              f"{E_all[idx[k]][2]:+.5f}]   err {es_i[k]:.4e}")
    print(f"  median {np.median(es_i):.4e}   p90 {np.percentile(es_i,90):.4e}   "
          f"max {es_i.max():.4e}   ({int(np.sum(es_i > 1e-2))} states above 1e-2)")

    print(f"\n=== {n} states, {nfail} failures ===\n")
    print(f"{'quantity':<34} {'value':>14}")
    print("-" * 50)
    print(f"{'stress error vs PROM (rel. Frob.)':<34} {es:14.4e}")
    print(f"{'PROM wall clock (s)':<34} {tp.sum():14.2f}")
    print(f"{'HPROM wall clock (s)':<34} {th.sum():14.2f}")
    print(f"{'speedup vs PROM':<34} {tp.sum() / th.sum():13.2f}x")
    print(f"{'speedup vs FOM':<34} {4.99 * tp.sum() / th.sum():13.2f}x")
    np.savez_compressed(HERE / "hprom_reduced.npz", err_s=es,
                        t_prom=tp, t_hprom=th)
    print("HPROM_REDUCED_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
