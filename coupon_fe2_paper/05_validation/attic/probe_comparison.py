#!/usr/bin/env python3
"""The comparison: PROM, HPROM and D-HPROM-ANN on the same states, all against
the FOM, in and out of the training envelope.

THE QUESTION. Earlier measurements showed the reduced BASIS generalizes out of
the envelope (POD projection degrading 5.7x) while the trained closure does
not (D-HPROM-ANN degrading 28x from a level three orders worse). The HPROM
sits between: it SOLVES the reduced equilibrium, like the PROM, but its
cubature weights are FITTED on in-envelope snapshots, like the network. So its
degradation says which of the two ingredients dominates -- the solve or the
data fitting.

EVERYTHING IS RECOMPUTED ON THE SAME STATES AGAINST THE SAME REFERENCE.
Earlier D-HPROM-ANN figures were per-state MEDIANS while PROM and HPROM
figures were relative FROBENIUS norms; comparing those directly would be
meaningless. Here all three are evaluated on one state list, against the
stored FOM stress, and reported both ways.

The 5 probe states where the FOM itself has no converged solution are excluded
by the isfinite filter, as they have no reference; they sit in the strongly
compressive corner declared out of scope from the start.
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


def frob(A, B):
    return float(np.linalg.norm(A - B) / np.linalg.norm(B))


def per_state(A, B):
    return np.linalg.norm(A - B, axis=1) / np.linalg.norm(B, axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-eval", type=int, default=N_EVAL)
    a_ = ap.parse_args()

    import torch
    b = np.load(ROOT / "04_training" / "decoder_basis_B_r39.npz")
    d = np.load(ROOT / "03_data" / "data.npz")
    sup = np.load(HERE / "ecm_supports.npz")
    m = np.load(ROOT / "04_training" / "nslave.npz")
    Phi, Phi_M, Phi_S, A_M = b["Phi_ROM"], b["Phi_M"], b["Phi_S"], b["A_M"]
    mu_m, mu_s = m["mu_mean"], m["mu_std"]

    from train_nslave import build_net
    net = build_net(3, Phi_S.shape[1], width=int(m["width"]), depth=int(m["depth"]))
    net.load_state_dict({k: torch.from_numpy(m[k]) for k in net.state_dict()})
    net.eval()

    from periodic_fom import PeriodicRVE
    from linear_prom import LinearPROM
    from _material_law_guard_claude import true_neo_hookean_active
    from reduced_mesh import ReducedAssembly

    with true_neo_hookean_active():
        rve = PeriodicRVE(str(ROOT / "03_data" / "rve_mesh"),
                          cell_area=float(d["cell_area"]))
        prom = LinearPROM(rve, Phi)
        full = str(ROOT / "03_data" / "rve_mesh") + ".mdpa"
        red_res = ReducedAssembly(full, HERE / "cmp_res", sup["z_res"],
                                  sup["w_res"], rve, prom.TPhi)
        red_sig = ReducedAssembly(full, HERE / "cmp_sig", sup["z_sig"],
                                  sup["w_sig"], rve, prom.TPhi)

        def hprom(E):
            n = max(1, int(np.ceil(lp.SUBSTEPS_PER_UNIT_STRAIN * np.linalg.norm(E))))
            q = np.zeros(Phi.shape[1])
            for k in range(1, n + 1):
                g = rve._g(E * (k / n))
                for _ in range(lp.NEWTON_MAX_IT):
                    K, R = red_res.assemble(q, g)
                    dq = np.linalg.solve(red_res.TPhi.T @ (K @ red_res.TPhi),
                                         red_res.TPhi.T @ R)
                    q = q + dq
                    if np.linalg.norm(dq) / max(np.linalg.norm(q), 1e-30) < TOL:
                        break
                else:
                    raise RuntimeError("HPROM Newton failed")
            red_sig.assemble(q, rve._g(E))
            return rve.homogenized_stress(E, assembler=red_sig.asm)

        def dhprom_ann(E):
            with torch.no_grad():
                qs = net(torch.from_numpy(((E - mu_m) / mu_s)[None, :])).numpy()[0]
            u_ind = Phi_M @ (A_M @ E) + Phi_S @ qs
            rve.assembler.Assemble(rve.T @ u_ind + rve._g(E))
            return rve.homogenized_stress(E)

        results = {}
        for nm in ("test", "probe"):
            E_s, S_s = d[f"E_{nm}"], d[f"S_{nm}"]
            ok = np.isfinite(S_s).all(axis=1)
            E_s, S_s = E_s[ok], S_s[ok]
            idx = np.random.default_rng(3).choice(
                E_s.shape[0], min(a_.n_eval, E_s.shape[0]), replace=False)
            E_s, S_s = E_s[idx], S_s[idx]
            print(f"\n{nm}: {len(idx)} states", flush=True)
            cols = {}
            for label, fn in (("PROM", lambda E: prom.solve(E)[0]),
                              ("HPROM", hprom),
                              ("D-HPROM-ANN", dhprom_ann)):
                out, t, nfail = [], 0.0, 0
                keep = []
                for j in range(E_s.shape[0]):
                    try:
                        t0 = time.perf_counter()
                        out.append(fn(E_s[j]))
                        t += time.perf_counter() - t0
                        keep.append(j)
                    except RuntimeError:
                        nfail += 1
                A = np.array(out)
                B = S_s[keep]
                cols[label] = dict(frob=frob(A, B), per=per_state(A, B),
                                   t=t, nfail=nfail, n=len(keep))
                print(f"  {label:13s} n={len(keep):3d} fail={nfail} "
                      f"frob {cols[label]['frob']:.4e}  "
                      f"median {np.median(cols[label]['per']):.4e}  "
                      f"max {cols[label]['per'].max():.4e}  {t:.1f}s", flush=True)
            results[nm] = cols

    print("\n=== TABLE 1 (clean): relative Frobenius stress error vs FOM ===\n")
    print(f"{'method':<14} {'in-envelope':>14} {'out-of-envelope':>16} "
          f"{'degradation':>12}")
    print("-" * 60)
    for label in ("PROM", "HPROM", "D-HPROM-ANN"):
        a = results["test"][label]["frob"]
        p = results["probe"][label]["frob"]
        print(f"{label:<14} {a:14.4e} {p:16.4e} {p / a:11.1f}x")

    print("\n=== TABLE 2 (diagnostic, per state) ===\n")
    print(f"{'method':<14} {'set':>6} {'median':>12} {'p90':>12} {'max':>12} "
          f"{'fail':>5}")
    print("-" * 68)
    for label in ("PROM", "HPROM", "D-HPROM-ANN"):
        for nm in ("test", "probe"):
            c = results[nm][label]
            print(f"{label:<14} {nm:>6} {np.median(c['per']):12.4e} "
                  f"{np.percentile(c['per'], 90):12.4e} {c['per'].max():12.4e} "
                  f"{c['nfail']:5d}")

    np.savez_compressed(HERE / "probe_comparison.npz",
                        **{f"{nm}_{lb}": results[nm][lb]["per"]
                           for nm in results for lb in results[nm]})
    print("\nPROBE_COMPARISON_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
