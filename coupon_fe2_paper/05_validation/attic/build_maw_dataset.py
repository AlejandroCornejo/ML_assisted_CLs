#!/usr/bin/env python3
"""Build the per-state (A_j, b_j) blocks MAW-ECM prunes on, for both rules.

WHY MAW-ECM CAN GO SO MUCH LOWER than the fixed-weight rule, which is the
whole point and worth stating in numbers:

                      constraints              weight vectors   min points
    fixed-weight ECM  3 or 4 x 495 states      ONE, shared      40 - 135
    MAW-ECM           3 or 4, per state        one PER state    ~4 - 10

With adaptive weights each state satisfies only its own handful of constraints,
so a support of ~10 is reachable. Fixed weights needed 40-135 because a single
vector had to serve all 495 states at once.

WHERE THE RESIDUAL IS EVALUATED, and a correction worth recording. Evaluated
at the FOM solution the projected residual would vanish identically,

    (T Phi_D)^T f_int = Phi_D^T (T^T f_int) = 0

since the FOM satisfies T^T f_int = 0 by construction -- which would make the
target degenerate, admitting the trivial w = 0 under non-negativity. But that
is not what is computed here. The residual is evaluated at the DECODER's
reconstruction d(q), not at the FOM snapshot, because that is where the
HPROM-ANN actually evaluates it: on the manifold, off equilibrium. Measured
target magnitude is 4.3e+05, not zero.

The volume row is kept regardless. It is no longer needed to avoid degeneracy,
but it preserves volume exactly, which is one of the two structural guarantees
that made the classic rule generalize out of the envelope at 1.2x.

Candidates are the classic ECM supports already built, so MAW-ECM refines a
good starting point rather than searching from scratch -- matching the
previous project's own architecture, where classic ECM always runs first.
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
          str(PROJ / "fe2_extension"), str(PROJ / "core"), str(PROJ / "mawecm")):
    if p not in sys.path:
        sys.path.insert(0, p)
KRATOS_PATH = "/home/kratos/Kratos_Eigen_Check/bin/Release"
if KRATOS_PATH not in sys.path:
    sys.path.append(KRATOS_PATH)

N_STATES = 495
RES_SVD_TOL = 1.0e-3      # the manifold rule's 40-element support


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-states", type=int, default=N_STATES)
    a_ = ap.parse_args()

    from periodic_fom import PeriodicRVE
    from _material_law_guard_claude import true_neo_hookean_active
    from numpy_decoder import NumpyDecoder
    from hprom_full import rank_for, run_ecm
    from hprom_ann import build_manifold_integrand

    b = np.load(ROOT / "04_training" / "decoder_basis_B_r39.npz")
    d = np.load(ROOT / "03_data" / "data.npz")
    sup = np.load(HERE / "ecm_supports.npz")

    U_train, E_train = d["U_train"], d["E_train"]
    ok = np.isfinite(U_train).all(axis=1)
    U_train, E_train = U_train[ok], E_train[ok]
    q_all = np.ascontiguousarray(b["q_M"].T)[ok]

    with true_neo_hookean_active():
        rve = PeriodicRVE(str(ROOT / "03_data" / "rve_mesh"),
                          cell_area=float(d["cell_area"]))
        dec = NumpyDecoder(ROOT / "04_training" / "decoder_basis_B_r39.npz",
                           ROOT / "04_training" / "nslave.npz")
        a = rve.assembler
        ne, ndl = a.n_elems, a.n_local_dof
        dofs = np.asarray(a.rows_R, dtype=np.int64).reshape(ne, ndl)

        # residual candidates: the manifold ECM's own support
        print("rebuilding the manifold residual ECM support...", flush=True)
        Um, svm = build_manifold_integrand(rve, dec, q_all, E_train,
                                           a_.n_states, verbose=False)
        z_res, w_res = run_ecm(np.ascontiguousarray(
            Um[:, :rank_for(svm, RES_SVD_TOL)]))
        o = np.argsort(z_res)
        z_res, w_res = z_res[o], w_res[o]
        z_sig = np.asarray(sup["z_sig"], dtype=np.int64)
        w_sig = np.asarray(sup["w_sig"], dtype=float)
        print(f"  residual candidates {z_res.size}, stress candidates {z_sig.size}",
              flush=True)

        step = max(1, q_all.shape[0] // a_.n_states)
        sel = np.arange(0, q_all.shape[0], step)[:a_.n_states]

        A_res = np.zeros((sel.size, 4, z_res.size))   # 3 residual rows + volume
        b_res = np.zeros((sel.size, 4))
        # 4 PK1 components + a volume row. The volume row is NOT optional: the
        # classic ECM imposes sum(w) = n_elements on BOTH rules via
        # constrain_sum_of_weights, and omitting it here silently dropped that
        # guarantee from the MAW stress rule -- measured, its per-state weight
        # sums then drifted over 1082..1443, a 4.9% spread, where the residual
        # rule (which did carry the row) held 1546.0000 to 2.5e-16. Two things
        # break without it: the softmax weight regression can only represent a
        # FIXED sum, and volume conservation is one of the two structural
        # guarantees behind the classic rule degrading only 1.2x out of the
        # training envelope.
        A_sig = np.zeros((sel.size, 5, z_sig.size))
        b_sig = np.zeros((sel.size, 5))
        t0 = time.perf_counter()
        for k, j in enumerate(sel):
            q = q_all[j]
            dd, Phi_D = dec.value_and_jac(q)
            TPD = np.asarray(rve.T @ Phi_D)
            a.Assemble(rve.T @ dd + rve._g(E_train[j]))

            f_int = a._f_int.reshape(ne, ndl)
            c_res = np.einsum("eld,el->ed", TPD[dofs], f_int)      # (ne, 3)
            A_res[k, :3, :] = c_res[z_res].T
            b_res[k, :3] = c_res.sum(axis=0)
            A_res[k, 3, :] = 1.0                                    # volume row
            b_res[k, 3] = float(ne)

            Sv = a._S_voigt
            St = np.zeros(Sv.shape[:2] + (2, 2))
            St[..., 0, 0] = Sv[..., 0]
            St[..., 1, 1] = Sv[..., 1]
            St[..., 0, 1] = Sv[..., 2]
            St[..., 1, 0] = Sv[..., 2]
            P = np.matmul(a._F, St)
            c_sig = np.einsum("eg,egij->eij", a.w_detJ, P).reshape(ne, 4) / rve.denom
            A_sig[k, :4, :] = c_sig[z_sig].T
            b_sig[k, :4] = c_sig.sum(axis=0)
            A_sig[k, 4, :] = 1.0
            b_sig[k, 4] = float(ne)
            if (k + 1) % 100 == 0:
                print(f"  {k + 1}/{sel.size}", flush=True)

    print(f"built in {time.perf_counter() - t0:.1f}s")
    print(f"residual target |b[:3]| median {np.median(np.abs(b_res[:, :3])):.3e} "
          f"(zero by construction at the FOM solution)")
    print(f"stress   target |b|     median {np.median(np.abs(b_sig)):.3e}")

    # Does the all-ones weight vector reproduce the targets? It must, since b
    # is the full-mesh sum -- but only on the CANDIDATE subset for A, so this
    # measures how much the classic ECM support already loses.
    for nm, A, bb, w in (("residual", A_res, b_res, w_res),
                         ("stress", A_sig, b_sig, w_sig)):
        r = np.array([np.linalg.norm(A[k] @ w - bb[k])
                      / max(np.linalg.norm(bb[k]), 1e-300)
                      for k in range(A.shape[0])])
        print(f"{nm:9s}: classic ECM weights reproduce b to median {np.median(r):.3e}")

    np.savez_compressed(HERE / "maw_dataset.npz",
                        A_res=A_res, b_res=b_res, z_res=z_res, w_res=w_res,
                        A_sig=A_sig, b_sig=b_sig, z_sig=z_sig, w_sig=w_sig,
                        q_train=q_all[sel], E_train=E_train[sel], n_elements=ne)
    print("MAW_DATASET_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
