#!/usr/bin/env python3
"""Save the per-element integrand contributions on the FULL mesh, once.

WHY THIS EXISTS. Every question left open about MAW-ECM is a question about
which SUPPORT and which WEIGHTS to use, and each of those was costing a full
495-state reassembly of the RVE to evaluate. Worse, `build_maw_dataset.py`
stores A only on the classic ECM's own candidate columns, so it cannot even
express the one comparison that matters most -- the classic rule at the SAME
point count as MAW -- and `build_manifold_integrand` computes exactly the
matrix needed and then throws it away, keeping only its POD basis.

So store the raw contributions:

    C_res  (ne, n_states, 3)   element-wise Phi_D(q)^T f_int
    C_sig  (ne, n_states, 4)   element-wise PK1, already divided by rve.denom

From these, the constraint blocks for ANY support z follow in pure numpy,

    A[k] = C[z, k, :].T   stacked with a row of ones      (volume row)
    b[k] = C[:, k, :].sum(axis=0)  with n_elements        (full-mesh target)

which makes every subsequent sweep -- classic ECM at any rank, MAW at any
n_stop, any weight field evaluated on any support -- free. b is computed from
the full-mesh sum, so it is the true target rather than a candidate-subset
approximation.

THE COMPARISON THIS UNLOCKS, and the reason it is the point of the script:
MAW-ECM at 10 points was being compared against the classic rule at 40 (residual)
and 73 (stress) points. That flatters the classic rule's accuracy by giving it
4x and 7x the cost. The honest question is which rule is more accurate AT EQUAL
POINT COUNT, and, past that, what each rule's accuracy-vs-cost curve looks like.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-states", type=int, default=N_STATES)
    a_ = ap.parse_args()

    from periodic_fom import PeriodicRVE
    from _material_law_guard_claude import true_neo_hookean_active
    from numpy_decoder import NumpyDecoder

    b = np.load(ROOT / "04_training" / "decoder_basis_B_r39.npz")
    d = np.load(ROOT / "03_data" / "data.npz")

    U_train, E_train = d["U_train"], d["E_train"]
    ok = np.isfinite(U_train).all(axis=1)
    E_train = E_train[ok]
    q_all = np.ascontiguousarray(b["q_M"].T)[ok]

    # Same state selection as build_maw_dataset.py, so the two datasets index
    # the same physical states and results stay comparable across scripts.
    step = max(1, q_all.shape[0] // a_.n_states)
    sel = np.arange(0, q_all.shape[0], step)[:a_.n_states]

    with true_neo_hookean_active():
        rve = PeriodicRVE(str(ROOT / "03_data" / "rve_mesh"),
                          cell_area=float(d["cell_area"]))
        dec = NumpyDecoder(ROOT / "04_training" / "decoder_basis_B_r39.npz",
                           ROOT / "04_training" / "nslave.npz")
        a = rve.assembler
        ne, ndl = a.n_elems, a.n_local_dof
        dofs = np.asarray(a.rows_R, dtype=np.int64).reshape(ne, ndl)

        C_res = np.empty((ne, sel.size, 3))
        C_sig = np.empty((ne, sel.size, 4))
        t0 = time.perf_counter()
        for k, j in enumerate(sel):
            dd, Phi_D = dec.value_and_jac(q_all[j])
            TPD = np.asarray(rve.T @ Phi_D)
            a.Assemble(rve.T @ dd + rve._g(E_train[j]))

            f_int = a._f_int.reshape(ne, ndl)
            C_res[:, k, :] = np.einsum("eld,el->ed", TPD[dofs], f_int)

            Sv = a._S_voigt
            St = np.zeros(Sv.shape[:2] + (2, 2))
            St[..., 0, 0] = Sv[..., 0]
            St[..., 1, 1] = Sv[..., 1]
            St[..., 0, 1] = Sv[..., 2]
            St[..., 1, 0] = Sv[..., 2]
            P = np.matmul(a._F, St)
            C_sig[:, k, :] = np.einsum(
                "eg,egij->eij", a.w_detJ, P).reshape(ne, 4) / rve.denom
            if (k + 1) % 100 == 0:
                print(f"  {k + 1}/{sel.size}  "
                      f"{time.perf_counter() - t0:.0f}s", flush=True)

    print(f"assembled {sel.size} states in {time.perf_counter() - t0:.1f}s")

    # The all-ones weight vector on the FULL mesh must reproduce b exactly,
    # since b is defined as that same sum. This is a tautology check on the
    # bookkeeping, not on physics -- but it is the check that would have caught
    # the support-mapping bugs that cost real time earlier in this project.
    for nm, C in (("res", C_res), ("sig", C_sig)):
        bb = C.sum(axis=0)
        r = np.abs(C.sum(axis=0) - bb).max()
        print(f"{nm}: full-mesh identity {r:.3e} (must be 0)")
        print(f"{nm}: |b| median {np.median(np.linalg.norm(bb, axis=1)):.4e}")

    np.savez(HERE / "full_integrand.npz", C_res=C_res, C_sig=C_sig,
             q_train=q_all[sel], E_train=E_train[sel], n_elements=ne,
             state_index=sel)
    sz = (HERE / "full_integrand.npz").stat().st_size / 1e6
    print(f"wrote full_integrand.npz  ({sz:.0f} MB)")
    print("FULL_INTEGRAND_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
