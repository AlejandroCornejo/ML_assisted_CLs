#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage 9b: Compute ECM Weights for HPROM-RBF.

Same ECM/RSVD logic as Stage 5b, but reading Stage 9a dataset outputs.
"""

import os
import numpy as np

from empirical_cubature_method import EmpiricalCubatureMethod
from randomized_singular_value_decomposition import RandomizedSingularValueDecomposition


DATA_DIR = "stage_9_ecm_dataset_rbf"
OUT_DIR = "stage_9_hprom_rbf_data"

Q_FILE = os.path.join(DATA_DIR, "Q_ecm.dat")
B_FILE = os.path.join(DATA_DIR, "b_full.dat")
C_FILE = os.path.join(DATA_DIR, "C_hom.dat")
BH_FILE = os.path.join(DATA_DIR, "b_hom.dat")
META_FILE = os.path.join(DATA_DIR, "meta.npz")

RSVD_TOL_RES = 1e-5
RSVD_TOL_EPS = 1e-6
RSVD_TOL_SIG = 1e-6

ECM_TOL_RES = 0.0
ECM_TOL_EPS = 0.0
ECM_TOL_SIG = 0.0

ECM_MAX_UNSUCCESSFUL_IT = 200


def rel_error(a, b):
    num = np.linalg.norm(a - b)
    den = max(np.linalg.norm(b), 1e-30)
    return float(num / den)


def target_sum(C):
    return C @ np.ones(C.shape[1])


def print_snapshot_residual_stats(b_res, nq, Ns):
    if int(nq) <= 0 or int(Ns) <= 0:
        return
    b_blocks = np.asarray(b_res).reshape(int(Ns), int(nq))
    bn = np.linalg.norm(b_blocks, axis=1)
    q = np.quantile(bn, [0.0, 0.5, 0.9, 0.99, 1.0])
    med = float(q[1])
    hard_gate = max(1.0e-6, 1000.0 * med)
    n_out = int(np.count_nonzero(bn > hard_gate))

    print("\n[RES] Snapshot equilibrium indicator (||b_s|| over nq rows):")
    print(
        "  min={:.3e}, median={:.3e}, q90={:.3e}, q99={:.3e}, max={:.3e}".format(
            float(q[0]), float(q[1]), float(q[2]), float(q[3]), float(q[4])
        )
    )
    print(f"  outliers above {hard_gate:.3e}: {n_out}/{Ns}")
    if n_out > 0:
        idx = np.argsort(bn)[-min(10, n_out):][::-1]
        print("  top outlier snapshot ids (0-based):")
        for i in idx:
            print(f"    s={int(i):5d}, ||b_s||={float(bn[i]):.3e}")


def run_rsvd_on_transpose(M_T, rsvd_tol, label=""):
    print(f"\n[{label}] RSVD on matrix^T  (shape {M_T.shape})")
    A = np.ascontiguousarray(M_T)

    rsvd = RandomizedSingularValueDecomposition(
        COMPUTE_U=True,
        COMPUTE_V=False,
        RELATIVE_SVD=True,
        USE_RANDOMIZATION=True,
    )

    U, s, _, eSVD = rsvd.Calculate(A, truncation_tolerance=float(rsvd_tol))
    if U.size == 0:
        raise RuntimeError(f"[{label}] RSVD returned empty basis")

    print(f"[{label}] RSVD done")
    print(f"[{label}]   U.shape = {U.shape}")
    print(f"[{label}]   kept    = {s.size}")
    print(f"[{label}]   eSVD    = {eSVD:.3e}")
    return U, s, float(eSVD)


def run_ecm(U_basis, n_elem, ecm_tol, init_candidates, label=""):
    print(f"\n[{label}] Running Empirical Cubature Method")
    ecm = EmpiricalCubatureMethod(
        ECM_tolerance=float(ecm_tol),
        Filter_tolerance=0.0,
        Plotting=False,
        MaximumNumberUnsuccesfulIterations=int(ECM_MAX_UNSUCCESSFUL_IT),
    )

    ecm.SetUp(
        ResidualsBasis=U_basis,
        InitialCandidatesSet=init_candidates,
        constrain_sum_of_weights=False,
        constrain_conditions=False,
        number_of_conditions=0,
    )
    ecm.Run()

    Z = np.array(ecm.z, dtype=int).ravel()
    w_sel = np.array(ecm.w, dtype=float).ravel()

    w_full = np.zeros(int(n_elem), dtype=float)
    w_full[Z] = w_sel
    print(f"[{label}] Selected |Z| = {Z.size}  ({100.0 * Z.size / n_elem:.1f}% of {n_elem} elements)")
    return Z, w_sel, w_full


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    for f in [Q_FILE, B_FILE, C_FILE, BH_FILE, META_FILE]:
        if not os.path.isfile(f):
            raise FileNotFoundError(f)

    meta = np.load(META_FILE, allow_pickle=True)
    nq = int(meta["nq"])
    Ns = int(meta["N_s"])
    n_elem = int(meta["n_elem"])

    n_rows_res = nq * Ns
    n_rows_hom = 6 * Ns
    n_rows_eps = 3 * Ns
    n_rows_sig = 3 * Ns

    print("=" * 60)
    print("  Stage 9b: ECM Weight Computation (HPROM-RBF)")
    print("=" * 60)
    print(f"  nq (RBF reduced vars) = {nq}")
    print(f"  Ns (snapshots)        = {Ns}")
    print(f"  n_elem                = {n_elem}")
    print(f"  Q_ecm rows            = {n_rows_res}")
    print(f"  C_hom rows            = {n_rows_hom}  (eps={n_rows_eps}, sig={n_rows_sig})")

    Q_res = np.memmap(Q_FILE, dtype=np.float64, mode="r", shape=(n_rows_res, n_elem))
    b_res = np.memmap(B_FILE, dtype=np.float64, mode="r", shape=(n_rows_res,))
    C_hom = np.memmap(C_FILE, dtype=np.float64, mode="r", shape=(n_rows_hom, n_elem))
    b_hom = np.memmap(BH_FILE, dtype=np.float64, mode="r", shape=(n_rows_hom,))

    print("\n[Sanity] Checking consistency: Q·1 ≈ b")
    b_check_res = target_sum(Q_res)
    err_b_res = rel_error(b_check_res, b_res)
    abs_b_res = np.linalg.norm(b_check_res - b_res)
    max_b_res = np.max(np.abs(b_check_res - b_res))
    print(f"  [RES]  ||Q·1 − b|| / ||b|| = {err_b_res:.3e}")
    print(f"  [RES]  ||Q·1 − b||         = {abs_b_res:.3e}  (max |.| = {max_b_res:.3e})")
    if err_b_res > 1e-10:
        print("  [RES][WARN] b_full is NOT consistent with Q_res!")

    b_check_hom = target_sum(C_hom)
    err_b_hom = rel_error(b_check_hom, b_hom)
    abs_b_hom = np.linalg.norm(b_check_hom - b_hom)
    max_b_hom = np.max(np.abs(b_check_hom - b_hom))
    print(f"  [HOM]  ||C·1 − b|| / ||b|| = {err_b_hom:.3e}")
    print(f"  [HOM]  ||C·1 − b||         = {abs_b_hom:.3e}  (max |.| = {max_b_hom:.3e})")
    if err_b_hom > 1e-10:
        print("  [HOM][WARN] b_hom is NOT consistent with C_hom!")
    print_snapshot_residual_stats(b_res, nq=nq, Ns=Ns)

    U_res, s_res, eSVD_res = run_rsvd_on_transpose(Q_res.T, RSVD_TOL_RES, label="RES")
    Z_res, w_res_sel, w_res_full = run_ecm(
        U_basis=U_res,
        n_elem=n_elem,
        ecm_tol=ECM_TOL_RES,
        init_candidates=None,
        label="RES",
    )
    b_hp_res = Q_res @ w_res_full
    err_hp_res = rel_error(b_hp_res, b_res)
    abs_hp_res = np.linalg.norm(b_hp_res - b_res)
    max_hp_res = np.max(np.abs(b_hp_res - b_res))
    print(f"[RES] Final check ||Q·w − b|| / ||b|| = {err_hp_res:.3e}")
    print(f"[RES] Final check ||Q·w − b||         = {abs_hp_res:.3e}  (max |.| = {max_hp_res:.3e})")

    C_eps = C_hom[0:n_rows_eps, :]
    b_eps = b_hom[0:n_rows_eps]
    U_eps, s_eps, eSVD_eps = run_rsvd_on_transpose(C_eps.T, RSVD_TOL_EPS, label="EPS")
    Z_eps, w_eps_sel, w_eps_full = run_ecm(
        U_basis=U_eps,
        n_elem=n_elem,
        ecm_tol=ECM_TOL_EPS,
        init_candidates=np.array(Z_res, dtype=int),
        label="EPS",
    )
    b_hp_eps = C_eps @ w_eps_full
    err_hp_eps = rel_error(b_hp_eps, b_eps)
    abs_hp_eps = np.linalg.norm(b_hp_eps - b_eps)
    max_hp_eps = np.max(np.abs(b_hp_eps - b_eps))
    print(f"[EPS] Final check ||C_eps·w − b_eps|| / ||b_eps|| = {err_hp_eps:.3e}")
    print(f"[EPS] Final check ||C_eps·w − b_eps||             = {abs_hp_eps:.3e}  (max |.| = {max_hp_eps:.3e})")

    C_sig = C_hom[n_rows_eps:n_rows_eps + n_rows_sig, :]
    b_sig = b_hom[n_rows_eps:n_rows_eps + n_rows_sig]
    U_sig, s_sig, eSVD_sig = run_rsvd_on_transpose(C_sig.T, RSVD_TOL_SIG, label="SIG")
    Z_sig, w_sig_sel, w_sig_full = run_ecm(
        U_basis=U_sig,
        n_elem=n_elem,
        ecm_tol=ECM_TOL_SIG,
        init_candidates=np.array(Z_eps, dtype=int),
        label="SIG",
    )
    b_hp_sig = C_sig @ w_sig_full
    err_hp_sig = rel_error(b_hp_sig, b_sig)
    abs_hp_sig = np.linalg.norm(b_hp_sig - b_sig)
    max_hp_sig = np.max(np.abs(b_hp_sig - b_sig))
    print(f"[SIG] Final check ||C_sig·w − b_sig|| / ||b_sig|| = {err_hp_sig:.3e}")
    print(f"[SIG] Final check ||C_sig·w − b_sig||             = {abs_hp_sig:.3e}  (max |.| = {max_hp_sig:.3e})")

    Z_union = np.union1d(np.union1d(Z_res, Z_eps), Z_sig).astype(int)

    print("\n" + "=" * 60)
    print("  ELEMENT SELECTION SUMMARY (HPROM-RBF)")
    print("=" * 60)
    print(f"  |Z_res|   = {Z_res.size:5d}  ({100.0 * Z_res.size / n_elem:.1f}%)")
    print(f"  |Z_eps|   = {Z_eps.size:5d}  ({100.0 * Z_eps.size / n_elem:.1f}%)")
    print(f"  |Z_sig|   = {Z_sig.size:5d}  ({100.0 * Z_sig.size / n_elem:.1f}%)")
    print(f"  |Z_union| = {Z_union.size:5d}  ({100.0 * Z_union.size / n_elem:.1f}%)")

    out_file = os.path.join(OUT_DIR, "ecm_weights_all.npz")
    np.savez(
        out_file,
        Z_res=Z_res,
        Z_eps=Z_eps,
        Z_sig=Z_sig,
        Z_union=Z_union,
        w_res=w_res_sel,
        w_res_full=w_res_full,
        w_eps=w_eps_sel,
        w_eps_full=w_eps_full,
        w_sig=w_sig_sel,
        w_sig_full=w_sig_full,
        nq=nq,
        Ns=Ns,
        n_elem=n_elem,
        RSVD_TOL_RES=RSVD_TOL_RES,
        eSVD_res=eSVD_res,
        rel_error_res=err_hp_res,
        RSVD_TOL_EPS=RSVD_TOL_EPS,
        eSVD_eps=eSVD_eps,
        rel_error_eps=err_hp_eps,
        RSVD_TOL_SIG=RSVD_TOL_SIG,
        eSVD_sig=eSVD_sig,
        rel_error_sig=err_hp_sig,
    )
    print(f"\n[DONE] Saved → {out_file}")


if __name__ == "__main__":
    main()
