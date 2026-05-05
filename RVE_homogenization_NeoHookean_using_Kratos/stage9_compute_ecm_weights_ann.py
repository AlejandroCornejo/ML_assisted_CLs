#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage 9b: Compute ECM Weights for HPROM-ANN.
Computes the optimal element weights (RSVD + ECM) using the residual dataset 
generated in Stage 9a (using the ANN manifold tangent).
"""

import os
import numpy as np
from empirical_cubature_method import EmpiricalCubatureMethod
from randomized_singular_value_decomposition import RandomizedSingularValueDecomposition

# ============================================================
# CONFIGURATION
# ============================================================
DATA_DIR = "stage_9_ecm_dataset_ann"
OUT_DIR = "stage_9_hprom_ann_data"

Q_FILE = os.path.join(DATA_DIR, "Q_ecm.dat")
B_FILE = os.path.join(DATA_DIR, "b_full.dat")
C_FILE = os.path.join(DATA_DIR, "C_hom.dat")
BH_FILE = os.path.join(DATA_DIR, "b_hom.dat")
META_FILE = os.path.join(DATA_DIR, "meta.npz")

# RSVD tolerance controls the rank of the basis used for ECM
RSVD_TOL_RES = 1e-6
RSVD_TOL_EPS = 1e-6
RSVD_TOL_SIG = 1e-6

ECM_MAX_UNSUCCESSFUL_IT = 200

# ============================================================
# HELPERS
# ============================================================

def rel_error(a, b):
    num = np.linalg.norm(a - b)
    den = max(np.linalg.norm(b), 1e-30)
    return float(num / den)

def target_sum(C):
    return C @ np.ones(C.shape[1])

def print_snapshot_residual_stats(b_res, nq, Ns):
    if int(nq) <= 0 or int(Ns) <= 0: return
    b_blocks = np.asarray(b_res).reshape(int(Ns), int(nq))
    bn = np.linalg.norm(b_blocks, axis=1)
    q = np.quantile(bn, [0.0, 0.5, 0.9, 0.99, 1.0])
    med = float(q[1])
    hard_gate = max(1.0e-6, 1000.0 * med)
    n_out = int(np.count_nonzero(bn > hard_gate))

    print("\n[RES] Snapshot equilibrium indicator (||b_s|| over nq rows):")
    print("  min={:.3e}, median={:.3e}, q90={:.3e}, q99={:.3e}, max={:.3e}".format(
            float(q[0]), float(q[1]), float(q[2]), float(q[3]), float(q[4])))
    print(f"  outliers above {hard_gate:.3e}: {n_out}/{Ns}")

def run_rsvd_on_transpose(M_T, rsvd_tol, label=""):
    print(f"\n[{label}] RSVD on matrix^T  (shape {M_T.shape})")
    rsvd = RandomizedSingularValueDecomposition(COMPUTE_U=True, RELATIVE_SVD=True, USE_RANDOMIZATION=True)
    U, s, _, eSVD = rsvd.Calculate(np.ascontiguousarray(M_T), truncation_tolerance=float(rsvd_tol))
    if U.size == 0: raise RuntimeError(f"[{label}] RSVD returned empty basis")
    print(f"[{label}] RSVD done | kept={s.size} | eSVD={eSVD:.3e}")
    return U, s, float(eSVD)

def run_ecm(U_basis, n_elem, ecm_tol, init_candidates, label=""):
    print(f"[{label}] Running ECM...")
    ecm = EmpiricalCubatureMethod(ECM_tolerance=float(ecm_tol), Filter_tolerance=0.0, MaximumNumberUnsuccesfulIterations=int(ECM_MAX_UNSUCCESSFUL_IT))
    ecm.SetUp(ResidualsBasis=U_basis, InitialCandidatesSet=init_candidates, constrain_sum_of_weights=False)
    ecm.Run()
    Z = np.array(ecm.z, dtype=int).ravel()
    w_sel = np.array(ecm.w, dtype=float).ravel()
    w_full = np.zeros(int(n_elem), dtype=float)
    w_full[Z] = w_sel
    print(f"[{label}] Selected |Z| = {Z.size} ({100.*Z.size/n_elem:.1f}% of elements)")
    return Z, w_sel, w_full

# ============================================================
# MAIN
# ============================================================

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for f in [Q_FILE, B_FILE, C_FILE, BH_FILE, META_FILE]:
        if not os.path.isfile(f): raise FileNotFoundError(f)

    meta = np.load(META_FILE)
    nq, Ns, n_elem = int(meta["n_primary"]), int(meta["N_s"]), int(meta["n_elem"])
    
    print("=" * 60)
    print("  Stage 9b: ECM Weight Computation (HPROM-ANN)")
    print("=" * 60)
    print(f"  nq (reduced vars) = {nq}")
    print(f"  Ns (snapshots)    = {Ns}")
    print(f"  n_elem            = {n_elem}")

    Q_res = np.memmap(Q_FILE, dtype=np.float64, mode="r", shape=(nq * Ns, n_elem))
    b_res = np.memmap(B_FILE, dtype=np.float64, mode="r", shape=(nq * Ns,))
    C_hom = np.memmap(C_FILE, dtype=np.float64, mode="r", shape=(6 * Ns, n_elem))
    b_hom = np.memmap(BH_FILE, dtype=np.float64, mode="r", shape=(6 * Ns,))

    print("\n[Sanity] Checking consistency: Q·1 ≈ b")
    b_check_res = target_sum(Q_res)
    err_b_res = rel_error(b_check_res, b_res)
    print(f"  [RES] ||Q·1 − b||/||b|| = {err_b_res:.3e}")
    
    b_check_hom = target_sum(C_hom)
    err_b_hom = rel_error(b_check_hom, b_hom)
    print(f"  [HOM] ||C·1 − b||/||b|| = {err_b_hom:.3e}")
    
    print_snapshot_residual_stats(b_res, nq=nq, Ns=Ns)

    # 1. Residual ECM
    U_res, _, eS_res = run_rsvd_on_transpose(Q_res.T, RSVD_TOL_RES, label="RES")
    Z_res, w_res_sel, w_res_full = run_ecm(U_res, n_elem, 0.0, None, "RES")
    err_hp_res = rel_error(Q_res @ w_res_full, b_res)
    print(f"[RES] Quality check ||Q·w − b||/||b|| = {err_hp_res:.3e}")

    # 2. Strain ECM
    C_eps, b_eps = C_hom[0:3*Ns, :], b_hom[0:3*Ns]
    U_eps, _, eS_eps = run_rsvd_on_transpose(C_eps.T, RSVD_TOL_EPS, label="EPS")
    Z_eps, w_eps_sel, w_eps_full = run_ecm(U_eps, n_elem, 0.0, Z_res, "EPS")
    err_hp_eps = rel_error(C_eps @ w_eps_full, b_eps)
    print(f"[EPS] Quality check ||C·w − b||/||b|| = {err_hp_eps:.3e}")

    # 3. Stress ECM
    C_sig, b_sig = C_hom[3*Ns:6*Ns, :], b_hom[3*Ns:6*Ns]
    U_sig, _, eS_sig = run_rsvd_on_transpose(C_sig.T, RSVD_TOL_SIG, label="SIG")
    Z_sig, w_sig_sel, w_sig_full = run_ecm(U_sig, n_elem, 0.0, Z_eps, "SIG")
    err_hp_sig = rel_error(C_sig @ w_sig_full, b_sig)
    print(f"[SIG] Quality check ||C·w − b||/||b|| = {err_hp_sig:.3e}")

    Z_union = np.union1d(np.union1d(Z_res, Z_eps), Z_sig).astype(int)
    print("\n" + "=" * 60)
    print("  ELEMENT SELECTION SUMMARY (HPROM-ANN)")
    print("=" * 60)
    print(f"  |Z_res|   = {Z_res.size:5d} ({100.*Z_res.size/n_elem:.1f}%)")
    print(f"  |Z_union| = {Z_union.size:5d} ({100.*Z_union.size/n_elem:.1f}%)")

    out_file = os.path.join(OUT_DIR, "ecm_weights_all.npz")
    np.savez(out_file, Z_res=Z_res, Z_eps=Z_eps, Z_sig=Z_sig, Z_union=Z_union,
             w_res=w_res_sel, w_res_full=w_res_full,
             w_eps=w_eps_sel, w_eps_full=w_eps_full,
             w_sig=w_sig_sel, w_sig_full=w_sig_full,
             nq=nq, Ns=Ns, n_elem=n_elem,
             rel_error_res=err_hp_res, rel_error_eps=err_hp_eps, rel_error_sig=err_hp_sig)
    print(f"\n[DONE] Weights saved → {out_file}")

if __name__ == "__main__": main()
