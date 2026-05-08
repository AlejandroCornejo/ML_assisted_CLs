#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage 5b: Compute ECM Weights
Runs the Empirical Cubature Method on the residual dataset to select
a sparse set of elements and their integration weights for HPROM.
"""

import os
import argparse
import numpy as np

from empirical_cubature_method import EmpiricalCubatureMethod
from randomized_singular_value_decomposition import RandomizedSingularValueDecomposition

# ============================================================
# CONFIGURATION
# ============================================================
DATA_DIR = "stage_5_ecm_dataset"
OUT_DIR  = "stage_5_hprom_data"

Q_FILE   = os.path.join(DATA_DIR, "Q_ecm.dat")
B_FILE   = os.path.join(DATA_DIR, "b_full.dat")
C_FILE   = os.path.join(DATA_DIR, "C_hom.dat")
BH_FILE  = os.path.join(DATA_DIR, "b_hom.dat")
META_FILE = os.path.join(DATA_DIR, "meta.npz")

# RSVD tolerance controls the rank of the basis used for ECM
# Smaller = more modes kept = more accurate but slower selection
RSVD_TOL_RES = 1e-6
RSVD_TOL_EPS = 1e-6
RSVD_TOL_SIG = 1e-6

# ECM tolerance: 0.0 means "keep adding elements until no improvement"
ECM_TOL_RES = 0.0
ECM_TOL_EPS = 0.0
ECM_TOL_SIG = 0.0

ECM_MAX_UNSUCCESSFUL_IT = 200


# ============================================================
# HELPERS
# ============================================================

def rel_error(a, b):
    num = np.linalg.norm(a - b)
    den = max(np.linalg.norm(b), 1e-30)
    return num / den


def target_sum(C):
    """Sum across all elements (columns) = full-mesh result."""
    return C @ np.ones(C.shape[1])


def print_snapshot_residual_stats(b_res, nq, Ns):
    """
    Inspect reduced residual magnitudes per snapshot to detect potential
    non-converged training states in Stage 5a replay.
    """
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
    """
    Compute RSVD of M (given as M^T) to get the left singular vectors U.
    These form the basis for the ECM.
    """
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


def run_ecm(U_basis, n_elem, ecm_tol, init_candidates, label="", max_unsuccessful_it=ECM_MAX_UNSUCCESSFUL_IT):
    """Run the greedy ECM element selection."""
    print(f"\n[{label}] Running Empirical Cubature Method")
    ecm = EmpiricalCubatureMethod(
        ECM_tolerance=float(ecm_tol),
        Filter_tolerance=0.0,
        Plotting=False,
        MaximumNumberUnsuccesfulIterations=int(max_unsuccessful_it),
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

    print(f"[{label}] Selected |Z| = {Z.size}  ({100.*Z.size/n_elem:.1f}% of {n_elem} elements)")
    return Z, w_sel, w_full


def parse_args():
    p = argparse.ArgumentParser(description="Stage 5b: compute ECM weights from a Stage-5 dataset.")
    p.add_argument("--data-dir", type=str, default=DATA_DIR, help="Input dataset directory (contains Q_ecm.dat, C_hom.dat, meta.npz).")
    p.add_argument("--out-dir", type=str, default=OUT_DIR, help="Output directory for ECM weights npz.")
    p.add_argument("--rsvd-tol-res", type=float, default=RSVD_TOL_RES)
    p.add_argument("--rsvd-tol-eps", type=float, default=RSVD_TOL_EPS)
    p.add_argument("--rsvd-tol-sig", type=float, default=RSVD_TOL_SIG)
    p.add_argument("--ecm-tol-res", type=float, default=ECM_TOL_RES)
    p.add_argument("--ecm-tol-eps", type=float, default=ECM_TOL_EPS)
    p.add_argument("--ecm-tol-sig", type=float, default=ECM_TOL_SIG)
    p.add_argument("--max-unsuccessful-it", type=int, default=ECM_MAX_UNSUCCESSFUL_IT)
    p.add_argument(
        "--ecm-coupling-mode",
        type=str,
        default="coupled",
        choices=["coupled", "independent"],
        help=(
            "coupled: EPS starts from Z_res and SIG starts from Z_eps (legacy behavior). "
            "independent: RES/EPS/SIG are selected independently from full candidates."
        ),
    )
    return p.parse_args()


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args()
    data_dir = str(args.data_dir)
    out_dir = str(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    q_file = os.path.join(data_dir, "Q_ecm.dat")
    b_file = os.path.join(data_dir, "b_full.dat")
    c_file = os.path.join(data_dir, "C_hom.dat")
    bh_file = os.path.join(data_dir, "b_hom.dat")
    meta_file = os.path.join(data_dir, "meta.npz")

    # --- Check files ---
    for f in [q_file, b_file, c_file, bh_file, meta_file]:
        if not os.path.isfile(f):
            raise FileNotFoundError(f)

    # --- Load metadata ---
    meta = np.load(meta_file, allow_pickle=True)
    nq      = int(meta["nq"])
    Ns_res  = int(meta["N_s_res"])
    Ns_hom  = int(meta["N_s_hom"])
    n_elem  = int(meta["n_elem"])
    A0_ref = float(np.ravel(meta["A_total"])[0]) if "A_total" in meta else None

    n_rows_res = nq * Ns_res
    n_rows_hom = 6 * Ns_hom
    n_rows_eps = 3 * Ns_hom
    n_rows_sig = 3 * Ns_hom

    print("=" * 60)
    print("  Stage 5b: ECM Weight Computation")
    print("=" * 60)
    print(f"  nq (POD modes)  = {nq}")
    print(f"  Ns_res (snapshots for residual)      = {Ns_res}")
    print(f"  Ns_hom (snapshots for homogenization)= {Ns_hom}")
    print(f"  n_elem          = {n_elem}")
    print(f"  Q_ecm rows      = {n_rows_res}")
    print(f"  C_hom rows      = {n_rows_hom}  (eps={n_rows_eps}, sig={n_rows_sig})")
    print(f"  data_dir        = {data_dir}")
    print(f"  out_dir         = {out_dir}")
    print(f"  coupling mode   = {args.ecm_coupling_mode}")
    if A0_ref is not None:
        print(f"  reference area A0 = {A0_ref:.6e}")

    # --- Load memmaps ---
    Q_res = np.memmap(q_file, dtype=np.float64, mode="r", shape=(n_rows_res, n_elem))
    b_res = np.memmap(b_file, dtype=np.float64, mode="r", shape=(n_rows_res,))

    C_hom = np.memmap(c_file, dtype=np.float64, mode="r", shape=(n_rows_hom, n_elem))
    b_hom = np.memmap(bh_file, dtype=np.float64, mode="r", shape=(n_rows_hom,))

    # --- Sanity checks ---
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
    print_snapshot_residual_stats(b_res, nq=nq, Ns=Ns_res)

    # =========================================================
    # 1. ECM for the RESIDUAL (projected internal forces)
    # =========================================================
    U_res, s_res, eSVD_res = run_rsvd_on_transpose(Q_res.T, args.rsvd_tol_res, label="RES")

    Z_res, w_res_sel, w_res_full = run_ecm(
        U_basis=U_res,
        n_elem=n_elem,
        ecm_tol=args.ecm_tol_res,
        init_candidates=None,
        label="RES",
        max_unsuccessful_it=args.max_unsuccessful_it,
    )

    b_hp_res = Q_res @ w_res_full
    err_hp_res = rel_error(b_hp_res, b_res)
    abs_hp_res = np.linalg.norm(b_hp_res - b_res)
    max_hp_res = np.max(np.abs(b_hp_res - b_res))
    print(f"[RES] Final check ||Q·w − b|| / ||b|| = {err_hp_res:.3e}")
    print(f"[RES] Final check ||Q·w − b||         = {abs_hp_res:.3e}  (max |.| = {max_hp_res:.3e})")

    # =========================================================
    # 2. ECM for STRAIN homogenization
    # =========================================================
    # IMPORTANT:
    # Stage 5a stores homogenization rows per snapshot as:
    #   [eps_xx, eps_yy, eps_xy, sig_xx, sig_yy, sig_xy]
    # Therefore EPS/SIG rows are interleaved in 6-row blocks and must be split
    # by block, not by a single contiguous top/bottom cut.
    C_blk = np.asarray(C_hom, dtype=float).reshape(Ns_hom, 6, n_elem)
    b_blk = np.asarray(b_hom, dtype=float).reshape(Ns_hom, 6)

    C_eps = C_blk[:, 0:3, :].reshape(n_rows_eps, n_elem)
    b_eps = b_blk[:, 0:3].reshape(n_rows_eps)

    U_eps, s_eps, eSVD_eps = run_rsvd_on_transpose(C_eps.T, args.rsvd_tol_eps, label="EPS")

    eps_init = np.array(Z_res, dtype=int) if str(args.ecm_coupling_mode).lower() == "coupled" else None
    Z_eps, w_eps_sel, w_eps_full = run_ecm(
        U_basis=U_eps,
        n_elem=n_elem,
        ecm_tol=args.ecm_tol_eps,
        init_candidates=eps_init,
        label="EPS",
        max_unsuccessful_it=args.max_unsuccessful_it,
    )

    b_hp_eps = C_eps @ w_eps_full
    err_hp_eps = rel_error(b_hp_eps, b_eps)
    abs_hp_eps = np.linalg.norm(b_hp_eps - b_eps)
    max_hp_eps = np.max(np.abs(b_hp_eps - b_eps))
    print(f"[EPS] Final check ||C_eps·w − b_eps|| / ||b_eps|| = {err_hp_eps:.3e}")
    print(f"[EPS] Final check ||C_eps·w − b_eps||             = {abs_hp_eps:.3e}  (max |.| = {max_hp_eps:.3e})")

    # =========================================================
    # 3. ECM for STRESS homogenization
    # =========================================================
    C_sig = C_blk[:, 3:6, :].reshape(n_rows_sig, n_elem)
    b_sig = b_blk[:, 3:6].reshape(n_rows_sig)

    U_sig, s_sig, eSVD_sig = run_rsvd_on_transpose(C_sig.T, args.rsvd_tol_sig, label="SIG")

    sig_init = np.array(Z_eps, dtype=int) if str(args.ecm_coupling_mode).lower() == "coupled" else None
    Z_sig, w_sig_sel, w_sig_full = run_ecm(
        U_basis=U_sig,
        n_elem=n_elem,
        ecm_tol=args.ecm_tol_sig,
        init_candidates=sig_init,
        label="SIG",
        max_unsuccessful_it=args.max_unsuccessful_it,
    )

    b_hp_sig = C_sig @ w_sig_full
    err_hp_sig = rel_error(b_hp_sig, b_sig)
    abs_hp_sig = np.linalg.norm(b_hp_sig - b_sig)
    max_hp_sig = np.max(np.abs(b_hp_sig - b_sig))
    print(f"[SIG] Final check ||C_sig·w − b_sig|| / ||b_sig|| = {err_hp_sig:.3e}")
    print(f"[SIG] Final check ||C_sig·w − b_sig||             = {abs_hp_sig:.3e}  (max |.| = {max_hp_sig:.3e})")

    # =========================================================
    # 4. Union of selected elements
    # =========================================================
    Z_union = np.union1d(np.union1d(Z_res, Z_eps), Z_sig).astype(int)

    print("\n" + "=" * 60)
    print("  ELEMENT SELECTION SUMMARY")
    print("=" * 60)
    print(f"  |Z_res|   = {Z_res.size:5d}  ({100.*Z_res.size/n_elem:.1f}%)")
    print(f"  |Z_eps|   = {Z_eps.size:5d}  ({100.*Z_eps.size/n_elem:.1f}%)")
    print(f"  |Z_sig|   = {Z_sig.size:5d}  ({100.*Z_sig.size/n_elem:.1f}%)")
    print(f"  |Z_union| = {Z_union.size:5d}  ({100.*Z_union.size/n_elem:.1f}%)")
    if str(args.ecm_coupling_mode).lower() == "coupled":
        print("  coupling detail: EPS initialized with Z_res, SIG initialized with Z_eps")
    else:
        print("  coupling detail: EPS and SIG initialized independently (full candidate sets)")

    # =========================================================
    # 5. Save
    # =========================================================
    out_file = os.path.join(out_dir, "ecm_weights_all.npz")

    np.savez(
        out_file,
        # Element sets
        Z_res=Z_res, Z_eps=Z_eps, Z_sig=Z_sig, Z_union=Z_union,
        # Weights (sparse and full)
        w_res=w_res_sel, w_res_full=w_res_full,
        w_eps=w_eps_sel, w_eps_full=w_eps_full,
        w_sig=w_sig_sel, w_sig_full=w_sig_full,
        # Metadata
        nq=nq, Ns_res=Ns_res, Ns_hom=Ns_hom, n_elem=n_elem,
        # Tolerances and errors
        RSVD_TOL_RES=float(args.rsvd_tol_res), eSVD_res=eSVD_res, rel_error_res=err_hp_res,
        RSVD_TOL_EPS=float(args.rsvd_tol_eps), eSVD_eps=eSVD_eps, rel_error_eps=err_hp_eps,
        RSVD_TOL_SIG=float(args.rsvd_tol_sig), eSVD_sig=eSVD_sig, rel_error_sig=err_hp_sig,
        ECM_TOL_RES=float(args.ecm_tol_res),
        ECM_TOL_EPS=float(args.ecm_tol_eps),
        ECM_TOL_SIG=float(args.ecm_tol_sig),
        ECM_COUPLING_MODE=np.array([str(args.ecm_coupling_mode)]),
        data_dir=np.array([data_dir]),
        A0_ref=np.array([float(A0_ref if A0_ref is not None else np.nan)], dtype=float),
        hom_reference_measure=np.array([float(A0_ref if A0_ref is not None else np.nan)], dtype=float),
    )

    if A0_ref is not None:
        with open(os.path.join(out_dir, "reference_measure_A0.txt"), "w", encoding="utf-8") as f:
            f.write(f"{A0_ref:.16e}\n")

    print(f"\n[DONE] Saved → {out_file}")


if __name__ == "__main__":
    main()
