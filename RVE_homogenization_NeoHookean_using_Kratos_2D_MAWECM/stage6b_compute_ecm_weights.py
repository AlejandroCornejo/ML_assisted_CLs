#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Stage 6b (2D-MAWECM): compute classical ECM weights from Stage6a dataset."""

import argparse
import os
import sys

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_MAIN_CLASSIC_DIR = os.path.abspath(os.path.join(_THIS_DIR, "..", "RVE_homogenization_NeoHookean_using_Kratos"))
if _MAIN_CLASSIC_DIR not in sys.path:
    sys.path.append(_MAIN_CLASSIC_DIR)

from empirical_cubature_method import EmpiricalCubatureMethod
from randomized_singular_value_decomposition import RandomizedSingularValueDecomposition


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage6b: classical ECM weights (2D-MAWECM)")
    p.add_argument("--data-dir", type=str, default="stage_6a_ecm_dataset")
    p.add_argument("--out-dir", type=str, default="stage_6b_hprom_ecm")
    p.add_argument("--rsvd-tol-res", type=float, default=1e-6)
    p.add_argument("--rsvd-tol-eps", type=float, default=1e-6)
    p.add_argument("--rsvd-tol-sig", type=float, default=1e-6)
    p.add_argument("--ecm-tol-res", type=float, default=0.0)
    p.add_argument("--ecm-tol-eps", type=float, default=0.0)
    p.add_argument("--ecm-tol-sig", type=float, default=0.0)
    p.add_argument("--max-unsuccessful-it", type=int, default=200)
    p.add_argument(
        "--ecm-coupling-mode",
        type=str,
        default="cascade",
        choices=["independent", "cascade", "single"],
    )
    p.add_argument(
        "--single-block-normalization",
        type=str,
        default="fro",
        choices=["fro", "row", "none"],
    )
    p.add_argument(
        "--rsvd-tol-single",
        type=float,
        default=-1.0,
        help="If <=0, uses min(rsvd-tol-res,eps,sig).",
    )
    return p.parse_args()


def _rel_error(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def _run_rsvd_on_transpose(m_t: np.ndarray, tol: float, label: str):
    print(f"\n[{label}] RSVD on matrix^T shape={m_t.shape}, tol={tol:.3e}")
    rsvd = RandomizedSingularValueDecomposition(
        COMPUTE_U=True,
        COMPUTE_V=False,
        RELATIVE_SVD=True,
        USE_RANDOMIZATION=True,
    )
    u, s, _, esvd = rsvd.Calculate(np.ascontiguousarray(m_t), truncation_tolerance=float(tol))
    if u.size == 0:
        raise RuntimeError(f"[{label}] RSVD returned empty basis")
    print(f"[{label}] kept={s.size}, eSVD={float(esvd):.3e}, U.shape={u.shape}")
    return np.asarray(u, dtype=float), np.asarray(s, dtype=float), float(esvd)


def _run_ecm(u_basis: np.ndarray, n_elem: int, ecm_tol: float, init_candidates, label: str, max_unsuccessful_it: int):
    print(f"\n[{label}] ECM run: basis_cols={u_basis.shape[1]}, ecm_tol={ecm_tol:.3e}")
    ecm = EmpiricalCubatureMethod(
        ECM_tolerance=float(ecm_tol),
        Filter_tolerance=0.0,
        Plotting=False,
        MaximumNumberUnsuccesfulIterations=int(max_unsuccessful_it),
    )
    ecm.SetUp(
        ResidualsBasis=u_basis,
        InitialCandidatesSet=init_candidates,
        constrain_sum_of_weights=True,
        constrain_conditions=False,
        number_of_conditions=0,
    )
    ecm.Run()

    z = np.asarray(ecm.z, dtype=int).reshape(-1)
    w_sel = np.asarray(ecm.w, dtype=float).reshape(-1)
    w_full = np.zeros(int(n_elem), dtype=float)
    w_full[z] = w_sel
    print(f"[{label}] |Z|={z.size} ({100.0 * z.size / max(n_elem,1):.1f}% of {n_elem})")
    return z, w_sel, w_full


def _single_block_scales(q_res: np.ndarray, c_eps: np.ndarray, c_sig: np.ndarray, mode: str):
    if mode == "none":
        return 1.0, 1.0, 1.0
    if mode == "row":
        return (
            1.0 / max(np.sqrt(q_res.shape[0]), 1.0),
            1.0 / max(np.sqrt(c_eps.shape[0]), 1.0),
            1.0 / max(np.sqrt(c_sig.shape[0]), 1.0),
        )
    # fro
    return (
        1.0 / max(np.linalg.norm(q_res), 1e-30),
        1.0 / max(np.linalg.norm(c_eps), 1e-30),
        1.0 / max(np.linalg.norm(c_sig), 1e-30),
    )


def main():
    args = _parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    meta_file = os.path.join(args.data_dir, "meta.npz")
    q_file = os.path.join(args.data_dir, "Q_ecm.dat")
    b_file = os.path.join(args.data_dir, "b_full.dat")
    c_file = os.path.join(args.data_dir, "C_hom.dat")
    bh_file = os.path.join(args.data_dir, "b_hom.dat")
    for f in (meta_file, q_file, b_file, c_file, bh_file):
        if not os.path.exists(f):
            raise FileNotFoundError(f)

    meta = np.load(meta_file, allow_pickle=True)
    nq = int(np.ravel(meta["nq"])[0])
    n_elem = int(np.ravel(meta["n_elem"])[0])
    ns_res = int(np.ravel(meta["N_s_res"])[0])
    ns_hom = int(np.ravel(meta["N_s_hom"])[0])
    a0_ref = float(np.ravel(meta["A_total"])[0]) if "A_total" in meta else np.nan

    nr_res = nq * ns_res
    nr_hom = 6 * ns_hom
    nr_eps = 3 * ns_hom
    nr_sig = 3 * ns_hom

    print("=" * 72)
    print("Stage 6b: Classical ECM weights (2D-MAWECM)")
    print("=" * 72)
    print(f"data_dir            : {args.data_dir}")
    print(f"out_dir             : {args.out_dir}")
    print(f"nq / n_elem         : {nq} / {n_elem}")
    print(f"Ns_res / Ns_hom     : {ns_res} / {ns_hom}")
    print(f"coupling_mode       : {args.ecm_coupling_mode}")

    q_res = np.memmap(q_file, dtype=np.float64, mode="r", shape=(nr_res, n_elem))
    b_res = np.memmap(b_file, dtype=np.float64, mode="r", shape=(nr_res,))
    c_hom = np.memmap(c_file, dtype=np.float64, mode="r", shape=(nr_hom, n_elem))
    b_hom = np.memmap(bh_file, dtype=np.float64, mode="r", shape=(nr_hom,))

    # Sanity
    e_res = _rel_error(q_res @ np.ones(n_elem), b_res)
    e_hom = _rel_error(c_hom @ np.ones(n_elem), b_hom)
    print(f"[Sanity] rel ||Q*1-b||={e_res:.3e}, ||C*1-b||={e_hom:.3e}")

    c_blk = np.asarray(c_hom, dtype=float).reshape(ns_hom, 6, n_elem)
    b_blk = np.asarray(b_hom, dtype=float).reshape(ns_hom, 6)
    c_eps = c_blk[:, 0:3, :].reshape(nr_eps, n_elem)
    b_eps = b_blk[:, 0:3].reshape(nr_eps)
    c_sig = c_blk[:, 3:6, :].reshape(nr_sig, n_elem)
    b_sig = b_blk[:, 3:6].reshape(nr_sig)

    mode = str(args.ecm_coupling_mode)
    if mode == "single":
        tol_single = float(args.rsvd_tol_single)
        if tol_single <= 0.0:
            tol_single = float(min(args.rsvd_tol_res, args.rsvd_tol_eps, args.rsvd_tol_sig))

        a_res, a_eps, a_sig = _single_block_scales(np.asarray(q_res), c_eps, c_sig, str(args.single_block_normalization))
        m_all_t = np.ascontiguousarray(
            np.concatenate([
                float(a_res) * np.asarray(q_res.T, dtype=float),
                float(a_eps) * np.asarray(c_eps.T, dtype=float),
                float(a_sig) * np.asarray(c_sig.T, dtype=float),
            ], axis=1)
        )
        u_all, _, esvd_all = _run_rsvd_on_transpose(m_all_t, tol_single, label="ALL")
        ecm_tol_all = float(min(args.ecm_tol_res, args.ecm_tol_eps, args.ecm_tol_sig))
        z_all, w_all_sel, w_all_full = _run_ecm(u_all, n_elem, ecm_tol_all, None, "ALL", args.max_unsuccessful_it)

        z_res = z_all.copy()
        z_eps = z_all.copy()
        z_sig = z_all.copy()
        w_res = w_all_sel.copy()
        w_eps = w_all_sel.copy()
        w_sig = w_all_sel.copy()
        w_res_full = w_all_full.copy()
        w_eps_full = w_all_full.copy()
        w_sig_full = w_all_full.copy()

        esvd_res = np.nan
        esvd_eps = np.nan
        esvd_sig = np.nan
    else:
        u_res, _, esvd_res = _run_rsvd_on_transpose(np.asarray(q_res.T), args.rsvd_tol_res, label="RES")
        u_eps, _, esvd_eps = _run_rsvd_on_transpose(np.asarray(c_eps.T), args.rsvd_tol_eps, label="EPS")
        u_sig, _, esvd_sig = _run_rsvd_on_transpose(np.asarray(c_sig.T), args.rsvd_tol_sig, label="SIG")

        z_res, w_res, w_res_full = _run_ecm(u_res, n_elem, args.ecm_tol_res, None, "RES", args.max_unsuccessful_it)
        init_eps = z_res if mode == "cascade" else None
        z_eps, w_eps, w_eps_full = _run_ecm(u_eps, n_elem, args.ecm_tol_eps, init_eps, "EPS", args.max_unsuccessful_it)
        init_sig = z_eps if mode == "cascade" else None
        z_sig, w_sig, w_sig_full = _run_ecm(u_sig, n_elem, args.ecm_tol_sig, init_sig, "SIG", args.max_unsuccessful_it)

        esvd_all = np.nan

    # Final checks
    err_res = _rel_error(np.asarray(q_res) @ w_res_full, np.asarray(b_res))
    err_eps = _rel_error(np.asarray(c_eps) @ w_eps_full, np.asarray(b_eps))
    err_sig = _rel_error(np.asarray(c_sig) @ w_sig_full, np.asarray(b_sig))

    z_union = np.union1d(np.union1d(z_res, z_eps), z_sig).astype(int)

    print("\n" + "=" * 72)
    print("Selection summary")
    print("=" * 72)
    print(f"|Z_res|={z_res.size} ({100.0 * z_res.size / n_elem:.1f}%)")
    print(f"|Z_eps|={z_eps.size} ({100.0 * z_eps.size / n_elem:.1f}%)")
    print(f"|Z_sig|={z_sig.size} ({100.0 * z_sig.size / n_elem:.1f}%)")
    print(f"|Z_union|={z_union.size} ({100.0 * z_union.size / n_elem:.1f}%)")
    print(f"final errors: res={err_res:.3e}, eps={err_eps:.3e}, sig={err_sig:.3e}")

    out_file = os.path.join(args.out_dir, "ecm_weights_all.npz")
    np.savez(
        out_file,
        Z_res=z_res,
        Z_eps=z_eps,
        Z_sig=z_sig,
        Z_union=z_union,
        w_res=np.asarray(w_res, dtype=float),
        w_res_full=np.asarray(w_res_full, dtype=float),
        w_eps=np.asarray(w_eps, dtype=float),
        w_eps_full=np.asarray(w_eps_full, dtype=float),
        w_sig=np.asarray(w_sig, dtype=float),
        w_sig_full=np.asarray(w_sig_full, dtype=float),
        nq=np.array([nq], dtype=np.int64),
        Ns_res=np.array([ns_res], dtype=np.int64),
        Ns_hom=np.array([ns_hom], dtype=np.int64),
        n_elem=np.array([n_elem], dtype=np.int64),
        rel_error_res=np.array([err_res], dtype=float),
        rel_error_eps=np.array([err_eps], dtype=float),
        rel_error_sig=np.array([err_sig], dtype=float),
        RSVD_TOL_RES=np.array([float(args.rsvd_tol_res)], dtype=float),
        RSVD_TOL_EPS=np.array([float(args.rsvd_tol_eps)], dtype=float),
        RSVD_TOL_SIG=np.array([float(args.rsvd_tol_sig)], dtype=float),
        ECM_TOL_RES=np.array([float(args.ecm_tol_res)], dtype=float),
        ECM_TOL_EPS=np.array([float(args.ecm_tol_eps)], dtype=float),
        ECM_TOL_SIG=np.array([float(args.ecm_tol_sig)], dtype=float),
        eSVD_res=np.array([float(esvd_res)], dtype=float),
        eSVD_eps=np.array([float(esvd_eps)], dtype=float),
        eSVD_sig=np.array([float(esvd_sig)], dtype=float),
        eSVD_all=np.array([float(esvd_all)], dtype=float),
        coupling_mode=np.array([mode]),
        single_block_normalization=np.array([str(args.single_block_normalization)]),
        A0_ref=np.array([float(a0_ref)], dtype=float),
        hom_reference_measure=np.array([float(a0_ref)], dtype=float),
        data_dir=np.array([str(args.data_dir)]),
    )

    with open(os.path.join(args.out_dir, "reference_measure_A0.txt"), "w", encoding="utf-8") as f:
        f.write(f"{float(a0_ref):.16e}\n")

    print(f"\n[DONE] Saved -> {out_file}")


if __name__ == "__main__":
    main()
