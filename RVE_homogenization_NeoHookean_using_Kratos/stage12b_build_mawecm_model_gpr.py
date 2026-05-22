#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage 12b: Build MAW-ECM model (offline pruning + weight regression) for HPROM-GPR.

Default behavior follows the residual-centric MAW setup:
  - Stage12b first computes classical split ECM (self-contained) from Stage12a data.
  - MAW is then trained only for residual support/weights.
  - Homogenization supports/weights are kept classical (no MAW) unless requested.
  - Residual RHS targets come from the dataset (b_full), not from anchor reconstruction.
  - Graph regularization defaults to structured Stage12 mesh connectivity when available.
"""

from __future__ import annotations

import os
import argparse
import numpy as np

from mawecm_graph_utils import build_knn_graph_laplacian, build_cell_graph_laplacian
from mawecm_pruning import run_mawecm_pruning
from mawecm_rbf_weights import fit_mawecm_rbf, eval_mawecm_rbf
from empirical_cubature_method import EmpiricalCubatureMethod
from randomized_singular_value_decomposition import RandomizedSingularValueDecomposition


DATASET_DIR = "stage_12_mawecm_dataset_gpr_ls"
FIXED_ECM_DIR = "stage_9_hprom_gpr_data_ls"
OUT_DIR = "stage_12_hprom_mawecm_gpr_data_ls"


def _meta_int(meta, key):
    return int(np.ravel(meta[key])[0])


def _meta_float(meta, key):
    return float(np.ravel(meta[key])[0])


def _load_stage12a_dataset(dataset_dir, require_hom=False, load_hom=True):
    meta_path = os.path.join(dataset_dir, "meta.npz")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(meta_path)

    meta = np.load(meta_path, allow_pickle=True)
    nq = _meta_int(meta, "nq")
    n_elem = _meta_int(meta, "n_elem")
    ns_res = _meta_int(meta, "N_s_res")
    ns_hom_meta = _meta_int(meta, "N_s_hom")

    q_file = os.path.join(dataset_dir, "Q_ecm.dat")
    b_file = os.path.join(dataset_dir, "b_full.dat")
    c_file = os.path.join(dataset_dir, "C_hom.dat")
    bh_file = os.path.join(dataset_dir, "b_hom.dat")
    qh_file = os.path.join(dataset_dir, "q_p_hom.npy")
    muh_file = os.path.join(dataset_dir, "mu_hom.npy")
    idh_file = os.path.join(dataset_dir, "sample_ids_hom.npy")

    for p in (q_file, b_file):
        if not os.path.exists(p):
            raise FileNotFoundError(p)

    q_res_mm = np.memmap(q_file, dtype=np.float64, mode="r", shape=(nq * ns_res, n_elem))
    b_res_mm = np.memmap(b_file, dtype=np.float64, mode="r", shape=(nq * ns_res,))
    q_p_res = np.asarray(np.load(os.path.join(dataset_dir, "q_p_res.npy")), dtype=float)
    mu_res = np.asarray(np.load(os.path.join(dataset_dir, "mu_res.npy")), dtype=float)
    ids_res = np.asarray(np.load(os.path.join(dataset_dir, "sample_ids_res.npy")), dtype=np.int64)
    cells_struct_file = os.path.join(dataset_dir, "structured_mesh_cells_res.npy")
    grid_struct_file = os.path.join(dataset_dir, "structured_mesh_grid_shape_res.npy")
    cells_struct_res = (
        np.asarray(np.load(cells_struct_file), dtype=np.int64)
        if os.path.exists(cells_struct_file)
        else None
    )
    grid_struct_res = (
        np.asarray(np.load(grid_struct_file), dtype=np.int64)
        if os.path.exists(grid_struct_file)
        else None
    )

    if q_p_res.shape != (ns_res, nq):
        raise RuntimeError(f"q_p_res shape {q_p_res.shape} != ({ns_res},{nq})")
    q_res = np.asarray(q_res_mm, dtype=float)
    b_res = np.asarray(b_res_mm, dtype=float)
    q_ok_rows = np.isfinite(q_res).all(axis=1)
    b_ok_rows = np.isfinite(b_res)
    row_ok = q_ok_rows & b_ok_rows
    sample_ok = row_ok.reshape(ns_res, nq).all(axis=1)
    sample_ok &= np.isfinite(q_p_res).all(axis=1)
    sample_ok &= np.isfinite(mu_res).all(axis=1)
    n_bad = int(np.sum(~sample_ok))
    if n_bad > 0:
        bad_ids = np.flatnonzero(~sample_ok)[:10]
        raise RuntimeError(
            "Stage12a dataset contains non-finite residual samples. "
            f"Bad samples: {n_bad}/{ns_res}, first ids={bad_ids.tolist()}. "
            "Rebuild Stage12a with the latest structured admissibility fix."
        )

    hom_files = (c_file, bh_file, qh_file, muh_file, idh_file)
    hom_ready = all(os.path.exists(p) for p in hom_files)
    ns_hom = int(ns_hom_meta)
    c_hom = None
    b_hom = None
    q_p_hom = np.zeros((0, nq), dtype=float)
    mu_hom = np.zeros((0, 3), dtype=float)
    ids_hom = np.zeros((0, 2), dtype=np.int64)
    if bool(load_hom) and ns_hom > 0 and hom_ready:
        c_hom = np.memmap(c_file, dtype=np.float64, mode="r", shape=(6 * ns_hom, n_elem))
        b_hom = np.memmap(bh_file, dtype=np.float64, mode="r", shape=(6 * ns_hom,))
        q_p_hom = np.asarray(np.load(qh_file), dtype=float)
        mu_hom = np.asarray(np.load(muh_file), dtype=float)
        ids_hom = np.asarray(np.load(idh_file), dtype=np.int64)
        if q_p_hom.shape != (ns_hom, nq):
            raise RuntimeError(f"q_p_hom shape {q_p_hom.shape} != ({ns_hom},{nq})")
    else:
        if bool(load_hom) and ns_hom > 0 and bool(require_hom):
            missing = [p for p in hom_files if not os.path.exists(p)]
            raise FileNotFoundError(
                "Stage12a homogenization data is required but missing. "
                f"Missing: {missing}"
            )
        ns_hom = 0
        if bool(require_hom):
            raise RuntimeError("Stage12a homogenization snapshots are required but N_s_hom=0.")

    return {
        "meta": meta,
        "nq": nq,
        "n_elem": n_elem,
        "ns_res": ns_res,
        "ns_hom": ns_hom,
        "Q_res": q_res,
        "b_res": b_res,
        "C_hom": c_hom,
        "b_hom": b_hom,
        "q_p_res": q_p_res,
        "q_p_hom": q_p_hom,
        "mu_res": mu_res,
        "mu_hom": mu_hom,
        "ids_res": ids_res,
        "ids_hom": ids_hom,
        "cells_struct_res": cells_struct_res,
        "grid_struct_res": grid_struct_res,
    }


def _load_fixed_ecm(fixed_ecm_dir):
    ecm_path = os.path.join(fixed_ecm_dir, "ecm_weights_all.npz")
    if not os.path.exists(ecm_path):
        raise FileNotFoundError(ecm_path)
    ecm = np.load(ecm_path, allow_pickle=True)
    return {k: ecm[k] for k in ecm.files}


def _rel_error(a, b):
    aa = np.asarray(a, dtype=float).reshape(-1)
    bb = np.asarray(b, dtype=float).reshape(-1)
    return float(np.linalg.norm(aa - bb) / max(np.linalg.norm(bb), 1.0e-30))


def _run_rsvd_on_transpose(M_T, rsvd_tol, label=""):
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
    print(f"  [Classic-{label}] RSVD kept={s.size}, eSVD={float(eSVD):.3e}")
    return np.asarray(U, dtype=float), np.asarray(s, dtype=float), float(eSVD)


def _run_ecm(U_basis, n_elem, ecm_tol, init_candidates, label="", max_unsuccessful_it=200):
    ecm = EmpiricalCubatureMethod(
        ECM_tolerance=float(ecm_tol),
        Filter_tolerance=0.0,
        Plotting=False,
        MaximumNumberUnsuccesfulIterations=int(max_unsuccessful_it),
    )
    ecm.SetUp(
        ResidualsBasis=np.asarray(U_basis, dtype=float),
        InitialCandidatesSet=init_candidates,
        constrain_sum_of_weights=False,
        constrain_conditions=False,
        number_of_conditions=0,
    )
    ecm.Run()

    z = np.asarray(ecm.z, dtype=np.int64).reshape(-1)
    w_sel = np.asarray(ecm.w, dtype=float).reshape(-1)
    w_full = np.zeros(int(n_elem), dtype=float)
    w_full[z] = w_sel
    print(
        f"  [Classic-{label}] |Z|={z.size} "
        f"({100.0 * z.size / max(int(n_elem), 1):.1f}% of {int(n_elem)})"
    )
    return z, w_sel, w_full


def _normalize_classic_coupling_mode(mode_raw):
    mode = str(mode_raw).strip().lower()
    if mode in ("independent", "decoupled"):
        return "independent"
    if mode in ("cascade", "coupled", "sequential"):
        return "cascade"
    raise ValueError(
        f"Unsupported --classic-ecm-coupling-mode='{mode_raw}'. "
        "Use one of: independent, cascade."
    )


def _compute_classic_ecm_from_dataset(dataset, args, include_homogenization=True):
    nq = int(dataset["nq"])
    ns_res = int(dataset["ns_res"])
    ns_hom = int(dataset["ns_hom"])
    n_elem = int(dataset["n_elem"])

    q_res = np.asarray(dataset["Q_res"], dtype=float)
    b_res = np.asarray(dataset["b_res"], dtype=float)
    print("-" * 72)
    print("Stage 12b classical ECM bootstrap (self-contained)")
    print(f"  nq={nq}, Ns_res={ns_res}, Ns_hom={ns_hom}, n_elem={n_elem}")
    print(f"  coupling mode={args.classic_ecm_coupling_mode}")
    print(f"  include homogenization={bool(include_homogenization)}")

    u_res, _, e_res = _run_rsvd_on_transpose(
        q_res.T, float(args.classic_rsvd_tol_res), label="RES"
    )
    e_eps = np.nan
    e_sig = np.nan
    c_eps = None
    c_sig = None
    b_eps = None
    b_sig = None
    coupling = _normalize_classic_coupling_mode(args.classic_ecm_coupling_mode)
    if bool(include_homogenization):
        if ns_hom <= 0 or dataset["C_hom"] is None or dataset["b_hom"] is None:
            raise RuntimeError(
                "Classical homogenization ECM requested but Stage12a dataset has no homogenization blocks."
            )
        c_hom = np.asarray(dataset["C_hom"], dtype=float)
        b_hom = np.asarray(dataset["b_hom"], dtype=float)
        c_blk = c_hom.reshape(ns_hom, 6, n_elem)
        b_blk = b_hom.reshape(ns_hom, 6)
        c_eps = c_blk[:, 0:3, :].reshape(3 * ns_hom, n_elem)
        b_eps = b_blk[:, 0:3].reshape(3 * ns_hom)
        c_sig = c_blk[:, 3:6, :].reshape(3 * ns_hom, n_elem)
        b_sig = b_blk[:, 3:6].reshape(3 * ns_hom)
        u_eps, _, e_eps = _run_rsvd_on_transpose(
            c_eps.T, float(args.classic_rsvd_tol_eps), label="EPS"
        )
        u_sig, _, e_sig = _run_rsvd_on_transpose(
            c_sig.T, float(args.classic_rsvd_tol_sig), label="SIG"
        )

    z_res, w_res, w_res_full = _run_ecm(
        U_basis=u_res,
        n_elem=n_elem,
        ecm_tol=float(args.classic_ecm_tol_res),
        init_candidates=None,
        label="RES",
        max_unsuccessful_it=int(args.classic_max_unsuccessful_it),
    )
    if bool(include_homogenization):
        eps_init = np.asarray(z_res, dtype=np.int64) if coupling == "cascade" else None
        z_eps, w_eps, w_eps_full = _run_ecm(
            U_basis=u_eps,
            n_elem=n_elem,
            ecm_tol=float(args.classic_ecm_tol_eps),
            init_candidates=eps_init,
            label="EPS",
            max_unsuccessful_it=int(args.classic_max_unsuccessful_it),
        )
        sig_init = np.asarray(z_eps, dtype=np.int64) if coupling == "cascade" else None
        z_sig, w_sig, w_sig_full = _run_ecm(
            U_basis=u_sig,
            n_elem=n_elem,
            ecm_tol=float(args.classic_ecm_tol_sig),
            init_candidates=sig_init,
            label="SIG",
            max_unsuccessful_it=int(args.classic_max_unsuccessful_it),
        )
    else:
        z_eps = np.zeros(0, dtype=np.int64)
        z_sig = np.zeros(0, dtype=np.int64)
        w_eps = np.zeros(0, dtype=float)
        w_sig = np.zeros(0, dtype=float)
        w_eps_full = np.zeros(n_elem, dtype=float)
        w_sig_full = np.zeros(n_elem, dtype=float)

    err_res = _rel_error(q_res @ w_res_full, b_res)
    err_eps = _rel_error(c_eps @ w_eps_full, b_eps) if bool(include_homogenization) else np.nan
    err_sig = _rel_error(c_sig @ w_sig_full, b_sig) if bool(include_homogenization) else np.nan
    z_union = np.union1d(np.union1d(z_res, z_eps), z_sig).astype(np.int64)
    if bool(include_homogenization):
        print(
            f"  [Classic] final errors: res={err_res:.3e}, eps={err_eps:.3e}, sig={err_sig:.3e}"
        )
        print(
            f"  [Classic] |Z_res|={z_res.size}, |Z_eps|={z_eps.size}, "
            f"|Z_sig|={z_sig.size}, |Z_union|={z_union.size}"
        )
    else:
        print(f"  [Classic] final residual error: res={err_res:.3e}")
        print(f"  [Classic] |Z_res|={z_res.size}")

    return {
        "Z_res": z_res,
        "Z_eps": z_eps,
        "Z_sig": z_sig,
        "Z_union": z_union,
        "w_res": w_res,
        "w_eps": w_eps,
        "w_sig": w_sig,
        "w_res_full": w_res_full,
        "w_eps_full": w_eps_full,
        "w_sig_full": w_sig_full,
        "n_elem": np.array([n_elem], dtype=np.int64),
        "nq": np.array([nq], dtype=np.int64),
        "Ns_res": np.array([ns_res], dtype=np.int64),
        "Ns_hom": np.array([ns_hom], dtype=np.int64),
        "RSVD_TOL_RES": np.array([float(args.classic_rsvd_tol_res)], dtype=float),
        "RSVD_TOL_EPS": np.array([float(args.classic_rsvd_tol_eps)], dtype=float),
        "RSVD_TOL_SIG": np.array([float(args.classic_rsvd_tol_sig)], dtype=float),
        "eSVD_res": np.array([float(e_res)], dtype=float),
        "eSVD_eps": np.array([float(e_eps)], dtype=float),
        "eSVD_sig": np.array([float(e_sig)], dtype=float),
        "ECM_TOL_RES": np.array([float(args.classic_ecm_tol_res)], dtype=float),
        "ECM_TOL_EPS": np.array([float(args.classic_ecm_tol_eps)], dtype=float),
        "ECM_TOL_SIG": np.array([float(args.classic_ecm_tol_sig)], dtype=float),
        "ECM_COUPLING_MODE": np.array(
            [str(coupling) if bool(include_homogenization) else "residual_only"]
        ),
        "ECM_COUPLING_MODE_INPUT": np.array([str(args.classic_ecm_coupling_mode)]),
        "rel_error_res": np.array([float(err_res)], dtype=float),
        "rel_error_eps": np.array([float(err_eps)], dtype=float),
        "rel_error_sig": np.array([float(err_sig)], dtype=float),
        "data_dir": np.array([str(args.dataset_dir)]),
        "A0_ref": np.array([_meta_float(dataset["meta"], "A0_ref")], dtype=float),
        "hom_reference_measure": np.array([_meta_float(dataset["meta"], "A0_ref")], dtype=float),
    }


def _build_blocks_res(dataset, z_ini, w_ini, rhs_mode="dataset"):
    nq = int(dataset["nq"])
    ns = int(dataset["ns_res"])
    Q_res = dataset["Q_res"]
    b_res = dataset["b_res"]
    mode = str(rhs_mode).strip().lower()
    A_blocks = []
    b_blocks = []
    for s in range(ns):
        r0, r1 = nq * s, nq * (s + 1)
        A_full = np.asarray(Q_res[r0:r1, :], dtype=float)
        A = A_full[:, z_ini]
        if mode == "dataset":
            b = np.asarray(b_res[r0:r1], dtype=float)
        elif mode == "anchor":
            b = A @ w_ini
        else:
            raise ValueError(f"Unsupported residual rhs mode '{rhs_mode}'.")
        A_blocks.append(A)
        b_blocks.append(b)
    q_train = np.asarray(dataset["q_p_res"], dtype=float)
    mu_train = np.asarray(dataset["mu_res"], dtype=float)
    ids = np.asarray(dataset["ids_res"], dtype=np.int64)
    return A_blocks, b_blocks, q_train, mu_train, ids


def _build_blocks_eps(dataset, z_ini, w_ini, rhs_mode="anchor"):
    ns = int(dataset["ns_hom"])
    if ns <= 0 or dataset["C_hom"] is None or dataset["b_hom"] is None:
        raise RuntimeError("EPS MAW requested but dataset has no homogenization blocks.")
    C_hom = dataset["C_hom"]
    b_hom = dataset["b_hom"]
    mode = str(rhs_mode).strip().lower()
    A_blocks = []
    b_blocks = []
    for s in range(ns):
        r0, r1 = 6 * s, 6 * s + 3
        A_full = np.asarray(C_hom[r0:r1, :], dtype=float)
        A = A_full[:, z_ini]
        if mode == "dataset":
            b = np.asarray(b_hom[r0:r1], dtype=float)
        elif mode == "anchor":
            b = A @ w_ini
        else:
            raise ValueError(f"Unsupported eps rhs mode '{rhs_mode}'.")
        A_blocks.append(A)
        b_blocks.append(b)
    q_train = np.asarray(dataset["q_p_hom"], dtype=float)
    mu_train = np.asarray(dataset["mu_hom"], dtype=float)
    ids = np.asarray(dataset["ids_hom"], dtype=np.int64)
    return A_blocks, b_blocks, q_train, mu_train, ids


def _build_blocks_sig(dataset, z_ini, w_ini, rhs_mode="anchor"):
    ns = int(dataset["ns_hom"])
    if ns <= 0 or dataset["C_hom"] is None or dataset["b_hom"] is None:
        raise RuntimeError("SIG MAW requested but dataset has no homogenization blocks.")
    C_hom = dataset["C_hom"]
    b_hom = dataset["b_hom"]
    mode = str(rhs_mode).strip().lower()
    A_blocks = []
    b_blocks = []
    for s in range(ns):
        r0, r1 = 6 * s + 3, 6 * s + 6
        A_full = np.asarray(C_hom[r0:r1, :], dtype=float)
        A = A_full[:, z_ini]
        if mode == "dataset":
            b = np.asarray(b_hom[r0:r1], dtype=float)
        elif mode == "anchor":
            b = A @ w_ini
        else:
            raise ValueError(f"Unsupported sig rhs mode '{rhs_mode}'.")
        A_blocks.append(A)
        b_blocks.append(b)
    q_train = np.asarray(dataset["q_p_hom"], dtype=float)
    mu_train = np.asarray(dataset["mu_hom"], dtype=float)
    ids = np.asarray(dataset["ids_hom"], dtype=np.int64)
    return A_blocks, b_blocks, q_train, mu_train, ids


def _build_graph_for_target(dataset, q_train, target, args):
    mode = str(args.graph_mode).strip().lower()
    if mode == "structured_stage12":
        if str(target).strip().lower() != "res":
            raise RuntimeError(
                "graph-mode=structured_stage12 is currently supported only for residual target."
            )
        cells = dataset.get("cells_struct_res", None)
        if cells is None:
            raise RuntimeError(
                "structured_mesh_cells_res.npy not found in dataset. "
                "Rebuild Stage12a with --residual-state-source stage7_structured_mesh."
            )
        return build_cell_graph_laplacian(
            n_nodes=int(q_train.shape[0]),
            cells=np.asarray(cells, dtype=np.int64),
            cell_type="auto",
            weight_mode="binary",
        )
    if mode == "knn":
        sigma = None if float(args.graph_sigma) <= 0.0 else float(args.graph_sigma)
        return build_knn_graph_laplacian(
            q_train=q_train,
            knn=int(args.graph_knn),
            metric="euclidean",
            kernel=str(args.graph_kernel),
            sigma=sigma,
        )
    raise ValueError(f"Unsupported --graph-mode='{args.graph_mode}'.")


def _select_initial_support_and_weights(ecm_fixed, target, n_elem, candidate_pool="fixed_support"):
    t = str(target).strip().lower()
    pool = str(candidate_pool).strip().lower()
    if pool not in ("fixed_support", "full_mesh"):
        raise ValueError(f"Unsupported candidate pool '{candidate_pool}'.")

    if t == "res":
        z_fix = np.asarray(ecm_fixed["Z_res"], dtype=np.int64).reshape(-1)
        w_fix = np.asarray(ecm_fixed["w_res"], dtype=float).reshape(-1)
        if pool == "fixed_support":
            z = z_fix
            w = w_fix
        else:
            z = np.arange(int(n_elem), dtype=np.int64)
            if "w_res_full" in ecm_fixed:
                wf = np.asarray(ecm_fixed["w_res_full"], dtype=float).reshape(-1)
                if wf.size != int(n_elem):
                    raise RuntimeError(
                        f"w_res_full size {wf.size} != n_elem {int(n_elem)}."
                    )
                w = np.asarray(wf, dtype=float)
                if np.sum(np.abs(w)) <= 1.0e-30:
                    w = np.ones(int(n_elem), dtype=float)
            else:
                w = np.ones(int(n_elem), dtype=float)
            # Ensure all full-mesh candidates are initially active in MAW pruning.
            w = np.maximum(w, 1.0e-12)
    elif t == "eps":
        z = np.asarray(ecm_fixed["Z_eps"], dtype=np.int64).reshape(-1)
        w = np.asarray(ecm_fixed["w_eps"], dtype=float).reshape(-1)
    elif t == "sig":
        z = np.asarray(ecm_fixed["Z_sig"], dtype=np.int64).reshape(-1)
        w = np.asarray(ecm_fixed["w_sig"], dtype=float).reshape(-1)
    else:
        raise ValueError(f"Unsupported target {target}")
    if z.size != w.size:
        raise RuntimeError(f"Fixed ECM size mismatch for {target}: |Z|={z.size}, |w|={w.size}")
    return z, w


def _run_target_maw(
    target,
    dataset,
    ecm_fixed,
    n_elem,
    args,
):
    if target == "res":
        z_ini, w_ini = _select_initial_support_and_weights(
            ecm_fixed,
            target,
            n_elem=n_elem,
            candidate_pool=str(args.res_candidate_pool),
        )
        rhs_mode = str(args.res_target_source)
    else:
        z_ini, w_ini = _select_initial_support_and_weights(
            ecm_fixed,
            target,
            n_elem=n_elem,
            candidate_pool="fixed_support",
        )
        rhs_mode = "anchor"

    if target == "res":
        A_blocks, b_blocks, q_train, mu_train, ids = _build_blocks_res(
            dataset, z_ini, w_ini, rhs_mode=rhs_mode
        )
        n_centers = int(args.rbf_centers_res)
    elif target == "eps":
        A_blocks, b_blocks, q_train, mu_train, ids = _build_blocks_eps(
            dataset, z_ini, w_ini, rhs_mode=rhs_mode
        )
        n_centers = int(args.rbf_centers_eps)
    else:
        A_blocks, b_blocks, q_train, mu_train, ids = _build_blocks_sig(
            dataset, z_ini, w_ini, rhs_mode=rhs_mode
        )
        n_centers = int(args.rbf_centers_sig)

    K_graph = _build_graph_for_target(
        dataset=dataset,
        q_train=q_train,
        target=target,
        args=args,
    )

    prune_opts = {
        "K_graph": K_graph,
        "alpha_smooth": float(args.alpha_smooth),
        "criterion": int(args.criterion),
        "number_of_candidates_to_try": int(args.n_candidates_to_try) if int(args.n_candidates_to_try) > 0 else None,
        "incremental_smoothing": bool(int(args.incremental_smoothing)),
        "use_total_as_criterion": bool(int(args.use_total_as_criterion)),
        "tol_rank_rel": float(args.tol_rank_rel),
        "tol_neg_factor": float(args.tol_neg_factor),
        "tol_zero": float(args.tol_zero),
        "max_active_set_iters": int(args.max_as_iters),
        "max_reduced_dim": int(args.max_reduced_dim),
        "verbose": True,
    }

    maw = run_mawecm_pruning(
        A_blocks=A_blocks,
        b_blocks=b_blocks,
        z_ini=z_ini,
        w_ini=w_ini,
        q_train=q_train,
        options=prune_opts,
    )

    W_train = np.asarray(maw["W_support"], dtype=float)
    z_red = np.asarray(maw["Z_support"], dtype=np.int64)
    if z_red.size == 0:
        raise RuntimeError(f"MAW returned empty support for target={target}.")

    renorm_target = float(np.sum(w_ini))
    rbf = fit_mawecm_rbf(
        q_train=q_train,
        W_train=W_train,
        n_centers=n_centers,
        poly_mode=int(args.rbf_poly_mode),
        lambda_reg=float(args.rbf_lambda),
        length_scale_factor=float(args.rbf_length_scale_factor),
    )

    W_recon = eval_mawecm_rbf(
        q_query=q_train,
        model=rbf,
        clip_nonnegative=bool(int(args.rbf_clip_nonnegative)),
        renorm_target=renorm_target if bool(int(args.rbf_renorm)) else None,
    )
    rel_recon = float(np.linalg.norm(W_recon - W_train) / max(np.linalg.norm(W_train), 1.0e-30))

    # Fixed anchor vectors for compatibility with existing Stage-9/10-style consumers.
    w_full = np.zeros(int(n_elem), dtype=float)
    w_anchor = np.mean(W_train, axis=1)
    w_full[z_red] = w_anchor

    print(
        f"  [MAW-{target}] rhs={rhs_mode} |Z_ini|={z_ini.size} -> |Z_red|={z_red.size}, "
        f"RBF train-rel={rel_recon:.3e}, prune-elapsed={maw['elapsed_sec']:.2f}s"
    )

    return {
        "target": target,
        "z_ini": z_ini,
        "w_ini": w_ini,
        "q_train": q_train,
        "mu_train": mu_train,
        "ids": ids,
        "A_m": int(np.asarray(b_blocks[0]).size),
        "b_train": np.asarray(np.vstack([np.asarray(bj, dtype=float) for bj in b_blocks]), dtype=float),
        "maw": maw,
        "z_red": z_red,
        "W_train": W_train,
        "rbf": rbf,
        "w_full": w_full,
        "w_sel": w_full[z_red],
        "renorm_target": float(renorm_target),
        "recon_rel": rel_recon,
        "rhs_mode": np.array([rhs_mode]),
    }


def _build_fixed_hom_target_result(dataset, ecm_fixed, target):
    t = str(target).strip().lower()
    if t == "eps":
        z = np.asarray(ecm_fixed["Z_eps"], dtype=np.int64).reshape(-1)
        w = np.asarray(ecm_fixed["w_eps"], dtype=float).reshape(-1)
        q_train = np.asarray(dataset["q_p_hom"], dtype=float)
        mu_train = np.asarray(dataset["mu_hom"], dtype=float)
        ids = np.asarray(dataset["ids_hom"], dtype=np.int64)
    elif t == "sig":
        z = np.asarray(ecm_fixed["Z_sig"], dtype=np.int64).reshape(-1)
        w = np.asarray(ecm_fixed["w_sig"], dtype=float).reshape(-1)
        q_train = np.asarray(dataset["q_p_hom"], dtype=float)
        mu_train = np.asarray(dataset["mu_hom"], dtype=float)
        ids = np.asarray(dataset["ids_hom"], dtype=np.int64)
    else:
        raise ValueError(f"Unsupported fixed hom target '{target}'.")
    if z.size != w.size:
        raise RuntimeError(f"Fixed ECM size mismatch for {target}: |Z|={z.size}, |w|={w.size}")
    n_elem = int(dataset["n_elem"])
    w_full = np.zeros(n_elem, dtype=float)
    w_full[z] = w
    return {
        "target": t,
        "z_ini": z.copy(),
        "w_ini": w.copy(),
        "q_train": q_train,
        "mu_train": mu_train,
        "ids": ids,
        "A_m": int(3),
        "b_train": np.zeros((q_train.shape[0], 3), dtype=float),
        "maw": None,
        "z_red": z.copy(),
        "W_train": np.tile(w[:, None], (1, q_train.shape[0])),
        "rbf": None,
        "w_full": w_full,
        "w_sel": w.copy(),
        "renorm_target": float(np.sum(w)),
        "recon_rel": 0.0,
        "rhs_mode": np.array(["fixed_classic"]),
        "fixed_classic": True,
    }


def _build_disabled_hom_target_result(dataset, target):
    t = str(target).strip().lower()
    q_train = np.zeros((0, int(dataset["nq"])), dtype=float)
    mu_train = np.zeros((0, 3), dtype=float)
    ids = np.zeros((0, 2), dtype=np.int64)
    return {
        "target": t,
        "z_ini": np.zeros(0, dtype=np.int64),
        "w_ini": np.zeros(0, dtype=float),
        "q_train": q_train,
        "mu_train": mu_train,
        "ids": ids,
        "A_m": int(3),
        "b_train": np.zeros((0, 3), dtype=float),
        "maw": None,
        "z_red": np.zeros(0, dtype=np.int64),
        "W_train": np.zeros((0, 0), dtype=float),
        "rbf": None,
        "w_full": np.zeros(int(dataset["n_elem"]), dtype=float),
        "w_sel": np.zeros(0, dtype=float),
        "renorm_target": 0.0,
        "recon_rel": 0.0,
        "rhs_mode": np.array(["disabled"]),
        "fixed_classic": True,
    }


def _parse_targets(text):
    vals = [v.strip().lower() for v in str(text).split(",") if v.strip()]
    valid = []
    for v in vals:
        if v not in ("res", "eps", "sig"):
            raise ValueError(f"Invalid target '{v}'. Use subset of: res,eps,sig")
        if v not in valid:
            valid.append(v)
    if not valid:
        raise ValueError("No targets selected.")
    return valid


def _save_stage12b_file(out_file, dataset, ecm_fixed, results, args):
    n_elem = int(dataset["n_elem"])

    # Compatibility vectors
    z_res = np.asarray(results["res"]["z_red"], dtype=np.int64)
    z_eps = np.asarray(results["eps"]["z_red"], dtype=np.int64)
    z_sig = np.asarray(results["sig"]["z_red"], dtype=np.int64)
    z_union = np.union1d(np.union1d(z_res, z_eps), z_sig).astype(np.int64)

    w_res_full = np.asarray(results["res"]["w_full"], dtype=float)
    w_eps_full = np.asarray(results["eps"]["w_full"], dtype=float)
    w_sig_full = np.asarray(results["sig"]["w_full"], dtype=float)

    maw_enabled_targets = [t for t in ("res", "eps", "sig") if results[t]["maw"] is not None]
    fixed_dir_tag = (
        str(args.fixed_ecm_dir)
        if str(args.classic_ecm_source).strip().lower() == "file"
        else "__self_stage12_classic__"
    )
    payload = {
        "Z_res": z_res,
        "Z_eps": z_eps,
        "Z_sig": z_sig,
        "Z_union": z_union,
        "w_res": np.asarray(results["res"]["w_sel"], dtype=float),
        "w_eps": np.asarray(results["eps"]["w_sel"], dtype=float),
        "w_sig": np.asarray(results["sig"]["w_sel"], dtype=float),
        "w_res_full": w_res_full,
        "w_eps_full": w_eps_full,
        "w_sig_full": w_sig_full,
        "n_elem": np.array([n_elem], dtype=np.int64),
        "nq": np.array([int(dataset["nq"])], dtype=np.int64),
        "Ns_res": np.array([int(dataset["ns_res"])], dtype=np.int64),
        "Ns_hom": np.array([int(dataset["ns_hom"])], dtype=np.int64),
        "A0_ref": np.array([_meta_float(dataset["meta"], "A0_ref")], dtype=float),
        "hom_reference_measure": np.array([_meta_float(dataset["meta"], "A0_ref")], dtype=float),
        "ECM_COUPLING_MODE": np.array(
            ["mawecm_residual_only" if bool(int(args.disable_homogenization)) else "mawecm"]
        ),
        "ECM_COUPLING_MODE_INPUT": np.array(
            ["mawecm_residual_only" if bool(int(args.disable_homogenization)) else "mawecm"]
        ),
        "hprom_model_type": np.array(["MAW_ECM_GPR"]),
        "data_dir": np.array([str(args.dataset_dir)]),
        "fixed_ecm_dir": np.array([fixed_dir_tag]),
        "classic_ecm_source": np.array([str(args.classic_ecm_source)]),
        "MAW_VERSION": np.array(["v1"]),
        "maw_q_dim": np.array([int(dataset["q_p_res"].shape[1])], dtype=np.int64),
        "maw_targets": np.array([",".join(maw_enabled_targets)]),
        "maw_hom_mode": np.array(
            ["disabled" if bool(int(args.disable_homogenization)) else str(args.hom_mode)]
        ),
        "maw_res_candidate_pool": np.array([str(args.res_candidate_pool)]),
        "maw_res_target_source": np.array([str(args.res_target_source)]),
        "maw_graph_type": np.array([str(args.graph_mode)]),
        "maw_knn": np.array([int(args.graph_knn)], dtype=np.int64),
        "maw_graph_kernel": np.array([str(args.graph_kernel)]),
        "maw_graph_sigma": np.array([float(args.graph_sigma)], dtype=float),
        "maw_alpha_smooth": np.array([float(args.alpha_smooth)], dtype=float),
        "maw_stage2_criterion": np.array([int(args.criterion)], dtype=np.int64),
        "maw_incremental_smoothing": np.array([int(args.incremental_smoothing)], dtype=np.int64),
        "maw_postprocess_mode": np.array(["clip+renorm" if int(args.rbf_renorm) else "clip"]),
    }
    if dataset.get("grid_struct_res", None) is not None:
        payload["maw_res_structured_grid_shape"] = np.asarray(
            dataset["grid_struct_res"], dtype=np.int64
        )

    for t in ("res", "eps", "sig"):
        rec = results[t]
        maw = rec["maw"]
        rbf = rec["rbf"]
        prefix = f"maw_{t}_"

        payload[prefix + "Z_ini"] = np.asarray(rec["z_ini"], dtype=np.int64)
        payload[prefix + "w_ini"] = np.asarray(rec["w_ini"], dtype=float)
        payload[prefix + "sample_ids"] = np.asarray(rec["ids"], dtype=np.int64)
        payload[prefix + "q_train"] = np.asarray(rec["q_train"], dtype=float)
        payload[prefix + "mu_train"] = np.asarray(rec["mu_train"], dtype=float)
        payload[prefix + "W_train"] = np.asarray(rec["W_train"], dtype=float)
        payload[prefix + "b_train"] = np.asarray(rec["b_train"], dtype=float)
        payload[prefix + "m_constraints"] = np.array([int(rec["A_m"])], dtype=np.int64)
        payload[prefix + "recon_rel"] = np.array([float(rec["recon_rel"])], dtype=float)
        payload[prefix + "rhs_mode"] = np.array([str(np.ravel(rec["rhs_mode"])[0])])
        payload[prefix + "is_fixed_classic"] = np.array(
            [1 if bool(rec.get("fixed_classic", False)) else 0],
            dtype=np.int64,
        )
        payload[prefix + "renorm_target"] = np.array([float(rec["renorm_target"])], dtype=float)
        if maw is not None and rbf is not None:
            payload[prefix + "n_stop"] = np.array([int(maw["n_stop"])], dtype=np.int64)
            payload[prefix + "prune_history_active_counts"] = np.asarray(
                maw["active_counts"], dtype=np.int64
            )
            payload[prefix + "prune_history_removed_global"] = np.asarray(
                maw["removed_local"], dtype=np.int64
            )
            payload[prefix + "tol_neg"] = np.array([float(maw["tol_neg"])], dtype=float)
            payload[prefix + "alpha_smooth"] = np.array(
                [float(maw["options"]["alpha_smooth"])], dtype=float
            )
            payload[prefix + "criterion"] = np.array(
                [int(maw["options"]["criterion"])], dtype=np.int64
            )
            payload[prefix + "elapsed_sec"] = np.array([float(maw["elapsed_sec"])], dtype=float)

            payload[prefix + "rbf_centers"] = np.asarray(rbf["centers"], dtype=float)
            payload[prefix + "rbf_center_ids"] = np.asarray(rbf["center_ids"], dtype=np.int64)
            payload[prefix + "rbf_length_scales"] = np.asarray(
                rbf["length_scales"], dtype=float
            )
            payload[prefix + "rbf_alpha"] = np.asarray(rbf["Alpha"], dtype=float)
            payload[prefix + "rbf_beta"] = np.asarray(rbf["Beta"], dtype=float)
            payload[prefix + "rbf_scale"] = np.asarray(rbf["scale"], dtype=float)
            payload[prefix + "rbf_poly_mode"] = np.array(
                [int(rbf["poly_mode"])], dtype=np.int64
            )
            payload[prefix + "rbf_lambda"] = np.array(
                [float(rbf["lambda_reg"])], dtype=float
            )
            payload[prefix + "rbf_n_centers"] = np.array(
                [int(rbf["n_centers"])], dtype=np.int64
            )

    # Keep classical ECM baseline for traceability and comparisons.
    for k in (
        "Z_res",
        "Z_eps",
        "Z_sig",
        "Z_union",
        "w_res",
        "w_eps",
        "w_sig",
        "w_res_full",
        "w_eps_full",
        "w_sig_full",
        "rel_error_res",
        "rel_error_eps",
        "rel_error_sig",
        "ECM_COUPLING_MODE",
        "ECM_COUPLING_MODE_INPUT",
    ):
        if k in ecm_fixed:
            payload[f"classic_{k}"] = ecm_fixed[k]

    # Preserve HROM mesh metadata only on explicit request; otherwise force rebuild for new supports.
    if bool(int(args.preserve_hrom_metadata)):
        for k in (
            "hrom_mesh_base",
            "hrom_full_mesh_base",
            "hrom_selection_key",
            "hrom_condition_mode",
            "hrom_element_full_indices",
            "hrom_element_ids_0based",
            "hrom_condition_ids_0based",
            "hrom_n_elem",
            "hrom_n_cond",
            "w_res_hrom",
            "w_eps_hrom",
            "w_sig_hrom",
        ):
            if k in ecm_fixed:
                payload[k] = ecm_fixed[k]
    elif "hrom_full_mesh_base" in ecm_fixed:
        payload["hrom_full_mesh_base"] = ecm_fixed["hrom_full_mesh_base"]

    np.savez(out_file, **payload)


def parse_args():
    p = argparse.ArgumentParser(description="Stage 12b: build MAW-ECM model for HPROM-GPR.")
    p.add_argument("--dataset-dir", type=str, default=DATASET_DIR)
    p.add_argument(
        "--classic-ecm-source",
        type=str,
        default="compute",
        choices=["compute", "file"],
        help="Classical ECM source: self-compute from Stage12a dataset, or load from --fixed-ecm-dir.",
    )
    p.add_argument("--fixed-ecm-dir", type=str, default=FIXED_ECM_DIR)
    p.add_argument("--out-dir", type=str, default=OUT_DIR)
    p.add_argument("--targets", type=str, default="res")
    p.add_argument(
        "--hom-mode",
        type=str,
        default="fixed_classic",
        choices=["fixed_classic", "fixed_stage9", "maw"],
        help="Homogenization handling: keep classical split ECM weights fixed (recommended) or train MAW for eps/sig.",
    )
    p.add_argument(
        "--classic-ecm-coupling-mode",
        type=str,
        default="cascade",
        choices=["independent", "cascade"],
        help="Classical split ECM mode used when --classic-ecm-source=compute.",
    )
    p.add_argument("--classic-rsvd-tol-res", type=float, default=1.0e-5)
    p.add_argument("--classic-rsvd-tol-eps", type=float, default=1.0e-5)
    p.add_argument("--classic-rsvd-tol-sig", type=float, default=1.0e-5)
    p.add_argument("--classic-ecm-tol-res", type=float, default=0.0)
    p.add_argument("--classic-ecm-tol-eps", type=float, default=0.0)
    p.add_argument("--classic-ecm-tol-sig", type=float, default=0.0)
    p.add_argument("--classic-max-unsuccessful-it", type=int, default=200)
    p.add_argument(
        "--disable-homogenization",
        type=int,
        default=1,
        choices=[0, 1],
        help="If 1, Stage12b builds residual-only model and skips homogenization ECM/MAW paths.",
    )
    p.add_argument(
        "--res-candidate-pool",
        type=str,
        default="fixed_support",
        choices=["fixed_support", "full_mesh"],
        help="Initial candidate pool for residual MAW.",
    )
    p.add_argument(
        "--res-target-source",
        type=str,
        default="dataset",
        choices=["dataset", "anchor"],
        help="Residual RHS target source for MAW pruning.",
    )
    p.add_argument(
        "--preserve-hrom-metadata",
        type=int,
        default=0,
        choices=[0, 1],
        help="If 1, keep existing HROM mesh metadata from fixed ECM file.",
    )

    p.add_argument(
        "--graph-mode",
        type=str,
        default="structured_stage12",
        choices=["structured_stage12", "knn"],
        help="Graph operator used in MAW pruning.",
    )
    p.add_argument("--graph-knn", type=int, default=8)
    p.add_argument("--graph-kernel", type=str, default="gaussian", choices=["gaussian", "binary"])
    p.add_argument("--graph-sigma", type=float, default=0.0)
    p.add_argument("--alpha-smooth", type=float, default=0.1)
    p.add_argument("--criterion", type=int, default=2, choices=[0, 1, 2])
    p.add_argument("--n-candidates-to-try", type=int, default=20)
    p.add_argument("--incremental-smoothing", type=int, default=1, choices=[0, 1])
    p.add_argument("--use-total-as-criterion", type=int, default=0, choices=[0, 1])
    p.add_argument("--tol-rank-rel", type=float, default=1.0e-12)
    p.add_argument("--tol-neg-factor", type=float, default=10.0)
    p.add_argument("--tol-zero", type=float, default=1.0e-12)
    p.add_argument("--max-as-iters", type=int, default=30)
    p.add_argument("--max-reduced-dim", type=int, default=2500)

    p.add_argument("--rbf-centers-res", type=int, default=0)
    p.add_argument("--rbf-centers-eps", type=int, default=0)
    p.add_argument("--rbf-centers-sig", type=int, default=0)
    p.add_argument("--rbf-poly-mode", type=int, default=1, choices=[0, 1, 2])
    p.add_argument("--rbf-lambda", type=float, default=1.0e-10)
    p.add_argument("--rbf-length-scale-factor", type=float, default=1.0)
    p.add_argument("--rbf-clip-nonnegative", type=int, default=1, choices=[0, 1])
    p.add_argument("--rbf-renorm", type=int, default=1, choices=[0, 1])
    return p.parse_args()


def main():
    args = parse_args()
    targets = _parse_targets(args.targets)
    disable_hom = bool(int(args.disable_homogenization))
    if disable_hom:
        targets = ["res"]
        args.targets = "res"
    if "res" not in targets:
        raise ValueError("Residual target 'res' is required in --targets.")
    hom_mode_raw = str(args.hom_mode).strip().lower()
    hom_mode = "fixed_classic" if hom_mode_raw == "fixed_stage9" else hom_mode_raw
    if disable_hom and hom_mode == "maw":
        raise ValueError("Homogenization is disabled; --hom-mode maw is not allowed.")
    args.hom_mode = hom_mode

    os.makedirs(args.out_dir, exist_ok=True)
    out_file = os.path.join(args.out_dir, "ecm_weights_all.npz")

    classic_source = str(args.classic_ecm_source).strip().lower()
    require_hom_dataset = not disable_hom
    load_hom_dataset = not disable_hom
    dataset = _load_stage12a_dataset(
        str(args.dataset_dir),
        require_hom=require_hom_dataset,
        load_hom=load_hom_dataset,
    )
    if classic_source == "compute":
        ecm_fixed = _compute_classic_ecm_from_dataset(
            dataset, args, include_homogenization=(not disable_hom)
        )
        print("  [Stage12b] Classical ECM source: self-computed from Stage12a dataset.")
    else:
        ecm_fixed = _load_fixed_ecm(str(args.fixed_ecm_dir))
        print(f"  [Stage12b] Classical ECM source: loaded from {args.fixed_ecm_dir}")

    n_elem = int(dataset["n_elem"])
    print("=" * 72)
    print("Stage 12b: MAW-ECM model build")
    print("=" * 72)
    print(f"dataset_dir : {args.dataset_dir}")
    print(f"classic src : {classic_source}")
    print(f"disable hom: {int(disable_hom)}")
    print(f"fixed_ecm   : {args.fixed_ecm_dir}")
    print(f"out_file    : {out_file}")
    print(f"maw targets : {targets}")
    print(f"hom_mode    : {hom_mode}")
    print(f"res pool/rhs: {args.res_candidate_pool} / {args.res_target_source}")
    print(f"graph mode  : {args.graph_mode}")
    print(f"n_elem      : {n_elem}")
    print(f"nq          : {dataset['nq']}")
    if disable_hom:
        print(f"N_s_res     : {dataset['ns_res']}")
    else:
        print(f"N_s_res/hom : {dataset['ns_res']} / {dataset['ns_hom']}")

    results = {}
    results["res"] = _run_target_maw(
        target="res",
        dataset=dataset,
        ecm_fixed=ecm_fixed,
        n_elem=n_elem,
        args=args,
    )

    if disable_hom:
        for t in ("eps", "sig"):
            results[t] = _build_disabled_hom_target_result(dataset, target=t)
        print("  [Stage12b] Homogenization targets are disabled: eps/sig supports forced empty.")
    else:
        for t in ("eps", "sig"):
            if hom_mode == "maw" and t in targets:
                results[t] = _run_target_maw(
                    target=t,
                    dataset=dataset,
                    ecm_fixed=ecm_fixed,
                    n_elem=n_elem,
                    args=args,
                )
            else:
                results[t] = _build_fixed_hom_target_result(dataset, ecm_fixed, target=t)
                print(
                    f"  [MAW-{t}] fixed classical homogenization support kept: |Z|={results[t]['z_red'].size}"
                )

    _save_stage12b_file(
        out_file=out_file,
        dataset=dataset,
        ecm_fixed=ecm_fixed,
        results=results,
        args=args,
    )

    with open(os.path.join(args.out_dir, "reference_measure_A0.txt"), "w", encoding="utf-8") as f:
        f.write(f"{_meta_float(dataset['meta'], 'A0_ref'):.16e}\n")

    print("\n[DONE] Stage 12b MAW-ECM model saved.")
    print(f"  -> {out_file}")


if __name__ == "__main__":
    main()
