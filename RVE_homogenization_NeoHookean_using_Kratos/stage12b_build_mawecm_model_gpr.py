#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage 12b: Build MAW-ECM model (offline pruning + weight regression) for HPROM-GPR.

Default behavior follows the residual-centric MAW setup:
  - Stage12b first computes classical split ECM (self-contained) from Stage12a data.
  - MAW is then trained only for residual support/weights.
  - Homogenization supports/weights are kept classical (no MAW) unless requested.
  - Every local MAW target is anchored at the classical ECM rule: b_j=A_j*w_ini.
  - Phase 1 is unregularized; phase 2 is enabled only through explicit flags.
  - If homogenization MAW is requested, eps/sig are trained independently and
    require positive, sum-preserving weights.
"""

from __future__ import annotations

import os
import argparse
import numpy as np

from mawecm_graph_utils import (
    build_knn_graph_laplacian,
    filter_degenerate_tetrahedra,
)
from mawecm_pruning import run_mawecm_pruning
from mawecm_rbf_weights import fit_mawecm_rbf, eval_mawecm_rbf
from mawecm_ann_weights import fit_mawecm_ann, eval_mawecm_ann
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
    qm_res_file = os.path.join(dataset_dir, "q_m_res.npy")
    qh_file = os.path.join(dataset_dir, "q_m_hom.npy")
    muh_file = os.path.join(dataset_dir, "mu_hom.npy")
    idh_file = os.path.join(dataset_dir, "sample_ids_hom.npy")

    for p in (q_file, b_file, qm_res_file):
        if not os.path.exists(p):
            raise FileNotFoundError(p)

    q_res_mm = np.memmap(q_file, dtype=np.float64, mode="r", shape=(nq * ns_res, n_elem))
    b_res_mm = np.memmap(b_file, dtype=np.float64, mode="r", shape=(nq * ns_res,))
    q_m_res = np.asarray(np.load(qm_res_file), dtype=float)
    mu_res = np.asarray(np.load(os.path.join(dataset_dir, "mu_res.npy")), dtype=float)
    ids_res = np.asarray(np.load(os.path.join(dataset_dir, "sample_ids_res.npy")), dtype=np.int64)
    cells_struct_file = os.path.join(dataset_dir, "parameter_mesh_cells_res.npy")
    grid_struct_file = os.path.join(dataset_dir, "structured_mesh_grid_shape_res.npy")
    cells_struct_res = (
        np.asarray(np.load(cells_struct_file), dtype=np.int64)
        if os.path.exists(cells_struct_file)
        else None
    )
    if cells_struct_res is not None and (
        cells_struct_res.ndim != 2 or cells_struct_res.shape[1] != 4
    ):
        raise RuntimeError(
            "Stage12b requires parameter_mesh_cells_res.npy with tet4 Delaunay "
            f"connectivity; got shape {cells_struct_res.shape}. Rebuild Stage7a and Stage12a."
        )
    grid_struct_res = (
        np.asarray(np.load(grid_struct_file), dtype=np.int64)
        if os.path.exists(grid_struct_file)
        else None
    )

    if q_m_res.shape != (ns_res, nq):
        raise RuntimeError(f"q_m_res shape {q_m_res.shape} != ({ns_res},{nq})")
    q_res = np.asarray(q_res_mm, dtype=float)
    b_res = np.asarray(b_res_mm, dtype=float)
    q_ok_rows = np.isfinite(q_res).all(axis=1)
    b_ok_rows = np.isfinite(b_res)
    row_ok = q_ok_rows & b_ok_rows
    sample_ok = row_ok.reshape(ns_res, nq).all(axis=1)
    sample_ok &= np.isfinite(q_m_res).all(axis=1)
    sample_ok &= np.isfinite(mu_res).all(axis=1)
    n_bad = int(np.sum(~sample_ok))
    if n_bad > 0:
        bad_ids = np.flatnonzero(~sample_ok)[:10]
        raise RuntimeError(
            "Stage12a dataset contains non-finite residual samples. "
            f"Bad samples: {n_bad}/{ns_res}, first ids={bad_ids.tolist()}. "
            "Rebuild Stage12a from the actual-sample Delaunay parameter mesh."
        )

    hom_files = (c_file, bh_file, qh_file, muh_file, idh_file)
    hom_ready = all(os.path.exists(p) for p in hom_files)
    ns_hom = int(ns_hom_meta)
    c_hom = None
    b_hom = None
    q_m_hom = np.zeros((0, nq), dtype=float)
    mu_hom = np.zeros((0, 3), dtype=float)
    ids_hom = np.zeros((0, 2), dtype=np.int64)
    if bool(load_hom) and ns_hom > 0 and hom_ready:
        c_hom = np.memmap(c_file, dtype=np.float64, mode="r", shape=(6 * ns_hom, n_elem))
        b_hom = np.memmap(bh_file, dtype=np.float64, mode="r", shape=(6 * ns_hom,))
        q_m_hom = np.asarray(np.load(qh_file), dtype=float)
        mu_hom = np.asarray(np.load(muh_file), dtype=float)
        ids_hom = np.asarray(np.load(idh_file), dtype=np.int64)
        if q_m_hom.shape != (ns_hom, nq):
            raise RuntimeError(f"q_m_hom shape {q_m_hom.shape} != ({ns_hom},{nq})")
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
        "q_m_res": q_m_res,
        "q_m_hom": q_m_hom,
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


def _run_rsvd_on_transpose(M_T, rsvd_tol, label="", use_randomization=True):
    A = np.ascontiguousarray(M_T)
    rsvd = RandomizedSingularValueDecomposition(
        COMPUTE_U=True,
        COMPUTE_V=False,
        RELATIVE_SVD=True,
        USE_RANDOMIZATION=bool(use_randomization),
    )
    U, s, _, eSVD = rsvd.Calculate(A, truncation_tolerance=float(rsvd_tol))
    if U.size == 0:
        svd_label = "RSVD" if bool(use_randomization) else "SVD"
        raise RuntimeError(f"[{label}] {svd_label} returned empty basis")
    svd_label = "RSVD" if bool(use_randomization) else "SVD"
    print(f"  [Classic-{label}] {svd_label} kept={s.size}, eSVD={float(eSVD):.3e}")
    return np.asarray(U, dtype=float), np.asarray(s, dtype=float), float(eSVD)


def _run_ecm(
    U_basis,
    n_elem,
    ecm_tol,
    init_candidates,
    label="",
    max_unsuccessful_it=200,
    constrain_sum_of_weights=True,
):
    ecm = EmpiricalCubatureMethod(
        ECM_tolerance=float(ecm_tol),
        Filter_tolerance=0.0,
        Plotting=False,
        MaximumNumberUnsuccesfulIterations=int(max_unsuccessful_it),
    )
    ecm.SetUp(
        ResidualsBasis=np.asarray(U_basis, dtype=float),
        InitialCandidatesSet=init_candidates,
        constrain_sum_of_weights=bool(constrain_sum_of_weights),
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
        f"({100.0 * z.size / max(int(n_elem), 1):.1f}% of {int(n_elem)}), "
        f"sum(w)={float(np.sum(w_sel)):.6e}"
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

    use_classic_randomized_svd = bool(int(args.classic_rsvd_randomized))
    u_res, _, e_res = _run_rsvd_on_transpose(
        q_res.T,
        float(args.classic_rsvd_tol_res),
        label="RES",
        use_randomization=use_classic_randomized_svd,
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
            c_eps.T,
            float(args.classic_rsvd_tol_eps),
            label="EPS",
            use_randomization=use_classic_randomized_svd,
        )
        u_sig, _, e_sig = _run_rsvd_on_transpose(
            c_sig.T,
            float(args.classic_rsvd_tol_sig),
            label="SIG",
            use_randomization=use_classic_randomized_svd,
        )

    z_res, w_res, w_res_full = _run_ecm(
        U_basis=u_res,
        n_elem=n_elem,
        ecm_tol=float(args.classic_ecm_tol_res),
        init_candidates=None,
        label="RES",
        max_unsuccessful_it=int(args.classic_max_unsuccessful_it),
        constrain_sum_of_weights=bool(int(args.classic_constrain_sum_weights)),
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
            constrain_sum_of_weights=bool(int(args.classic_constrain_sum_weights)),
        )
        sig_init = np.asarray(z_eps, dtype=np.int64) if coupling == "cascade" else None
        z_sig, w_sig, w_sig_full = _run_ecm(
            U_basis=u_sig,
            n_elem=n_elem,
            ecm_tol=float(args.classic_ecm_tol_sig),
            init_candidates=sig_init,
            label="SIG",
            max_unsuccessful_it=int(args.classic_max_unsuccessful_it),
            constrain_sum_of_weights=bool(int(args.classic_constrain_sum_weights)),
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
        "CLASSIC_RSVD_RANDOMIZED": np.array([int(args.classic_rsvd_randomized)], dtype=np.int64),
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
    q_train = np.asarray(dataset["q_m_res"], dtype=float)
    mu_train = np.asarray(dataset["mu_res"], dtype=float)
    ids = np.asarray(dataset["ids_res"], dtype=np.int64)
    return A_blocks, b_blocks, q_train, mu_train, ids


def _build_blocks_eps(dataset, z_ini, w_ini, rhs_mode="anchor", component_index=None):
    ns = int(dataset["ns_hom"])
    if ns <= 0 or dataset["C_hom"] is None or dataset["b_hom"] is None:
        raise RuntimeError("EPS MAW requested but dataset has no homogenization blocks.")
    C_hom = dataset["C_hom"]
    b_hom = dataset["b_hom"]
    mode = str(rhs_mode).strip().lower()
    A_blocks = []
    b_blocks = []
    for s in range(ns):
        if component_index is None:
            r0, r1 = 6 * s, 6 * s + 3
        else:
            comp = int(component_index)
            if comp < 0 or comp >= 3:
                raise ValueError(f"Invalid eps component_index={component_index}.")
            r0, r1 = 6 * s + comp, 6 * s + comp + 1
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
    q_train = np.asarray(dataset["q_m_hom"], dtype=float)
    mu_train = np.asarray(dataset["mu_hom"], dtype=float)
    ids = np.asarray(dataset["ids_hom"], dtype=np.int64)
    return A_blocks, b_blocks, q_train, mu_train, ids


def _build_blocks_sig(dataset, z_ini, w_ini, rhs_mode="anchor", component_index=None):
    ns = int(dataset["ns_hom"])
    if ns <= 0 or dataset["C_hom"] is None or dataset["b_hom"] is None:
        raise RuntimeError("SIG MAW requested but dataset has no homogenization blocks.")
    C_hom = dataset["C_hom"]
    b_hom = dataset["b_hom"]
    mode = str(rhs_mode).strip().lower()
    A_blocks = []
    b_blocks = []
    for s in range(ns):
        if component_index is None:
            r0, r1 = 6 * s + 3, 6 * s + 6
        else:
            comp = int(component_index)
            if comp < 0 or comp >= 3:
                raise ValueError(f"Invalid sig component_index={component_index}.")
            r0, r1 = 6 * s + 3 + comp, 6 * s + 3 + comp + 1
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
    q_train = np.asarray(dataset["q_m_hom"], dtype=float)
    mu_train = np.asarray(dataset["mu_hom"], dtype=float)
    ids = np.asarray(dataset["ids_hom"], dtype=np.int64)
    return A_blocks, b_blocks, q_train, mu_train, ids


def _augment_blocks_with_sum_constraint(A_blocks, b_blocks, target_sum):
    target = float(target_sum)
    A_aug = []
    b_aug = []
    for A, b in zip(A_blocks, b_blocks):
        Aj = np.asarray(A, dtype=float)
        bj = np.asarray(b, dtype=float).reshape(-1)
        A_aug.append(np.vstack([Aj, np.ones((1, int(Aj.shape[1])), dtype=float)]))
        b_aug.append(np.concatenate([bj, np.array([target], dtype=float)]))
    return A_aug, b_aug


def _scale_hom_blocks_by_component(A_blocks, b_blocks, target):
    """Balance eps/sig component rows without changing their exact feasible set."""
    if not A_blocks:
        return A_blocks, b_blocks, np.ones(0, dtype=float)

    m = int(np.asarray(A_blocks[0]).shape[0])
    if m <= 0:
        return A_blocks, b_blocks, np.ones(0, dtype=float)
    for A, b in zip(A_blocks, b_blocks):
        if int(np.asarray(A).shape[0]) != m or int(np.asarray(b).reshape(-1).size) != m:
            raise RuntimeError(
                f"Cannot component-scale MAW-{target}: variable local row counts before rowspace compression."
            )

    row_norms = np.zeros((len(A_blocks), m), dtype=float)
    rhs_abs = np.zeros((len(A_blocks), m), dtype=float)
    for i, (A, b) in enumerate(zip(A_blocks, b_blocks)):
        Aj = np.asarray(A, dtype=float)
        bj = np.asarray(b, dtype=float).reshape(-1)
        row_norms[i, :] = np.linalg.norm(Aj, axis=1)
        rhs_abs[i, :] = np.abs(bj)

    char = np.maximum(
        np.sqrt(np.mean(row_norms * row_norms, axis=0)),
        np.sqrt(np.mean(rhs_abs * rhs_abs, axis=0)),
    )
    char = np.maximum(char, 1.0e-30)
    scales = 1.0 / char

    A_scaled = []
    b_scaled = []
    for A, b in zip(A_blocks, b_blocks):
        Aj = np.asarray(A, dtype=float)
        bj = np.asarray(b, dtype=float).reshape(-1)
        A_scaled.append(Aj * scales[:, None])
        b_scaled.append(bj * scales)

    print(
        f"  [MAW-{target}] component row scaling: "
        + ", ".join(f"s{i + 1}={scales[i]:.3e}" for i in range(scales.size)),
        flush=True,
    )
    return A_scaled, b_scaled, scales


def _compress_constraints_rowspace(A_blocks, b_blocks, tol_rel, label):
    """Remove linearly dependent local constraint rows without changing the feasible set."""
    tol_rel = float(max(tol_rel, 0.0))
    A_eff = []
    b_eff = []
    ranks = []
    m_full = []
    n_rank_zero = 0

    for A, b in zip(A_blocks, b_blocks):
        Aj = np.asarray(A, dtype=float)
        bj = np.asarray(b, dtype=float).reshape(-1)
        m = int(Aj.shape[0])
        m_full.append(m)
        if m == 0:
            rank = 0
            A_hat = np.zeros((0, int(Aj.shape[1])), dtype=float)
            b_hat = np.zeros((0,), dtype=float)
        else:
            U, s, _ = np.linalg.svd(Aj, full_matrices=False)
            smax = float(np.max(s)) if s.size else 0.0
            tol = tol_rel * max(smax, 1.0)
            rank = int(np.sum(s > tol))
            if rank <= 0:
                A_hat = np.zeros((0, int(Aj.shape[1])), dtype=float)
                b_hat = np.zeros((0,), dtype=float)
            elif rank >= m:
                A_hat = Aj
                b_hat = bj
            else:
                U_r = U[:, :rank]
                A_hat = U_r.T @ Aj
                b_hat = U_r.T @ bj

        if rank == 0:
            n_rank_zero += 1
        ranks.append(rank)
        A_eff.append(np.asarray(A_hat, dtype=float))
        b_eff.append(np.asarray(b_hat, dtype=float))

    ranks_arr = np.asarray(ranks, dtype=np.int64)
    m_full_arr = np.asarray(m_full, dtype=np.int64)
    info = {
        "n_blocks": int(ranks_arr.size),
        "m_full_min": int(np.min(m_full_arr)) if m_full_arr.size else 0,
        "m_full_max": int(np.max(m_full_arr)) if m_full_arr.size else 0,
        "rank_min": int(np.min(ranks_arr)) if ranks_arr.size else 0,
        "rank_mean": float(np.mean(ranks_arr)) if ranks_arr.size else 0.0,
        "rank_max": int(np.max(ranks_arr)) if ranks_arr.size else 0,
        "n_rank_zero": int(n_rank_zero),
        "n_row_reduced": int(np.sum(ranks_arr < m_full_arr)) if ranks_arr.size else 0,
    }
    print(
        f"  [MAW-{label}] rowspace compression: "
        f"rank(min/mean/max)=({info['rank_min']}/{info['rank_mean']:.2f}/{info['rank_max']}), "
        f"reduced_blocks={info['n_row_reduced']}/{info['n_blocks']}, rank0={info['n_rank_zero']}"
    )
    return A_eff, b_eff, info


def _pack_b_blocks_for_save(b_blocks):
    """Pack possibly variable-size local RHS blocks into a rectangular diagnostic array."""
    sizes = np.asarray([np.asarray(b, dtype=float).size for b in b_blocks], dtype=np.int64)
    if sizes.size == 0:
        return np.zeros((0, 0), dtype=float), sizes
    max_size = int(np.max(sizes))
    packed = np.full((int(sizes.size), max_size), np.nan, dtype=float)
    for i, b in enumerate(b_blocks):
        bj = np.asarray(b, dtype=float).reshape(-1)
        packed[i, : bj.size] = bj
    return packed, sizes


def _validate_maw_constraints(
    A_blocks,
    b_blocks,
    W_train,
    strict_rel_tol,
    strict_negative_tol,
):
    W = np.asarray(W_train, dtype=float)
    if W.ndim != 2 or W.shape[1] != len(A_blocks):
        raise RuntimeError(
            f"Invalid MAW weight field shape {W.shape}; expected (*,{len(A_blocks)})."
        )

    rel_errors = np.zeros(len(A_blocks), dtype=float)
    for j, (A, b) in enumerate(zip(A_blocks, b_blocks)):
        Aj = np.asarray(A, dtype=float)
        bj = np.asarray(b, dtype=float).reshape(-1)
        residual = Aj @ W[:, j] - bj
        rel_errors[j] = np.linalg.norm(residual) / max(np.linalg.norm(bj), 1.0e-30)

    max_rel = float(np.max(rel_errors)) if rel_errors.size else 0.0
    min_weight = float(np.min(W)) if W.size else 0.0
    if max_rel > float(strict_rel_tol):
        raise RuntimeError(
            f"MAW local constraints failed: max_rel={max_rel:.3e} > {strict_rel_tol:.3e}."
        )
    if min_weight < -float(strict_negative_tol):
        raise RuntimeError(
            f"MAW non-negativity failed: min_weight={min_weight:.3e} "
            f"< -{strict_negative_tol:.3e}."
        )
    return {"max_rel": max_rel, "min_weight": min_weight}


def _select_prune_graph_subset(coord_train, target, args):
    """Deterministic subset used only by graph-regularized MAW pruning."""
    n = int(np.asarray(coord_train).shape[0])
    size = int(getattr(args, "maw_prune_graph_subsample_size", 0))
    if size <= 0 or size >= n:
        return None
    mode = str(getattr(args, "maw_prune_graph_subsample_mode", "uniform")).strip().lower()
    if mode == "uniform":
        idx = np.unique(np.round(np.linspace(0, n - 1, size)).astype(np.int64))
        if idx.size < size:
            missing = size - int(idx.size)
            pool = np.setdiff1d(np.arange(n, dtype=np.int64), idx, assume_unique=False)
            idx = np.sort(np.concatenate([idx, pool[:missing]]))
    elif mode == "random":
        seed = int(getattr(args, "maw_prune_graph_subsample_seed", 11))
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(n, size=size, replace=False).astype(np.int64))
    else:
        raise ValueError(
            f"Unsupported --maw-prune-graph-subsample-mode='{mode}'. "
            "Use 'uniform' or 'random'."
        )
    print(
        f"  [MAW-{target}] graph-pruning subset: {int(idx.size)}/{n} states "
        f"(mode={mode}); final ANN physics loss still uses the full set.",
        flush=True,
    )
    return idx


def _plot_maw_weight_fields_3d(
    q_train,
    W_train,
    z_red,
    rbf_model,
    out_dir,
    target,
    max_plots,
    fmt,
    clip_nonnegative,
    renorm_target,
    coord_label="q",
    regressor_type="rbf",
):
    """Plot w(x1,x2;x3) as three surfaces at representative fixed coordinate-3 values."""
    q = np.asarray(q_train, dtype=float)
    w = np.asarray(W_train, dtype=float)
    z = np.asarray(z_red, dtype=np.int64).reshape(-1)
    tag = str(target).strip().lower()
    coord_label = str(coord_label).strip().lower() or "q"
    regressor_type = str(regressor_type).strip().lower()

    if q.ndim != 2 or q.shape[1] < 3:
        print(
            f"  [MAW-plots][{tag}][WARN] q_train shape {q.shape} is not a 3D state domain."
        )
        return 0
    if w.ndim != 2 or w.shape != (z.size, q.shape[0]):
        print(
            f"  [MAW-plots][{tag}][WARN] incompatible W_train/q/support shapes: "
            f"{w.shape}, {q.shape}, {z.shape}."
        )
        return 0

    save_dir = os.path.join(out_dir, f"maw_weight_fields_{tag}")
    os.makedirs(save_dir, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(save_dir, ".mplconfig"))
    os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams["text.usetex"] = False
    mpl.rcParams["font.size"] = 10
    mpl.rcParams["axes.titlesize"] = 10
    mpl.rcParams["axes.labelsize"] = 10
    mpl.rcParams["xtick.labelsize"] = 9
    mpl.rcParams["ytick.labelsize"] = 9

    q_dim = int(q.shape[1])
    n_q1, n_q2 = 70, 55
    q1_vec = np.linspace(float(np.min(q[:, 0])), float(np.max(q[:, 0])), n_q1)
    q2_vec = np.linspace(float(np.min(q[:, 1])), float(np.max(q[:, 1])), n_q2)
    q1_grid, q2_grid = np.meshgrid(q1_vec, q2_vec, indexing="xy")
    q3_values = np.quantile(q[:, 2], [0.10, 0.50, 0.90])
    axis_labels = {
        "q": (r"$q^m_1$", r"$q^m_2$", r"$q^m_3$"),
        "mu": (r"$\mu_1$", r"$\mu_2$", r"$\mu_3$"),
    }.get(coord_label, (r"$x_1$", r"$x_2$", r"$x_3$"))

    # Evaluate in bounded-memory batches because a full Stage12 RBF may contain
    # thousands of centers.
    slice_weights = []
    batch_size = 256
    for q3 in q3_values:
        query = np.zeros((q1_grid.size, q_dim), dtype=float)
        query[:, 0] = q1_grid.ravel()
        query[:, 1] = q2_grid.ravel()
        query[:, 2] = float(q3)
        if q_dim > 3:
            query[:, 3:] = np.median(q[:, 3:], axis=0)[None, :]
        parts = []
        for i0 in range(0, query.shape[0], batch_size):
            q_batch = query[i0 : i0 + batch_size]
            if regressor_type == "ann":
                parts.append(eval_mawecm_ann(q_batch, rbf_model))
            else:
                parts.append(
                    eval_mawecm_rbf(
                        q_query=q_batch,
                        model=rbf_model,
                        clip_nonnegative=bool(clip_nonnegative),
                        renorm_target=renorm_target,
                    )
                )
        slice_weights.append(np.hstack(parts))

    n_all = int(z.size)
    n_do = n_all if int(max_plots) <= 0 else min(n_all, int(max_plots))
    summary_rows = []
    q3_band = max(float(np.ptp(q[:, 2])) / 20.0, 1.0e-12)
    print(
        f"  [MAW-plots][{tag}] plotting {n_do}/{n_all} weight surfaces...",
        flush=True,
    )

    for k in range(n_do):
        if k == 0 or (k + 1) == n_do or (k + 1) % 10 == 0:
            print(
                f"  [MAW-plots][{tag}] plot {k + 1}/{n_do}",
                flush=True,
            )
        fields = [
            np.asarray(weights[k], dtype=float).reshape(n_q2, n_q1)
            for weights in slice_weights
        ]
        value_min = min(float(np.min(field)) for field in fields)
        value_max = max(float(np.max(field)) for field in fields)
        value_mean = float(np.mean([np.mean(field) for field in fields]))
        rel_range = (value_max - value_min) / max(abs(value_mean), 1.0e-30)
        z_pad = 0.03 * max(value_max - value_min, abs(value_mean), 1.0e-12)
        curvature = []
        for field in fields:
            d2_q1 = np.gradient(np.gradient(field, axis=1), axis=1)
            d2_q2 = np.gradient(np.gradient(field, axis=0), axis=0)
            curvature.append(
                float(
                    np.sqrt(np.mean((d2_q1 + d2_q2) ** 2))
                    / max(abs(value_mean), value_max - value_min, 1.0e-30)
                )
            )

        fig, axes = plt.subplots(
            1,
            3,
            figsize=(15.8, 5.0),
            constrained_layout=True,
            subplot_kw={"projection": "3d"},
        )
        for j, (ax, field, q3) in enumerate(zip(axes, fields, q3_values)):
            ax.plot_surface(
                q1_grid,
                q2_grid,
                field,
                color="#2f7f9f",
                alpha=0.90,
                linewidth=0.0,
                antialiased=True,
                rcount=n_q2,
                ccount=n_q1,
            )
            near = np.abs(q[:, 2] - q3) <= q3_band
            if np.any(near):
                ax.scatter(
                    q[near, 0],
                    q[near, 1],
                    w[k, near],
                    c="black",
                    s=7,
                    alpha=0.35,
                    linewidths=0.0,
                    depthshade=False,
                )
            ax.set_xlabel(axis_labels[0])
            ax.set_ylabel(axis_labels[1])
            ax.set_zlabel(r"$w_e$")
            ax.set_zlim(value_min - z_pad, value_max + z_pad)
            ax.view_init(elev=28.0, azim=-58.0)
            ax.set_title(
                rf"{axis_labels[2]}={q3:.3e} | curvature={curvature[j]:.2e}",
                fontsize=10,
            )
        elem_id = int(z[k])
        fig.suptitle(
            f"MAW-{tag} weight surfaces, element index {elem_id} | "
            f"relative range={rel_range:.2e}",
            fontsize=12,
        )
        out_file = os.path.join(
            save_dir,
            f"weight_surfaces_elem_{elem_id:05d}.{str(fmt).strip('.')}",
        )
        fig.savefig(out_file, dpi=150, bbox_inches="tight")
        plt.close(fig)
        summary_rows.append(
            [elem_id, value_min, value_max, value_mean, rel_range, *curvature]
        )

    summary_file = os.path.join(save_dir, "weight_surface_smoothness.csv")
    header = (
        "element_index,min,max,mean,relative_range,"
        "curvature_q3_p10,curvature_q3_p50,curvature_q3_p90"
    )
    np.savetxt(
        summary_file,
        np.asarray(summary_rows, dtype=float),
        delimiter=",",
        header=header,
        comments="",
    )
    print(
        f"  [MAW-plots][{tag}] saved {n_do} three-surface weight-field plots in "
        f"{save_dir}"
    )
    return n_do


def _build_graph_for_target(dataset, q_train, target, args):
    mode = str(args.graph_mode).strip().lower()
    if mode == "parameter_mesh_stage12":
        cells = dataset.get("cells_struct_res", None)
        if cells is None:
            raise RuntimeError(
                "parameter_mesh_cells_res.npy not found in dataset. "
                "Rebuild Stage12a from the Stage7 Delaunay parameter mesh."
            )
        target_key = str(target).strip().lower()
        if target_key == "res":
            mu_train = np.asarray(dataset["mu_res"], dtype=float)
            q_label = "q_m_res"
        elif target_key in ("eps", "sig"):
            mu_train = np.asarray(dataset["mu_hom"], dtype=float)
            q_label = "q_m_hom"
        else:
            raise ValueError(f"Unsupported MAW graph target '{target}'.")
        if mu_train.shape != (int(q_train.shape[0]), 3):
            raise RuntimeError(
                f"Stage12 parameter graph requires mu aligned with {q_label}; "
                f"got mu={mu_train.shape}, {q_label}={q_train.shape} for target={target_key}."
            )
        cells = np.asarray(cells, dtype=np.int64)
        if cells.size and (int(np.min(cells)) < 0 or int(np.max(cells)) >= int(q_train.shape[0])):
            raise RuntimeError(
                "Stage12 parameter graph cells are incompatible with the selected target samples. "
                f"target={target_key}, n_samples={int(q_train.shape[0])}, "
                f"cell id range=[{int(np.min(cells))},{int(np.max(cells))}]. "
                "Rebuild Stage12a with homogenization enabled from the same Stage7 parameter mesh."
            )
        _, laplacian, graph_stats = filter_degenerate_tetrahedra(
            points=mu_train,
            cells=cells,
            normalized_det_tol=1.0e-14,
            require_all_nodes=True,
            require_connected=True,
        )
        print(
            f"  [MAW-graph][{target_key}] Stage12 parameter tetrahedra: "
            f"kept={graph_stats['n_cells_kept']}/"
            f"{graph_stats['n_cells_total']}, "
            f"removed_degenerate={graph_stats['n_cells_removed']}, "
            f"edges={graph_stats['n_graph_edges']}, "
            f"components={graph_stats['n_components']}, "
            f"isolated={graph_stats['n_isolated_nodes']} "
            f"(normalized |det J_mu| > "
            f"{graph_stats['normalized_det_tol']:.1e})."
        )
        return laplacian
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


def _hom_component_classic_keys(target, component_index):
    t = str(target).strip().lower()
    if component_index is None:
        return None
    comp = int(component_index)
    if t not in ("eps", "sig") or comp not in (0, 1, 2):
        return None
    suffix = "xy" if comp == 2 else "norm"
    return (
        f"Z_{t}_{suffix}",
        f"w_{t}_{suffix}",
        f"w_{t}_{suffix}_full",
        suffix,
    )


def _load_hom_component_classic_rule(ecm_fixed, target, component_index, n_elem):
    keys = _hom_component_classic_keys(target, component_index)
    if keys is None:
        return None
    z_key, w_key, w_full_key, suffix = keys
    if z_key not in ecm_fixed or w_key not in ecm_fixed:
        return None

    z = np.asarray(ecm_fixed[z_key], dtype=np.int64).reshape(-1)
    w = np.asarray(ecm_fixed[w_key], dtype=float).reshape(-1)
    if z.size != w.size:
        raise RuntimeError(
            f"Fixed component ECM size mismatch for {target}_{component_index} "
            f"({suffix}): |Z|={z.size}, |w|={w.size}."
        )
    if z.size == 0:
        return None
    if np.any(z < 0) or np.any(z >= int(n_elem)):
        raise RuntimeError(
            f"Fixed component ECM support {z_key} has indices outside [0,{int(n_elem)})."
        )

    w_full = np.zeros(int(n_elem), dtype=float)
    if w_full_key in ecm_fixed:
        wf = np.asarray(ecm_fixed[w_full_key], dtype=float).reshape(-1)
        if wf.size != int(n_elem):
            raise RuntimeError(
                f"{w_full_key} size {wf.size} != n_elem {int(n_elem)}."
            )
        w_full[:] = wf
    else:
        w_full[z] = w

    return {
        "z": z,
        "w": w,
        "w_full": w_full,
        "z_key": z_key,
        "w_key": w_key,
        "w_full_key": w_full_key,
        "suffix": suffix,
    }


def _select_initial_support_and_weights(
    ecm_fixed,
    target,
    n_elem,
    candidate_pool="fixed_support",
    component_index=None,
):
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
    elif t in ("eps", "sig"):
        component_rule = _load_hom_component_classic_rule(
            ecm_fixed=ecm_fixed,
            target=t,
            component_index=component_index,
            n_elem=n_elem,
        )
        if component_rule is not None:
            z = np.asarray(component_rule["z"], dtype=np.int64).reshape(-1)
            w = np.asarray(component_rule["w"], dtype=float).reshape(-1)
            print(
                f"  [MAW-{t}_{component_index}] using classical component "
                f"bootstrap '{component_rule['suffix']}' from "
                f"{component_rule['z_key']}: |Z|={z.size}",
                flush=True,
            )
        else:
            z = np.asarray(ecm_fixed[f"Z_{t}"], dtype=np.int64).reshape(-1)
            w = np.asarray(ecm_fixed[f"w_{t}"], dtype=float).reshape(-1)
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
    label=None,
    hom_component_index=None,
    min_support_override=None,
    phase1_stop_size_override=None,
):
    tag = str(label or target).strip().lower()
    if target == "res":
        z_ini, w_ini = _select_initial_support_and_weights(
            ecm_fixed,
            target,
            n_elem=n_elem,
            candidate_pool=str(args.res_candidate_pool),
        )
        rhs_mode = "anchor"
    else:
        z_ini, w_ini = _select_initial_support_and_weights(
            ecm_fixed,
            target,
            n_elem=n_elem,
            candidate_pool="fixed_support",
            component_index=hom_component_index,
        )
        rhs_mode = "anchor"

    if target == "res":
        A_blocks, b_blocks, q_train, mu_train, ids = _build_blocks_res(
            dataset, z_ini, w_ini, rhs_mode=rhs_mode
        )
        n_centers = int(args.rbf_centers_res)
    elif target == "eps":
        A_blocks, b_blocks, q_train, mu_train, ids = _build_blocks_eps(
            dataset,
            z_ini,
            w_ini,
            rhs_mode=rhs_mode,
            component_index=hom_component_index,
        )
        n_centers = int(args.rbf_centers_eps)
    elif target == "sig":
        A_blocks, b_blocks, q_train, mu_train, ids = _build_blocks_sig(
            dataset,
            z_ini,
            w_ini,
            rhs_mode=rhs_mode,
            component_index=hom_component_index,
        )
        n_centers = int(args.rbf_centers_sig)
    else:
        raise ValueError(f"Unsupported MAW target '{target}'.")

    component_scales = np.ones(
        int(np.asarray(A_blocks[0]).shape[0]) if A_blocks else 0,
        dtype=float,
    )
    if target in ("eps", "sig"):
        A_blocks, b_blocks, component_scales = _scale_hom_blocks_by_component(
            A_blocks=A_blocks,
            b_blocks=b_blocks,
            target=tag,
        )

    sum_target = float(args.sum_weights_target)
    if sum_target <= 0.0:
        sum_target = float(np.sum(w_ini))
    if bool(int(args.enforce_sum_weights)):
        A_blocks, b_blocks = _augment_blocks_with_sum_constraint(
            A_blocks=A_blocks,
            b_blocks=b_blocks,
            target_sum=sum_target,
        )
        print(f"  [MAW-{tag}] enforcing sum(w)={sum_target:.6e} in each local block.")

    rowspace_info = {
        "n_blocks": len(A_blocks),
        "m_full_min": int(min(A.shape[0] for A in A_blocks)) if A_blocks else 0,
        "m_full_max": int(max(A.shape[0] for A in A_blocks)) if A_blocks else 0,
        "rank_min": np.nan,
        "rank_mean": np.nan,
        "rank_max": np.nan,
        "n_rank_zero": 0,
        "n_row_reduced": 0,
    }
    if target in ("eps", "sig") and bool(int(args.maw_hom_rowspace_compress)):
        A_blocks, b_blocks, rowspace_info = _compress_constraints_rowspace(
            A_blocks=A_blocks,
            b_blocks=b_blocks,
            tol_rel=float(args.maw_hom_rowspace_tol_rel),
            label=tag,
        )

    coord_label = "mu" if target in ("eps", "sig") else "q"
    coord_train = np.asarray(mu_train if coord_label == "mu" else q_train, dtype=float)
    print(
        f"  [MAW-{tag}] weight-field coordinate: {coord_label} "
        f"(shape={coord_train.shape})",
        flush=True,
    )

    use_global_graph = bool(int(args.use_global_graph_2ndstage))
    prune_idx = None
    A_blocks_prune = A_blocks
    b_blocks_prune = b_blocks
    coord_train_prune = coord_train
    if use_global_graph:
        prune_idx = _select_prune_graph_subset(coord_train, tag, args)
        if prune_idx is not None:
            if str(args.graph_mode).strip().lower() != "knn":
                raise RuntimeError(
                    "Graph-pruning subsampling is currently supported only with "
                    "--graph-mode knn."
                )
            A_blocks_prune = [A_blocks[int(i)] for i in prune_idx]
            b_blocks_prune = [b_blocks[int(i)] for i in prune_idx]
            coord_train_prune = coord_train[prune_idx, :]

    K_graph = None
    if use_global_graph:
        K_graph = _build_graph_for_target(
            dataset=dataset,
            q_train=coord_train_prune,
            target=target,
            args=args,
        )

    min_support = {
        "res": int(args.maw_min_support_size_res),
        "eps": int(args.maw_min_support_size_eps),
        "sig": int(args.maw_min_support_size_sig),
    }[target]
    if min_support_override is not None:
        min_support = int(min_support_override)
    phase1_stop_size = {
        "res": int(args.maw_phase1_stop_size_res),
        "eps": int(args.maw_phase1_stop_size_eps),
        "sig": int(args.maw_phase1_stop_size_sig),
    }[target]
    if phase1_stop_size_override is not None:
        phase1_stop_size = int(phase1_stop_size_override)

    prune_opts = {
        "alpha_smooth": float(args.alpha_smooth) if use_global_graph else 0.0,
        "criterion": int(args.criterion),
        "number_of_candidates_to_try": int(args.n_candidates_to_try) if int(args.n_candidates_to_try) > 0 else None,
        "incremental_smoothing": bool(int(args.incremental_smoothing)),
        "use_total_as_criterion": bool(int(args.use_total_as_criterion)),
        "tol_rank_rel": float(args.tol_rank_rel),
        "tol_neg_factor": float(args.tol_neg_factor),
        "tol_zero": float(args.tol_zero),
        "max_active_set_iters": int(args.max_as_iters),
        "max_reduced_dim": int(args.max_reduced_dim),
        "graph_cg_maxiter": int(args.maw_graph_cg_maxiter),
        "graph_cg_rtol": float(args.maw_graph_cg_rtol),
        "warn_max_reduced_dim": False,
        "enforce_nonnegativity": True,
        "use_global_graph_2ndstage": use_global_graph,
        "smooth_laplacian_all_iterations": bool(int(args.smooth_laplacian_all_iterations)),
        "max_number_zeros_active_set_loop": int(args.max_number_zeros_active_set_loop_maw_ecm),
        "verbose": True,
    }
    if K_graph is not None:
        prune_opts["K_graph"] = K_graph
    if min_support > 0:
        prune_opts["n_stop"] = min_support
    if phase1_stop_size > 0:
        prune_opts["phase1_stop_size"] = phase1_stop_size

    W_ini = np.tile(np.asarray(w_ini, dtype=float)[:, None], (1, len(A_blocks)))
    validation_ini = _validate_maw_constraints(
        A_blocks=A_blocks,
        b_blocks=b_blocks,
        W_train=W_ini,
        strict_rel_tol=max(float(args.strict_constraint_rel_tol), 1.0e-14),
        strict_negative_tol=float(args.strict_negative_tol),
    )
    print(
        f"  [MAW-{tag}] init feasibility: "
        f"max_rel={validation_ini['max_rel']:.3e}, "
        f"min_w={validation_ini['min_weight']:.3e}"
    )

    maw = run_mawecm_pruning(
        A_blocks=A_blocks_prune,
        b_blocks=b_blocks_prune,
        z_ini=z_ini,
        w_ini=w_ini,
        q_train=coord_train_prune,
        options=prune_opts,
    )

    W_train = np.asarray(maw["W_support"], dtype=float)
    z_red = np.asarray(maw["Z_support"], dtype=np.int64)
    if z_red.size == 0:
        raise RuntimeError(f"MAW returned empty support for target={target}.")
    support_local = np.asarray(maw["i_support_local"], dtype=np.int64)
    A_blocks_red = [np.asarray(A[:, support_local], dtype=float) for A in A_blocks]
    A_blocks_red_prune = [
        np.asarray(A[:, support_local], dtype=float) for A in A_blocks_prune
    ]
    validation = _validate_maw_constraints(
        A_blocks=A_blocks_red_prune,
        b_blocks=b_blocks_prune,
        W_train=W_train,
        strict_rel_tol=float(args.strict_constraint_rel_tol),
        strict_negative_tol=float(args.strict_negative_tol),
    )

    renorm_target = float(np.sum(w_ini))
    regressor_type = "rbf"
    if target in ("eps", "sig"):
        regressor_type = str(args.maw_hom_weight_regressor).strip().lower()
    rbf = None
    ann = None
    weight_model = None

    if regressor_type == "ann":
        print(
            f"  [MAW-{tag}] fitting ANN weights: "
            f"samples={int(coord_train_prune.shape[0])}, support={int(z_red.size)}, "
            f"coord={coord_label}, hidden={args.maw_ann_hidden_dims}, "
            f"epochs={int(args.maw_ann_epochs)}, "
            f"physics_states={int(coord_train.shape[0])}",
            flush=True,
        )
        ann = fit_mawecm_ann(
            q_train=coord_train_prune,
            W_train=W_train,
            target_sum=renorm_target,
            constraint_A_blocks=A_blocks_red_prune,
            constraint_b_blocks=b_blocks_prune,
            physics_q_train=coord_train,
            physics_constraint_A_blocks=A_blocks_red,
            physics_constraint_b_blocks=b_blocks,
            hidden_dims=str(args.maw_ann_hidden_dims),
            activation=str(args.maw_ann_activation),
            epochs=int(args.maw_ann_epochs),
            batch_size=int(args.maw_ann_batch_size),
            lr=float(args.maw_ann_lr),
            weight_decay=float(args.maw_ann_weight_decay),
            val_fraction=float(args.maw_ann_val_fraction),
            patience=int(args.maw_ann_patience),
            lr_scheduler=bool(int(args.maw_ann_lr_scheduler)),
            lr_scheduler_factor=float(args.maw_ann_lr_scheduler_factor),
            lr_scheduler_patience=int(args.maw_ann_lr_scheduler_patience),
            min_lr=float(args.maw_ann_min_lr),
            seed=int(args.maw_ann_seed),
            mse_weight=float(args.maw_ann_mse_weight),
            physics_weight=float(args.maw_ann_physics_weight),
            verbose=True,
            label=f"MAW-{tag}-ANN",
        )
        weight_model = ann
        print(
            f"  [MAW-{tag}] ANN fit done: "
            f"train_rel={float(ann.get('train_rel_error', np.nan)):.3e}, "
            f"val_rel={float(ann.get('val_rel_error', np.nan)):.3e}, "
            f"constraint_rel={float(ann.get('train_constraint_rel_error', np.nan)):.3e}",
            flush=True,
        )
        print(f"  [MAW-{tag}] validating fitted ANN on training states...", flush=True)
        W_recon = eval_mawecm_ann(coord_train_prune, ann)
    else:
        effective_centers = int(coord_train_prune.shape[0]) if int(n_centers) <= 0 else min(int(n_centers), int(coord_train_prune.shape[0]))
        print(
            f"  [MAW-{tag}] fitting RBF weights: "
            f"samples={int(coord_train_prune.shape[0])}, support={int(z_red.size)}, "
            f"coord={coord_label}, "
            f"centers={effective_centers}, poly_mode={int(args.rbf_poly_mode)}, "
            f"lambda={float(args.rbf_lambda):.1e}",
            flush=True,
        )
        rbf = fit_mawecm_rbf(
            q_train=coord_train_prune,
            W_train=W_train,
            n_centers=n_centers,
            poly_mode=int(args.rbf_poly_mode),
            lambda_reg=float(args.rbf_lambda),
            length_scale_factor=float(args.rbf_length_scale_factor),
            verbose=True,
            label=f"MAW-{tag}-RBF",
        )
        weight_model = rbf
        print(
            f"  [MAW-{tag}] RBF fit done: "
            f"train_rel={float(rbf.get('train_rel_error', np.nan)):.3e}",
            flush=True,
        )

        print(f"  [MAW-{tag}] validating fitted RBF on training states...", flush=True)
        W_recon = eval_mawecm_rbf(
            q_query=coord_train_prune,
            model=rbf,
            clip_nonnegative=bool(int(args.rbf_clip_nonnegative)),
            renorm_target=renorm_target if bool(int(args.rbf_renorm)) else None,
        )
    rel_recon = float(np.linalg.norm(W_recon - W_train) / max(np.linalg.norm(W_train), 1.0e-30))

    # Fixed anchor vectors for compatibility with existing Stage-9/10-style consumers.
    w_full = np.zeros(int(n_elem), dtype=float)
    w_anchor = np.mean(W_train, axis=1)
    w_full[z_red] = w_anchor

    b_train_packed, m_constraints_per_block = _pack_b_blocks_for_save(b_blocks_prune)

    print(
        f"  [MAW-{tag}] rhs={rhs_mode} |Z_ini|={z_ini.size} -> |Z_red|={z_red.size}, "
        f"{regressor_type.upper()} train-rel={rel_recon:.3e}, max_rel={validation['max_rel']:.3e}, "
        f"min_w={validation['min_weight']:.3e}, prune-elapsed={maw['elapsed_sec']:.2f}s"
    )

    return {
        "target": tag,
        "base_target": target,
        "hom_component_index": -1 if hom_component_index is None else int(hom_component_index),
        "z_ini": z_ini,
        "w_ini": w_ini,
        "q_train": q_train,
        "mu_train": mu_train,
        "coord_train": coord_train_prune,
        "coord_train_full": coord_train,
        "prune_subset_indices": (
            np.asarray(prune_idx, dtype=np.int64)
            if prune_idx is not None
            else np.arange(coord_train.shape[0], dtype=np.int64)
        ),
        "coord_label": coord_label,
        "component_scales": component_scales,
        "ids": ids,
        "A_m": int(np.max(m_constraints_per_block)) if m_constraints_per_block.size else 0,
        "m_constraints_per_block": m_constraints_per_block,
        "b_train": b_train_packed,
        "maw": maw,
        "z_red": z_red,
        "W_train": W_train,
        "regressor_type": regressor_type,
        "weight_model": weight_model,
        "rbf": rbf,
        "ann": ann,
        "w_full": w_full,
        "w_sel": w_full[z_red],
        "renorm_target": float(renorm_target),
        "recon_rel": rel_recon,
        "rhs_mode": np.array([rhs_mode]),
        "sum_target": float(sum_target),
        "sum_constraint": int(bool(int(args.enforce_sum_weights))),
        "rowspace_info": dict(rowspace_info),
    }


def _build_fixed_hom_target_result(dataset, ecm_fixed, target):
    t = str(target).strip().lower()
    if t == "eps":
        z = np.asarray(ecm_fixed["Z_eps"], dtype=np.int64).reshape(-1)
        w = np.asarray(ecm_fixed["w_eps"], dtype=float).reshape(-1)
        q_train = np.asarray(dataset["q_m_hom"], dtype=float)
        mu_train = np.asarray(dataset["mu_hom"], dtype=float)
        ids = np.asarray(dataset["ids_hom"], dtype=np.int64)
    elif t == "sig":
        z = np.asarray(ecm_fixed["Z_sig"], dtype=np.int64).reshape(-1)
        w = np.asarray(ecm_fixed["w_sig"], dtype=float).reshape(-1)
        q_train = np.asarray(dataset["q_m_hom"], dtype=float)
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


def _build_fixed_hom_component_result(dataset, ecm_fixed, target, component_index):
    """Build one component-wise homogenization record with constant classical ECM weights."""
    t = str(target).strip().lower()
    if t not in ("eps", "sig"):
        raise ValueError(f"Unsupported fixed hom component target '{target}'.")
    comp = int(component_index)
    if comp < 0 or comp > 2:
        raise ValueError(f"Invalid {t} component index {component_index}; expected 0, 1 or 2.")

    q_train = np.asarray(dataset["q_m_hom"], dtype=float)
    mu_train = np.asarray(dataset["mu_hom"], dtype=float)
    ids = np.asarray(dataset["ids_hom"], dtype=np.int64)
    n_elem = int(dataset["n_elem"])

    component_rule = _load_hom_component_classic_rule(
        ecm_fixed=ecm_fixed,
        target=t,
        component_index=comp,
        n_elem=n_elem,
    )
    if component_rule is not None:
        z = np.asarray(component_rule["z"], dtype=np.int64).reshape(-1)
        w = np.asarray(component_rule["w"], dtype=float).reshape(-1)
        w_full = np.asarray(component_rule["w_full"], dtype=float).reshape(-1)
        rhs_label = f"fixed_classic_component_{component_rule['suffix']}"
    else:
        base = _build_fixed_hom_target_result(dataset, ecm_fixed, target=t)
        z = np.asarray(base["z_red"], dtype=np.int64).reshape(-1)
        w = np.asarray(base["w_sel"], dtype=float).reshape(-1)
        w_full = np.zeros(n_elem, dtype=float)
        w_full[z] = w
        rhs_label = "fixed_classic_component"

    return {
        "target": f"{t}_{comp}",
        "base_target": t,
        "hom_component_index": comp,
        "z_ini": z.copy(),
        "w_ini": w.copy(),
        "q_train": q_train,
        "mu_train": mu_train,
        "coord_train": mu_train,
        "coord_label": "mu",
        "component_scales": np.ones(0, dtype=float),
        "ids": ids,
        "A_m": int(1),
        "m_constraints_per_block": np.ones(q_train.shape[0], dtype=np.int64),
        "b_train": np.zeros((q_train.shape[0], 1), dtype=float),
        "maw": None,
        "z_red": z.copy(),
        "W_train": np.tile(w[:, None], (1, q_train.shape[0])),
        "regressor_type": "fixed_classic",
        "weight_model": None,
        "rbf": None,
        "ann": None,
        "w_full": w_full,
        "w_sel": w.copy(),
        "renorm_target": float(np.sum(w)),
        "recon_rel": 0.0,
        "rhs_mode": np.array([rhs_label]),
        "fixed_classic": True,
        "sum_target": float(np.sum(w)),
        "sum_constraint": int(1),
        "rowspace_info": {},
    }


def _build_componentwise_hom_target_result(dataset, target, component_results):
    t = str(target).strip().lower()
    if t not in ("eps", "sig"):
        raise ValueError(f"Unsupported component-wise hom target '{target}'.")
    comps = list(component_results)
    if len(comps) != 3:
        raise RuntimeError(f"Component-wise {t} requires exactly 3 component records.")

    n_elem = int(dataset["n_elem"])
    z_union = np.zeros(0, dtype=np.int64)
    z_ini_union = np.zeros(0, dtype=np.int64)
    w_full_stack = []
    for rec in comps:
        z_union = np.union1d(z_union, np.asarray(rec["z_red"], dtype=np.int64).reshape(-1))
        z_ini_union = np.union1d(z_ini_union, np.asarray(rec["z_ini"], dtype=np.int64).reshape(-1))
        w_full_stack.append(np.asarray(rec["w_full"], dtype=float).reshape(-1))
    z_union = z_union.astype(np.int64)
    z_ini_union = z_ini_union.astype(np.int64)
    w_full = np.mean(np.vstack(w_full_stack), axis=0) if w_full_stack else np.zeros(n_elem, dtype=float)

    q_train = np.asarray(dataset["q_m_hom"], dtype=float)
    mu_train = np.asarray(dataset["mu_hom"], dtype=float)
    ids = np.asarray(dataset["ids_hom"], dtype=np.int64)
    return {
        "target": t,
        "z_ini": z_ini_union.copy(),
        "w_ini": w_full[z_ini_union].copy(),
        "q_train": q_train,
        "mu_train": mu_train,
        "ids": ids,
        "A_m": int(3),
        "m_constraints_per_block": np.full(q_train.shape[0], 3, dtype=np.int64),
        "b_train": np.zeros((q_train.shape[0], 3), dtype=float),
        "maw": None,
        "z_red": z_union.copy(),
        "W_train": np.tile(w_full[z_union][:, None], (1, q_train.shape[0])),
        "regressor_type": "componentwise",
        "weight_model": None,
        "rbf": None,
        "ann": None,
        "w_full": w_full,
        "w_sel": w_full[z_union],
        "renorm_target": float(np.sum(w_full)),
        "recon_rel": 0.0,
        "rhs_mode": np.array(["componentwise_maw"]),
        "fixed_classic": False,
        "componentwise": True,
        "component_records": comps,
        "sum_target": float(np.mean([float(rec.get("sum_target", np.sum(rec["w_full"]))) for rec in comps])),
        "sum_constraint": int(1),
        "rowspace_info": {},
    }


def _build_fixed_res_target_result(dataset, ecm_fixed):
    z = np.asarray(ecm_fixed["Z_res"], dtype=np.int64).reshape(-1)
    w = np.asarray(ecm_fixed["w_res"], dtype=float).reshape(-1)
    if z.size != w.size:
        raise RuntimeError(f"Fixed ECM size mismatch for res: |Z|={z.size}, |w|={w.size}")
    n_elem = int(dataset["n_elem"])
    w_full = np.zeros(n_elem, dtype=float)
    if "w_res_full" in ecm_fixed:
        wf = np.asarray(ecm_fixed["w_res_full"], dtype=float).reshape(-1)
        if wf.size == n_elem:
            w_full[:] = wf
        else:
            w_full[z] = w
    else:
        w_full[z] = w
    q_train = np.asarray(dataset["q_m_res"], dtype=float)
    mu_train = np.asarray(dataset["mu_res"], dtype=float)
    ids = np.asarray(dataset["ids_res"], dtype=np.int64)
    return {
        "target": "res",
        "z_ini": z.copy(),
        "w_ini": w.copy(),
        "q_train": q_train,
        "mu_train": mu_train,
        "ids": ids,
        "A_m": int(dataset["nq"]),
        "b_train": np.zeros((q_train.shape[0], int(dataset["nq"])), dtype=float),
        "maw": None,
        "z_red": z.copy(),
        "W_train": np.tile(w[:, None], (1, q_train.shape[0])),
        "rbf": None,
        "w_full": w_full,
        "w_sel": w_full[z],
        "renorm_target": float(np.sum(w)),
        "recon_rel": 0.0,
        "rhs_mode": np.array(["fixed_classic"]),
        "fixed_classic": True,
        "sum_target": float(np.sum(w)),
        "sum_constraint": int(0),
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


def _parse_component_sizes(text, fallback, name):
    raw = str(text).strip()
    if raw == "":
        return [int(fallback)] * 3
    vals = [v.strip() for v in raw.split(",") if v.strip()]
    if len(vals) != 3:
        raise ValueError(f"{name} must be empty or three comma-separated integers, got '{text}'.")
    out = [int(v) for v in vals]
    if any(v < 0 for v in out):
        raise ValueError(f"{name} entries must be non-negative, got {out}.")
    return out


def _parse_component_flags(text, name):
    raw = str(text).strip()
    if raw == "":
        return [0, 0, 0]
    vals = [v.strip() for v in raw.split(",") if v.strip()]
    if len(vals) != 3:
        raise ValueError(f"{name} must be empty or three comma-separated 0/1 flags, got '{text}'.")
    out = [int(v) for v in vals]
    if any(v not in (0, 1) for v in out):
        raise ValueError(f"{name} entries must be 0 or 1, got {out}.")
    return out


def _append_maw_model_payload(payload, rec, prefix, z_key):
    """Save one MAW weight model under an explicit prefix."""
    maw = rec["maw"]
    rbf = rec.get("rbf", None)
    ann = rec.get("ann", None)
    weight_model = rec.get("weight_model", rbf if rbf is not None else ann)
    regressor_type = str(rec.get("regressor_type", "rbf" if rbf is not None else "none")).strip().lower()

    payload[z_key] = np.asarray(rec["z_red"], dtype=np.int64)
    payload[prefix + "Z_ini"] = np.asarray(rec["z_ini"], dtype=np.int64)
    payload[prefix + "w_ini"] = np.asarray(rec["w_ini"], dtype=float)
    payload[prefix + "sample_ids"] = np.asarray(rec["ids"], dtype=np.int64)
    payload[prefix + "q_train"] = np.asarray(rec["q_train"], dtype=float)
    payload[prefix + "mu_train"] = np.asarray(rec["mu_train"], dtype=float)
    payload[prefix + "coord_label"] = np.array([str(rec.get("coord_label", "q"))])
    payload[prefix + "coord_train"] = np.asarray(
        rec.get("coord_train", rec["q_train"]),
        dtype=float,
    )
    if "coord_train_full" in rec:
        payload[prefix + "coord_train_full"] = np.asarray(rec["coord_train_full"], dtype=float)
    if "prune_subset_indices" in rec:
        payload[prefix + "prune_subset_indices"] = np.asarray(
            rec["prune_subset_indices"],
            dtype=np.int64,
        )
    payload[prefix + "component_scales"] = np.asarray(
        rec.get("component_scales", np.ones(0, dtype=float)),
        dtype=float,
    )
    payload[prefix + "regressor_type"] = np.array([regressor_type])
    payload[prefix + "W_train"] = np.asarray(rec["W_train"], dtype=float)
    payload[prefix + "b_train"] = np.asarray(rec["b_train"], dtype=float)
    payload[prefix + "m_constraints"] = np.array([int(rec["A_m"])], dtype=np.int64)
    payload[prefix + "m_constraints_per_block"] = np.asarray(
        rec.get(
            "m_constraints_per_block",
            np.full(
                int(np.asarray(rec["q_train"]).shape[0]),
                int(rec["A_m"]),
                dtype=np.int64,
            ),
        ),
        dtype=np.int64,
    )
    payload[prefix + "rhs_mode"] = np.array([str(np.ravel(rec["rhs_mode"])[0])])
    payload[prefix + "sum_constraint"] = np.array([int(rec.get("sum_constraint", 0))], dtype=np.int64)
    payload[prefix + "sum_target"] = np.array([float(rec.get("sum_target", rec["renorm_target"]))], dtype=float)
    payload[prefix + "renorm_target"] = np.array([float(rec["renorm_target"])], dtype=float)
    payload[prefix + "recon_rel"] = np.array([float(rec["recon_rel"])], dtype=float)
    payload[prefix + "hom_component_index"] = np.array(
        [int(rec.get("hom_component_index", -1))],
        dtype=np.int64,
    )
    payload[prefix + "is_fixed_classic"] = np.array(
        [1 if bool(rec.get("fixed_classic", False)) else 0],
        dtype=np.int64,
    )

    if regressor_type == "fixed_classic" or bool(rec.get("fixed_classic", False)):
        payload[prefix + "regressor_type"] = np.array(["fixed_classic"])
        payload[prefix + "w_fixed"] = np.asarray(rec["w_sel"], dtype=float)
        return

    rowspace_info = dict(rec.get("rowspace_info", {}))
    payload[prefix + "rowspace_rank_min"] = np.array(
        [float(rowspace_info.get("rank_min", np.nan))], dtype=float
    )
    payload[prefix + "rowspace_rank_mean"] = np.array(
        [float(rowspace_info.get("rank_mean", np.nan))], dtype=float
    )
    payload[prefix + "rowspace_rank_max"] = np.array(
        [float(rowspace_info.get("rank_max", np.nan))], dtype=float
    )
    payload[prefix + "rowspace_reduced_blocks"] = np.array(
        [int(rowspace_info.get("n_row_reduced", 0))], dtype=np.int64
    )
    payload[prefix + "rowspace_rank0_count"] = np.array(
        [int(rowspace_info.get("n_rank_zero", 0))], dtype=np.int64
    )

    if maw is None or weight_model is None:
        return

    payload[prefix + "n_stop"] = np.array([int(maw["n_stop"])], dtype=np.int64)
    payload[prefix + "prune_history_active_counts"] = np.asarray(maw["active_counts"], dtype=np.int64)
    payload[prefix + "prune_history_removed_global"] = np.asarray(maw["removed_local"], dtype=np.int64)
    payload[prefix + "tol_neg"] = np.array([float(maw["tol_neg"])], dtype=float)
    payload[prefix + "alpha_smooth"] = np.array([float(maw["options"]["alpha_smooth"])], dtype=float)
    payload[prefix + "criterion"] = np.array([int(maw["options"]["criterion"])], dtype=np.int64)
    payload[prefix + "elapsed_sec"] = np.array([float(maw["elapsed_sec"])], dtype=float)
    payload[prefix + "phase1_start_size"] = np.array([int(maw["phase1_start_size"])], dtype=np.int64)
    payload[prefix + "phase1_end_size"] = np.array([int(maw["phase1_end_size"])], dtype=np.int64)
    payload[prefix + "phase2_started"] = np.array([int(bool(maw["phase2_started"]))], dtype=np.int64)
    payload[prefix + "phase2_start_size"] = np.array([int(maw["phase2_start_size"])], dtype=np.int64)
    payload[prefix + "phase2_end_size"] = np.array([int(maw["phase2_end_size"])], dtype=np.int64)
    payload[prefix + "phase2_attempts"] = np.array([int(maw["phase2_attempts"])], dtype=np.int64)
    payload[prefix + "phase2_successes"] = np.array([int(maw["phase2_successes"])], dtype=np.int64)
    payload[prefix + "stage_history"] = np.asarray(maw["stage_history"])

    if regressor_type == "ann":
        payload[prefix + "ann_x_mean"] = np.asarray(ann["x_mean"], dtype=float)
        payload[prefix + "ann_x_std"] = np.asarray(ann["x_std"], dtype=float)
        payload[prefix + "ann_activation"] = np.array([str(ann.get("activation", "silu"))])
        payload[prefix + "ann_hidden_dims"] = np.asarray(
            ann.get("hidden_dims", np.zeros(0, dtype=np.int64)),
            dtype=np.int64,
        )
        payload[prefix + "ann_target_sum"] = np.array([float(ann["target_sum"])], dtype=float)
        payload[prefix + "ann_n_layers"] = np.array([int(ann["n_layers"])], dtype=np.int64)
        payload[prefix + "ann_best_epoch"] = np.array([int(ann.get("best_epoch", -1))], dtype=np.int64)
        payload[prefix + "ann_best_val_loss"] = np.array([float(ann.get("best_val_loss", np.nan))], dtype=float)
        payload[prefix + "ann_train_rel_error"] = np.array([float(ann.get("train_rel_error", np.nan))], dtype=float)
        payload[prefix + "ann_val_rel_error"] = np.array([float(ann.get("val_rel_error", np.nan))], dtype=float)
        payload[prefix + "ann_physics_weight"] = np.array([float(ann.get("physics_weight", 0.0))], dtype=float)
        payload[prefix + "ann_train_constraint_rel_error"] = np.array(
            [float(ann.get("train_constraint_rel_error", np.nan))],
            dtype=float,
        )
        payload[prefix + "ann_val_constraint_rel_error"] = np.array(
            [float(ann.get("val_constraint_rel_error", np.nan))],
            dtype=float,
        )
        payload[prefix + "ann_elapsed_sec"] = np.array([float(ann.get("elapsed_sec", np.nan))], dtype=float)
        payload[prefix + "ann_final_lr"] = np.array([float(ann.get("final_lr", np.nan))], dtype=float)
        payload[prefix + "ann_lr_scheduler"] = np.array([int(ann.get("lr_scheduler", 0))], dtype=np.int64)
        for i in range(int(ann["n_layers"])):
            payload[prefix + f"ann_W_{i}"] = np.asarray(ann[f"W_{i}"], dtype=float)
            payload[prefix + f"ann_b_{i}"] = np.asarray(ann[f"b_{i}"], dtype=float)
    elif regressor_type == "rbf":
        payload[prefix + "rbf_centers"] = np.asarray(rbf["centers"], dtype=float)
        payload[prefix + "rbf_center_ids"] = np.asarray(rbf["center_ids"], dtype=np.int64)
        payload[prefix + "rbf_length_scales"] = np.asarray(rbf["length_scales"], dtype=float)
        payload[prefix + "rbf_alpha"] = np.asarray(rbf["Alpha"], dtype=float)
        payload[prefix + "rbf_beta"] = np.asarray(rbf["Beta"], dtype=float)
        payload[prefix + "rbf_scale"] = np.asarray(rbf["scale"], dtype=float)
        payload[prefix + "rbf_poly_mode"] = np.array([int(rbf["poly_mode"])], dtype=np.int64)
        payload[prefix + "rbf_lambda"] = np.array([float(rbf["lambda_reg"])], dtype=float)
        payload[prefix + "rbf_n_centers"] = np.array([int(rbf["n_centers"])], dtype=np.int64)


def _save_stage12b_file(out_file, dataset, ecm_fixed, results, args):
    n_elem = int(dataset["n_elem"])
    hom_componentwise = bool(int(getattr(args, "maw_hom_componentwise", 0))) and all(
        f"{base}_{comp}" in results
        for base in ("eps", "sig")
        for comp in range(3)
    )

    # Compatibility vectors
    z_res = np.asarray(results["res"]["z_red"], dtype=np.int64)
    z_eps = np.asarray(results["eps"]["z_red"], dtype=np.int64)
    z_sig = np.asarray(results["sig"]["z_red"], dtype=np.int64)
    z_union = np.union1d(np.union1d(z_res, z_eps), z_sig).astype(np.int64)

    w_res_full = np.asarray(results["res"]["w_full"], dtype=float)
    w_eps_full = np.asarray(results["eps"]["w_full"], dtype=float)
    w_sig_full = np.asarray(results["sig"]["w_full"], dtype=float)

    if hom_componentwise:
        maw_enabled_targets = [
            t for t in ("res", "eps", "sig") if results[t]["maw"] is not None
        ] + [
            f"{base}_{comp}"
            for base in ("eps", "sig")
            for comp in range(3)
            if results[f"{base}_{comp}"]["maw"] is not None
        ]
    else:
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
        "maw_q_dim": np.array([int(dataset["q_m_res"].shape[1])], dtype=np.int64),
        "maw_targets": np.array([",".join(maw_enabled_targets)]),
        "maw_res_mode": np.array([str(args.res_mode)]),
        "maw_hom_mode": np.array(
            ["disabled" if bool(int(args.disable_homogenization)) else str(args.hom_mode)]
        ),
        "maw_res_candidate_pool": np.array([str(args.res_candidate_pool)]),
        "maw_res_target_source": np.array([str(args.res_target_source)]),
        "maw_graph_type": np.array([str(args.graph_mode)]),
        "maw_knn": np.array([int(args.graph_knn)], dtype=np.int64),
        "maw_graph_kernel": np.array([str(args.graph_kernel)]),
        "maw_graph_sigma": np.array([float(args.graph_sigma)], dtype=float),
        "maw_prune_graph_subsample_size": np.array(
            [int(args.maw_prune_graph_subsample_size)], dtype=np.int64
        ),
        "maw_prune_graph_subsample_mode": np.array(
            [str(args.maw_prune_graph_subsample_mode)]
        ),
        "maw_prune_graph_subsample_seed": np.array(
            [int(args.maw_prune_graph_subsample_seed)], dtype=np.int64
        ),
        "maw_alpha_smooth": np.array([float(args.alpha_smooth)], dtype=float),
        "maw_use_global_graph_2ndstage": np.array(
            [int(args.use_global_graph_2ndstage)], dtype=np.int64
        ),
        "maw_smooth_laplacian_all_iterations": np.array(
            [int(args.smooth_laplacian_all_iterations)], dtype=np.int64
        ),
        "maw_min_support_size_res": np.array([int(args.maw_min_support_size_res)], dtype=np.int64),
        "maw_min_support_size_eps": np.array([int(args.maw_min_support_size_eps)], dtype=np.int64),
        "maw_min_support_size_sig": np.array([int(args.maw_min_support_size_sig)], dtype=np.int64),
        "maw_phase1_stop_size_res": np.array([int(args.maw_phase1_stop_size_res)], dtype=np.int64),
        "maw_phase1_stop_size_eps": np.array([int(args.maw_phase1_stop_size_eps)], dtype=np.int64),
        "maw_phase1_stop_size_sig": np.array([int(args.maw_phase1_stop_size_sig)], dtype=np.int64),
        "maw_max_number_zeros_active_set_loop": np.array(
            [int(args.max_number_zeros_active_set_loop_maw_ecm)], dtype=np.int64
        ),
        "maw_enforce_sum_weights": np.array([int(args.enforce_sum_weights)], dtype=np.int64),
        "maw_sum_weights_target_input": np.array([float(args.sum_weights_target)], dtype=float),
        "classic_constrain_sum_weights": np.array(
            [int(args.classic_constrain_sum_weights)], dtype=np.int64
        ),
        "maw_stage2_criterion": np.array([int(args.criterion)], dtype=np.int64),
        "maw_incremental_smoothing": np.array([int(args.incremental_smoothing)], dtype=np.int64),
        "maw_hom_weight_regressor": np.array([str(args.maw_hom_weight_regressor)]),
        "maw_hom_componentwise": np.array([int(hom_componentwise)], dtype=np.int64),
        "maw_rbf_clip_nonnegative": np.array([int(args.rbf_clip_nonnegative)], dtype=np.int64),
        "maw_rbf_renorm": np.array([int(args.rbf_renorm)], dtype=np.int64),
        "maw_postprocess_mode": np.array(
            [
                (
                    "softmax_sum"
                    if str(args.maw_hom_weight_regressor).strip().lower() == "ann"
                    else (
                        ("clip+" if int(args.rbf_clip_nonnegative) else "")
                        + ("renorm" if int(args.rbf_renorm) else "none")
                    )
                )
            ]
        ),
    }
    if dataset.get("grid_struct_res", None) is not None:
        payload["maw_res_structured_grid_shape"] = np.asarray(
            dataset["grid_struct_res"], dtype=np.int64
        )

    for t in ("res", "eps", "sig"):
        rec = results[t]
        maw = rec["maw"]
        rbf = rec.get("rbf", None)
        ann = rec.get("ann", None)
        weight_model = rec.get("weight_model", rbf if rbf is not None else ann)
        regressor_type = str(rec.get("regressor_type", "rbf" if rbf is not None else "none")).strip().lower()
        prefix = f"maw_{t}_"

        payload[prefix + "Z_ini"] = np.asarray(rec["z_ini"], dtype=np.int64)
        payload[prefix + "w_ini"] = np.asarray(rec["w_ini"], dtype=float)
        payload[prefix + "sample_ids"] = np.asarray(rec["ids"], dtype=np.int64)
        payload[prefix + "q_train"] = np.asarray(rec["q_train"], dtype=float)
        payload[prefix + "mu_train"] = np.asarray(rec["mu_train"], dtype=float)
        payload[prefix + "coord_label"] = np.array([str(rec.get("coord_label", "q"))])
        payload[prefix + "coord_train"] = np.asarray(
            rec.get("coord_train", rec["q_train"]),
            dtype=float,
        )
        payload[prefix + "component_scales"] = np.asarray(
            rec.get("component_scales", np.ones(0, dtype=float)),
            dtype=float,
        )
        payload[prefix + "regressor_type"] = np.array([regressor_type])
        payload[prefix + "W_train"] = np.asarray(rec["W_train"], dtype=float)
        payload[prefix + "b_train"] = np.asarray(rec["b_train"], dtype=float)
        payload[prefix + "m_constraints"] = np.array([int(rec["A_m"])], dtype=np.int64)
        payload[prefix + "m_constraints_per_block"] = np.asarray(
            rec.get(
                "m_constraints_per_block",
                np.full(
                    int(np.asarray(rec["q_train"]).shape[0]),
                    int(rec["A_m"]),
                    dtype=np.int64,
                ),
            ),
            dtype=np.int64,
        )
        rowspace_info = dict(rec.get("rowspace_info", {}))
        payload[prefix + "rowspace_rank_min"] = np.array(
            [float(rowspace_info.get("rank_min", np.nan))], dtype=float
        )
        payload[prefix + "rowspace_rank_mean"] = np.array(
            [float(rowspace_info.get("rank_mean", np.nan))], dtype=float
        )
        payload[prefix + "rowspace_rank_max"] = np.array(
            [float(rowspace_info.get("rank_max", np.nan))], dtype=float
        )
        payload[prefix + "rowspace_reduced_blocks"] = np.array(
            [int(rowspace_info.get("n_row_reduced", 0))], dtype=np.int64
        )
        payload[prefix + "rowspace_rank0_count"] = np.array(
            [int(rowspace_info.get("n_rank_zero", 0))], dtype=np.int64
        )
        payload[prefix + "recon_rel"] = np.array([float(rec["recon_rel"])], dtype=float)
        payload[prefix + "rhs_mode"] = np.array([str(np.ravel(rec["rhs_mode"])[0])])
        payload[prefix + "sum_constraint"] = np.array([int(rec.get("sum_constraint", 0))], dtype=np.int64)
        payload[prefix + "sum_target"] = np.array([float(rec.get("sum_target", rec["renorm_target"]))], dtype=float)
        payload[prefix + "is_fixed_classic"] = np.array(
            [1 if bool(rec.get("fixed_classic", False)) else 0],
            dtype=np.int64,
        )
        payload[prefix + "renorm_target"] = np.array([float(rec["renorm_target"])], dtype=float)
        if maw is not None and weight_model is not None:
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
            payload[prefix + "phase1_start_size"] = np.array(
                [int(maw["phase1_start_size"])], dtype=np.int64
            )
            payload[prefix + "phase1_end_size"] = np.array(
                [int(maw["phase1_end_size"])], dtype=np.int64
            )
            payload[prefix + "phase2_started"] = np.array(
                [int(bool(maw["phase2_started"]))], dtype=np.int64
            )
            payload[prefix + "phase2_start_size"] = np.array(
                [int(maw["phase2_start_size"])], dtype=np.int64
            )
            payload[prefix + "phase2_end_size"] = np.array(
                [int(maw["phase2_end_size"])], dtype=np.int64
            )
            payload[prefix + "phase2_attempts"] = np.array(
                [int(maw["phase2_attempts"])], dtype=np.int64
            )
            payload[prefix + "phase2_successes"] = np.array(
                [int(maw["phase2_successes"])], dtype=np.int64
            )
            payload[prefix + "stage_history"] = np.asarray(maw["stage_history"])

            if regressor_type == "ann":
                payload[prefix + "ann_x_mean"] = np.asarray(ann["x_mean"], dtype=float)
                payload[prefix + "ann_x_std"] = np.asarray(ann["x_std"], dtype=float)
                payload[prefix + "ann_activation"] = np.array([str(ann.get("activation", "silu"))])
                payload[prefix + "ann_hidden_dims"] = np.asarray(
                    ann.get("hidden_dims", np.zeros(0, dtype=np.int64)),
                    dtype=np.int64,
                )
                payload[prefix + "ann_target_sum"] = np.array([float(ann["target_sum"])], dtype=float)
                payload[prefix + "ann_n_layers"] = np.array([int(ann["n_layers"])], dtype=np.int64)
                payload[prefix + "ann_best_epoch"] = np.array([int(ann.get("best_epoch", -1))], dtype=np.int64)
                payload[prefix + "ann_best_val_loss"] = np.array([float(ann.get("best_val_loss", np.nan))], dtype=float)
                payload[prefix + "ann_train_rel_error"] = np.array([float(ann.get("train_rel_error", np.nan))], dtype=float)
                payload[prefix + "ann_val_rel_error"] = np.array([float(ann.get("val_rel_error", np.nan))], dtype=float)
                payload[prefix + "ann_physics_weight"] = np.array([float(ann.get("physics_weight", 0.0))], dtype=float)
                payload[prefix + "ann_train_constraint_rel_error"] = np.array(
                    [float(ann.get("train_constraint_rel_error", np.nan))],
                    dtype=float,
                )
                payload[prefix + "ann_val_constraint_rel_error"] = np.array(
                    [float(ann.get("val_constraint_rel_error", np.nan))],
                    dtype=float,
                )
                payload[prefix + "ann_elapsed_sec"] = np.array([float(ann.get("elapsed_sec", np.nan))], dtype=float)
                payload[prefix + "ann_final_lr"] = np.array([float(ann.get("final_lr", np.nan))], dtype=float)
                payload[prefix + "ann_lr_scheduler"] = np.array([int(ann.get("lr_scheduler", 0))], dtype=np.int64)
                for i in range(int(ann["n_layers"])):
                    payload[prefix + f"ann_W_{i}"] = np.asarray(ann[f"W_{i}"], dtype=float)
                    payload[prefix + f"ann_b_{i}"] = np.asarray(ann[f"b_{i}"], dtype=float)
            else:
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

    if hom_componentwise:
        for base in ("eps", "sig"):
            for comp in range(3):
                key = f"{base}_{comp}"
                rec = results[key]
                _append_maw_model_payload(
                    payload=payload,
                    rec=rec,
                    prefix=f"maw_{base}_{comp}_",
                    z_key=f"Z_{base}_{comp}",
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
        "w_eps_components_full",
        "w_sig_components_full",
        "Z_eps_norm",
        "w_eps_norm",
        "w_eps_norm_full",
        "Z_eps_xy",
        "w_eps_xy",
        "w_eps_xy_full",
        "Z_sig_norm",
        "w_sig_norm",
        "w_sig_norm_full",
        "Z_sig_xy",
        "w_sig_xy",
        "w_sig_xy_full",
        "HOM_ECM_SPLIT_MODE",
        "HOM_ECM_SPLIT_MODE_INPUT",
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
        "--res-mode",
        type=str,
        default="maw",
        choices=["maw", "fixed_classic"],
        help="Residual handling: adaptive MAW-ECM (default) or keep the classical ECM residual rule fixed.",
    )
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
    p.add_argument(
        "--classic-rsvd-randomized",
        type=int,
        default=0,
        choices=[0, 1],
        help="If 0, use deterministic numpy SVD for the classical ECM bootstrap basis.",
    )
    p.add_argument("--classic-ecm-tol-res", type=float, default=0.0)
    p.add_argument("--classic-ecm-tol-eps", type=float, default=0.0)
    p.add_argument("--classic-ecm-tol-sig", type=float, default=0.0)
    p.add_argument("--classic-max-unsuccessful-it", type=int, default=200)
    p.add_argument(
        "--classic-constrain-sum-weights",
        type=int,
        default=1,
        choices=[0, 1],
        help="If 1, enforce the classical ECM bootstrap sum-of-weights constraint.",
    )
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
        choices=["fixed_support"],
        help="MAW starts from the classical residual ECM support z_ini.",
    )
    p.add_argument(
        "--res-target-source",
        type=str,
        default="anchor",
        choices=["anchor"],
        help="Local targets are b_j=A_j*w_ini, matching the validated 2D workflow.",
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
        default="parameter_mesh_stage12",
        choices=["parameter_mesh_stage12", "knn"],
        help=(
            "Graph operator used in MAW pruning. parameter_mesh_stage12 uses the "
            "Delaunay tetrahedra of actual Stage7 states."
        ),
    )
    p.add_argument("--graph-knn", type=int, default=8)
    p.add_argument("--graph-kernel", type=str, default="gaussian", choices=["gaussian", "binary"])
    p.add_argument("--graph-sigma", type=float, default=0.0)
    p.add_argument(
        "--maw-prune-graph-subsample-size",
        type=int,
        default=0,
        help=(
            "If >0 and graph phase-2 is enabled, prune on this many graph states "
            "instead of all states. The final ANN can still use all states through "
            "the physics loss. Currently supported with --graph-mode knn."
        ),
    )
    p.add_argument(
        "--maw-prune-graph-subsample-mode",
        type=str,
        default="uniform",
        choices=["uniform", "random"],
    )
    p.add_argument("--maw-prune-graph-subsample-seed", type=int, default=11)
    p.add_argument("--alpha-smooth", type=float, default=0.1)
    p.add_argument(
        "--use-global-graph-2ndstage",
        type=int,
        default=0,
        choices=[0, 1],
        help="Enable graph-coupled phase 2. Default 0 uses local active-set phase 2.",
    )
    p.add_argument(
        "--smooth-laplacian-all-iterations",
        type=int,
        default=0,
        choices=[0, 1],
        help="If 1, phase 2 is forced from the first deletion attempt.",
    )
    p.add_argument(
        "--maw-min-support-size-res",
        type=int,
        default=0,
        help="Hard lower bound for residual MAW support. <=0 uses the algebraic minimum.",
    )
    p.add_argument("--maw-min-support-size-eps", type=int, default=0)
    p.add_argument("--maw-min-support-size-sig", type=int, default=0)
    p.add_argument(
        "--maw-min-support-size-eps-components",
        type=str,
        default="",
        help=(
            "Component-wise eps n_stop as 'xx,yy,xy' when --maw-hom-componentwise 1. "
            "Empty uses --maw-min-support-size-eps for all three."
        ),
    )
    p.add_argument(
        "--maw-min-support-size-sig-components",
        type=str,
        default="",
        help=(
            "Component-wise sig n_stop as 'xx,yy,xy' when --maw-hom-componentwise 1. "
            "Empty uses --maw-min-support-size-sig for all three."
        ),
    )
    p.add_argument(
        "--maw-phase1-stop-size-res",
        type=int,
        default=0,
        help="Residual support size where phase 1 intentionally hands control to phase 2.",
    )
    p.add_argument("--maw-phase1-stop-size-eps", type=int, default=0)
    p.add_argument("--maw-phase1-stop-size-sig", type=int, default=0)
    p.add_argument(
        "--maw-phase1-stop-size-eps-components",
        type=str,
        default="",
        help=(
            "Component-wise eps phase-1 stop sizes as 'xx,yy,xy'. "
            "Empty uses --maw-phase1-stop-size-eps for all three."
        ),
    )
    p.add_argument(
        "--maw-phase1-stop-size-sig-components",
        type=str,
        default="",
        help=(
            "Component-wise sig phase-1 stop sizes as 'xx,yy,xy'. "
            "Empty uses --maw-phase1-stop-size-sig for all three."
        ),
    )
    p.add_argument(
        "--maw-fixed-classic-eps-components",
        type=str,
        default="",
        help=(
            "Component-wise eps fixed-classic flags as 'xx,yy,xy'. "
            "Use e.g. 0,0,1 to keep gamma_xy on the traditional ECM rule."
        ),
    )
    p.add_argument(
        "--maw-fixed-classic-sig-components",
        type=str,
        default="",
        help=(
            "Component-wise sig fixed-classic flags as 'xx,yy,xy'. "
            "Use e.g. 0,0,1 to keep sigma_xy on the traditional ECM rule."
        ),
    )
    p.add_argument(
        "--max-number-zeros-active-set-loop-maw-ecm",
        type=int,
        default=1,
        help=(
            "Enable phase-2 active-set elimination when >0. Use 0 for phase-1 only."
        ),
    )
    p.add_argument(
        "--enforce-sum-weights",
        type=int,
        default=1,
        choices=[0, 1],
        help="If 1, append sum(w)=target to every MAW local constraint block.",
    )
    p.add_argument(
        "--sum-weights-target",
        type=float,
        default=-1.0,
        help="Target sum(w) for MAW local constraints. <=0 uses sum(w_ini).",
    )
    p.add_argument("--criterion", type=int, default=2, choices=[0, 1, 2])
    p.add_argument("--n-candidates-to-try", type=int, default=0)
    p.add_argument("--incremental-smoothing", type=int, default=1, choices=[0, 1])
    p.add_argument("--use-total-as-criterion", type=int, default=0, choices=[0, 1])
    p.add_argument("--tol-rank-rel", type=float, default=1.0e-12)
    p.add_argument("--tol-neg-factor", type=float, default=10.0)
    p.add_argument("--tol-zero", type=float, default=1.0e-12)
    p.add_argument("--max-as-iters", type=int, default=30)
    p.add_argument("--max-reduced-dim", type=int, default=2500)
    p.add_argument("--maw-graph-cg-maxiter", type=int, default=1000)
    p.add_argument("--maw-graph-cg-rtol", type=float, default=1.0e-9)
    p.add_argument("--strict-constraint-rel-tol", type=float, default=1.0e-8)
    p.add_argument("--strict-negative-tol", type=float, default=1.0e-12)
    p.add_argument(
        "--maw-hom-rowspace-compress",
        type=int,
        default=1,
        choices=[0, 1],
        help=(
            "If 1, eps/sig MAW local blocks are projected to their numerical row-space "
            "after adding the sum constraint. This removes redundant equations before pruning."
        ),
    )
    p.add_argument("--maw-hom-rowspace-tol-rel", type=float, default=1.0e-12)
    p.add_argument(
        "--maw-hom-componentwise",
        type=int,
        default=0,
        choices=[0, 1],
        help=(
            "If 1, build six independent MAW rules for eps_xx, eps_yy, gamma_xy, "
            "sig_xx, sig_yy and sig_xy. Each component keeps its own positive, "
            "sum-preserving adaptive weights."
        ),
    )
    p.add_argument("--rbf-centers-res", type=int, default=0)
    p.add_argument("--rbf-centers-eps", type=int, default=0)
    p.add_argument("--rbf-centers-sig", type=int, default=0)
    p.add_argument("--rbf-poly-mode", type=int, default=1, choices=[0, 1, 2])
    p.add_argument("--rbf-lambda", type=float, default=1.0e-10)
    p.add_argument("--rbf-length-scale-factor", type=float, default=1.0)
    p.add_argument("--rbf-clip-nonnegative", type=int, default=1, choices=[0, 1])
    p.add_argument("--rbf-renorm", type=int, default=1, choices=[0, 1])
    p.add_argument(
        "--maw-hom-weight-regressor",
        type=str,
        default="rbf",
        choices=["rbf", "ann"],
        help="Regressor for eps/sig MAW weight fields. Residual MAW still uses RBF.",
    )
    p.add_argument("--maw-ann-hidden-dims", type=str, default="128,128,128")
    p.add_argument("--maw-ann-activation", type=str, default="silu", choices=["silu", "tanh", "relu", "gelu"])
    p.add_argument("--maw-ann-epochs", type=int, default=2000)
    p.add_argument("--maw-ann-batch-size", type=int, default=2048)
    p.add_argument("--maw-ann-lr", type=float, default=1.0e-3)
    p.add_argument("--maw-ann-weight-decay", type=float, default=1.0e-6)
    p.add_argument("--maw-ann-val-fraction", type=float, default=0.1)
    p.add_argument("--maw-ann-patience", type=int, default=200)
    p.add_argument("--maw-ann-lr-scheduler", type=int, default=1, choices=[0, 1])
    p.add_argument("--maw-ann-lr-scheduler-factor", type=float, default=0.5)
    p.add_argument("--maw-ann-lr-scheduler-patience", type=int, default=100)
    p.add_argument("--maw-ann-min-lr", type=float, default=1.0e-6)
    p.add_argument("--maw-ann-seed", type=int, default=11)
    p.add_argument(
        "--maw-ann-mse-weight",
        type=float,
        default=10.0,
        help="Extra probability-MSE weight added to the ANN KL loss.",
    )
    p.add_argument(
        "--maw-ann-physics-weight",
        type=float,
        default=0.0,
        help=(
            "Extra local constraint loss for ANN MAW hom weights: "
            "mean ||A_j w(mu_j)-b_j||^2/||b_j||^2. Use >0 for component-normalized "
            "physics-aware training."
        ),
    )
    p.add_argument(
        "--save-weight-field-plots",
        type=int,
        default=1,
        choices=[0, 1],
        help="Save three-view 3D state-space plots of each adaptive MAW weight field.",
    )
    p.add_argument(
        "--max-weight-field-plots",
        type=int,
        default=0,
        help="Maximum plots per MAW target. 0 saves all selected-element fields.",
    )
    p.add_argument("--weight-plot-format", type=str, default="png")
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
    hom_componentwise = (not disable_hom) and hom_mode == "maw" and bool(int(args.maw_hom_componentwise))
    eps_component_nstop = _parse_component_sizes(
        args.maw_min_support_size_eps_components,
        fallback=int(args.maw_min_support_size_eps),
        name="--maw-min-support-size-eps-components",
    )
    sig_component_nstop = _parse_component_sizes(
        args.maw_min_support_size_sig_components,
        fallback=int(args.maw_min_support_size_sig),
        name="--maw-min-support-size-sig-components",
    )
    eps_component_phase1 = _parse_component_sizes(
        args.maw_phase1_stop_size_eps_components,
        fallback=int(args.maw_phase1_stop_size_eps),
        name="--maw-phase1-stop-size-eps-components",
    )
    sig_component_phase1 = _parse_component_sizes(
        args.maw_phase1_stop_size_sig_components,
        fallback=int(args.maw_phase1_stop_size_sig),
        name="--maw-phase1-stop-size-sig-components",
    )
    eps_component_fixed = _parse_component_flags(
        args.maw_fixed_classic_eps_components,
        name="--maw-fixed-classic-eps-components",
    )
    sig_component_fixed = _parse_component_flags(
        args.maw_fixed_classic_sig_components,
        name="--maw-fixed-classic-sig-components",
    )
    if disable_hom and hom_mode == "maw":
        raise ValueError("Homogenization is disabled; --hom-mode maw is not allowed.")
    if (not disable_hom) and hom_mode == "maw":
        missing_hom_targets = [t for t in ("eps", "sig") if t not in targets]
        if missing_hom_targets:
            raise ValueError(
                "--hom-mode maw requires separated homogenization MAW targets. "
                f"Missing targets: {missing_hom_targets}. Use --targets res,eps,sig."
            )
        if not bool(int(args.classic_constrain_sum_weights)):
            raise ValueError(
                "--hom-mode maw requires --classic-constrain-sum-weights 1 so the "
                "initial eps/sig ECM rules are volume/sum preserving."
            )
        if not bool(int(args.enforce_sum_weights)):
            raise ValueError(
                "--hom-mode maw requires --enforce-sum-weights 1 for every local "
                "eps/sig MAW block."
            )
        if str(args.maw_hom_weight_regressor).strip().lower() == "rbf" and not bool(int(args.rbf_clip_nonnegative)):
            raise ValueError(
                "--hom-mode maw requires --rbf-clip-nonnegative 1 so online eps/sig "
                "weights remain nonnegative."
            )
        if str(args.maw_hom_weight_regressor).strip().lower() == "rbf" and not bool(int(args.rbf_renorm)):
            raise ValueError(
                "--hom-mode maw requires --rbf-renorm 1 so online eps/sig weights "
                "preserve the target sum."
            )
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
    fixed_ecm_label = str(args.fixed_ecm_dir) if classic_source == "file" else "<unused>"
    print(f"fixed_ecm   : {fixed_ecm_label}")
    print(f"out_file    : {out_file}")
    print(f"maw targets : {targets}")
    print(f"res_mode    : {args.res_mode}")
    print(f"hom_mode    : {hom_mode}")
    print(f"hom compwise: {int(hom_componentwise)}")
    if hom_componentwise:
        print(f"eps n_stop components [xx,yy,xy]: {eps_component_nstop}")
        print(f"sig n_stop components [xx,yy,xy]: {sig_component_nstop}")
        print(f"eps fixed-classic components [xx,yy,xy]: {eps_component_fixed}")
        print(f"sig fixed-classic components [xx,yy,xy]: {sig_component_fixed}")
    print(f"res pool/rhs: {args.res_candidate_pool} / {args.res_target_source}")
    print(f"graph mode  : {args.graph_mode}")
    print(f"n_elem      : {n_elem}")
    print(f"nq          : {dataset['nq']}")
    if disable_hom:
        print(f"N_s_res     : {dataset['ns_res']}")
    else:
        print(f"N_s_res/hom : {dataset['ns_res']} / {dataset['ns_hom']}")

    results = {}
    if str(args.res_mode).strip().lower() == "fixed_classic":
        results["res"] = _build_fixed_res_target_result(dataset, ecm_fixed)
        print(
            f"  [RES] fixed classical residual support kept: |Z|={results['res']['z_red'].size} "
            "(no residual MAW pruning)"
        )
    else:
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
            if hom_mode == "maw" and t in targets and hom_componentwise:
                comp_records = []
                for comp in range(3):
                    comp_key = f"{t}_{comp}"
                    fixed_flags = eps_component_fixed if t == "eps" else sig_component_fixed
                    if int(fixed_flags[comp]) == 1:
                        rec = _build_fixed_hom_component_result(
                            dataset=dataset,
                            ecm_fixed=ecm_fixed,
                            target=t,
                            component_index=comp,
                        )
                        print(
                            f"  [Stage12b] Fixed classical target '{comp_key}' kept: "
                            f"|Z|={rec['z_red'].size}",
                            flush=True,
                        )
                    else:
                        print(f"  [Stage12b] Starting MAW target '{comp_key}'...", flush=True)
                        rec = _run_target_maw(
                            target=t,
                            dataset=dataset,
                            ecm_fixed=ecm_fixed,
                            n_elem=n_elem,
                            args=args,
                            label=comp_key,
                            hom_component_index=comp,
                            min_support_override=(
                                eps_component_nstop[comp] if t == "eps" else sig_component_nstop[comp]
                            ),
                            phase1_stop_size_override=(
                                eps_component_phase1[comp] if t == "eps" else sig_component_phase1[comp]
                            ),
                        )
                    results[comp_key] = rec
                    comp_records.append(rec)
                    print(f"  [Stage12b] Finished target '{comp_key}'.", flush=True)
                results[t] = _build_componentwise_hom_target_result(
                    dataset=dataset,
                    target=t,
                    component_results=comp_records,
                )
                sizes = [int(rec["z_red"].size) for rec in comp_records]
                print(
                    f"  [MAW-{t}] component-wise supports kept: {sizes}; "
                    f"union |Z|={results[t]['z_red'].size}",
                    flush=True,
                )
            elif hom_mode == "maw" and t in targets:
                print(f"  [Stage12b] Starting MAW target '{t}'...", flush=True)
                results[t] = _run_target_maw(
                    target=t,
                    dataset=dataset,
                    ecm_fixed=ecm_fixed,
                    n_elem=n_elem,
                    args=args,
                )
                print(f"  [Stage12b] Finished MAW target '{t}'.", flush=True)
            else:
                results[t] = _build_fixed_hom_target_result(dataset, ecm_fixed, target=t)
                print(
                    f"  [MAW-{t}] fixed classical homogenization support kept: |Z|={results[t]['z_red'].size}"
                )

    if bool(int(args.save_weight_field_plots)):
        plot_keys = ["res"]
        if hom_componentwise:
            plot_keys.extend([f"{base}_{comp}" for base in ("eps", "sig") for comp in range(3)])
        else:
            plot_keys.extend(["eps", "sig"])
        for t in plot_keys:
            rec = results[t]
            if rec["maw"] is None or rec.get("weight_model", rec.get("rbf", None)) is None:
                continue
            print(f"  [Stage12b] Starting weight-field plots for '{t}'...", flush=True)
            rec["n_weight_field_plots"] = _plot_maw_weight_fields_3d(
                q_train=rec.get("coord_train", rec["q_train"]),
                W_train=rec["W_train"],
                z_red=rec["z_red"],
                rbf_model=rec.get("weight_model", rec.get("rbf")),
                out_dir=args.out_dir,
                target=t,
                max_plots=int(args.max_weight_field_plots),
                fmt=str(args.weight_plot_format),
                clip_nonnegative=bool(int(args.rbf_clip_nonnegative)),
                renorm_target=(
                    float(rec["renorm_target"]) if bool(int(args.rbf_renorm)) else None
                ),
                coord_label=rec.get("coord_label", "q"),
                regressor_type=rec.get("regressor_type", "rbf"),
            )
            print(f"  [Stage12b] Finished weight-field plots for '{t}'.", flush=True)

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
