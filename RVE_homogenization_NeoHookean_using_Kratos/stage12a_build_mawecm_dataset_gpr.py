#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage 12a: Build MAW-ECM dataset for HPROM-GPR.

Residual-only workflow (default):
  - residual projection (Q_ecm, b_full)
  - MAW metadata for residual samples (q_p_res, mu_res, sample_ids_res)
  - state source can be trajectory sampling or Stage7 structured mesh nodes

Optional legacy homogenization export can be enabled explicitly.

Residual projection uses the GPR-manifold tangent:
  J_m = (phi_p + phi_s*J0) + phi_s * d(q_s_corr)/d(q_p)
"""

import os
import sys
import argparse
import numpy as np
from scipy.spatial import cKDTree

# Add Kratos path
KRATOS_PATH = "/home/kratos/Kratos_Eigen_Check/bin/Release"
if KRATOS_PATH not in sys.path:
    sys.path.append(KRATOS_PATH)

import KratosMultiphysics as KM
from fom_solver_rve import (
    DeformationGradientFromGreenLagrange2D,
    RVEHomogenizationDatasetGenerator,
    SetUpDofEquationIdsAndDisplacementAdaptor,
    SetDisplacementFromEquationVector,
    UpdateCurrentCoordinatesFromDisplacement,
    PrecomputeElementIntegrationWeights,
    EvaluateGaussPointData,
    SetInputMeshFilename,
    StripMdpaExtension,
    DetectMaterialSubModelParts,
    ConfigureElementModelerForMaterialParts,
)
from sparse_gp_manifold_model import (
    load_sparse_gp_model,
    evaluate_sparse_gp_map_and_jacobian_qp,
)
from ecm_sampling_utils import get_stratified_indices, get_param_aware_indices


SNAPSHOTS_DIR = "stage_1_training_set_fom"
BASIS_DIR = "stage_2_pod_rve"
GPR_DIR = "stage_7_gpr_data_ls"
STRUCTURED_MESH_DIR = "stage_7_ann_data_ls"
OUT_DIR = "stage_12_mawecm_dataset_gpr_ls"
SNAPSHOT_PERCENT_RES = 5.0
SNAPSHOT_PERCENT_HOM = 5.0
SEED = 42
SAMPLING_MODE = "param_aware"  # options: "param_aware", "stratified"
PARAM_AWARE_TIME_WEIGHT = 0.20
RESIDUAL_FIT_MODE = "none"  # options: none, gauss_newton
FIT_MAX_ITERS = 8
FIT_REL_TOL = 1e-6
FIT_L2_REG = 1e-10
FIT_STEP_TOL = 1e-12
RESIDUAL_STATE_SOURCE = "stage7_structured_mesh"  # options: trajectory_sampling, stage7_structured_mesh


def _build_free_map(n_dof, free_dofs):
    g2f = -np.ones(int(n_dof), dtype=int)
    for i, gdof in enumerate(np.asarray(free_dofs, dtype=np.int64)):
        g2f[int(gdof)] = int(i)
    return g2f


def _allocate_memmaps(out_dir, nq, ns_res, ns_hom, n_elem, include_homogenization):
    os.makedirs(out_dir, exist_ok=True)
    q_path = os.path.join(out_dir, "Q_ecm.dat")
    b_path = os.path.join(out_dir, "b_full.dat")
    c_path = os.path.join(out_dir, "C_hom.dat")
    bh_path = os.path.join(out_dir, "b_hom.dat")

    for p in (q_path, b_path):
        if os.path.exists(p):
            os.remove(p)
    for p in (c_path, bh_path):
        if os.path.exists(p):
            os.remove(p)

    q_ecm = np.memmap(q_path, dtype="float64", mode="w+", shape=(int(nq) * int(ns_res), int(n_elem)))
    b_full = np.memmap(b_path, dtype="float64", mode="w+", shape=(int(nq) * int(ns_res),))
    c_hom = None
    b_hom = None
    if bool(include_homogenization):
        c_hom = np.memmap(c_path, dtype="float64", mode="w+", shape=(6 * int(ns_hom), int(n_elem)))
        b_hom = np.memmap(bh_path, dtype="float64", mode="w+", shape=(6 * int(ns_hom),))
    return q_ecm, b_full, c_hom, b_hom


def _extract_snapshot_u(u_all, step_idx, n_dof):
    if u_all.ndim != 2:
        raise ValueError(f"Invalid snapshot array shape {u_all.shape}. Expected 2D.")
    if u_all.shape[1] == n_dof:
        return np.asarray(u_all[step_idx, :], dtype=float)
    if u_all.shape[0] == n_dof:
        return np.asarray(u_all[:, step_idx], dtype=float)
    raise ValueError(
        f"Cannot infer displacement layout from shape {u_all.shape} with n_dof={n_dof}."
    )


def _infer_n_u_steps(u_meta, n_dof):
    if u_meta.ndim != 2:
        return 0
    if u_meta.shape[1] == n_dof:
        return int(u_meta.shape[0])
    if u_meta.shape[0] == n_dof:
        return int(u_meta.shape[1])
    return 0


def _pick_snapshot_indices(e_traj, n_steps, n_pick, mode, seed, time_weight):
    n_pick = int(max(1, min(int(n_steps), int(n_pick))))
    if str(mode).strip().lower() == "param_aware":
        return get_param_aware_indices(
            e_traj[:n_steps, :3],
            n_pick,
            seed=int(seed),
            time_weight=float(time_weight),
        )
    return get_stratified_indices(int(n_steps), n_pick, seed=int(seed))


def _load_stage7_structured_mesh_nodes(structured_mesh_dir, n_primary):
    mu_path = os.path.join(structured_mesh_dir, "parameter_mesh_nodes_mu.npy")
    qls_path = os.path.join(structured_mesh_dir, "parameter_mesh_nodes_q_ls.npy")
    cells_path = os.path.join(structured_mesh_dir, "parameter_mesh_cells.npy")
    grid_path = os.path.join(structured_mesh_dir, "parameter_mesh_structured_grid_shape.npy")

    for p in (mu_path, qls_path, cells_path):
        if not os.path.exists(p):
            raise FileNotFoundError(p)

    mu_nodes = np.asarray(np.load(mu_path), dtype=float)
    qls_nodes = np.asarray(np.load(qls_path), dtype=float)
    cells = np.asarray(np.load(cells_path), dtype=np.int64)
    grid_shape = (
        np.asarray(np.load(grid_path), dtype=np.int64)
        if os.path.exists(grid_path)
        else np.zeros(0, dtype=np.int64)
    )

    if mu_nodes.ndim != 2 or mu_nodes.shape[1] < 3:
        raise RuntimeError(f"Invalid parameter_mesh_nodes_mu shape: {mu_nodes.shape}")
    if qls_nodes.ndim != 2 or qls_nodes.shape[0] != mu_nodes.shape[0]:
        raise RuntimeError(
            f"Invalid parameter_mesh_nodes_q_ls shape: {qls_nodes.shape}, expected ({mu_nodes.shape[0]},*)"
        )
    if qls_nodes.shape[1] < int(n_primary):
        raise RuntimeError(
            f"parameter_mesh_nodes_q_ls has dim={qls_nodes.shape[1]} < n_primary={int(n_primary)}"
        )
    if cells.ndim != 2 or cells.shape[1] not in (4, 8):
        raise RuntimeError(f"Invalid parameter_mesh_cells shape: {cells.shape}")
    if np.min(cells) < 0 or np.max(cells) >= mu_nodes.shape[0]:
        raise RuntimeError(
            "parameter_mesh_cells has indices out of bounds for parameter_mesh_nodes_*."
        )

    q_p_nodes = np.asarray(qls_nodes[:, : int(n_primary)], dtype=float)
    mu3_nodes = np.asarray(mu_nodes[:, :3], dtype=float)
    return q_p_nodes, mu3_nodes, cells, grid_shape


def _parse_trajectory_indices(text):
    if not str(text).strip():
        return None
    try:
        vals = sorted(set(int(v.strip()) for v in str(text).split(",") if v.strip()))
    except ValueError as exc:
        raise ValueError("--trajectory-indices must be comma-separated integers.") from exc
    if any(v <= 0 for v in vals):
        raise ValueError("--trajectory-indices must be >= 1.")
    return vals


def _build_residual_projection_cache(elements, process_info, g2f):
    cache = []
    active = 0
    for elem in elements:
        ids = np.asarray(elem.EquationIdVector(process_info), dtype=int)
        local_pos = np.flatnonzero(ids >= 0)
        if local_pos.size == 0:
            cache.append(None)
            continue

        local_dofs = ids[local_pos]
        rows = g2f[local_dofs]
        valid = rows >= 0
        if not np.any(valid):
            cache.append(None)
            continue

        rhs_pick = local_pos[valid].astype(int, copy=False)
        rows_valid = np.asarray(rows[valid], dtype=int)
        cache.append((rhs_pick, rows_valid))
        active += 1
    return cache, active


def _fill_hom_block_from_kratos(elements, model_part, area_e, c_block):
    eps_gp, sig_gp, _ = EvaluateGaussPointData(elements, model_part)
    c_block[0:3, :] = (area_e[:, None] * np.mean(eps_gp, axis=1)).T
    c_block[3:6, :] = (area_e[:, None] * np.mean(sig_gp, axis=1)).T


def _evaluate_gpr_qs_and_jac(q_p, gpr_model, n_primary):
    q_s, j_qs_qp = evaluate_sparse_gp_map_and_jacobian_qp(
        np.asarray(q_p, dtype=float), gpr_model, n_primary
    )
    return np.asarray(q_s, dtype=float), np.asarray(j_qs_qp, dtype=float)


def _evaluate_gpr_origin_terms(gpr_model, n_primary):
    return _evaluate_gpr_qs_and_jac(np.zeros(n_primary, dtype=float), gpr_model, n_primary)


def _apply_manifold_origin_correction(q_p, q_s_raw, j_qs_qp_raw, q0_ref, j0_ref):
    q_vec = np.asarray(q_p, dtype=float).reshape(-1)
    q_s_corr = np.asarray(q_s_raw, dtype=float).reshape(-1) - np.asarray(q0_ref, dtype=float).reshape(-1) - (
        np.asarray(j0_ref, dtype=float) @ q_vec
    )
    j_corr = np.asarray(j_qs_qp_raw, dtype=float) - np.asarray(j0_ref, dtype=float)
    return q_s_corr, j_corr


def _fit_qp_gauss_newton(
    q_init,
    w_free,
    phi_p,
    phi_s,
    gpr_model,
    n_primary,
    n_secondary,
    q0_ref,
    j0_ref,
    max_iters,
    rel_tol,
    l2_reg,
    step_tol,
):
    q = np.asarray(q_init, dtype=float).copy()
    w_target = np.asarray(w_free, dtype=float).reshape(-1)
    target_norm = max(np.linalg.norm(w_target), 1e-30)
    phi_p_eff = np.asarray(phi_p, dtype=float) + np.asarray(phi_s, dtype=float) @ np.asarray(j0_ref, dtype=float)
    w0_const = np.asarray(phi_s, dtype=float) @ np.asarray(q0_ref, dtype=float).reshape(-1)

    q_s_raw, j_qs_qp_raw = _evaluate_gpr_qs_and_jac(q, gpr_model, n_primary)
    q_s, j_qs_qp = _apply_manifold_origin_correction(q, q_s_raw, j_qs_qp_raw, q0_ref, j0_ref)
    if j_qs_qp.shape != (n_secondary, n_primary):
        raise RuntimeError(
            f"Invalid GPR Jacobian shape {j_qs_qp.shape}, expected {(n_secondary, n_primary)}."
        )
    w_rec = w0_const + phi_p_eff @ q + phi_s @ q_s
    rel0 = float(np.linalg.norm(w_rec - w_target) / target_norm)
    rel = rel0
    n_it = 0

    for it in range(int(max_iters)):
        if rel <= float(rel_tol):
            break
        J_m = phi_p_eff + phi_s @ j_qs_qp
        res = w_rec - w_target
        jtj = J_m.T @ J_m
        if l2_reg > 0.0:
            jtj = jtj + float(l2_reg) * np.eye(jtj.shape[0], dtype=float)
        rhs = J_m.T @ res
        try:
            dq = np.linalg.solve(jtj, rhs)
        except np.linalg.LinAlgError:
            dq, *_ = np.linalg.lstsq(jtj, rhs, rcond=None)
        q -= dq
        n_it = it + 1
        if np.linalg.norm(dq) <= float(step_tol) * max(np.linalg.norm(q), 1.0):
            q_s_raw, j_qs_qp_raw = _evaluate_gpr_qs_and_jac(q, gpr_model, n_primary)
            q_s, j_qs_qp = _apply_manifold_origin_correction(q, q_s_raw, j_qs_qp_raw, q0_ref, j0_ref)
            w_rec = w0_const + phi_p_eff @ q + phi_s @ q_s
            rel = float(np.linalg.norm(w_rec - w_target) / target_norm)
            break
        q_s_raw, j_qs_qp_raw = _evaluate_gpr_qs_and_jac(q, gpr_model, n_primary)
        q_s, j_qs_qp = _apply_manifold_origin_correction(q, q_s_raw, j_qs_qp_raw, q0_ref, j0_ref)
        if j_qs_qp.shape != (n_secondary, n_primary):
            raise RuntimeError(
                f"Invalid GPR Jacobian shape {j_qs_qp.shape}, expected {(n_secondary, n_primary)}."
            )
        w_rec = w0_const + phi_p_eff @ q + phi_s @ q_s
        rel = float(np.linalg.norm(w_rec - w_target) / target_norm)

    return q, q_s, j_qs_qp, rel0, rel, int(n_it)


def _parse_args():
    p = argparse.ArgumentParser(
        description="Stage 12a MAW-ECM-GPR: build ECM dataset with MAW manifold metadata."
    )
    p.add_argument("--snapshots-dir", type=str, default=SNAPSHOTS_DIR)
    p.add_argument("--basis-dir", type=str, default=BASIS_DIR)
    p.add_argument("--gpr-dir", type=str, default=GPR_DIR)
    p.add_argument("--structured-mesh-dir", type=str, default=STRUCTURED_MESH_DIR)
    p.add_argument("--out-dir", type=str, default=OUT_DIR)
    p.add_argument(
        "--residual-state-source",
        type=str,
        default=RESIDUAL_STATE_SOURCE,
        choices=["trajectory_sampling", "stage7_structured_mesh"],
        help=(
            "Source of residual states used to build Stage12a. "
            "'trajectory_sampling' uses snapshot-percent over FOM trajectories; "
            "'stage7_structured_mesh' uses parameter_mesh_nodes_(mu,q_ls) from Stage7."
        ),
    )
    p.add_argument("--snapshot-percent-res", type=float, default=SNAPSHOT_PERCENT_RES)
    p.add_argument("--snapshot-percent-hom", type=float, default=SNAPSHOT_PERCENT_HOM)
    p.add_argument(
        "--disable-homogenization",
        type=int,
        default=1,
        choices=[0, 1],
        help="If 1, build residual-only Stage12a dataset (no C_hom/b_hom/q_p_hom exports).",
    )
    p.add_argument(
        "--sampling-mode",
        type=str,
        default=SAMPLING_MODE,
        choices=["param_aware", "stratified"],
    )
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--param-aware-time-weight", type=float, default=PARAM_AWARE_TIME_WEIGHT)
    p.add_argument(
        "--trajectory-indices",
        type=str,
        default="",
        help="Optional comma-separated trajectory indices (e.g. '2' or '1,2,5').",
    )
    p.add_argument(
        "--first-n-steps",
        type=int,
        default=0,
        help="If >0, only first N snapshots per selected trajectory are considered.",
    )
    p.add_argument(
        "--residual-fit-mode",
        type=str,
        default=RESIDUAL_FIT_MODE,
        choices=["none", "gauss_newton"],
        help="Optional manifold-consistent fit for residual snapshots before building Q_ecm.",
    )
    p.add_argument("--fit-max-iters", type=int, default=FIT_MAX_ITERS)
    p.add_argument("--fit-rel-tol", type=float, default=FIT_REL_TOL)
    p.add_argument("--fit-l2-reg", type=float, default=FIT_L2_REG)
    p.add_argument("--fit-step-tol", type=float, default=FIT_STEP_TOL)
    return p.parse_args()


def main():
    args = _parse_args()
    print("--- Stage 12a MAW-ECM-GPR: Building ECM Dataset ---")

    disable_hom = bool(int(args.disable_homogenization))
    include_hom = not disable_hom
    residual_state_source = str(args.residual_state_source).strip().lower()
    pct_res = float(args.snapshot_percent_res)
    pct_hom = float(args.snapshot_percent_hom)
    if residual_state_source == "trajectory_sampling":
        if pct_res <= 0.0:
            raise ValueError("snapshot-percent-res must be > 0.")
        if pct_res > 100.0:
            raise ValueError("snapshot-percent-res must be <= 100.")
        if include_hom:
            if pct_hom <= 0.0:
                raise ValueError("snapshot-percent-hom must be > 0 when homogenization is enabled.")
            if pct_hom > 100.0:
                raise ValueError("snapshot-percent-hom must be <= 100.")
        else:
            pct_hom = 0.0
    else:
        pct_res = 100.0
        pct_hom = 100.0 if include_hom else 0.0

    snapshots_dir = str(args.snapshots_dir)
    basis_dir = str(args.basis_dir)
    gpr_dir = str(args.gpr_dir)
    structured_mesh_dir = str(args.structured_mesh_dir)
    out_dir = str(args.out_dir)
    first_n_steps = int(args.first_n_steps)
    if first_n_steps < 0:
        raise ValueError("--first-n-steps must be >= 0.")
    fit_mode = str(args.residual_fit_mode).strip().lower()
    fit_max_iters = int(args.fit_max_iters)
    fit_rel_tol = float(args.fit_rel_tol)
    fit_l2_reg = float(args.fit_l2_reg)
    fit_step_tol = float(args.fit_step_tol)
    if fit_max_iters < 0:
        raise ValueError("--fit-max-iters must be >= 0.")
    if fit_rel_tol < 0.0:
        raise ValueError("--fit-rel-tol must be >= 0.")
    if fit_l2_reg < 0.0:
        raise ValueError("--fit-l2-reg must be >= 0.")
    if fit_step_tol < 0.0:
        raise ValueError("--fit-step-tol must be >= 0.")

    selected_traj_ids = _parse_trajectory_indices(args.trajectory_indices)

    phi_p = np.asarray(np.load(os.path.join(gpr_dir, "phi_p.npy")), dtype=float)
    phi_s = np.asarray(np.load(os.path.join(gpr_dir, "phi_s.npy")), dtype=float)
    free_dofs = np.asarray(np.load(os.path.join(basis_dir, "free_dofs.npy")), dtype=np.int64)
    dir_dofs = np.asarray(np.load(os.path.join(basis_dir, "dirichlet_dofs.npy")), dtype=np.int64)
    eq_map_ref = np.asarray(np.load(os.path.join(basis_dir, "eq_map.npy")), dtype=np.int64)
    gpr_model = load_sparse_gp_model(os.path.join(gpr_dir, "sparse_gp_model.npz"))

    n_primary = int(phi_p.shape[1])
    n_secondary = int(phi_s.shape[1])
    n_free = int(phi_p.shape[0])
    if int(phi_s.shape[0]) != n_free:
        raise ValueError("phi_p and phi_s row count mismatch.")
    if int(gpr_model["n_primary"]) != n_primary:
        raise ValueError("GPR model n_primary does not match phi_p.")
    if int(gpr_model["n_secondary"]) != n_secondary:
        raise ValueError("GPR model n_secondary does not match phi_s.")

    include_macro = bool(gpr_model["include_macro_strain_input"])
    if include_macro:
        raise RuntimeError(
            "GPR model was trained with macro-strain manifold inputs (N(q,mu)), "
            "which is no longer supported. Retrain Stage 7 without macro inputs."
        )
    n_total_dofs_ref = int(len(free_dofs) + len(dir_dofs))
    print(
        f"[Info] n_primary={n_primary}, n_secondary={n_secondary}, "
        f"include_macro_strain_input={int(include_macro)}"
    )
    print(
        f"[Info] residual fit mode={fit_mode}, fit_max_iters={fit_max_iters}, "
        f"fit_rel_tol={fit_rel_tol:.2e}, fit_l2_reg={fit_l2_reg:.2e}, fit_step_tol={fit_step_tol:.2e}"
    )
    print(f"[Info] residual-only mode (disable homogenization) = {int(disable_hom)}")
    print("[Info] Manifold correction active in Stage 12a MAW-ECM-GPR: N(0)=0 and J(0)=0.")
    all_tasks = []
    q_p_nodes_struct = None
    q_p_nodes_eval = None
    q_s_nodes_eval_raw = None
    mu_nodes_struct = None
    mu_nodes_eval = None
    cells_struct = None
    grid_shape_struct = None
    q_nn_idx = None
    q_dist = None
    mu_dist = None
    if residual_state_source == "trajectory_sampling":
        trajectories = sorted([d for d in os.listdir(snapshots_dir) if d.startswith("trajectory_")])
        if selected_traj_ids is not None:
            selected_names = {f"trajectory_{i}" for i in selected_traj_ids}
            trajectories = [t for t in trajectories if t in selected_names]
        if not trajectories:
            raise FileNotFoundError(
                f"No trajectory folders found in {snapshots_dir} for selection={selected_traj_ids}."
            )

        frac_res = pct_res / 100.0
        frac_hom = pct_hom / 100.0 if include_hom else 0.0
        for traj in trajectories:
            u_file = os.path.join(snapshots_dir, traj, f"{traj}_U.npy")
            e_file = os.path.join(snapshots_dir, traj, f"{traj}_applied_strain.npy")
            if not (os.path.exists(u_file) and os.path.exists(e_file)):
                print(f"  [Skip] {traj}: missing U or applied_strain file")
                continue

            u_meta = np.load(u_file, mmap_mode="r")
            e_meta = np.load(e_file, mmap_mode="r")
            if e_meta.ndim != 2 or e_meta.shape[1] < 3:
                print(f"  [Skip] {traj}: invalid applied_strain shape {e_meta.shape}")
                continue

            n_u_steps = _infer_n_u_steps(u_meta, n_total_dofs_ref)
            if n_u_steps <= 0:
                print(f"  [Skip] {traj}: cannot infer U layout with n_dof={n_total_dofs_ref}")
                continue

            n_steps = min(n_u_steps, int(e_meta.shape[0]))
            if first_n_steps > 0:
                n_steps = min(n_steps, first_n_steps)
            if n_steps <= 0:
                print(f"  [Skip] {traj}: no snapshots after first-n-steps filter")
                continue

            n_pick_res = int(np.ceil(frac_res * n_steps))
            n_pick_hom = int(np.ceil(frac_hom * n_steps)) if include_hom else 0
            idx_res = _pick_snapshot_indices(
                e_meta,
                n_steps=n_steps,
                n_pick=n_pick_res,
                mode=args.sampling_mode,
                seed=int(args.seed) + 2 * len(all_tasks),
                time_weight=args.param_aware_time_weight,
            )
            if include_hom:
                idx_hom = _pick_snapshot_indices(
                    e_meta,
                    n_steps=n_steps,
                    n_pick=n_pick_hom,
                    mode=args.sampling_mode,
                    seed=int(args.seed) + 2 * len(all_tasks) + 1,
                    time_weight=args.param_aware_time_weight,
                )
            else:
                idx_hom = np.zeros(0, dtype=int)

            all_tasks.append(
                {
                    "traj": traj,
                    "u_file": u_file,
                    "e_file": e_file,
                    "idx_res": np.asarray(idx_res, dtype=int),
                    "idx_hom": np.asarray(idx_hom, dtype=int),
                }
            )

        total_snapshots_res = int(sum(np.asarray(t["idx_res"], dtype=int).size for t in all_tasks))
        total_snapshots_hom = int(sum(np.asarray(t["idx_hom"], dtype=int).size for t in all_tasks))
        if total_snapshots_res <= 0:
            raise RuntimeError("No residual snapshots selected.")
        if include_hom and total_snapshots_hom <= 0:
            raise RuntimeError("No homogenization snapshots selected.")

        print(f"[Info] Residual state source: {residual_state_source}")
        print(f"[Info] Sampling mode: {args.sampling_mode}")
        print(f"[Info] Target residual snapshots      : {pct_res}% across {len(all_tasks)} trajectories.")
        if include_hom:
            print(f"[Info] Target homogenization snapshots: {pct_hom}% across {len(all_tasks)} trajectories.")
        else:
            print("[Info] Homogenization sampling is disabled in Stage12a.")
        if selected_traj_ids is not None:
            print(f"[Info] Selected trajectory indices: {selected_traj_ids}")
        if first_n_steps > 0:
            print(f"[Info] First-N filter per trajectory: {first_n_steps}")
        print(f"[Info] Total residual snapshots      : {total_snapshots_res}")
        if include_hom:
            print(f"[Info] Total homogenization snapshots: {total_snapshots_hom}")
    else:
        q_p_nodes_struct, mu_nodes_struct, cells_struct, grid_shape_struct = _load_stage7_structured_mesh_nodes(
            structured_mesh_dir=structured_mesh_dir,
            n_primary=n_primary,
        )
        q_p_bank_file = os.path.join(structured_mesh_dir, "q_p_train.npy")
        q_s_bank_file = os.path.join(structured_mesh_dir, "q_s_train.npy")
        mu_bank_file = os.path.join(structured_mesh_dir, "ls_targets_train.npy")
        if (not os.path.exists(q_p_bank_file)) or (not os.path.exists(q_s_bank_file)) or (not os.path.exists(mu_bank_file)):
            raise FileNotFoundError(
                "Structured mode requires q_p_train.npy, q_s_train.npy and ls_targets_train.npy "
                f"in {structured_mesh_dir}."
            )
        q_p_bank = np.asarray(np.load(q_p_bank_file), dtype=float)
        q_s_bank = np.asarray(np.load(q_s_bank_file), dtype=float)
        mu_bank = np.asarray(np.load(mu_bank_file), dtype=float)
        if q_p_bank.ndim != 2 or q_p_bank.shape[1] != n_primary:
            raise RuntimeError(
                f"Invalid q_p_train.npy shape {q_p_bank.shape}; expected (*,{n_primary})."
            )
        if q_s_bank.ndim != 2 or q_s_bank.shape[0] != q_p_bank.shape[0] or q_s_bank.shape[1] != n_secondary:
            raise RuntimeError(
                f"Invalid q_s_train.npy shape {q_s_bank.shape}; expected ({q_p_bank.shape[0]},{n_secondary})."
            )
        if mu_bank.ndim != 2 or mu_bank.shape[1] < 3 or mu_bank.shape[0] != q_p_bank.shape[0]:
            raise RuntimeError(
                f"Invalid ls_targets_train.npy shape {mu_bank.shape}; expected ({q_p_bank.shape[0]},>=3)."
            )
        mu_tree = cKDTree(mu_bank[:, :3])
        mu_dist, q_nn_idx = mu_tree.query(mu_nodes_struct[:, :3], k=1)
        q_p_nodes_eval = np.asarray(q_p_bank[np.asarray(q_nn_idx, dtype=np.int64)], dtype=float)
        q_s_nodes_eval_raw = np.asarray(q_s_bank[np.asarray(q_nn_idx, dtype=np.int64)], dtype=float)
        mu_nodes_eval = np.asarray(mu_bank[np.asarray(q_nn_idx, dtype=np.int64), :3], dtype=float)
        q_dist = np.linalg.norm(q_p_nodes_eval - q_p_nodes_struct, axis=1)
        total_snapshots_res = int(q_p_nodes_struct.shape[0])
        total_snapshots_hom = int(total_snapshots_res if include_hom else 0)
        print(f"[Info] Residual state source: {residual_state_source}")
        print(f"[Info] Structured mesh dir: {structured_mesh_dir}")
        print(f"[Info] Structured mesh nodes (residual): {total_snapshots_res}")
        print(f"[Info] Structured mesh cells: {int(cells_struct.shape[0])}")
        print(
            "[Info] Structured admissibility projection (mu-node -> nearest training snapshot): "
            f"mean_dist_mu={float(np.mean(mu_dist)):.3e}, max_dist_mu={float(np.max(mu_dist)):.3e}, "
            f"mean_dist_q={float(np.mean(q_dist)):.3e}, max_dist_q={float(np.max(q_dist)):.3e}"
        )
        if include_hom:
            print("[Info] Homogenization enabled: Stage12a will reuse structured residual states for hom blocks.")
        else:
            print("[Info] Homogenization sampling is disabled in Stage12a.")

    with open("ProjectParameters.json", "r", encoding="utf-8") as f:
        parameters = KM.Parameters(f.read())
    SetInputMeshFilename(parameters, "rve_geometry")

    mesh_base = parameters["modelers"][0]["parameters"]["input_filename"].GetString()
    mdpa_path = f"{StripMdpaExtension(mesh_base)}.mdpa"
    if os.path.exists(mdpa_path):
        material_parts = DetectMaterialSubModelParts(mdpa_path)
        parameters = ConfigureElementModelerForMaterialParts(parameters, material_parts)
        parameters["solver_settings"]["material_import_settings"]["materials_filename"].SetString(
            "StructuralMaterials.json"
        )
        print(f"[Info] Material parts: {material_parts}")

    model = KM.Model()
    sim = RVEHomogenizationDatasetGenerator(model, parameters)
    sim.Initialize()
    mp = sim._GetSolver().GetComputingModelPart()
    pi = mp.ProcessInfo

    n_dof, eq_map, ta = SetUpDofEquationIdsAndDisplacementAdaptor(mp)
    if int(n_dof) != n_total_dofs_ref:
        raise RuntimeError(f"DOF mismatch: runtime={n_dof}, expected={n_total_dofs_ref}.")
    if eq_map.shape == eq_map_ref.shape and not np.array_equal(eq_map, eq_map_ref):
        raise RuntimeError("eq_map mismatch between Stage 2 metadata and runtime model.")

    elements = list(mp.Elements)
    n_elem = len(elements)
    _, area_e = PrecomputeElementIntegrationWeights(elements)
    g2f = _build_free_map(n_dof, free_dofs)

    residual_cache, n_active_elems = _build_residual_projection_cache(elements, pi, g2f)
    rhs_vectors = [KM.Vector() for _ in range(n_elem)]
    q_block = np.zeros((n_primary, n_elem), dtype=float)
    c_block = np.zeros((6, n_elem), dtype=float)
    print(f"[Info] Residual projection active on {n_active_elems}/{n_elem} elements.")

    sim._InitializeDomainCenterIfNeeded(mp)
    x0c, y0c = float(sim._x0c), float(sim._y0c)
    dof_x = np.zeros(n_dof, dtype=float)
    dof_y = np.zeros(n_dof, dtype=float)
    is_x_dof = np.zeros(n_dof, dtype=bool)
    for i, node in enumerate(mp.Nodes):
        xr = float(node.X0) - x0c
        yr = float(node.Y0) - y0c
        idx_x = int(eq_map[i, 0])
        idx_y = int(eq_map[i, 1])
        if 0 <= idx_x < n_dof:
            dof_x[idx_x] = xr
            dof_y[idx_x] = yr
            is_x_dof[idx_x] = True
        if 0 <= idx_y < n_dof:
            dof_x[idx_y] = xr
            dof_y[idx_y] = yr
            is_x_dof[idx_y] = False

    x_free = dof_x[free_dofs]
    y_free = dof_y[free_dofs]
    is_x_free = is_x_dof[free_dofs]

    def _affine_free(e_macro):
        F = DeformationGradientFromGreenLagrange2D(np.asarray(e_macro, dtype=float))
        ux = (F[0, 0] - 1.0) * x_free + F[0, 1] * y_free
        uy = F[1, 0] * x_free + (F[1, 1] - 1.0) * y_free
        return np.where(is_x_free, ux, uy)

    Q_ecm, b_full, C_hom, b_hom = _allocate_memmaps(
        out_dir, n_primary, total_snapshots_res, total_snapshots_hom, n_elem, include_homogenization=include_hom
    )
    q_p_res = np.zeros((total_snapshots_res, n_primary), dtype=float)
    mu_res = np.zeros((total_snapshots_res, 3), dtype=float)
    sample_ids_res = np.zeros((total_snapshots_res, 2), dtype=np.int64)
    q_p_hom = np.zeros((total_snapshots_hom, n_primary), dtype=float)
    mu_hom = np.zeros((total_snapshots_hom, 3), dtype=float)
    sample_ids_hom = np.zeros((total_snapshots_hom, 2), dtype=np.int64)

    fit_rel_before = []
    fit_rel_after = []
    fit_iters = []
    q0_const, j0_const = _evaluate_gpr_origin_terms(gpr_model, n_primary)
    phi_p_eff_const = phi_p + phi_s @ j0_const
    w0_const = phi_s @ q0_const
    s_res_global = 0
    s_hom_global = 0
    structured_state_scale = None
    if residual_state_source == "trajectory_sampling":
        for task in all_tasks:
            traj_name = str(task["traj"])
            u_path = str(task["u_file"])
            e_path = str(task["e_file"])
            idx_res = np.unique(np.asarray(task["idx_res"], dtype=int).reshape(-1))
            idx_hom = np.unique(np.asarray(task["idx_hom"], dtype=int).reshape(-1))
            try:
                traj_id = int(traj_name.split("_")[-1])
            except Exception:
                traj_id = -1
            idx_union = np.union1d(idx_res, idx_hom) if include_hom else idx_res
            set_res = set(int(v) for v in idx_res.tolist())
            set_hom = set(int(v) for v in idx_hom.tolist())

            if include_hom:
                print(f"  > Processing {traj_name} (res={len(idx_res)} steps, hom={len(idx_hom)} steps)...")
            else:
                print(f"  > Processing {traj_name} (res={len(idx_res)} steps)...")
            u_all = np.load(u_path, mmap_mode="r")
            e_all = np.load(e_path, mmap_mode="r")

            for k in idx_union:
                ks = int(k)
                u_snap = _extract_snapshot_u(u_all, ks, int(n_dof))
                e_macro = np.asarray(e_all[ks, :3], dtype=float)
                u_free = u_snap[free_dofs]
                u_aff_free = _affine_free(e_macro)
                w_free = u_free - u_aff_free
                q_p_init = w_free @ phi_p

                SetDisplacementFromEquationVector(u_snap, eq_map, ta)
                UpdateCurrentCoordinatesFromDisplacement(mp)

                q_p_curr = None
                if ks in set_res:
                    if fit_mode == "gauss_newton":
                        q_p, _, j_qs_qp, rel0, relf, nit = _fit_qp_gauss_newton(
                            q_init=q_p_init,
                            w_free=w_free,
                            phi_p=phi_p,
                            phi_s=phi_s,
                            gpr_model=gpr_model,
                            n_primary=n_primary,
                            n_secondary=n_secondary,
                            q0_ref=q0_const,
                            j0_ref=j0_const,
                            max_iters=fit_max_iters,
                            rel_tol=fit_rel_tol,
                            l2_reg=fit_l2_reg,
                            step_tol=fit_step_tol,
                        )
                        fit_rel_before.append(float(rel0))
                        fit_rel_after.append(float(relf))
                        fit_iters.append(int(nit))
                    else:
                        q_p = q_p_init
                        q_s_raw, j_qs_qp_raw = _evaluate_gpr_qs_and_jac(q_p, gpr_model, n_primary)
                        _, j_qs_qp = _apply_manifold_origin_correction(
                            q_p, q_s_raw, j_qs_qp_raw, q0_const, j0_const
                        )
                        if j_qs_qp.shape != (n_secondary, n_primary):
                            raise RuntimeError(
                                f"Invalid GPR Jacobian shape {j_qs_qp.shape}, expected {(n_secondary, n_primary)}."
                            )
                    q_p_curr = np.asarray(q_p, dtype=float).reshape(-1)
                    q_p_res[s_res_global, :] = q_p_curr
                    mu_res[s_res_global, :] = e_macro[:3]
                    sample_ids_res[s_res_global, 0] = int(traj_id)
                    sample_ids_res[s_res_global, 1] = int(ks)
                    J_m = phi_p_eff_const + phi_s @ j_qs_qp

                    q_block.fill(0.0)
                    for i, elem in enumerate(elements):
                        cached = residual_cache[i]
                        if cached is None:
                            continue
                        rhs_pick, rows = cached
                        rhs = rhs_vectors[i]
                        elem.CalculateRightHandSide(rhs, pi)
                        rhs_arr = np.asarray(rhs, dtype=float)
                        if not np.isfinite(rhs_arr).all():
                            raise RuntimeError(
                                f"[Stage12a] Non-finite element RHS detected (trajectory={traj_name}, step={ks}, elem={i})."
                            )
                        q_col = J_m[rows, :].T @ rhs_arr[rhs_pick]
                        if not np.isfinite(q_col).all():
                            raise RuntimeError(
                                f"[Stage12a] Non-finite projected residual column (trajectory={traj_name}, step={ks}, elem={i})."
                            )
                        q_block[:, i] = q_col

                    r0, r1 = n_primary * s_res_global, n_primary * (s_res_global + 1)
                    if not np.isfinite(q_block).all():
                        raise RuntimeError(
                            f"[Stage12a] Non-finite residual block (trajectory={traj_name}, step={ks})."
                        )
                    Q_ecm[r0:r1, :] = q_block
                    b_full[r0:r1] = np.sum(q_block, axis=1)
                    if not np.isfinite(np.asarray(b_full[r0:r1], dtype=float)).all():
                        raise RuntimeError(
                            f"[Stage12a] Non-finite residual rhs block (trajectory={traj_name}, step={ks})."
                        )
                    s_res_global += 1

                if include_hom and ks in set_hom:
                    if q_p_curr is None:
                        if fit_mode == "gauss_newton":
                            q_p, _, _, rel0, relf, nit = _fit_qp_gauss_newton(
                                q_init=q_p_init,
                                w_free=w_free,
                                phi_p=phi_p,
                                phi_s=phi_s,
                                gpr_model=gpr_model,
                                n_primary=n_primary,
                                n_secondary=n_secondary,
                                q0_ref=q0_const,
                                j0_ref=j0_const,
                                max_iters=fit_max_iters,
                                rel_tol=fit_rel_tol,
                                l2_reg=fit_l2_reg,
                                step_tol=fit_step_tol,
                            )
                            fit_rel_before.append(float(rel0))
                            fit_rel_after.append(float(relf))
                            fit_iters.append(int(nit))
                        else:
                            q_p = q_p_init
                        q_p_curr = np.asarray(q_p, dtype=float).reshape(-1)
                    q_p_hom[s_hom_global, :] = q_p_curr
                    mu_hom[s_hom_global, :] = e_macro[:3]
                    sample_ids_hom[s_hom_global, 0] = int(traj_id)
                    sample_ids_hom[s_hom_global, 1] = int(ks)
                    _fill_hom_block_from_kratos(elements, mp, area_e, c_block)
                    h0, h1 = 6 * s_hom_global, 6 * (s_hom_global + 1)
                    C_hom[h0:h1, :] = c_block
                    b_hom[h0:h1] = np.sum(c_block, axis=1)
                    s_hom_global += 1
    else:
        if fit_mode == "gauss_newton":
            print("[Warn] residual-fit-mode=gauss_newton is ignored for stage7_structured_mesh source.")
        for inode in range(int(total_snapshots_res)):
            q_p_target = np.asarray(q_p_nodes_struct[inode, :], dtype=float).reshape(-1)
            q_p_curr = np.asarray(q_p_nodes_eval[inode, :], dtype=float).reshape(-1)
            q_s_state_raw = np.asarray(q_s_nodes_eval_raw[inode, :], dtype=float).reshape(-1)
            mu_target = np.asarray(mu_nodes_struct[inode, :], dtype=float).reshape(-1)
            e_macro = np.asarray(mu_nodes_eval[inode, :], dtype=float).reshape(-1)
            q_s_map_raw, j_qs_qp_raw = _evaluate_gpr_qs_and_jac(q_p_curr, gpr_model, n_primary)
            _, j_qs_qp = _apply_manifold_origin_correction(
                q_p_curr, q_s_map_raw, j_qs_qp_raw, q0_const, j0_const
            )
            if j_qs_qp.shape != (n_secondary, n_primary):
                raise RuntimeError(
                    f"Invalid GPR Jacobian shape {j_qs_qp.shape}, expected {(n_secondary, n_primary)}."
                )

            w_free = phi_p @ q_p_curr + phi_s @ q_s_state_raw
            u_aff_free = _affine_free(e_macro)
            u_free = u_aff_free + w_free
            u_snap = np.zeros(int(n_dof), dtype=float)
            u_snap[np.asarray(free_dofs, dtype=np.int64)] = u_free

            SetDisplacementFromEquationVector(u_snap, eq_map, ta)
            UpdateCurrentCoordinatesFromDisplacement(mp)

            q_p_res[s_res_global, :] = q_p_curr
            mu_res[s_res_global, :] = e_macro[:3]
            sample_ids_res[s_res_global, 0] = -7
            sample_ids_res[s_res_global, 1] = int(q_nn_idx[inode])
            J_m = phi_p_eff_const + phi_s @ j_qs_qp

            q_block.fill(0.0)
            for i, elem in enumerate(elements):
                cached = residual_cache[i]
                if cached is None:
                    continue
                rhs_pick, rows = cached
                rhs = rhs_vectors[i]
                try:
                    elem.CalculateRightHandSide(rhs, pi)
                except RuntimeError as exc:
                    raise RuntimeError(
                        "[Stage12a][structured] Inadmissible residual state "
                        f"(node={inode}, nn_id={int(q_nn_idx[inode])}, "
                        f"nn_dist_q={float(q_dist[inode]):.3e}, "
                        f"mu_target=[{mu_target[0]:.6e},{mu_target[1]:.6e},{mu_target[2]:.6e}], "
                        f"mu_eval=[{e_macro[0]:.6e},{e_macro[1]:.6e},{e_macro[2]:.6e}], "
                        f"q_target=[{q_p_target[0]:.6e},{q_p_target[1]:.6e},{q_p_target[2]:.6e}], "
                        f"q_eval=[{q_p_curr[0]:.6e},{q_p_curr[1]:.6e},{q_p_curr[2]:.6e}])."
                    ) from exc
                rhs_arr = np.asarray(rhs, dtype=float)
                if not np.isfinite(rhs_arr).all():
                    raise RuntimeError(
                        "[Stage12a][structured] Non-finite element RHS detected "
                        f"(node={inode}, nn_id={int(q_nn_idx[inode])}, elem={i})."
                    )
                q_col = J_m[rows, :].T @ rhs_arr[rhs_pick]
                if not np.isfinite(q_col).all():
                    raise RuntimeError(
                        "[Stage12a][structured] Non-finite projected residual column detected "
                        f"(node={inode}, nn_id={int(q_nn_idx[inode])}, elem={i})."
                    )
                q_block[:, i] = q_col

            r0, r1 = n_primary * s_res_global, n_primary * (s_res_global + 1)
            if not np.isfinite(q_block).all():
                raise RuntimeError(
                    "[Stage12a][structured] Non-finite residual block detected "
                    f"(node={inode}, nn_id={int(q_nn_idx[inode])})."
                )
            Q_ecm[r0:r1, :] = q_block
            b_full[r0:r1] = np.sum(q_block, axis=1)
            if not np.isfinite(np.asarray(b_full[r0:r1], dtype=float)).all():
                raise RuntimeError(
                    "[Stage12a][structured] Non-finite residual rhs block detected "
                    f"(node={inode}, nn_id={int(q_nn_idx[inode])})."
                )
            s_res_global += 1

            if include_hom:
                q_p_hom[s_hom_global, :] = q_p_curr
                mu_hom[s_hom_global, :] = e_macro[:3]
                sample_ids_hom[s_hom_global, 0] = -7
                sample_ids_hom[s_hom_global, 1] = int(q_nn_idx[inode])
                _fill_hom_block_from_kratos(elements, mp, area_e, c_block)
                h0, h1 = 6 * s_hom_global, 6 * (s_hom_global + 1)
                C_hom[h0:h1, :] = c_block
                b_hom[h0:h1] = np.sum(c_block, axis=1)
                s_hom_global += 1

    if s_res_global != total_snapshots_res:
        raise RuntimeError(
            f"Residual snapshot count mismatch: expected {total_snapshots_res}, got {s_res_global}"
        )
    if include_hom and s_hom_global != total_snapshots_hom:
        raise RuntimeError(
            f"Homogenization snapshot count mismatch: expected {total_snapshots_hom}, got {s_hom_global}"
        )

    Q_ecm.flush()
    b_full.flush()
    if C_hom is not None:
        C_hom.flush()
    if b_hom is not None:
        b_hom.flush()
    np.save(os.path.join(out_dir, "q_p_res.npy"), q_p_res)
    np.save(os.path.join(out_dir, "mu_res.npy"), mu_res)
    np.save(os.path.join(out_dir, "sample_ids_res.npy"), sample_ids_res)
    if residual_state_source == "stage7_structured_mesh":
        np.save(os.path.join(out_dir, "structured_mesh_cells_res.npy"), np.asarray(cells_struct, dtype=np.int64))
        np.save(
            os.path.join(out_dir, "structured_mesh_grid_shape_res.npy"),
            np.asarray(grid_shape_struct, dtype=np.int64),
        )
        np.save(os.path.join(out_dir, "q_p_res_target_structured.npy"), np.asarray(q_p_nodes_struct, dtype=float))
        np.save(os.path.join(out_dir, "mu_res_target_structured.npy"), np.asarray(mu_nodes_struct, dtype=float))
        np.save(os.path.join(out_dir, "mu_res_eval_structured.npy"), np.asarray(mu_nodes_eval, dtype=float))
        np.save(os.path.join(out_dir, "q_p_res_nn_dist_structured.npy"), np.asarray(q_dist, dtype=float))
        np.save(os.path.join(out_dir, "mu_res_nn_dist_structured.npy"), np.asarray(mu_dist, dtype=float))
        np.save(os.path.join(out_dir, "q_p_res_nn_index_structured.npy"), np.asarray(q_nn_idx, dtype=np.int64))
    else:
        for p in (
            "structured_mesh_cells_res.npy",
            "structured_mesh_grid_shape_res.npy",
            "q_p_res_target_structured.npy",
            "mu_res_target_structured.npy",
            "mu_res_eval_structured.npy",
            "q_p_res_nn_dist_structured.npy",
            "mu_res_nn_dist_structured.npy",
            "q_p_res_nn_index_structured.npy",
        ):
            fpath = os.path.join(out_dir, p)
            if os.path.exists(fpath):
                os.remove(fpath)
    if include_hom:
        np.save(os.path.join(out_dir, "q_p_hom.npy"), q_p_hom)
        np.save(os.path.join(out_dir, "mu_hom.npy"), mu_hom)
        np.save(os.path.join(out_dir, "sample_ids_hom.npy"), sample_ids_hom)
    else:
        for p in ("q_p_hom.npy", "mu_hom.npy", "sample_ids_hom.npy"):
            fpath = os.path.join(out_dir, p)
            if os.path.exists(fpath):
                os.remove(fpath)

    if fit_mode == "gauss_newton" and fit_rel_after:
        rb = np.asarray(fit_rel_before, dtype=float)
        ra = np.asarray(fit_rel_after, dtype=float)
        it = np.asarray(fit_iters, dtype=float)
        print(
            "[Fit] relative reconstruction residual: "
            f"before mean={np.mean(rb):.3e}, after mean={np.mean(ra):.3e}, "
            f"after max={np.max(ra):.3e}, iters mean={np.mean(it):.2f}, iters max={np.max(it):.0f}"
        )

    a0_ref = float(np.sum(area_e))
    np.savez(
        os.path.join(out_dir, "meta.npz"),
        nq=n_primary,
        n_primary=n_primary,
        n_secondary=n_secondary,
        n_elem=n_elem,
        N_s_res=total_snapshots_res,
        N_s_hom=total_snapshots_hom if include_hom else 0,
        snapshot_percent_res=pct_res,
        snapshot_percent_hom=pct_hom if include_hom else 0.0,
        sampling_mode=np.array([args.sampling_mode]),
        residual_state_source=np.array([residual_state_source]),
        structured_admissible_projection=np.array(
            [1 if residual_state_source == "stage7_structured_mesh" else 0], dtype=np.int64
        ),
        structured_q_nn_mean_dist=np.array(
            [float(np.mean(q_dist)) if residual_state_source == "stage7_structured_mesh" else np.nan], dtype=float
        ),
        structured_q_nn_max_dist=np.array(
            [float(np.max(q_dist)) if residual_state_source == "stage7_structured_mesh" else np.nan], dtype=float
        ),
        structured_mu_nn_mean_dist=np.array(
            [float(np.mean(mu_dist)) if residual_state_source == "stage7_structured_mesh" else np.nan], dtype=float
        ),
        structured_mu_nn_max_dist=np.array(
            [float(np.max(mu_dist)) if residual_state_source == "stage7_structured_mesh" else np.nan], dtype=float
        ),
        param_aware_time_weight=np.array([args.param_aware_time_weight]),
        snapshots_dir=np.array([snapshots_dir]),
        structured_mesh_dir=np.array([structured_mesh_dir]),
        basis_dir=np.array([basis_dir]),
        gpr_data_dir=np.array([gpr_dir]),
        first_n_steps=np.array([first_n_steps], dtype=np.int64),
        trajectory_indices=np.array(
            selected_traj_ids if (selected_traj_ids is not None and residual_state_source == "trajectory_sampling") else [],
            dtype=np.int64,
        ),
        include_macro_strain_input=np.array([1 if include_macro else 0], dtype=np.int64),
        A_total=a0_ref,
        A0_ref=np.array([a0_ref], dtype=float),
        hom_reference_measure=np.array([a0_ref], dtype=float),
        residual_fit_mode=np.array([fit_mode]),
        fit_max_iters=np.array([fit_max_iters], dtype=np.int64),
        fit_rel_tol=np.array([fit_rel_tol], dtype=float),
        fit_l2_reg=np.array([fit_l2_reg], dtype=float),
        fit_step_tol=np.array([fit_step_tol], dtype=float),
        fit_rel_before_mean=np.array([float(np.mean(fit_rel_before)) if fit_rel_before else np.nan], dtype=float),
        fit_rel_after_mean=np.array([float(np.mean(fit_rel_after)) if fit_rel_after else np.nan], dtype=float),
        fit_rel_after_max=np.array([float(np.max(fit_rel_after)) if fit_rel_after else np.nan], dtype=float),
        fit_iters_mean=np.array([float(np.mean(fit_iters)) if fit_iters else np.nan], dtype=float),
        fit_iters_max=np.array([float(np.max(fit_iters)) if fit_iters else np.nan], dtype=float),
        manifold_origin_correction=np.array([1], dtype=np.int64),
        disable_homogenization=np.array([1 if disable_hom else 0], dtype=np.int64),
        row_ordering_res=np.array(["sample-major-qrow"]),
        row_ordering_hom=np.array(["sample-major-6row" if include_hom else "disabled"]),
    )
    with open(os.path.join(out_dir, "reference_measure_A0.txt"), "w", encoding="utf-8") as f:
        f.write(f"{a0_ref:.16e}\n")

    sim.Finalize()

    print("\n[DONE] Stage 12a MAW-ECM dataset generation complete.")
    print(f"      - Q_ecm shape: {Q_ecm.shape}")
    if include_hom:
        print(f"      - C_hom shape: {C_hom.shape}")
    else:
        print("      - C_hom: disabled")
    print(f"      - q_p_res shape: {q_p_res.shape}")
    if include_hom:
        print(f"      - q_p_hom shape: {q_p_hom.shape}")
    else:
        print("      - q_p_hom: disabled")
    print(f"      - Datasets saved to: {out_dir}")


if __name__ == "__main__":
    main()
