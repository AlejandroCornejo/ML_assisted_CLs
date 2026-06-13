#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage 9a: Build ECM dataset for HPROM-GPR.

Stage-5-style dataset builder with independent sampling for:
  - residual projection (Q_ecm, b_full)
  - homogenization targets (C_hom, b_hom)

Residual projection uses the GPR-manifold tangent:
  J_m = (phi_m*A_m + phi_s*J0) + phi_s * d(q_s_corr)/d(q_m)
"""

import os
import sys
import argparse
import numpy as np

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
GPR_DIR = "stage_7_gpr_data"
OUT_DIR = "stage_9_ecm_dataset_gpr"
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


def _build_free_map(n_dof, free_dofs):
    g2f = -np.ones(int(n_dof), dtype=int)
    for i, gdof in enumerate(np.asarray(free_dofs, dtype=np.int64)):
        g2f[int(gdof)] = int(i)
    return g2f


def _allocate_memmaps(out_dir, nq, ns_res, ns_hom, n_elem):
    os.makedirs(out_dir, exist_ok=True)
    q_path = os.path.join(out_dir, "Q_ecm.dat")
    b_path = os.path.join(out_dir, "b_full.dat")
    c_path = os.path.join(out_dir, "C_hom.dat")
    bh_path = os.path.join(out_dir, "b_hom.dat")

    for p in (q_path, b_path, c_path, bh_path):
        if os.path.exists(p):
            os.remove(p)

    q_ecm = np.memmap(q_path, dtype="float64", mode="w+", shape=(int(nq) * int(ns_res), int(n_elem)))
    b_full = np.memmap(b_path, dtype="float64", mode="w+", shape=(int(nq) * int(ns_res),))
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
    a_m,
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
    phi_master = np.asarray(phi_p, dtype=float) @ np.asarray(a_m, dtype=float)
    phi_p_eff = phi_master + np.asarray(phi_s, dtype=float) @ np.asarray(j0_ref, dtype=float)
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
        description="Stage 9a-GPR: build ECM dataset with separate residual/hom sampling."
    )
    p.add_argument("--snapshots-dir", type=str, default=SNAPSHOTS_DIR)
    p.add_argument("--basis-dir", type=str, default=BASIS_DIR)
    p.add_argument("--gpr-dir", type=str, default=GPR_DIR)
    p.add_argument("--out-dir", type=str, default=OUT_DIR)
    p.add_argument("--snapshot-percent-res", type=float, default=SNAPSHOT_PERCENT_RES)
    p.add_argument("--snapshot-percent-hom", type=float, default=SNAPSHOT_PERCENT_HOM)
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
    print("--- Stage 9a-GPR: Building ECM Dataset ---")

    pct_res = float(args.snapshot_percent_res)
    pct_hom = float(args.snapshot_percent_hom)
    if pct_res <= 0.0 or pct_hom <= 0.0:
        raise ValueError("snapshot percentages must be > 0.")
    if pct_res > 100.0 or pct_hom > 100.0:
        raise ValueError("snapshot percentages must be <= 100.")

    snapshots_dir = str(args.snapshots_dir)
    basis_dir = str(args.basis_dir)
    gpr_dir = str(args.gpr_dir)
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

    phi_m_path = os.path.join(gpr_dir, "phi_m.npy")
    if not os.path.exists(phi_m_path):
        raise FileNotFoundError(
            f"{phi_m_path}. Rebuild Stage7a/7b with the validated Joaquin-style LS split."
        )
    phi_p = np.asarray(np.load(phi_m_path), dtype=float)
    phi_s = np.asarray(np.load(os.path.join(gpr_dir, "phi_s.npy")), dtype=float)
    a_m_path = os.path.join(gpr_dir, "A_m.npy")
    if not os.path.exists(a_m_path):
        raise FileNotFoundError(
            f"{a_m_path}. Rebuild Stage7a/7b with the validated Joaquin-style LS split."
        )
    a_m = np.asarray(np.load(a_m_path), dtype=float)
    free_dofs = np.asarray(np.load(os.path.join(basis_dir, "free_dofs.npy")), dtype=np.int64)
    dir_dofs = np.asarray(np.load(os.path.join(basis_dir, "dirichlet_dofs.npy")), dtype=np.int64)
    eq_map_ref = np.asarray(np.load(os.path.join(basis_dir, "eq_map.npy")), dtype=np.int64)
    gpr_model = load_sparse_gp_model(os.path.join(gpr_dir, "sparse_gp_model.npz"))

    n_primary = int(phi_p.shape[1])
    n_secondary = int(phi_s.shape[1])
    n_free = int(phi_p.shape[0])
    if int(phi_s.shape[0]) != n_free:
        raise ValueError("phi_p and phi_s row count mismatch.")
    if a_m.shape != (n_primary, n_primary):
        raise ValueError(
            f"A_m shape mismatch: got {a_m.shape}, expected {(n_primary, n_primary)}."
        )
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
    print("[Info] Manifold correction active in Stage 9a-GPR: N(0)=0 and J(0)=0.")
    trajectories = sorted([d for d in os.listdir(snapshots_dir) if d.startswith("trajectory_")])
    if selected_traj_ids is not None:
        selected_names = {f"trajectory_{i}" for i in selected_traj_ids}
        trajectories = [t for t in trajectories if t in selected_names]
    if not trajectories:
        raise FileNotFoundError(
            f"No trajectory folders found in {snapshots_dir} for selection={selected_traj_ids}."
        )

    total_snapshots_res = 0
    total_snapshots_hom = 0
    all_tasks = []
    frac_res = pct_res / 100.0
    frac_hom = pct_hom / 100.0

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
        n_pick_hom = int(np.ceil(frac_hom * n_steps))
        idx_res = _pick_snapshot_indices(
            e_meta,
            n_steps=n_steps,
            n_pick=n_pick_res,
            mode=args.sampling_mode,
            seed=int(args.seed) + 2 * len(all_tasks),
            time_weight=args.param_aware_time_weight,
        )
        idx_hom = _pick_snapshot_indices(
            e_meta,
            n_steps=n_steps,
            n_pick=n_pick_hom,
            mode=args.sampling_mode,
            seed=int(args.seed) + 2 * len(all_tasks) + 1,
            time_weight=args.param_aware_time_weight,
        )

        total_snapshots_res += int(len(idx_res))
        total_snapshots_hom += int(len(idx_hom))
        all_tasks.append((traj, u_file, e_file, idx_res, idx_hom))

    if total_snapshots_res <= 0:
        raise RuntimeError("No residual snapshots selected.")
    if total_snapshots_hom <= 0:
        raise RuntimeError("No homogenization snapshots selected.")

    print(f"[Info] Sampling mode: {args.sampling_mode}")
    print(f"[Info] Target residual snapshots      : {pct_res}% across {len(all_tasks)} trajectories.")
    print(f"[Info] Target homogenization snapshots: {pct_hom}% across {len(all_tasks)} trajectories.")
    if selected_traj_ids is not None:
        print(f"[Info] Selected trajectory indices: {selected_traj_ids}")
    if first_n_steps > 0:
        print(f"[Info] First-N filter per trajectory: {first_n_steps}")
    print(f"[Info] Total residual snapshots      : {total_snapshots_res}")
    print(f"[Info] Total homogenization snapshots: {total_snapshots_hom}")

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
        out_dir, n_primary, total_snapshots_res, total_snapshots_hom, n_elem
    )

    fit_rel_before = []
    fit_rel_after = []
    fit_iters = []
    q0_const, j0_const = _evaluate_gpr_origin_terms(gpr_model, n_primary)
    phi_master = phi_p @ a_m
    phi_p_eff_const = phi_master + phi_s @ j0_const
    s_res_global = 0
    s_hom_global = 0
    for traj_name, u_path, e_path, idx_res, idx_hom in all_tasks:
        idx_res = np.unique(np.asarray(idx_res, dtype=int).reshape(-1))
        idx_hom = np.unique(np.asarray(idx_hom, dtype=int).reshape(-1))
        idx_union = np.union1d(idx_res, idx_hom)
        set_res = set(int(v) for v in idx_res.tolist())
        set_hom = set(int(v) for v in idx_hom.tolist())

        print(f"  > Processing {traj_name} (res={len(idx_res)} steps, hom={len(idx_hom)} steps)...")
        u_all = np.load(u_path, mmap_mode="r")
        e_all = np.load(e_path, mmap_mode="r")

        for k in idx_union:
            ks = int(k)
            u_snap = _extract_snapshot_u(u_all, ks, int(n_dof))
            e_macro = np.asarray(e_all[ks, :3], dtype=float)

            SetDisplacementFromEquationVector(u_snap, eq_map, ta)
            UpdateCurrentCoordinatesFromDisplacement(mp)

            if ks in set_res:
                u_free = u_snap[free_dofs]
                u_aff_free = _affine_free(e_macro)
                w_free = u_free - u_aff_free
                q_master_init = w_free @ phi_p
                try:
                    q_p_init = np.linalg.solve(a_m, q_master_init)
                except np.linalg.LinAlgError:
                    q_p_init = np.linalg.lstsq(a_m, q_master_init, rcond=None)[0]
                if fit_mode == "gauss_newton":
                    q_p, _, j_qs_qp, rel0, relf, nit = _fit_qp_gauss_newton(
                        q_init=q_p_init,
                        w_free=w_free,
                        phi_p=phi_p,
                        a_m=a_m,
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
                    q_block[:, i] = J_m[rows, :].T @ rhs_arr[rhs_pick]

                r0, r1 = n_primary * s_res_global, n_primary * (s_res_global + 1)
                Q_ecm[r0:r1, :] = q_block
                b_full[r0:r1] = np.sum(q_block, axis=1)
                s_res_global += 1

            if ks in set_hom:
                _fill_hom_block_from_kratos(elements, mp, area_e, c_block)
                h0, h1 = 6 * s_hom_global, 6 * (s_hom_global + 1)
                C_hom[h0:h1, :] = c_block
                b_hom[h0:h1] = np.sum(c_block, axis=1)
                s_hom_global += 1

    if s_res_global != total_snapshots_res:
        raise RuntimeError(
            f"Residual snapshot count mismatch: expected {total_snapshots_res}, got {s_res_global}"
        )
    if s_hom_global != total_snapshots_hom:
        raise RuntimeError(
            f"Homogenization snapshot count mismatch: expected {total_snapshots_hom}, got {s_hom_global}"
        )

    Q_ecm.flush()
    b_full.flush()
    C_hom.flush()
    b_hom.flush()

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
        N_s_hom=total_snapshots_hom,
        snapshot_percent_res=pct_res,
        snapshot_percent_hom=pct_hom,
        sampling_mode=np.array([args.sampling_mode]),
        param_aware_time_weight=np.array([args.param_aware_time_weight]),
        snapshots_dir=np.array([snapshots_dir]),
        basis_dir=np.array([basis_dir]),
        gpr_data_dir=np.array([gpr_dir]),
        first_n_steps=np.array([first_n_steps], dtype=np.int64),
        trajectory_indices=np.array(selected_traj_ids if selected_traj_ids is not None else [], dtype=np.int64),
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
    )
    with open(os.path.join(out_dir, "reference_measure_A0.txt"), "w", encoding="utf-8") as f:
        f.write(f"{a0_ref:.16e}\\n")

    sim.Finalize()

    print("\\n[DONE] Dataset generation complete.")
    print(f"      - Q_ecm shape: {Q_ecm.shape}")
    print(f"      - C_hom shape: {C_hom.shape}")
    print(f"      - Datasets saved to: {out_dir}")


if __name__ == "__main__":
    main()
