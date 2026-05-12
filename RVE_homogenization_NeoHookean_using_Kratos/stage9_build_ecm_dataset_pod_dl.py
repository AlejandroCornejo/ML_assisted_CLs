#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage 9a: Build ECM dataset for HPROM-POD-DL.

Stage-5-style dataset builder with independent sampling for:
  - residual projection (Q_ecm, b_full)
  - homogenization targets (C_hom, b_hom)

Residual projection uses the POD-DL manifold tangent:
  J_m = phi_q * d(q_hat)/d(z)
with q_hat(z) = decode(z) - q_ref.
"""

import os
import sys
import argparse
import numpy as np
import torch

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
from pod_dl_manifold_model import load_pod_dl_model
from ecm_sampling_utils import get_stratified_indices, get_param_aware_indices


SNAPSHOTS_DIR = "stage_1_training_set_fom"
BASIS_DIR = "stage_2_pod_rve"
POD_DL_DIR = "stage_7_pod_dl_data"
OUT_DIR = "stage_9_ecm_dataset_pod_dl"
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


def _is_finite(arr):
    return bool(np.all(np.isfinite(np.asarray(arr))))


def _evaluate_qhat_and_jac(z_state, pod_dl_model, q_ref_t):
    with torch.no_grad():
        q_map = pod_dl_model.decode_from_latent(z_state.unsqueeze(0)).reshape(-1)
        q_hat = q_map - q_ref_t
    q_hat_np = q_hat.detach().cpu().numpy().reshape(-1).astype(float)

    with torch.enable_grad():
        z_in = z_state.detach().clone().requires_grad_(True)

        def _decode_only(z_vec):
            return pod_dl_model.decode_from_latent(z_vec.unsqueeze(0)).reshape(-1)

        j_q = torch.autograd.functional.jacobian(_decode_only, z_in)
    j_q_np = j_q.detach().cpu().numpy()
    if j_q_np.ndim == 1:
        j_q_np = j_q_np.reshape(-1, 1)
    return q_hat_np, np.asarray(j_q_np, dtype=float)


def _encode_z_from_qhat(q_hat_target, pod_dl_model, q_ref_np, device):
    q_raw = np.asarray(q_hat_target, dtype=np.float32).reshape(-1) + np.asarray(q_ref_np, dtype=np.float32).reshape(-1)
    with torch.no_grad():
        q_t = torch.from_numpy(q_raw).unsqueeze(0).to(device)
        z_t = pod_dl_model.encode(q_t).reshape(-1)
    return z_t.detach().clone()


def _fit_z_gauss_newton(
    z_init,
    w_free,
    phi_q,
    pod_dl_model,
    q_ref_t,
    n_q,
    n_latent,
    max_iters,
    rel_tol,
    l2_reg,
    step_tol,
):
    z_state = z_init.detach().clone()
    w_target = np.asarray(w_free, dtype=float).reshape(-1)
    target_norm = max(np.linalg.norm(w_target), 1e-30)

    q_hat, j_q = _evaluate_qhat_and_jac(z_state, pod_dl_model, q_ref_t)
    if j_q.shape != (n_q, n_latent):
        raise RuntimeError(f"Invalid decoder Jacobian shape {j_q.shape}, expected {(n_q, n_latent)}.")

    w_rec = phi_q @ q_hat
    rel0 = float(np.linalg.norm(w_rec - w_target) / target_norm)
    rel = rel0
    n_it = 0

    for it in range(int(max_iters)):
        if rel <= float(rel_tol):
            break

        J_m = phi_q @ j_q
        res = w_rec - w_target
        jtj = J_m.T @ J_m
        if l2_reg > 0.0:
            jtj = jtj + float(l2_reg) * np.eye(jtj.shape[0], dtype=float)
        rhs = J_m.T @ res
        try:
            dz = np.linalg.solve(jtj, rhs)
        except np.linalg.LinAlgError:
            dz, *_ = np.linalg.lstsq(jtj, rhs, rcond=None)

        with torch.no_grad():
            dz_t = torch.from_numpy(np.asarray(dz, dtype=np.float32)).to(z_state.device)
            z_state = z_state - dz_t

        n_it = it + 1
        if np.linalg.norm(dz) <= float(step_tol) * max(np.linalg.norm(z_state.detach().cpu().numpy()), 1.0):
            q_hat, j_q = _evaluate_qhat_and_jac(z_state, pod_dl_model, q_ref_t)
            w_rec = phi_q @ q_hat
            rel = float(np.linalg.norm(w_rec - w_target) / target_norm)
            break

        q_hat, j_q = _evaluate_qhat_and_jac(z_state, pod_dl_model, q_ref_t)
        if j_q.shape != (n_q, n_latent):
            raise RuntimeError(f"Invalid decoder Jacobian shape {j_q.shape}, expected {(n_q, n_latent)}.")

        w_rec = phi_q @ q_hat
        rel = float(np.linalg.norm(w_rec - w_target) / target_norm)

    return z_state, q_hat, j_q, rel0, rel, int(n_it)


def _parse_args():
    p = argparse.ArgumentParser(
        description="Stage 9a-POD-DL: build ECM dataset with separate residual/hom sampling."
    )
    p.add_argument("--snapshots-dir", type=str, default=SNAPSHOTS_DIR)
    p.add_argument("--basis-dir", type=str, default=BASIS_DIR)
    p.add_argument("--pod-dl-dir", type=str, default=POD_DL_DIR)
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
    print("--- Stage 9a-POD-DL: Building ECM Dataset ---")

    pct_res = float(args.snapshot_percent_res)
    pct_hom = float(args.snapshot_percent_hom)
    if pct_res <= 0.0 or pct_hom <= 0.0:
        raise ValueError("snapshot percentages must be > 0.")
    if pct_res > 100.0 or pct_hom > 100.0:
        raise ValueError("snapshot percentages must be <= 100.")

    snapshots_dir = str(args.snapshots_dir)
    basis_dir = str(args.basis_dir)
    pod_dl_dir = str(args.pod_dl_dir)
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pod_dl_model, checkpoint, _ = load_pod_dl_model(model_dir=pod_dl_dir, device=device)

    basis_full = np.asarray(np.load(os.path.join(basis_dir, "pod_basis_free.npy")), dtype=float)
    free_dofs = np.asarray(np.load(os.path.join(basis_dir, "free_dofs.npy")), dtype=np.int64)
    dir_dofs = np.asarray(np.load(os.path.join(basis_dir, "dirichlet_dofs.npy")), dtype=np.int64)
    eq_map_ref = np.asarray(np.load(os.path.join(basis_dir, "eq_map.npy")), dtype=np.int64)

    n_q = int(checkpoint["q_dim"])
    n_latent = int(checkpoint["latent_dim"])
    if basis_full.shape[1] < n_q:
        raise ValueError(
            f"Basis has {basis_full.shape[1]} columns but POD-DL model expects q_dim={n_q}."
        )
    phi_q = np.asarray(basis_full[:, :n_q], dtype=float)
    n_free = int(phi_q.shape[0])

    if int(len(free_dofs)) != n_free:
        raise ValueError(f"phi_q rows ({n_free}) and free_dofs length ({len(free_dofs)}) mismatch.")

    with torch.no_grad():
        q_zero = torch.zeros((1, n_q), dtype=torch.float32, device=device)
        z_ref = pod_dl_model.encode(q_zero).reshape(-1)
        q_ref_t = pod_dl_model.decode_from_latent(z_ref.unsqueeze(0)).reshape(-1)
        q_ref_np = q_ref_t.detach().cpu().numpy().reshape(-1).astype(float)

    n_total_dofs_ref = int(len(free_dofs) + len(dir_dofs))
    print(
        f"[Info] n_q={n_q}, latent_dim={n_latent}, "
        f"||decode(encode(0))||={float(torch.norm(q_ref_t).cpu().item()):.3e}"
    )
    print(
        f"[Info] residual fit mode={fit_mode}, fit_max_iters={fit_max_iters}, "
        f"fit_rel_tol={fit_rel_tol:.2e}, fit_l2_reg={fit_l2_reg:.2e}, fit_step_tol={fit_step_tol:.2e}"
    )

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
    q_block = np.zeros((n_latent, n_elem), dtype=float)
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
        out_dir, n_latent, total_snapshots_res, total_snapshots_hom, n_elem
    )

    fit_rel_before = []
    fit_rel_after = []
    fit_iters = []

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

                q_init = w_free @ phi_q
                z_init = _encode_z_from_qhat(q_init, pod_dl_model, q_ref_np, device)

                if fit_mode == "gauss_newton":
                    z_fit, _, j_q, rel0, relf, nit = _fit_z_gauss_newton(
                        z_init=z_init,
                        w_free=w_free,
                        phi_q=phi_q,
                        pod_dl_model=pod_dl_model,
                        q_ref_t=q_ref_t,
                        n_q=n_q,
                        n_latent=n_latent,
                        max_iters=fit_max_iters,
                        rel_tol=fit_rel_tol,
                        l2_reg=fit_l2_reg,
                        step_tol=fit_step_tol,
                    )
                    fit_rel_before.append(float(rel0))
                    fit_rel_after.append(float(relf))
                    fit_iters.append(int(nit))
                else:
                    z_fit = z_init
                    _, j_q = _evaluate_qhat_and_jac(z_fit, pod_dl_model, q_ref_t)

                if j_q.shape != (n_q, n_latent):
                    raise RuntimeError(
                        f"Invalid decoder Jacobian shape {j_q.shape}, expected {(n_q, n_latent)}."
                    )

                J_m = phi_q @ j_q
                if not _is_finite(J_m):
                    raise RuntimeError("Non-finite POD-DL tangent detected while building Q_ecm.")

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

                r0, r1 = n_latent * s_res_global, n_latent * (s_res_global + 1)
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
        nq=n_latent,
        n_q=n_q,
        n_latent=n_latent,
        n_elem=n_elem,
        N_s_res=total_snapshots_res,
        N_s_hom=total_snapshots_hom,
        snapshot_percent_res=pct_res,
        snapshot_percent_hom=pct_hom,
        sampling_mode=np.array([args.sampling_mode]),
        param_aware_time_weight=np.array([args.param_aware_time_weight]),
        snapshots_dir=np.array([snapshots_dir]),
        basis_dir=np.array([basis_dir]),
        pod_dl_data_dir=np.array([pod_dl_dir]),
        first_n_steps=np.array([first_n_steps], dtype=np.int64),
        trajectory_indices=np.array(selected_traj_ids if selected_traj_ids is not None else [], dtype=np.int64),
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
    )

    with open(os.path.join(out_dir, "reference_measure_A0.txt"), "w", encoding="utf-8") as f:
        f.write(f"{a0_ref:.16e}\n")

    sim.Finalize()

    print("\n[DONE] Dataset generation complete.")
    print(f"      - Q_ecm shape: {Q_ecm.shape}")
    print(f"      - C_hom shape: {C_hom.shape}")
    print(f"      - Datasets saved to: {out_dir}")


if __name__ == "__main__":
    main()
