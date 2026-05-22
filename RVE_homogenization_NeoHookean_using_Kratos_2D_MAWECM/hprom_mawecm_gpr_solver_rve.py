#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""HPROM-MAWECM-GPR solver (2D-MAWECM).

Residual hyper-reduction:
- fixed residual support Z_res from MAW offline
- adaptive residual weights w_res(q_state) via RBF (MAW)

State update uses Joaquin-style master/slave decoder:
    q_s = N_gpr(q_m)
    u = u_aff + Phi_m (A_m q_m) + Phi_s q_s
"""

import os
import sys
import time
from typing import Dict

import numpy as np

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
    InitializeNonLinearIteration,
    FinalizeNonLinearIteration,
    BuildDynamicSegmentSteps,
    CalculateHomogenizedFromAssemblerWithElementWeights,
    VectorizedAssembler,
    REFERENCE_STEPS_FOR_UNIT_AMPLITUDE,
    MIN_STEPS_PER_SEGMENT,
    setup_kratos_parameters,
)
from prom_gpr_solver_rve import (
    LoadPromGprModel,
    _recover_mu_from_E,
    _initial_qm_from_mu,
)
from mawecm_rbf_weights import eval_mawecm_rbf, eval_mawecm_rbf_with_jacobian
from sparse_gp_manifold_model import evaluate_sparse_gp_map_and_jacobian_qp


_W_TOL = 1.0e-14


def _build_free_dof_index_map(n_dof: int, free_dofs: np.ndarray) -> np.ndarray:
    idx = -np.ones(int(n_dof), dtype=np.int64)
    f = np.asarray(free_dofs, dtype=np.int64).reshape(-1)
    idx[f] = np.arange(f.size, dtype=np.int64)
    return idx


def _prepare_unit_rhs_scatter(assembler_unit: VectorizedAssembler, free_dof_index_map: np.ndarray):
    ne = int(assembler_unit.n_elems)
    ndl = int(assembler_unit.n_local_dof)
    eq_ids = np.asarray(assembler_unit.local_eq_ids, dtype=np.int64).reshape(ne, ndl)
    free_ids = free_dof_index_map[eq_ids]  # (ne, ndl), -1 for constrained
    valid = free_ids >= 0
    if not np.any(valid):
        return {
            "ne": ne,
            "ndl": ndl,
            "valid": valid,
            "row_idx": np.zeros(0, dtype=np.int64),
            "col_idx": np.zeros(0, dtype=np.int64),
        }
    row_idx = free_ids[valid]
    col_idx = np.broadcast_to(np.arange(ne, dtype=np.int64)[:, None], (ne, ndl))[valid]
    return {
        "ne": ne,
        "ndl": ndl,
        "valid": valid,
        "row_idx": row_idx,
        "col_idx": col_idx,
    }


def _scatter_rhs_loc_by_element_free(rhs_loc: np.ndarray, n_free: int, scatter_cache):
    """
    Scatter local per-element RHS contributions to free-DOF stacked columns.

    Parameters
    ----------
    rhs_loc : ndarray, shape (n_elem_res, n_local_dof)
        Local RHS contribution per element and local dof.
    """
    ne = int(scatter_cache["ne"])
    ndl = int(scatter_cache["ndl"])
    valid = scatter_cache["valid"]
    row_idx = scatter_cache["row_idx"]
    col_idx = scatter_cache["col_idx"]

    rhs_loc = np.asarray(rhs_loc, dtype=float).reshape(ne, ndl)
    if row_idx.size == 0:
        return np.zeros((int(n_free), ne), dtype=float)

    vals = rhs_loc[valid]
    out = np.zeros((int(n_free), ne), dtype=float)
    np.add.at(out, (row_idx, col_idx), vals)
    return out


def _assemble_unit_rhs_by_element_free(
    assembler_unit: VectorizedAssembler,
    free_dof_index_map: np.ndarray,
    n_free: int,
    scatter_cache=None,
    rhs_loc=None,
):
    """
    Build per-element unit-weight RHS contributions restricted to free DOFs.

    Returns
    -------
    R_free_e : ndarray, shape (n_free, n_elem_res)
        Column e stores the free-DOF RHS contribution of residual element e
        for unit MAW weight.
    """
    if scatter_cache is None:
        scatter_cache = _prepare_unit_rhs_scatter(assembler_unit, free_dof_index_map)
    if rhs_loc is None:
        ne = int(scatter_cache["ne"])
        ndl = int(scatter_cache["ndl"])
        rhs_loc = -np.asarray(assembler_unit._f_int, dtype=float).reshape(ne, ndl)
    return _scatter_rhs_loc_by_element_free(rhs_loc=rhs_loc, n_free=n_free, scatter_cache=scatter_cache)


def _stable_unique_preserve_order(idx: np.ndarray) -> np.ndarray:
    seen = set()
    out = []
    for v in np.asarray(idx, dtype=np.int64).reshape(-1).tolist():
        iv = int(v)
        if iv in seen:
            continue
        seen.add(iv)
        out.append(iv)
    return np.asarray(out, dtype=np.int64)


def _map_support_indices_to_current_mesh(idx_raw, n_elem: int, full_to_local, label: str):
    idx_raw = np.asarray(idx_raw, dtype=np.int64).reshape(-1)
    if idx_raw.size == 0:
        return idx_raw

    if full_to_local is None:
        if np.any(idx_raw < 0) or np.any(idx_raw >= int(n_elem)):
            raise RuntimeError(f"{label} contains out-of-range element ids for current mesh.")
        return _stable_unique_preserve_order(idx_raw)

    mapped = []
    missing = []
    for fid in idx_raw.tolist():
        key = int(fid)
        if key in full_to_local:
            mapped.append(full_to_local[key])
        else:
            missing.append(key)

    if missing:
        # Backward compatibility: ids may already be local.
        idx_try = np.asarray(idx_raw, dtype=np.int64)
        if np.any(idx_try < 0) or np.any(idx_try >= int(n_elem)):
            raise RuntimeError(
                f"Could not map {label} from full mesh to HROM local indices. "
                f"Missing full ids count={len(missing)} (example={missing[:8]})."
            )
        return _stable_unique_preserve_order(idx_try)
    return _stable_unique_preserve_order(np.asarray(mapped, dtype=np.int64))


def _extract_hrom_aligned_maw_arrays(mawecm_data: Dict[str, object], n_elem: int):
    """
    Return MAW residual support and homogenization weights aligned with the
    currently loaded mesh.

    Works with:
    - full mesh MAW data (maw_res_z_support + w_eps_full/w_sig_full)
    - HROM-aligned data carrying hrom_* metadata and w_*_hrom.
    """
    z_res_raw = np.asarray(mawecm_data["maw_res_z_support"], dtype=np.int64).reshape(-1)
    z_eps_raw = np.asarray(mawecm_data["Z_eps"], dtype=np.int64).reshape(-1) if "Z_eps" in mawecm_data else None
    z_sig_raw = np.asarray(mawecm_data["Z_sig"], dtype=np.int64).reshape(-1) if "Z_sig" in mawecm_data else None
    w_eps_full_raw = np.asarray(mawecm_data["w_eps_full"], dtype=float).reshape(-1)
    w_sig_full_raw = np.asarray(mawecm_data["w_sig_full"], dtype=float).reshape(-1)

    has_hrom_map = ("hrom_element_full_indices" in mawecm_data) and ("hrom_n_elem" in mawecm_data)
    if has_hrom_map:
        hrom_full_idx = np.asarray(mawecm_data["hrom_element_full_indices"], dtype=np.int64).reshape(-1)
        hrom_n_elem = int(np.ravel(mawecm_data["hrom_n_elem"])[0])
        if hrom_n_elem != hrom_full_idx.size:
            raise RuntimeError(
                f"HROM metadata mismatch: hrom_n_elem={hrom_n_elem}, "
                f"len(hrom_element_full_indices)={hrom_full_idx.size}"
            )
    else:
        hrom_full_idx = None

    using_hrom = (
        has_hrom_map
        and hrom_full_idx.size == int(n_elem)
        and ("w_eps_hrom" in mawecm_data)
        and ("w_sig_hrom" in mawecm_data)
    )

    if using_hrom:
        w_eps_full = np.asarray(mawecm_data["w_eps_hrom"], dtype=float).reshape(-1)
        w_sig_full = np.asarray(mawecm_data["w_sig_hrom"], dtype=float).reshape(-1)
        if w_eps_full.size != n_elem or w_sig_full.size != n_elem:
            raise RuntimeError(
                "HROM-projected MAW homogenization weights size mismatch with current mesh. "
                f"n_elem={n_elem}, len(w_eps_hrom)={w_eps_full.size}, len(w_sig_hrom)={w_sig_full.size}"
            )

        full_to_local = {int(fid): int(i) for i, fid in enumerate(hrom_full_idx.tolist())}
        z_res = _map_support_indices_to_current_mesh(z_res_raw, n_elem=n_elem, full_to_local=full_to_local, label="maw_res_z_support")
        if z_eps_raw is not None:
            z_eps = _map_support_indices_to_current_mesh(z_eps_raw, n_elem=n_elem, full_to_local=full_to_local, label="Z_eps")
        else:
            z_eps = np.flatnonzero(np.abs(w_eps_full) > _W_TOL).astype(np.int64)
        if z_sig_raw is not None:
            z_sig = _map_support_indices_to_current_mesh(z_sig_raw, n_elem=n_elem, full_to_local=full_to_local, label="Z_sig")
        else:
            z_sig = np.flatnonzero(np.abs(w_sig_full) > _W_TOL).astype(np.int64)
        mesh_mode = "hrom"
    else:
        if w_eps_full_raw.size != n_elem or w_sig_full_raw.size != n_elem:
            raise RuntimeError(
                "MAW homogenization weights size mismatch with current mesh. "
                f"n_elem={n_elem}, len(w_eps_full)={w_eps_full_raw.size}, len(w_sig_full)={w_sig_full_raw.size}"
            )
        w_eps_full = w_eps_full_raw
        w_sig_full = w_sig_full_raw
        z_res = _map_support_indices_to_current_mesh(z_res_raw, n_elem=n_elem, full_to_local=None, label="maw_res_z_support")
        if z_eps_raw is not None:
            z_eps = _map_support_indices_to_current_mesh(z_eps_raw, n_elem=n_elem, full_to_local=None, label="Z_eps")
        else:
            z_eps = np.flatnonzero(np.abs(w_eps_full) > _W_TOL).astype(np.int64)
        if z_sig_raw is not None:
            z_sig = _map_support_indices_to_current_mesh(z_sig_raw, n_elem=n_elem, full_to_local=None, label="Z_sig")
        else:
            z_sig = np.flatnonzero(np.abs(w_sig_full) > _W_TOL).astype(np.int64)
        mesh_mode = "full"

    return z_res, z_eps, z_sig, w_eps_full, w_sig_full, mesh_mode


def _build_full_to_local_dof_map(full_mesh_base: str, mp_local, eq_map_local: np.ndarray):
    params_full = setup_kratos_parameters(str(full_mesh_base))
    model_full = KM.Model()
    sim_full = RVEHomogenizationDatasetGenerator(model_full, params_full)
    sim_full.Initialize()
    try:
        mp_full = sim_full._GetSolver().GetComputingModelPart()
        _, eq_map_full, _ = SetUpDofEquationIdsAndDisplacementAdaptor(mp_full)

        node_to_full = {}
        for i, node in enumerate(mp_full.Nodes):
            node_to_full[int(node.Id)] = (int(eq_map_full[i, 0]), int(eq_map_full[i, 1]))

        full_to_local = {}
        missing_nodes = []
        for i, node in enumerate(mp_local.Nodes):
            nid = int(node.Id)
            if nid not in node_to_full:
                missing_nodes.append(nid)
                continue
            fx, fy = node_to_full[nid]
            lx = int(eq_map_local[i, 0])
            ly = int(eq_map_local[i, 1])
            if fx >= 0 and lx >= 0:
                full_to_local[int(fx)] = int(lx)
            if fy >= 0 and ly >= 0:
                full_to_local[int(fy)] = int(ly)

        if missing_nodes:
            raise RuntimeError(
                f"Could not map {len(missing_nodes)} local nodes to full mesh node IDs "
                f"(example={missing_nodes[:8]})."
            )
        return full_to_local
    finally:
        sim_full.Finalize()


def _remap_free_basis_to_local(
    free_dofs_full: np.ndarray,
    phi_m_full: np.ndarray,
    phi_s_full: np.ndarray,
    full_to_local: Dict[int, int],
):
    free_full = np.asarray(free_dofs_full, dtype=np.int64).reshape(-1)
    mask = np.array([int(d) in full_to_local for d in free_full], dtype=bool)
    row_idx = np.flatnonzero(mask)
    if row_idx.size == 0:
        raise RuntimeError("No overlapping free DOFs between full basis and HROM mesh.")

    free_local = np.array([int(full_to_local[int(free_full[i])]) for i in row_idx], dtype=np.int64)
    if np.unique(free_local).size != free_local.size:
        raise RuntimeError("Non-unique mapped free DOFs while remapping full basis to HROM.")

    phi_m_loc = np.asarray(phi_m_full[row_idx, :], dtype=float)
    phi_s_loc = np.asarray(phi_s_full[row_idx, :], dtype=float)
    return free_local, phi_m_loc, phi_s_loc, int(row_idx.size)


def _load_maw_rbf_from_npz(data):
    return {
        "centers": np.asarray(data["maw_res_rbf_centers"], dtype=float),
        "center_ids": np.asarray(data["maw_res_rbf_center_ids"], dtype=np.int64),
        "length_scales": np.asarray(data["maw_res_rbf_length_scales"], dtype=float),
        "Alpha": np.asarray(data["maw_res_rbf_alpha"], dtype=float),
        "Beta": np.asarray(data["maw_res_rbf_beta"], dtype=float),
        "scale": np.asarray(data["maw_res_rbf_scale"], dtype=float),
        "poly_mode": int(np.ravel(data["maw_res_rbf_poly_mode"])[0]),
        "lambda_reg": float(np.ravel(data["maw_res_rbf_lambda"])[0]),
        "n_centers": int(np.ravel(data["maw_res_rbf_n_centers"])[0]),
    }


def LoadHpromMawEcmGprModel(
    stage2a_dir: str = "stage_2a_pod_data",
    stage3_dataset_file: str = "stage_3_prom_regression_dataset/stage3_prom_regression_dataset.npz",
    stage4_gpr_dir: str = "stage_4_prom_gpr_sparse",
    mawecm_file: str = "stage_8b_hprom_mawecm_res_rbf/ecm_weights_all.npz",
    stage2b_dir: str = "stage_2b_ls_master",
):
    model_pack = LoadPromGprModel(
        stage2a_dir=stage2a_dir,
        stage3_dataset_file=stage3_dataset_file,
        stage4_gpr_dir=stage4_gpr_dir,
        stage2b_dir=stage2b_dir,
    )
    if not os.path.exists(mawecm_file):
        raise FileNotFoundError(mawecm_file)
    data = np.load(mawecm_file, allow_pickle=True)

    required = [
        "Z_res",
        "w_eps_full",
        "w_sig_full",
        "A0_ref",
        "maw_res_state_space",
        "maw_res_z_support",
        "maw_res_q_train",
        "maw_res_W_train",
        "maw_res_rbf_centers",
        "maw_res_rbf_alpha",
    ]
    missing = [k for k in required if k not in data]
    if missing:
        raise RuntimeError(f"MAW-ECM file missing keys: {missing}")

    out = {k: data[k] for k in data.files}
    out["maw_rbf_model"] = _load_maw_rbf_from_npz(data)
    return model_pack, out


def RunHpromMawEcmGprBatchSimulation(
    parameters,
    model_pack: Dict[str, object],
    mawecm_data: Dict[str, object],
    strain_path,
    out_dir=None,
    trajectory_index=None,
    reference_amplitude=None,
    reference_steps=REFERENCE_STEPS_FOR_UNIT_AMPLITUDE,
    prom_corrector_max_iters=10,
    prom_corrector_rel_tol=1.0e-5,
    prom_corrector_abs_tol=1.0e-10,
    prom_corrector_dq_abs_tol=1.0e-7,
    prom_corrector_dq_rel_tol=1.0e-6,
    prom_corrector_res_floor_for_dq=1.0e-1,
    prom_corrector_min_rel_drop_stop=1.0e-2,
    prom_corrector_stagnation_relnorm_gate=1.0e-4,
    prom_corrector_max_dq_norm=0.5,
    prom_corrector_damping_after_iter=10,
    prom_corrector_damping_factor=0.5,
    prom_use_old_stiffness_in_first_iteration=True,
    prom_old_stiffness_residual_cutoff=1.0e5,
    prom_corrector_l2_reg=1.0e-10,
    prom_fail_on_nonconvergence=True,
    update_maw_weights_each_iter=1,
    include_weight_tangent=1,
    clip_nonnegative=1,
    renorm_weights=0,
    homogenization_mode="full_fom",
    track_q_pod=0,
    sync_modelpart_each_newton_iter=0,
    call_entity_hooks_each_newton_iter=0,
    use_analysis_stage_solution_step_hooks=0,
):
    if strain_path is None:
        raise ValueError("strain_path must be provided.")

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)

    phi_m_full = np.asarray(model_pack["phi_m"], dtype=float)
    phi_s_full = np.asarray(model_pack["phi_s"], dtype=float)
    a_m = np.asarray(model_pack["A_m"], dtype=float)
    c_m = np.asarray(model_pack["C_m"], dtype=float)
    c_s = np.asarray(model_pack["C_s"], dtype=float)

    free_dofs_full = np.asarray(model_pack["free_dofs"], dtype=np.int64)
    mu_space = str(model_pack["mu_space"])
    mapping = str(model_pack["mapping"])
    q_m_dim = int(model_pack["q_m_dim"])
    q_s_dim = int(model_pack["q_s_dim"])
    track_qpod = bool(int(track_q_pod))
    q_pod_dim = int(model_pack["q_pod_dim"]) if track_qpod else 0
    gpr_model = model_pack["gpr_model"]
    if int(gpr_model["input_dim"]) != q_m_dim:
        raise RuntimeError(
            f"Sparse-GPR input_dim mismatch: model={int(gpr_model['input_dim'])}, q_m_dim={q_m_dim}."
        )
    if int(gpr_model["output_dim"]) != q_s_dim:
        raise RuntimeError(
            f"Sparse-GPR output_dim mismatch: model={int(gpr_model['output_dim'])}, q_s_dim={q_s_dim}."
        )

    maw_rbf = mawecm_data["maw_rbf_model"]
    maw_state_space = str(np.ravel(mawecm_data["maw_res_state_space"])[0]).strip().lower()
    if maw_state_space != "q_m":
        raise RuntimeError(
            f"Unsupported MAW state space '{maw_state_space}'. "
            "This implementation is locked to q_m to match the intended workflow."
        )
    need_qpod_state = False
    renorm_target = float(np.ravel(mawecm_data["maw_res_renorm_target"])[0]) if "maw_res_renorm_target" in mawecm_data else None
    hom_mode = str(homogenization_mode).strip().lower()
    if hom_mode not in {"full_fom", "ecm_separate"}:
        raise RuntimeError(
            "Unsupported homogenization_mode for HPROM-MAWECM-GPR. "
            "Use one of: {'full_fom', 'ecm_separate'}."
        )

    dt = parameters["solver_settings"]["time_stepping"]["time_step"].GetDouble()
    e_wp = np.asarray(strain_path, dtype=float)
    if e_wp.ndim != 2 or e_wp.shape[1] != 3:
        raise ValueError(f"strain_path must have shape [n,3], got {e_wp.shape}")

    n_seg = e_wp.shape[0] - 1
    seg_steps, _ = BuildDynamicSegmentSteps(
        e_wp,
        reference_steps=reference_steps,
        min_steps=MIN_STEPS_PER_SEGMENT,
        reference_amplitude=reference_amplitude,
    )
    step_offsets = np.concatenate(([0], np.cumsum(seg_steps)))
    n_steps_total = int(step_offsets[-1])
    end_time = dt * float(n_steps_total)
    if n_steps_total <= 0:
        raise RuntimeError("Dynamic step allocation produced zero total steps.")

    model = KM.Model()
    sim = RVEHomogenizationDatasetGenerator(model, parameters)
    sim.Initialize()

    mp = sim._GetSolver().GetComputingModelPart()
    entities = list(mp.Elements) + list(mp.Conditions)
    n_dof, eq_map_runtime, ta_disp = SetUpDofEquationIdsAndDisplacementAdaptor(mp)
    elements_all = list(mp.Elements)
    n_elem = len(elements_all)

    z_res, z_eps, z_sig, w_eps_full, w_sig_full, mesh_mode = _extract_hrom_aligned_maw_arrays(
        mawecm_data=mawecm_data,
        n_elem=n_elem,
    )
    if mesh_mode == "hrom":
        if "hrom_full_mesh_base" not in mawecm_data:
            raise RuntimeError(
                "MAW data has HROM-projected weights but missing 'hrom_full_mesh_base' for DOF remapping."
            )
        full_mesh_base = str(np.ravel(mawecm_data["hrom_full_mesh_base"])[0])
        full_to_local = _build_full_to_local_dof_map(
            full_mesh_base=full_mesh_base,
            mp_local=mp,
            eq_map_local=np.asarray(eq_map_runtime, dtype=np.int64),
        )
        free_dofs, phi_m, phi_s, n_overlap = _remap_free_basis_to_local(
            free_dofs_full=free_dofs_full,
            phi_m_full=phi_m_full,
            phi_s_full=phi_s_full,
            full_to_local=full_to_local,
        )
    else:
        free_dofs = np.asarray(free_dofs_full, dtype=np.int64)
        phi_m = np.asarray(phi_m_full, dtype=float)
        phi_s = np.asarray(phi_s_full, dtype=float)
        n_overlap = int(free_dofs.size)

    if phi_m.shape[0] != free_dofs.size or phi_s.shape[0] != free_dofs.size:
        raise RuntimeError(
            f"Basis/free DOF mismatch after remap: free={free_dofs.size}, "
            f"phi_m rows={phi_m.shape[0]}, phi_s rows={phi_s.shape[0]}."
        )
    # Fixed residual weights from file (classical ECM equivalent when MAW updates are disabled).
    w_res_support_fixed = None
    if "w_res_hrom" in mawecm_data:
        w_try = np.asarray(mawecm_data["w_res_hrom"], dtype=float).reshape(-1)
        if w_try.size == n_elem:
            w_res_support_fixed = w_try[z_res]
    if w_res_support_fixed is None and "w_res_full" in mawecm_data:
        w_try = np.asarray(mawecm_data["w_res_full"], dtype=float).reshape(-1)
        if w_try.size == n_elem:
            w_res_support_fixed = w_try[z_res]
    elem_res = [elements_all[int(i)] for i in z_res.tolist()]
    assembler_hom_full = None
    assembler_hom_union = None
    w_eps_sel = None
    w_sig_sel = None
    a0_ref = None
    z_hom_union = None

    if hom_mode == "full_fom":
        z_hom_full = np.arange(n_elem, dtype=np.int64)
        elem_hom_full = [elements_all[int(i)] for i in z_hom_full.tolist()]
        assembler_hom_full = VectorizedAssembler(
            mp,
            n_dof,
            eq_map_runtime,
            elements=elem_hom_full,
            element_scales=None,
            log_label="HPROM-MAW-HomAssemblerFull",
        )
    elif hom_mode == "ecm_separate":
        if "A0_ref" in mawecm_data:
            a0_ref = float(np.ravel(mawecm_data["A0_ref"])[0])
        elif "hom_reference_measure" in mawecm_data:
            a0_ref = float(np.ravel(mawecm_data["hom_reference_measure"])[0])
        else:
            raise RuntimeError(
                "MAW-ECM file missing A0_ref / hom_reference_measure required for homogenization_mode='ecm_separate'."
            )
        if z_eps.size == 0 or z_sig.size == 0:
            raise RuntimeError(
                "homogenization_mode='ecm_separate' requires non-empty Z_eps and Z_sig."
            )

        # Single homogenization assembler on union support with separate weight vectors
        # for strain and stress (no residual coupling).
        z_hom_union = np.union1d(z_eps, z_sig).astype(np.int64)
        elem_hom_union = [elements_all[int(i)] for i in z_hom_union.tolist()]
        w_eps_sel = np.asarray(w_eps_full[z_hom_union], dtype=float)
        w_sig_sel = np.asarray(w_sig_full[z_hom_union], dtype=float)
        assembler_hom_union = VectorizedAssembler(
            mp,
            n_dof,
            eq_map_runtime,
            elements=elem_hom_union,
            element_scales=None,
            log_label="HPROM-MAW-HomAssemblerUnion",
        )

    sim._InitializeDomainCenterIfNeeded(mp)
    x0c, y0c = float(sim._x0c), float(sim._y0c)

    dof_x = np.zeros(n_dof, dtype=float)
    dof_y = np.zeros(n_dof, dtype=float)
    is_x_dof = np.zeros(n_dof, dtype=bool)
    for i, node in enumerate(mp.Nodes):
        xr = float(node.X0) - x0c
        yr = float(node.Y0) - y0c
        ix = int(eq_map_runtime[i, 0])
        iy = int(eq_map_runtime[i, 1])
        if 0 <= ix < n_dof:
            dof_x[ix] = xr
            dof_y[ix] = yr
            is_x_dof[ix] = True
        if 0 <= iy < n_dof:
            dof_x[iy] = xr
            dof_y[iy] = yr
            is_x_dof[iy] = False

    free_mask = np.zeros(n_dof, dtype=bool)
    free_mask[free_dofs] = True
    dir_dofs = np.nonzero(~free_mask)[0].astype(np.int64)

    x_free = dof_x[free_dofs]
    y_free = dof_y[free_dofs]
    is_x_free = is_x_dof[free_dofs]
    x_dir = dof_x[dir_dofs]
    y_dir = dof_y[dir_dofs]
    is_x_dir = is_x_dof[dir_dofs]

    def _affine_component(e_vec, xx, yy, is_x):
        f = DeformationGradientFromGreenLagrange2D(e_vec)
        ux = (f[0, 0] - 1.0) * xx + f[0, 1] * yy
        uy = f[1, 0] * xx + (f[1, 1] - 1.0) * yy
        return np.where(is_x, ux, uy)

    def _eval_maw_weights(state_vec):
        q_query = np.asarray(state_vec, dtype=float).reshape(1, -1)
        w_col = eval_mawecm_rbf(
            q_query=q_query,
            model=maw_rbf,
            clip_nonnegative=bool(int(clip_nonnegative)),
            renorm_target=(renorm_target if (bool(int(renorm_weights)) and renorm_target is not None) else None),
        )
        w = np.asarray(w_col[:, 0], dtype=float).reshape(-1)
        if w.size != z_res.size:
            raise RuntimeError(
                f"Adaptive MAW weight size mismatch: got {w.size}, expected {z_res.size}."
            )
        return w

    def _eval_maw_weights_and_jac(state_vec):
        q_query = np.asarray(state_vec, dtype=float).reshape(1, -1)
        w_col, dw_col = eval_mawecm_rbf_with_jacobian(
            q_query=q_query,
            model=maw_rbf,
            clip_nonnegative=bool(int(clip_nonnegative)),
            renorm_target=(renorm_target if (bool(int(renorm_weights)) and renorm_target is not None) else None),
        )
        w = np.asarray(w_col[:, 0], dtype=float).reshape(-1)
        dw = np.asarray(dw_col[:, 0, :], dtype=float)  # (n_res, q_m_dim)
        if w.size != z_res.size:
            raise RuntimeError(
                f"Adaptive MAW weight size mismatch: got {w.size}, expected {z_res.size}."
            )
        if dw.shape != (z_res.size, q_m_dim):
            raise RuntimeError(
                "Adaptive MAW weight Jacobian size mismatch: "
                f"got {dw.shape}, expected ({z_res.size}, {q_m_dim})."
            )
        return w, dw

    def _state_for_maw(state_mode: str, q_m_vec: np.ndarray):
        if state_mode != "q_m":
            raise RuntimeError(f"Unsupported MAW state space '{state_mode}'.")
        return q_m_vec

    # Manifold correction for consistency at origin: N(0)=0 and J(0)=0.
    q_s0_raw, j0_raw = evaluate_sparse_gp_map_and_jacobian_qp(
        np.zeros(q_m_dim, dtype=float), gpr_model, q_m_dim
    )
    q_s0_raw = np.asarray(q_s0_raw, dtype=float).reshape(-1)
    j0_raw = np.asarray(j0_raw, dtype=float)
    if q_s0_raw.size != q_s_dim:
        raise RuntimeError(
            f"Sparse-GPR zero-state output mismatch: got {q_s0_raw.size}, expected {q_s_dim}."
        )
    if j0_raw.shape != (q_s_dim, q_m_dim):
        raise RuntimeError(
            f"Sparse-GPR zero-state Jacobian mismatch: got {j0_raw.shape}, expected ({q_s_dim},{q_m_dim})."
        )
    phi_master = phi_m @ a_m
    phi_master_eff = phi_master + (phi_s @ j0_raw)
    w0_const = phi_s @ q_s0_raw

    results_eps = [np.zeros(3, dtype=float)]
    results_sig = [np.zeros(3, dtype=float)]
    u_hist = [np.zeros(n_dof, dtype=float)]
    e_applied_hist = [np.zeros(3, dtype=float)]
    qm_hist = [np.zeros(q_m_dim, dtype=float)]
    qs_hist = [np.zeros(q_s_dim, dtype=float)]
    qpod_hist = [np.zeros(q_pod_dim, dtype=float)] if track_qpod else None
    wres_hist = [np.zeros(z_res.size, dtype=float)]

    n_corr = max(0, int(prom_corrector_max_iters))
    damp_after = int(prom_corrector_damping_after_iter)
    damp_factor = float(prom_corrector_damping_factor)
    if not (0.0 < damp_factor <= 1.0):
        raise ValueError(
            f"prom_corrector_damping_factor must be in (0,1], got {damp_factor}."
        )
    update_each_iter = bool(int(update_maw_weights_each_iter))
    use_weight_tangent = bool(int(include_weight_tangent)) and update_each_iter
    free_dof_index_map = _build_free_dof_index_map(n_dof=n_dof, free_dofs=free_dofs)
    assembler_hr = VectorizedAssembler(
        mp,
        n_dof,
        eq_map_runtime,
        elements=elem_res,
        element_scales=np.ones(z_res.size, dtype=float),
        log_label="HPROM-MAW-ResidualAssembler",
    )
    unit_rhs_scatter = _prepare_unit_rhs_scatter(
        assembler_unit=assembler_hr,
        free_dof_index_map=free_dof_index_map,
    ) if use_weight_tangent else None

    print(
        f"  [HPROM-MAWECM-GPR] Solving trajectory with {n_steps_total} increments "
        f"(corrector iters={n_corr}, update_weights_each_iter={int(update_each_iter)})"
    )
    print(
        f"  [HPROM-MAWECM-GPR] Residual support: |Z_res|={z_res.size} "
        f"({100.0*z_res.size/max(n_elem,1):.1f}% of {n_elem})"
    )
    if hom_mode == "full_fom":
        print(
            f"  [HPROM-MAWECM-GPR] Hom support: full mesh "
            f"({n_elem}/{n_elem}, 100.0%), mesh_mode={mesh_mode}"
        )
    else:
        print(
            f"  [HPROM-MAWECM-GPR] Hom support (separate ECM): "
            f"|Z_eps|={z_eps.size} ({100.0*z_eps.size/max(n_elem,1):.1f}%), "
            f"|Z_sig|={z_sig.size} ({100.0*z_sig.size/max(n_elem,1):.1f}%), "
            f"|Z_union|={z_hom_union.size} ({100.0*z_hom_union.size/max(n_elem,1):.1f}%), "
            f"mesh_mode={mesh_mode}"
        )
    print(
        f"  [HPROM-MAWECM-GPR] DOF overlap full->mesh: free_used={n_overlap}/{free_dofs_full.size}"
    )
    print(f"  [HPROM-MAWECM-GPR] MAW state space: {maw_state_space}")
    print(f"  [HPROM-MAWECM-GPR] Homogenization mode: {hom_mode}")
    print(f"  [HPROM-MAWECM-GPR] Include d(w)/d(q_m) tangent term: {int(use_weight_tangent)}")
    print(
        "  [HPROM-MAWECM-GPR] Newton modelpart sync/hooks: "
        f"sync={int(bool(sync_modelpart_each_newton_iter))}, "
        f"hooks={int(bool(call_entity_hooks_each_newton_iter))}"
    )
    print(
        f"  [HPROM-MAWECM-GPR] AnalysisStage step hooks: "
        f"{int(bool(use_analysis_stage_solution_step_hooks))}"
    )
    if (not update_each_iter) and (w_res_support_fixed is not None):
        print("  [HPROM-MAWECM-GPR] MAW updates disabled: using fixed residual weights from file.")
    print("  [HPROM-MAWECM-GPR] Manifold correction active: N(0)=0 and J(0)=0.")

    t_eval_gpr = 0.0
    t_eval_maw = 0.0
    t_sync_newton = 0.0
    t_res_asm = 0.0
    t_unit_asm = 0.0
    t_lin_solve = 0.0
    t_hom_asm = 0.0

    q_m_state = None
    k_red_old = None
    for step in range(1, n_steps_total + 1):
        time_val = float(step) * float(dt)
        mp.CloneTimeStep(time_val)
        mp.ProcessInfo[KM.DELTA_TIME] = dt
        mp.ProcessInfo[KM.TIME] = time_val
        mp.ProcessInfo[KM.STEP] = step

        sim.time, sim.step, sim.end_time = time_val, step, end_time
        if bool(use_analysis_stage_solution_step_hooks):
            sim.InitializeSolutionStep()

        s = int(np.searchsorted(step_offsets, step, side="left") - 1)
        s = max(0, min(s, n_seg - 1))
        xi = float(step - step_offsets[s]) / float(max(seg_steps[s], 1))
        e = (1.0 - xi) * e_wp[s, :] + xi * e_wp[s + 1, :]

        mu = _recover_mu_from_E(e, mapping=mapping, mu_space=mu_space)
        if q_m_state is None:
            q_m = _initial_qm_from_mu(mu, model_pack=model_pack)
        else:
            q_m = np.asarray(q_m_state, dtype=float).reshape(-1).copy()

        q_s = np.zeros(q_s_dim, dtype=float)
        q_s_phys = np.zeros(q_s_dim, dtype=float)
        q_pod = np.zeros(q_pod_dim, dtype=float) if (track_qpod or need_qpod_state) else None

        u_aff_free = _affine_component(e, x_free, y_free, is_x_free)
        u_aff_dir = _affine_component(e, x_dir, y_dir, is_x_dir)
        u = np.zeros(n_dof, dtype=float)
        u[dir_dofs] = u_aff_dir

        w_res_iter = None
        dw_res_iter = None
        if not update_each_iter:
            t0 = time.perf_counter()
            q_s_phys, _ = evaluate_sparse_gp_map_and_jacobian_qp(q_m, gpr_model, q_m_dim)
            t_eval_gpr += time.perf_counter() - t0
            q_s_phys = np.asarray(q_s_phys, dtype=float).reshape(-1)
            q_s = q_s_phys - q_s0_raw - (j0_raw @ q_m)
            q_master0 = a_m @ q_m
            if track_qpod or need_qpod_state:
                q_pod = (c_m @ q_master0) + (c_s @ q_s_phys)
            if w_res_support_fixed is not None:
                w_res_iter = w_res_support_fixed.copy()
            else:
                maw_state0 = _state_for_maw(maw_state_space, q_m)
                if use_weight_tangent:
                    t1 = time.perf_counter()
                    w_res_iter, dw_res_iter = _eval_maw_weights_and_jac(maw_state0)
                    t_eval_maw += time.perf_counter() - t1
                else:
                    t1 = time.perf_counter()
                    w_res_iter = _eval_maw_weights(maw_state0)
                    t_eval_maw += time.perf_counter() - t1
            assembler_hr.SetElementScales(w_res_iter)

        converged = False
        nonfinite_detected = False
        nr0 = None
        ndq0 = None
        nr_last = np.nan
        ndq_last = np.nan
        dq_rel_last = np.nan
        nr_prev = None
        nr_best = np.inf
        rrel_best = np.inf
        q_m_step_start = q_m.copy()
        q_m_best = q_m.copy()
        k_red_last = None
        w_last = None

        for it in range(n_corr + 1):
            mp.ProcessInfo[KM.NL_ITERATION_NUMBER] = it + 1

            t0 = time.perf_counter()
            q_s_raw, dqs_raw = evaluate_sparse_gp_map_and_jacobian_qp(q_m, gpr_model, q_m_dim)
            t_eval_gpr += time.perf_counter() - t0
            q_s_raw = np.asarray(q_s_raw, dtype=float).reshape(-1)
            dqs_raw = np.asarray(dqs_raw, dtype=float)
            if q_s_raw.size != q_s_dim:
                raise RuntimeError(
                    f"Sparse-GPR output size mismatch: got {q_s_raw.size}, expected {q_s_dim}."
                )
            if dqs_raw.shape != (q_s_dim, q_m_dim):
                raise RuntimeError(
                    f"Sparse-GPR Jacobian shape mismatch: got {dqs_raw.shape}, expected ({q_s_dim},{q_m_dim})."
                )
            if (not np.all(np.isfinite(q_s_raw))) or (not np.all(np.isfinite(dqs_raw))):
                nonfinite_detected = True
                break

            q_s_phys = q_s_raw
            q_s = q_s_phys - q_s0_raw - (j0_raw @ q_m)
            dqs_dqm = dqs_raw - j0_raw
            q_master = a_m @ q_m
            if track_qpod or need_qpod_state:
                q_pod = (c_m @ q_master) + (c_s @ q_s_phys)
            u[free_dofs] = u_aff_free + w0_const + (phi_master_eff @ q_m) + (phi_s @ q_s)

            if update_each_iter:
                maw_state_iter = _state_for_maw(maw_state_space, q_m)
                if use_weight_tangent:
                    t1 = time.perf_counter()
                    w_res_iter, dw_res_iter = _eval_maw_weights_and_jac(maw_state_iter)
                    t_eval_maw += time.perf_counter() - t1
                else:
                    t1 = time.perf_counter()
                    w_res_iter = _eval_maw_weights(maw_state_iter)
                    t_eval_maw += time.perf_counter() - t1
                assembler_hr.SetElementScales(w_res_iter)
            w_last = w_res_iter

            if bool(sync_modelpart_each_newton_iter):
                t1 = time.perf_counter()
                SetDisplacementFromEquationVector(u, eq_map_runtime, ta_disp)
                UpdateCurrentCoordinatesFromDisplacement(mp, step=0)
                t_sync_newton += time.perf_counter() - t1

            if bool(call_entity_hooks_each_newton_iter):
                InitializeNonLinearIteration(entities, mp.ProcessInfo)
            t1 = time.perf_counter()
            k_hr, rhs_hr = assembler_hr.Assemble(u)
            t_res_asm += time.perf_counter() - t1
            if bool(call_entity_hooks_each_newton_iter):
                FinalizeNonLinearIteration(entities, mp.ProcessInfo)

            r_f = rhs_hr[free_dofs]
            du_dqm = phi_master_eff + (phi_s @ dqs_dqm)
            r_red = du_dqm.T @ r_f
            nr = float(np.linalg.norm(r_red))
            if not np.isfinite(nr):
                nonfinite_detected = True
                break
            nr_last = nr
            if nr0 is None:
                nr0 = max(nr, 1.0e-30)
            r_rel = nr / nr0
            if nr < nr_best:
                nr_best = nr
                rrel_best = r_rel
                q_m_best = q_m.copy()

            kff = k_hr[free_dofs][:, free_dofs]
            k_red = du_dqm.T @ (kff @ du_dqm)
            if use_weight_tangent and (dw_res_iter is not None):
                # Consistent MAW term: J_u^T [sum_e r_e \otimes d w_e/dq_m]
                # where r_e are unit-weight element RHS contributions on free DOFs.
                t1 = time.perf_counter()
                ne = int(z_res.size)
                ndl = int(assembler_hr.n_local_dof)
                rhs_scaled_loc = -np.asarray(assembler_hr._f_int, dtype=float).reshape(ne, ndl)
                scale_vec = np.asarray(w_res_iter, dtype=float).reshape(ne, 1)
                # Recover unit-weight element RHS from currently assembled scaled residual:
                # rhs_scaled = w_e * rhs_unit  => rhs_unit = rhs_scaled / w_e.
                rhs_unit_loc = np.divide(
                    rhs_scaled_loc,
                    scale_vec,
                    out=np.zeros_like(rhs_scaled_loc),
                    where=np.abs(scale_vec) > 1.0e-14,
                )
                r_free_e = _scatter_rhs_loc_by_element_free(
                    rhs_loc=rhs_unit_loc,
                    n_free=free_dofs.size,
                    scatter_cache=unit_rhs_scatter,
                )  # (n_free, n_res)
                drw_dqm = r_free_e @ dw_res_iter  # (n_free, q_m_dim)
                k_red = k_red + (du_dqm.T @ drw_dqm)
                t_unit_asm += time.perf_counter() - t1
            if float(prom_corrector_l2_reg) > 0.0:
                k_red = k_red + float(prom_corrector_l2_reg) * np.eye(q_m_dim, dtype=float)
            k_red_last = k_red

            if nr <= float(prom_corrector_abs_tol):
                converged = True
                print(f"  > It {it:02d}: ||R_r|| = {nr:.3e}, rel = {r_rel:.3e}")
                print(f"  > Converged in {it} iterations.")
                break

            if (
                ndq_last == ndq_last
                and ndq_last <= float(prom_corrector_dq_abs_tol)
                and r_rel <= float(prom_corrector_rel_tol)
                and nr <= float(prom_corrector_res_floor_for_dq)
            ):
                converged = True
                print(
                    f"  > It {it:02d}: ||R_r|| = {nr:.3e}, rel = {r_rel:.3e}, "
                    f"||dq_m|| = {ndq_last:.3e} -> converged(small-dq+rel)"
                )
                print(f"  > Converged in {it} iterations.")
                break

            if nr_prev is not None:
                rel_drop = abs(nr_prev - nr) / max(nr_prev, 1.0e-30)
                if (
                    rel_drop <= float(prom_corrector_min_rel_drop_stop)
                    and r_rel <= float(prom_corrector_stagnation_relnorm_gate)
                    and nr <= float(prom_corrector_res_floor_for_dq)
                ):
                    converged = True
                    print(
                        f"  > It {it:02d}: ||R_r|| = {nr:.3e}, rel = {r_rel:.3e}, "
                        f"rel_drop={rel_drop:.3e} -> converged(stagnation)"
                    )
                    print(f"  > Converged in {it} iterations.")
                    break
            nr_prev = nr

            if r_rel <= float(prom_corrector_rel_tol) and nr <= float(prom_corrector_res_floor_for_dq):
                converged = True
                print(f"  > It {it:02d}: ||R_r|| = {nr:.3e}, rel = {r_rel:.3e} -> converged(rel)")
                print(f"  > Converged in {it} iterations.")
                break

            if it >= n_corr:
                break

            k_solve = k_red
            used_old = False
            if (
                it == 0
                and bool(prom_use_old_stiffness_in_first_iteration)
                and k_red_old is not None
                and k_red_old.shape == k_red.shape
                and nr <= float(prom_old_stiffness_residual_cutoff)
            ):
                k_solve = k_red_old
                used_old = True

            try:
                t1 = time.perf_counter()
                dq = np.linalg.solve(k_solve, r_red)
                t_lin_solve += time.perf_counter() - t1
            except np.linalg.LinAlgError:
                t1 = time.perf_counter()
                dq = np.linalg.lstsq(k_solve, r_red, rcond=None)[0]
                t_lin_solve += time.perf_counter() - t1
            if used_old and (not np.all(np.isfinite(dq))):
                try:
                    t1 = time.perf_counter()
                    dq = np.linalg.solve(k_red, r_red)
                    t_lin_solve += time.perf_counter() - t1
                    used_old = False
                except np.linalg.LinAlgError:
                    t1 = time.perf_counter()
                    dq = np.linalg.lstsq(k_red, r_red, rcond=None)[0]
                    t_lin_solve += time.perf_counter() - t1
                    used_old = False

            ndq = float(np.linalg.norm(dq))
            ndq_last = ndq
            if ndq0 is None:
                ndq0 = max(ndq, 1.0e-30)
            dq_rel = ndq / ndq0
            dq_rel_last = dq_rel
            if ndq > float(prom_corrector_max_dq_norm) and ndq > 0.0:
                scale = float(prom_corrector_max_dq_norm) / ndq
                dq = dq * scale
                ndq = float(np.linalg.norm(dq))
                ndq_last = ndq
                dq_rel = ndq / ndq0
                dq_rel_last = dq_rel

            alpha = 1.0
            if it > damp_after:
                alpha = min(max(damp_factor, 1.0e-6), 1.0)
            dq_step = alpha * dq
            ndq = float(np.linalg.norm(dq_step))
            ndq_last = ndq
            dq_rel = ndq / ndq0
            dq_rel_last = dq_rel
            q_m = q_m + dq_step
            if used_old:
                print("    > using previous reduced stiffness (K_old) at first iteration")
            if alpha < 1.0:
                print(f"    > reduced-step damping alpha={alpha:.3f}")

            dq_ok = (ndq <= float(prom_corrector_dq_abs_tol)) or (dq_rel <= float(prom_corrector_dq_rel_tol))
            if dq_ok and nr <= float(prom_corrector_res_floor_for_dq):
                converged = True
                print(
                    f"  > It {it:02d}: ||R_r|| = {nr:.3e}, rel = {r_rel:.3e}, "
                    f"||dq_m|| = {ndq:.3e} (rel {dq_rel:.3e}) -> converged(dq)"
                )
                print(f"  > Converged in {it} iterations.")
                break

            print(f"  > It {it:02d}: ||R_r|| = {nr:.3e}, rel = {r_rel:.3e}")

        if not converged:
            quasi_converged = (
                np.isfinite(nr_best)
                and np.isfinite(rrel_best)
                and (rrel_best <= float(prom_corrector_rel_tol))
                and (nr_best <= float(prom_corrector_res_floor_for_dq))
            )
            if quasi_converged:
                q_m = q_m_best.copy()
                converged = True
                print(
                    "  [HPROM-MAWECM-GPR] Step accepted as quasi-converged: "
                    f"best ||R_r||={nr_best:.3e}, rel={rrel_best:.3e}"
                )
            else:
                msg = (
                    f"[HPROM-MAWECM-GPR] Step {step:03d}/{n_steps_total} not converged in {n_corr} iters. "
                    f"last ||R_r||={nr_last:.3e}, ||dq_m||={ndq_last:.3e}."
                )
                if nonfinite_detected:
                    msg += " Non-finite state detected."
                if bool(prom_fail_on_nonconvergence):
                    raise RuntimeError(msg)
                if np.isfinite(nr_best):
                    q_m = q_m_best.copy()
                else:
                    q_m = q_m_step_start.copy()
                print(f"  [HPROM-MAWECM-GPR][WARN] {msg}")
                if nonfinite_detected:
                    print("  [HPROM-MAWECM-GPR][WARN] non-finite state detected; rolled back to best finite iterate.")
            k_red_old = None
        elif k_red_last is not None and np.all(np.isfinite(k_red_last)):
            k_red_old = k_red_last.copy()
        else:
            k_red_old = None

        t0 = time.perf_counter()
        q_s_raw_f, _ = evaluate_sparse_gp_map_and_jacobian_qp(q_m, gpr_model, q_m_dim)
        t_eval_gpr += time.perf_counter() - t0
        q_s_phys = np.asarray(q_s_raw_f, dtype=float).reshape(-1)
        q_s = q_s_phys - q_s0_raw - (j0_raw @ q_m)
        q_master = a_m @ q_m
        if track_qpod or need_qpod_state:
            q_pod = (c_m @ q_master) + (c_s @ q_s_phys)
        u[free_dofs] = u_aff_free + w0_const + (phi_master_eff @ q_m) + (phi_s @ q_s)
        if update_each_iter:
            maw_state_acc = _state_for_maw(maw_state_space, q_m)
            t1 = time.perf_counter()
            w_last = _eval_maw_weights(maw_state_acc)
            t_eval_maw += time.perf_counter() - t1
        elif w_res_support_fixed is not None:
            w_last = w_res_support_fixed

        SetDisplacementFromEquationVector(u, eq_map_runtime, ta_disp)
        UpdateCurrentCoordinatesFromDisplacement(mp, step=0)
        if hom_mode == "full_fom":
            InitializeNonLinearIteration(entities, mp.ProcessInfo)
            t1 = time.perf_counter()
            _, _ = assembler_hom_full.Assemble(u)
            t_hom_asm += time.perf_counter() - t1
            FinalizeNonLinearIteration(entities, mp.ProcessInfo)

            ref_full = float(np.sum(np.asarray(assembler_hom_full.area_e, dtype=float)))
            eps_h, sig_h = CalculateHomogenizedFromAssemblerWithElementWeights(
                assembler_hom_full,
                w_eps=None,
                w_sig=None,
                reference_measure=ref_full,
            )
        else:
            InitializeNonLinearIteration(entities, mp.ProcessInfo)
            t1 = time.perf_counter()
            _, _ = assembler_hom_union.Assemble(u)
            t_hom_asm += time.perf_counter() - t1
            FinalizeNonLinearIteration(entities, mp.ProcessInfo)
            eps_h, sig_h = CalculateHomogenizedFromAssemblerWithElementWeights(
                assembler_hom_union,
                w_eps=w_eps_sel,
                w_sig=w_sig_sel,
                reference_measure=a0_ref,
            )
            eps_h = np.asarray(eps_h, dtype=float)
            sig_h = np.asarray(sig_h, dtype=float)

        if bool(use_analysis_stage_solution_step_hooks):
            sim.FinalizeSolutionStep()

        results_eps.append(np.asarray(eps_h, dtype=float))
        results_sig.append(np.asarray(sig_h, dtype=float))
        u_hist.append(u.copy())
        e_applied_hist.append(e.copy())
        qm_hist.append(q_m.copy())
        qs_hist.append(q_s_phys.copy())
        if track_qpod:
            qpod_hist.append(q_pod.copy())
        wres_hist.append(np.asarray(w_last if w_last is not None else np.zeros(z_res.size), dtype=float).copy())
        q_m_state = q_m.copy()

        print(
            f"  [HPROM-MAWECM-GPR] Step {step:03d}/{n_steps_total}: "
            f"||q_m||={np.linalg.norm(q_m):.3e}, ||q_s||={np.linalg.norm(q_s_phys):.3e}, "
            f"sum(w_res)={float(np.sum(wres_hist[-1])):.3e}"
        )

    sim.Finalize()

    strain_hist = np.stack(results_eps)
    stress_hist = np.stack(results_sig)
    u_hist = np.stack(u_hist)
    e_applied_hist = np.stack(e_applied_hist)
    qm_hist = np.stack(qm_hist)
    qs_hist = np.stack(qs_hist)
    if track_qpod:
        qpod_hist = np.stack(qpod_hist)
    wres_hist = np.stack(wres_hist)

    print(
        "  [HPROM-MAWECM-GPR][timing] "
        f"gpr_eval={t_eval_gpr:.2f}s, maw_eval={t_eval_maw:.2f}s, "
        f"newton_sync={t_sync_newton:.2f}s, res_asm={t_res_asm:.2f}s, "
        f"unit_asm={t_unit_asm:.2f}s, lin_solve={t_lin_solve:.2f}s, hom_asm={t_hom_asm:.2f}s"
    )

    if out_dir is not None:
        tag = f"trajectory_{trajectory_index}" if trajectory_index is not None else "hprom_mawecm_gpr_run"
        np.save(os.path.join(out_dir, f"{tag}_strain.npy"), strain_hist)
        np.save(os.path.join(out_dir, f"{tag}_stress.npy"), stress_hist)
        np.save(os.path.join(out_dir, f"{tag}_U.npy"), u_hist)
        np.save(os.path.join(out_dir, f"{tag}_applied_strain.npy"), e_applied_hist)
        np.save(os.path.join(out_dir, f"{tag}_q_m.npy"), qm_hist)
        np.save(os.path.join(out_dir, f"{tag}_q_s.npy"), qs_hist)
        if track_qpod:
            np.save(os.path.join(out_dir, f"{tag}_q_pod.npy"), qpod_hist)
        np.save(os.path.join(out_dir, f"{tag}_w_res.npy"), wres_hist)

    return strain_hist, stress_hist
