#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""HPROM-RBF solver (2D-MAWECM, classical ECM hyper-reduction).

Residual/Jacobian are assembled only on ECM-selected residual elements.
State update follows Joaquin-style master/slave decoder:
    q_s = N_slave(q_m)
    u = u_aff + Phi_m (A_m q_m) + Phi_s q_s
"""

import os
import sys
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
from prom_rbf_solver_rve import (
    LoadPromRbfModel,
    _recover_mu_from_E,
    _initial_qm_from_mu,
    _assert_qm_input_dimension,
    _decode_qs_from_qm,
    _decoder_jacobian_qs_wrt_qm,
)


_W_TOL = 1.0e-14


def _extract_hrom_aligned_ecm_arrays(ecm_data: Dict[str, object], n_elem: int):
    """
    Return residual support and full-length residual/homogenization weights aligned
    with the currently loaded mesh.

    Works with:
    - full mesh ECM data (Z_res, w_*_full)
    - HROM-aligned ECM data (hrom_element_full_indices + w_*_hrom)
    """
    z_res_raw = np.asarray(ecm_data.get("Z_res", np.array([], dtype=int)), dtype=np.int64).reshape(-1)
    w_res_full_raw = np.asarray(ecm_data.get("w_res_full", np.array([], dtype=float)), dtype=float).reshape(-1)
    w_eps_full_raw = np.asarray(ecm_data.get("w_eps_full", np.array([], dtype=float)), dtype=float).reshape(-1)
    w_sig_full_raw = np.asarray(ecm_data.get("w_sig_full", np.array([], dtype=float)), dtype=float).reshape(-1)

    if z_res_raw.size == 0:
        raise RuntimeError("ECM file must contain non-empty Z_res.")

    has_hrom_map = ("hrom_element_full_indices" in ecm_data) and ("hrom_n_elem" in ecm_data)
    if has_hrom_map:
        hrom_full_idx = np.asarray(ecm_data["hrom_element_full_indices"], dtype=np.int64).reshape(-1)
        hrom_n_elem = int(np.ravel(ecm_data["hrom_n_elem"])[0])
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
        and ("w_res_hrom" in ecm_data)
        and ("w_eps_hrom" in ecm_data)
        and ("w_sig_hrom" in ecm_data)
    )

    if using_hrom:
        w_res_full = np.asarray(ecm_data["w_res_hrom"], dtype=float).reshape(-1)
        w_eps_full = np.asarray(ecm_data["w_eps_hrom"], dtype=float).reshape(-1)
        w_sig_full = np.asarray(ecm_data["w_sig_hrom"], dtype=float).reshape(-1)
        if w_res_full.size != n_elem or w_eps_full.size != n_elem or w_sig_full.size != n_elem:
            raise RuntimeError(
                "HROM-projected ECM weights size mismatch with current mesh. "
                f"n_elem={n_elem}, len(w_res_hrom)={w_res_full.size}, "
                f"len(w_eps_hrom)={w_eps_full.size}, len(w_sig_hrom)={w_sig_full.size}"
            )

        full_to_local = {int(fid): int(i) for i, fid in enumerate(hrom_full_idx.tolist())}
        z_local = []
        missing = []
        for fid in z_res_raw.tolist():
            key = int(fid)
            if key in full_to_local:
                z_local.append(full_to_local[key])
            else:
                missing.append(key)
        if missing:
            # Fallback: if Z_res is already local indexing, keep it.
            z_try = np.asarray(z_res_raw, dtype=np.int64)
            if np.any(z_try < 0) or np.any(z_try >= n_elem):
                raise RuntimeError(
                    "Could not map Z_res from full mesh to HROM local indices. "
                    f"Missing full ids count={len(missing)} (example={missing[:8]})."
                )
            z_res = z_try
        else:
            z_res = np.asarray(z_local, dtype=np.int64)
        mesh_mode = "hrom"
    else:
        if w_res_full_raw.size != n_elem or w_eps_full_raw.size != n_elem or w_sig_full_raw.size != n_elem:
            raise RuntimeError(
                "ECM weights size mismatch with current mesh. "
                f"n_elem={n_elem}, len(w_res_full)={w_res_full_raw.size}, "
                f"len(w_eps_full)={w_eps_full_raw.size}, len(w_sig_full)={w_sig_full_raw.size}"
            )
        w_res_full = w_res_full_raw
        w_eps_full = w_eps_full_raw
        w_sig_full = w_sig_full_raw
        z_res = z_res_raw
        mesh_mode = "full"

    if np.any(z_res < 0) or np.any(z_res >= n_elem):
        raise RuntimeError("Z_res contains out-of-range element ids for the current mesh.")

    return z_res, w_res_full, w_eps_full, w_sig_full, mesh_mode


def _build_full_to_local_dof_map(full_mesh_base: str, mp_local, eq_map_local: np.ndarray):
    """
    Build map full_dof_index -> local_dof_index using node IDs as anchor.
    """
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


def LoadHpromRbfModel(
    stage2a_dir: str = "stage_2a_pod_data",
    stage3_dataset_file: str = "stage_3_prom_regression_dataset/stage3_prom_regression_dataset.npz",
    stage4_rbf_dir: str = "stage_4_prom_rbf_grid",
    ecm_file: str = "stage_6b_hprom_ecm/ecm_weights_all.npz",
    stage2b_dir: str = "stage_2b_ls_master",
):
    model_pack = LoadPromRbfModel(
        stage2a_dir=stage2a_dir,
        stage3_dataset_file=stage3_dataset_file,
        stage4_rbf_dir=stage4_rbf_dir,
        stage2b_dir=stage2b_dir,
    )
    if not os.path.exists(ecm_file):
        raise FileNotFoundError(f"ECM file not found: {ecm_file}")
    ecm_npz = np.load(ecm_file, allow_pickle=True)
    ecm_data = {k: ecm_npz[k] for k in ecm_npz.files}
    return model_pack, ecm_data


def RunHpromRbfBatchSimulation(
    parameters,
    model_pack: Dict[str, object],
    ecm_data: Dict[str, object],
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
    prom_use_old_stiffness_in_first_iteration=True,
    prom_old_stiffness_residual_cutoff=1.0e5,
    prom_corrector_l2_reg=1.0e-10,
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
    target_space = str(model_pack["target_space"])
    mu_space = str(model_pack["mu_space"])
    mapping = str(model_pack["mapping"])
    q_m_dim = int(model_pack["q_m_dim"])
    q_s_dim = int(model_pack["q_s_dim"])
    q_pod_dim = int(model_pack["q_pod_dim"])
    rbf_input_space = str(model_pack.get("rbf_input_space", "mu")).strip().lower()

    mu_mean = np.asarray(model_pack["mu_mean"], dtype=float)
    mu_std = np.asarray(model_pack["mu_std"], dtype=float)
    x_mean = np.asarray(model_pack.get("x_mean", mu_mean), dtype=float)
    x_std = np.asarray(model_pack.get("x_std", mu_std), dtype=float)
    y_mean = np.asarray(model_pack["y_mean"], dtype=float)
    y_std = np.asarray(model_pack["y_std"], dtype=float)
    rbf_model = model_pack["rbf_model"]
    _assert_qm_input_dimension(q_m_dim, x_mean)
    if rbf_input_space != "q_m":
        raise RuntimeError(
            "HPROM-RBF q_m-Newton requires Stage4 RBF trained with '--rbf-input-space q_m'. "
            f"Current model input_space='{rbf_input_space}'. Retrain Stage4."
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

    z_res, w_res_full, w_eps_full, w_sig_full, mesh_mode = _extract_hrom_aligned_ecm_arrays(
        ecm_data=ecm_data,
        n_elem=n_elem,
    )

    if mesh_mode == "hrom":
        if "hrom_full_mesh_base" not in ecm_data:
            raise RuntimeError(
                "ECM data has HROM-projected weights but missing 'hrom_full_mesh_base' for DOF remapping."
            )
        full_mesh_base = str(np.ravel(ecm_data["hrom_full_mesh_base"])[0])
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

    w_res_sel = w_res_full[z_res]
    elem_res = [elements_all[int(i)] for i in z_res.tolist()]

    z_hom = np.union1d(
        np.flatnonzero(np.abs(w_eps_full) > _W_TOL),
        np.flatnonzero(np.abs(w_sig_full) > _W_TOL),
    ).astype(np.int64)
    if z_hom.size == 0:
        z_hom = np.arange(n_elem, dtype=np.int64)
    elem_hom = [elements_all[int(i)] for i in z_hom.tolist()]
    w_eps_sel = w_eps_full[z_hom]
    w_sig_sel = w_sig_full[z_hom]

    assembler_hr = VectorizedAssembler(
        mp,
        n_dof,
        eq_map_runtime,
        elements=elem_res,
        element_scales=w_res_sel,
        log_label="HPROM-RBF-ResidualAssembler",
    )
    assembler_hom = VectorizedAssembler(
        mp,
        n_dof,
        eq_map_runtime,
        elements=elem_hom,
        element_scales=None,
        log_label="HPROM-RBF-HomAssembler",
    )

    if "A0_ref" in ecm_data:
        a0_ref = float(np.ravel(ecm_data["A0_ref"])[0])
    elif "hom_reference_measure" in ecm_data:
        a0_ref = float(np.ravel(ecm_data["hom_reference_measure"])[0])
    else:
        raise RuntimeError("ECM file missing A0_ref / hom_reference_measure.")

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

    results_eps = [np.zeros(3, dtype=float)]
    results_sig = [np.zeros(3, dtype=float)]
    u_hist = [np.zeros(n_dof, dtype=float)]
    e_applied_hist = [np.zeros(3, dtype=float)]
    qm_hist = [np.zeros(q_m_dim, dtype=float)]
    qs_hist = [np.zeros(q_s_dim, dtype=float)]
    qpod_hist = [np.zeros(q_pod_dim, dtype=float)]

    n_corr = max(0, int(prom_corrector_max_iters))
    print(
        f"  [HPROM-RBF] Solving trajectory with {n_steps_total} increments "
        f"(reduced corrector iters={n_corr})"
    )
    print(f"  [HPROM-RBF] Residual support: |Z_res|={z_res.size} ({100.0*z_res.size/max(n_elem,1):.1f}% of {n_elem})")
    print(
        f"  [HPROM-RBF] Hom support: |Z_hom|={z_hom.size} ({100.0*z_hom.size/max(n_elem,1):.1f}% of {n_elem}), "
        f"mesh_mode={mesh_mode}"
    )
    print(
        f"  [HPROM-RBF] DOF overlap full->mesh: free_used={n_overlap}/{free_dofs_full.size}"
    )

    q_m_state = None
    k_red_old = None
    for step in range(1, n_steps_total + 1):
        time_val = float(step) * float(dt)
        mp.CloneTimeStep(time_val)
        mp.ProcessInfo[KM.DELTA_TIME] = dt
        mp.ProcessInfo[KM.TIME] = time_val
        mp.ProcessInfo[KM.STEP] = step

        sim.time, sim.step, sim.end_time = time_val, step, end_time
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
        q_pod = np.zeros(q_pod_dim, dtype=float)

        def _decode_qs_local(qm_vec: np.ndarray) -> np.ndarray:
            return _decode_qs_from_qm(
                qm_vec,
                rbf_model=rbf_model,
                x_mean=x_mean,
                x_std=x_std,
                y_mean=y_mean,
                y_std=y_std,
                target_space=target_space,
                q_m_dim=q_m_dim,
                q_s_dim=q_s_dim,
            )
        _decode_qs_local._analytic_kwargs = {
            "rbf_model": rbf_model,
            "x_mean": x_mean,
            "x_std": x_std,
            "y_mean": y_mean,
            "y_std": y_std,
            "target_space": target_space,
            "q_m_dim": q_m_dim,
            "q_s_dim": q_s_dim,
        }

        u_aff_free = _affine_component(e, x_free, y_free, is_x_free)
        u_aff_dir = _affine_component(e, x_dir, y_dir, is_x_dir)
        u = np.zeros(n_dof, dtype=float)
        u[dir_dofs] = u_aff_dir

        converged = False
        nonfinite_detected = False
        nr0 = None
        nr_last = np.nan
        ndq_last = np.nan
        ndq0 = None
        nr_prev = None
        nr_best = np.inf
        rrel_best = np.inf
        q_m_step_start = q_m.copy()
        q_m_best = q_m.copy()
        k_red_last = None
        for it in range(n_corr + 1):
            mp.ProcessInfo[KM.NL_ITERATION_NUMBER] = it + 1

            q_s = _decode_qs_local(q_m)
            q_master = a_m @ q_m
            q_pod = (c_m @ q_master) + (c_s @ q_s)
            u[free_dofs] = u_aff_free + (phi_m @ q_master) + (phi_s @ q_s)

            SetDisplacementFromEquationVector(u, eq_map_runtime, ta_disp)
            UpdateCurrentCoordinatesFromDisplacement(mp, step=0)

            InitializeNonLinearIteration(entities, mp.ProcessInfo)
            k_hr, rhs_hr = assembler_hr.Assemble(u)
            FinalizeNonLinearIteration(entities, mp.ProcessInfo)

            r_f = rhs_hr[free_dofs]
            dqs_dqm = _decoder_jacobian_qs_wrt_qm(
                q_m,
                _decode_qs_local,
                q_s_dim=q_s_dim,
            )
            du_dqm = (phi_m @ a_m) + (phi_s @ dqs_dqm)
            r_red = du_dqm.T @ r_f
            nr = float(np.linalg.norm(r_red))
            if not np.isfinite(nr):
                nonfinite_detected = True
                break
            nr_last = nr
            if nr0 is None:
                nr0 = max(nr, 1e-30)
            r_rel = nr / nr0
            if nr < nr_best:
                nr_best = nr
                rrel_best = r_rel
                q_m_best = q_m.copy()

            kff = k_hr[free_dofs][:, free_dofs]
            k_red = du_dqm.T @ (kff @ du_dqm)
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
                print(
                    f"  > It {it:02d}: ||R_r|| = {nr:.3e}, rel = {r_rel:.3e} -> converged(rel)"
                )
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
                dq = np.linalg.solve(k_solve, r_red)
            except np.linalg.LinAlgError:
                dq = np.linalg.lstsq(k_solve, r_red, rcond=None)[0]
            if used_old and (not np.all(np.isfinite(dq))):
                try:
                    dq = np.linalg.solve(k_red, r_red)
                    used_old = False
                except np.linalg.LinAlgError:
                    dq = np.linalg.lstsq(k_red, r_red, rcond=None)[0]
                    used_old = False

            ndq = float(np.linalg.norm(dq))
            ndq_last = ndq
            if ndq0 is None:
                ndq0 = max(ndq, 1.0e-30)
            dq_rel = ndq / ndq0
            if ndq > float(prom_corrector_max_dq_norm) and ndq > 0.0:
                scale = float(prom_corrector_max_dq_norm) / ndq
                dq = dq * scale
                ndq = float(np.linalg.norm(dq))
                ndq_last = ndq
                dq_rel = ndq / ndq0
            q_m = q_m + dq
            if used_old:
                print("    > using previous reduced stiffness (K_old) at first iteration")

            dq_ok = (ndq <= float(prom_corrector_dq_abs_tol)) or (dq_rel <= float(prom_corrector_dq_rel_tol))
            if dq_ok and nr <= float(prom_corrector_res_floor_for_dq):
                converged = True
                print(
                    f"  > It {it:02d}: ||R_r|| = {nr:.3e}, rel = {r_rel:.3e}, "
                    f"||dq_m|| = {ndq:.3e} (rel {dq_rel:.3e}) -> converged(dq)"
                )
                print(f"  > Converged in {it} iterations.")
                break

            print(
                f"  > It {it:02d}: ||R_r|| = {nr:.3e}, rel = {r_rel:.3e}"
            )

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
                    "  [HPROM-RBF] Step accepted as quasi-converged: "
                    f"best ||R_r||={nr_best:.3e}, rel={rrel_best:.3e}"
                )
            else:
                if np.isfinite(nr_best):
                    q_m = q_m_best.copy()
                else:
                    q_m = q_m_step_start.copy()
                print(
                    f"  [HPROM-RBF][WARN] Step {step:03d}/{n_steps_total} not converged in {n_corr} iters. "
                    f"last ||R_r||={nr_last:.3e}, ||dq_m||={ndq_last:.3e}"
                )
                if nonfinite_detected:
                    print("  [HPROM-RBF][WARN] non-finite state detected; rolled back to best finite iterate.")
            k_red_old = None
        elif k_red_last is not None and np.all(np.isfinite(k_red_last)):
            k_red_old = k_red_last.copy()
        else:
            k_red_old = None

        q_s = _decode_qs_local(q_m)
        q_master = a_m @ q_m
        q_pod = (c_m @ q_master) + (c_s @ q_s)
        u[free_dofs] = u_aff_free + (phi_m @ q_master) + (phi_s @ q_s)

        SetDisplacementFromEquationVector(u, eq_map_runtime, ta_disp)
        UpdateCurrentCoordinatesFromDisplacement(mp, step=0)
        InitializeNonLinearIteration(entities, mp.ProcessInfo)
        _, _ = assembler_hom.Assemble(u)
        FinalizeNonLinearIteration(entities, mp.ProcessInfo)

        eps_h, sig_h = CalculateHomogenizedFromAssemblerWithElementWeights(
            assembler_hom,
            w_eps=w_eps_sel,
            w_sig=w_sig_sel,
            reference_measure=float(a0_ref),
        )

        sim.FinalizeSolutionStep()

        results_eps.append(np.asarray(eps_h, dtype=float))
        results_sig.append(np.asarray(sig_h, dtype=float))
        u_hist.append(u.copy())
        e_applied_hist.append(e.copy())
        qm_hist.append(q_m.copy())
        qs_hist.append(q_s.copy())
        qpod_hist.append(q_pod.copy())
        q_m_state = q_m.copy()

        print(
            f"  [HPROM-RBF] Step {step:03d}/{n_steps_total}: "
            f"||q_m||={np.linalg.norm(q_m):.3e}, ||q_s||={np.linalg.norm(q_s):.3e}, ||q_pod||={np.linalg.norm(q_pod):.3e}"
        )

    sim.Finalize()

    strain_hist = np.stack(results_eps)
    stress_hist = np.stack(results_sig)
    u_hist = np.stack(u_hist)
    e_applied_hist = np.stack(e_applied_hist)
    qm_hist = np.stack(qm_hist)
    qs_hist = np.stack(qs_hist)
    qpod_hist = np.stack(qpod_hist)

    if out_dir is not None:
        tag = f"trajectory_{trajectory_index}" if trajectory_index is not None else "hprom_rbf_run"
        np.save(os.path.join(out_dir, f"{tag}_strain.npy"), strain_hist)
        np.save(os.path.join(out_dir, f"{tag}_stress.npy"), stress_hist)
        np.save(os.path.join(out_dir, f"{tag}_U.npy"), u_hist)
        np.save(os.path.join(out_dir, f"{tag}_applied_strain.npy"), e_applied_hist)
        np.save(os.path.join(out_dir, f"{tag}_q_m.npy"), qm_hist)
        np.save(os.path.join(out_dir, f"{tag}_q_s.npy"), qs_hist)
        np.save(os.path.join(out_dir, f"{tag}_q_pod.npy"), qpod_hist)

    return strain_hist, stress_hist
