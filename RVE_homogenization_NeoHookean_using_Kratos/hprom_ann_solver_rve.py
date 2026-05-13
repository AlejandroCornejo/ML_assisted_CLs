#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
HPROM-ANN solver:
  - Hyper-reduced assembly on ECM-selected elements
  - Nonlinear ANN manifold projection with tangent Jacobian
"""

import os
import sys
import time
import numpy as np
import torch
try:
    from torch.func import jacfwd as _torch_jacfwd
except Exception:
    _torch_jacfwd = None

# Add Kratos path
KRATOS_PATH = "/home/kratos/Kratos_Eigen_Check/bin/Release"
if KRATOS_PATH not in sys.path:
    sys.path.append(KRATOS_PATH)

import KratosMultiphysics as KM
from fom_solver_rve import VectorizedAssembler
from fom_solver_rve import (
    DeformationGradientFromGreenLagrange2D,
    RVEHomogenizationDatasetGenerator,
    SetUpDofEquationIdsAndDisplacementAdaptor,
    SetDisplacementFromEquationVector,
    UpdateCurrentCoordinatesFromDisplacement,
    AssembleGlobalSystem,
    InitializeNonLinearIteration,
    FinalizeNonLinearIteration,
    BuildDynamicSegmentSteps,
    CalculateHomogenizedStressAndStrainKratosReference,
    REFERENCE_STEPS_FOR_UNIT_AMPLITUDE,
    MIN_STEPS_PER_SEGMENT,
    USE_OLD_STIFFNESS_IN_FIRST_ITERATION,
    NEWTON_TOL_ABS,
)
from hprom_solver_rve import (
    AssembleHyperReducedSystem,
    ResetHyperReductionAssemblyStats,
    GetHyperReductionAssemblyStats,
    ResolveResidualHyperReductionSelection,
    ResolveActiveFreeDofsAndBasisRows,
    ResolveHomogenizationWeightSelection,
    CalculateHomogenizedFromAssemblerWithElementWeights,
    GetReferenceIntegrationMeasureFromMesh,
)
from prom_ann_solver_rve import LoadPromAnnModel


def LoadHpromAnnModel(
    basis_dir="stage_2_pod_rve",
    ann_data_dir="stage_7_ann_data",
    hprom_ann_dir="stage_9_hprom_ann_data",
):
    phi_p, phi_s, free_dofs, dir_dofs, eq_map, ann_model, device, include_macro = LoadPromAnnModel(
        basis_dir=basis_dir,
        ann_data_dir=ann_data_dir,
    )
    Xc, Yc = np.load(os.path.join(basis_dir, "domain_center.npy"))
    ecm = np.load(os.path.join(hprom_ann_dir, "ecm_weights_all.npz"))
    ecm_data = {k: ecm[k] for k in ecm.files}

    return (
        phi_p,
        phi_s,
        free_dofs,
        dir_dofs,
        eq_map,
        Xc,
        Yc,
        ann_model,
        device,
        ecm_data,
        include_macro,
    )


def RunHpromAnnBatchSimulation(
    parameters,
    phi_p,
    phi_s,
    free_dofs,
    ann_model,
    device,
    ecm_data,
    out_dir="stage_10_hprom_ann_results",
    strain_path=None,
    trajectory_index=None,
    relnorm_cutoff=1e-5,
    max_its=25,
    abs_res_cutoff=NEWTON_TOL_ABS,
    dq_abs_cutoff=1.0e-6,
    max_res_for_rel_convergence=1.0e-1,
    min_rel_drop_stop=1.0e-2,
    stagnation_relnorm_gate=1.0e-4,
    max_dq_norm=0.5,
    old_stiffness_residual_cutoff=1.0e5,
    regularization=1.0e-10,
    reference_amplitude=None,
    reference_steps=REFERENCE_STEPS_FOR_UNIT_AMPLITUDE,
    use_old_stiffness_in_first_iteration=USE_OLD_STIFFNESS_IN_FIRST_ITERATION,
    verbose_iterations=False,
    use_fast_dirichlet_bc=True,
    eq_map_full=None,
    Xc=None,
    Yc=None,
    return_stats=False,
):
    t_wall_total_start = time.perf_counter()
    os.makedirs(out_dir, exist_ok=True)

    free_dofs_ref = np.asarray(free_dofs, dtype=np.int64).reshape(-1)
    phi_p_ref = np.asarray(phi_p, dtype=float)
    phi_s_ref = np.asarray(phi_s, dtype=float)
    if phi_p_ref.shape[0] != phi_s_ref.shape[0]:
        raise ValueError("phi_p and phi_s must have the same number of rows (n_free).")
    if phi_p_ref.shape[0] != free_dofs_ref.size:
        raise RuntimeError(
            f"[HPROM-ANN] Basis/free_dofs mismatch: phi rows={phi_p_ref.shape[0]} "
            f"!= len(free_dofs)={free_dofs_ref.size}."
        )

    n_primary = int(phi_p_ref.shape[1])
    n_secondary = int(phi_s_ref.shape[1])

    dt = parameters["solver_settings"]["time_stepping"]["time_step"].GetDouble()
    E_wp = np.array(strain_path, dtype=float)
    n_seg = len(E_wp) - 1
    seg_steps, _ = BuildDynamicSegmentSteps(
        E_wp,
        reference_steps=reference_steps,
        min_steps=MIN_STEPS_PER_SEGMENT,
        reference_amplitude=reference_amplitude,
    )
    step_offsets = np.concatenate(([0], np.cumsum(seg_steps)))
    n_steps_total = int(step_offsets[-1])
    end_time = dt * float(n_steps_total)

    model_kratos = KM.Model()
    sim = RVEHomogenizationDatasetGenerator(model_kratos, parameters)
    sim.Initialize()
    mp = sim._GetSolver().GetComputingModelPart()

    n_total_dof, eq_id_map, ta = SetUpDofEquationIdsAndDisplacementAdaptor(mp)
    vec_full_assembler = VectorizedAssembler(mp, n_total_dof, eq_id_map, log_label="HPROMANNFullSyncAssembler")
    elements = list(mp.Elements)
    entities = list(mp.Elements) + list(mp.Conditions)
    full_mesh_base = str(np.ravel(ecm_data["hrom_full_mesh_base"])[0]) if "hrom_full_mesh_base" in ecm_data else "rve_geometry"
    free_dofs, dir_dofs, basis_rows = ResolveActiveFreeDofsAndBasisRows(
        mp,
        n_total_dof,
        eq_id_map,
        free_dofs_reference=free_dofs_ref,
        eq_map_reference=eq_map_full,
        full_mesh_base=full_mesh_base,
        solver_label="HPROM-ANN",
    )
    phi_p = phi_p_ref[basis_rows, :]
    phi_s = phi_s_ref[basis_rows, :]
    Z_res, w_res_selected, using_hrom_mesh = ResolveResidualHyperReductionSelection(
        ecm_data,
        n_current_elements=len(elements),
        solver_label="HPROM-ANN",
    )
    Z_union = np.asarray(ecm_data["Z_union"], dtype=np.int64).reshape(-1) if "Z_union" in ecm_data else Z_res
    n_elem_reference = int(np.ravel(ecm_data["n_elem"])[0]) if "n_elem" in ecm_data else len(elements)

    if Xc is None or Yc is None:
        sim._InitializeDomainCenterIfNeeded(mp)
        x0c, y0c = float(sim._x0c), float(sim._y0c)
    else:
        # Keep affine lifting center consistent with basis/training reference mesh.
        x0c, y0c = float(Xc), float(Yc)

    w_eps_hom, w_sig_hom, using_weighted_hom = ResolveHomogenizationWeightSelection(
        ecm_data,
        n_current_elements=len(elements),
        solver_label="HPROM-ANN",
    )
    hom_reference_measure = GetReferenceIntegrationMeasureFromMesh(full_mesh_base)
    print(f"  [HPROM-ANN] Homogenization reference measure A0 (full mesh): {hom_reference_measure:.6e}")
    with open(os.path.join(out_dir, "reference_measure_A0.txt"), "w", encoding="utf-8") as f:
        f.write(f"{float(hom_reference_measure):.16e}\n")
    if not using_weighted_hom:
        raise RuntimeError(
            "[HPROM-ANN] ECM homogenization weights are required (w_eps/w_sig not available)."
        )
    print("  [HPROM-ANN] Using ECM-weighted homogenization (w_eps / w_sig).")
    dof_x = np.zeros(n_total_dof, dtype=float)
    dof_y = np.zeros(n_total_dof, dtype=float)
    is_x_dof = np.zeros(n_total_dof, dtype=bool)
    for i, node in enumerate(mp.Nodes):
        xr = float(node.X0) - x0c
        yr = float(node.Y0) - y0c
        idx_x = int(eq_id_map[i, 0])
        idx_y = int(eq_id_map[i, 1])
        if 0 <= idx_x < n_total_dof:
            dof_x[idx_x] = xr
            dof_y[idx_x] = yr
            is_x_dof[idx_x] = True
        if 0 <= idx_y < n_total_dof:
            dof_x[idx_y] = xr
            dof_y[idx_y] = yr
            is_x_dof[idx_y] = False
    x_free = dof_x[free_dofs]
    y_free = dof_y[free_dofs]
    is_x_free = is_x_dof[free_dofs]
    x_dir = dof_x[dir_dofs]
    y_dir = dof_y[dir_dofs]
    is_x_dir = is_x_dof[dir_dofs]

    results_eps = [np.zeros(3, dtype=float)]
    results_sig = [np.zeros(3, dtype=float)]
    q_hist = [np.zeros(n_primary, dtype=float)]

    ann_input_dim = int(ann_model.input_scaler.mean.numel())
    expected_input_dim = int(n_primary)
    if ann_input_dim != expected_input_dim:
        raise ValueError(
            f"ANN input size mismatch: model expects {ann_input_dim}, "
            f"but solver was configured for {expected_input_dim}."
        )

    def _is_finite(arr):
        return bool(np.all(np.isfinite(np.asarray(arr))))

    def _compute_affine_component(e_vec, x_loc, y_loc, is_x_loc):
        F = DeformationGradientFromGreenLagrange2D(e_vec)
        ux = (F[0, 0] - 1.0) * x_loc + F[0, 1] * y_loc
        uy = F[1, 0] * x_loc + (F[1, 1] - 1.0) * y_loc
        return np.where(is_x_loc, ux, uy)

    def _compute_affine_free_displacement(e_vec):
        return _compute_affine_component(e_vec, x_free, y_free, is_x_free)

    def _compute_affine_dirichlet_displacement(e_vec):
        return _compute_affine_component(e_vec, x_dir, y_dir, is_x_dir)

    def _capture_current_displacement_vector():
        disp_vec = np.zeros(n_total_dof, dtype=float)
        for i, node in enumerate(mp.Nodes):
            d = node.GetSolutionStepValue(KM.DISPLACEMENT)
            idx_x, idx_y = eq_id_map[i, 0], eq_id_map[i, 1]
            if idx_x < n_total_dof:
                disp_vec[idx_x] = d[0]
            if idx_y < n_total_dof:
                disp_vec[idx_y] = d[1]
        return disp_vec

    def _apply_total_free_displacement(u_total_free, base_disp_vec=None):
        if base_disp_vec is None:
            disp_vec = _capture_current_displacement_vector()
        else:
            disp_vec = np.asarray(base_disp_vec, dtype=float).copy()
        disp_vec[free_dofs] = np.asarray(u_total_free, dtype=float).reshape(-1)
        SetDisplacementFromEquationVector(disp_vec, eq_id_map, ta)
        UpdateCurrentCoordinatesFromDisplacement(mp, step=0)
        return disp_vec

    def _build_ann_input(q_tensor, e_tensor):
        return q_tensor

    def _evaluate_qs_and_jac(qp_vec, e_vec):
        qp_arr = np.asarray(qp_vec, dtype=float).reshape(-1)
        e_arr = np.asarray(e_vec, dtype=float).reshape(3)

        qp_tensor = torch.from_numpy(qp_arr.astype(np.float32)).unsqueeze(0).to(device)
        e_tensor = torch.from_numpy(e_arr.astype(np.float32)).unsqueeze(0).to(device)

        with torch.no_grad():
            q_s_map_tensor = ann_model(_build_ann_input(qp_tensor, e_tensor))

        with torch.enable_grad():
            qp_in_vec = qp_tensor.squeeze(0).clone().detach().requires_grad_(True)

            def ann_from_qvec(q_vec):
                q_local = q_vec.unsqueeze(0)
                out = ann_model(_build_ann_input(q_local, e_tensor))
                return out.squeeze(0)

            if _torch_jacfwd is not None:
                try:
                    jac_ann = _torch_jacfwd(ann_from_qvec)(qp_in_vec)
                except Exception:
                    jac_ann = torch.autograd.functional.jacobian(ann_from_qvec, qp_in_vec)
            else:
                jac_ann = torch.autograd.functional.jacobian(ann_from_qvec, qp_in_vec)

        q_s_map = q_s_map_tensor.detach().cpu().numpy().reshape(-1)
        jac_np = jac_ann.detach().cpu().numpy()
        return q_s_map, jac_np

    def _solve_reduced_system(K_sys, rhs):
        try:
            dq_loc = np.linalg.solve(K_sys, rhs)
        except np.linalg.LinAlgError:
            dq_loc, *_ = np.linalg.lstsq(K_sys, rhs, rcond=None)
        if _is_finite(dq_loc):
            return dq_loc

        K_reg = K_sys + float(regularization) * np.eye(K_sys.shape[0], dtype=K_sys.dtype)
        try:
            dq_loc = np.linalg.solve(K_reg, rhs)
        except np.linalg.LinAlgError:
            dq_loc, *_ = np.linalg.lstsq(K_reg, rhs, rcond=None)
        return dq_loc

    q_p = np.zeros(n_primary, dtype=float)
    Kr_old = None
    q0_const, J0_const = _evaluate_qs_and_jac(np.zeros(n_primary, dtype=float), np.zeros(3, dtype=float))
    phi_p_eff = phi_p + phi_s @ J0_const
    w0_const = phi_s @ q0_const

    print(f"  [HPROM-ANN] Solving for {n_steps_total} dynamic increments...")
    print(f"  [HPROM-ANN] Active mesh elements: {len(elements)} (reference full mesh: {n_elem_reference})")
    print("  [HPROM-ANN] Manifold correction active: N(0)=0 and J(0)=0.")
    print(
        f"  [HPROM-ANN] ECM residual elements: {len(Z_res)} / {len(elements)} | "
        f"union reference size: {len(Z_union)} / {n_elem_reference}"
    )
    if using_hrom_mesh:
        print("  [HPROM-ANN] Using reduced HROM mesh residual weights (w_res_hrom).")
    t_map = 0.0
    t_assembly = 0.0
    t_projection = 0.0
    t_solve = 0.0
    t_full_sync = 0.0
    step_iters = []
    ResetHyperReductionAssemblyStats()

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
        E = (1.0 - xi) * E_wp[s, :] + xi * E_wp[s + 1, :]
        u_aff_free = _compute_affine_free_displacement(E)
        sim.batch_strain = E.copy()

        if use_fast_dirichlet_bc:
            disp_base_step = np.zeros(n_total_dof, dtype=float)
            disp_base_step[dir_dofs] = _compute_affine_dirichlet_displacement(E)
        else:
            sim.ApplyBoundaryConditions()
            disp_base_step = _capture_current_displacement_vector()

        if step == 1 or step % 100 == 0 or step == n_steps_total:
            print(f"\n[HPROM-ANN] Step {step:04d}/{n_steps_total} | E={E}")
        verbose_step = bool(verbose_iterations)

        it = 0
        res_norm_0 = None
        converged = False
        nonfinite_detected = False
        Kr_last = None
        dq_norm_prev = None
        prev_res_norm = None
        q_step_start = q_p.copy()
        best_q = q_step_start.copy()
        best_res = np.inf
        best_rel = np.inf
        it_step_count = 0

        while it < max_its:
            mp.ProcessInfo[KM.NL_ITERATION_NUMBER] = it + 1
            it_step_count += 1
            t0 = time.perf_counter()
            q_s_map, J_ann_raw = _evaluate_qs_and_jac(q_p, E)
            t_map += time.perf_counter() - t0
            q_s = q_s_map - q0_const - J0_const @ q_p
            J_ann = J_ann_raw - J0_const

            if verbose_step:
                print(f"    > q_p norm: {np.linalg.norm(q_p):.3e} | q_s norm: {np.linalg.norm(q_s):.3e}")
            if (not _is_finite(q_p)) or (not _is_finite(q_s)):
                print("  [HPROM-ANN] WARNING: non-finite reduced state detected.")
                nonfinite_detected = True
                break

            if J_ann.shape != (n_secondary, n_primary):
                raise RuntimeError(
                    f"Invalid ANN Jacobian shape {J_ann.shape}; expected ({n_secondary}, {n_primary})."
                )
            if not _is_finite(J_ann):
                print("  [HPROM-ANN] WARNING: non-finite ANN Jacobian detected.")
                nonfinite_detected = True
                break

            u_fluc = w0_const + phi_p_eff @ q_p + phi_s @ q_s
            if not _is_finite(u_fluc):
                print("  [HPROM-ANN] WARNING: non-finite reconstructed displacement detected.")
                nonfinite_detected = True
                break
            u_free = u_aff_free + u_fluc

            J_manifold = phi_p_eff + phi_s @ J_ann
            if not _is_finite(J_manifold):
                print("  [HPROM-ANN] WARNING: non-finite manifold Jacobian detected.")
                nonfinite_detected = True
                break

            u_eq_curr = _apply_total_free_displacement(u_free, base_disp_vec=disp_base_step)

            InitializeNonLinearIteration(entities, mp.ProcessInfo)
            t0 = time.perf_counter()
            K_hp, rhs_hp = AssembleHyperReducedSystem(
                mp, n_total_dof, elements, Z_res, w_res_selected, u_eq=u_eq_curr
            )
            t_assembly += time.perf_counter() - t0
            FinalizeNonLinearIteration(entities, mp.ProcessInfo)

            r_full = rhs_hp[free_dofs]
            K_free_sparse = K_hp[free_dofs][:, free_dofs]
            if (not _is_finite(r_full)) or (not _is_finite(K_free_sparse.data)):
                print("  [HPROM-ANN] WARNING: non-finite full residual/stiffness detected.")
                nonfinite_detected = True
                break

            t0 = time.perf_counter()
            KJ = K_free_sparse @ J_manifold
            r_r = J_manifold.T @ r_full
            K_r = J_manifold.T @ KJ
            t_projection += time.perf_counter() - t0
            Kr_last = K_r
            if (not _is_finite(r_r)) or (not _is_finite(K_r)):
                print("  [HPROM-ANN] WARNING: non-finite reduced residual/stiffness detected.")
                nonfinite_detected = True
                break

            res_norm = float(np.linalg.norm(r_r))
            if not np.isfinite(res_norm):
                print("  [HPROM-ANN] WARNING: non-finite reduced residual norm detected.")
                nonfinite_detected = True
                break
            if res_norm_0 is None:
                res_norm_0 = max(res_norm, 1e-30)
            rel_res = res_norm / (res_norm_0 + 1e-12)
            if res_norm < best_res:
                best_res = float(res_norm)
                best_q = q_p.copy()
                best_rel = float(rel_res)
            print(f"  > It {it:02d}: ||R_r|| = {res_norm:.3e}, rel = {rel_res:.3e}")

            if res_norm < float(abs_res_cutoff):
                print(f"  > Converged in {it} iterations.")
                converged = True
                break
            if (
                dq_norm_prev is not None
                and dq_norm_prev < float(dq_abs_cutoff)
                and rel_res < float(relnorm_cutoff)
                and res_norm < float(max_res_for_rel_convergence)
            ):
                print(f"  > Converged in {it} iterations (small update + reduced residual).")
                converged = True
                break
            if prev_res_norm is not None:
                rel_drop = abs(prev_res_norm - res_norm) / max(prev_res_norm, 1e-30)
                if (
                    rel_drop < float(min_rel_drop_stop)
                    and rel_res < float(stagnation_relnorm_gate)
                    and res_norm < float(max_res_for_rel_convergence)
                ):
                    print(
                        f"  > Converged in {it} iterations (stagnation criterion: rel_drop={rel_drop:.3e})."
                    )
                    converged = True
                    break
            prev_res_norm = float(res_norm)

            K_solve = K_r
            used_old = False
            if (
                it == 0
                and use_old_stiffness_in_first_iteration
                and Kr_old is not None
                and Kr_old.shape == K_r.shape
                and res_norm < float(old_stiffness_residual_cutoff)
            ):
                K_solve = Kr_old
                used_old = True

            t0 = time.perf_counter()
            dq_p = _solve_reduced_system(K_solve, r_r)
            if used_old and not _is_finite(dq_p):
                dq_p = _solve_reduced_system(K_r, r_r)
                used_old = False
            t_solve += time.perf_counter() - t0
            if not _is_finite(dq_p):
                print("  [HPROM-ANN] WARNING: non-finite reduced update detected.")
                nonfinite_detected = True
                break

            dq_norm = float(np.linalg.norm(dq_p))
            if dq_norm > float(max_dq_norm) and dq_norm > 0.0:
                scale = float(max_dq_norm) / dq_norm
                dq_p *= scale
                dq_norm = float(np.linalg.norm(dq_p))
                print(f"    > large reduced update clipped with scale={scale:.3e}")

            alpha = 1.0 if it <= 10 else 0.5
            q_trial = q_p + alpha * dq_p
            if not _is_finite(q_trial):
                print("  [HPROM-ANN] WARNING: non-finite reduced state update detected.")
                nonfinite_detected = True
                break
            q_p = q_trial
            dq_norm_prev = abs(alpha) * dq_norm
            if used_old:
                print("    > using previous reduced stiffness (K_old) at first iteration")
            it += 1

        if not converged:
            quasi_converged = (
                np.isfinite(best_res)
                and np.isfinite(best_rel)
                and (best_rel < float(relnorm_cutoff))
                and (best_res < float(max_res_for_rel_convergence))
            )
            if quasi_converged:
                q_p = best_q.copy()
                converged = True
                print(
                    "  [HPROM-ANN] Step accepted as quasi-converged: "
                    f"best ||R_r||={best_res:.3e}, rel={best_rel:.3e}."
                )
                Kr_old = None
            else:
                print(f"  [HPROM-ANN] WARNING: step {step} did not converge in {max_its} iterations.")
                if nonfinite_detected:
                    print("  [HPROM-ANN] WARNING: non-finite state encountered; rolling back to best finite iterate.")
                if np.isfinite(best_res):
                    q_p = best_q.copy()
                    print(f"  [HPROM-ANN] Using best finite iterate with ||R_r||={best_res:.3e}.")
                else:
                    q_p = q_step_start.copy()
                    print("  [HPROM-ANN] Reverting to previous-step reduced state.")
                Kr_old = None
        elif Kr_last is not None and _is_finite(Kr_last):
            Kr_old = Kr_last.copy()
        else:
            Kr_old = None

        q_s_final_map, _ = _evaluate_qs_and_jac(q_p, E)
        q_s_final = q_s_final_map - q0_const - J0_const @ q_p
        u_fluc_final = w0_const + phi_p_eff @ q_p + phi_s @ q_s_final
        if not _is_finite(u_fluc_final):
            q_p = q_step_start.copy()
            q_s_final_map, _ = _evaluate_qs_and_jac(q_p, E)
            q_s_final = q_s_final_map - q0_const - J0_const @ q_p
            u_fluc_final = w0_const + phi_p_eff @ q_p + phi_s @ q_s_final
        if not _is_finite(u_fluc_final):
            raise RuntimeError("HPROM-ANN accepted state is non-finite after rollback.")

        if not use_fast_dirichlet_bc:
            sim.ApplyBoundaryConditions()
            disp_base_step = _capture_current_displacement_vector()
        _apply_total_free_displacement(u_aff_free + u_fluc_final, base_disp_vec=disp_base_step)

        InitializeNonLinearIteration(entities, mp.ProcessInfo)
        u_curr = _capture_current_displacement_vector()
        t0 = time.perf_counter()
        _, _ = vec_full_assembler.Assemble(u_curr)
        t_full_sync += time.perf_counter() - t0
        FinalizeNonLinearIteration(entities, mp.ProcessInfo)

        if using_weighted_hom:
            hom_eps, hom_sig = CalculateHomogenizedFromAssemblerWithElementWeights(
                vec_full_assembler,
                w_eps=w_eps_hom,
                w_sig=w_sig_hom,
                reference_measure=hom_reference_measure,
            )
        else:
            hom_eps, hom_sig = CalculateHomogenizedStressAndStrainKratosReference(
                mp,
                reference_area_e=np.asarray(vec_full_assembler.area_e, dtype=float),
                reference_measure=hom_reference_measure,
            )
        sim.FinalizeSolutionStep()
        step_iters.append(int(it_step_count))

        q_hist.append(q_p.copy())
        results_eps.append(hom_eps)
        results_sig.append(hom_sig)

    sim.Finalize()
    t_wall_total = time.perf_counter() - t_wall_total_start
    t_accounted = t_map + t_assembly + t_projection + t_solve + t_full_sync
    t_other = max(t_wall_total - t_accounted, 0.0)
    iters = np.asarray(step_iters, dtype=float)
    hr_stats = GetHyperReductionAssemblyStats()
    timing_stats = {
        "n_steps": int(len(step_iters)),
        "newton_iters_total": int(np.sum(iters)) if iters.size else 0,
        "newton_iters_mean_per_step": float(np.mean(iters)) if iters.size else 0.0,
        "newton_iters_max_per_step": int(np.max(iters)) if iters.size else 0,
        "map": float(t_map),
        "assembly": float(t_assembly),
        "projection": float(t_projection),
        "solve": float(t_solve),
        "full_sync": float(t_full_sync),
        "accounted": float(t_accounted),
        "other": float(t_other),
        "total": float(t_wall_total),
    }
    for k, v in hr_stats.items():
        timing_stats[f"hr_{k}"] = float(v)
    if timing_stats["newton_iters_total"] > 0:
        timing_stats["mean_time_per_newton_iter"] = float(
            timing_stats["total"] / float(timing_stats["newton_iters_total"])
        )
        timing_stats["mean_map_time_per_newton_iter"] = float(
            timing_stats["map"] / float(timing_stats["newton_iters_total"])
        )
    else:
        timing_stats["mean_time_per_newton_iter"] = 0.0
        timing_stats["mean_map_time_per_newton_iter"] = 0.0

    print(
        f"\n[HPROM-ANN] Timing: map={t_map:.3f}s, assembly={t_assembly:.3f}s, "
        f"projection={t_projection:.3f}s, solve={t_solve:.3f}s, full_sync={t_full_sync:.3f}s, "
        f"accounted={t_accounted:.3f}s, other={t_other:.3f}s, total={t_wall_total:.3f}s, "
        f"iters(total={timing_stats['newton_iters_total']}, mean/step={timing_stats['newton_iters_mean_per_step']:.2f})"
    )
    print(
        f"  [HPROM-ANN] Hyper-assembly stats: calls={int(hr_stats['calls_total'])}, "
        f"vec={int(hr_stats['vectorized_calls'])}, local={int(hr_stats['local_calls'])}, "
        f"vec_fail={int(hr_stats['vectorized_failures'])}, "
        f"cache(rebuild={int(hr_stats['cache_rebuilds'])}, reuse={int(hr_stats['cache_reuses'])}), "
        f"u_eq_none={int(hr_stats['calls_without_u_eq'])}, "
        f"mean_vec={1e3*hr_stats['mean_vectorized_time_per_call']:.3f}ms, "
        f"mean_local={1e3*hr_stats['mean_local_time_per_call']:.3f}ms"
    )

    np.savez(os.path.join(out_dir, "hprom_ann_timing_stats.npz"), **{
        k: np.array([v], dtype=float) for k, v in timing_stats.items()
    })
    with open(os.path.join(out_dir, "hprom_ann_timing_stats.txt"), "w", encoding="utf-8") as f:
        for k, v in timing_stats.items():
            f.write(f"{k}={v}\n")

    tag = f"trajectory_{trajectory_index}" if trajectory_index is not None else "hprom_ann_run"
    np.save(os.path.join(out_dir, f"{tag}_q_p.npy"), np.stack(q_hist))
    np.save(os.path.join(out_dir, f"{tag}_strain.npy"), np.stack(results_eps))
    np.save(os.path.join(out_dir, f"{tag}_stress.npy"), np.stack(results_sig))

    if return_stats:
        return np.array(results_eps), np.array(results_sig), timing_stats
    return np.array(results_eps), np.array(results_sig)
