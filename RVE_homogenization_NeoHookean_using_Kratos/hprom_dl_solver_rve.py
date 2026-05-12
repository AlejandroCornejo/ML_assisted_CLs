#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
HPROM-POD-DL solver:
  - Hyper-reduced residual assembly on ECM-selected elements
  - POD-DL latent manifold tangent projection
  - ECM-weighted homogenization with full-mesh reference measure A0
"""

import os
import sys
import numpy as np
import torch

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
    InitializeNonLinearIteration,
    FinalizeNonLinearIteration,
    BuildDynamicSegmentSteps,
    REFERENCE_STEPS_FOR_UNIT_AMPLITUDE,
    MIN_STEPS_PER_SEGMENT,
    USE_OLD_STIFFNESS_IN_FIRST_ITERATION,
    NEWTON_TOL_ABS,
)
from hprom_solver_rve import (
    AssembleHyperReducedSystem,
    ResolveResidualHyperReductionSelection,
    ResolveActiveFreeDofsAndBasisRows,
    ResolveHomogenizationWeightSelection,
    CalculateHomogenizedFromAssemblerWithElementWeights,
    GetReferenceIntegrationMeasureFromMesh,
)
from prom_dl_solver_rve import LoadPromDlModel


def LoadHpromDlModel(
    basis_dir="stage_2_pod_rve",
    pod_dl_data_dir="stage_7_pod_dl_data",
    hprom_dl_dir="stage_9_hprom_pod_dl_data",
):
    phi_q, free_dofs, dir_dofs, eq_map, pod_dl_model, device, checkpoint = LoadPromDlModel(
        basis_dir=basis_dir,
        pod_dl_data_dir=pod_dl_data_dir,
    )
    Xc, Yc = np.load(os.path.join(basis_dir, "domain_center.npy"))

    ecm = np.load(os.path.join(hprom_dl_dir, "ecm_weights_all.npz"), allow_pickle=True)
    ecm_data = {k: ecm[k] for k in ecm.files}

    return (
        phi_q,
        free_dofs,
        dir_dofs,
        eq_map,
        Xc,
        Yc,
        pod_dl_model,
        device,
        checkpoint,
        ecm_data,
    )


def _is_finite(arr):
    return bool(np.all(np.isfinite(np.asarray(arr))))


def _solve_reduced_system(K_sys, rhs, regularization=1.0e-10):
    try:
        dz = np.linalg.solve(K_sys, rhs)
    except np.linalg.LinAlgError:
        dz, *_ = np.linalg.lstsq(K_sys, rhs, rcond=None)
    if _is_finite(dz):
        return dz

    K_reg = K_sys + float(regularization) * np.eye(K_sys.shape[0], dtype=K_sys.dtype)
    try:
        dz = np.linalg.solve(K_reg, rhs)
    except np.linalg.LinAlgError:
        dz, *_ = np.linalg.lstsq(K_reg, rhs, rcond=None)
    return dz


def _decode_q_and_jacobian(model, z_state, q_ref):
    with torch.no_grad():
        q_map = model.decode_from_latent(z_state.unsqueeze(0)).reshape(-1)
        q_hat = q_map - q_ref
    q_np = q_hat.detach().cpu().numpy().reshape(-1)

    with torch.enable_grad():
        z_in = z_state.detach().clone().requires_grad_(True)

        def decode_only(z_vec):
            return model.decode_from_latent(z_vec.unsqueeze(0)).reshape(-1)

        j_q = torch.autograd.functional.jacobian(decode_only, z_in)
    j_q_np = j_q.detach().cpu().numpy()
    if j_q_np.ndim == 1:
        j_q_np = j_q_np.reshape(-1, 1)
    return q_np, j_q_np


def RunHpromDlBatchSimulation(
    parameters,
    phi_q,
    free_dofs,
    pod_dl_model,
    device,
    ecm_data,
    out_dir="stage_10_hprom_dl_results",
    strain_path=None,
    trajectory_index=None,
    relnorm_cutoff=1e-5,
    max_its=25,
    abs_res_cutoff=NEWTON_TOL_ABS,
    dz_abs_cutoff=1.0e-6,
    max_res_for_rel_convergence=1.0e-1,
    min_rel_drop_stop=1.0e-2,
    stagnation_relnorm_gate=1.0e-4,
    max_dz_norm=0.5,
    old_stiffness_residual_cutoff=1.0e5,
    regularization=1.0e-10,
    use_fast_dirichlet_bc=True,
    reference_amplitude=None,
    reference_steps=REFERENCE_STEPS_FOR_UNIT_AMPLITUDE,
    use_old_stiffness_in_first_iteration=USE_OLD_STIFFNESS_IN_FIRST_ITERATION,
    eq_map_full=None,
    Xc=None,
    Yc=None,
):
    os.makedirs(out_dir, exist_ok=True)

    free_dofs_ref = np.asarray(free_dofs, dtype=np.int64).reshape(-1)
    phi_q_ref = np.asarray(phi_q, dtype=np.float64)
    if phi_q_ref.ndim != 2:
        raise ValueError(f"phi_q must be 2D, got shape {phi_q_ref.shape}.")
    if phi_q_ref.shape[0] != free_dofs_ref.size:
        raise RuntimeError(
            f"[HPROM-DL] Basis/free_dofs mismatch: phi rows={phi_q_ref.shape[0]} "
            f"!= len(free_dofs)={free_dofs_ref.size}."
        )

    n_q = int(phi_q_ref.shape[1])

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
    vec_full_assembler = VectorizedAssembler(mp, n_total_dof, eq_id_map, log_label="HPROMDLFullSyncAssembler")
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
        solver_label="HPROM-DL",
    )
    phi_q = phi_q_ref[basis_rows, :]

    if phi_q.shape[0] != len(free_dofs):
        raise RuntimeError(
            f"[HPROM-DL] Active basis/free_dofs mismatch: phi rows={phi_q.shape[0]}, free dofs={len(free_dofs)}"
        )

    Z_res, w_res_selected, using_hrom_mesh = ResolveResidualHyperReductionSelection(
        ecm_data,
        n_current_elements=len(elements),
        solver_label="HPROM-DL",
    )
    Z_union = np.asarray(ecm_data["Z_union"], dtype=np.int64).reshape(-1) if "Z_union" in ecm_data else Z_res
    n_elem_reference = int(np.ravel(ecm_data["n_elem"])[0]) if "n_elem" in ecm_data else len(elements)

    if Xc is None or Yc is None:
        sim._InitializeDomainCenterIfNeeded(mp)
        x0c, y0c = float(sim._x0c), float(sim._y0c)
    else:
        x0c, y0c = float(Xc), float(Yc)

    w_eps_hom, w_sig_hom, using_weighted_hom = ResolveHomogenizationWeightSelection(
        ecm_data,
        n_current_elements=len(elements),
        solver_label="HPROM-DL",
    )
    hom_reference_measure = GetReferenceIntegrationMeasureFromMesh(full_mesh_base)
    print(f"  [HPROM-DL] Homogenization reference measure A0 (full mesh): {hom_reference_measure:.6e}")
    with open(os.path.join(out_dir, "reference_measure_A0.txt"), "w", encoding="utf-8") as f:
        f.write(f"{float(hom_reference_measure):.16e}\n")
    if not using_weighted_hom:
        raise RuntimeError(
            "[HPROM-DL] ECM homogenization weights are required (w_eps/w_sig not available)."
        )
    print("  [HPROM-DL] Using ECM-weighted homogenization (w_eps / w_sig).")

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

    with torch.no_grad():
        q_zero = torch.zeros((1, n_q), dtype=torch.float32, device=device)
        z_ref = pod_dl_model.encode(q_zero).reshape(-1)
        q_ref = pod_dl_model.decode_from_latent(z_ref.unsqueeze(0)).reshape(-1)

    z_state = z_ref.detach().clone()
    n_latent = int(z_state.numel())
    Kr_old = None

    results_eps = [np.zeros(3, dtype=float)]
    results_sig = [np.zeros(3, dtype=float)]
    z_hist = [z_state.detach().cpu().numpy().reshape(-1)]

    print(f"  [HPROM-DL] Solving for {n_steps_total} dynamic increments...")
    print(f"  [HPROM-DL] reduced dims: q_dim={n_q}, latent_dim={n_latent}")
    print(f"  [HPROM-DL] ||decode(encode(0))|| = {float(torch.norm(q_ref).cpu().item()):.3e}")
    print(f"  [HPROM-DL] Active mesh elements: {len(elements)} (reference full mesh: {n_elem_reference})")
    print(
        f"  [HPROM-DL] ECM residual elements: {len(Z_res)} / {len(elements)} | "
        f"union reference size: {len(Z_union)} / {n_elem_reference}"
    )
    if using_hrom_mesh:
        print("  [HPROM-DL] Using reduced HROM mesh residual weights (w_res_hrom).")

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
            print(f"\n[HPROM-DL] Step {step:04d}/{n_steps_total} | E={E}")

        it = 0
        res_norm_0 = None
        converged = False
        nonfinite_detected = False
        Kr_last = None
        dz_norm_prev = None
        prev_res_norm = None
        z_step_start = z_state.detach().clone()
        best_z = z_step_start.detach().clone()
        best_res = np.inf
        best_rel = np.inf

        while it < max_its:
            mp.ProcessInfo[KM.NL_ITERATION_NUMBER] = it + 1
            q_np, j_q = _decode_q_and_jacobian(pod_dl_model, z_state, q_ref)
            z_norm = float(torch.norm(z_state).detach().cpu().item())
            q_norm = float(np.linalg.norm(q_np))
            print(f"    > ||z||: {z_norm:.3e} | ||q||: {q_norm:.3e}")

            if (not _is_finite(q_np)) or (not _is_finite(j_q)):
                print("  [HPROM-DL] WARNING: non-finite decoded state or decoder Jacobian detected.")
                nonfinite_detected = True
                break
            if j_q.shape != (n_q, n_latent):
                raise RuntimeError(
                    f"Invalid decoder Jacobian shape {j_q.shape}; expected ({n_q}, {n_latent})."
                )

            u_fluc = phi_q @ q_np
            if not _is_finite(u_fluc):
                print("  [HPROM-DL] WARNING: non-finite reconstructed displacement detected.")
                nonfinite_detected = True
                break
            u_free = u_aff_free + u_fluc

            J_manifold = phi_q @ j_q
            if not _is_finite(J_manifold):
                print("  [HPROM-DL] WARNING: non-finite manifold Jacobian detected.")
                nonfinite_detected = True
                break

            _apply_total_free_displacement(u_free, base_disp_vec=disp_base_step)

            InitializeNonLinearIteration(entities, mp.ProcessInfo)
            u_curr = _capture_current_displacement_vector()
            K_sparse, rhs_vec = AssembleHyperReducedSystem(
                mp,
                n_total_dof,
                elements,
                Z_res,
                w_res_selected,
                u_eq=u_curr,
            )
            FinalizeNonLinearIteration(entities, mp.ProcessInfo)

            r_full = rhs_vec[free_dofs]
            K_full = K_sparse[free_dofs][:, free_dofs].toarray()

            if (not _is_finite(r_full)) or (not _is_finite(K_full)):
                print("  [HPROM-DL] WARNING: non-finite full residual/stiffness detected.")
                nonfinite_detected = True
                break

            r_r = J_manifold.T @ r_full
            K_r = J_manifold.T @ K_full @ J_manifold
            Kr_last = K_r
            if (not _is_finite(r_r)) or (not _is_finite(K_r)):
                print("  [HPROM-DL] WARNING: non-finite reduced residual/stiffness detected.")
                nonfinite_detected = True
                break

            res_norm = float(np.linalg.norm(r_r))
            if not np.isfinite(res_norm):
                print("  [HPROM-DL] WARNING: non-finite reduced residual norm detected.")
                nonfinite_detected = True
                break
            if res_norm_0 is None:
                res_norm_0 = max(res_norm, 1e-30)
            rel_res = res_norm / (res_norm_0 + 1e-12)

            if res_norm < best_res:
                best_res = float(res_norm)
                best_z = z_state.detach().clone()
                best_rel = float(rel_res)

            print(f"  > It {it:02d}: ||R_r|| = {res_norm:.3e}, rel = {rel_res:.3e}")

            if res_norm < float(abs_res_cutoff):
                print(f"  > Converged in {it} iterations.")
                converged = True
                break
            if (
                dz_norm_prev is not None
                and dz_norm_prev < float(dz_abs_cutoff)
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
                        f"  > Converged in {it} iterations (stagnation criterion: "
                        f"rel_drop={rel_drop:.3e})."
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

            dz = _solve_reduced_system(K_solve, r_r, regularization=regularization)
            if used_old and not _is_finite(dz):
                dz = _solve_reduced_system(K_r, r_r, regularization=regularization)
                used_old = False
            if not _is_finite(dz):
                print("  [HPROM-DL] WARNING: non-finite reduced update detected.")
                nonfinite_detected = True
                break

            dz_norm = float(np.linalg.norm(dz))
            if dz_norm > float(max_dz_norm) and dz_norm > 0.0:
                scale = float(max_dz_norm) / dz_norm
                dz *= scale
                dz_norm = float(np.linalg.norm(dz))
                print(f"    > large reduced update clipped with scale={scale:.3e}")

            alpha = 1.0 if it <= 10 else 0.5
            with torch.no_grad():
                dz_torch = torch.from_numpy(dz.astype(np.float32)).to(device)
                z_trial = z_state + alpha * dz_torch
                if not _is_finite(z_trial.detach().cpu().numpy()):
                    print("  [HPROM-DL] WARNING: non-finite latent update detected.")
                    nonfinite_detected = True
                    break
                z_state = z_trial

            dz_norm_prev = abs(alpha) * dz_norm
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
                z_state = best_z.detach().clone()
                converged = True
                print(
                    "  [HPROM-DL] Step accepted as quasi-converged: "
                    f"best ||R_r||={best_res:.3e}, rel={best_rel:.3e}."
                )
                Kr_old = None
            else:
                print(f"  [HPROM-DL] WARNING: step {step} did not converge in {max_its} iterations.")
                if nonfinite_detected:
                    print("  [HPROM-DL] WARNING: non-finite state encountered; rolling back to best finite iterate.")
                if np.isfinite(best_res):
                    z_state = best_z.detach().clone()
                    print(f"  [HPROM-DL] Using best finite iterate with ||R_r||={best_res:.3e}.")
                else:
                    z_state = z_step_start.detach().clone()
                    print("  [HPROM-DL] Reverting to previous-step latent state.")
                Kr_old = None
        elif Kr_last is not None and _is_finite(Kr_last):
            Kr_old = Kr_last.copy()
        else:
            Kr_old = None

        q_final, _ = _decode_q_and_jacobian(pod_dl_model, z_state, q_ref)
        u_fluc_final = phi_q @ q_final
        if not _is_finite(u_fluc_final):
            z_state = z_step_start.detach().clone()
            q_final, _ = _decode_q_and_jacobian(pod_dl_model, z_state, q_ref)
            u_fluc_final = phi_q @ q_final
        if not _is_finite(u_fluc_final):
            raise RuntimeError("HPROM-DL accepted state is non-finite after rollback.")

        if not use_fast_dirichlet_bc:
            sim.ApplyBoundaryConditions()
            disp_base_step = _capture_current_displacement_vector()
        u_eq_final = _apply_total_free_displacement(u_aff_free + u_fluc_final, base_disp_vec=disp_base_step)

        sim.FinalizeSolutionStep()

        InitializeNonLinearIteration(entities, mp.ProcessInfo)
        _, _ = vec_full_assembler.Assemble(u_eq_final)
        FinalizeNonLinearIteration(entities, mp.ProcessInfo)

        eps_h, sig_h = CalculateHomogenizedFromAssemblerWithElementWeights(
            vec_full_assembler,
            w_eps=w_eps_hom,
            w_sig=w_sig_hom,
            reference_measure=hom_reference_measure,
        )
        results_eps.append(eps_h)
        results_sig.append(sig_h)
        z_hist.append(z_state.detach().cpu().numpy().reshape(-1))

    sim.Finalize()
    np.save(os.path.join(out_dir, "hprom_dl_run_z.npy"), np.asarray(z_hist, dtype=float))
    return np.array(results_eps), np.array(results_sig)
