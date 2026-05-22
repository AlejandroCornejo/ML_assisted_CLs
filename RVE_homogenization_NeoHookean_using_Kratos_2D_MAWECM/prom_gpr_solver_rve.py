#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""PROM sparse-GPR solver (2D-MAWECM, online trajectory evaluation).

Master/slave reduced kinematics:
    q_master = A_m q_m
    q_s = N_gpr(q_m)
    u = u_aff + Phi_m q_master + Phi_s q_s

Unknown of Newton is q_m. The decoder Jacobian dq_s/dq_m is analytic from
the sparse-GP model (ARD RBF kernel).
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
)
from sparse_gp_manifold_model import (
    load_sparse_gp_model,
    evaluate_sparse_gp_map_and_jacobian_qp,
)


def _recover_mu_from_E(e_vec: np.ndarray, mapping: str, mu_space: str) -> np.ndarray:
    exx = float(e_vec[0])
    gamma = float(e_vec[2])
    if mapping == "small_strain":
        gx = exx
        gxy = gamma
    elif mapping == "green_lagrange_upper":
        disc = 1.0 + 2.0 * exx
        if disc <= 0.0:
            raise RuntimeError(
                f"Invalid strain state for Green-Lagrange inversion: 1+2*Exx={disc:.3e} <= 0."
            )
        gx = np.sqrt(disc) - 1.0
        den = 1.0 + gx
        if abs(den) <= 1e-14:
            raise RuntimeError("Invalid strain state: denominator (1+Gx) too small for Gxy recovery.")
        gxy = gamma / den
    else:
        raise RuntimeError(f"Unsupported mapping='{mapping}' for mu recovery.")

    if mu_space == "gx_gxy":
        return np.array([gx, gxy], dtype=float)
    if mu_space == "f11_f12":
        return np.array([1.0 + gx, gxy], dtype=float)
    raise RuntimeError(f"Unsupported mu_space='{mu_space}'.")


def _load_stage2b_array(stage2b_dir: str, name: str) -> np.ndarray:
    p = os.path.join(stage2b_dir, name)
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    return np.asarray(np.load(p), dtype=float)


def _initial_qm_from_mu(mu_vec: np.ndarray, model_pack: Dict[str, object]) -> np.ndarray:
    mu = np.asarray(mu_vec, dtype=float).reshape(-1)
    q_m_dim = int(model_pack["q_m_dim"])
    a_init = model_pack.get("q_m_init_from_mu_A", None)
    b_init = model_pack.get("q_m_init_from_mu_b", None)
    if a_init is not None and b_init is not None:
        a_init = np.asarray(a_init, dtype=float)
        b_init = np.asarray(b_init, dtype=float).reshape(-1)
        if a_init.shape[0] == q_m_dim and a_init.shape[1] == mu.size and b_init.size == q_m_dim:
            return (a_init @ mu + b_init).astype(float, copy=False)
    if mu.size == q_m_dim:
        return mu.copy()
    return np.zeros(q_m_dim, dtype=float)


def LoadPromGprModel(
    stage2a_dir: str = "stage_2a_pod_data",
    stage3_dataset_file: str = "stage_3_prom_regression_dataset/stage3_prom_regression_dataset.npz",
    stage4_gpr_dir: str = "stage_4_prom_gpr_sparse",
    stage2b_dir: str = "stage_2b_ls_master",
):
    phi_rom = np.asarray(np.load(os.path.join(stage2a_dir, "pod_basis_free.npy")), dtype=float)
    free_dofs = np.asarray(np.load(os.path.join(stage2a_dir, "free_dofs.npy")), dtype=np.int64)
    dir_dofs = np.asarray(np.load(os.path.join(stage2a_dir, "dirichlet_dofs.npy")), dtype=np.int64)
    eq_map = np.asarray(np.load(os.path.join(stage2a_dir, "eq_map.npy")), dtype=np.int64)

    stage3 = np.load(stage3_dataset_file, allow_pickle=True)
    q_m_dim = int(np.asarray(stage3["q_m_all"]).shape[1])
    q_s_dim = int(np.asarray(stage3["q_s_all"]).shape[1])
    q_pod_dim = int(np.asarray(stage3["q_pod_all"]).shape[1])
    mu_space = str(np.ravel(stage3["mu_space"])[0]) if "mu_space" in stage3 else "gx_gxy"
    mapping = str(np.ravel(stage3["mapping"])[0]) if "mapping" in stage3 else "green_lagrange_upper"
    stage3_target_space = str(np.ravel(stage3["target_space"])[0]) if "target_space" in stage3 else "q_s"

    a_m = np.asarray(stage3["A_m"], dtype=float) if "A_m" in stage3 else _load_stage2b_array(stage2b_dir, "A_m.npy")
    c_m = np.asarray(stage3["C_m"], dtype=float) if "C_m" in stage3 else _load_stage2b_array(stage2b_dir, "C_m.npy")
    c_s = np.asarray(stage3["C_s"], dtype=float) if "C_s" in stage3 else _load_stage2b_array(stage2b_dir, "C_s.npy")
    phi_m = _load_stage2b_array(stage2b_dir, "phi_m.npy")
    phi_s = _load_stage2b_array(stage2b_dir, "phi_s.npy")

    q_m_init_A_file = os.path.join(stage2b_dir, "q_m_init_from_mu_A.npy")
    q_m_init_b_file = os.path.join(stage2b_dir, "q_m_init_from_mu_b.npy")
    q_m_init_A = np.asarray(np.load(q_m_init_A_file), dtype=float) if os.path.exists(q_m_init_A_file) else None
    q_m_init_b = np.asarray(np.load(q_m_init_b_file), dtype=float) if os.path.exists(q_m_init_b_file) else None

    model_file = os.path.join(stage4_gpr_dir, "sparse_gp_model.npz")
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Sparse-GPR model not found: {model_file}")
    gpr_model = load_sparse_gp_model(model_file)

    meta_file = os.path.join(stage4_gpr_dir, "sparse_gp_meta.npz")
    if os.path.exists(meta_file):
        meta = np.load(meta_file, allow_pickle=True)
        input_space = str(np.ravel(meta["input_space"])[0]).strip().lower()
        target_space = str(np.ravel(meta["target_space"])[0]).strip().lower()
    else:
        input_space = "q_m"
        target_space = "q_s"

    if stage3_target_space != "q_s":
        raise RuntimeError(
            f"Stage3 target_space must be q_s for master/slave PROM-GPR. Got '{stage3_target_space}'."
        )
    if input_space != "q_m":
        raise RuntimeError(
            "PROM-GPR q_m-Newton requires Stage4 sparse-GPR trained with '--gpr-input-space q_m'. "
            f"Current model input_space='{input_space}'."
        )
    if target_space != "q_s":
        raise RuntimeError(
            "PROM-GPR master/slave requires Stage4 sparse-GPR target 'q_s'. "
            f"Current model target_space='{target_space}'."
        )

    if int(gpr_model["input_dim"]) != int(q_m_dim):
        raise RuntimeError(
            f"Sparse-GPR input_dim mismatch: model={int(gpr_model['input_dim'])}, q_m_dim={q_m_dim}."
        )
    if int(gpr_model["output_dim"]) != int(q_s_dim):
        raise RuntimeError(
            f"Sparse-GPR output_dim mismatch: model={int(gpr_model['output_dim'])}, q_s_dim={q_s_dim}."
        )

    if phi_rom.shape[1] != q_pod_dim:
        raise RuntimeError(
            f"POD basis columns ({phi_rom.shape[1]}) do not match q_pod_dim ({q_pod_dim})."
        )
    if phi_m.shape[0] != phi_rom.shape[0] or phi_s.shape[0] != phi_rom.shape[0]:
        raise RuntimeError("phi_m/phi_s row size mismatch with Stage2a free-DOF basis.")
    if phi_m.shape[1] != q_m_dim:
        raise RuntimeError(f"phi_m columns ({phi_m.shape[1]}) must match q_m_dim ({q_m_dim}).")
    if phi_s.shape[1] != q_s_dim:
        raise RuntimeError(f"phi_s columns ({phi_s.shape[1]}) must match q_s_dim ({q_s_dim}).")
    if a_m.shape != (q_m_dim, q_m_dim):
        raise RuntimeError(f"A_m shape mismatch: expected ({q_m_dim},{q_m_dim}), got {a_m.shape}.")
    if c_m.shape != (q_pod_dim, q_m_dim):
        raise RuntimeError(f"C_m shape mismatch: expected ({q_pod_dim},{q_m_dim}), got {c_m.shape}.")
    if c_s.shape != (q_pod_dim, q_s_dim):
        raise RuntimeError(f"C_s shape mismatch: expected ({q_pod_dim},{q_s_dim}), got {c_s.shape}.")

    return {
        "phi_rom": phi_rom,
        "phi_m": phi_m,
        "phi_s": phi_s,
        "A_m": a_m,
        "C_m": c_m,
        "C_s": c_s,
        "free_dofs": free_dofs,
        "dir_dofs": dir_dofs,
        "eq_map": eq_map,
        "mu_space": mu_space,
        "mapping": mapping,
        "q_m_dim": q_m_dim,
        "q_s_dim": q_s_dim,
        "q_pod_dim": q_pod_dim,
        "q_m_init_from_mu_A": q_m_init_A,
        "q_m_init_from_mu_b": q_m_init_b,
        "gpr_model": gpr_model,
        "model_input_space": input_space,
        "model_target_space": target_space,
    }


def RunPromGprBatchSimulation(
    parameters,
    model_pack: Dict[str, object],
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
    track_q_pod=0,
):
    if strain_path is None:
        raise ValueError("strain_path must be provided.")
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)

    phi_m = np.asarray(model_pack["phi_m"], dtype=float)
    phi_s = np.asarray(model_pack["phi_s"], dtype=float)
    a_m = np.asarray(model_pack["A_m"], dtype=float)
    c_m = np.asarray(model_pack["C_m"], dtype=float)
    c_s = np.asarray(model_pack["C_s"], dtype=float)
    free_dofs = np.asarray(model_pack["free_dofs"], dtype=np.int64)
    mu_space = str(model_pack["mu_space"])
    mapping = str(model_pack["mapping"])
    q_m_dim = int(model_pack["q_m_dim"])
    q_s_dim = int(model_pack["q_s_dim"])
    q_pod_dim = int(model_pack["q_pod_dim"])
    gpr_model = model_pack["gpr_model"]
    track_qpod = bool(int(track_q_pod))

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

    eq_map_ref = np.asarray(model_pack["eq_map"], dtype=np.int64)
    dir_dofs_ref = np.asarray(model_pack["dir_dofs"], dtype=np.int64)
    if eq_map_runtime.shape == eq_map_ref.shape and not np.array_equal(eq_map_runtime, eq_map_ref):
        raise RuntimeError("eq_map mismatch between runtime model and Stage2a metadata.")

    free_mask = np.zeros(n_dof, dtype=bool)
    free_mask[free_dofs] = True
    dir_dofs_runtime = np.nonzero(~free_mask)[0].astype(np.int64)
    if dir_dofs_ref.size == dir_dofs_runtime.size and not np.array_equal(dir_dofs_ref, dir_dofs_runtime):
        raise RuntimeError("dirichlet_dofs mismatch between runtime model and Stage2a metadata.")

    assembler = VectorizedAssembler(mp, n_dof, eq_map_runtime)
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

    x_free = dof_x[free_dofs]
    y_free = dof_y[free_dofs]
    is_x_free = is_x_dof[free_dofs]
    dir_dofs = dir_dofs_runtime
    x_dir = dof_x[dir_dofs]
    y_dir = dof_y[dir_dofs]
    is_x_dir = is_x_dof[dir_dofs]

    def _affine_component(e_vec, xx, yy, is_x):
        f = DeformationGradientFromGreenLagrange2D(e_vec)
        ux = (f[0, 0] - 1.0) * xx + f[0, 1] * yy
        uy = f[1, 0] * xx + (f[1, 1] - 1.0) * yy
        return np.where(is_x, ux, uy)

    # Manifold correction for consistency at origin: N(0)=0, J(0)=0.
    # Keep algebra equivalent to raw map by adding:
    #   u = u_aff + w0_const + phi_master_eff q_m + phi_s q_s_corr
    # where q_s_corr = N(q_m)-N0-J0 q_m, phi_master_eff = phi_master + phi_s J0, w0_const = phi_s N0.
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
    print("  [PROM-GPR] Manifold correction active: N(0)=0 and J(0)=0.")

    results_eps = [np.zeros(3, dtype=float)]
    results_sig = [np.zeros(3, dtype=float)]
    u_hist = [np.zeros(n_dof, dtype=float)]
    e_applied_hist = [np.zeros(3, dtype=float)]
    qm_hist = [np.zeros(q_m_dim, dtype=float)]
    qs_hist = [np.zeros(q_s_dim, dtype=float)]
    qpod_hist = [np.zeros(q_pod_dim, dtype=float)] if track_qpod else None

    n_corr = max(0, int(prom_corrector_max_iters))
    print(
        f"  [PROM-GPR] Solving trajectory with {n_steps_total} increments "
        f"(reduced corrector iters={n_corr})."
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
        q_pod = np.zeros(q_pod_dim, dtype=float) if track_qpod else None

        u_aff_free = _affine_component(e, x_free, y_free, is_x_free)
        u_aff_dir = _affine_component(e, x_dir, y_dir, is_x_dir)
        u = np.zeros(n_dof, dtype=float)
        u[dir_dofs] = u_aff_dir

        converged = False
        nonfinite_detected = False
        nrm0 = None
        nrm_prev = None
        nrm_last = np.nan
        ndq_last = np.nan
        ndq0 = None
        k_red_last = None

        for it in range(n_corr + 1):
            mp.ProcessInfo[KM.NL_ITERATION_NUMBER] = it + 1

            q_s_raw, dqs_raw = evaluate_sparse_gp_map_and_jacobian_qp(q_m, gpr_model, q_m_dim)
            q_s_raw = np.asarray(q_s_raw, dtype=float).reshape(-1)
            dqs_raw = np.asarray(dqs_raw, dtype=float)
            if q_s_raw.size != q_s_dim:
                raise RuntimeError(f"Sparse-GPR output size mismatch: got {q_s_raw.size}, expected {q_s_dim}.")
            if dqs_raw.shape != (q_s_dim, q_m_dim):
                raise RuntimeError(
                    f"Sparse-GPR Jacobian shape mismatch: got {dqs_raw.shape}, expected ({q_s_dim},{q_m_dim})."
                )
            q_s = q_s_raw - q_s0_raw - (j0_raw @ q_m)   # corrected slave coordinates used internally
            dqs_dqm = dqs_raw - j0_raw

            q_master = a_m @ q_m
            if track_qpod:
                q_pod = (c_m @ q_master) + (c_s @ q_s)
            q_s_phys = q_s_raw  # physical slave coordinates (for diagnostics/projections)
            u[free_dofs] = u_aff_free + w0_const + (phi_master_eff @ q_m) + (phi_s @ q_s)

            SetDisplacementFromEquationVector(u, eq_map_runtime, ta_disp)
            UpdateCurrentCoordinatesFromDisplacement(mp, step=0)

            InitializeNonLinearIteration(entities, mp.ProcessInfo)
            k_full, rhs = assembler.Assemble(u)
            FinalizeNonLinearIteration(entities, mp.ProcessInfo)

            r_f = rhs[free_dofs]
            du_dqm = phi_master_eff + (phi_s @ dqs_dqm)
            r_red = du_dqm.T @ r_f

            nrm = float(np.linalg.norm(r_red))
            if not np.isfinite(nrm):
                nonfinite_detected = True
                break
            nrm_last = nrm
            if nrm0 is None:
                nrm0 = max(nrm, 1.0e-30)
            r_rel = nrm / nrm0

            kff = k_full[free_dofs][:, free_dofs]
            k_red = du_dqm.T @ (kff @ du_dqm)
            if float(prom_corrector_l2_reg) > 0.0:
                k_red = k_red + float(prom_corrector_l2_reg) * np.eye(q_m_dim, dtype=float)
            k_red_last = k_red

            if nrm <= float(prom_corrector_abs_tol):
                converged = True
                print(f"  > It {it:02d}: ||R_r|| = {nrm:.3e}, rel = {r_rel:.3e}")
                print(f"  > Converged in {it} iterations.")
                break

            if (
                np.isfinite(ndq_last)
                and ndq_last <= float(prom_corrector_dq_abs_tol)
                and r_rel <= float(prom_corrector_rel_tol)
                and nrm <= float(prom_corrector_res_floor_for_dq)
            ):
                converged = True
                print(
                    f"  > It {it:02d}: ||R_r|| = {nrm:.3e}, rel = {r_rel:.3e}, "
                    f"||dq_m|| = {ndq_last:.3e} -> converged(small-dq+rel)"
                )
                print(f"  > Converged in {it} iterations.")
                break

            if nrm_prev is not None:
                rel_drop = abs(nrm_prev - nrm) / max(nrm_prev, 1.0e-30)
                if (
                    rel_drop <= float(prom_corrector_min_rel_drop_stop)
                    and r_rel <= float(prom_corrector_stagnation_relnorm_gate)
                    and nrm <= float(prom_corrector_res_floor_for_dq)
                ):
                    converged = True
                    print(
                        f"  > It {it:02d}: ||R_r|| = {nrm:.3e}, rel = {r_rel:.3e}, "
                        f"rel_drop={rel_drop:.3e} -> converged(stagnation)"
                    )
                    print(f"  > Converged in {it} iterations.")
                    break
            nrm_prev = nrm

            if r_rel <= float(prom_corrector_rel_tol) and nrm <= float(prom_corrector_res_floor_for_dq):
                converged = True
                print(f"  > It {it:02d}: ||R_r|| = {nrm:.3e}, rel = {r_rel:.3e} -> converged(rel)")
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
                and nrm <= float(prom_old_stiffness_residual_cutoff)
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

            alpha = 1.0
            if it > int(prom_corrector_damping_after_iter):
                alpha = float(prom_corrector_damping_factor)
                alpha = min(max(alpha, 1.0e-6), 1.0)
            dq_step = alpha * dq
            ndq = float(np.linalg.norm(dq_step))
            ndq_last = ndq
            dq_rel = ndq / ndq0
            q_m = q_m + dq_step
            if used_old:
                print("    > using previous reduced stiffness (K_old) at first iteration")
            if alpha < 1.0:
                print(f"    > reduced-step damping alpha={alpha:.3f}")

            dq_ok = (ndq <= float(prom_corrector_dq_abs_tol)) or (dq_rel <= float(prom_corrector_dq_rel_tol))
            if dq_ok and nrm <= float(prom_corrector_res_floor_for_dq):
                converged = True
                print(
                    f"  > It {it:02d}: ||R_r|| = {nrm:.3e}, rel = {r_rel:.3e}, "
                    f"||dq_m|| = {ndq:.3e} (rel {dq_rel:.3e}) -> converged(dq)"
                )
                print(f"  > Converged in {it} iterations.")
                break

            print(f"  > It {it:02d}: ||R_r|| = {nrm:.3e}, rel = {r_rel:.3e}")

        if not converged:
            msg = (
                f"[PROM-GPR] Step {step}/{n_steps_total} did not converge in {n_corr} corrector iterations. "
                f"last ||R_r||={nrm_last:.3e}, last ||dq_m||={ndq_last:.3e}."
            )
            if nonfinite_detected:
                msg += " Non-finite state detected."
            if bool(prom_fail_on_nonconvergence):
                raise RuntimeError(msg)
            print(f"  [PROM-GPR][WARN] {msg}")
            k_red_old = None
        elif k_red_last is not None and np.all(np.isfinite(k_red_last)):
            k_red_old = k_red_last.copy()
        else:
            k_red_old = None

        # Push accepted state and compute homogenization with full mesh (PROM-style)
        q_s_raw_f, _ = evaluate_sparse_gp_map_and_jacobian_qp(q_m, gpr_model, q_m_dim)
        q_s_phys = np.asarray(q_s_raw_f, dtype=float).reshape(-1)
        q_s = q_s_phys - q_s0_raw - (j0_raw @ q_m)
        q_master = a_m @ q_m
        if track_qpod:
            q_pod = (c_m @ q_master) + (c_s @ q_s_phys)
        u[free_dofs] = u_aff_free + w0_const + (phi_master_eff @ q_m) + (phi_s @ q_s)
        SetDisplacementFromEquationVector(u, eq_map_runtime, ta_disp)
        UpdateCurrentCoordinatesFromDisplacement(mp, step=0)

        eps_h, sig_h = CalculateHomogenizedFromAssemblerWithElementWeights(
            assembler,
            w_eps=None,
            w_sig=None,
            reference_measure=float(np.sum(np.asarray(assembler.area_e, dtype=float))),
        )

        sim.FinalizeSolutionStep()
        results_eps.append(np.asarray(eps_h, dtype=float))
        results_sig.append(np.asarray(sig_h, dtype=float))
        u_hist.append(u.copy())
        e_applied_hist.append(e.copy())
        qm_hist.append(q_m.copy())
        qs_hist.append(q_s_phys.copy())
        if track_qpod:
            qpod_hist.append(q_pod.copy())
        q_m_state = q_m.copy()
        print(
            f"  [PROM-GPR] Step {step:03d}/{n_steps_total}: "
            f"||q_m||={np.linalg.norm(q_m):.3e}, ||q_s||={np.linalg.norm(q_s_phys):.3e}"
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

    if out_dir is not None:
        tag = f"trajectory_{trajectory_index}" if trajectory_index is not None else "prom_gpr_run"
        np.save(os.path.join(out_dir, f"{tag}_strain.npy"), strain_hist)
        np.save(os.path.join(out_dir, f"{tag}_stress.npy"), stress_hist)
        np.save(os.path.join(out_dir, f"{tag}_U.npy"), u_hist)
        np.save(os.path.join(out_dir, f"{tag}_applied_strain.npy"), e_applied_hist)
        np.save(os.path.join(out_dir, f"{tag}_q_m.npy"), qm_hist)
        np.save(os.path.join(out_dir, f"{tag}_q_s.npy"), qs_hist)
        if track_qpod:
            np.save(os.path.join(out_dir, f"{tag}_q_pod.npy"), qpod_hist)

    return strain_hist, stress_hist
