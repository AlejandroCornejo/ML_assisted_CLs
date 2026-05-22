import os
import sys
import time
import numpy as np

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
    CalculateHomogenizedFromAssemblerWithElementWeights,
    REFERENCE_STEPS_FOR_UNIT_AMPLITUDE,
    MIN_STEPS_PER_SEGMENT,
    USE_OLD_STIFFNESS_IN_FIRST_ITERATION,
    NEWTON_TOL_ABS,
    DISP_TOL_ABS,
)
from sparse_gp_manifold_model import (
    load_sparse_gp_model,
    evaluate_sparse_gp_map_and_jacobian_qp,
)


def _load_optional_mu_affine_initializer(gpr_data_dir, n_primary):
    init_path = os.path.join(gpr_data_dir, "qp_init_mu_affine.npz")
    if not os.path.exists(init_path):
        return None
    data = np.load(init_path, allow_pickle=True)
    if "b_aff" not in data:
        raise RuntimeError(f"[PROM-GPR] Invalid initializer file (missing b_aff): {init_path}")
    b_aff = np.asarray(data["b_aff"], dtype=float)
    if b_aff.ndim != 2 or b_aff.shape[0] < 2:
        raise RuntimeError(f"[PROM-GPR] Invalid b_aff shape in {init_path}: {b_aff.shape}")
    qp_dim = int(np.ravel(data["qp_dim"])[0]) if "qp_dim" in data else int(b_aff.shape[1])
    mu_dim = int(np.ravel(data["mu_dim"])[0]) if "mu_dim" in data else int(b_aff.shape[0] - 1)
    if qp_dim != int(n_primary):
        raise RuntimeError(
            f"[PROM-GPR] qp_init_mu_affine qp_dim={qp_dim} incompatible with n_primary={n_primary}."
        )
    if mu_dim != int(b_aff.shape[0] - 1):
        raise RuntimeError(
            f"[PROM-GPR] qp_init_mu_affine mu_dim metadata mismatch ({mu_dim} vs {b_aff.shape[0] - 1})."
        )
    return {
        "path": init_path,
        "b_aff": b_aff,
        "mu_dim": mu_dim,
        "qp_dim": qp_dim,
    }


def LoadPromGprModel(basis_dir="stage_2_pod_rve", gpr_data_dir="stage_7_gpr_data"):
    phi_p = np.load(os.path.join(gpr_data_dir, "phi_p.npy"))
    phi_s = np.load(os.path.join(gpr_data_dir, "phi_s.npy"))
    free_dofs = np.load(os.path.join(basis_dir, "free_dofs.npy"))
    dir_dofs = np.load(os.path.join(basis_dir, "dirichlet_dofs.npy"))
    eq_map = np.load(os.path.join(basis_dir, "eq_map.npy"))

    gpr_model = load_sparse_gp_model(os.path.join(gpr_data_dir, "sparse_gp_model.npz"))

    n_p = int(phi_p.shape[1])
    n_s = int(phi_s.shape[1])
    input_dim = int(gpr_model["input_dim"])
    output_dim = int(gpr_model["output_dim"])
    include_macro_strain_input = bool(gpr_model["include_macro_strain_input"])
    if include_macro_strain_input:
        raise RuntimeError(
            "GPR model was trained with macro-strain manifold inputs (N(q,mu)), "
            "which is no longer supported. Retrain Stage 7 without macro inputs."
        )

    expected_input_dim = n_p
    if input_dim != expected_input_dim:
        raise ValueError(
            f"GPR input_dim={input_dim} incompatible with n_primary={n_p} "
            f"and include_macro_strain_input={int(include_macro_strain_input)}."
        )
    if output_dim != n_s:
        raise ValueError(
            f"GPR output_dim={output_dim} incompatible with n_secondary={n_s}."
        )
    if "n_primary" in gpr_model and int(gpr_model["n_primary"]) != n_p:
        raise ValueError(
            f"GPR metadata n_primary={int(gpr_model['n_primary'])} does not match phi_p ({n_p})."
        )
    if "n_secondary" in gpr_model and int(gpr_model["n_secondary"]) != n_s:
        raise ValueError(
            f"GPR metadata n_secondary={int(gpr_model['n_secondary'])} does not match phi_s ({n_s})."
        )
    gpr_model["qp_init_mu_affine"] = _load_optional_mu_affine_initializer(gpr_data_dir, n_primary=n_p)

    return (
        phi_p,
        phi_s,
        free_dofs,
        dir_dofs,
        eq_map,
        gpr_model,
        False,
    )


def RunPromGprBatchSimulation(
    parameters,
    phi_p,
    phi_s,
    free_dofs,
    gpr_model,
    strain_path,
    out_dir=None,
    relnorm_cutoff=1e-5,
    max_its=25,
    abs_res_cutoff=NEWTON_TOL_ABS,
    dq_abs_cutoff=1.0e-6,
    max_res_for_rel_convergence=1.0e-1,
    min_rel_drop_stop=1.0e-2,
    stagnation_relnorm_gate=1.0e-4,
    max_dq_norm=0.5,
    damping_after_iter=10,
    damping_factor=0.5,
    old_stiffness_residual_cutoff=1.0e5,
    regularization=1.0e-10,
    use_fast_dirichlet_bc=True,
    reference_amplitude=None,
    reference_steps=REFERENCE_STEPS_FOR_UNIT_AMPLITUDE,
    use_old_stiffness_in_first_iteration=USE_OLD_STIFFNESS_IN_FIRST_ITERATION,
    verbose_iterations=False,
    qp_init_mode="previous",
    evaluate_homogenization=True,
):
    t_wall_total_start = time.perf_counter()

    free_dofs = np.asarray(free_dofs, dtype=np.int64)
    if phi_p.shape[0] != phi_s.shape[0]:
        raise ValueError("phi_p and phi_s must have the same number of rows (n_free).")

    n_primary = int(phi_p.shape[1])
    n_secondary = int(phi_s.shape[1])
    damping_after_iter = int(damping_after_iter)
    damping_factor = float(damping_factor)
    if not (0.0 < damping_factor <= 1.0):
        raise ValueError(f"damping_factor must be in (0, 1], got {damping_factor}.")

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
    vec_assembler = VectorizedAssembler(mp, n_total_dof, eq_id_map)
    hom_reference_measure = float(np.sum(np.asarray(vec_assembler.area_e, dtype=float)))
    if out_dir is not None and bool(evaluate_homogenization):
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "reference_measure_A0.txt"), "w", encoding="utf-8") as f:
            f.write(f"{hom_reference_measure:.16e}\n")
    elements = list(mp.Elements)
    entities = list(mp.Elements) + list(mp.Conditions)

    # Affine lifting helper for free DOFs: u_aff = (F-I)(X-Xc)
    sim._InitializeDomainCenterIfNeeded(mp)
    x0c, y0c = float(sim._x0c), float(sim._y0c)
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
    free_mask = np.zeros(n_total_dof, dtype=bool)
    free_mask[free_dofs] = True
    dir_dofs = np.nonzero(~free_mask)[0].astype(np.int64)
    x_dir = dof_x[dir_dofs]
    y_dir = dof_y[dir_dofs]
    is_x_dir = is_x_dof[dir_dofs]

    results_eps = [np.zeros(3, dtype=float)]
    results_sig = [np.zeros(3, dtype=float)]
    q_hist = [np.zeros(n_primary, dtype=float)]

    gpr_input_dim = int(gpr_model["input_dim"])
    expected_input_dim = int(n_primary)
    if gpr_input_dim != expected_input_dim:
        raise ValueError(
            f"GPR input size mismatch: model expects {gpr_input_dim}, "
            f"but solver was configured for {expected_input_dim}."
        )
    qp_init_mode = str(qp_init_mode).strip().lower()
    if qp_init_mode not in ("previous", "zero", "mu_affine"):
        raise ValueError(
            f"Unsupported qp_init_mode='{qp_init_mode}'. Use one of: previous, zero, mu_affine."
        )
    qp_aff = gpr_model.get("qp_init_mu_affine", None)
    if qp_init_mode == "mu_affine" and qp_aff is None:
        raise RuntimeError(
            "[PROM-GPR] qp_init_mode='mu_affine' requested but qp_init_mu_affine.npz is missing "
            f"in model directory."
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

    def _build_gpr_input(qp_vec, e_vec):
        qp = np.asarray(qp_vec, dtype=float).reshape(-1)
        return qp

    def _evaluate_qs_and_jac(qp_vec, e_vec):
        x_in = _build_gpr_input(qp_vec, e_vec)
        q_s_map, j_qs_qp = evaluate_sparse_gp_map_and_jacobian_qp(x_in, gpr_model, n_primary)
        return np.asarray(q_s_map, dtype=float).reshape(-1), np.asarray(j_qs_qp, dtype=float)

    def _initial_qp_guess(e_vec, q_prev):
        if qp_init_mode == "previous":
            return np.asarray(q_prev, dtype=float).copy()
        if qp_init_mode == "zero":
            return np.zeros(n_primary, dtype=float)
        mu_dim = int(qp_aff["mu_dim"])
        mu = np.asarray(e_vec, dtype=float).reshape(-1)[:mu_dim]
        if mu.size < mu_dim:
            raise RuntimeError(
                f"[PROM-GPR] qp_init_mu_affine expects mu_dim={mu_dim} but got only {mu.size} strain components."
            )
        x_aug = np.concatenate([mu, np.array([1.0], dtype=float)])
        q0 = x_aug @ np.asarray(qp_aff["b_aff"], dtype=float)
        return np.asarray(q0, dtype=float).reshape(-1)

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
    J_full = np.zeros((n_total_dof, n_primary), dtype=float)
    q0_const, J0_const = _evaluate_qs_and_jac(np.zeros(n_primary, dtype=float), np.zeros(3, dtype=float))
    phi_p_eff = phi_p + phi_s @ J0_const
    w0_const = phi_s @ q0_const

    print(f"  [PROM-GPR] Solving for {n_steps_total} dynamic increments...")
    print(f"  [PROM-GPR] Full elements assembled each Newton step: {len(elements)} / {len(elements)}")
    print("  [PROM-GPR] Manifold correction active: N(0)=0 and J(0)=0.")
    print(f"  [PROM-GPR] q_p initializer mode: {qp_init_mode}")
    if qp_init_mode == "mu_affine" and qp_aff is not None:
        print(f"  [PROM-GPR] Using affine initializer: {qp_aff['path']}")
    t_map = 0.0
    t_assembly = 0.0
    t_projection = 0.0
    t_solve = 0.0
    t_full_sync = 0.0

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
            print(f"\n[PROM-GPR] Step {step:04d}/{n_steps_total} | E={E}")
        verbose_step = bool(verbose_iterations)

        q_p = _initial_qp_guess(E, q_p)
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

        while it < max_its:
            mp.ProcessInfo[KM.NL_ITERATION_NUMBER] = it + 1
            t0 = time.perf_counter()
            q_s_map, J_rbf_raw = _evaluate_qs_and_jac(q_p, E)
            t_map += time.perf_counter() - t0
            q_s = q_s_map - q0_const - J0_const @ q_p
            J_rbf = J_rbf_raw - J0_const

            if verbose_step:
                qp_norm = float(np.linalg.norm(q_p))
                qs_norm = float(np.linalg.norm(q_s))
                print(f"    > q_p norm: {qp_norm:.3e} | q_s norm: {qs_norm:.3e}")

            if (not _is_finite(q_p)) or (not _is_finite(q_s)):
                print("  [PROM-GPR] WARNING: non-finite reduced state detected.")
                nonfinite_detected = True
                break

            if J_rbf.shape != (n_secondary, n_primary):
                raise RuntimeError(
                    f"Invalid GPR Jacobian shape {J_rbf.shape}; expected ({n_secondary}, {n_primary})."
                )
            if not _is_finite(J_rbf):
                print("  [PROM-GPR] WARNING: non-finite GPR Jacobian detected.")
                nonfinite_detected = True
                break

            u_fluc = w0_const + phi_p_eff @ q_p + phi_s @ q_s
            if not _is_finite(u_fluc):
                print("  [PROM-GPR] WARNING: non-finite reconstructed displacement detected.")
                nonfinite_detected = True
                break
            u_free = u_aff_free + u_fluc

            J_manifold = phi_p_eff + phi_s @ J_rbf
            if not _is_finite(J_manifold):
                print("  [PROM-GPR] WARNING: non-finite manifold Jacobian detected.")
                nonfinite_detected = True
                break

            u_eq_curr = _apply_total_free_displacement(u_free, base_disp_vec=disp_base_step)

            InitializeNonLinearIteration(entities, mp.ProcessInfo)
            t0 = time.perf_counter()
            K_sparse, rhs_vec = vec_assembler.Assemble(u_eq_curr)
            t_assembly += time.perf_counter() - t0
            FinalizeNonLinearIteration(entities, mp.ProcessInfo)

            if (not _is_finite(rhs_vec)) or (not _is_finite(K_sparse.data)):
                print("  [PROM-GPR] WARNING: non-finite full residual/stiffness detected.")
                nonfinite_detected = True
                break

            t0 = time.perf_counter()
            J_full.fill(0.0)
            J_full[free_dofs, :] = J_manifold
            KJ = K_sparse @ J_full
            r_r = J_full.T @ rhs_vec
            K_r = J_full.T @ KJ
            t_projection += time.perf_counter() - t0
            Kr_last = K_r
            if (not _is_finite(r_r)) or (not _is_finite(K_r)):
                print("  [PROM-GPR] WARNING: non-finite reduced residual/stiffness detected.")
                nonfinite_detected = True
                break

            res_norm = float(np.linalg.norm(r_r))
            if not np.isfinite(res_norm):
                print("  [PROM-GPR] WARNING: non-finite reduced residual norm detected.")
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
                print("  > Converged in {0} iterations (small update + reduced residual).".format(it))
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
                        "  > Converged in {0} iterations (stagnation criterion: rel_drop={1:.3e}).".format(
                            it, rel_drop
                        )
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
                print("  [PROM-GPR] WARNING: non-finite reduced update detected.")
                nonfinite_detected = True
                break

            dq_norm = float(np.linalg.norm(dq_p))
            if dq_norm > float(max_dq_norm) and dq_norm > 0.0:
                scale = float(max_dq_norm) / dq_norm
                dq_p *= scale
                dq_norm = float(np.linalg.norm(dq_p))
                print(f"    > large reduced update clipped with scale={scale:.3e}")

            alpha = 1.0
            if it > int(damping_after_iter):
                alpha = float(damping_factor)
            q_trial = q_p + alpha * dq_p
            if not _is_finite(q_trial):
                print("  [PROM-GPR] WARNING: non-finite reduced state update detected.")
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
                    "  [PROM-GPR] Step accepted as quasi-converged: "
                    f"best ||R_r||={best_res:.3e}, rel={best_rel:.3e}."
                )
                Kr_old = None
            else:
                print(f"  [PROM-GPR] WARNING: step {step} did not converge in {max_its} iterations.")
                if nonfinite_detected:
                    print("  [PROM-GPR] WARNING: non-finite state encountered; rolling back to best finite iterate.")
                if np.isfinite(best_res):
                    q_p = best_q.copy()
                    print(f"  [PROM-GPR] Using best finite iterate with ||R_r||={best_res:.3e}.")
                else:
                    q_p = q_step_start.copy()
                    print("  [PROM-GPR] Reverting to previous-step reduced state.")
                Kr_old = None
        elif Kr_last is not None and _is_finite(Kr_last):
            Kr_old = Kr_last.copy()
        else:
            Kr_old = None

        # Ensure the accepted reduced state is explicitly pushed to Kratos.
        q_s_final_map, _ = _evaluate_qs_and_jac(q_p, E)
        q_s_final = q_s_final_map - q0_const - J0_const @ q_p
        u_fluc_final = w0_const + phi_p_eff @ q_p + phi_s @ q_s_final
        if not _is_finite(u_fluc_final):
            q_p = q_step_start.copy()
            q_s_final_map, _ = _evaluate_qs_and_jac(q_p, E)
            q_s_final = q_s_final_map - q0_const - J0_const @ q_p
            u_fluc_final = w0_const + phi_p_eff @ q_p + phi_s @ q_s_final
        if not _is_finite(u_fluc_final):
            raise RuntimeError("PROM-GPR accepted state is non-finite after rollback.")
        if not use_fast_dirichlet_bc:
            sim.ApplyBoundaryConditions()
            disp_base_step = _capture_current_displacement_vector()
        u_eq_final = _apply_total_free_displacement(
            u_aff_free + u_fluc_final, base_disp_vec=disp_base_step
        )

        if bool(evaluate_homogenization):
            InitializeNonLinearIteration(entities, mp.ProcessInfo)
            t0 = time.perf_counter()
            _, _ = vec_assembler.Assemble(u_eq_final)
            t_full_sync += time.perf_counter() - t0
            FinalizeNonLinearIteration(entities, mp.ProcessInfo)
            eps_h, sig_h = CalculateHomogenizedFromAssemblerWithElementWeights(
                vec_assembler,
                reference_measure=hom_reference_measure,
            )
        else:
            eps_h = np.zeros(3, dtype=float)
            sig_h = np.zeros(3, dtype=float)
        sim.FinalizeSolutionStep()
        results_eps.append(eps_h)
        results_sig.append(sig_h)
        q_hist.append(q_p.copy())

    sim.Finalize()
    t_wall_total = time.perf_counter() - t_wall_total_start
    t_accounted = t_map + t_assembly + t_projection + t_solve + t_full_sync
    t_other = max(t_wall_total - t_accounted, 0.0)
    print(
        f"\n[PROM-GPR] Timing: map={t_map:.3f}s, assembly={t_assembly:.3f}s, "
        f"projection={t_projection:.3f}s, solve={t_solve:.3f}s, full_sync={t_full_sync:.3f}s, "
        f"accounted={t_accounted:.3f}s, other={t_other:.3f}s, total={t_wall_total:.3f}s"
    )
    if out_dir is not None:
        np.save(os.path.join(out_dir, "prom_gpr_run_q_p.npy"), np.stack(q_hist))
    return np.array(results_eps), np.array(results_sig)
