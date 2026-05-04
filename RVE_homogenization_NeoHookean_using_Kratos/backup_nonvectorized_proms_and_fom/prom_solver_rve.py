#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PROM-only RVE solver (Finite Deformation BCs)
Reduced Order Model using the POD basis for free fluctuations.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add Kratos path
KRATOS_PATH = "/home/kratos/Kratos_Eigen_Check/bin/Release"
if KRATOS_PATH not in sys.path:
    sys.path.append(KRATOS_PATH)

import KratosMultiphysics as KM
from fom_solver_rve import (
    DeformationGradientFromGreenLagrange2D,
    SetUpDofEquationIdsAndDisplacementAdaptor,
    SetDisplacementFromEquationVector,
    UpdateCurrentCoordinatesFromDisplacement,
    AssembleGlobalSystem,
    InitializeNonLinearIteration,
    FinalizeNonLinearIteration,
    CalculateHomogenizedStressAndStrainKratosReference,
    DetectMaterialSubModelParts,
    ConfigureElementModelerForMaterialParts,
    StripMdpaExtension,
    BuildDynamicSegmentSteps,
    RVEHomogenizationDatasetGenerator,
    REFERENCE_STEPS_FOR_UNIT_AMPLITUDE,
    MIN_STEPS_PER_SEGMENT,
    USE_OLD_STIFFNESS_IN_FIRST_ITERATION,
    NEWTON_TOL_ABS,
    DISP_TOL_ABS,
)

# =============================================================================
# PROM Newton-Raphson Loop
# =============================================================================

def RunPromBatchSimulation(
    parameters,
    phi_f,  # (n_free, r)
    free_dofs,
    dir_dofs,
    eq_map,
    Xc, Yc,
    out_dir="prom_results",
    save_plot=True,
    strain_path=None,
    trajectory_index=None,
    reference_amplitude=None,
    reference_steps=REFERENCE_STEPS_FOR_UNIT_AMPLITUDE,
    max_newton_it=20,
    use_old_stiffness_in_first_iteration=USE_OLD_STIFFNESS_IN_FIRST_ITERATION,
    use_fast_dirichlet_bc=True,
):
    os.makedirs(out_dir, exist_ok=True)
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

    model = KM.Model()
    sim = RVEHomogenizationDatasetGenerator(model, parameters)
    sim.Initialize()
    mp = sim._GetSolver().GetComputingModelPart()
    elements = list(mp.Elements)
    entities = list(mp.Elements) + list(mp.Conditions)
    n_dof, eq_id_map, ta_disp = SetUpDofEquationIdsAndDisplacementAdaptor(mp)

    free_dofs = np.asarray(free_dofs, dtype=np.int64)
    free_mask = np.zeros(n_dof, dtype=bool)
    free_mask[free_dofs] = True
    dir_dofs_local = np.nonzero(~free_mask)[0].astype(np.int64)

    x0c = float(Xc)
    y0c = float(Yc)
    dof_x = np.zeros(n_dof, dtype=float)
    dof_y = np.zeros(n_dof, dtype=float)
    is_x_dof = np.zeros(n_dof, dtype=bool)
    for i, node in enumerate(mp.Nodes):
        xr = float(node.X0) - x0c
        yr = float(node.Y0) - y0c
        idx_x = int(eq_id_map[i, 0])
        idx_y = int(eq_id_map[i, 1])
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
    x_dir = dof_x[dir_dofs_local]
    y_dir = dof_y[dir_dofs_local]
    is_x_dir = is_x_dof[dir_dofs_local]

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
        disp_vec = np.zeros(n_dof, dtype=float)
        for i, node in enumerate(mp.Nodes):
            d = node.GetSolutionStepValue(KM.DISPLACEMENT)
            idx_x, idx_y = eq_id_map[i, 0], eq_id_map[i, 1]
            if idx_x < n_dof:
                disp_vec[idx_x] = d[0]
            if idx_y < n_dof:
                disp_vec[idx_y] = d[1]
        return disp_vec

    def _apply_total_free_displacement(u_total_free, base_disp_vec=None):
        if base_disp_vec is None:
            disp_vec = _capture_current_displacement_vector()
        else:
            disp_vec = np.asarray(base_disp_vec, dtype=float).copy()
        disp_vec[free_dofs] = np.asarray(u_total_free, dtype=float).reshape(-1)
        SetDisplacementFromEquationVector(disp_vec, eq_id_map, ta_disp)
        UpdateCurrentCoordinatesFromDisplacement(mp, step=0)

    # PROM Initialization
    r = phi_f.shape[1]
    q = np.zeros(r, dtype=float)  # Reduced state
    
    Q_hist, strain_hist, stress_hist = [], [], []
    
    # Initial state
    Q_hist.append(q.copy())
    strain_hist.append(np.zeros(3))
    stress_hist.append(np.zeros(3))
    Kr_old = None

    for step in range(1, n_steps_total + 1):
        time_val = float(step) * float(dt)
        mp.CloneTimeStep(time_val)
        mp.ProcessInfo[KM.DELTA_TIME] = dt
        mp.ProcessInfo[KM.TIME] = time_val
        mp.ProcessInfo[KM.STEP] = step

        sim.time, sim.step, sim.end_time = time_val, step, end_time
        sim.InitializeSolutionStep()

        # Interpolate Waypoint
        s = int(np.searchsorted(step_offsets, step, side="left") - 1)
        s = max(0, min(s, n_seg - 1))
        xi = float(step - step_offsets[s]) / float(max(seg_steps[s], 1))
        E_t = (1.0 - xi) * E_wp[s, :] + xi * E_wp[s + 1, :]
        
        sim.batch_strain = E_t.copy()
        u_aff_free = _compute_affine_free_displacement(E_t)
        if use_fast_dirichlet_bc:
            disp_base_step = np.zeros(n_dof, dtype=float)
            disp_base_step[dir_dofs_local] = _compute_affine_dirichlet_displacement(E_t)
        else:
            sim.ApplyBoundaryConditions()
            disp_base_step = _capture_current_displacement_vector()

        print(f"\n[PROM] Step {step:03d} | t={time_val:.6f}")
        
        converged = False
        Kr_last = None
        for it in range(max_newton_it):
            mp.ProcessInfo[KM.NL_ITERATION_NUMBER] = it + 1
            
            # Reconstruction: u = u_aff + Phi_f * q
            u_free = u_aff_free + phi_f @ q
            _apply_total_free_displacement(u_free, base_disp_vec=disp_base_step)

            InitializeNonLinearIteration(entities, mp.ProcessInfo)
            K, rhs = AssembleGlobalSystem(mp, n_dof, entities)
            FinalizeNonLinearIteration(entities, mp.ProcessInfo)

            # Project to Reduced Space
            # System: Phif^T * K * Phif * dq = Phif^T * rhs
            Kf_phi = K[free_dofs, :][:, free_dofs] @ phi_f
            Kr = phi_f.T @ Kf_phi
            Kr_last = Kr
            rr = phi_f.T @ rhs[free_dofs]
            
            nR = np.linalg.norm(rr)
            if it > 0 and nR < NEWTON_TOL_ABS:
                print(f"  > It {it:02d}: ||r_r|| = {nR:.3e} (CONVERGED)")
                converged = True; break
            
            K_solve = Kr
            used_old = False
            if (
                it == 0
                and use_old_stiffness_in_first_iteration
                and Kr_old is not None
                and Kr_old.shape == Kr.shape
            ):
                K_solve = Kr_old
                used_old = True

            try:
                dq = np.linalg.solve(K_solve, rr)
            except np.linalg.LinAlgError:
                if used_old:
                    try:
                        dq = np.linalg.solve(Kr, rr)
                        used_old = False
                    except np.linalg.LinAlgError:
                        dq, *_ = np.linalg.lstsq(Kr, rr, rcond=None)
                        used_old = False
                else:
                    dq, *_ = np.linalg.lstsq(Kr, rr, rcond=None)
            nq = np.linalg.norm(dq)
            solve_tag = " (K_old)" if used_old else ""
            print(f"  > It {it:02d}: ||r_r|| = {nR:.3e}, ||dq|| = {nq:.3e}{solve_tag}")
            
            q += dq
            if nq < DISP_TOL_ABS:
                print(f"  > It {it:02d}: ||dq|| = {nq:.3e} (CONVERGED)")
                converged = True; break

        if not converged:
            print(f" [WARNING] Step {step} did not converge.")
        if Kr_last is not None:
            Kr_old = Kr_last.copy()

        u_fluc_final = phi_f @ q
        if not use_fast_dirichlet_bc:
            sim.ApplyBoundaryConditions()
            disp_base_step = _capture_current_displacement_vector()
        _apply_total_free_displacement(u_aff_free + u_fluc_final, base_disp_vec=disp_base_step)

        InitializeNonLinearIteration(entities, mp.ProcessInfo)
        _, _ = AssembleGlobalSystem(mp, n_dof, entities)
        FinalizeNonLinearIteration(entities, mp.ProcessInfo)

        hom_eps, hom_sig = CalculateHomogenizedStressAndStrainKratosReference(mp)
        sim.FinalizeSolutionStep()
        
        Q_hist.append(q.copy())
        strain_hist.append(hom_eps)
        stress_hist.append(hom_sig)

    sim.Finalize()
    
    # Save Results
    tag = f"trajectory_{trajectory_index}" if trajectory_index else "prom_run"
    np.save(os.path.join(out_dir, f"{tag}_q.npy"), np.stack(Q_hist))
    np.save(os.path.join(out_dir, f"{tag}_strain.npy"), np.stack(strain_hist))
    np.save(os.path.join(out_dir, f"{tag}_stress.npy"), np.stack(stress_hist))
    
    if save_plot:
        _plot_prom(np.stack(strain_hist), np.stack(stress_hist), out_dir, tag)
    
    return np.stack(strain_hist), np.stack(stress_hist)

def _plot_prom(strain_hist, stress_hist, out_dir, tag):
    # Try to find FOM data for comparison
    fom_dir = os.path.join("stage_1_training_set_fom", tag)
    fom_sig_file = os.path.join(fom_dir, f"{tag}_stress.npy")
    fom_eps_file = os.path.join(fom_dir, f"{tag}_strain.npy")
    
    fom_sig = np.load(fom_sig_file) if os.path.exists(fom_sig_file) else None
    fom_eps = np.load(fom_eps_file) if os.path.exists(fom_eps_file) else None

    plt.figure(figsize=(8, 6))
    
    # Plot PROM
    plt.plot(strain_hist[:, 0], stress_hist[:, 0], 'r-', label="PROM sigma_xx")
    plt.plot(strain_hist[:, 1], stress_hist[:, 1], 'g-', label="PROM sigma_yy")
    plt.plot(strain_hist[:, 2], stress_hist[:, 2], 'b-', label="PROM sigma_xy")
    
    # Plot FOM if available
    if fom_sig is not None and fom_eps is not None:
        plt.plot(fom_eps[:, 0], fom_sig[:, 0], 'k--', alpha=0.6, label="FOM sigma_xx")
        plt.plot(fom_eps[:, 1], fom_sig[:, 1], 'k:',  alpha=0.6, label="FOM sigma_yy")
        plt.plot(fom_eps[:, 2], fom_sig[:, 2], 'k-.', alpha=0.6, label="FOM sigma_xy")

    plt.xlabel("Strain [-]")
    plt.ylabel("Stress [Pa]")
    plt.grid(True)
    plt.legend()
    plt.title(f"Results Comparison: {tag}")
    plt.savefig(os.path.join(out_dir, f"{tag}_prom_plots.png"))
    plt.close()

if __name__ == "__main__":
    import argparse
    from fom_solver_rve import SetInputMeshFilename, LoadStrainWaypointsFromFile
    
    p = argparse.ArgumentParser(description="PROM Solver for RVE")
    p.add_argument("--trajectory-index", type=int, default=1)
    p.add_argument("--mesh", type=str, default="rve_geometry")
    p.add_argument("--model-dir", type=str, default="pod_rbf_model")
    p.add_argument("--out-dir", type=str, default="prom_results")
    p.add_argument("--emax", type=float, default=0.10)
    p.add_argument("--young-mpa", type=float, default=1628.0)
    p.add_argument("--poisson", type=float, default=0.4)
    p.add_argument(
        "--no-old-stiffness-first-it",
        action="store_true",
        help="Disable reuse of previous-step reduced stiffness in Newton iteration 0.",
    )
    args = p.parse_args()

    # Load Model Metadata
    phi_f = np.load(os.path.join(args.model_dir, "pod_basis_free.npy"))
    free_dofs = np.load(os.path.join(args.model_dir, "free_dofs.npy"))
    dir_dofs = np.load(os.path.join(args.model_dir, "dirichlet_dofs.npy"))
    eq_map = np.load(os.path.join(args.model_dir, "eq_map.npy"))
    Xc, Yc = np.load(os.path.join(args.model_dir, "domain_center.npy"))

    with open("ProjectParameters.json", "r") as f:
        parameters = KM.Parameters(f.read())
    SetInputMeshFilename(parameters, args.mesh)
    
    # Material Setup (Mirrors FOM)
    mdpa_path = f"{StripMdpaExtension(args.mesh)}.mdpa"
    material_parts = DetectMaterialSubModelParts(mdpa_path)
    parameters = ConfigureElementModelerForMaterialParts(parameters, material_parts)
    parameters["solver_settings"]["material_import_settings"]["materials_filename"].SetString(
        "StructuralMaterials.json"
    )

    # Load Path
    strain_path, meta = LoadStrainWaypointsFromFile("stage_0_trajectory/stage_0_trajectories.npz", args.trajectory_index)
    
    RunPromBatchSimulation(
        parameters, phi_f, free_dofs, dir_dofs, eq_map, Xc, Yc,
        out_dir=args.out_dir,
        strain_path=strain_path,
        trajectory_index=args.trajectory_index,
        reference_amplitude=meta.get("reference_amplitude", args.emax),
        reference_steps=meta.get("ref_steps", REFERENCE_STEPS_FOR_UNIT_AMPLITUDE),
        use_old_stiffness_in_first_iteration=not args.no_old_stiffness_first_it,
    )
