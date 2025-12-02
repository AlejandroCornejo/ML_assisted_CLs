#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PROM–POD solver for the J2 RVE problem.

- Builds the Kratos model and vectorized FE/J2 machinery (same as FOM).
- Loads a POD basis Phi (from your snapshot-based POD).
- Chooses one macro-strain batch (theta, phi) via user parameters.
- Runs a quasi-static time stepping using a reduced Newton in q:

      u_D = Dirichlet lifting (from Kratos BCs)
      u(free) = u_prev_free + Phi_f @ q
      R_f(u) = internal force residual (free DOFs)
      r(q)   = Phi_f^T R_f(u(q))
      K_r    = Phi_f^T K_ff(u(q)) Phi_f

  Solve K_r Δq = -r until convergence.

- Homogenizes stress/strain at each step from J2 and compares PROM vs FOM
  if all_strain_histories.npz / all_stress_histories.npz exist.

Requires:
  - ProjectParameters.json
  - j2_plane_stress_plastic_strain_rve_simo_optimized.py
  - modes/U_modes_tol_*.npy
  - (optional, for comparison) data_set/all_strain_histories.npz,
    data_set/all_stress_histories.npz and data_set/batch_log.txt
"""

import numpy as np
import os
import importlib
import matplotlib.pyplot as plt

import KratosMultiphysics as KM
import KratosMultiphysics.analysis_stage as analysis_stage
from KratosMultiphysics.StructuralMechanicsApplication import \
    python_solvers_wrapper_structural as structural_solvers

from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from fom_solver_rve import (
    build_node_global_map, precompute_mesh_arrays,
    extract_dirichlet_bcs, assemble_K_and_fint_vec, homogenize_from_J2_vec
)

from j2_plane_stress_plastic_strain_rve_simo_optimized import (
    VonMisesIsotropicPlasticityPlaneStress
)

# ----------------------------------------------------------------------
# User parameters
# ----------------------------------------------------------------------
# POD basis to use
basis_file = "modes/U_modes_tol_1e-06.npy"

# Choose which macro-strain direction to reproduce via PROM
# (same grid as in your dataset generator: theta, phi in {0, 25, 50} deg)
theta = 50.0   # [deg]
phi   = 50.0   # [deg]
max_stretch_factor = 0.01  # lambda

# Time stepping (same as in ProjectParameters.json ideally)
project_parameters_file = "ProjectParameters.json"

# Newton settings (reduced system)
max_newton_it = 40
tol_rel = 1e-3
tol_abs = 1e-2
max_ls_it = 5   # backtracking line-search iterations (analogous to FOM)

# For mapping (theta, phi) to batch index in your original dataset
angle_increment = 25.0    # must match generator
n_phi = int(50.0 / angle_increment) + 1  # = 3

# ----------------------------------------------------------------------
# Custom AnalysisStage just to get BCs (same pattern as in generator)
# ----------------------------------------------------------------------
class RVE_homogenization_PROM(analysis_stage.AnalysisStage):
    def __init__(self, model, project_parameters, batch_strain):
        super().__init__(model, project_parameters)
        self.batch_strain = np.array(batch_strain, dtype=float)

    def _CreateSolver(self):
        return structural_solvers.CreateSolver(self.model, self.project_parameters)

    def __CreateListOfProcesses(self):
        order_processes_initialization = self._GetOrderOfProcessesInitialization()
        self._list_of_processes        = self._CreateProcesses("processes", order_processes_initialization)
        deprecated_output_processes    = self._CheckDeprecatedOutputProcesses(self._list_of_processes)
        order_processes_initialization = self._GetOrderOfOutputProcessesInitialization()
        self._list_of_output_processes = self._CreateProcesses("output_processes", order_processes_initialization)
        self._list_of_processes.extend(self._list_of_output_processes)
        self._list_of_output_processes.extend(deprecated_output_processes)

    def ApplyBoundaryConditions(self):
        """
        Macro-strain → nodal displacements (same formula as in dataset generator).
        """
        super().ApplyBoundaryConditions()

        Ex, Ey, Exy = self.batch_strain

        for node in self._GetSolver().GetComputingModelPart().Nodes:
            x_coord = node.X0
            y_coord = node.Y0
            displ_x = (Ex * x_coord + Exy * y_coord) * self.time / self.end_time
            displ_y = (Ey * y_coord + Exy * x_coord) * self.time / self.end_time
            displ_z = 0.0

            if node.IsFixed(KM.DISPLACEMENT_X):
                node.SetSolutionStepValue(KM.DISPLACEMENT_X, displ_x)
            if node.IsFixed(KM.DISPLACEMENT_Y):
                node.SetSolutionStepValue(KM.DISPLACEMENT_Y, displ_y)
            if node.IsFixed(KM.DISPLACEMENT_Z):
                node.SetSolutionStepValue(KM.DISPLACEMENT_Z, displ_z)


# ----------------------------------------------------------------------
# Main PROM driver
# ----------------------------------------------------------------------
def main():
    # --- Load ProjectParameters and time stepping ---
    if not os.path.isfile(project_parameters_file):
        raise FileNotFoundError(project_parameters_file)

    with open(project_parameters_file, "r") as parameter_file:
        parameters = KM.Parameters(parameter_file.read())

    dt       = parameters["solver_settings"]["time_stepping"]["time_step"].GetDouble()
    end_time = parameters["problem_data"]["end_time"].GetDouble()

    # Macro-strain vector for this (theta, phi)
    batch_strain = max_stretch_factor * np.array([
        np.cos(np.radians(phi)),                              # Exx
        np.sin(np.radians(theta)) * np.cos(np.radians(phi)),  # Eyy
        np.sin(np.radians(theta)) * np.sin(np.radians(phi)),  # Exy
    ])

    print(f"[PROM] Using batch_strain = {batch_strain}")

    # --- Build Kratos model and solver wrapper ---
    model = KM.Model()
    simulation = RVE_homogenization_PROM(model, parameters, batch_strain)
    simulation.Initialize()

    mp = simulation._GetSolver().GetComputingModelPart()
    idx_ux, idx_uy, n_dof = build_node_global_map(mp)

    conn, B_all, w_all, area_all, pattern_I, pattern_J = \
        precompute_mesh_arrays(mp, idx_ux, idx_uy)

    n_elem, n_gp = B_all.shape[0], B_all.shape[1]

    # Material (from first element)
    elem0 = next(iter(mp.Elements))
    props0 = elem0.Properties
    E        = props0[KM.YOUNG_MODULUS]
    nu       = props0[KM.POISSON_RATIO]
    sigma_y0 = props0[KM.YIELD_STRESS]
    H        = 0.0

    mat = VonMisesIsotropicPlasticityPlaneStress(E, nu, sigma_y0, H)

    # Plastic state
    eps_p_n = np.zeros((n_elem, n_gp, 3), dtype=float)
    alpha_n = np.zeros((n_elem, n_gp), dtype=float)

    # --- Load POD basis ---
    if not os.path.isfile(basis_file):
        raise FileNotFoundError(basis_file)
    Phi = np.load(basis_file)   # (n_dof, n_modes)
    if Phi.shape[0] != n_dof:
        raise ValueError(f"POD basis rows {Phi.shape[0]} != n_dof {n_dof}")

    n_modes = Phi.shape[1]
    print(f"[PROM] Loaded Phi: shape = {Phi.shape}, n_modes = {n_modes}")

    # --- Time loop (reduced Newton in q with K_old + line search) ---
    time = 0.0
    step = 0

    u = np.zeros(n_dof)               # full-space displacement (PROM)
    q = np.zeros(n_modes)             # reduced coordinates
    Phi_f = None                      # restriction to free DOFs
    free_dofs = None
    dirichlet_dofs = None

    # Predictor reduced stiffness (analogous to K_old in FOM)
    K_old_red = None

    strain_prom_hist = [np.zeros(3)]
    stress_prom_hist = [np.zeros(3)]

    while time < end_time - 1e-12:
        step += 1
        time += dt
        simulation.time = time
        simulation.step = step

        print(f"\n[PROM] Time step {step}, t = {time:.6f}")

        # 1) Apply BCs and read Dirichlet values
        simulation.ApplyBoundaryConditions()
        dirichlet_dofs, dirichlet_vals = extract_dirichlet_bcs(
            mp, idx_ux, idx_uy, step_index=0
        )

        all_dofs = np.arange(n_dof, dtype=int)
        mask = np.ones(n_dof, dtype=bool)
        mask[dirichlet_dofs] = False
        free_dofs = all_dofs[mask]

        if Phi_f is None:
            # Restrict basis to free DOFs once
            Phi_f = Phi[free_dofs, :]
            print(f"[PROM] Phi_f shape = {Phi_f.shape} (|free| x n_modes)")

        # Lifting vector u_D for this step
        u_D = np.zeros(n_dof)
        if dirichlet_dofs.size > 0:
            u_D[dirichlet_dofs] = dirichlet_vals

        # Initial reduced guess: keep previous q (quasi-static path)
        q_curr = q.copy()

        # Newton in q (with predictor K_old_red and line search)
        converged = False
        norm_r0 = None

        for it in range(max_newton_it):
            # 2) Build full displacement from current q_curr
            u_curr = np.zeros_like(u)
            u_curr[dirichlet_dofs] = u_D[dirichlet_dofs]
            u_curr[free_dofs]      = u_D[free_dofs] + Phi_f @ q_curr

            # 3) Assemble at u_curr
            K_glob, f_int, _, _ = assemble_K_and_fint_vec(
                mat, conn, B_all, w_all,
                eps_p_n, alpha_n,      # committed state at t_n
                u_curr, n_dof, pattern_I, pattern_J
            )

            R_f = f_int[free_dofs]
            r   = Phi_f.T @ R_f   # reduced residual

            norm_r = np.linalg.norm(r, ord=np.inf)
            if it == 0:
                norm_r0 = max(norm_r, 1.0)

            rel = norm_r / norm_r0

            print(f"  [NR it {it:02d}] ||r||_inf = {norm_r:.3e}, rel = {rel:.3e}")

            # 4) Convergence check
            if (rel < tol_rel) or (norm_r < tol_abs):
                print("    -> PROM converged")
                converged = True
                # At convergence, keep q_curr and u_curr
                u[:] = u_curr
                break

            # 5) Reduced tangent with predictor (analogous to K_old / K_ff in FOM)
            K_ff = K_glob[free_dofs][:, free_dofs]
            if it == 0 and K_old_red is not None:
                # Use predictor reduced stiffness from previous step
                K_red = K_old_red
            else:
                # Fresh reduced tangent
                K_red = Phi_f.T @ (K_ff @ Phi_f)

            # 6) Newton step in reduced space
            dq = spsolve(K_red, -r)

            # 7) Backtracking line search in q (analogous to FOM line search in u_f)
            q_base = q_curr.copy()
            alpha_ls = 1.0
            for ls_it in range(max_ls_it):
                q_trial = q_base + alpha_ls * dq

                # Rebuild u for this q_trial
                u_trial = np.zeros_like(u)
                u_trial[dirichlet_dofs] = u_D[dirichlet_dofs]
                u_trial[free_dofs]      = u_D[free_dofs] + Phi_f @ q_trial

                # Residual at trial state (still with committed eps_p_n, alpha_n)
                _, f_int_trial, _, _ = assemble_K_and_fint_vec(
                    mat, conn, B_all, w_all,
                    eps_p_n, alpha_n,
                    u_trial, n_dof, pattern_I, pattern_J
                )
                R_f_trial = f_int_trial[free_dofs]
                r_trial   = Phi_f.T @ R_f_trial
                norm_r_trial = np.linalg.norm(r_trial, ord=np.inf)

                if norm_r_trial < norm_r:
                    # Accept trial step
                    q_curr = q_trial
                    # (we do NOT update eps_p_n / alpha_n here, same as FOM)
                    break
                else:
                    alpha_ls *= 0.5
            else:
                # If line search fails to reduce residual, take a small step
                q_curr = q_base + 0.1 * dq

        # --- end of Newton loop ---

        if not converged:
            print(f"  [WARN] PROM did not fully converge at step {step} "
                  f"(||r||_inf={norm_r:.3e})")

        # 8) At the end of the step, recompute at final (q_curr, u) to:
        #    - get consistent eps_p, alpha
        #    - build predictor K_old_red for next step
        u_final = np.zeros_like(u)
        u_final[dirichlet_dofs] = u_D[dirichlet_dofs]
        u_final[free_dofs]      = u_D[free_dofs] + Phi_f @ q_curr

        K_glob_conv, f_int_conv, eps_p, alpha = assemble_K_and_fint_vec(
            mat, conn, B_all, w_all,
            eps_p_n, alpha_n,
            u_final, n_dof, pattern_I, pattern_J
        )

        # Update K_old_red (predictor reduced stiffness for next step)
        K_ff_conv  = K_glob_conv[free_dofs][:, free_dofs]
        K_old_red  = Phi_f.T @ (K_ff_conv @ Phi_f)

        # Commit plastic state
        eps_p_n[:] = eps_p
        alpha_n[:] = alpha

        # Update global u and q to final values
        u[:] = u_final
        q[:] = q_curr

        # 9) Homogenize
        strain_prom, stress_prom = homogenize_from_J2_vec(
            mat, conn, B_all, w_all, area_all,
            eps_p_n, alpha_n, u
        )

        strain_prom_hist.append(strain_prom)
        stress_prom_hist.append(stress_prom)

    strain_prom_hist = np.stack(strain_prom_hist, axis=0)
    stress_prom_hist = np.stack(stress_prom_hist, axis=0)

    # ------------------------------------------------------------------
    # Try to load FOM dataset for comparison (if available)
    # ------------------------------------------------------------------
    fom_strain = None
    fom_stress = None

    strain_npz = "data_set/all_strain_histories.npz"
    stress_npz = "data_set/all_stress_histories.npz"

    if os.path.isfile(strain_npz) and os.path.isfile(stress_npz):
        strain_data = np.load(strain_npz)
        stress_data = np.load(stress_npz)

        strain_tensor = strain_data["strain"]   # (n_batches, n_steps+1, 3)
        stress_tensor = stress_data["stress"]   # (n_batches, n_steps+1, 3)

        # Map (theta, phi) -> batch index consistent with your generator
        ith_theta = int(round(theta / angle_increment))
        jth_phi   = int(round(phi   / angle_increment))
        batch_index = ith_theta * n_phi + jth_phi   # 0-based

        if batch_index < 0 or batch_index >= strain_tensor.shape[0]:
            print(f"[WARN] Computed batch_index {batch_index} out of range "
                  f"(0..{strain_tensor.shape[0]-1}), skipping FOM comparison.")
        else:
            fom_strain = strain_tensor[batch_index, :, :]
            fom_stress = stress_tensor[batch_index, :, :]

            print(f"[PROM] Comparing to FOM batch index {batch_index+1} "
                  f"(theta={theta}, phi={phi})")

    # ------------------------------------------------------------------
    # Plots: PROM vs FOM (if FOM available)
    # ------------------------------------------------------------------
    os.makedirs("prom_results", exist_ok=True)
    labels = [r"$\sigma_{xx}$", r"$\sigma_{yy}$", r"$\sigma_{xy}$"]
    colors = ["r", "b", "k"]

    plt.figure(figsize=(8, 6))
    for i in range(3):
        if fom_strain is not None and fom_stress is not None:
            plt.plot(
                fom_strain[:, i], fom_stress[:, i],
                color=colors[i], linestyle="-", marker="o",
                label=f"FOM {labels[i]}"
            )
        plt.plot(
            strain_prom_hist[:, i], stress_prom_hist[:, i],
            color=colors[i], linestyle="--", marker="",
            label=f"PROM {labels[i]}"
        )

    plt.xlabel("Strain component [-]")
    plt.ylabel("Stress [Pa]")
    plt.title(f"POD-PROM J2 RVE (theta={theta:.1f}, phi={phi:.1f})")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    fig_name = f"prom_results/stress_strain_PROM_vs_FOM_theta{theta:.0f}_phi{phi:.0f}.png"
    plt.savefig(fig_name, dpi=300)
    plt.show()
    print(f"[PROM] Saved comparison plot to {fig_name}")

    # Save raw histories
    np.savez("prom_results/prom_histories_theta{:.0f}_phi{:.0f}.npz".format(theta, phi),
             strain_prom=strain_prom_hist,
             stress_prom=stress_prom_hist)


if __name__ == "__main__":
    main()
