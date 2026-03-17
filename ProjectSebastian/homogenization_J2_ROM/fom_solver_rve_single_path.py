#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import KratosMultiphysics as KM
import KratosMultiphysics.analysis_stage as analysis_stage
from KratosMultiphysics.StructuralMechanicsApplication import \
    python_solvers_wrapper_structural as structural_solvers
import KratosMultiphysics.StructuralMechanicsApplication as SMApp  # noqa: F401
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
import os
import matplotlib.pyplot as plt

from j2_plane_stress_plastic_strain_rve_simo_optimized import (
    VonMisesIsotropicPlasticityPlaneStress
)

# Default directory for data generation when running this file as a script
DEFAULT_OUTPUT_DIR = "training_set"


def precompute_mesh_arrays(mp, idx_ux, idx_uy):
    """
    Precompute element connectivity, B-matrices, GP weights and areas
    in fully vectorized arrays.

    Returns
    -------
    conn      : (n_elem, nd)        global DOF ids per element
    B_all     : (n_elem, n_gp, 3, nd)
    w_all     : (n_elem, n_gp)
    area_all  : (n_elem,)
    pattern_I, pattern_J : 1D arrays for global K sparsity
    """
    elems = list(mp.Elements)
    n_elem = len(elems)

    # Assume all elements have same topology and GPs
    geom0 = elems[0].GetGeometry()
    nnode = geom0.PointsNumber()
    nd    = 2 * nnode

    Ns   = np.array(geom0.ShapeFunctionsValues())
    n_gp = Ns.shape[0]

    conn     = np.zeros((n_elem, nd), dtype=int)
    B_all    = np.zeros((n_elem, n_gp, 3, nd), dtype=float)
    w_all    = np.zeros((n_elem, n_gp), dtype=float)
    area_all = np.zeros(n_elem, dtype=float)

    # One-time loop over elements (preprocessing, not in the NR loop)
    for e_idx, elem in enumerate(elems):
        geom = elem.GetGeometry()
        area = geom.Area()
        area_all[e_idx] = area

        # DOF indices
        col_ids = []
        for node in geom:
            col_ids.append(idx_ux[node.Id])
            col_ids.append(idx_uy[node.Id])
        conn[e_idx, :] = np.array(col_ids, dtype=int)

        for igauss in range(n_gp):
            DNDe = np.array(geom.ShapeFunctionDerivatives(1, igauss))
            J    = np.array(geom.Jacobian(igauss))
            DNDX = DNDe @ np.linalg.inv(J)
            B_loc = build_B_from_DNDX(DNDX)  # (3, nd)
            B_all[e_idx, igauss, :, :] = B_loc
            w_all[e_idx, igauss] = area / n_gp

    # Global sparsity pattern for K: for each element, all nd×nd pairs
    n_dof = 2 * mp.NumberOfNodes()
    nd    = conn.shape[1]

    # conn: (n_elem, nd)
    I_local = np.repeat(conn[:, :, None], nd, axis=2)  # (n_elem, nd, nd)
    J_local = np.repeat(conn[:, None, :], nd, axis=1)  # (n_elem, nd, nd)

    pattern_I = I_local.ravel()
    pattern_J = J_local.ravel()

    return conn, B_all, w_all, area_all, pattern_I, pattern_J


def build_B_from_DNDX(DNDX):
    """
    DNDX: (nnode, 2) with columns [dN/dx, dN/dy] in GLOBAL coords.
    Returns B (3, 2*nnode) for 2D small strain with ENGINEERING shear (γxy).
    Voigt order: [εxx, εyy, γxy].
    """
    nnode = DNDX.shape[0]
    B = np.zeros((3, 2*nnode))
    for a in range(nnode):
        Nx, Ny = DNDX[a, 0], DNDX[a, 1]
        # εxx
        B[0, 2*a    ] = Nx
        B[0, 2*a + 1] = 0.0
        # εyy
        B[1, 2*a    ] = 0.0
        B[1, 2*a + 1] = Ny
        # γxy (engineering γxy = 2 εxy)
        B[2, 2*a    ] = Ny
        B[2, 2*a + 1] = Nx
    return B


def build_node_global_map(mp):
    """
    Deterministic map consistent with TensorAdaptor ordering:
    [ux(node0), uy(node0), ux(node1), uy(node1), ...] following mp.Nodes iteration.
    """
    idx_ux, idx_uy = {}, {}
    for k, node in enumerate(mp.Nodes):
        idx_ux[node.Id] = 2*k
        idx_uy[node.Id] = 2*k + 1
    n_dof = 2 * mp.NumberOfNodes()
    return idx_ux, idx_uy, n_dof


def assemble_global_B_vec(conn, B_all, n_dof):
    """
    Fully vectorized construction of the global strain–displacement matrix B.

    Parameters
    ----------
    conn   : (n_elem, nd)
        Global DOF indices for each element.
    B_all  : (n_elem, n_gp, 3, nd)
        Element B matrices for all Gauss points.
    n_dof  : int
        Total number of DOFs in the model.

    Returns
    -------
    B_glob : csr_matrix, shape (3 * n_elem * n_gp, n_dof)
        Global B operator such that eps_flat = B_glob @ u.
    """
    n_elem, n_gp, _, nd = B_all.shape
    n_gp_total = n_elem * n_gp
    n_rows = 3 * n_gp_total

    # Flatten (elem, gp) -> single index for B
    B_flat = B_all.reshape(n_gp_total, 3, nd)   # (n_gp_total, 3, nd)

    # ----- Row indices -----
    # For each GP index g, rows are [3g, 3g+1, 3g+2], repeated for each local dof.
    gp_ids        = np.arange(n_gp_total)              # (n_gp_total,)
    rows_per_gp   = 3 * gp_ids[:, None] + np.arange(3) # (n_gp_total, 3)
    rows_full     = np.repeat(rows_per_gp[:, :, None], nd, axis=2)  # (n_gp_total, 3, nd)

    # ----- Column indices -----
    # Repeat connectivity for each GP of each element
    conn_expanded = np.repeat(conn, n_gp, axis=0)  # (n_gp_total, nd)
    cols_full     = np.broadcast_to(conn_expanded[:, None, :],
                                    (n_gp_total, 3, nd))            # (n_gp_total, 3, nd)

    # ----- Values -----
    vals_full = B_flat                                # (n_gp_total, 3, nd)

    # Flatten everything
    rows = rows_full.ravel()
    cols = cols_full.ravel()
    vals = vals_full.ravel()

    # Safety check
    assert rows.size == cols.size == vals.size, \
        f"rows={rows.size}, cols={cols.size}, vals={vals.size}"

    B_glob = coo_matrix((vals, (rows, cols)), shape=(n_rows, n_dof)).tocsr()
    return B_glob


def assemble_K_and_fint_vec(mat, conn, B_all, w_all,
                            eps_p_n, alpha_n,
                            u, n_dof, pattern_I, pattern_J):
    """
    Fully vectorized assembly of K(u) and f_int(u).

    Parameters
    ----------
    mat       : VonMisesIsotropicPlasticityPlaneStress (global)
    conn      : (n_elem, nd)
    B_all     : (n_elem, n_gp, 3, nd)
    w_all     : (n_elem, n_gp)
    eps_p_n   : (n_elem, n_gp, 3) committed plastic strain
    alpha_n   : (n_elem, n_gp)    committed eq. plastic strain
    u         : (n_dof,)
    n_dof     : int
    pattern_I, pattern_J : sparsity pattern for K

    Returns
    -------
    K_glob    : csr_matrix (n_dof, n_dof)
    fint_glob : (n_dof,)
    eps_p     : (n_elem, n_gp, 3) current plastic strain
    alpha     : (n_elem, n_gp)    current eq. plastic strain
    """

    n_elem, nd = conn.shape
    n_gp       = B_all.shape[1]

    # 1) Element displacements: u_e[e, j] = u[conn[e, j]]
    u_e = u[conn]  # (n_elem, nd)

    # 2) Strains at all GPs: eps[e,g,i] = B_all[e,g,i,j] * u_e[e,j]
    eps = np.einsum('egij,ej->egi', B_all, u_e)  # (n_elem, n_gp, 3)

    # 3) J2 return mapping (flatten all GPs into one batch)
    eps_flat     = eps.reshape(-1, 3)
    eps_p_n_flat = eps_p_n.reshape(-1, 3)
    alpha_n_flat = alpha_n.reshape(-1)

    sigma_flat, eps_p_flat, alpha_flat, Ctan_flat = \
        mat._return_mapping_batch(eps_flat, eps_p_n_flat, alpha_n_flat)

    # Reshape back
    sigma = sigma_flat.reshape(n_elem, n_gp, 3)          # (n_elem, n_gp, 3)
    eps_p = eps_p_flat.reshape(n_elem, n_gp, 3)          # (n_elem, n_gp, 3)
    alpha = alpha_flat.reshape(n_elem, n_gp)             # (n_elem, n_gp)
    Ctan  = Ctan_flat.reshape(n_elem, n_gp, 3, 3)        # (n_elem, n_gp, 3, 3)

    # 4) Internal force
    wsigma = w_all[..., None] * sigma                    # (n_elem, n_gp, 3)
    # fint_e[e, j] = sum_{g,i} wsigma[e,g,i] * B_all[e,g,i,j]
    fint_e = np.einsum('egi,egij->ej', wsigma, B_all)    # (n_elem, nd)

    fint_glob = np.zeros(n_dof, dtype=float)
    # assemble element contributions without explicit Python loop
    np.add.at(fint_glob, conn, fint_e)

    # 5) Tangent stiffness
    wC = w_all[..., None, None] * Ctan                   # (n_elem, n_gp, 3, 3)
    # Ke[e,m,n] = sum_{g,i,j} B[e,g,i,m] * wC[e,g,i,j] * B[e,g,j,n]
    Ke = np.einsum('egim,egij,egjn->emn', B_all, wC, B_all)  # (n_elem, nd, nd)

    K_vals = Ke.reshape(-1)  # must match pattern_I, pattern_J order

    K_glob = coo_matrix((K_vals, (pattern_I, pattern_J)),
                        shape=(n_dof, n_dof)).tocsr()

    return K_glob, fint_glob, eps_p, alpha


def extract_dirichlet_bcs(mp, idx_ux, idx_uy, step_index=0):
    """
    Read which nodes are fixed and what value they want at this step.
    Assumes processes/ApplyBoundaryConditions have already set:
      - FIXED flags
      - DISPLACEMENT_X/Y values
    """
    dofs = []
    vals = []

    for node in mp.Nodes:
        if node.IsFixed(KM.DISPLACEMENT_X):
            dofs.append(idx_ux[node.Id])
            vals.append(node.GetSolutionStepValue(KM.DISPLACEMENT_X, step_index))
        if node.IsFixed(KM.DISPLACEMENT_Y):
            dofs.append(idx_uy[node.Id])
            vals.append(node.GetSolutionStepValue(KM.DISPLACEMENT_Y, step_index))

    if not dofs:
        return np.zeros(0, dtype=int), np.zeros(0)
    return np.array(dofs, dtype=int), np.array(vals, dtype=float)


def homogenize_from_J2_vec(mat, conn, B_all, w_all, area_all,
                           eps_p_n, alpha_n, u):
    """
    Homogenization using committed J2 state (eps_p_n, alpha_n).

    Uses:
      eps = B u
      sigma from J2 (starting from committed state, but we do NOT update it)
      area-weighted average over the RVE.
    """

    n_elem, nd = conn.shape
    n_gp       = B_all.shape[1]

    # Element displacements
    u_e = u[conn]                          # (n_elem, nd)

    # Strains at all GPs
    eps = np.einsum('egij,ej->egi', B_all, u_e)  # (n_elem, n_gp, 3)

    # J2 evaluation from COMMITTED state (no update of eps_p_n/alpha_n)
    eps_flat     = eps.reshape(-1, 3)
    eps_p_n_flat = eps_p_n.reshape(-1, 3)
    alpha_n_flat = alpha_n.reshape(-1)

    sigma_flat, _, _, _ = mat._return_mapping_batch(
        eps_flat, eps_p_n_flat, alpha_n_flat
    )
    sigma = sigma_flat.reshape(n_elem, n_gp, 3)

    # Weights: w_all already = area / n_gp
    w = w_all[..., None]                   # (n_elem, n_gp, 1)

    # Raw integrals
    hom_strain_raw = np.sum(w * eps, axis=(0, 1))    # (3,)
    hom_stress_raw = np.sum(w * sigma, axis=(0, 1))  # (3,)

    RVE_area = np.sum(area_all)

    if RVE_area > 0.0:
        hom_strain = hom_strain_raw / RVE_area
        hom_stress = hom_stress_raw / RVE_area
    else:
        hom_strain = hom_strain_raw
        hom_stress = hom_stress_raw

    return hom_strain, hom_stress


# =========================================================
# Custom AnalysisStage (only for BC + model loading)
# =========================================================

class RVE_homogenization_dataset_generator(analysis_stage.AnalysisStage):
    def __init__(self, model, project_parameters):
        super().__init__(model, project_parameters)
        self.batch_strain = np.array([])

    def _CreateSolver(self):
        # We still create Kratos solver but never call SolveSolutionStep().
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
        Macro-strain → nodal displacements.
        """
        super().ApplyBoundaryConditions()

        Ex  = self.batch_strain[0]
        Ey  = self.batch_strain[1]
        Exy = self.batch_strain[2]

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


# =========================================================
# Single-batch FOM driver (theta, phi, out_dir)
# =========================================================

def run_fom_batch(theta,
                  phi,
                  parameters,
                  max_stretch_factor=0.01,
                  max_newton_it=100,
                  max_ls_it=5,
                  out_dir=DEFAULT_OUTPUT_DIR,
                  save_plot=True):
    """
    Run ONE FOM batch for given (theta, phi) and save per-batch .npy files.

    Parameters
    ----------
    theta, phi : float
        Loading direction angles [deg].
    parameters : KM.Parameters
        ProjectParameters.
    max_stretch_factor : float
        Lambda scaling for macro strain.
    max_newton_it : int
    max_ls_it     : int
    out_dir       : str
        Directory where .npy (and optional .png) will be saved.
    save_plot     : bool

    Returns
    -------
    strain_history_BC : (n_steps+1, 3)
    stress_history_BC : (n_steps+1, 3)
    U_history         : (n_steps+1, n_dof)
    log_line          : str
        Info line with theta, phi, and macro strain.
    """

    os.makedirs(out_dir, exist_ok=True)

    # Time stepping from Kratos settings
    dt       = parameters["solver_settings"]["time_stepping"]["time_step"].GetDouble()
    end_time = parameters["problem_data"]["end_time"].GetDouble()

    # Build model + analysis stage
    global_model = KM.Model()
    simulation = RVE_homogenization_dataset_generator(global_model, parameters)

    # Macro-strain definition
    simulation.batch_strain = max_stretch_factor * np.array([
        np.cos(np.radians(phi)),                               # E_xx
        np.sin(np.radians(theta)) * np.cos(np.radians(phi)),   # E_yy
        np.sin(np.radians(theta)) * np.sin(np.radians(phi)),   # E_xy
    ])

    log_line = (
        f"theta={theta:.2f}, phi={phi:.2f}, "
        f"strain={simulation.batch_strain.tolist()}"
    )

    # History containers
    strain_history_BC = [np.zeros(3)]
    stress_history_BC = [np.zeros(3)]

    simulation.Initialize()

    mp = simulation._GetSolver().GetComputingModelPart()
    idx_ux, idx_uy, n_dof = build_node_global_map(mp)

    # Global mesh arrays
    conn, B_all, w_all, area_all, pattern_I, pattern_J = \
        precompute_mesh_arrays(mp, idx_ux, idx_uy)

    n_elem, n_gp = B_all.shape[0], B_all.shape[1]

    # Material params (assumed same for all elements)
    elem0 = next(iter(mp.Elements))
    props0 = elem0.Properties
    E        = props0[KM.YOUNG_MODULUS]
    nu       = props0[KM.POISSON_RATIO]
    sigma_y0 = props0[KM.YIELD_STRESS]
    H        = 0.0  # perfect plasticity or as needed

    # Single global J2 object (no per-GP objects)
    mat = VonMisesIsotropicPlasticityPlaneStress(E, nu, sigma_y0, H)

    # Global plastic state: committed (n) and current (n+1)
    eps_p_n = np.zeros((n_elem, n_gp, 3), dtype=float)
    alpha_n = np.zeros((n_elem, n_gp), dtype=float)
    eps_p   = np.zeros_like(eps_p_n)
    alpha   = np.zeros_like(alpha_n)

    # initial displacements
    u = np.zeros(n_dof)

    time = 0.0
    step = 0

    # POD: store initial condition
    U_history = [u.copy()]

    # Predictor tangent (local to this batch)
    K_old = None

    while time < end_time - 1e-12:
        step += 1
        time += dt
        simulation.time = time
        simulation.step = step

        # 1) Macro-strain → nodal displacements (Dirichlet targets)
        simulation.ApplyBoundaryConditions()

        # 2) Set Dirichlet values into u for this step (initial guess)
        dirichlet_dofs, dirichlet_vals = extract_dirichlet_bcs(
            mp, idx_ux, idx_uy, step_index=0
        )
        if dirichlet_dofs.size > 0:
            u[dirichlet_dofs] = dirichlet_vals

        # 3) Newton–Raphson on FREE DOFs with K_old predictor + line search
        all_dofs = np.arange(n_dof, dtype=int)
        mask = np.ones(n_dof, dtype=bool)
        mask[dirichlet_dofs] = False
        free_dofs = all_dofs[mask]

        converged = False

        for it in range(max_newton_it):
            # Assemble at current u (full K and residual)
            K_glob, f_int, eps_p, alpha = assemble_K_and_fint_vec(
                mat, conn, B_all, w_all,
                eps_p_n, alpha_n,
                u, n_dof, pattern_I, pattern_J
            )

            R_f    = f_int[free_dofs]
            norm_R = np.linalg.norm(R_f, ord=np.inf)

            if it == 0:
                R0 = max(norm_R, 1.0)   # avoid division by zero and tiny scales

            rel = norm_R / R0  # relative residual

            print(f"[theta={theta:.2f}, phi={phi:.2f} | Step {step:03d} | "
                  f"NR it {it:02d}] ||R_f||_inf = {norm_R:.3e}   rel = {rel:.3e}")

            # Convergence test: relative OR absolute
            if rel < 1e-3 or norm_R < 1e-2:
                print(f"    -> Converged (rel={rel:.3e}, abs={norm_R:.3e})")
                converged = True
                break

            # Choose stiffness for the Newton step
            if it == 0 and K_old is not None:
                # Predictor: use stiffness from previous time step
                K_ff = K_old[free_dofs][:, free_dofs]
            else:
                # Pure Newton: use current tangent
                K_ff = K_glob[free_dofs][:, free_dofs]

            delta_u_f = spsolve(K_ff, -R_f)

            # -------------------------
            # Backtracking line search
            # -------------------------
            u_trial = u.copy()
            alpha_ls = 1.0
            for ls_it in range(max_ls_it):
                u_trial[free_dofs] = u[free_dofs] + alpha_ls * delta_u_f

                # recompute residual at u_trial (do NOT update committed state)
                _, f_int_trial, _, _ = assemble_K_and_fint_vec(
                    mat, conn, B_all, w_all,
                    eps_p_n, alpha_n,          # committed state at tn
                    u_trial, n_dof, pattern_I, pattern_J
                )

                R_f_trial    = f_int_trial[free_dofs]
                norm_R_trial = np.linalg.norm(R_f_trial, ord=np.inf)

                if norm_R_trial < norm_R:
                    # accept trial
                    u = u_trial
                    break
                else:
                    alpha_ls *= 0.5
            else:
                # if line search fails to reduce residual, take a small step
                u[free_dofs] += 0.1 * delta_u_f

        if not converged:
            print(f"Max Newton iterations reached at step {step} "
                  f"(||R_f||_inf = {norm_R:.3e})")

        # ---- after convergence of this time step ----
        # Recompute K_glob at converged u and store as predictor
        K_glob_conv, _, eps_p, alpha = assemble_K_and_fint_vec(
            mat, conn, B_all, w_all,
            eps_p_n, alpha_n,     # committed state at tn
            u, n_dof, pattern_I, pattern_J
        )
        K_old = K_glob_conv.copy()

        # Commit plastic state
        eps_p_n[:] = eps_p
        alpha_n[:] = alpha

        # 4) Homogenize from J2 at converged u
        step_strain_BC, step_stress_BC = homogenize_from_J2_vec(
            mat, conn, B_all, w_all, area_all,
            eps_p_n, alpha_n,
            u
        )

        strain_history_BC.append(step_strain_BC)
        stress_history_BC.append(step_stress_BC)

        # POD: store converged displacement
        U_history.append(u.copy())

    # Stack arrays
    strain_history_BC = np.stack(strain_history_BC, axis=0)
    stress_history_BC = np.stack(stress_history_BC, axis=0)
    U_history         = np.stack(U_history, axis=0)  # (n_steps+1, n_dof)

    # -------------------------------------------------
    # Save per-(theta, phi) .npy files
    # -------------------------------------------------
    tag = f"theta{theta:.1f}_phi{phi:.1f}"

    strain_file = os.path.join(out_dir, f"strain_{tag}.npy")
    stress_file = os.path.join(out_dir, f"stress_{tag}.npy")
    U_file      = os.path.join(out_dir, f"U_{tag}.npy")

    np.save(strain_file, strain_history_BC)
    np.save(stress_file, stress_history_BC)
    np.save(U_file,      U_history)

    print(f"[FOM] Saved strain to {strain_file}")
    print(f"[FOM] Saved stress to {stress_file}")
    print(f"[FOM] Saved displacements to {U_file}")

    # -------------------------------------------------
    # Optional plot for this batch
    # -------------------------------------------------
    if save_plot:
        Sxx = stress_history_BC[:, 0]
        Syy = stress_history_BC[:, 1]
        Sxy = stress_history_BC[:, 2]
        Exx = strain_history_BC[:, 0]
        Eyy = strain_history_BC[:, 1]
        Exy = strain_history_BC[:, 2]

        plt.figure()
        plt.plot(Exx, Sxx, marker='o', color='r', label="σ_xx (J2)")
        plt.plot(Eyy, Syy, marker='o', color='b', label="σ_yy (J2)")
        plt.plot(Exy, Sxy, marker='o', color='k', label="σ_xy (J2)")
        plt.xlabel("Strain [-]")
        plt.ylabel("Stress [Pa]")
        plt.title(f"theta={theta:.1f}, phi={phi:.1f}: homogenized response (J2)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        fig_name = os.path.join(out_dir, f"stress_strain_{tag}.png")
        plt.savefig(fig_name)
        plt.close()
        print(f"[FOM] Saved plot to {fig_name}")

    return strain_history_BC, stress_history_BC, U_history, log_line


# =========================================================
# Main: loop over theta, phi grid and call run_fom_batch
# =========================================================

def main(out_dir=DEFAULT_OUTPUT_DIR):
    with open("ProjectParameters.json", 'r') as parameter_file:
        parameters = KM.Parameters(parameter_file.read())

    angle_increment    = 25.0
    max_stretch_factor = 0.01  # lambda

    theta_vals = np.arange(0.0, 50.0 + 1e-8, angle_increment)
    phi_vals   = np.arange(0.0, 50.0 + 1e-8, angle_increment)

    log_lines = []
    batch = 0

    for theta in theta_vals:
        for phi in phi_vals:
            batch += 1
            print(f"\n[INFO] Starting batch {batch} with theta={theta:.2f}, phi={phi:.2f}")

            _, _, _, log_line = run_fom_batch(
                theta=theta,
                phi=phi,
                parameters=parameters,
                max_stretch_factor=max_stretch_factor,
                max_newton_it=100,
                max_ls_it=5,
                out_dir=out_dir,
                save_plot=True
            )

            log_lines.append(f"Batch {batch}: " + log_line)

    # Save a simple log in the same directory
    log_file = os.path.join(out_dir, "batch_log.txt")
    with open(log_file, "w") as f:
        f.write(f"Total batches: {batch}\n")
        f.write("Batch info (theta, phi, strain):\n")
        for line in log_lines:
            f.write(line + "\n")

    print(f"[FOM] Log stored in {log_file}")
    print(f"[FOM] Per-batch .npy files stored in directory: {out_dir}")


if __name__ == "__main__":
    main()
