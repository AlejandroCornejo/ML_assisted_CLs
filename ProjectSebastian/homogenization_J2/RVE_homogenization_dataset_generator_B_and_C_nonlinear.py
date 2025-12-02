#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import KratosMultiphysics as KM
import KratosMultiphysics.analysis_stage as analysis_stage
import importlib
from KratosMultiphysics.StructuralMechanicsApplication import \
    python_solvers_wrapper_structural as structural_solvers
import KratosMultiphysics.StructuralMechanicsApplication as SMApp  # noqa: F401
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
import os
import matplotlib.pyplot as plt

from j2_plane_stress_plastic_strain_rve_simo import VonMisesIsotropicPlasticityPlaneStress

"""
RVE small-strain homogenization with our OWN nonlinear J2 solver.

Kratos is only used to:
  - Read model, geometry, Properties, BC processes (ProjectParameters.json)
  - Mark which nodes are fixed and prescribe DISPLACEMENT_X/Y (ApplyBoundaryConditions)

We do:
  - Build B_glob and gp_meta_B (for ε = B u)
  - Build one J2 material per Gauss point
  - Assemble K(u) and f_int(u) element-wise:
        K(u)     = sum_e sum_gp B_e^T (w_e C_tangent_gp) B_e
        f_int(u) = sum_e sum_gp B_e^T (w_e sigma_gp)
  - Newton–Raphson per time step (displacement-driven):
        R(u) = f_int(u) = 0  on FREE DOFs
  - At converged u(t): ε = B u, σ from J2, homogenize σ, ε.

No Kratos structural solver, no GL/PK2, no Kratos-based homogenization.
"""

REL_EPS = 1e-14


# =========================================================
# Small helpers
# =========================================================


def init_j2_materials_per_gauss_point(mp):
    """
    Create one VonMisesIsotropicPlasticityPlaneStress object
    per (element, gauss_point).
    """
    materials = {}

    for elem in mp.Elements:
        geom = elem.GetGeometry()

        # Number of Gauss points from shape function values
        N_gp = np.array(geom.ShapeFunctionsValues())  # (n_gp, nnode)
        n_gp = N_gp.shape[0]

        props = elem.Properties

        # --- Material parameters from Kratos Properties ---
        E        = props[KM.YOUNG_MODULUS]
        nu       = props[KM.POISSON_RATIO]
        sigma_y0 = props[KM.YIELD_STRESS]

        # Perfect plasticity (H = 0)
        H = 0.0

        for igauss in range(n_gp):
            key = (elem.Id, igauss)
            materials[key] = VonMisesIsotropicPlasticityPlaneStress(E, nu, sigma_y0, H)

    return materials


def precompute_element_data(mp, idx_ux, idx_uy):
    elem_data = {}
    for elem in mp.Elements:
        geom = elem.GetGeometry()
        nnode = geom.PointsNumber()

        # DOF indices
        col_ids = []
        for node in geom:
            col_ids.append(idx_ux[node.Id])
            col_ids.append(idx_uy[node.Id])

        Ns   = np.array(geom.ShapeFunctionsValues())
        n_gp = Ns.shape[0]
        area = geom.Area()

        B_list = []
        w_list = []

        for igauss in range(n_gp):
            DNDe = np.array(geom.ShapeFunctionDerivatives(1, igauss))
            J    = np.array(geom.Jacobian(igauss))
            DNDX = DNDe @ np.linalg.inv(J)

            B_loc = build_B_from_DNDX(DNDX)     # 3 × 2*nnode
            B_list.append(B_loc)
            w_list.append(area/n_gp)

        elem_data[elem.Id] = {
            "col_ids": np.array(col_ids, dtype=int),
            "B": np.stack(B_list, axis=0),       # (n_gp,3,2*nnode)
            "w": np.array(w_list)
        }
    return elem_data


def precompute_sparsity_pattern(elem_data, n_dof):
    I = []
    J = []
    for data in elem_data.values():
        col_ids = data["col_ids"]
        for a in col_ids:
            for b in col_ids:
                I.append(a)
                J.append(b)
    return np.array(I, dtype=int), np.array(J, dtype=int)


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


def assemble_global_B(mp, idx_ux, idx_uy):
    """
    Constructs the global strain–displacement operator B and the mapping
    between each 3-row block and the corresponding Gauss point.
    """
    rows, cols, vals = [], [], []
    gp_meta = []          # will record (element_id, gauss_index)
    row_base = 0          # row counter in the global B

    for elem in mp.Elements:
        geom = elem.GetGeometry()
        nnode = geom.PointsNumber()

        # DOF indices in the global vector for this element
        col_ids = []
        for node in geom:
            col_ids.append(idx_ux[node.Id])
            col_ids.append(idx_uy[node.Id])

        N_gp = np.array(geom.ShapeFunctionsValues())   # shape (n_gp, nnode)
        n_gp = N_gp.shape[0]

        for igauss in range(n_gp):

            # Compute derivatives in global coordinates
            DNDe = np.array(geom.ShapeFunctionDerivatives(1, igauss))
            J    = np.array(geom.Jacobian(igauss))
            DNDX = DNDe @ np.linalg.inv(J)

            # Local 3×(2*nnode) B-matrix
            B_loc = build_B_from_DNDX(DNDX)

            # Insert B_loc into the global sparse structure
            for i in range(3):              # 3 rows per Gauss point
                r = row_base + i
                for a in range(2*nnode):    # 2 dofs per node
                    rows.append(r)
                    cols.append(col_ids[a])
                    vals.append(B_loc[i, a])

            # Record mapping from row block → (element, gp index)
            gp_meta.append((elem.Id, igauss))
            row_base += 3

    n_rows = row_base
    n_cols = 2 * len(idx_ux)   # total DOFs = 2 per node

    B_glob = coo_matrix((vals, (rows, cols)), shape=(n_rows, n_cols)).tocsr()
    return B_glob, gp_meta


def assemble_K_and_fint(mp, idx_ux, idx_uy, materials, u, elem_data, pattern_I, pattern_J):

    n_dof = 2 * mp.NumberOfNodes()
    fint_glob = np.zeros(n_dof)
    K_vals    = np.zeros(len(pattern_I))

    counter = 0

    for elem in mp.Elements:

        data    = elem_data[elem.Id]
        col_ids = data["col_ids"]
        B_all   = data["B"]            # (n_gp, 3, nd)
        w_all   = data["w"]            # (n_gp,)
        n_gp    = B_all.shape[0]
        nd      = len(col_ids)

        u_e = u[col_ids]
        eps_all = B_all @ u_e

        sigma_all = np.zeros((n_gp, 3))
        Ctan_all  = np.zeros((n_gp, 3, 3))

        for g, eps_gp in enumerate(eps_all):
            mat = materials[(elem.Id, g)]
            s, C = mat.MaterialResponseAndTangent(eps_gp)
            sigma_all[g] = s
            Ctan_all[g]  = C

        wsigma = w_all[:, None] * sigma_all
        fint_e = np.einsum("gi, gij -> j", wsigma, B_all)
        fint_glob[col_ids] += fint_e

        wC  = w_all[:, None, None] * Ctan_all
        Ke  = np.einsum("gim, gij, gjn -> mn", B_all, wC, B_all)

        size = nd * nd
        K_vals[counter:counter+size] = Ke.ravel()
        counter += size

    K_glob_sparse = coo_matrix((K_vals, (pattern_I, pattern_J)),
                               shape=(n_dof, n_dof)).tocsr()

    return K_glob_sparse, fint_glob


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


def homogenize_from_J2(mp, B_glob, gp_meta_B, materials, u):
    """
    J2 version of homogenization (NO commit):

      - Strain:  ε_gp = (B_glob @ u)[gp]
      - Stress:  σ_gp from the J2 material at each Gauss point
                 via Evaluate(eps_gp) using committed state.

    Then area/n_ip weighting same as Alejandro.
    """
    # 1) ε_B = B u  (all GPs, concatenated)
    eps_B = B_glob @ u                     # shape (3 * n_gp_total,)
    eps_B_reshaped = eps_B.reshape(-1, 3)  # (n_gp_total, 3)

    # 2) σ from J2 material at each GP (pure evaluation, no commit)
    n_gp_total = eps_B_reshaped.shape[0]
    sigma_B = np.zeros_like(eps_B_reshaped)

    for igp, (elem_id, igauss) in enumerate(gp_meta_B):
        mat = materials[(elem_id, igauss)]
        eps_gp = eps_B_reshaped[igp, :]
        sigma_gp, _, _, _ = mat.Evaluate(eps_gp)
        sigma_B[igp, :] = sigma_gp

    # 3) Same area-weighted homogenization as Alejandro

    # count IPs per element
    n_ips_dict = {}
    for elem_id, igauss in gp_meta_B:
        n_ips_dict[elem_id] = n_ips_dict.get(elem_id, 0) + 1

    # element areas & total area
    area_dict = {}
    RVE_area = 0.0
    for elem in mp.Elements:
        area = elem.GetGeometry().Area()
        area_dict[elem.Id] = area
        RVE_area += area

    hom_strain_raw = np.zeros(3)
    hom_stress_raw = np.zeros(3)

    for igp, (elem_id, igauss) in enumerate(gp_meta_B):
        area = area_dict[elem_id]
        n_ips_elem = n_ips_dict[elem_id]
        w = area / n_ips_elem  # same logic: area * (sum_ip / n_ips)

        hom_strain_raw += w * eps_B_reshaped[igp, :]
        hom_stress_raw += w * sigma_B[igp, :]

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
# Main
# =========================================================

with open("ProjectParameters.json", 'r') as parameter_file:
    parameters = KM.Parameters(parameter_file.read())

analysis_stage_module_name = parameters["analysis_stage"].GetString()
analysis_stage_class_name = analysis_stage_module_name.split('.')[-1]
analysis_stage_class_name = ''.join(x.title() for x in analysis_stage_class_name.split('_'))

analysis_stage_module = importlib.import_module(analysis_stage_module_name)
analysis_stage_class = getattr(analysis_stage_module, analysis_stage_class_name)

log_lines = []

# Time stepping from Kratos settings
dt       = parameters["solver_settings"]["time_stepping"]["time_step"].GetDouble()
end_time = parameters["problem_data"]["end_time"].GetDouble()

theta = 25.0
phi   = 50.0
angle_increment    = 25.0
max_stretch_factor = 0.005  # lambda

all_strain_histories_BC = []
all_stress_histories_BC = []
batch = 0

# Newton settings
max_newton_it = 100
max_ls_it     = 5   # max line-search backtracking iterations

K_old = None   # previous step stiffness (for modified Newton / predictor)

while theta <= 25.0 + 1e-8:
    while phi <= 50.0 + 1e-8:
        batch += 1
        print(f"\n[INFO] Starting batch {batch} with theta={theta:.2f}, phi={phi:.2f}")

        global_model = KM.Model()
        simulation = RVE_homogenization_dataset_generator(global_model, parameters)

        # Same macro-strain as original script
        simulation.batch_strain = max_stretch_factor * np.array([
            np.cos(np.radians(phi)),                               # E_xx
            np.sin(np.radians(theta))  * np.cos(np.radians(phi)),  # E_yy
            (np.sin(np.radians(theta)) * np.sin(np.radians(phi))), # E_xy
        ])

        log_lines.append(
            f"Batch {batch}: theta={theta:.2f}, phi={phi:.2f}, "
            f"strain={simulation.batch_strain.tolist()}"
        )

        # History for this batch
        strain_history_BC = [np.zeros(3)]
        stress_history_BC = [np.zeros(3)]

        simulation.Initialize()

        mp = simulation._GetSolver().GetComputingModelPart()
        materials = init_j2_materials_per_gauss_point(mp)
        idx_ux, idx_uy, n_dof = build_node_global_map(mp)
        elem_data = precompute_element_data(mp, idx_ux, idx_uy)
        pattern_I, pattern_J = precompute_sparsity_pattern(elem_data, n_dof)
        B_glob, gp_meta_B = assemble_global_B(mp, idx_ux, idx_uy)

        # initial displacements
        u = np.zeros(n_dof)

        time = 0.0
        step = 0

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
                K_glob, f_int = assemble_K_and_fint(
                    mp, idx_ux, idx_uy, materials, u, elem_data, pattern_I, pattern_J
                )

                R_f = f_int[free_dofs]
                norm_R = np.linalg.norm(R_f, ord=np.inf)

                if it == 0:
                    R0 = max(norm_R, 1.0)   # avoid division by zero and tiny scales

                rel = norm_R / R0  # relative residual

                print(f"[Batch {batch} | Step {step:03d} | "
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
                alpha   = 1.0
                for ls_it in range(max_ls_it):
                    u_trial[free_dofs] = u[free_dofs] + alpha * delta_u_f

                    # recompute residual at u_trial
                    _, f_int_trial = assemble_K_and_fint(
                        mp, idx_ux, idx_uy, materials,
                        u_trial, elem_data, pattern_I, pattern_J
                    )
                    R_f_trial = f_int_trial[free_dofs]
                    norm_R_trial = np.linalg.norm(R_f_trial, ord=np.inf)

                    if norm_R_trial < norm_R:
                        # accept trial
                        u = u_trial
                        break
                    else:
                        alpha *= 0.5

                else:
                    # if line search fails to reduce residual, take a small step
                    u[free_dofs] += 0.1 * delta_u_f

            if not converged:
                print(f"Max Newton iterations reached at step {step} "
                      f"(||R_f||_inf = {norm_R:.3e})")

            # ---- after convergence of this time step ----
            # Recompute K_glob at converged u and store as predictor
            K_glob_conv, _ = assemble_K_and_fint(
                mp, idx_ux, idx_uy, materials, u, elem_data, pattern_I, pattern_J
            )
            K_old = K_glob_conv.copy()

            # 3b) Commit plastic state for this converged time step
            for mat in materials.values():
                mat.Commit()

            # 4) Homogenize from J2 at converged u (no commit inside)
            step_strain_BC, step_stress_BC = homogenize_from_J2(
                mp, B_glob, gp_meta_B, materials, u
            )

            strain_history_BC.append(step_strain_BC)
            stress_history_BC.append(step_stress_BC)

        # Optionally: simulation.Finalize()

        all_strain_histories_BC.append(np.stack(strain_history_BC, axis=0))
        all_stress_histories_BC.append(np.stack(stress_history_BC, axis=0))

        phi += angle_increment

    phi = 0.0
    theta += angle_increment

# =========================================================
# Save and simple plots
# =========================================================

strain_tensor_BC = np.stack(all_strain_histories_BC, axis=0)
stress_tensor_BC = np.stack(all_stress_histories_BC, axis=0)

os.makedirs("data_set", exist_ok=True)
np.savez("data_set/all_stress_histories.npz", stress=stress_tensor_BC)
np.savez("data_set/all_strain_histories.npz", strain=strain_tensor_BC)

print("Results stored in:")
print("  data_set/all_stress_histories.npz")
print("  data_set/all_strain_histories.npz")

for batch_idx in range(stress_tensor_BC.shape[0]):
    Sxx = stress_tensor_BC[batch_idx, :, 0]
    Syy = stress_tensor_BC[batch_idx, :, 1]
    Sxy = stress_tensor_BC[batch_idx, :, 2]
    Exx = strain_tensor_BC[batch_idx, :, 0]
    Eyy = strain_tensor_BC[batch_idx, :, 1]
    Exy = strain_tensor_BC[batch_idx, :, 2]

    plt.figure()
    plt.plot(Exx, Sxx, marker='o', color='r', label="σ_xx (J2)")
    plt.plot(Eyy, Syy, marker='o', color='b', label="σ_yy (J2)")
    plt.plot(Exy, Sxy, marker='o', color='k', label="σ_xy (J2)")
    plt.xlabel("Strain [-]")
    plt.ylabel("Stress [Pa]")
    plt.title(f"Batch {batch_idx+1}: homogenized response (J2)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"data_set/batch_{batch_idx+1}_stress_strain.png")
    plt.close()

with open("data_set/batch_log.txt", "w") as f:
    f.write(f"Total batches: {batch}\n")
    f.write("Batch info (theta, phi, strain):\n")
    for line in log_lines:
        f.write(line + "\n")

print("Plots stored in data_set/batch_*_stress_strain.png")
print("Log stored in data_set/batch_log.txt")
