#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import KratosMultiphysics as KM
import KratosMultiphysics.analysis_stage as analysis_stage
import importlib
from KratosMultiphysics.StructuralMechanicsApplication import \
    python_solvers_wrapper_structural as structural_solvers
import KratosMultiphysics.StructuralMechanicsApplication as SMApp  # noqa: F401
import numpy as np
import os
import matplotlib.pyplot as plt

from j2_plane_stress_plastic_strain_rve_simo import VonMisesIsotropicPlasticityPlaneStress

"""
RVE homogenization with multipath (0 -> E1 -> E2) in strain space.

- Kratos uses its own CL (SmallStrainIsotropicPlasticityPlaneStressVonMisesVonMises)
- "Ours" uses VonMisesIsotropicPlasticityPlaneStress in Python, driven by
  Kratos' GREEN_LAGRANGE_STRAIN_VECTOR at the IPs.

For each batch:
  - Choose base direction angles (theta, phi) [deg]
  - Compute E1 from (theta, phi, max_stretch_factor)
  - Sample gamma ~ U(0.100, 1.000)   (||E2|| = gamma * ||E1||)
           psi   ~ U(0.0, 180.0) [deg] (angle(E1, E2) in Voigt space)
  - Build E2 accordingly
  - Run a two-segment path:
        t in [0, T/2]:    E(t) = alpha(t)*E1,  alpha = t/(T/2)
        t in [T/2,   T]:  E(t) = (1-beta(t))*E1 + beta(t)*E2,  beta = (t-T/2)/(T/2)
  - BCs use:
        u_x = (Ex * x + Exy * y) * (t / T)
        u_y = (Ey * y + Exy * x) * (t / T)
    so we set batch_strain(t) = E(t) / (t / T)

We save:
  - data_set/all_histories_kratos.npz  (stress, strain)
  - data_set/all_histories_ours.npz    (stress, strain)
  - data_set/batch_#_stress_strain_compare.png
  - data_set/batch_log.txt
"""


# =====================================================================
# AnalysisStage + homogenization functions
# =====================================================================

class RVE_homogenization_dataset_generator(analysis_stage.AnalysisStage):

    def __init__(self, model, project_parameters):
        super().__init__(model, project_parameters)
        # Will be updated from the driver at each time step
        self.batch_strain = np.array([])

    def _CreateSolver(self):
        return structural_solvers.CreateSolver(self.model, self.project_parameters)

    def __CreateListOfProcesses(self):
        order_processes_initialization = self._GetOrderOfProcessesInitialization()
        self._list_of_processes        = self._CreateProcesses("processes", order_processes_initialization)
        deprecated_output_processes    = self._CheckDeprecatedOutputProcesses(self._list_of_processes)
        order_processes_initialization = self._GetOrderOfOutputProcessesInitialization()
        self._list_of_output_processes = self._CreateProcesses("output_processes", order_processes_initialization)
        # Adding the output processes to the regular processes
        self._list_of_processes.extend(self._list_of_output_processes)
        self._list_of_output_processes.extend(deprecated_output_processes)

    def ApplyBoundaryConditions(self):
        """
        Macro-strain → nodal displacements.

        Using convention:
          u_x = (Ex * x + Exy * y) * (time / end_time)
          u_y = (Ey * y + Exy * x) * (time / end_time)
          u_z = 0
        where batch_strain = [Ex, Ey, Exy].
        """
        super().ApplyBoundaryConditions()

        Ex  = self.batch_strain[0]
        Ey  = self.batch_strain[1]
        Exy = self.batch_strain[2]

        for node in self._GetSolver().GetComputingModelPart().Nodes:
            x_coord = node.X0
            y_coord = node.Y0
            z_coord = node.Z0

            factor  = self.time / self.end_time
            displ_x = (Ex * x_coord + Exy * y_coord) * factor
            displ_y = (Ey * y_coord + Exy * x_coord) * factor
            displ_z = 0.0

            if node.IsFixed(KM.DISPLACEMENT_X):
                node.SetSolutionStepValue(KM.DISPLACEMENT_X, displ_x)
            if node.IsFixed(KM.DISPLACEMENT_Y):
                node.SetSolutionStepValue(KM.DISPLACEMENT_Y, displ_y)
            if node.IsFixed(KM.DISPLACEMENT_Z):
                node.SetSolutionStepValue(KM.DISPLACEMENT_Z, displ_z)

    def CalculateHomogenizedStressAndStrain(self):
        """
        Homogenized GL strain and PK2 stress using Kratos constitutive law.

        NOTE: assumes BCs are already applied.
        """
        process_info = self._GetSolver().GetComputingModelPart().ProcessInfo
        computing_model_part = self._GetSolver().GetComputingModelPart()

        # Get number of IPs and voigt size from the first element
        for element in computing_model_part.Elements:
            dummy_strain = np.array(
                element.CalculateOnIntegrationPoints(KM.GREEN_LAGRANGE_STRAIN_VECTOR, process_info)
            )
            break  # assumes all elements have same number of IPs

        n_ips = dummy_strain.shape[0]
        voigt_size = dummy_strain.shape[1]

        homogenized_stress = np.zeros(voigt_size)
        homogenized_strain = np.zeros(voigt_size)
        RVE_area = 0.0

        for element in computing_model_part.Elements:
            strain = element.CalculateOnIntegrationPoints(KM.GREEN_LAGRANGE_STRAIN_VECTOR, process_info)
            stress = element.CalculateOnIntegrationPoints(KM.PK2_STRESS_VECTOR, process_info)

            strain_arr = np.array(strain)
            stress_arr = np.array(stress)

            strain_sum = np.sum(strain_arr, axis=0)
            stress_sum = np.sum(stress_arr, axis=0)

            element_area = element.GetGeometry().Area()

            RVE_area += element_area
            homogenized_stress += element_area * stress_sum / n_ips
            homogenized_strain += element_area * strain_sum / n_ips

        homogenized_stress /= RVE_area
        homogenized_strain /= RVE_area

        return homogenized_strain, homogenized_stress

    def CalculateHomogenizedStressAndStrainOurs(self):
        """
        Homogenized strain (from Kratos GL strain) and homogenized stress
        computed with our VonMisesIsotropicPlasticityPlaneStress model.

        NOTE:
        - Uses persistent J2 models per (elem, ip) to keep plastic history.
        - Commits all J2 models at the end of each call (i.e., per time step).
        """
        process_info = self._GetSolver().GetComputingModelPart().ProcessInfo
        computing_model_part = self._GetSolver().GetComputingModelPart()

        # Get number of IPs and voigt size from the first element
        for element in computing_model_part.Elements:
            dummy_strain = np.array(
                element.CalculateOnIntegrationPoints(KM.GREEN_LAGRANGE_STRAIN_VECTOR, process_info)
            )
            break  # assumes all elements have same number of IPs

        n_ips = dummy_strain.shape[0]
        voigt_size = dummy_strain.shape[1]  # should be 3 for 2D plane-stress case

        homogenized_strain = np.zeros(voigt_size)
        homogenized_stress_ours = np.zeros(voigt_size)
        RVE_area = 0.0

        # Persistent J2 models per (element, ip) to keep plastic history
        if not hasattr(self, "_j2_models"):
            # key: (elem_id, ip_index) -> VonMisesIsotropicPlasticityPlaneStress
            self._j2_models = {}

        # --- main loop over elements/IPs ---
        for element in computing_model_part.Elements:
            geom = element.GetGeometry()
            element_area = geom.Area()
            props = element.Properties

            # Material parameters from Kratos Properties
            E        = props[KM.YOUNG_MODULUS]
            nu       = props[KM.POISSON_RATIO]
            sigma_y0 = props[KM.YIELD_STRESS]
            H        = 0.0  # perfect plasticity (HARDENING_CURVE=3 in Kratos)

            # Green-Lagrange strain at IPs (≈ small strain for this problem)
            strain_list = element.CalculateOnIntegrationPoints(
                KM.GREEN_LAGRANGE_STRAIN_VECTOR, process_info
            )  # list of length n_ips

            for ip, eps_ip in enumerate(strain_list):
                eps_np = np.array(eps_ip, dtype=float)  # [Exx, Eyy, 2Exy]

                key = (element.Id, ip)
                if key not in self._j2_models:
                    self._j2_models[key] = VonMisesIsotropicPlasticityPlaneStress(
                        E, nu, sigma_y0, H
                    )

                mat = self._j2_models[key]
                sigma_ours, _ = mat.MaterialResponseAndTangent(eps_np)

                # weight = element area / number of IPs
                w = element_area / n_ips

                homogenized_strain      += w * eps_np
                homogenized_stress_ours += w * sigma_ours

            RVE_area += element_area

        # Normalize by total RVE area
        if RVE_area > 0.0:
            homogenized_strain      /= RVE_area
            homogenized_stress_ours /= RVE_area

        # --- commit all J2 models for the next global step ---
        for mat in self._j2_models.values():
            mat.Commit()

        return homogenized_strain, homogenized_stress_ours


# =====================================================================
# Helper: build E1 and E2 (same as before, no randomness here)
# =====================================================================

def _build_E1_E2(theta_deg, phi_deg, max_stretch_factor, gamma, psi_deg):
    """
    Build E1 and E2 in strain space (Exx, Eyy, Exy) for the multipath.

    E1 (base direction):
        E1 = lambda * [cos(phi), sin(theta)*cos(phi), sin(theta)*sin(phi)]

    E2: has norm gamma * ||E1|| and angle psi_deg w.r.t. E1 in R^3.
    """
    theta = np.radians(theta_deg)
    phi   = np.radians(phi_deg)
    psi   = np.radians(psi_deg)

    # Same direction as your single-path code
    dir1 = np.array([
        np.cos(phi),                    # Exx
        np.sin(theta) * np.cos(phi),    # Eyy
        np.sin(theta) * np.sin(phi)     # Exy
    ], dtype=float)

    norm_dir1 = np.linalg.norm(dir1)
    if norm_dir1 < 1e-14:
        raise ValueError("E1 direction is zero; choose different theta/phi.")

    e1 = dir1 / norm_dir1
    E1 = max_stretch_factor * dir1

    # Handle the almost-aligned case cleanly
    if abs(np.sin(psi)) < 1e-8:
        E2 = gamma * E1 * np.sign(np.cos(psi))
        return E1, E2

    # Build a perpendicular direction in strain space
    a = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(a, e1)) > 0.99:
        a = np.array([0.0, 1.0, 0.0])
    v = a - np.dot(a, e1) * e1
    nv = np.linalg.norm(v)
    if nv < 1e-14:
        a = np.array([0.0, 0.0, 1.0])
        v = a - np.dot(a, e1) * e1
        nv = np.linalg.norm(v)
        if nv < 1e-14:
            raise RuntimeError("Failed to construct a perpendicular direction in strain space.")
    e_perp1 = v / nv

    # Direction of E2 at angle psi from e1
    dir2 = np.cos(psi) * e1 + np.sin(psi) * e_perp1
    dir2 /= np.linalg.norm(dir2)

    # ||E2|| = gamma * ||E1||
    E2_norm = gamma * np.linalg.norm(E1)
    E2 = E2_norm * dir2

    return E1, E2


# =====================================================================
# MAIN: multipath driver with SAME RANDOM LOGIC as your Kratos-only script
# =====================================================================

with open("ProjectParameters.json", 'r') as parameter_file:
    parameters = KM.Parameters(parameter_file.read())

analysis_stage_module_name = parameters["analysis_stage"].GetString()
analysis_stage_class_name = analysis_stage_module_name.split('.')[-1]
analysis_stage_class_name = ''.join(x.title() for x in analysis_stage_class_name.split('_'))

analysis_stage_module = importlib.import_module(analysis_stage_module_name)
analysis_stage_class = getattr(analysis_stage_module, analysis_stage_class_name)

end_time = parameters["problem_data"]["end_time"].GetDouble()
t_mid    = 0.5 * end_time

log_lines = []

theta = 0.0
phi   = 0.0
angle_increment    = 25.0
max_stretch_factor = 0.005  # lambda for E1

# --- SAME RANDOM LOGIC as the Kratos-only multipath script ---
np.random.seed(1234)

# Here we will store the strain and stress histories for all batches
all_strain_histories       = []   # Kratos
all_stress_histories       = []   # Kratos
all_strain_histories_ours  = []   # Ours (J2)
all_stress_histories_ours  = []   # Ours (J2)

batch = 0

while theta <= 25.0 + 1e-8:
    while phi <= 25.0 + 1e-8:
        batch += 1
        print(f"\n[INFO] Batch {batch}: theta={theta}, phi={phi}")

        # --- RANDOM gamma and psi (same ranges as your other script) ---
        gamma   = np.random.uniform(0.100, 1.000)   # magnitude factor
        psi_deg = np.random.uniform(0.0,   180.0)   # angle between E1 and E2 [deg]

        # Build E1 and E2
        E1, E2 = _build_E1_E2(theta, phi, max_stretch_factor, gamma, psi_deg)

        print(f"       gamma = {gamma:.3f}, psi = {psi_deg:.1f} deg")
        print(f"       E1 = {E1}")
        print(f"       E2 = {E2}")

        # New analysis_stage per batch
        global_model = KM.Model()
        simulation = RVE_homogenization_dataset_generator(global_model, parameters)

        log_lines.append(
            "Batch {}: theta={:.2f}, phi={:.2f}, "
            "gamma={:.3f}, psi_deg={:.1f}, "
            "E1={}, E2={}".format(
                batch, theta, phi, gamma, psi_deg,
                E1.tolist(), E2.tolist()
            )
        )

        # histories for this batch, start with zeros
        strain_history       = [np.zeros(3)]  # Kratos
        stress_history       = [np.zeros(3)]  # Kratos
        strain_history_ours  = [np.zeros(3)]  # Ours
        stress_history_ours  = [np.zeros(3)]  # Ours

        simulation.Initialize()

        while simulation.KeepAdvancingSolutionLoop():
            simulation.time = simulation._AdvanceTime()
            time = simulation.time
            t    = time

            # 0 → E1 → E2 multipath
            if t <= t_mid:
                # Segment 1: 0 -> E1
                alpha_seg = t / max(t_mid, 1e-14)
                E_t = alpha_seg * E1
                seg_id = 1
            else:
                # Segment 2: E1 -> E2
                beta_seg = (t - t_mid) / max(end_time - t_mid, 1e-14)
                E_t = (1.0 - beta_seg) * E1 + beta_seg * E2
                seg_id = 2

            # Inverse scaling trick: BCs use (time/end_time)
            lambda_t = t / max(end_time, 1e-14)
            if lambda_t < 1e-14:
                effective_batch_strain = np.zeros(3)
            else:
                effective_batch_strain = E_t / lambda_t

            simulation.batch_strain = effective_batch_strain

            print(f"  [Step t={t:.5f}, seg={seg_id}] "
                  f"E(t)={E_t}, batch_strain={simulation.batch_strain}")

            # Standard Kratos solution loop
            simulation.InitializeSolutionStep()
            simulation._GetSolver().Predict()
            is_converged = simulation._GetSolver().SolveSolutionStep()
            simulation.FinalizeSolutionStep()

            # Kratos homogenized GL strain + PK2 stress
            step_strain, step_stress = simulation.CalculateHomogenizedStressAndStrain()

            # Our J2-based homogenized stress
            step_strain_ours, step_stress_ours = simulation.CalculateHomogenizedStressAndStrainOurs()

            strain_history.append(step_strain)
            stress_history.append(step_stress)
            strain_history_ours.append(step_strain_ours)
            stress_history_ours.append(step_stress_ours)

        simulation.Finalize()

        all_strain_histories.append(np.stack(strain_history, axis=0))
        all_stress_histories.append(np.stack(stress_history, axis=0))
        all_strain_histories_ours.append(np.stack(strain_history_ours, axis=0))
        all_stress_histories_ours.append(np.stack(stress_history_ours, axis=0))

        phi += angle_increment

    phi = 0.0
    theta += angle_increment

# ==============================
# Stack tensors
# ==============================
strain_tensor      = np.stack(all_strain_histories,      axis=0)  # [batch, steps, 3] Kratos
stress_tensor      = np.stack(all_stress_histories,      axis=0)  # [batch, steps, 3] Kratos
strain_tensor_ours = np.stack(all_strain_histories_ours, axis=0)  # [batch, steps, 3] Ours
stress_tensor_ours = np.stack(all_stress_histories_ours, axis=0)  # [batch, steps, 3] Ours

# ==============================
# Save datasets
# ==============================
os.makedirs("data_set", exist_ok=True)

np.savez("data_set/all_histories_kratos.npz",
         stress=stress_tensor,
         strain=strain_tensor)

np.savez("data_set/all_histories_ours.npz",
         stress=stress_tensor_ours,
         strain=strain_tensor_ours)

print("Results stored in:")
print("  data_set/all_histories_kratos.npz")
print("  data_set/all_histories_ours.npz")

# ==============================
# Plots per batch: Kratos vs Ours
# ==============================
for batch_idx in range(stress_tensor.shape[0]):
    Sxx_k = stress_tensor[batch_idx, :, 0]
    Syy_k = stress_tensor[batch_idx, :, 1]
    Sxy_k = stress_tensor[batch_idx, :, 2]
    Exx_k = strain_tensor[batch_idx, :, 0]
    Eyy_k = strain_tensor[batch_idx, :, 1]
    Exy_k = strain_tensor[batch_idx, :, 2]

    Sxx_o = stress_tensor_ours[batch_idx, :, 0]
    Syy_o = stress_tensor_ours[batch_idx, :, 1]
    Sxy_o = stress_tensor_ours[batch_idx, :, 2]
    Exx_o = strain_tensor_ours[batch_idx, :, 0]
    Eyy_o = strain_tensor_ours[batch_idx, :, 1]
    Exy_o = strain_tensor_ours[batch_idx, :, 2]

    plt.figure()
    # Kratos
    plt.plot(Exx_k, Sxx_k, 'r-o',  label="σ_xx Kratos")
    plt.plot(Eyy_k, Syy_k, 'b-o',  label="σ_yy Kratos")
    plt.plot(Exy_k, Sxy_k, 'k-o',  label="σ_xy Kratos")
    # Ours
    plt.plot(Exx_o, Sxx_o, 'r--x', label="σ_xx Ours")
    plt.plot(Eyy_o, Syy_o, 'b--x', label="σ_yy Ours")
    plt.plot(Exy_o, Sxy_o, 'k--x', label="σ_xy Ours")

    plt.xlabel("Strain [-]")
    plt.ylabel("Stress [Pa]")
    plt.title(f"Batch {batch_idx+1}: Kratos vs Ours (multipath)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"data_set/batch_{batch_idx+1}_stress_strain_compare.png")
    plt.close()

with open("data_set/batch_log.txt", "w") as f:
    f.write(f"Total batches: {batch}\n")
    f.write("Batch info (theta, phi, gamma, psi_deg, E1, E2):\n")
    for line in log_lines:
        f.write(line + "\n")

print("Plots stored in data_set/batch_*_stress_strain_compare.png")
print("Log stored in data_set/batch_log.txt")
