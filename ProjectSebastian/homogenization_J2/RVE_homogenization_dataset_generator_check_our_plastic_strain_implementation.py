
import KratosMultiphysics as KM
import KratosMultiphysics.analysis_stage as analysis_stage
import importlib
from KratosMultiphysics.StructuralMechanicsApplication import python_solvers_wrapper_structural as structural_solvers
import KratosMultiphysics.StructuralMechanicsApplication as SMApp
import numpy as np
import os
import matplotlib.pyplot as plt
from j2_plane_stress_plastic_strain_rve_simo import VonMisesIsotropicPlasticityPlaneStress

"""
@file RVE_homogenization_dataset_generator.py
@brief Generator for RVE homogenization datasets using Kratos Multiphysics.

This script runs homogenization simulations on a Representative Volume Element (RVE)
for different combinations of applied strains (defined by theta and phi angles).
For each batch (combination of angles), it stores the homogenized strain and stress histories
in .npz files and generates plots of the stress-strain curves.
Additionally, it creates a log file with information about each batch.

- Saves stress and strain histories in: data_set/all_stress_histories.npz and data_set/all_strain_histories.npz
- Saves a log of the batches in: data_set/batch_log.txt
- Saves stress-strain plots for each batch in: data_set/batch_X_stress_strain_plots.png

@author Alejandro Cornejo
@date 2025

@section Usage
1. Configure the ProjectParameters.json file with the model parameters.
2. Run this script to generate the homogenization dataset.

@section Details
- Uses KratosMultiphysics and StructuralMechanicsApplication.
- Each batch corresponds to a combination of theta and phi.
- The history of each batch is stored as a tensor [batch, steps, voigt_size].
"""

class RVE_homogenization_dataset_generator(analysis_stage.AnalysisStage):

    def __init__(self, model, project_parameters):
        super().__init__(model, project_parameters)

        self.batch_strain = np.array([])

    def _CreateSolver(self):
        return structural_solvers.CreateSolver(self.model, self.project_parameters)

    def __CreateListOfProcesses(self):
        order_processes_initialization = self._GetOrderOfProcessesInitialization()
        self._list_of_processes        = self._CreateProcesses("processes", order_processes_initialization)
        deprecated_output_processes    = self._CheckDeprecatedOutputProcesses(self._list_of_processes)
        order_processes_initialization = self._GetOrderOfOutputProcessesInitialization()
        self._list_of_output_processes = self._CreateProcesses("output_processes", order_processes_initialization)
        self._list_of_processes.extend(self._list_of_output_processes) # Adding the output processes to the regular processes
        self._list_of_output_processes.extend(deprecated_output_processes)

    def ApplyBoundaryConditions(self):
        super().ApplyBoundaryConditions()

        Ex = self.batch_strain[0]
        Ey = self.batch_strain[1]
        Exy = self.batch_strain[2]

        for node in self._GetSolver().GetComputingModelPart().Nodes:
            # NOTE: here we assume that one corner of the RVE is at (0,0,0)
            x_coord = node.X0
            y_coord = node.Y0
            z_coord = node.Z0
            displ_x = (Ex * x_coord + Exy * y_coord) * self.time / self.end_time
            displ_y = (Ey * y_coord + Exy * x_coord) * self.time / self.end_time
            displ_z = 0.0  # Assuming no displacement in Z direction for 2D RVE

            if node.IsFixed(KM.DISPLACEMENT_X):
                node.SetSolutionStepValue(KM.DISPLACEMENT_X, displ_x)
            if node.IsFixed(KM.DISPLACEMENT_Y):
                node.SetSolutionStepValue(KM.DISPLACEMENT_Y, displ_y)
            if node.IsFixed(KM.DISPLACEMENT_Z):
                node.SetSolutionStepValue(KM.DISPLACEMENT_Z, displ_z)

    def CalculateHomogenizedStressAndStrain(self): # NOTE: it assumed that the BCs are already applied

        process_info = self._GetSolver().GetComputingModelPart().ProcessInfo
        computing_model_part = self._GetSolver().GetComputingModelPart()

        for element in computing_model_part.Elements:
            dummy_strain = np.array(element.CalculateOnIntegrationPoints(KM.GREEN_LAGRANGE_STRAIN_VECTOR, process_info))
            break # NOTE: this assumes that all elements have the same number of IP
        n_ips = dummy_strain.shape[0]
        voigt_size  = dummy_strain.shape[1]

        homogenized_stress = np.zeros(voigt_size)
        homogenized_strain  = np.zeros(voigt_size)
        RVE_area = 0.0

        for element in computing_model_part.Elements:
            strain = element.CalculateOnIntegrationPoints(KM.GREEN_LAGRANGE_STRAIN_VECTOR, process_info)
            stress = element.CalculateOnIntegrationPoints(KM.PK2_STRESS_VECTOR, process_info)

            stress_vector_sum_ip = np.sum(np.array(stress), axis=0)
            strain_vector_sum_ip = np.sum(np.array(strain), axis=0)

            element_area = element.GetGeometry().Area()

            RVE_area += element_area
            homogenized_stress += element_area * stress_vector_sum_ip / n_ips
            homogenized_strain  += element_area * strain_vector_sum_ip / n_ips

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
            H        = 0.0  # perfect plasticity

            # Green-Lagrange strain at IPs (≈ small strain for this problem)
            strain_list = element.CalculateOnIntegrationPoints(
                KM.GREEN_LAGRANGE_STRAIN_VECTOR, process_info
            )  # list of length n_ips

            for ip, eps_ip in enumerate(strain_list):
                eps_np = np.array(eps_ip, dtype=float)  # [Exx, Eyy, 2Exy] for 2D

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


#====================================================================================================
#====================================================================================================

with open("ProjectParameters.json", 'r') as parameter_file:
    parameters = KM.Parameters(parameter_file.read())

analysis_stage_module_name = parameters["analysis_stage"].GetString()
analysis_stage_class_name = analysis_stage_module_name.split('.')[-1]
analysis_stage_class_name = ''.join(x.title() for x in analysis_stage_class_name.split('_'))

analysis_stage_module = importlib.import_module(analysis_stage_module_name)
analysis_stage_class = getattr(analysis_stage_module, analysis_stage_class_name)

log_lines = []

theta = 0.0
phi = 0.0
angle_increment = 25.0
max_stretch_factor  = 0.005 # lambda

# Here we will store the strain and stress histories for all batches
all_strain_histories       = []   # Kratos
all_stress_histories       = []   # Kratos
all_strain_histories_ours  = []   # Ours (J2)
all_stress_histories_ours  = []   # Ours (J2)

batch = 0

while theta <= 25.0:
    while phi <= 25.0:
        batch += 1

        # NOTE: Each batch creates a new analysis_stage
        global_model = KM.Model()
        simulation = RVE_homogenization_dataset_generator(global_model, parameters)

        simulation.batch_strain = max_stretch_factor * np.array([
                np.cos(np.radians(phi)),                               # E_xx
                np.sin(np.radians(theta))  * np.cos(np.radians(phi)),  # E_yy
                (np.sin(np.radians(theta)) * np.sin(np.radians(phi))), # E_xy
        ])

        log_lines.append(
            f"Batch {batch}: theta={theta:.2f}, phi={phi:.2f}, strain={simulation.batch_strain.tolist()}"
        )

        # strains/stresses for one batch history, start with null values
        strain_history       = [[0, 0, 0]]  # Kratos
        stress_history       = [[0, 0, 0]]  # Kratos
        strain_history_ours  = [[0, 0, 0]]  # Ours
        stress_history_ours  = [[0, 0, 0]]  # Ours

        simulation.Initialize()
        while simulation.KeepAdvancingSolutionLoop():
            simulation.time = simulation._AdvanceTime()
            simulation.InitializeSolutionStep()
            simulation._GetSolver().Predict()
            is_converged = simulation._GetSolver().SolveSolutionStep()
            simulation.FinalizeSolutionStep()

            # --- Kratos homogenized GL strain + PK2 stress
            step_strain, step_stress = simulation.CalculateHomogenizedStressAndStrain()

            # --- Our homogenized strain + Cauchy stress from VonMisesIsotropicPlasticityPlaneStress
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
strain_tensor      = np.stack(all_strain_histories, axis=0)       # [batch, steps, 3] Kratos
stress_tensor      = np.stack(all_stress_histories, axis=0)       # [batch, steps, 3] Kratos
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
    plt.title(f"Batch {batch_idx+1}: Kratos vs Ours")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"data_set/batch_{batch_idx+1}_stress_strain_compare.png")
    plt.close()

with open("data_set/batch_log.txt", "w") as f:
    f.write(f"Total batches: {batch}\n")
    f.write("Batch info (theta, phi, strain):\n")
    for line in log_lines:
        f.write(line + "\n")

print("Plots stored in data_set/batch_*_stress_strain_compare.png")
print("Log stored in data_set/batch_log.txt")
