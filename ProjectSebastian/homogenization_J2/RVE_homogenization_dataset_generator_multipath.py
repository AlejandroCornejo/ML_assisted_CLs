import KratosMultiphysics as KM
import KratosMultiphysics.analysis_stage as analysis_stage
import importlib
from KratosMultiphysics.StructuralMechanicsApplication import \
    python_solvers_wrapper_structural as structural_solvers
import KratosMultiphysics.StructuralMechanicsApplication as SMApp  # noqa: F401
import numpy as np
import os
import matplotlib.pyplot as plt

"""
RVE_homogenization_dataset_generator with multipath (0 -> E1 -> E2) in strain space.

For each batch (theta, phi), we:
  - Build a base strain direction E1(theta, phi, max_stretch_factor)
  - Sample multipath parameters gamma and psi_deg
  - Build E2 with ||E2|| = gamma * ||E1|| and angle(E1, E2) = psi_deg
  - Run a two-segment path in strain space:
        Segment 1 (0 -> T/2):   E(t) = alpha(t) * E1
        Segment 2 (T/2 -> T):   E(t) = (1 - beta(t)) * E1 + beta(t) * E2
  - Use batch_strain(t) = E(t) / (time / end_time) so that
        u_x = (Ex * x + Exy * y) * (time / end_time)
        u_y = (Ey * y + Exy * x) * (time / end_time)
    reproduces exactly E(t).
"""


# ----------------------------------------------------------------------
# Helper: unified multipath builder E1, E2
# ----------------------------------------------------------------------
def _build_E1_E2(theta_deg, phi_deg, max_stretch_factor, gamma, psi_deg):
    """
    Build E1 and E2 for the multipath in strain space.
    """
    theta = np.radians(theta_deg)
    phi   = np.radians(phi_deg)

    # Original spherical-like mapping for E1
    E1 = max_stretch_factor * np.array([
        np.cos(phi),                      # Exx
        np.sin(theta) * np.cos(phi),      # Eyy
        np.sin(theta) * np.sin(phi),      # Exy
    ], dtype=float)

    r1 = np.linalg.norm(E1)
    if r1 < 1e-14:
        raise ValueError("E1 has near-zero norm; cannot define multipath.")

    e1  = E1 / r1
    psi = np.radians(psi_deg)

    # If psi ~ 0 or pi, just align E2 along E1 with different magnitude
    if abs(np.sin(psi)) < 1e-8:
        E2 = gamma * E1 * np.sign(np.cos(psi))
        return E1, E2

    # Build an orthonormal direction e_perp1 perpendicular to e1
    a = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(a, e1)) > 0.99:
        a = np.array([0.0, 1.0, 0.0])
    v = a - np.dot(a, e1) * e1
    v_norm = np.linalg.norm(v)
    if v_norm < 1e-14:
        raise RuntimeError("Failed to construct a perpendicular direction.")
    e_perp1 = v / v_norm

    # Direction for E2 with angle psi w.r.t E1
    e2_dir = np.cos(psi) * e1 + np.sin(psi) * e_perp1
    e2_dir /= np.linalg.norm(e2_dir)

    # Norm of E2 = gamma * norm(E1)
    E2 = gamma * r1 * e2_dir

    return E1, E2


# ----------------------------------------------------------------------
# AnalysisStage for Kratos RVE homogenization
# ----------------------------------------------------------------------
class RVE_homogenization_dataset_generator(analysis_stage.AnalysisStage):

    def __init__(self, model, project_parameters):
        super().__init__(model, project_parameters)
        self.batch_strain = np.zeros(3)

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
        super().ApplyBoundaryConditions()

        Ex, Ey, Exy = self.batch_strain

        for node in self._GetSolver().GetComputingModelPart().Nodes:
            x_coord = node.X0
            y_coord = node.Y0
            z_coord = node.Z0
            displ_x = (Ex * x_coord + Exy * y_coord) * self.time / self.end_time
            displ_y = (Ey * y_coord + Exy * x_coord) * self.time / self.end_time
            displ_z = 0.0

            if node.IsFixed(KM.DISPLACEMENT_X):
                node.SetSolutionStepValue(KM.DISPLACEMENT_X, displ_x)
            if node.IsFixed(KM.DISPLACEMENT_Y):
                node.SetSolutionStepValue(KM.DISPLACEMENT_Y, displ_y)
            if node.IsFixed(KM.DISPLACEMENT_Z):
                node.SetSolutionStepValue(KM.DISPLACEMENT_Z, displ_z)

    def CalculateHomogenizedStressAndStrain(self):
        process_info = self._GetSolver().GetComputingModelPart().ProcessInfo
        computing_model_part = self._GetSolver().GetComputingModelPart()

        for element in computing_model_part.Elements:
            dummy_strain = np.array(
                element.CalculateOnIntegrationPoints(KM.GREEN_LAGRANGE_STRAIN_VECTOR, process_info)
            )
            break
        n_ips = dummy_strain.shape[0]
        voigt_size = dummy_strain.shape[1]

        homogenized_stress = np.zeros(voigt_size)
        homogenized_strain = np.zeros(voigt_size)
        RVE_area = 0.0

        for element in computing_model_part.Elements:
            strain = element.CalculateOnIntegrationPoints(KM.GREEN_LAGRANGE_STRAIN_VECTOR, process_info)
            stress = element.CalculateOnIntegrationPoints(KM.PK2_STRESS_VECTOR, process_info)

            strain_array = np.array(strain)
            stress_array = np.array(stress)

            stress_ip_sum = np.sum(stress_array, axis=0)
            strain_ip_sum = np.sum(strain_array, axis=0)

            element_area = element.GetGeometry().Area()

            RVE_area += element_area
            homogenized_stress += element_area * stress_ip_sum / n_ips
            homogenized_strain += element_area * strain_ip_sum / n_ips

        homogenized_stress /= RVE_area
        homogenized_strain /= RVE_area

        return homogenized_strain, homogenized_stress


# =====================================================================
# MAIN
# =====================================================================

with open("ProjectParameters.json", 'r') as parameter_file:
    parameters = KM.Parameters(parameter_file.read())

analysis_stage_module_name = parameters["analysis_stage"].GetString()
analysis_stage_class_name = analysis_stage_module_name.split('.')[-1]
analysis_stage_class_name = ''.join(x.title() for x in analysis_stage_class_name.split('_'))

analysis_stage_module = importlib.import_module(analysis_stage_module_name)
analysis_stage_class = getattr(analysis_stage_module, analysis_stage_class_name)

end_time = parameters["problem_data"]["end_time"].GetDouble()

log_lines = []

theta = 0.0
phi   = 0.0
angle_increment    = 25.0
max_stretch_factor = 0.005

np.random.seed(1234)

all_strain_histories = []
all_stress_histories = []

batch = 0

while theta <= 25.0 + 1e-8:
    while phi <= 25.0 + 1e-8:
        batch += 1
        print(f"\n[INFO] Batch {batch}: theta={theta}, phi={phi}")

        gamma   = np.random.uniform(0.100, 1.000)
        psi_deg = np.random.uniform(0.0, 180.0)

        E1, E2 = _build_E1_E2(theta, phi, max_stretch_factor, gamma, psi_deg)

        print(f"       gamma = {gamma:.3f}, psi = {psi_deg:.1f} deg")
        print(f"       E1 = {E1}")
        print(f"       E2 = {E2}")

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

        strain_history = [np.zeros(3)]
        stress_history = [np.zeros(3)]

        simulation.Initialize()

        t_mid = 0.5 * end_time

        while simulation.KeepAdvancingSolutionLoop():
            simulation.time = simulation._AdvanceTime()
            t = simulation.time

            if t <= t_mid:
                alpha_seg = t / max(t_mid, 1e-14)
                E_t = alpha_seg * E1
                seg_id = 1
            else:
                beta_seg = (t - t_mid) / max(end_time - t_mid, 1e-14)
                E_t = (1.0 - beta_seg) * E1 + beta_seg * E2
                seg_id = 2

            lambda_t = t / max(end_time, 1e-14)
            if lambda_t < 1e-14:
                effective_batch_strain = np.zeros(3)
            else:
                effective_batch_strain = E_t / lambda_t

            simulation.batch_strain = effective_batch_strain

            print(f"  [Step t={t:.5f}, seg={seg_id}] "
                  f"E(t)={E_t}, batch_strain={simulation.batch_strain}")

            simulation.InitializeSolutionStep()
            simulation._GetSolver().Predict()
            is_converged = simulation._GetSolver().SolveSolutionStep()
            simulation.FinalizeSolutionStep()

            step_strain, step_stress = simulation.CalculateHomogenizedStressAndStrain()
            strain_history.append(step_strain)
            stress_history.append(step_stress)

        simulation.Finalize()

        all_strain_histories.append(np.stack(strain_history, axis=0))
        all_stress_histories.append(np.stack(stress_history, axis=0))

        phi += angle_increment

    phi = 0.0
    theta += angle_increment

strain_tensor = np.stack(all_strain_histories, axis=0)
stress_tensor = np.stack(all_stress_histories, axis=0)

os.makedirs("data_set", exist_ok=True)
np.savez("data_set/all_stress_histories.npz", stress=stress_tensor)
np.savez("data_set/all_strain_histories.npz", strain=strain_tensor)

print("Results stored in data_set/all_stress_histories.npz and data_set/all_strain_histories.npz")

for batch_idx in range(stress_tensor.shape[0]):
    Sxx = stress_tensor[batch_idx, :, 0]
    Syy = stress_tensor[batch_idx, :, 1]
    Sxy = stress_tensor[batch_idx, :, 2]
    Exx = strain_tensor[batch_idx, :, 0]
    Eyy = strain_tensor[batch_idx, :, 1]
    Exy = strain_tensor[batch_idx, :, 2]

    plt.plot(Exx, Sxx, marker='o', color='r', label="XX")
    plt.plot(Eyy, Syy, marker='o', color='b', label="YY")
    plt.plot(Exy, Sxy, marker='o', color='k', label="XY")
    plt.xlabel("Strain [-]")
    plt.ylabel("Stress [Pa]")
    plt.title(f"Batch {batch_idx+1}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"data_set/batch_{batch_idx+1}_stress_strain_plots.png")
    plt.close()

with open("data_set/batch_log.txt", "w") as f:
    f.write(f"Total batches: {batch}\n")
    f.write("Batch info (theta, phi, gamma, psi_deg, E1, E2):\n")
    for line in log_lines:
        f.write(line + "\n")
