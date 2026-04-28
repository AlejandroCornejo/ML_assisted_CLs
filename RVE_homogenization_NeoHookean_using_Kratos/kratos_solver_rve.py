#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import json
import hashlib
import numpy as np
import matplotlib.pyplot as plt

import KratosMultiphysics as KM
import KratosMultiphysics.analysis_stage as analysis_stage
from KratosMultiphysics.StructuralMechanicsApplication import (
    python_solvers_wrapper_structural as structural_solvers,
)
import KratosMultiphysics.StructuralMechanicsApplication as SMApp  # noqa: F401
import KratosMultiphysics.ConstitutiveLawsApplication as CLA  # noqa: F401


DEFAULT_OUTPUT_DIR = "training_set_kratos"
DEFAULT_YOUNG_MPA = 1628.0
DEFAULT_POISSON = 0.4
PAPER_YOUNG_MPA = 70000.0
PAPER_POISSON = 0.29

# 0 -> Exx, 1 -> Eyy, 2 -> Exy
UNIAXIAL_COMPONENT = 0

# Dynamic path-time discretization (same policy as fom_solver_rve.py)
REFERENCE_STEPS_FOR_UNIT_AMPLITUDE = 400
MIN_STEPS_PER_SEGMENT = 1


def GreenLagrangeVoigtToTensor2D(strain_voigt):
    exx = float(strain_voigt[0])
    eyy = float(strain_voigt[1])
    gxy = float(strain_voigt[2])  # engineering shear = 2 * E12
    e12 = 0.5 * gxy
    return np.array([[exx, e12], [e12, eyy]], dtype=float)


def DeformationGradientFromGreenLagrange2D(strain_voigt):
    e_tensor = GreenLagrangeVoigtToTensor2D(strain_voigt)
    c_tensor = 2.0 * e_tensor + np.eye(2, dtype=float)  # C = F^T F

    eigvals, eigvecs = np.linalg.eigh(c_tensor)
    if np.min(eigvals) <= 0.0:
        raise RuntimeError(
            "Invalid Green-Lagrange strain state: C=2E+I is not positive definite."
        )

    f_tensor = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
    return f_tensor


class RVEKratosHomogenizationStage(analysis_stage.AnalysisStage):
    def __init__(self, model, project_parameters):
        super().__init__(model, project_parameters)
        self.batch_strain = np.zeros(3, dtype=float)

        self._center_initialized = False
        self._x0c = 0.0
        self._y0c = 0.0

    def _CreateSolver(self):
        return structural_solvers.CreateSolver(self.model, self.project_parameters)

    def __CreateListOfProcesses(self):
        order = self._GetOrderOfProcessesInitialization()
        self._list_of_processes = self._CreateProcesses("processes", order)
        dep = self._CheckDeprecatedOutputProcesses(self._list_of_processes)

        order_out = self._GetOrderOfOutputProcessesInitialization()
        self._list_of_output_processes = self._CreateProcesses("output_processes", order_out)

        self._list_of_processes.extend(self._list_of_output_processes)
        self._list_of_output_processes.extend(dep)

    def _InitializeDomainCenterIfNeeded(self, mp):
        if self._center_initialized:
            return
        xs = [node.X0 for node in mp.Nodes]
        ys = [node.Y0 for node in mp.Nodes]
        if len(xs) == 0:
            self._x0c = 0.0
            self._y0c = 0.0
        else:
            xmin, xmax = float(min(xs)), float(max(xs))
            ymin, ymax = float(min(ys)), float(max(ys))
            self._x0c = 0.5 * (xmin + xmax)
            self._y0c = 0.5 * (ymin + ymax)
        self._center_initialized = True

    def ApplyBoundaryConditions(self):
        super().ApplyBoundaryConditions()

        f_tensor = DeformationGradientFromGreenLagrange2D(self.batch_strain)
        mp = self._GetSolver().GetComputingModelPart()

        self._InitializeDomainCenterIfNeeded(mp)
        x0c, y0c = self._x0c, self._y0c

        for node in mp.Nodes:
            xr = node.X0 - x0c
            yr = node.Y0 - y0c

            # Affine macro displacement: u_macro = (F - I) X
            ux = (f_tensor[0, 0] - 1.0) * xr + f_tensor[0, 1] * yr
            uy = f_tensor[1, 0] * xr + (f_tensor[1, 1] - 1.0) * yr

            if node.IsFixed(KM.DISPLACEMENT_X):
                node.SetSolutionStepValue(KM.DISPLACEMENT_X, ux)
            if node.IsFixed(KM.DISPLACEMENT_Y):
                node.SetSolutionStepValue(KM.DISPLACEMENT_Y, uy)
            if node.IsFixed(KM.DISPLACEMENT_Z):
                node.SetSolutionStepValue(KM.DISPLACEMENT_Z, 0.0)


def BuildDynamicSegmentSteps(
    path_points,
    reference_steps=REFERENCE_STEPS_FOR_UNIT_AMPLITUDE,
    min_steps=MIN_STEPS_PER_SEGMENT,
    reference_amplitude=None,
):
    p = np.asarray(path_points, dtype=float)
    if p.ndim == 1:
        p = p.reshape(-1, 1)
    if p.ndim != 2 or p.shape[0] < 2:
        raise ValueError("path_points must have shape (n_waypoints, n_components), with n_waypoints >= 2.")

    dp = np.diff(p, axis=0)
    # Consistency with Stage 0: use Exy = 0.5 * Gxy for segment length calculation
    dp_scaled = dp.copy()
    if dp_scaled.shape[1] == 3:
        dp_scaled[:, 2] *= 0.5

    seg_len = np.linalg.norm(dp_scaled, axis=1)
    if reference_amplitude is None:
        # Use max norm of the scaled points for ref_amp if not provided
        p_scaled = p.copy()
        if p_scaled.shape[1] == 3:
            p_scaled[:, 2] *= 0.5
        ref_amp = float(np.max(np.linalg.norm(p_scaled, axis=1)))
        if ref_amp <= 0.0:
            ref_amp = float(np.max(seg_len))
    else:
        ref_amp = float(abs(reference_amplitude))
    if (not np.isfinite(ref_amp)) or ref_amp <= 0.0:
        ref_amp = 1.0

    seg_steps = np.maximum(
        int(min_steps),
        np.ceil(float(reference_steps) * seg_len / ref_amp).astype(int),
    )
    return seg_steps, ref_amp, seg_len


def ParseStrainWaypoints(path_string):
    rows = []
    for chunk in str(path_string).split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        vals = [float(x.strip()) for x in chunk.split(",")]
        if len(vals) != 3:
            raise ValueError("Each waypoint must have 3 comma-separated values: Ex,Ey,Gxy")
        rows.append(vals)
    arr = np.array(rows, dtype=float)
    if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] != 3:
        raise ValueError("strain-waypoints must define at least 2 waypoints with 3 components each.")
    return arr


def _ValidateStrainPathArray(arr, source):
    p = np.array(arr, dtype=float)
    if p.ndim != 2 or p.shape[0] < 2 or p.shape[1] != 3:
        raise ValueError(f"{source} must contain a (n_waypoints,3) array with n_waypoints >= 2.")
    return p


def LoadStrainWaypointsFromFile(path, trajectory_index=1):
    if int(trajectory_index) < 1:
        raise ValueError("trajectory-index must be >= 1.")

    fpath = str(path)
    if not os.path.exists(fpath):
        raise FileNotFoundError(f"strain-waypoints-file not found: {fpath}")

    ext = os.path.splitext(fpath)[1].lower()
    meta = {}

    if ext == ".npy":
        arr = np.load(fpath)
        traj = _ValidateStrainPathArray(arr, "npy file")
        return traj, meta

    if ext == ".npz":
        data = np.load(fpath)
        idx = int(trajectory_index)
        key_main = f"trajectory_{idx}"
        if key_main in data:
            arr = data[key_main]
        elif "trajectory" in data:
            arr = data["trajectory"]
        else:
            available = ", ".join(list(data.keys()))
            raise ValueError(
                f"npz file does not contain '{key_main}' or 'trajectory'. Available keys: {available}"
            )

        if "ref_steps" in data:
            meta["ref_steps"] = int(np.ravel(data["ref_steps"])[0])
        if "reference_amplitude" in data:
            meta["reference_amplitude"] = float(np.ravel(data["reference_amplitude"])[0])

        traj = _ValidateStrainPathArray(arr, "npz trajectory")
        return traj, meta

    if ext == ".json":
        with open(fpath, "r") as f:
            obj = json.load(f)

        if isinstance(obj, dict):
            idx = int(trajectory_index)
            key_main = f"trajectory_{idx}"
            if key_main in obj:
                arr = obj[key_main]
            elif "trajectory" in obj:
                arr = obj["trajectory"]
            else:
                raise ValueError("json file must contain 'trajectory' or 'trajectory_<index>'.")

            if "ref_steps" in obj:
                meta["ref_steps"] = int(obj["ref_steps"])
            if "reference_amplitude" in obj:
                meta["reference_amplitude"] = float(obj["reference_amplitude"])
        else:
            arr = obj

        traj = _ValidateStrainPathArray(arr, "json trajectory")
        return traj, meta

    raise ValueError(
        f"Unsupported strain-waypoints-file extension: '{ext}'. Use .npz, .npy or .json"
    )


def GetPaperTrajectoryWaypoints():
    # Same trajectory structure reported in Bravo et al. (2024), interpreted as [Ex, Ey, Gxy].
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [-0.2, 0.0, 0.0],
            [0.0, 0.0, 0.2],
            [0.0, 0.0, -0.2],
            [2.0, 2.0, 0.2],
        ],
        dtype=float,
    )


def StripMdpaExtension(mesh_name):
    s = str(mesh_name)
    if s.endswith(".mdpa"):
        return s[:-5]
    return s


def SetInputMeshFilename(parameters, mesh_name):
    mesh_base = StripMdpaExtension(mesh_name)
    parameters["modelers"][0]["parameters"]["input_filename"].SetString(mesh_base)
    parameters["problem_data"]["problem_name"].SetString(mesh_base)


def FindSubModelPartNamesInMdpa(mdpa_path):
    names = []
    with open(mdpa_path, "r") as f:
        for line in f:
            s = line.strip()
            if s.startswith("Begin SubModelPart "):
                parts = s.split()
                if len(parts) >= 3:
                    names.append(parts[2])
    return names


def _MaterialPartSortKey(name):
    if name == "material":
        return (0, 0, name)
    if name.startswith("material_"):
        suffix = name.split("_", 1)[1]
        if suffix.isdigit():
            return (1, int(suffix), name)
    return (2, 0, name)


def DetectMaterialSubModelParts(mdpa_path):
    names = FindSubModelPartNamesInMdpa(mdpa_path)
    mats = sorted([n for n in names if n.startswith("material")], key=_MaterialPartSortKey)
    if not mats:
        raise RuntimeError(
            f"No material* submodelparts found in mdpa: {mdpa_path}. "
            f"Expected names like material, material_1, material_2."
        )
    return mats


def ConfigureElementModelerForMaterialParts(parameters, material_parts):
    params_dict = json.loads(parameters.PrettyPrintJsonString())
    old_list = params_dict["modelers"][1]["parameters"]["elements_list"]
    element_name = old_list[0]["element_name"] if len(old_list) > 0 else "TotalLagrangianElement2D6N"

    params_dict["modelers"][1]["parameters"]["elements_list"] = [
        {
            "model_part_name": f"Structure.{name}",
            "element_name": element_name,
        }
        for name in material_parts
    ]
    return KM.Parameters(json.dumps(params_dict))


def WriteRuntimeMaterialsFile(
    material_parts,
    young_mpa,
    poisson,
    out_filename="StructuralMaterials.runtime.json",
    density=7850.0,
    thickness=0.05,
):
    young_pa = float(young_mpa) * 1.0e6
    props = []
    for i, name in enumerate(material_parts, start=1):
        props.append(
            {
                "model_part_name": f"Structure.{name}",
                "properties_id": i,
                "Material": {
                    "constitutive_law": {"name": "HyperElasticPlaneStrain2DLaw"},
                    "Variables": {
                        "DENSITY": float(density),
                        "YOUNG_MODULUS": float(young_pa),
                        "POISSON_RATIO": float(poisson),
                        "THICKNESS": float(thickness),
                    },
                    "Tables": None,
                },
            }
        )

    with open(out_filename, "w") as f:
        json.dump({"properties": props}, f, indent=4)

    return out_filename


def SanitizeDirichletGeometriesInMdpa(mdpa_path):
    with open(mdpa_path, "r") as f:
        lines = f.readlines()

    out = []
    in_dirichlet = False
    skip_geometries = False
    changed = False

    for line in lines:
        s = line.strip()

        if s.startswith("Begin SubModelPart "):
            parts = s.split()
            if len(parts) >= 3 and parts[2] == "dirichlet":
                in_dirichlet = True

        if in_dirichlet and s == "Begin SubModelPartGeometries":
            skip_geometries = True
            changed = True
            continue

        if skip_geometries and s == "End SubModelPartGeometries":
            skip_geometries = False
            continue

        if not skip_geometries:
            out.append(line)

        if in_dirichlet and s == "End SubModelPart":
            in_dirichlet = False

    if changed:
        with open(mdpa_path, "w") as f:
            f.writelines(out)

    return changed


def SetUpDofEquationIds(mp):
    dofs = KM.DofsArrayType()
    KM.DofArrayUtilities.SetUpDofArray(mp, dofs)
    KM.DofArrayUtilities.SetDofEquationIds(dofs)

    eq_map = np.empty((mp.NumberOfNodes(), 2), dtype=int)
    for i, node in enumerate(mp.Nodes):
        eq_map[i, 0] = node.GetDof(KM.DISPLACEMENT_X).EquationId
        eq_map[i, 1] = node.GetDof(KM.DISPLACEMENT_Y).EquationId

    return len(dofs), eq_map


def ExtractEquationDisplacements(mp, n_dof, eq_map):
    u_eq = np.zeros(n_dof, dtype=float)
    for i, node in enumerate(mp.Nodes):
        u_eq[eq_map[i, 0]] = node.GetSolutionStepValue(KM.DISPLACEMENT_X, 0)
        u_eq[eq_map[i, 1]] = node.GetSolutionStepValue(KM.DISPLACEMENT_Y, 0)
    return u_eq


def CalculateHomogenizedStressAndStrain(mp):
    process_info = mp.ProcessInfo

    first = next(iter(mp.Elements), None)
    if first is None:
        return np.zeros(3, dtype=float), np.zeros(3, dtype=float)

    dummy = np.array(
        first.CalculateOnIntegrationPoints(KM.GREEN_LAGRANGE_STRAIN_VECTOR, process_info),
        dtype=float,
    )
    n_ip, voigt = dummy.shape

    eps_h = np.zeros(voigt, dtype=float)
    sig_h = np.zeros(voigt, dtype=float)
    area = 0.0

    for elem in mp.Elements:
        eps = np.array(
            elem.CalculateOnIntegrationPoints(KM.GREEN_LAGRANGE_STRAIN_VECTOR, process_info),
            dtype=float,
        )
        sig = np.array(
            elem.CalculateOnIntegrationPoints(KM.PK2_STRESS_VECTOR, process_info),
            dtype=float,
        )

        A = float(elem.GetGeometry().Area())
        area += A
        eps_h += A * np.sum(eps, axis=0) / float(n_ip)
        sig_h += A * np.sum(sig, axis=0) / float(n_ip)

    if area <= 0.0:
        return eps_h, sig_h
    return eps_h / area, sig_h / area


def UpdateCurrentCoordinatesFromDisplacement(mp, step=0):
    KM.VariableUtils().UpdateCurrentPosition(mp.Nodes, KM.DISPLACEMENT, int(step))


def ComputeEquivalentStressStrain2D(strain_hist, stress_hist):
    Exx = strain_hist[:, 0]
    Eyy = strain_hist[:, 1]
    Gxy = strain_hist[:, 2]
    E12 = 0.5 * Gxy  # Kratos strain vector uses engineering shear in Voigt form.

    Emean = 0.5 * (Exx + Eyy)
    Edev_xx = Exx - Emean
    Edev_yy = Eyy - Emean
    Edev_xy = E12
    Edev_norm2 = Edev_xx * Edev_xx + Edev_yy * Edev_yy + 2.0 * Edev_xy * Edev_xy
    eps_eq = np.sqrt(np.maximum((2.0 / 3.0) * Edev_norm2, 0.0))

    Sxx = stress_hist[:, 0]
    Syy = stress_hist[:, 1]
    Sxy = stress_hist[:, 2]
    Smean = 0.5 * (Sxx + Syy)
    Sdev_xx = Sxx - Smean
    Sdev_yy = Syy - Smean
    Sdev_xy = Sxy
    Sdev_norm2 = Sdev_xx * Sdev_xx + Sdev_yy * Sdev_yy + 2.0 * Sdev_xy * Sdev_xy
    sig_eq = np.sqrt(np.maximum((3.0 / 2.0) * Sdev_norm2, 0.0))

    return eps_eq, sig_eq


def RunKratosBatchSimulation(
    parameters,
    out_dir=DEFAULT_OUTPUT_DIR,
    save_plot=True,
    save_data=True,
    theta1=0.0,
    phi1=0.0,
    theta2=0.0,
    phi2=0.0,
    alpha1=0.01,
    alpha2=0.01,
    alpha_path=None,
    strain_path=None,
    trajectory_index=None,
    reference_amplitude=None,
    reference_steps=REFERENCE_STEPS_FOR_UNIT_AMPLITUDE,
):
    os.makedirs(out_dir, exist_ok=True)

    dt = parameters["solver_settings"]["time_stepping"]["time_step"].GetDouble()

    if strain_path is None:
        if alpha_path is None:
            alpha_path = [0.0, float(alpha1), float(alpha2)]
        else:
            alpha_path = [float(a) for a in alpha_path]
            if len(alpha_path) < 2:
                raise ValueError("alpha_path must have at least 2 waypoints.")

        n_wp = len(alpha_path)
        E_wp = np.zeros((n_wp, 3), dtype=float)
        for i, a in enumerate(alpha_path):
            E_wp[i, UNIAXIAL_COMPONENT] = a
    else:
        E_wp = np.array(strain_path, dtype=float)
        if E_wp.ndim != 2 or E_wp.shape[0] < 2 or E_wp.shape[1] != 3:
            raise ValueError("strain_path must have shape (n_waypoints, 3) with n_waypoints >= 2.")

    n_wp = E_wp.shape[0]
    n_seg = n_wp - 1

    seg_steps, ref_amp_used, seg_len = BuildDynamicSegmentSteps(
        E_wp,
        reference_steps=reference_steps,
        min_steps=MIN_STEPS_PER_SEGMENT,
        reference_amplitude=reference_amplitude,
    )
    step_offsets = np.concatenate(([0], np.cumsum(seg_steps)))
    n_steps_total = int(step_offsets[-1])
    end_time = dt * float(n_steps_total)
    if n_steps_total <= 0:
        raise RuntimeError("Dynamic step allocation produced zero total steps.")

    # Clone and adapt per run (avoid writing over the other solver's GiD output path).
    params = parameters.Clone()
    params["problem_data"]["end_time"].SetDouble(end_time)
    if params["output_processes"].Has("gid_output"):
        gid_list = params["output_processes"]["gid_output"]
        if gid_list.size() > 0:
            gid_list[0]["Parameters"]["output_name"].SetString("gid_output/rve_geometry_kratos")

    E1 = np.zeros(3, dtype=float)
    E2 = np.zeros(3, dtype=float)
    if n_wp >= 2:
        E1[:] = E_wp[1, :]
    if n_wp >= 3:
        E2[:] = E_wp[2, :]
    elif n_wp >= 2:
        E2[:] = E_wp[1, :]
    log_line = (
        f"theta1={theta1:.2f}, phi1={phi1:.2f}, alpha1={alpha1:.6f}, "
        f"theta2={theta2:.2f}, phi2={phi2:.2f}, alpha2={alpha2:.6f}, "
        f"E1={E1.tolist()}, E2={E2.tolist()}, E_path={E_wp.tolist()}"
    )

    print(
        f"[KRATOS] dynamic path timing: ref_steps={int(reference_steps)}, ref_amp={ref_amp_used:.6e}, "
        f"seg_len={seg_len.tolist()}, seg_steps={seg_steps.tolist()}, "
        f"total_steps={n_steps_total}, end_time={end_time:.6e}"
    )

    model = KM.Model()
    sim = RVEKratosHomogenizationStage(model, params)
    sim.Initialize()
    mp = sim._GetSolver().GetComputingModelPart()

    n_dof, eq_map = SetUpDofEquationIds(mp)

    strain_hist = [np.zeros(3, dtype=float)]
    stress_hist = [np.zeros(3, dtype=float)]
    U_hist = [ExtractEquationDisplacements(mp, n_dof, eq_map)]

    for step in range(1, n_steps_total + 1):
        time_val = float(step) * float(dt)

        mp.CloneTimeStep(time_val)
        mp.ProcessInfo[KM.DELTA_TIME] = dt
        mp.ProcessInfo[KM.TIME] = time_val
        mp.ProcessInfo[KM.STEP] = step

        sim.time = time_val
        sim.step = step
        sim.end_time = end_time

        s = int(np.searchsorted(step_offsets, step, side="left") - 1)
        if s < 0:
            s = 0
        if s >= n_seg:
            s = n_seg - 1
        local_step = int(step - step_offsets[s])  # 1..seg_steps[s]
        xi = float(local_step) / float(max(int(seg_steps[s]), 1))
        E_t = (1.0 - xi) * E_wp[s, :] + xi * E_wp[s + 1, :]

        sim.batch_strain = E_t.copy()

        sim.InitializeSolutionStep()
        sim.ApplyBoundaryConditions()
        sim._GetSolver().Predict()
        is_converged = sim._GetSolver().SolveSolutionStep()
        if is_converged is False:
            raise RuntimeError(f"Kratos solver did not converge at step {step}.")

        # Keep current coordinates synchronized with DISPLACEMENT before GP queries.
        UpdateCurrentCoordinatesFromDisplacement(mp, step=0)
        eps_h, sig_h = CalculateHomogenizedStressAndStrain(mp)
        strain_hist.append(eps_h)
        stress_hist.append(sig_h)
        U_hist.append(ExtractEquationDisplacements(mp, n_dof, eq_map))

        sim.FinalizeSolutionStep()
        if save_data or save_plot:
            sim.OutputSolutionStep()

    sim.Finalize()

    strain_hist = np.stack(strain_hist, axis=0)
    stress_hist = np.stack(stress_hist, axis=0)
    U_hist = np.stack(U_hist, axis=0)

    # Simplified naming convention
    if trajectory_index:
        tag = f"trajectory_{trajectory_index}"
    else:
        wp_hash = hashlib.sha1(np.round(E_wp, 10).tobytes()).hexdigest()[:8]
        tag = f"run_{wp_hash}"

    if save_data:
        np.save(os.path.join(out_dir, f"strain_{tag}.npy"), strain_hist)
        np.save(os.path.join(out_dir, f"stress_{tag}.npy"), stress_hist)
        np.save(os.path.join(out_dir, f"U_{tag}.npy"), U_hist)

    if save_plot:
        Exx, Eyy, Exy = strain_hist[:, 0], strain_hist[:, 1], strain_hist[:, 2]
        Sxx, Syy, Sxy = stress_hist[:, 0], stress_hist[:, 1], stress_hist[:, 2]

        plt.figure()
        plt.plot(Exx, Sxx, marker="o", color="r", label="sigma_xx")
        plt.plot(Eyy, Syy, marker="o", color="b", label="sigma_yy")
        plt.plot(Exy, Sxy, marker="o", color="k", label="sigma_xy")
        plt.xlabel("Strain [-]")
        plt.ylabel("Stress [Pa]")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"stress_strain_{tag}.png"))
        plt.close()

        eps_eq, sig_eq = ComputeEquivalentStressStrain2D(strain_hist, stress_hist)
        plt.figure()
        plt.plot(eps_eq, sig_eq, marker="o", color="g")
        plt.xlabel("Equivalent Green-Lagrange strain [-]")
        plt.ylabel("Equivalent PK2 stress [Pa]")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"stress_strain_equivalent_{tag}.png"))
        plt.close()

    return strain_hist, stress_hist, U_hist, log_line


def BuildDefaultTrainCases(amax):
    a25 = 0.25 * amax
    a50 = 0.50 * amax
    a75 = 0.75 * amax
    a100 = 1.00 * amax
    return [
        ("mono_pos_0p25_0p5_0p75_1p0", [0.0, +a25, +a50, +a75, +a100]),
        ("mono_neg_0p25_0p5_0p75_1p0", [0.0, -a25, -a50, -a75, -a100]),
    ]


def RunBatchGeneration(
    out_dir=DEFAULT_OUTPUT_DIR,
    amax=0.10,
    parameters=None,
    reference_steps=REFERENCE_STEPS_FOR_UNIT_AMPLITUDE,
):
    os.makedirs(out_dir, exist_ok=True)
    if parameters is None:
        with open("ProjectParameters.json", "r") as f:
            parameters = KM.Parameters(f.read())

    train_cases = BuildDefaultTrainCases(float(amax))

    log_lines = []
    for k, (name, alpha_path) in enumerate(train_cases, start=1):
        theta1 = phi1 = theta2 = phi2 = 0.0
        alpha1 = alpha_path[1] if len(alpha_path) > 1 else 0.0
        alpha2 = alpha_path[2] if len(alpha_path) > 2 else alpha1

        print(f"\n[INFO] KRATOS Batch {k}/{len(train_cases)}: {name} | alpha_path={alpha_path}")
        _, _, _, log_line = RunKratosBatchSimulation(
            parameters=parameters,
            out_dir=out_dir,
            save_plot=True,
            save_data=True,
            theta1=theta1,
            phi1=phi1,
            theta2=theta2,
            phi2=phi2,
            alpha1=alpha1,
            alpha2=alpha2,
            alpha_path=alpha_path,
            reference_amplitude=amax,
            reference_steps=reference_steps,
        )
        log_lines.append(f"Batch {k} ({name}): {log_line}")

    with open(os.path.join(out_dir, "batch_log.txt"), "w") as f:
        f.write(f"Total batches: {len(train_cases)}\n")
        for line in log_lines:
            f.write(line + "\n")


def _TryAmax(parameters, amax, reference_amplitude_for_step_control, reference_steps):
    train_cases = BuildDefaultTrainCases(float(amax))
    for name, alpha_path in train_cases:
        print(f"[SEARCH] trying {name} with amax={amax:.6f}")
        theta1 = phi1 = theta2 = phi2 = 0.0
        alpha1 = alpha_path[1] if len(alpha_path) > 1 else 0.0
        alpha2 = alpha_path[2] if len(alpha_path) > 2 else alpha1

        RunKratosBatchSimulation(
            parameters=parameters,
            out_dir=DEFAULT_OUTPUT_DIR,
            save_plot=False,
            save_data=False,
            theta1=theta1,
            phi1=phi1,
            theta2=theta2,
            phi2=phi2,
            alpha1=alpha1,
            alpha2=alpha2,
            alpha_path=alpha_path,
            reference_amplitude=reference_amplitude_for_step_control,
            reference_steps=reference_steps,
        )


def FindBestStableAmax(
    parameters,
    amax_min=0.10,
    amax_max=1.00,
    rel_tol=0.05,
    reference_amplitude_for_step_control=0.10,
    reference_steps=REFERENCE_STEPS_FOR_UNIT_AMPLITUDE,
    max_iter=20,
):
    low = float(amax_min)
    high = float(amax_max)

    try:
        _TryAmax(parameters, low, reference_amplitude_for_step_control, reference_steps)
    except Exception as exc:
        raise RuntimeError(
            f"Lower bound amax_min={low:.6f} is already unstable. "
            f"Decrease amax_min. Original error: {exc}"
        ) from exc

    try:
        _TryAmax(parameters, high, reference_amplitude_for_step_control, reference_steps)
        print(f"[SEARCH] amax_max={high:.6f} is stable. Returning upper bound.")
        return high
    except Exception:
        pass

    for it in range(1, int(max_iter) + 1):
        mid = 0.5 * (low + high)
        print(f"[SEARCH] iter={it:02d} | low={low:.6f} high={high:.6f} mid={mid:.6f}")
        try:
            _TryAmax(parameters, mid, reference_amplitude_for_step_control, reference_steps)
            low = mid
        except Exception as exc:
            print(f"[SEARCH] mid={mid:.6f} failed: {exc}")
            high = mid

        if (high - low) / max(low, 1e-12) <= float(rel_tol):
            break

    return low


def RunSingleStrainTrajectory(
    parameters,
    strain_path,
    out_dir=DEFAULT_OUTPUT_DIR,
    trajectory_index=None,
    reference_steps=REFERENCE_STEPS_FOR_UNIT_AMPLITUDE,
    reference_amplitude=0.10,
):
    theta1 = phi1 = theta2 = phi2 = 0.0
    alpha1 = float(strain_path[1, 0]) if strain_path.shape[0] > 1 else 0.0
    alpha2 = float(strain_path[2, 0]) if strain_path.shape[0] > 2 else alpha1

    _, _, _, log_line = RunKratosBatchSimulation(
        parameters=parameters,
        out_dir=out_dir,
        save_plot=True,
        save_data=True,
        theta1=theta1,
        phi1=phi1,
        theta2=theta2,
        phi2=phi2,
        alpha1=alpha1,
        alpha2=alpha2,
        strain_path=strain_path,
        trajectory_index=trajectory_index,
        reference_amplitude=reference_amplitude,
        reference_steps=reference_steps,
    )

    with open(os.path.join(out_dir, "batch_log.txt"), "w") as f:
        f.write("Total batches: 1\n")
        f.write(f"Batch 1 (single_trajectory): {log_line}\n")


def ParseArgs():
    parser = argparse.ArgumentParser(description="Kratos RVE NeoHookean runner")
    parser.add_argument("--mesh", type=str, default=None, help="Mesh base name (with or without .mdpa).")
    parser.add_argument(
        "--sanitize-mdpa",
        action="store_true",
        help="Remove dirichlet SubModelPartGeometries block from the selected .mdpa before running.",
    )
    parser.add_argument(
        "--ref-steps",
        type=int,
        default=None,
        help="Reference number of steps for a path length equal to --reference-amplitude.",
    )
    parser.add_argument(
        "--reference-amplitude",
        type=float,
        default=None,
        help="Reference path-length amplitude used in dynamic step scaling.",
    )
    parser.add_argument(
        "--strain-waypoints",
        type=str,
        default=None,
        help='Single trajectory as "Ex,Ey,Gxy; Ex,Ey,Gxy; ...". Runs one case instead of default mono +/- batch.',
    )
    parser.add_argument(
        "--strain-waypoints-file",
        type=str,
        default=None,
        help="Path to trajectory file (.npz/.npy/.json). For .npz use keys trajectory_1/trajectory_2.",
    )
    parser.add_argument(
        "--trajectory-index",
        type=int,
        default=1,
        help="Trajectory index for --strain-waypoints-file (e.g. 1 or 2).",
    )
    parser.add_argument(
        "--use-paper-trajectory",
        action="store_true",
        help="Use the 7-point trajectory from Bravo et al. (2024) as a single run.",
    )
    parser.add_argument(
        "--young-mpa",
        type=float,
        default=None,
        help="Young modulus in MPa for the runtime materials file (auto: paper value if --use-paper-trajectory, otherwise default value).",
    )
    parser.add_argument(
        "--poisson",
        type=float,
        default=None,
        help="Poisson ratio for the runtime materials file (auto: paper value if --use-paper-trajectory, otherwise default value).",
    )
    parser.add_argument("--find-amax", action="store_true", help="Find largest stable amax instead of generating dataset.")
    parser.add_argument("--amax", type=float, default=0.10, help="amax for regular dataset generation.")
    parser.add_argument("--amax-min", type=float, default=0.10, help="Lower bound for amax search.")
    parser.add_argument("--amax-max", type=float, default=1.00, help="Upper bound for amax search.")
    parser.add_argument("--amax-rel-tol", type=float, default=0.05, help="Relative tolerance for amax bisection.")
    parser.add_argument(
        "--search-ref-amplitude",
        type=float,
        default=0.10,
        help="Reference amplitude used in dynamic step allocation during search.",
    )
    parser.add_argument("--out-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Output directory for dataset generation.")
    return parser.parse_args()


if __name__ == "__main__":
    args = ParseArgs()
    with open("ProjectParameters.json", "r") as f:
        parameters = KM.Parameters(f.read())

    if args.mesh is not None:
        SetInputMeshFilename(parameters, args.mesh)

    mesh_base = parameters["modelers"][0]["parameters"]["input_filename"].GetString()
    if args.sanitize_mdpa:
        mdpa_path = f"{StripMdpaExtension(mesh_base)}.mdpa"
        changed = SanitizeDirichletGeometriesInMdpa(mdpa_path)
        if changed:
            print(f"[INFO] Sanitized dirichlet geometry block in: {mdpa_path}")
        else:
            print(f"[INFO] No dirichlet geometry block found in: {mdpa_path}")

    n_single_flags = int(args.strain_waypoints is not None) + int(args.strain_waypoints_file is not None) + int(args.use_paper_trajectory)
    if n_single_flags > 1:
        raise ValueError(
            "Use only one single-trajectory source: --strain-waypoints OR --strain-waypoints-file OR --use-paper-trajectory."
        )

    single_traj = None
    single_meta = {}
    if args.strain_waypoints is not None:
        single_traj = ParseStrainWaypoints(args.strain_waypoints)
    elif args.strain_waypoints_file is not None:
        single_traj, single_meta = LoadStrainWaypointsFromFile(
            args.strain_waypoints_file,
            trajectory_index=args.trajectory_index,
        )
        print(
            "[INFO] Loaded trajectory file: "
            f"{args.strain_waypoints_file} (index={args.trajectory_index}) "
            f"with {single_traj.shape[0]} waypoints."
        )
    elif args.use_paper_trajectory:
        single_traj = GetPaperTrajectoryWaypoints()

    ref_steps_eff = args.ref_steps
    if ref_steps_eff is None:
        ref_steps_eff = int(single_meta.get("ref_steps", REFERENCE_STEPS_FOR_UNIT_AMPLITUDE))
    ref_steps_eff = int(ref_steps_eff)

    mdpa_path = f"{StripMdpaExtension(mesh_base)}.mdpa"
    material_parts = DetectMaterialSubModelParts(mdpa_path)
    parameters = ConfigureElementModelerForMaterialParts(parameters, material_parts)

    young_mpa = args.young_mpa
    if young_mpa is None:
        young_mpa = PAPER_YOUNG_MPA if args.use_paper_trajectory else DEFAULT_YOUNG_MPA
    poisson = args.poisson
    if poisson is None:
        poisson = PAPER_POISSON if args.use_paper_trajectory else DEFAULT_POISSON

    runtime_materials = WriteRuntimeMaterialsFile(
        material_parts=material_parts,
        young_mpa=young_mpa,
        poisson=poisson,
    )
    parameters["solver_settings"]["material_import_settings"]["materials_filename"].SetString(runtime_materials)
    print(
        "[INFO] Material setup: "
        f"parts={material_parts}, E={young_mpa:.3f} MPa, nu={poisson:.4f}, file={runtime_materials}"
    )

    if args.find_amax:
        best_amax = FindBestStableAmax(
            parameters=parameters,
            amax_min=args.amax_min,
            amax_max=args.amax_max,
            rel_tol=args.amax_rel_tol,
            reference_amplitude_for_step_control=args.search_ref_amplitude,
            reference_steps=ref_steps_eff,
        )
        print(f"\n[SEARCH] Best stable amax found: {best_amax:.6f}")
    elif single_traj is not None:
        ref_amp = args.reference_amplitude
        if ref_amp is None:
            ref_amp = float(single_meta.get("reference_amplitude", 0.10))
        RunSingleStrainTrajectory(
            parameters=parameters,
            strain_path=single_traj,
            out_dir=args.out_dir,
            trajectory_index=args.trajectory_index,
            reference_steps=ref_steps_eff,
            reference_amplitude=ref_amp,
        )
    else:
        RunBatchGeneration(
            out_dir=args.out_dir,
            amax=args.amax,
            parameters=parameters,
            reference_steps=ref_steps_eff,
        )
