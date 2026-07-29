#!/usr/bin/env python3
"""Verify the effective D4 (square) material symmetry of the RVE FOM.

The RVE input convention is the Green--Lagrange strain vector
``[E11, E22, gamma12]``, where ``gamma12 = 2 E12``.  A material symmetry
operation R acts on it through ``E' = R.T @ E @ R``.  Therefore no change to
the existing strain-driven FOM interface is required.

The test runs three mixed finite-strain states and the four distinct actions
of D4 on a symmetric 2-D tensor: identity, reflection, 90-degree rotation,
and diagonal reflection.  The remaining four elements of D4 differ only by
R -> -R and act identically on C and E.

For each run we save the existing FOM histories, then compare:
  * homogenized PK2 stress, transformed as S' = R.T @ S @ R;
  * volume-averaged microscopic Neo-Hookean energy density.

Run from this directory with the Kratos Eigen environment enabled:
    source /home/kratos/set_up_kratos_eigen.sh
    python3 run_d4_symmetry_test.py
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


THIS_DIR = Path(__file__).resolve().parent
RVE_DIR = THIS_DIR.parent
if str(RVE_DIR) not in sys.path:
    sys.path.insert(0, str(RVE_DIR))

import KratosMultiphysics as KM  # noqa: E402
import fom_solver_rve as fom  # noqa: E402


# These states are deliberately non-axisymmetric and include shear.  Hence a
# wrong symmetry implementation is visible in both the normal and shear terms.
BASE_STRAINS = {
    "mixed_A": np.array([0.12, -0.04, 0.10], dtype=float),
    "mixed_B": np.array([-0.06, 0.09, -0.12], dtype=float),
    "mixed_C": np.array([0.05, 0.14, 0.18], dtype=float),
}

# Four distinct tensor actions.  D4 has eight matrices, but R and -R produce
# the same R.T @ E @ R for a second-order tensor.
D4_ACTIONS = {
    "identity": np.array([[1.0, 0.0], [0.0, 1.0]]),
    "reflect_x": np.array([[1.0, 0.0], [0.0, -1.0]]),
    "rotate_90": np.array([[0.0, -1.0], [1.0, 0.0]]),
    "reflect_diagonal": np.array([[0.0, 1.0], [1.0, 0.0]]),
}


def voigt_to_tensor(voigt: np.ndarray) -> np.ndarray:
    """Convert [xx, yy, gamma_xy] to a symmetric tensor."""
    xx, yy, gamma_xy = np.asarray(voigt, dtype=float).reshape(3)
    return np.array([[xx, 0.5 * gamma_xy], [0.5 * gamma_xy, yy]], dtype=float)


def tensor_to_voigt(tensor: np.ndarray) -> np.ndarray:
    """Convert a symmetric tensor to [xx, yy, gamma_xy]."""
    tensor = np.asarray(tensor, dtype=float)
    return np.array([tensor[0, 0], tensor[1, 1], 2.0 * tensor[0, 1]], dtype=float)


def strain_transform(strain: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    e_tensor = voigt_to_tensor(strain)
    return tensor_to_voigt(rotation.T @ e_tensor @ rotation)


def stress_transform(stress: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    # The FOM stores [S11, S22, S12] (not engineering shear) for stress.
    s11, s22, s12 = np.asarray(stress, dtype=float).reshape(3)
    s_tensor = np.array([[s11, s12], [s12, s22]], dtype=float)
    transformed = rotation.T @ s_tensor @ rotation
    return np.array([transformed[0, 0], transformed[1, 1], transformed[0, 1]], dtype=float)


def is_admissible(strain: np.ndarray) -> tuple[bool, np.ndarray]:
    c_tensor = np.eye(2) + 2.0 * voigt_to_tensor(strain)
    eigvals = np.linalg.eigvalsh(c_tensor)
    return bool(np.min(eigvals) > 0.0), eigvals


def relative_error(actual: np.ndarray, expected: np.ndarray, floor: float = 1.0e-14) -> float:
    return float(np.linalg.norm(np.asarray(actual) - np.asarray(expected)) / max(np.linalg.norm(expected), floor))


def make_parameters() -> KM.Parameters:
    """Create independent FOM parameters and suppress GiD output for this test."""
    with open(THIS_DIR / "ProjectParameters.json", encoding="utf-8") as parameter_file:
        config = json.load(parameter_file)

    # Histories are written explicitly by RunFomBatchSimulation.  Disabling
    # GiD prevents side effects outside results/ and keeps this experiment lean.
    config["output_processes"] = {"gid_output": [], "vtk_output": []}
    config["solver_settings"]["echo_level"] = 0
    parameters = KM.Parameters(json.dumps(config))
    fom.SetInputMeshFilename(parameters, "rve_geometry")
    material_parts = fom.DetectMaterialSubModelParts("rve_geometry.mdpa")
    parameters = fom.ConfigureElementModelerForMaterialParts(parameters, material_parts)
    fom.SetMaterialsFilename(parameters, "StructuralMaterials.json")
    return parameters


def microscopic_energy_density(u_final: np.ndarray) -> float:
    """Return the RVE-average microscopic Neo-Hookean energy density.

    This is evaluated directly at the converged Gauss points.  It is more
    fundamental than a path-work estimate and is independent of the path-step
    size used to reach the final state.
    """
    parameters = make_parameters()
    model = KM.Model()
    simulation = fom.RVEHomogenizationDatasetGenerator(model, parameters)
    simulation.Initialize()
    try:
        model_part = simulation._GetSolver().GetComputingModelPart()
        n_dof, equation_map, _ = fom.SetUpDofEquationIdsAndDisplacementAdaptor(model_part)
        if np.asarray(u_final).size != n_dof:
            raise RuntimeError(
                f"Stored displacement has {np.asarray(u_final).size} entries, expected {n_dof}."
            )
        assembler = fom.VectorizedAssembler(model_part, n_dof, equation_map, log_label="D4-energy")
        strain_gp, _ = assembler.ComputeStrainStressOnly(np.asarray(u_final, dtype=float))

        c11 = 1.0 + 2.0 * strain_gp[..., 0]
        c22 = 1.0 + 2.0 * strain_gp[..., 1]
        c12 = strain_gp[..., 2]
        det_c = c11 * c22 - c12 * c12
        if np.min(det_c) <= 0.0:
            raise RuntimeError("A converged state contains a non-positive det(C) at a Gauss point.")
        log_j = 0.5 * np.log(det_c)

        young = assembler.young[:, None]
        poisson = assembler.poisson[:, None]
        shear_modulus = young / (2.0 * (1.0 + poisson))
        lame_lambda = young * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))

        # Plane strain restriction of the compressible 3-D Neo-Hookean energy:
        # psi = mu/2 (tr(C)-2) - mu log(J) + lambda/2 log(J)^2.
        psi_gp = (
            0.5 * shear_modulus * (c11 + c22 - 2.0)
            - shear_modulus * log_j
            + 0.5 * lame_lambda * log_j * log_j
        )
        psi_element = np.mean(psi_gp, axis=1)
        return float(np.dot(assembler.area_e, psi_element) / np.sum(assembler.area_e))
    finally:
        simulation.Finalize()


def path_work(strain_history: np.ndarray, stress_history: np.ndarray) -> float:
    """Trapezoidal path-work diagnostic using the repository Voigt convention."""
    delta_strain = np.diff(np.asarray(strain_history, dtype=float), axis=0)
    stress_midpoint = 0.5 * (np.asarray(stress_history, dtype=float)[1:] + stress_history[:-1])
    return float(np.sum(stress_midpoint * delta_strain))


def run_one_case(
    case_name: str,
    applied_strain: np.ndarray,
    results_dir: Path,
    reference_steps: int,
    reference_amplitude: float,
) -> dict:
    admissible, eigvals = is_admissible(applied_strain)
    if not admissible:
        raise RuntimeError(f"{case_name}: C is not SPD; eigenvalues={eigvals.tolist()}.")

    case_dir = results_dir / case_name
    case_dir.mkdir(parents=True, exist_ok=True)
    log_path = case_dir / "solver.log"
    parameters = make_parameters()
    strain_path = np.vstack((np.zeros(3), np.asarray(applied_strain, dtype=float)))

    start = time.perf_counter()
    with open(log_path, "w", encoding="utf-8") as log_file:
        with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
            _, stress_history = fom.RunFomBatchSimulation(
                parameters=parameters,
                out_dir=str(case_dir),
                save_results=True,
                save_plot=False,
                strain_path=strain_path,
                trajectory_index=1,
                reference_amplitude=float(reference_amplitude),
                reference_steps=int(reference_steps),
            )
    elapsed_seconds = time.perf_counter() - start

    applied_history = np.load(case_dir / "trajectory_1_applied_strain.npy")
    homogeneous_strain_history = np.load(case_dir / "trajectory_1_strain.npy")
    stress_history = np.asarray(stress_history, dtype=float)
    displacement_history = np.load(case_dir / "trajectory_1_U.npy")
    energy_fem = microscopic_energy_density(displacement_history[-1])

    return {
        "case": case_name,
        "input_strain": np.asarray(applied_strain, dtype=float),
        "C_eigenvalues": eigvals,
        "stress_final": stress_history[-1],
        "homogeneous_strain_final": homogeneous_strain_history[-1],
        "energy_fem": energy_fem,
        "path_work_applied": path_work(applied_history, stress_history),
        "path_work_homogenized": path_work(homogeneous_strain_history, stress_history),
        "n_steps": int(stress_history.shape[0] - 1),
        "wall_seconds": elapsed_seconds,
        "case_dir": str(case_dir),
    }


def to_json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def write_report(records: list[dict], results_dir: Path) -> None:
    by_base: dict[str, dict[str, dict]] = {}
    for record in records:
        by_base.setdefault(record["base_case"], {})[record["action"]] = record

    comparison_rows = []
    for base_name, cases in by_base.items():
        reference = cases["identity"]
        for action_name, record in cases.items():
            rotation = D4_ACTIONS[action_name]
            expected_stress = stress_transform(reference["stress_final"], rotation)
            expected_strain = strain_transform(reference["homogeneous_strain_final"], rotation)
            energy_reference = float(reference["energy_fem"])
            energy_rel = abs(float(record["energy_fem"]) - energy_reference) / max(abs(energy_reference), 1.0e-14)
            comparison_rows.append(
                {
                    "base_case": base_name,
                    "action": action_name,
                    "stress_rel_error": relative_error(record["stress_final"], expected_stress),
                    "strain_rel_error": relative_error(record["homogeneous_strain_final"], expected_strain),
                    "energy_rel_error": float(energy_rel),
                    "energy_fem": float(record["energy_fem"]),
                    "energy_fem_reference": energy_reference,
                    "stress_final": record["stress_final"],
                    "stress_expected": expected_stress,
                }
            )

    comparison_path = results_dir / "d4_symmetry_metrics.csv"
    with open(comparison_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "base_case", "action", "stress_rel_error", "strain_rel_error", "energy_rel_error",
                "energy_fem", "energy_fem_reference", "stress_final", "stress_expected",
            ]
        )
        for row in comparison_rows:
            writer.writerow(
                [
                    row["base_case"], row["action"], f"{row['stress_rel_error']:.16e}",
                    f"{row['strain_rel_error']:.16e}", f"{row['energy_rel_error']:.16e}",
                    f"{row['energy_fem']:.16e}", f"{row['energy_fem_reference']:.16e}",
                    json.dumps(to_json_value(row["stress_final"])),
                    json.dumps(to_json_value(row["stress_expected"])),
                ]
            )

    max_stress = max(row["stress_rel_error"] for row in comparison_rows)
    max_strain = max(row["strain_rel_error"] for row in comparison_rows)
    max_energy = max(row["energy_rel_error"] for row in comparison_rows)
    result = {
        "test": "effective_D4_square_symmetry",
        "convention": {
            "strain_input": "[E11, E22, gamma12] with gamma12 = 2 E12",
            "stress_output": "[S11, S22, S12]",
            "material_action": "E' = R^T E R and S' = R^T S R",
            "energy": "RVE-average microscopic Neo-Hookean energy at converged Gauss points",
        },
        "base_strains": {key: value.tolist() for key, value in BASE_STRAINS.items()},
        "actions": {key: value.tolist() for key, value in D4_ACTIONS.items()},
        "max_relative_errors": {
            "stress": float(max_stress),
            "homogenized_strain": float(max_strain),
            "energy": float(max_energy),
        },
        "records": [{key: to_json_value(value) for key, value in record.items()} for record in records],
        "comparisons": [{key: to_json_value(value) for key, value in row.items()} for row in comparison_rows],
    }
    with open(results_dir / "d4_symmetry_summary.json", "w", encoding="utf-8") as json_file:
        json.dump(result, json_file, indent=2)

    lines = [
        "# Effective D4 symmetry test — FOM result",
        "",
        "The material transformation is \\(\\mathbf E'=\\mathbf R^T\\mathbf E\\mathbf R\\) and \\(\\mathbf S'=\\mathbf R^T\\mathbf S\\mathbf R\\).",
        "The reported energy is evaluated from the microscopic Neo-Hookean energy at the converged Gauss points.",
        "",
        "| Quantity | Maximum relative discrepancy |",
        "|---|---:|",
        f"| PK2 stress | {max_stress:.6e} |",
        f"| Homogenized strain | {max_strain:.6e} |",
        f"| Microscopic energy | {max_energy:.6e} |",
        "",
        "## Per-case comparison",
        "",
        "| Base state | D4 action | Stress error | Strain error | Energy error |",
        "|---|---|---:|---:|---:|",
    ]
    for row in comparison_rows:
        lines.append(
            f"| {row['base_case']} | {row['action']} | {row['stress_rel_error']:.3e} | "
            f"{row['strain_rel_error']:.3e} | {row['energy_rel_error']:.3e} |"
        )
    lines.extend(
        [
            "",
            "Interpretation: errors close to the FOM/mesh discretization tolerance support the expected square symmetry. "
            "They do not establish isotropy, because rotations such as 30 or 45 degrees are deliberately not part of D4.",
        ]
    )
    (results_dir / "d4_symmetry_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-steps", type=int, default=100, help="Steps for a reference-amplitude segment.")
    parser.add_argument("--reference-amplitude", type=float, default=0.15, help="Reference amplitude for adaptive FOM stepping.")
    parser.add_argument("--results-dir", type=Path, default=THIS_DIR / "results", help="Directory for all generated artifacts.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.reference_steps < 1 or args.reference_amplitude <= 0.0:
        raise ValueError("--reference-steps must be >= 1 and --reference-amplitude must be positive.")

    # All copied input files are intentionally local to this experiment.
    os.chdir(THIS_DIR)
    results_dir = args.results_dir.resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    records = []
    run_count = len(BASE_STRAINS) * len(D4_ACTIONS)
    current = 0
    for base_name, base_strain in BASE_STRAINS.items():
        for action_name, rotation in D4_ACTIONS.items():
            current += 1
            input_strain = strain_transform(base_strain, rotation)
            case_name = f"{base_name}__{action_name}"
            print(f"[{current:02d}/{run_count}] {case_name}: E={input_strain.tolist()}", flush=True)
            record = run_one_case(
                case_name=case_name,
                applied_strain=input_strain,
                results_dir=results_dir,
                reference_steps=args.reference_steps,
                reference_amplitude=args.reference_amplitude,
            )
            record["base_case"] = base_name
            record["action"] = action_name
            records.append(record)

    write_report(records, results_dir)
    summary = json.loads((results_dir / "d4_symmetry_summary.json").read_text(encoding="utf-8"))
    errors = summary["max_relative_errors"]
    print("\nD4 FOM symmetry test completed.")
    print(f"  max stress relative error : {errors['stress']:.6e}")
    print(f"  max strain relative error : {errors['homogenized_strain']:.6e}")
    print(f"  max energy relative error : {errors['energy']:.6e}")
    print(f"  report: {results_dir / 'd4_symmetry_summary.md'}")


if __name__ == "__main__":
    main()
