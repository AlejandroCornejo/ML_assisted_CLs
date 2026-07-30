#!/usr/bin/env python3
"""Finite-difference audit of the FOM macroscopic energy Hessian.

The PANN target is the direct microscopic energy density W(E), with

    e = [E11, E22, gamma12],  gamma12 = 2 E12,
    s = dW/de = [S11, S22, S12].

Consequently the relevant material tangent for the question ``is it SPD?`` is

    D = d s / d e = d^2 W / d e^2.

This script does not use a neural network.  It re-solves the full Kratos FOM
at a symmetric 19-point finite-difference stencil about selected converged
RVE states, evaluates the microscopic Neo-Hookean energy directly at the
converged Gauss points, and forms a symmetric Hessian from that energy.

The selected base states are loaded from the saved FOM displacement histories.
Every perturbation is then solved by continuation from that converged state.
This makes the test an FOM measurement rather than a surrogate diagnostic.

Run from this directory after enabling Kratos Eigen:

    source /home/kratos/set_up_kratos_eigen.sh
    python3 run_fom_energy_hessian_audit.py --step 1e-2

The default run uses six representative states.  A second run with a smaller
step can be made for a selected state to check finite-difference sensitivity:

    python3 run_fom_energy_hessian_audit.py --step 5e-3 --state extreme_biaxial_shear
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
RESULTS_DIR = HERE / "results"
PANN_DATA = REPO_ROOT / "pann" / "data" / "alltraj_stage10_direct_energy.npz"
STAGE1_DIR = REPO_ROOT / "trajectories" / "stage_1_training_set_fom"
# "stage10_mixed" only needs one arbitrary already-converged FOM state to probe
# curvature at -- its two reference arrays are copied locally rather than
# reviving the old stage_10_hprom_ann_ls_results_..._ann_hrom directory (that
# directory's trajectory is known not to match the canonical Stage-10 path
# used elsewhere; irrelevant here since this audit does not compare against it).
STAGE10_MIXED_DIR = HERE / "reference_states"

if str(REPO_ROOT / "core") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "core"))

import KratosMultiphysics as KM  # noqa: E402
import fom_solver_rve as fom  # noqa: E402


@dataclass(frozen=True)
class StateSpec:
    """One converged FOM state used as centre of a finite-difference stencil."""

    name: str
    description: str
    source: str
    target: tuple[float, float, float] | None = None
    source_index: int | None = None


STATE_SPECS = (
    StateSpec(
        "reference",
        "Reference configuration.",
        "stage1_trajectory_1",
        target=(0.0, 0.0, 0.0),
    ),
    StateSpec(
        "compressed_shear",
        "Most compressed Stage-1 corner with positive engineering shear.",
        "stage1_trajectory_6",
        target=(-0.1, -0.1, 0.1),
    ),
    StateSpec(
        "moderate_mixed",
        "Interior mixed state from the positive-shear training sheet.",
        "stage1_trajectory_3",
        target=(0.5, 0.5, 0.05),
    ),
    StateSpec(
        "large_biaxial",
        "Large equal biaxial extension in the zero-shear sheet.",
        "stage1_trajectory_1",
        target=(1.0, 1.0, 0.0),
    ),
    StateSpec(
        "extreme_biaxial_shear",
        "Largest tensile Stage-1 corner in the positive-shear sheet.",
        "stage1_trajectory_5",
        target=(2.0, 2.0, 0.1),
    ),
    StateSpec(
        "stage10_mixed",
        "Strictly held-out Stage-10 mixed state (history index 600).",
        "stage10",
        source_index=600,
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--step",
        type=float,
        default=1.0e-2,
        help="Central finite-difference step in each component of e (default: 1e-2).",
    )
    parser.add_argument(
        "--state",
        action="append",
        choices=[spec.name for spec in STATE_SPECS],
        help="Audit only this state. Repeat the option for several states.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run endpoints even if their compact endpoint JSON files already exist.",
    )
    parser.add_argument(
        "--keep-transient",
        action="store_true",
        help="Keep the temporary displacement histories created by the FOM solver.",
    )
    parser.add_argument(
        "--assemble",
        action="store_true",
        help="Assemble the per-state summaries already on disk; do not run Kratos.",
    )
    parser.add_argument(
        "--single-process",
        action="store_true",
        help="Run all requested states in this process (diagnostic only).",
    )
    return parser.parse_args()


def e_to_C(e: np.ndarray) -> np.ndarray:
    """Return C for e=[E11,E22,gamma12], gamma12=2E12."""
    e11, e22, gamma12 = np.asarray(e, dtype=float).reshape(3)
    return np.array(
        [[1.0 + 2.0 * e11, gamma12], [gamma12, 1.0 + 2.0 * e22]],
        dtype=float,
    )


def assert_admissible(e: np.ndarray, context: str) -> None:
    eig = np.linalg.eigvalsh(e_to_C(e))
    if np.min(eig) <= 0.0:
        raise ValueError(f"{context}: C is not positive definite; eigenvalues={eig.tolist()}")


def make_parameters() -> KM.Parameters:
    """Create isolated FOM parameters with paths independent of the cwd."""
    with open(REPO_ROOT / "core" / "ProjectParameters.json", encoding="utf-8") as parameter_file:
        config = json.load(parameter_file)

    # Histories are controlled by this script.  Disable GiD/VTK side effects.
    config["output_processes"] = {"gid_output": [], "vtk_output": []}
    config["solver_settings"]["echo_level"] = 0
    parameters = KM.Parameters(json.dumps(config))

    mesh_base = str(REPO_ROOT / "core" / "rve_geometry")
    fom.SetInputMeshFilename(parameters, mesh_base)
    material_parts = fom.DetectMaterialSubModelParts(str(REPO_ROOT / "core" / "rve_geometry.mdpa"))
    parameters = fom.ConfigureElementModelerForMaterialParts(parameters, material_parts)
    fom.SetMaterialsFilename(parameters, str(REPO_ROOT / "core" / "StructuralMaterials.json"))
    return parameters


def _microscopic_energy_density_verbose(u_final: np.ndarray) -> float:
    """Evaluate the RVE-average microscopic Neo-Hookean energy at a converged state."""
    parameters = make_parameters()
    model = KM.Model()
    simulation = fom.RVEHomogenizationDatasetGenerator(model, parameters)
    simulation.Initialize()
    try:
        model_part = simulation._GetSolver().GetComputingModelPart()
        n_dof, equation_map, _ = fom.SetUpDofEquationIdsAndDisplacementAdaptor(model_part)
        u_final = np.asarray(u_final, dtype=float).reshape(-1)
        if u_final.size != n_dof:
            raise RuntimeError(f"Stored displacement has {u_final.size} entries; expected {n_dof}.")

        assembler = fom.VectorizedAssembler(model_part, n_dof, equation_map, log_label="FOM-energy")
        strain_gp, _ = assembler.ComputeStrainStressOnly(u_final)

        c11 = 1.0 + 2.0 * strain_gp[..., 0]
        c22 = 1.0 + 2.0 * strain_gp[..., 1]
        c12 = strain_gp[..., 2]
        det_c = c11 * c22 - c12 * c12
        if np.min(det_c) <= 0.0:
            raise RuntimeError("A converged RVE state has det(C)<=0 at a Gauss point.")
        log_j = 0.5 * np.log(det_c)

        young = assembler.young[:, None]
        poisson = assembler.poisson[:, None]
        shear_modulus = young / (2.0 * (1.0 + poisson))
        lame_lambda = young * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
        psi_gp = (
            0.5 * shear_modulus * (c11 + c22 - 2.0)
            - shear_modulus * log_j
            + 0.5 * lame_lambda * log_j * log_j
        )
        psi_element = np.mean(psi_gp, axis=1)
        return float(np.dot(assembler.area_e, psi_element) / np.sum(assembler.area_e))
    finally:
        simulation.Finalize()


def microscopic_energy_density(u_final: np.ndarray) -> float:
    """Energy evaluation without Kratos' repeated initialization chatter on stdout."""
    with open(os.devnull, "w", encoding="utf-8") as null_stream:
        with contextlib.redirect_stdout(null_stream), contextlib.redirect_stderr(null_stream):
            return _microscopic_energy_density_verbose(u_final)


def source_paths(source: str) -> tuple[Path, Path]:
    if source.startswith("stage1_trajectory_"):
        trajectory = int(source.rsplit("_", maxsplit=1)[1])
        root = STAGE1_DIR / f"trajectory_{trajectory}"
        return (
            root / f"trajectory_{trajectory}_applied_strain.npy",
            root / f"trajectory_{trajectory}_U.npy",
        )
    if source == "stage10":
        return (
            STAGE10_MIXED_DIR / "stage10_mixed_applied_strain.npy",
            STAGE10_MIXED_DIR / "stage10_mixed_U.npy",
        )
    raise ValueError(f"Unknown state source: {source}")


def load_base_state(spec: StateSpec) -> tuple[np.ndarray, np.ndarray, dict]:
    applied_path, displacement_path = source_paths(spec.source)
    if not applied_path.exists() or not displacement_path.exists():
        raise FileNotFoundError(f"Missing FOM history for '{spec.name}': {applied_path} or {displacement_path}")

    applied = np.load(applied_path, mmap_mode="r")
    if spec.source_index is not None:
        idx = int(spec.source_index)
    else:
        target = np.asarray(spec.target, dtype=float)
        errors = np.max(np.abs(np.asarray(applied) - target[None, :]), axis=1)
        idx = int(np.argmin(errors))
        if float(errors[idx]) > 1.0e-12:
            raise RuntimeError(
                f"Could not recover exact base state '{spec.name}'. "
                f"Closest applied strain={np.asarray(applied[idx]).tolist()}, error={float(errors[idx]):.3e}."
            )
    e0 = np.asarray(applied[idx], dtype=float).copy()
    assert_admissible(e0, f"base state {spec.name}")
    displacement = np.load(displacement_path, mmap_mode="r")
    u0 = np.asarray(displacement[idx], dtype=float).copy()
    info = {
        "source": spec.source,
        "source_index": idx,
        "applied_history": str(applied_path),
        "displacement_history": str(displacement_path),
    }
    return e0, u0, info


def lookup_direct_label(e: np.ndarray) -> tuple[float | None, np.ndarray | None]:
    """Return existing direct W,S labels if the state belongs to the saved datasets."""
    if not PANN_DATA.exists():
        return None, None
    data = np.load(PANN_DATA)
    candidates = (
        (data["train_strain"], data["train_energy"], data["train_stress"]),
        (data["stage10_strain"], data["stage10_energy"], data["stage10_stress"]),
    )
    for strains, energy, stress in candidates:
        errors = np.max(np.abs(np.asarray(strains) - e[None, :]), axis=1)
        idx = int(np.argmin(errors))
        if float(errors[idx]) < 1.0e-12:
            return float(energy[idx]), np.asarray(stress[idx], dtype=float)
    return None, None


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    raise TypeError(f"Cannot JSON-serialize {type(value).__name__}")


def endpoint_label(delta: np.ndarray, atol: float = 1.0e-15) -> str:
    parts: list[str] = []
    for i, value in enumerate(np.asarray(delta, dtype=float)):
        if abs(value) <= atol:
            continue
        parts.append(("p" if value > 0.0 else "m") + str(i))
    return "zero" if not parts else "_".join(parts)


def solve_endpoint(
    *,
    state_name: str,
    e0: np.ndarray,
    u0: np.ndarray,
    delta: np.ndarray,
    h: float,
    run_dir: Path,
    force: bool,
    keep_transient: bool,
) -> dict:
    """Solve the FOM at e0+delta by continuation from the stored converged state."""
    target = np.asarray(e0 + delta, dtype=float)
    assert_admissible(target, f"endpoint {state_name}/{endpoint_label(delta)}")
    label = endpoint_label(delta)
    case_dir = run_dir / state_name / label
    case_dir.mkdir(parents=True, exist_ok=True)
    endpoint_path = case_dir / "endpoint.json"
    if endpoint_path.exists() and not force:
        with open(endpoint_path, encoding="utf-8") as stream:
            return json.load(stream)

    parameters = make_parameters()
    start = time.perf_counter()
    # The FOM evaluates the microscopic energy from its converged assembler.
    # This avoids starting a second Kratos model for every stencil endpoint.
    with open(case_dir / "solver.log", "w", encoding="utf-8") as log_file:
        with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
            _, stress_history, energy = fom.RunFomBatchSimulation(
                parameters=parameters,
                out_dir=str(case_dir),
                save_results=False,
                save_plot=False,
                strain_path=np.vstack((e0, target)),
                trajectory_index=1,
                reference_amplitude=max(float(h), 1.0e-12),
                reference_steps=10,
                initial_displacement=u0,
                use_old_stiffness_in_first_iteration=False,
                return_final_energy=True,
            )
    wall_seconds = time.perf_counter() - start
    record = {
        "state": state_name,
        "label": label,
        "delta": np.asarray(delta, dtype=float),
        "target_strain": target,
        "energy": float(energy),
        "legacy_homogenized_stress": np.asarray(stress_history[-1], dtype=float),
        "wall_seconds": float(wall_seconds),
    }
    with open(endpoint_path, "w", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, default=json_value)

    if not keep_transient:
        for filename in (
            "trajectory_1_U.npy",
            "trajectory_1_strain.npy",
            "trajectory_1_stress.npy",
            "trajectory_1_applied_strain.npy",
            "reference_measure_A0.txt",
        ):
            path = case_dir / filename
            if path.exists():
                path.unlink()
    return json.loads(json.dumps(record, default=json_value))


def make_stencil(h: float) -> dict[str, np.ndarray]:
    """Central 19-point stencil for a 3D Hessian."""
    stencil = {"zero": np.zeros(3, dtype=float)}
    for i in range(3):
        unit = np.zeros(3, dtype=float)
        unit[i] = h
        stencil[endpoint_label(unit)] = unit
        stencil[endpoint_label(-unit)] = -unit
    for i in range(3):
        for j in range(i + 1, 3):
            for sign_i in (-1.0, 1.0):
                for sign_j in (-1.0, 1.0):
                    delta = np.zeros(3, dtype=float)
                    delta[i] = sign_i * h
                    delta[j] = sign_j * h
                    stencil[endpoint_label(delta)] = delta
    if len(stencil) != 19:
        raise RuntimeError(f"Expected 19 stencil points, found {len(stencil)}.")
    return stencil


def energy_hessian_from_stencil(energy: dict[str, float], h: float) -> tuple[np.ndarray, np.ndarray]:
    gradient = np.empty(3, dtype=float)
    hessian = np.empty((3, 3), dtype=float)
    zero = float(energy["zero"])
    for i in range(3):
        unit = np.zeros(3, dtype=float)
        unit[i] = h
        plus = float(energy[endpoint_label(unit)])
        minus = float(energy[endpoint_label(-unit)])
        gradient[i] = (plus - minus) / (2.0 * h)
        hessian[i, i] = (plus - 2.0 * zero + minus) / (h * h)
    for i in range(3):
        for j in range(i + 1, 3):
            values: dict[tuple[int, int], float] = {}
            for sign_i in (-1.0, 1.0):
                for sign_j in (-1.0, 1.0):
                    delta = np.zeros(3, dtype=float)
                    delta[i] = sign_i * h
                    delta[j] = sign_j * h
                    values[(int(sign_i), int(sign_j))] = float(energy[endpoint_label(delta)])
            mixed = (
                values[(1, 1)]
                - values[(1, -1)]
                - values[(-1, 1)]
                + values[(-1, -1)]
            ) / (4.0 * h * h)
            hessian[i, j] = mixed
            hessian[j, i] = mixed
    return gradient, hessian


def run_state(spec: StateSpec, h: float, run_dir: Path, force: bool, keep_transient: bool) -> dict:
    e0, u0, source_info = load_base_state(spec)
    base_energy = microscopic_energy_density(u0)
    labeled_energy, labeled_stress = lookup_direct_label(e0)

    endpoint_records: dict[str, dict] = {
        "zero": {
            "state": spec.name,
            "label": "zero",
            "delta": np.zeros(3),
            "target_strain": e0,
            "energy": base_energy,
            "legacy_homogenized_stress": None,
            "wall_seconds": 0.0,
        }
    }
    stencil = make_stencil(h)
    for label, delta in stencil.items():
        if label == "zero":
            continue
        endpoint_records[label] = solve_endpoint(
            state_name=spec.name,
            e0=e0,
            u0=u0,
            delta=delta,
            h=h,
            run_dir=run_dir,
            force=force,
            keep_transient=keep_transient,
        )

    energy = {label: float(record["energy"]) for label, record in endpoint_records.items()}
    stress_fd, tangent_fd = energy_hessian_from_stencil(energy, h)
    eigvals, eigvecs = np.linalg.eigh(tangent_fd)
    return {
        "name": spec.name,
        "description": spec.description,
        "strain": e0,
        "C_eigenvalues": np.linalg.eigvalsh(e_to_C(e0)),
        "source": source_info,
        "base_energy": base_energy,
        "existing_direct_energy": labeled_energy,
        "existing_direct_stress": labeled_stress,
        "stress_fd": stress_fd,
        "stress_fd_minus_existing_direct": None if labeled_stress is None else stress_fd - labeled_stress,
        "tangent_fd": tangent_fd,
        "tangent_eigenvalues": eigvals,
        "min_eigenvector": eigvecs[:, 0],
        "min_eigenvalue": float(eigvals[0]),
        "is_spd": bool(eigvals[0] > 0.0),
        "endpoint_wall_seconds": float(sum(float(record["wall_seconds"]) for record in endpoint_records.values())),
        "endpoints": endpoint_records,
    }


def write_outputs(records: list[dict], h: float, run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    compact_records = []
    for record in records:
        compact = {key: value for key, value in record.items() if key != "endpoints"}
        compact_records.append(compact)

    summary = {
        "purpose": "FOM finite-difference Hessian of the direct macroscopic energy W(E)",
        "strain_convention": "e=[E11,E22,gamma12], gamma12=2E12",
        "stress_convention": "s=dW/de=[S11,S22,S12]",
        "tangent_definition": "D=d2W/de2=dS/de",
        "finite_difference": "central 19-point energy stencil",
        "step": float(h),
        "units": {"energy": "Pa", "stress": "Pa", "tangent": "Pa"},
        "records": compact_records,
    }
    with open(run_dir / "fom_energy_hessian_summary.json", "w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, default=json_value)

    np.savez(
        run_dir / "fom_energy_hessian_audit.npz",
        names=np.asarray([record["name"] for record in records]),
        descriptions=np.asarray([record["description"] for record in records]),
        strain=np.stack([record["strain"] for record in records]),
        C_eigenvalues=np.stack([record["C_eigenvalues"] for record in records]),
        base_energy=np.asarray([record["base_energy"] for record in records]),
        stress_fd=np.stack([record["stress_fd"] for record in records]),
        tangent_fd=np.stack([record["tangent_fd"] for record in records]),
        tangent_eigenvalues=np.stack([record["tangent_eigenvalues"] for record in records]),
        min_eigenvector=np.stack([record["min_eigenvector"] for record in records]),
        min_eigenvalue=np.asarray([record["min_eigenvalue"] for record in records]),
        step=np.asarray([h]),
    )

    lines = [
        "# FOM energy-tangent audit",
        "",
        "The reported matrix is the central finite-difference estimate of",
        "`D = d²W/de² = dS/de`, not the derivative of Cauchy stress.",
        "",
        f"Finite-difference step: `{h:.6g}`.",
        "",
        "| State | e=[E11,E22,gamma12] | eigenvalues of D [GPa] | SPD? |",
        "|---|---:|---:|:---:|",
    ]
    for record in records:
        e_text = ", ".join(f"{value:.6g}" for value in record["strain"])
        eig_text = ", ".join(f"{value / 1.0e9:.6g}" for value in record["tangent_eigenvalues"])
        lines.append(f"| {record['name']} | [{e_text}] | [{eig_text}] | {'yes' if record['is_spd'] else 'no'} |")
    (run_dir / "fom_energy_hessian_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_state_summary(record: dict, run_dir: Path) -> None:
    """Persist one completed state, so states may be run in separate processes.

    Kratos owns substantial C++ state during each solve.  Running each
    19-point stencil in its own Python process keeps this FOM audit robust and
    lets an interrupted batch resume safely from endpoint JSON files.
    """
    compact = {key: value for key, value in record.items() if key != "endpoints"}
    path = run_dir / record["name"] / "state_summary.json"
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(compact, stream, indent=2, default=json_value)


def load_state_summaries(run_dir: Path, selected: tuple[StateSpec, ...]) -> list[dict]:
    records = []
    missing = []
    for spec in selected:
        path = run_dir / spec.name / "state_summary.json"
        if not path.exists():
            missing.append(spec.name)
            continue
        with open(path, encoding="utf-8") as stream:
            records.append(json.load(stream))
    if missing:
        raise FileNotFoundError(
            "Cannot assemble because state summaries are missing for: " + ", ".join(missing)
        )
    return records


def main() -> None:
    args = parse_args()
    h = float(args.step)
    if not np.isfinite(h) or h <= 0.0:
        raise ValueError("--step must be a finite positive number.")
    selected = STATE_SPECS if args.state is None else tuple(spec for spec in STATE_SPECS if spec.name in set(args.state))
    run_dir = RESULTS_DIR / f"h_{h:.3e}"
    if args.assemble:
        records = load_state_summaries(run_dir, selected)
        write_outputs(records, h, run_dir)
        print(f"[FOM tangent] Assembled {len(records)} completed state summaries into {run_dir}.")
        return

    # Kratos allocates substantial C++ state for every nonlinear solve. One
    # isolated Python process per 19-point stencil is more robust than a long
    # monolithic batch and makes a partially completed audit resumable.
    if args.state is None and not args.single_process:
        run_dir.mkdir(parents=True, exist_ok=True)
        for spec in selected:
            summary_path = run_dir / spec.name / "state_summary.json"
            if summary_path.exists() and not args.force:
                print(f"[FOM tangent] Reusing completed state: {spec.name}")
                continue
            command = [sys.executable, str(Path(__file__).resolve()), "--step", str(h), "--state", spec.name]
            if args.force:
                command.append("--force")
            if args.keep_transient:
                command.append("--keep-transient")
            driver_log = run_dir / f"{spec.name}_driver.log"
            for attempt in range(1, 4):
                print(f"[FOM tangent] Launching isolated stencil {spec.name} (attempt {attempt}/3).")
                with open(driver_log, "a", encoding="utf-8") as log_file:
                    result = subprocess.run(command, stdout=log_file, stderr=subprocess.STDOUT, check=False)
                if result.returncode != 0:
                    raise RuntimeError(f"FOM tangent child '{spec.name}' exited with code {result.returncode}; see {driver_log}.")
                if summary_path.exists():
                    break
            else:
                raise RuntimeError(f"FOM tangent child '{spec.name}' did not write {summary_path} after three attempts.")
        records = load_state_summaries(run_dir, selected)
        write_outputs(records, h, run_dir)
        print(f"[FOM tangent] Assembled {len(records)} isolated FOM stencils into {run_dir}.")
        return

    print(f"[FOM tangent] Central energy stencil step h={h:.6g}; states={[spec.name for spec in selected]}")
    started = time.perf_counter()
    records = []
    for i, spec in enumerate(selected, start=1):
        print(f"[FOM tangent] ({i}/{len(selected)}) {spec.name}: {spec.description}")
        record = run_state(spec, h, run_dir, force=args.force, keep_transient=args.keep_transient)
        write_state_summary(record, run_dir)
        records.append(record)
        print(
            "[FOM tangent] "
            f"lambda_min={record['min_eigenvalue'] / 1.0e9:.6g} GPa, "
            f"eigs={record['tangent_eigenvalues'] / 1.0e9} GPa"
        )
    write_outputs(records, h, run_dir)
    print(f"[FOM tangent] Completed in {time.perf_counter() - started:.1f} s. Outputs: {run_dir}")


if __name__ == "__main__":
    main()
