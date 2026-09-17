"""Freeze material-B strain coordinates; do not solve the FOM or train models."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import scipy
from scipy.spatial import cKDTree
from scipy.stats import qmc


BASE = Path(__file__).resolve().parents[1]
DEFAULT_SPEC = Path(__file__).with_name("data_protocol_v1.json")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sobol(count: int, dimension: int, seed: int) -> np.ndarray:
    power = int(np.log2(count))
    if 2**power != count:
        raise ValueError("Sobol block sizes must be powers of two")
    return qmc.Sobol(d=dimension, scramble=True, seed=seed).random_base2(power)


def scale(points: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    return lower + points * (upper - lower)


def corners(lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    return np.asarray([[upper[j] if (mask >> j) & 1 else lower[j] for j in range(3)]
                       for mask in range(8)], dtype=float)


def face_points(spec: dict, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    setting = spec["sampling"]["fit_faces"]
    seeds = setting["seed_by_face"]
    count = setting["count_per_face"]
    rows = []
    face = 0
    for fixed_axis in range(3):
        free_axes = [j for j in range(3) if j != fixed_axis]
        for side in (0, 1):
            unit = sobol(count, 2, seeds[face])
            points = np.empty((count, 3), dtype=float)
            points[:, fixed_axis] = (lower if side == 0 else upper)[fixed_axis]
            points[:, free_axes] = scale(unit, lower[free_axes], upper[free_axes])
            rows.append(points)
            face += 1
    return np.vstack(rows)


def maximin(points: np.ndarray, count: int, initial=()) -> np.ndarray:
    """Deterministic strain-only farthest-point selection."""
    points = np.asarray(points, dtype=float)
    selected = list(map(int, initial))
    if not selected:
        selected = [0]
    distance = np.min(np.sum((points[:, None, :] - points[selected][None, :, :])**2, axis=2), axis=1)
    distance[selected] = -1.0
    while len(selected) < count:
        index = int(np.argmax(distance))
        selected.append(index)
        distance = np.minimum(distance, np.sum((points - points[index])**2, axis=1))
        distance[selected] = -1.0
    return np.asarray(selected, dtype=int)


def minimum_separation(groups: list[np.ndarray], lower: np.ndarray, upper: np.ndarray) -> float:
    points = (np.vstack(groups) - lower) / (upper - lower)
    distances, _ = cKDTree(points).query(points, k=2)
    return float(np.min(distances[:, 1]))


def minimum_macro_eigenvalue(points: np.ndarray) -> float:
    values = []
    for e11, e22, gamma in np.asarray(points):
        values.append(np.linalg.eigvalsh([[1 + 2*e11, gamma], [gamma, 1 + 2*e22]])[0])
    return float(min(values))


def build(spec: dict) -> tuple[dict[str, np.ndarray], dict]:
    lower = np.asarray(spec["bounds"]["lower"], dtype=float)
    upper = np.asarray(spec["bounds"]["upper"], dtype=float)
    sampling = spec["sampling"]
    fit_volume = scale(sobol(sampling["fit_volume"]["count"], 3,
                             sampling["fit_volume"]["seed"]), lower, upper)
    fit_faces = face_points(spec, lower, upper)
    fit_corners = corners(lower, upper)
    fit = np.vstack((fit_volume, fit_faces, fit_corners))
    validation = scale(sobol(sampling["validation"]["count"], 3,
                             sampling["validation"]["seed"]), lower, upper)
    test = scale(sobol(sampling["test"]["count"], 3,
                       sampling["test"]["seed"]), lower, upper)
    reference = np.asarray(sampling["reference_state"], dtype=float)[None, :]

    path_names = list(spec["paths"]["endpoints"])
    endpoints = np.asarray([spec["paths"]["endpoints"][name] for name in path_names], dtype=float)
    parameter = np.linspace(0.0, 1.0, spec["paths"]["points_per_path"])
    path_states = endpoints[:, None, :] * parameter[None, :, None]

    span = upper - lower
    normalized = lambda x: (x - lower) / span
    volume_audit = maximin(normalized(fit_volume), 16)
    validation_audit = maximin(normalized(validation), 16)
    test_audit = maximin(normalized(test), 16)
    boundary = np.vstack((fit_faces, fit_corners))
    corner_local = np.arange(len(fit_faces), len(boundary))
    boundary_audit = maximin(normalized(boundary), 16, initial=corner_local)
    audit_states = np.vstack((fit_volume[volume_audit], validation[validation_audit],
                              test[test_audit], boundary[boundary_audit]))
    audit_split = np.asarray((["fit_volume"]*16 + ["validation"]*16 + ["test"]*16
                              + ["fit_boundary"]*16))
    audit_index = np.concatenate((volume_audit, validation_audit, test_audit, boundary_audit))

    cold_fit = maximin(normalized(fit_volume), 8)
    cold_validation = maximin(normalized(validation), 8)
    cold_test = maximin(normalized(test), 8)
    cold_states = np.vstack((fit_volume[cold_fit], validation[cold_validation], test[cold_test]))
    cold_split = np.asarray(["fit_volume"]*8 + ["validation"]*8 + ["test"]*8)
    cold_index = np.concatenate((cold_fit, cold_validation, cold_test))

    all_requested = np.vstack((fit, validation, test, path_states.reshape(-1, 3), reference))
    face_counts = [int(np.sum(np.isclose(fit_faces[:, j], value, atol=0, rtol=0)))
                   for j in range(3) for value in (lower[j], upper[j])]
    separation = minimum_separation([fit, validation, test], lower, upper)
    checks = {
        "fit_count": len(fit) == sampling["fit_total"],
        "face_counts": face_counts == [sampling["fit_faces"]["count_per_face"]]*6,
        "corner_count": len(fit_corners) == sampling["fit_corners"],
        "validation_count": len(validation) == sampling["validation"]["count"],
        "test_count": len(test) == sampling["test"]["count"],
        "statistical_sets_have_no_duplicate": separation > 1e-12,
        "all_requested_states_inside_closed_box": bool(np.all(all_requested >= lower-1e-14)
                                                    and np.all(all_requested <= upper+1e-14)),
        "all_requested_macro_C_positive_definite": minimum_macro_eigenvalue(all_requested) > 0,
        "mesh_audit_count": len(audit_states) == spec["mesh_policy"]["audit_count"],
        "cold_start_count": len(cold_states) == spec["mesh_policy"]["cold_start_count"],
    }
    arrays = {
        "E_fit": fit,
        "fit_kind": np.asarray(["volume"]*len(fit_volume) + ["face"]*len(fit_faces)
                               + ["corner"]*len(fit_corners)),
        "E_validation": validation,
        "E_test": test,
        "E_reference": reference,
        "path_names": np.asarray(path_names),
        "path_parameter": parameter,
        "path_states": path_states,
        "mesh_audit_states": audit_states,
        "mesh_audit_split": audit_split,
        "mesh_audit_index": audit_index,
        "cold_start_states": cold_states,
        "cold_start_split": cold_split,
        "cold_start_index": cold_index,
        "lower": lower,
        "upper": upper,
    }
    summary = {
        "status": "coordinates_frozen" if all(checks.values()) else "failed",
        "counts": {"fit": len(fit), "fit_volume": len(fit_volume), "fit_faces": len(fit_faces),
                   "fit_corners": len(fit_corners), "validation": len(validation), "test": len(test),
                   "paths": len(path_names), "unique_nonreference_path_states":
                   int(len(path_names)*(len(parameter)-1)), "mesh_audit": len(audit_states),
                   "cold_start": len(cold_states)},
        "checks": checks,
        "minimum_normalized_separation_fit_validation_test": separation,
        "minimum_macro_C_eigenvalue_all_requested": minimum_macro_eigenvalue(all_requested),
        "face_counts": face_counts,
    }
    return arrays, summary


def deterministic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    with zipfile.ZipFile(path, mode="x", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=9) as archive:
        for name in sorted(arrays):
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, np.asarray(arrays[name]), allow_pickle=False)
            info = zipfile.ZipInfo(name + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, buffer.getvalue(), compress_type=zipfile.ZIP_DEFLATED,
                             compresslevel=9)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    for key in ("geometry_specification", "specification", "decision"):
        source = BASE / spec["domain_basis"][key]
        expected = spec["domain_basis"][key + "_sha256"]
        if digest(source) != expected:
            raise ValueError("Domain-basis hash changed: " + str(source))
    for role in ("working", "audit"):
        for suffix in ("mesh", "mesh_coordinates"):
            source = BASE / spec["mesh_policy"][role + "_" + suffix]
            expected = spec["mesh_policy"][role + "_" + suffix + "_sha256"]
            if digest(source) != expected:
                raise ValueError("Protocol mesh artifact hash changed: " + str(source))
    arrays, report = build(spec)
    if report["status"] != "coordinates_frozen":
        raise RuntimeError("Data-design checks failed: " + json.dumps(report["checks"]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    deterministic_npz(args.out, arrays)
    report.update(protocol_id=spec["id"], protocol_sha256=digest(args.spec),
                  generator_sha256=digest(Path(__file__)), design_file=str(args.out),
                  design_sha256=digest(args.out), numpy_version=np.__version__,
                  scipy_version=scipy.__version__, scope=spec["scope"])
    with args.report.open("x") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
