"""Run one restartable material-B data chunk under the frozen protocol."""
from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import sys
import time
from pathlib import Path

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
ROOT = BASE.parent
sys.path.insert(0, str(BASE))

from adaptive_continuation import solve as continuation_solve  # noqa: E402
from run_pilot import field_stats, pf  # noqa: E402
try:  # Script execution and package import use different module roots.
    from .prepare_design import deterministic_npz, digest  # type: ignore
except ImportError:
    from prepare_design import deterministic_npz, digest  # noqa: E402


DEFAULT_SPEC = HERE / "data_protocol_v1.json"
DEFAULT_DESIGN = BASE / "results/data_protocol_design_v1.npz"
DEFAULT_DESIGN_REPORT = BASE / "results/data_protocol_design_v1.json"


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def atomic_state(path: Path, **arrays) -> None:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(buffer.getvalue())
    os.replace(temporary, path)


def morton_order(points: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    unit = np.clip((points-lower)/(upper-lower), 0.0, 1.0)
    integer = np.floor(unit*(2**20-1)).astype(np.uint64)
    keys = np.zeros(len(points), dtype=np.uint64)
    for bit in range(20):
        for axis in range(3):
            keys |= ((integer[:, axis] >> bit) & 1) << (3*bit+axis)
    return np.argsort(keys, kind="stable")


def nearest_order(indices: np.ndarray, points: np.ndarray, lower: np.ndarray,
                  upper: np.ndarray) -> np.ndarray:
    remaining = list(map(int, indices))
    unit = (points-lower)/(upper-lower)
    current = (np.zeros(3)-lower)/(upper-lower)
    ordered = []
    while remaining:
        candidates = unit[remaining]
        local = int(np.argmin(np.sum((candidates-current)**2, axis=1)))
        chosen = remaining.pop(local)
        ordered.append(chosen)
        current = unit[chosen]
    return np.asarray(ordered, dtype=int)


def selection(data, split: str) -> tuple[np.ndarray, list[dict]]:
    if split in ("fit", "validation", "test"):
        key = {"fit": "E_fit", "validation": "E_validation", "test": "E_test"}[split]
        points = np.asarray(data[key])
        metadata = [dict(source_index=i) for i in range(len(points))]
    elif split == "paths":
        paths = np.asarray(data["path_states"])
        names = [str(x) for x in data["path_names"]]
        parameter = np.asarray(data["path_parameter"])
        points, metadata = [], []
        for path_index, name in enumerate(names):
            for step in range(1, len(parameter)):
                points.append(paths[path_index, step])
                metadata.append(dict(source_index=path_index*(len(parameter)-1)+step-1,
                                     path_index=path_index, path_name=name,
                                     path_parameter=float(parameter[step]), path_step=step))
        points = np.asarray(points)
    elif split == "audit":
        points = np.asarray(data["mesh_audit_states"])
        metadata = [dict(source_index=i, original_split=str(data["mesh_audit_split"][i]),
                         original_index=int(data["mesh_audit_index"][i])) for i in range(len(points))]
    elif split == "cold":
        points = np.asarray(data["cold_start_states"])
        metadata = [dict(source_index=i, original_split=str(data["cold_start_split"][i]),
                         original_index=int(data["cold_start_index"][i])) for i in range(len(points))]
    elif split == "reference":
        points = np.asarray(data["E_reference"])
        metadata = [dict(source_index=0)]
    else:
        raise ValueError("Unknown split: " + split)
    return points, metadata


def chunk_plan(data, split: str, chunk_index: int, chunk_count: int,
               lower: np.ndarray, upper: np.ndarray) -> list[dict]:
    points, metadata = selection(data, split)
    if not (0 <= chunk_index < chunk_count):
        raise ValueError("chunk-index must lie in [0, chunk-count)")
    if split == "paths":
        names = [str(x) for x in data["path_names"]]
        if chunk_count != len(names):
            raise ValueError("Path stages require one chunk per declared path")
        chosen = np.flatnonzero([row["path_index"] == chunk_index for row in metadata])
    elif split == "reference":
        if chunk_count != 1:
            raise ValueError("Reference stage requires one chunk")
        chosen = np.array([0])
    else:
        ordered = morton_order(points, lower, upper)
        chunks = np.array_split(ordered, chunk_count)
        chosen = nearest_order(chunks[chunk_index], points, lower, upper)
    return [dict(plan_position=position, global_index=int(index), strain=points[index].tolist(),
                 **metadata[int(index)]) for position, index in enumerate(chosen)]


def state_path(output: Path, position: int) -> Path:
    return output / "partial_states" / f"state_{position:05d}.npz"


def recover(report: dict, output: Path) -> None:
    known = {int(row["plan_position"]): row for row in report["states"]}
    plan = {int(row["plan_position"]): row for row in report["plan"]}
    folder = output / "partial_states"
    if folder.exists():
        for path in sorted(folder.glob("state_*.npz")):
            with np.load(path, allow_pickle=False) as saved:
                row = json.loads(str(saved["record_json"].item()))
                position = int(row["plan_position"])
                if position not in plan or row["global_index"] != plan[position]["global_index"]:
                    raise ValueError("Partial state does not match the frozen plan: " + str(path))
                if position in known and known[position] != row:
                    raise ValueError("Report and partial state disagree: " + str(path))
                known[position] = row
    report["states"] = [known[position] for position in sorted(known)]


def last_success(report: dict, output: Path) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    good = [row for row in report["states"] if row.get("ok")]
    if not good:
        return None, None
    row = max(good, key=lambda value: value["plan_position"])
    with np.load(state_path(output, row["plan_position"]), allow_pickle=False) as saved:
        return np.asarray(saved["strain"]), np.asarray(saved["q"])


def consolidate(report: dict, output: Path) -> Path:
    count = len(report["plan"])
    strain = np.asarray([row["strain"] for row in report["plan"]], dtype=float)
    stress = np.full((count, 3), np.nan)
    tangent = np.full((count, 3, 3), np.nan)
    energy = np.full(count, np.nan)
    minimum_j = np.full(count, np.nan)
    residual = np.full(count, np.nan)
    pk1_l2 = np.full(count, np.nan)
    pk1_max = np.full(count, np.nan)
    available = np.zeros(count, dtype=bool)
    warm_audit_index, warm_audit_q = [], []
    global_index = np.asarray([row["global_index"] for row in report["plan"]], dtype=int)
    records = {row["plan_position"]: row for row in report["states"]}
    for position in range(count):
        row = records[position]
        if not row.get("ok"):
            continue
        with np.load(state_path(output, position), allow_pickle=False) as saved:
            stress[position] = saved["stress"]
            tangent[position] = saved["tangent"]
            energy[position] = saved["energy"]
            minimum_j[position] = saved["minimum_micro_J"]
            residual[position] = saved["relative_reduced_residual"]
            pk1_l2[position] = saved["pk1_l2"]
            pk1_max[position] = saved["pk1_max"]
            available[position] = True
            cold_index = int(saved["cold_audit_index"])
            if cold_index >= 0:
                warm_audit_index.append(cold_index)
                warm_audit_q.append(np.asarray(saved["q"]))
    if warm_audit_q:
        warm_audit_q = np.stack(warm_audit_q)
    else:
        warm_audit_q = np.empty((0, 0), dtype=float)
    destination = output / "labels.npz"
    deterministic_npz(destination, dict(strain=strain, stress=stress, tangent=tangent,
        energy=energy, minimum_micro_J=minimum_j, relative_reduced_residual=residual,
        pk1_l2=pk1_l2, pk1_max=pk1_max, available=available, global_index=global_index,
        warm_audit_index=np.asarray(warm_audit_index, dtype=int), warm_audit_q=warm_audit_q))
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument("--design-report", type=Path, default=DEFAULT_DESIGN_REPORT)
    parser.add_argument("--split", choices=("fit", "validation", "test", "paths", "audit", "cold", "reference"), required=True)
    parser.add_argument("--mesh", choices=("working", "audit"), default="working")
    parser.add_argument("--chunk-index", type=int, required=True)
    parser.add_argument("--chunk-count", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after", type=int, default=0,
                        help="Pause after this many new records; zero means finish the chunk.")
    parser.add_argument("--confirm-full", action="store_true",
                        help="Required to run more than 32 states in one invocation.")
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    design_report = json.loads(args.design_report.read_text())
    if digest(args.spec) != design_report["protocol_sha256"]:
        raise ValueError("Protocol differs from the frozen coordinate report")
    if digest(args.design) != design_report["design_sha256"]:
        raise ValueError("Coordinate file differs from the frozen report")
    mesh_path = BASE / spec["mesh_policy"][args.mesh + "_mesh"]
    if digest(mesh_path) != spec["mesh_policy"][args.mesh + "_mesh_sha256"]:
        raise ValueError("Mesh differs from the frozen protocol")
    if args.split == "audit" and args.mesh != "audit":
        raise ValueError("The mesh-audit split must use the audit mesh")
    if args.mesh == "audit" and args.split != "audit":
        raise ValueError("The audit mesh is reserved for the mesh-audit split")
    with np.load(args.design, allow_pickle=False) as data:
        lower, upper = np.asarray(data["lower"]), np.asarray(data["upper"])
        plan = chunk_plan(data, args.split, args.chunk_index, args.chunk_count, lower, upper)
        cold_states = np.asarray(data["cold_start_states"])
    if not plan:
        raise ValueError("Selected chunk is empty")
    if args.stop_after < 0:
        raise ValueError("stop-after must be nonnegative")
    if len(plan) > 32 and args.stop_after == 0 and not args.confirm_full:
        raise RuntimeError("Refuse to run more than 32 states without --confirm-full")

    output = args.out.resolve()
    report_path = output / "report.json"
    driver_hash = digest(Path(__file__))
    if args.resume:
        report = json.loads(report_path.read_text())
        if report["driver_sha256"] != driver_hash or report["protocol_sha256"] != digest(args.spec):
            raise ValueError("Cannot resume with changed driver or protocol")
        if report["plan"] != plan or report["mesh_sha256"] != digest(mesh_path):
            raise ValueError("Cannot resume a different plan or mesh")
        if report["status"] == "complete":
            print("Chunk already complete:", output)
            return 0 if report["all_available"] else 1
        recover(report, output)
    else:
        output.mkdir(parents=True, exist_ok=False)
        (output / "partial_states").mkdir()
        (output / "run_data_stage_used.py").write_bytes(Path(__file__).read_bytes())
        sources = [args.spec, args.design, args.design_report, Path(__file__),
                   BASE / "adaptive_continuation.py", BASE / "run_pilot.py",
                   ROOT / "00_rve/periodic_fom.py",
                   ROOT.parent / "RVE_NeoHookean_Homogenization/core/fom_solver_rve.py",
                   ROOT.parent / "RVE_NeoHookean_Homogenization/core/StructuralMaterials.json", mesh_path]
        report = dict(status="running", protocol_id=spec["id"], protocol_sha256=digest(args.spec),
            design_sha256=digest(args.design), driver_sha256=driver_hash, mesh=args.mesh,
            mesh_sha256=digest(mesh_path), split=args.split, chunk_index=args.chunk_index,
            chunk_count=args.chunk_count, plan=plan, states=[], sources_sha256={str(p): digest(p) for p in sources},
            started_utc=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()), elapsed_seconds=0.0,
            scope=spec["scope"])
        atomic_json(report_path, report)

    completed = {row["plan_position"] for row in report["states"]}
    pending = [row for row in plan if row["plan_position"] not in completed]
    limit = len(pending) if args.stop_after == 0 else min(args.stop_after, len(pending))
    previous, q_previous = (None, None) if args.split in ("cold", "reference") else last_success(report, output)
    timer = time.perf_counter()
    sys.path.insert(0, str(ROOT.parent / "RVE_NeoHookean_Homogenization/core"))
    from _material_law_guard_claude import true_neo_hookean_active
    with true_neo_hookean_active():
        rve = pf.PeriodicRVE(mesh_path.with_suffix(""), cell_area=4.0)
        for planned in pending[:limit]:
            target = np.asarray(planned["strain"], dtype=float)
            cold_matches = np.flatnonzero(np.all(cold_states == target, axis=1))
            if len(cold_matches) > 1:
                raise ValueError("Duplicate state in frozen cold-start selection")
            cold_index = int(cold_matches[0]) if len(cold_matches) else -1
            independent = args.split in ("cold", "reference")
            start = None if independent else previous
            q_start = None if independent else q_previous
            record = dict(planned, ok=False)
            state_timer = time.perf_counter()
            try:
                stress, q, attempts = continuation_solve(rve, target,
                    spec["solver_policy"]["maximum_engineering_strain_vector_increment"],
                    spec["solver_policy"]["minimum_increment"], start, q_start)
                need_tangent = (args.split in ("test", "paths", "audit", "cold", "reference")
                                or cold_index >= 0)
                if need_tangent:
                    stress, tangent, q = rve.stress_and_tangent_consistent(
                        target, u_ind_init=q, E_start=target, return_state=True)
                else:
                    tangent = np.full((3, 3), np.nan)
                fields, _u, _pk1 = field_stats(rve, target, q)
                if fields["relative_reduced_residual"] > spec["solver_policy"]["relative_residual_tolerance"]:
                    raise RuntimeError("Final state exceeds frozen residual tolerance")
                energy = float(rve.homogenized_energy())
                record.update(ok=True, stress=np.asarray(stress).tolist(), tangent=np.asarray(tangent).tolist(),
                              energy=energy, fields=fields, attempts=attempts,
                              cold_audit_index=cold_index,
                              elapsed_seconds=time.perf_counter()-state_timer)
                atomic_state(state_path(output, planned["plan_position"]), strain=target, q=q,
                    stress=stress, tangent=tangent, energy=energy,
                    minimum_micro_J=fields["min_micro_J"],
                    relative_reduced_residual=fields["relative_reduced_residual"],
                    pk1_l2=fields["pk1_l2"], pk1_max=fields["pk1_max"],
                    cold_audit_index=cold_index,
                    record_json=json.dumps(record, allow_nan=False))
                if not independent:
                    previous, q_previous = target, q
            except Exception as error:
                record.update(error=repr(error), attempts=getattr(error, "attempts", []),
                              elapsed_seconds=time.perf_counter()-state_timer)
            report["states"].append(record)
            report["states"].sort(key=lambda row: row["plan_position"])
            report["elapsed_seconds"] += record["elapsed_seconds"]
            atomic_json(report_path, report)
            print(args.split, planned["plan_position"], planned["global_index"], record["ok"], flush=True)
    report["last_invocation_seconds"] = time.perf_counter()-timer
    if len(report["states"]) < len(plan):
        report["status"] = "paused"
        atomic_json(report_path, report)
        return 0
    labels = consolidate(report, output)
    report["status"] = "complete"
    report["all_available"] = all(row.get("ok") for row in report["states"])
    report["labels_sha256"] = digest(labels)
    report["rejected_increment_count"] = sum(not attempt["ok"] for row in report["states"]
                                               for attempt in row.get("attempts", []))
    atomic_json(report_path, report)
    shutil.rmtree(output / "partial_states")
    return 0 if report["all_available"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
