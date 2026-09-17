"""Verify and assemble all frozen material-B campaign chunks."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    from .prepare_design import deterministic_npz, digest
    from .run_campaign import jobs
    from .run_data_stage import selection
except ImportError:
    from prepare_design import deterministic_npz, digest
    from run_campaign import jobs
    from run_data_stage import selection


HERE = Path(__file__).resolve().parent
BASE = HERE.parent
sys.path.insert(0, str(BASE))
from audit_pilot import rank_one_screen  # noqa: E402
SPEC = HERE / "data_protocol_v1.json"
DESIGN = BASE / "results/data_protocol_design_v1.npz"
DESIGN_REPORT = BASE / "results/data_protocol_design_v1.json"
DOMAIN_SPEC = BASE / "asymmetric_box_spec.json"
PILOT_SPEC = BASE / "pilot_spec.json"
YOUNG = 1.628e9


def relative(first, second, floor=1.0) -> float:
    return float(np.linalg.norm(np.asarray(first)-np.asarray(second))
                 / max(np.linalg.norm(second), floor))


def load_group(campaign: Path, prefix: str, count: int, expected: np.ndarray) -> tuple[dict, dict]:
    rows, report_hashes, label_hashes = [], {}, {}
    for index in range(count):
        folder = campaign / "chunks" / f"{prefix}_{index:03d}"
        report_path, labels_path = folder / "report.json", folder / "labels.npz"
        report = json.loads(report_path.read_text())
        if report["status"] != "complete" or not report["all_available"]:
            raise RuntimeError("Incomplete or unavailable chunk: " + str(folder))
        if digest(labels_path) != report["labels_sha256"]:
            raise ValueError("Chunk label hash changed: " + str(folder))
        for source, expected_hash in report["sources_sha256"].items():
            if digest(Path(source)) != expected_hash:
                raise ValueError("Chunk source changed: " + source)
        with np.load(labels_path, allow_pickle=False) as saved:
            rows.append({name: np.asarray(saved[name]) for name in saved.files})
        report_hashes[str(report_path)] = digest(report_path)
        label_hashes[str(labels_path)] = digest(labels_path)
    ordinary = ("strain", "stress", "tangent", "energy", "minimum_micro_J",
                "relative_reduced_residual", "minimum_deformed_polygon_gap",
                "deformed_polygon_self_intersection", "max_periodic_jump_error",
                "pk1_l2", "pk1_max", "available", "global_index")
    merged = {name: np.concatenate([row[name] for row in rows], axis=0) for name in ordinary}
    order = np.argsort(merged["global_index"])
    merged = {name: values[order] for name, values in merged.items()}
    if not np.array_equal(merged["global_index"], np.arange(len(expected))):
        raise ValueError("Chunks do not cover each requested index exactly once: " + prefix)
    if not np.array_equal(merged["strain"], expected) or not np.all(merged["available"]):
        raise ValueError("Assembled strains/availability differ from frozen design: " + prefix)
    audit_index = np.concatenate([row["warm_audit_index"] for row in rows])
    audit_q_rows = [row["warm_audit_q"] for row in rows if len(row["warm_audit_index"])]
    merged["warm_audit_index"] = audit_index
    merged["warm_audit_q"] = (np.concatenate(audit_q_rows, axis=0) if audit_q_rows
                               else np.empty((0, 0), dtype=float))
    return merged, dict(report_sha256=report_hashes, labels_sha256=label_hashes)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(SPEC.read_text())
    design_report = json.loads(DESIGN_REPORT.read_text())
    manifest_path = args.campaign / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["status"] != "complete" or manifest["failed_jobs"]:
        raise RuntimeError("Campaign has not completed without failed jobs")
    if (manifest["protocol_sha256"] != digest(SPEC)
            or manifest["design_sha256"] != digest(DESIGN)
            or manifest["driver_sha256"] != digest(HERE / "run_data_stage.py")
            or manifest["orchestrator_sha256"] != digest(HERE / "run_campaign.py")
            or manifest["plan"] != jobs()):
        raise ValueError("Campaign manifest differs from frozen protocol/design/plan")
    if design_report["protocol_sha256"] != digest(SPEC) or design_report["design_sha256"] != digest(DESIGN):
        raise ValueError("Frozen coordinate report changed")
    with np.load(DESIGN, allow_pickle=False) as data:
        expected = {split: selection(data, split)[0]
                    for split in ("fit", "validation", "test", "paths", "audit", "cold", "reference")}
        path_names = np.asarray(data["path_names"])
        path_parameter = np.asarray(data["path_parameter"])
        cold_split = np.asarray(data["cold_start_split"])
        cold_original_index = np.asarray(data["cold_start_index"])
    definitions = {
        "fit": (64, "fit"), "validation": (8, "validation"), "test": (8, "test"),
        "paths": (10, "paths"), "audit_working": (4, "audit"),
        "audit_check": (4, "audit"), "cold": (4, "cold"), "reference": (1, "reference")}
    groups, provenance = {}, {}
    for prefix, (count, split) in definitions.items():
        groups[prefix], provenance[prefix] = load_group(args.campaign, prefix, count, expected[split])

    checks = []
    def add(name, passed, **values):
        checks.append(dict(check=name, passed=bool(passed), **values))
    main_groups = [groups[name] for name in ("fit", "validation", "test", "paths", "reference")]
    add("all requested working labels finite",
        all(np.isfinite(group["stress"]).all() and np.isfinite(group["energy"]).all()
            for group in main_groups))
    all_groups = list(groups.values())
    minimum_j = min(float(group["minimum_micro_J"].min()) for group in all_groups)
    maximum_residual = max(float(group["relative_reduced_residual"].max()) for group in all_groups)
    physical = spec["physical_screen"]
    add("working fields have positive quadrature J",
        minimum_j > physical["minimum_micro_J_exclusive"], value=minimum_j,
        threshold=physical["minimum_micro_J_exclusive"])
    add("working residual tolerance", maximum_residual <= spec["solver_policy"]["relative_residual_tolerance"],
        value=maximum_residual, threshold=spec["solver_policy"]["relative_residual_tolerance"])
    minimum_gap = min(float(group["minimum_deformed_polygon_gap"].min()) for group in all_groups)
    maximum_jump = max(float(group["max_periodic_jump_error"].max()) for group in all_groups)
    self_intersection_count = sum(int(np.count_nonzero(
        group["deformed_polygon_self_intersection"])) for group in all_groups)
    add("all generated states pass the deformed polygon-gap screen",
        minimum_gap > physical["minimum_deformed_polygon_gap_exclusive"], value=minimum_gap,
        threshold=physical["minimum_deformed_polygon_gap_exclusive"])
    add("all generated states pass the exact periodic-jump screen",
        maximum_jump < physical["maximum_periodic_jump_error_exclusive"], value=maximum_jump,
        threshold=physical["maximum_periodic_jump_error_exclusive"])
    add("no generated cavity polygon self-intersects",
        self_intersection_count == 0 or physical["allow_deformed_polygon_self_intersection"],
        count=self_intersection_count)
    minimum_energy = min(float(group["energy"].min()) for group in main_groups)
    energy_scale = max(float(np.abs(group["energy"]).max()) for group in main_groups)
    add("stored Neo-Hookean effective energies nonnegative within numerical tolerance",
        minimum_energy >= -1e-12*max(energy_scale, 1.0), value=minimum_energy,
        tolerance=-1e-12*max(energy_scale, 1.0))

    pilot = json.loads(PILOT_SPEC.read_text())["screening_tolerances"]
    reference = groups["reference"]
    reference_stress = float(np.linalg.norm(reference["stress"][0])/YOUNG)
    reference_energy = float(abs(reference["energy"][0])/YOUNG)
    add("reference stress", reference_stress <= pilot["reference_stress_over_young"],
        value=reference_stress, threshold=pilot["reference_stress_over_young"])
    add("reference energy", reference_energy <= pilot["reference_energy_over_young"],
        value=reference_energy, threshold=pilot["reference_energy_over_young"])
    tangents_finite = (np.isfinite(groups["test"]["tangent"]).all()
        and np.isfinite(groups["paths"]["tangent"]).all()
        and np.isfinite(groups["audit_working"]["tangent"]).all()
        and np.isfinite(groups["audit_check"]["tangent"]).all()
        and np.isfinite(groups["cold"]["tangent"]).all()
        and np.isfinite(reference["tangent"]).all())
    add("all required tangents finite", tangents_finite)
    rank_one_rows = []
    if tangents_finite:
        for group_name in ("test", "paths", "audit_working", "audit_check", "cold", "reference"):
            group = groups[group_name]
            for index in range(len(group["strain"])):
                rank_one_rows.append(dict(group=group_name, index=index,
                    curvature=rank_one_screen(group["strain"][index], group["stress"][index],
                                              group["tangent"][index])))
        worst_rank_one = min(rank_one_rows, key=lambda row: row["curvature"])
        add("sampled rank-one curvature at every tangent-labelled state",
            worst_rank_one["curvature"] >= 0.0, value=worst_rank_one["curvature"], threshold=0.0,
            worst_group=worst_rank_one["group"], worst_index=worst_rank_one["index"])
    else:
        worst_rank_one = dict(curvature=None, group=None, index=None)
        add("sampled rank-one curvature at every tangent-labelled state", False,
            value=None, threshold=0.0, worst_group=None, worst_index=None)

    domain = json.loads(DOMAIN_SPEC.read_text())
    mesh_rows = []
    for index, (working, check) in enumerate(zip(groups["audit_working"]["strain"],
                                                  groups["audit_check"]["strain"])):
        if not np.array_equal(working, check):
            raise ValueError("Working/check audit strain mismatch")
        mesh_rows.append(dict(index=index,
            stress=relative(groups["audit_working"]["stress"][index], groups["audit_check"]["stress"][index]),
            tangent=relative(groups["audit_working"]["tangent"][index], groups["audit_check"]["tangent"][index]),
            energy=relative(groups["audit_working"]["energy"][index], groups["audit_check"]["energy"][index]),
            pk1_l2=relative(groups["audit_working"]["pk1_l2"][index], groups["audit_check"]["pk1_l2"][index]),
            pk1_max=relative(groups["audit_working"]["pk1_max"][index], groups["audit_check"]["pk1_max"][index])))
    mesh_limits = dict(domain["reference_output_tolerances"], **domain["field_statistic_tolerances"])
    for key, threshold in mesh_limits.items():
        worst = max(mesh_rows, key=lambda row: row[key])
        add("campaign mesh audit: " + key, worst[key] <= threshold, value=worst[key],
            threshold=threshold, worst_index=worst["index"])

    warm_q = {}
    for group_name in ("fit", "validation", "test"):
        group = groups[group_name]
        for index, q in zip(group["warm_audit_index"], group["warm_audit_q"]):
            if int(index) in warm_q:
                raise ValueError("Duplicate warm cold-audit displacement")
            warm_q[int(index)] = q
    cold_q = {int(index): q for index, q in zip(groups["cold"]["warm_audit_index"],
                                                groups["cold"]["warm_audit_q"])}
    if set(warm_q) != set(range(len(expected["cold"]))) or set(cold_q) != set(warm_q):
        raise ValueError("Missing/duplicate warm or cold displacement audit")
    cold_rows = []
    split_map = {"fit_volume": "fit", "validation": "validation", "test": "test"}
    for cold_index in range(len(expected["cold"])):
        source_name = split_map[str(cold_split[cold_index])]
        source_index = int(cold_original_index[cold_index])
        warm, cold = groups[source_name], groups["cold"]
        cold_rows.append(dict(index=cold_index,
            stress=relative(warm["stress"][source_index], cold["stress"][cold_index]),
            tangent=relative(warm["tangent"][source_index], cold["tangent"][cold_index]),
            energy=relative(warm["energy"][source_index], cold["energy"][cold_index]),
            displacement=relative(warm_q[cold_index], cold_q[cold_index], 1e-12)))
    cold_tolerance = spec["solver_policy"]["cold_start_agreement_tolerance"]
    for key in ("stress", "tangent", "energy", "displacement"):
        worst = max(cold_rows, key=lambda row: row[key])
        add("cold-start agreement: " + key, worst[key] <= cold_tolerance, value=worst[key],
            threshold=cold_tolerance, worst_index=worst["index"])

    arrays = {}
    for name in ("fit", "validation", "test", "paths", "reference"):
        group = groups[name]
        suffix = name
        arrays.update({"E_"+suffix: group["strain"], "S_"+suffix: group["stress"],
                       "W_"+suffix: group["energy"], "D_"+suffix: group["tangent"],
                       "min_micro_J_"+suffix: group["minimum_micro_J"],
                       "min_polygon_gap_"+suffix: group["minimum_deformed_polygon_gap"],
                       "max_periodic_jump_"+suffix: group["max_periodic_jump_error"]})
    arrays.update(path_names=path_names, path_parameter=path_parameter,
                  E_mesh_audit=groups["audit_working"]["strain"],
                  S_mesh_audit_working=groups["audit_working"]["stress"],
                  D_mesh_audit_working=groups["audit_working"]["tangent"],
                  W_mesh_audit_working=groups["audit_working"]["energy"],
                  S_mesh_audit_check=groups["audit_check"]["stress"],
                  D_mesh_audit_check=groups["audit_check"]["tangent"],
                  W_mesh_audit_check=groups["audit_check"]["energy"],
                  E_cold=groups["cold"]["strain"], S_cold=groups["cold"]["stress"],
                  D_cold=groups["cold"]["tangent"], W_cold=groups["cold"]["energy"])
    deterministic_npz(args.out, arrays)
    result = dict(status="complete", passed=all(row["passed"] for row in checks), checks=checks,
                  counts={name: len(groups[name]["strain"]) for name in groups},
                  mesh_comparisons=mesh_rows, cold_comparisons=cold_rows,
                  minimum_micro_J=minimum_j, maximum_relative_residual=maximum_residual,
                  minimum_deformed_polygon_gap=minimum_gap,
                  maximum_periodic_jump_error=maximum_jump,
                  deformed_polygon_self_intersection_count=self_intersection_count,
                  minimum_sampled_rank_one_curvature=worst_rank_one["curvature"],
                  rank_one_screen=rank_one_rows, minimum_energy=minimum_energy,
                  protocol_sha256=digest(SPEC),
                  design_sha256=digest(DESIGN), campaign_manifest_sha256=digest(manifest_path),
                  assembled_data=str(args.out), assembled_data_sha256=digest(args.out),
                  assembler_sha256=digest(Path(__file__)), chunk_provenance=provenance,
                  scope=spec["scope"])
    with args.report.open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps(dict(passed=result["passed"], checks=checks), indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
