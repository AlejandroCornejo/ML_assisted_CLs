"""Assess the frozen expanded-box screen; no FOM solves or threshold changes."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


FD_KEYS = ("energy_gradient_relative_error", "tangent_relative_error", "fd_tangent_relative_asymmetry")
COLD_KEYS = ("stress", "tangent", "energy", "node_displacement")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def relative(a, b):
    return float(np.linalg.norm(np.asarray(a)-b) / max(np.linalg.norm(b), 1.))


def differences(reference, check):
    values = {k:relative(reference[k], check[k]) for k in ("stress", "tangent", "energy")}
    values.update({k:relative(reference["fields"][k], check["fields"][k]) for k in ("pk1_l2", "pk1_max")})
    return values


def assess(reference, check):
    if any(r["status"] != "complete" for r in (reference, check)):
        raise RuntimeError("Wait for both completed stages")
    if (reference["spec_sha256"] != check["spec_sha256"] or reference["spec"] != check["spec"]
            or reference["helper_sha256"] != check["helper_sha256"]):
        raise ValueError("Stages use different frozen specifications or continuation helpers")
    if reference["mesh_role"] != "reference" or check["mesh_role"] != "check":
        raise ValueError("Expected one reference and one denser-check stage")
    spec, checks = reference["spec"], []
    targets = spec["states"]
    def add(name, passed, **values):
        checks.append(dict(check=name, passed=bool(passed), **values))
    def exact_targets(report):
        rows = report["states"]
        return (report["all_targets_reached"] and len(rows) == len(targets)
                and {s["name"] for s in rows} == set(targets)
                and all(s["ok"] and s["strain"] == targets[s["name"]]
                        and s["origin"] == ("reused" if s["name"] in spec["reuse_states"] else "new") for s in rows))
    add("all exact targets reached on both meshes", exact_targets(reference) and exact_targets(check))
    ref_lookup = {s["name"]:s for s in reference["states"] if s["ok"]}
    comparisons = [dict(name=s["name"], **differences(ref_lookup[s["name"]], s))
                   for s in check["states"] if s["ok"] and s["name"] in ref_lookup]
    add("all unique target comparisons present", len(comparisons) == len(targets)
        and {c["name"] for c in comparisons} == set(targets))
    for key, threshold in dict(spec["reference_output_tolerances"], **spec["field_statistic_tolerances"]).items():
        worst = max(comparisons, key=lambda c:c[key], default=None)
        value = None if worst is None else worst[key]
        add("reference/check: "+key, value is not None and value <= threshold,
            value=value, threshold=threshold, worst_state=None if worst is None else worst["name"])
    rows = [(r["mesh_role"], s) for r in (reference, check) for s in r["states"] if s["ok"]]
    def physical(s):
        f, b = s["fields"], s["boundary"]
        return (s["physical_screen_passed"] and f["relative_reduced_residual"] <= spec["relative_residual_tolerance"]
                and f["min_micro_J"] > 0 and b["min_polygon_gap"] > spec["stop_polygon_gap"]
                and not b["self_intersection"] and b["max_periodic_jump_error"] < spec["periodic_jump_tolerance"]
                and s["sampled_min_rank_one_curvature"] >= spec["minimum_sampled_rank_one_curvature"])
    add("residual, positive quadrature J, endpoint polygons and sampled rank-one curvature", all(physical(s) for _, s in rows))
    fd = [(role, s["name"], d) for role, s in rows for d in s.get("derivatives", [])]
    expected_fd = {(role, name, step) for role in ("reference", "check")
                   for name in spec["derivative_states"] for step in spec["fd_steps"]}
    add("all unique selected derivative checks present", len(fd) == len(expected_fd)
        and {(role, name, d["step"]) for role, name, d in fd} == expected_fd)
    for key in FD_KEYS:
        value = max((d[key] for _, _, d in fd), default=None)
        add("finite-difference: "+key, value is not None and value <= spec["derivative_tolerance"],
            value=value, threshold=spec["derivative_tolerance"])
    cold = [(r["mesh_role"], c) for r in (reference, check) for c in r["cold_checks"]]
    expected_cold = {(role, name) for role in ("reference", "check") for name in spec["cold_check_states"]}
    add("all unique independent zero-start checks present", len(cold) == len(expected_cold)
        and {(role, c["name"]) for role, c in cold} == expected_cold
        and all(c["ok"] and c["strain"] == targets[c["name"]] for _, c in cold))
    for key in COLD_KEYS:
        values = [c["differences"][key] for _, c in cold if c["ok"]]
        value = max(values, default=None)
        add("independent zero-start agreement: "+key,
            len(values) == len(expected_cold) and value is not None and value <= spec["cold_agreement_tolerance"],
            value=value, threshold=spec["cold_agreement_tolerance"])
    cold_fields = [c["fields"] for _, c in cold if c["ok"]]
    add("independent zero-start residual and positive quadrature J",
        len(cold_fields) == len(expected_cold) and all(f["min_micro_J"] > 0
        and f["relative_reduced_residual"] <= spec["relative_residual_tolerance"] for f in cold_fields))
    fields = [s["fields"] for _, s in rows]+cold_fields
    attempts = [a for _, s in rows for a in s["attempts"]]+[a for _, c in cold for a in c.get("attempts", [])]
    return dict(status="complete", passed=all(c["passed"] for c in checks), spec=spec, checks=checks,
        comparisons=comparisons, comparison_count=len(comparisons), derivative_count=len(fd), cold_check_count=len(cold),
        new_target_count=sum(s["origin"] == "new" for _, s in rows),
        reused_target_count=sum(s["origin"] == "reused" for _, s in rows),
        reference_elements=reference["n_elements"], check_elements=check["n_elements"],
        rejected_increments=sum(not a["ok"] for a in attempts),
        minimum_micro_J=min((f["min_micro_J"] for f in fields), default=None),
        maximum_relative_residual=max((f["relative_reduced_residual"] for f in fields), default=None),
        minimum_polygon_gap=min((s["boundary"]["min_polygon_gap"] for _, s in rows), default=None),
        maximum_periodic_jump_error=max((s["boundary"]["max_periodic_jump_error"] for _, s in rows), default=None),
        minimum_sampled_rank_one_curvature=min((s["sampled_min_rank_one_curvature"] for _, s in rows), default=None),
        failed_stage_records={r["mesh_role"]:dict(failures=r["failures"], screen_failures=r["screen_failures"])
                              for r in (reference, check)},
        scope="Finite numerical preflight, not full-box approval, exact-solution error bound or stability/contact/uniqueness proof. "
              "Mesh differences use the denser-check norm (floor 1 in recorded units) in the denominator. Microscopic RMS "
              "and peak differences compare scalar statistics, not pointwise fields. Cold checks assess selected branch agreement only.")


def summarize(reference_dir, check_dir):
    reports = [json.loads((d / "report.json").read_text()) for d in (reference_dir, check_dir)]
    result = assess(*reports)
    field_hashes = {}
    for folder, report in zip((reference_dir, check_dir), reports):
        if digest(folder / "spec.json") != report["spec_sha256"]:
            raise ValueError("Frozen output specification changed")
        if json.loads((folder / "spec.json").read_text()) != report["spec"]:
            raise ValueError("Report and frozen specification differ")
        if not report["sources_sha256"]:
            raise ValueError("Missing source provenance")
        for name, expected in report["sources_sha256"].items():
            if digest(name) != expected:
                raise ValueError("Input/source changed: "+name)
        driver = next(name for name in report["sources_sha256"] if name.endswith("/run_box_extension.py"))
        if digest(folder / "run_box_extension_used.py") != report["sources_sha256"][driver]:
            raise ValueError("Used-driver snapshot differs")
        for s in report["states"]:
            if s["ok"]:
                field_hashes[s["fields_file"]] = digest(s["fields_file"])
                with np.load(s["fields_file"]) as f:
                    if not (np.array_equal(f["strain"], s["strain"]) and np.array_equal(f["stress"], s["stress"])
                            and np.array_equal(f["tangent"], s["tangent"])):
                        raise ValueError("Saved fields/report mismatch: "+s["name"])
    result.update(summary_source_sha256=digest(__file__),
                  reference_report_sha256=digest(reference_dir / "report.json"),
                  check_report_sha256=digest(check_dir / "report.json"), fields_sha256=field_hashes)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--check", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.reference, args.check)
    with args.out.open("x") as f:
        f.write(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps(dict(passed=result["passed"], checks=result["checks"]), indent=2))
    raise SystemExit(0 if result["passed"] else 1)
