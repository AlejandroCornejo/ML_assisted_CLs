"""Consolidate predeclared checks without rerunning FOM or changing thresholds."""
import argparse
import json
from pathlib import Path
from run_preflight import differences, digest, old_states


def summarize(reference_dir, check_dir):
    reference = json.loads((reference_dir / "report.json").read_text())
    check = json.loads((check_dir / "report.json").read_text())
    if reference["status"] != "complete" or check["status"] != "complete":
        raise RuntimeError("Wait for both preflight stages")
    if reference["spec_sha256"] != check["spec_sha256"]:
        raise ValueError("Stages use different predeclared specifications")
    spec = check["spec"]
    original, parent, _folder, _retry = old_states(spec)
    expected_strains = {f"pilot__{path}__{fraction:g}":row["strain"]
                        for (path, fraction), row in original.items()}
    domain_strains = {"domain__"+name:e for name, e in spec["domain_states"].items()}
    expected_strains.update(domain_strains)
    lookup = {s["name"]:s for s in reference["states"] if s["ok"]}
    comparisons, checks = [], []
    def exact_targets(report, targets):
        rows = report["states"]
        return (len(rows) == len(targets) and {s["name"] for s in rows} == set(targets)
                and all(s["ok"] and s["strain"] == targets[s["name"]] for s in rows))
    checks.append(dict(check="all targets reached on both stages",
        passed=bool(reference["numerical_states_complete"] and check["numerical_states_complete"]
                    and exact_targets(reference, domain_strains) and exact_targets(check, expected_strains))))
    for row in check["states"]:
        if not row["ok"]:
            continue
        if row["name"].startswith("domain__"):
            ref = lookup.get(row["name"])
        else:
            _prefix, path, fraction = row["name"].split("__")
            ref = original[path, float(fraction)]
        if ref is not None:
            comparisons.append(dict(name=row["name"], **differences(ref, row)))
    expected = 32+len(spec["domain_states"])
    checks.append(dict(check="all original and domain comparisons present",
        passed=len(comparisons) == expected and {c["name"] for c in comparisons} == set(expected_strains)))
    thresholds = dict(spec["reference_output_tolerances"],
                      pk1_l2=parent["parent_spec"]["screening_tolerances"]["mesh_pk1_l2_relative_difference"],
                      pk1_max=parent["parent_spec"]["screening_tolerances"]["mesh_pk1_max_relative_difference"])
    for key, threshold in thresholds.items():
        worst = max(comparisons, key=lambda c:c[key], default=None)
        value = None if worst is None else worst[key]
        checks.append(dict(check="reference/check: "+key, threshold=threshold, value=value,
                           worst_state=None if worst is None else worst["name"],
                           passed=value is not None and value <= threshold))
    all_states = [s for r in (reference, check) for s in r["states"] if s["ok"]]
    checks.append(dict(check="residuals, positive quadrature J and endpoint boundary screen",
        passed=all(s["fields"]["relative_reduced_residual"] <= 1e-7 and s["fields"]["min_micro_J"] > 0
                   and s["boundary"]["min_polygon_gap"] > 1e-6 and not s["boundary"]["self_intersection"]
                   and s["boundary"]["max_periodic_jump_error"] < 1e-8 for s in all_states)))
    derivatives = [d for s in all_states for d in s.get("derivatives", [])]
    checks.append(dict(check="all selected derivative checks present",
                       passed=len(derivatives) == 2*len(spec["derivative_states"])*len(spec["fd_steps"])))
    for key in ("energy_gradient_relative_error", "tangent_relative_error", "fd_tangent_relative_asymmetry"):
        value = max((d[key] for d in derivatives), default=None)
        threshold = parent["parent_spec"]["screening_tolerances"][key]
        checks.append(dict(check="finite-difference "+key, value=value, threshold=threshold,
                           passed=value is not None and value <= threshold))
    return dict(status="complete", summary_source_sha256=digest(Path(__file__)),
        reference_report_sha256=digest(reference_dir / "report.json"),
        check_report_sha256=digest(check_dir / "report.json"), spec=spec, checks=checks,
        parent_y_compression_report_sha256=digest(_retry / "report.json"),
        passed=all(c["passed"] for c in checks), comparisons=comparisons,
        minimum_sampled_rank_one_curvature=min((s["sampled_min_rank_one_curvature"] for s in all_states), default=None),
        rejected_increments=sum(not a["ok"] for s in all_states for a in s["attempts"]),
        reference_elements=reference["geometry"]["n_elements"], check_elements=check["geometry"]["n_elements"],
        minimum_micro_J=min((s["fields"]["min_micro_J"] for s in all_states), default=None),
        minimum_polygon_gap=min((s["boundary"]["min_polygon_gap"] for s in all_states), default=None),
        scope="Finite numerical preflight: not full-box/contact/stability/uniqueness proof or exact-solution "
        "error bound. Relative mesh differences use the denser check-mesh norm in the denominator. "
        "Previous failed mesh screens remain unchanged. Rank-one sampling is reported separately, not a certificate.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--check", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    result = summarize(args.reference, args.check)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print("Preflight passed:", result["passed"])
    print(result["checks"])
    raise SystemExit(0 if result["passed"] else 1)
