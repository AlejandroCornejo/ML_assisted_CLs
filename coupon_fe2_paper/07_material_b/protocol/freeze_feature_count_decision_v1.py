"""Record the validation-only reporting decision for the MC-RVE count sweep.

The decision records a representative feature count for the detailed
fixed/learned comparison after the full declared sensitivity study is terminal.
It reads the frozen validation audit only and cannot access test or path arrays.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path

from protocol.select_features import BASE


AUDIT = (BASE / "results/feature_count_analysis_v1/validation_audit_v1"
         / "validation_summary.json")
OUTPUT = BASE / "results/feature_count_analysis_v1/m06_reporting_decision_v1"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False,
                                     dir=path.parent, prefix=f".{path.name}.") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def median_score(summary: dict, model: str, count: int) -> float:
    row = next((row for row in summary["grouped_statistics"]
                if row["model"] == model and row["feature_count"] == count), None)
    if row is None:
        raise ValueError(f"Missing {model} at m={count}")
    return float(row["median_validation_score"])


def freeze(output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite frozen decision: {output}")
    summary = json.loads(AUDIT.read_text(encoding="utf-8"))
    if (summary.get("audit_id") != "multicavity_feature_count_validation_audit_v1"
            or summary.get("counts") != [2, 4, 6, 8, 16, 24, 32]
            or len(summary.get("rows", ())) != 84):
        raise ValueError("Unexpected count-sweep validation audit")
    required = {(model, count, seed) for model in summary["models"]
                for count in summary["counts"] for seed in summary["seeds"]}
    found = {(row["model"], row["feature_count"], row["seed"]) for row in summary["rows"]}
    if found != required:
        raise ValueError("Validation audit is incomplete")

    scores = {model: {str(count): median_score(summary, model, count)
                      for count in summary["counts"]}
              for model in summary["models"]}
    decision = dict(
        status="frozen_before_independent_evaluation",
        decision_id="multicavity_m06_reporting_decision_v1",
        created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        validation_audit=str(AUDIT),
        validation_audit_sha256=digest(AUDIT),
        selected_feature_count=6,
        evaluation_scope="All 12 constrained m=6 fits: four model variants and seeds 16, 29, and 47.",
        decision_basis=(
            "Validation-only sensitivity study. m=6 is the first declared count at which both learned "
            "cores enter the approximately 1e-7 validation-error regime after the sharp reduction from m=4; "
            "increasing m beyond 6 yields comparatively small and, for ICKAN, nonmonotone changes. "
            "The fixed variants remain near 1e-5 at m=6, making this the clearest controlled fixed/learned "
            "comparison point."),
        validation_median_scores=scores,
        test_or_path_labels_loaded=False,
        test_or_path_labels_used_for_selection=False,
        limitation=(
            "This reporting choice is an interpretation of the completed validation sensitivity study, not a "
            "claim that the entire seven-count grid was prospectively fixed. The m=2,4,6 block was an explicit "
            "validation-only amendment replacing unstarted m=40 fits."),
    )
    output.mkdir(parents=True)
    atomic_json(output / "decision.json", decision)
    return decision


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    result = freeze(args.output.resolve())
    print(json.dumps(dict(status=result["status"], feature_count=result["selected_feature_count"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
