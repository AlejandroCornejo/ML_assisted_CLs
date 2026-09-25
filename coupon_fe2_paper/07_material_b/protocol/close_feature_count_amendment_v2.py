"""Finalize the feature-count amendment after all child fits have terminated.

The amendment controller records job completion itself during ordinary runs.
This closeout utility exists for the recoverable case in which that controller
is interrupted after workers have written valid final artifacts but before its
status receipt has been updated.  It never trains, resumes, or evaluates a
model; it validates the declared 38 amendment jobs and atomically records their
terminal state.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from protocol.run_feature_count_amendment_v2 import (
    AMENDMENT_RULE,
    OUTPUT,
    _sha,
    _verify_completed,
)
from protocol.train_feature_count_sweep import RULE as LEGACY_RULE
from protocol.train_material_b import _atomic_json


def close(output: Path, amendment_rule: Path) -> dict:
    """Validate every declared job and replace stale running states by results."""
    status_path = output / "campaign_status.json"
    campaign = json.loads(status_path.read_text(encoding="utf-8"))
    if campaign.get("protocol_id") != "material_B_feature_count_amendment_v2":
        raise ValueError("Unexpected amendment campaign")
    legacy_sha = _sha(LEGACY_RULE)
    amendment_sha = _sha(amendment_rule)
    if (campaign.get("legacy_rule_sha256") != legacy_sha
            or campaign.get("amendment_rule_sha256") != amendment_sha):
        raise ValueError("Frozen rule hash does not match the campaign receipt")
    if len(campaign.get("jobs", ())) != campaign.get("declared_jobs") != 38:
        raise ValueError("Amendment job manifest is incomplete")

    incomplete = []
    for job in campaign["jobs"]:
        rule_sha = legacy_sha if job["kind"] == "carry_m32" else amendment_sha
        try:
            report = _verify_completed(job, rule_sha)
        except Exception as error:  # Preserve a non-terminal campaign when evidence is absent.
            incomplete.append(dict(slug=job["slug"], error=str(error)))
            continue
        job.update(status="complete", model_sha256=report["model_sha256"],
                   best_validation_score=report["best_validation_score"],
                   adam_steps=report["adam_steps"],
                   adam_stop_reason=report["adam_stop_reason"],
                   lbfgs_outer_calls=report["lbfgs_outer_calls"],
                   lbfgs_stop_reason=report["lbfgs_stop_reason"],
                   elapsed_seconds=report["elapsed_seconds"])
        job.pop("pid", None)

    if incomplete:
        campaign["status"] = "incomplete"
        campaign["closeout_errors"] = incomplete
        _atomic_json(status_path, campaign)
        raise RuntimeError(f"{len(incomplete)} amendment jobs lack valid final artifacts")
    campaign.pop("closeout_errors", None)
    campaign["status"] = "complete"
    campaign["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    campaign["closeout"] = dict(
        method="post-controller artifact validation",
        completed_utc=campaign["finished_utc"],
        note="No training or evaluation was run during closeout.",
    )
    _atomic_json(status_path, campaign)
    return campaign


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--rule", type=Path, default=AMENDMENT_RULE)
    args = parser.parse_args()
    campaign = close(args.output.resolve(), args.rule.resolve())
    print(json.dumps(dict(status=campaign["status"], jobs=len(campaign["jobs"])), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
