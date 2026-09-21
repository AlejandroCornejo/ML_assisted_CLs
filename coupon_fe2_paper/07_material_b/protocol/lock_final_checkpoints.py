"""Verify and hash-lock all 15 Material-B models without loading reserved labels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from protocol import train_material_b as base
from protocol import train_material_b_official as official
from protocol.close_free_lbfgs import AMENDMENT, FINAL_STATUS, utc, verify_free
from protocol.prepare_design import digest
from protocol.run_training_campaign import NAMES, OUTPUT, SEEDS, _slug, _verify_job


def lock(output: Path):
    campaign_path = output / "campaign_status.json"
    campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
    expected = {(name, seed, _slug(name, seed)) for seed in SEEDS for name in NAMES}
    actual = {(job["model"], job["seed"], job["slug"]) for job in campaign["jobs"]}
    if (campaign["status"] != "complete_with_budget_limited_adam"
            or actual != expected or len(campaign["jobs"]) != 15
            or campaign["official_rule_sha256"] != digest(official.RULE)
            or campaign["free_closeout_amendment_sha256"] != digest(AMENDMENT)):
        raise ValueError("Campaign is not the exact audited 12+3 closeout")
    audit_path = output / campaign["free_closeout_audit"]
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if (audit["amendment_sha256"] != digest(AMENDMENT)
            or len(audit["untouched_completed"]) != 12
            or len(audit["free_runs"]) != 3):
        raise ValueError("Invalid Free closeout audit")
    for item in audit["untouched_completed"]:
        if digest(output / item["slug"] / "model.pt") != item["model_sha256"]:
            raise ValueError(f"Previously completed model changed: {item['slug']}")
    entries = []
    for job in campaign["jobs"]:
        folder = output / job["slug"]
        if job["model"] == "Free":
            verified, _ = verify_free(folder, job["seed"], campaign["official_rule_sha256"])
            expected_status = FINAL_STATUS
        else:
            verified, _ = _verify_job(folder, job, campaign["official_rule_sha256"])
            expected_status = "complete"
        if verified != expected_status or job["status"] != expected_status:
            raise ValueError(f"Incomplete final job: {job['slug']}")
        report_path, model_path = folder / "run_report.json", folder / "model.pt"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        state = torch.load(folder / official.STATE_NAME, map_location="cpu",
                           weights_only=False)
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        if (state["phase"] != "complete" or state["review_required"]
                or report["identity"] != state["identity"]
                or checkpoint["identity"] != state["identity"]
                or report["best_validation_score"] != state["best_score"]
                or checkpoint["best_validation_score"] != state["best_score"]
                or report["best_origin"] != state["best_origin"]
                or checkpoint["best_origin"] != state["best_origin"]
                or state["best_score"] != min(event["score"]
                                              for event in state["validation_history"])
                or not all(torch.equal(value, state["best_model_state"][key])
                           for key, value in checkpoint["state_dict"].items())
                or report["test_labels_loaded"] is not False
                or report["path_labels_loaded"] is not False):
            raise ValueError(f"Checkpoint and report disagree: {job['slug']}")
        entries.append(dict(model=job["model"], seed=job["seed"],
            slug=job["slug"], status=verified,
            adam_stop_reason=report["adam_stop_reason"],
            lbfgs_stop_reason=report["lbfgs_stop_reason"],
            best_validation_score=report["best_validation_score"],
            best_origin=report["best_origin"],
            model_sha256=digest(model_path), report_sha256=digest(report_path),
            run_state_sha256=digest(folder / official.STATE_NAME)))
    manifest = dict(status="locked_mixed_plateau_and_adam_budget",
        locked_utc=utc(), campaign_status_sha256=digest(campaign_path),
        official_rule_sha256=digest(official.RULE),
        free_amendment_sha256=digest(AMENDMENT),
        free_closeout_audit_sha256=digest(audit_path),
        checkpoint_count=len(entries), entries=entries,
        reserved_test_labels_loaded=False, heldout_path_labels_loaded=False,
        evaluation_gate_amended=False)
    target = output / "final_checkpoint_manifest.json"
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite locked manifest: {target}")
    base._atomic_json(target, manifest)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    target = lock(args.output.resolve())
    print(target)


if __name__ == "__main__":
    main()
