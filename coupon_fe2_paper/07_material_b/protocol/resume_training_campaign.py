"""Approve a synchronized Adam review and resume the official 15-run campaign.

The frozen trainer and rule files remain unchanged. The review cadence is
explicitly amended: existing 5,000-step bookkeeping blocks may be approved
together only after every run reaches the review boundary. An approved horizon
is a safety ceiling, not a replacement for validation-based early stopping.
Original checkpoints and campaign status are copied into a review audit folder
before any checkpoint is changed. No reserved test or path labels are read.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch

from protocol import train_material_b as base
from protocol import train_material_b_official as official
from protocol.prepare_design import digest
from protocol.run_training_campaign import NAMES, OUTPUT, SEEDS, _slug, _verify_job
from protocol.select_features import BASE

ADAM_CEILING = 200_000


def _utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _expected_identity(name, seed, manifest):
    identity = base._identity(name, seed, official.RULE, base.FEATURES,
                              base.LABELS, manifest)
    identity["parent_preparation_sha256"] = digest(official.RECIPE)
    source = Path(official.__file__).resolve()
    identity["sources_sha256"][str(source.relative_to(BASE.parents[1]))] = digest(source)
    return identity


def inspect_campaign(output: Path):
    """Read-only preflight: all 15 terminal, with genuine Adam pauses at one step."""
    status_path = output / "campaign_status.json"
    campaign = json.loads(status_path.read_text(encoding="utf-8"))
    _, rule, manifest, _ = official.load_rule(base.FEATURES)
    rule_sha = digest(official.RULE)
    expected_jobs = {(name, seed, _slug(name, seed))
                     for seed in SEEDS for name in NAMES}
    actual_jobs = {(job["model"], job["seed"], job["slug"])
                   for job in campaign["jobs"]}
    if (campaign["status"] != "needs_review"
            or campaign["official_rule_sha256"] != rule_sha
            or len(campaign["jobs"]) != len(expected_jobs)
            or actual_jobs != expected_jobs):
        raise ValueError("Campaign is not a complete, frozen 15-run review batch")
    paused = []
    for job in campaign["jobs"]:
        if job["status"] not in ("needs_review", "complete"):
            raise ValueError(f"All 15 must finish or pause before review: {job['slug']}")
        folder = output / job["slug"]
        verified, _ = _verify_job(folder, job, rule_sha)
        if verified != job["status"] or (folder / "failure.json").exists():
            raise ValueError(f"Invalid review checkpoint: {job['slug']}")
        if verified == "complete":
            continue
        payload = torch.load(folder / official.STATE_NAME,
                             map_location="cpu", weights_only=False)
        if (payload["identity"] != _expected_identity(job["model"], job["seed"], manifest)
                or payload["phase"] != "adam"
                or payload["review_required"] is not True
                or payload["adam_step"] != payload["adam_limit"]
                or payload["lbfgs_call"] != 0):
            raise ValueError(f"Unsafe or changed review checkpoint: {job['slug']}")
        paused.append((job, payload))
    if not paused:
        raise ValueError("No Adam-paused run needs extension")
    steps = {payload["adam_step"] for _, payload in paused}
    if len(steps) != 1:
        raise ValueError("Runs reached different Adam review boundaries")
    return campaign, rule, paused, steps.pop()


def approve_batch(output: Path, campaign, rule, paused, step: int, reason: str,
                  target_step: int | None = None):
    """Back up first, then record each approved 5,000-step bookkeeping block."""
    if not reason.strip():
        raise ValueError("An explicit review reason is required")
    increment = rule["adam"]["review_extension_steps"]
    new_limit = step + 2 * increment if target_step is None else target_step
    if (new_limit <= step or new_limit > ADAM_CEILING
            or (new_limit - step) % increment):
        raise ValueError(f"Adam target must be aligned and at most {ADAM_CEILING:,} steps")
    blocks = (new_limit - step) // increment
    review_dir = output / "reviews" / f"adam_{step}_to_{new_limit}"
    review_dir.mkdir(parents=True, exist_ok=False)
    status_path = output / "campaign_status.json"
    try:
        shutil.copy2(status_path, review_dir / "campaign_status_before.json")
        for job, _ in paused:
            shutil.copy2(output / job["slug"] / official.STATE_NAME,
                         review_dir / f"{job['slug']}_before.pt")
        approved_utc = _utc()
        batch_id = review_dir.name
        audit = dict(batch_id=batch_id, approved_utc=approved_utc,
                     reason=reason.strip(), old_limit=step, new_limit=new_limit,
                     blocks=blocks, block_steps=increment, workers=None,
                     adam_ceiling=ADAM_CEILING,
                     ceiling_is_completion_criterion=False,
                     official_rule_sha256=digest(official.RULE), runs=[],
                     untouched_completed=[dict(slug=job["slug"],
                         model_sha256=digest(output / job["slug"] / "model.pt"))
                         for job in campaign["jobs"] if job["status"] == "complete"])
        for job, payload in paused:
            payload = copy.deepcopy(payload)
            old_hash = digest(output / job["slug"] / official.STATE_NAME)
            for block in range(blocks):
                old = step + block * increment
                payload["review_events"].append(dict(
                    phase="adam", old_limit=old, new_limit=old + increment,
                    approved_utc=approved_utc, batch_id=batch_id,
                    reason=reason.strip()))
            payload["adam_limit"] = new_limit
            payload["review_required"] = False
            base._atomic_torch(output / job["slug"] / official.STATE_NAME, payload)
            audit["runs"].append(dict(slug=job["slug"],
                checkpoint_sha256_before=old_hash,
                checkpoint_sha256_after=digest(output / job["slug"] / official.STATE_NAME)))
        base._atomic_json(review_dir / "approval.json", audit)
        return review_dir, audit
    except Exception:
        for job, _ in paused:
            backup = review_dir / f"{job['slug']}_before.pt"
            if backup.is_file():
                shutil.copy2(backup, output / job["slug"] / official.STATE_NAME)
        if (review_dir / "campaign_status_before.json").is_file():
            shutil.copy2(review_dir / "campaign_status_before.json", status_path)
        raise


def resume_campaign(output: Path, workers: int, reason: str, dry_run=False,
                    target_step: int | None = None):
    if not 1 <= workers <= 8:
        raise ValueError("Choose one to eight concurrent CPU fits")
    campaign, rule, paused, step = inspect_campaign(output)
    increment = rule["adam"]["review_extension_steps"]
    new_limit = step + 2 * increment if target_step is None else target_step
    if (new_limit <= step or new_limit > ADAM_CEILING
            or (new_limit - step) % increment):
        raise ValueError(f"Adam target must be aligned and at most {ADAM_CEILING:,} steps")
    if dry_run:
        print(json.dumps(dict(status="ready", runs=len(paused), adam_step=step,
                              proposed_limit=new_limit, workers=workers,
                              early_stop="validation_plateau",
                              ceiling_is_completion_criterion=False)))
        return 0
    review_dir, audit = approve_batch(output, campaign, rule, paused, step,
                                     reason, target_step=new_limit)
    audit["workers"] = workers
    base._atomic_json(review_dir / "approval.json", audit)
    campaign.setdefault("review_batches", []).append(dict(
        batch_id=audit["batch_id"], approved_utc=audit["approved_utc"],
        reason=audit["reason"], old_limit=step,
        new_limit=new_limit, workers=workers,
        audit=str((review_dir / "approval.json").relative_to(output))))
    campaign["status"] = "running"
    campaign["workers"] = workers
    campaign["resumed_utc"] = _utc()
    campaign.pop("finished_utc", None)
    paused_slugs = {job["slug"] for job, _ in paused}
    for job in campaign["jobs"]:
        if job["slug"] not in paused_slugs:
            continue
        job["status"] = "pending"
        for key in ("pid", "log", "exit_code", "started_utc", "finished_utc",
                    "phase", "adam_step", "lbfgs_call", "best_validation_score"):
            job.pop(key, None)
    base._atomic_json(output / "campaign_status.json", campaign)

    env = os.environ.copy()
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env[key] = "2"
    env["PYTHONPATH"] = os.pathsep.join((str(BASE.parents[1] / "coupon_fe2_paper/.pydeps"),
                                         str(BASE)))
    active = {}
    while True:
        for job in campaign["jobs"]:
            if len(active) >= workers:
                break
            if job["status"] != "pending":
                continue
            log_path = output / "logs" / f"{job['slug']}_resume_{step}.log"
            command = [sys.executable, "-B", "-m", "protocol.train_material_b_official",
                       "--model", job["model"], "--seed", str(job["seed"]),
                       "--output", str(output / job["slug"]), "--resume"]
            with log_path.open("xb") as log:
                process = subprocess.Popen(command, cwd=BASE.parents[1], env=env,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            job.update(status="running", pid=process.pid, started_utc=_utc(),
                       log=str(log_path.relative_to(output)))
            active[job["slug"]] = process
            base._atomic_json(output / "campaign_status.json", campaign)
        for job in campaign["jobs"]:
            if job["status"] != "running":
                continue
            process = active[job["slug"]]
            code = process.poll()
            if code is None:
                continue
            job["exit_code"] = code
            job["finished_utc"] = _utc()
            try:
                if code != 0:
                    raise RuntimeError(f"Trainer exited with status {code}")
                status, details = _verify_job(output / job["slug"], job,
                                              campaign["official_rule_sha256"])
                job.update(status=status, **details)
            except Exception as error:
                job.update(status="failed", error=str(error))
            del active[job["slug"]]
            base._atomic_json(output / "campaign_status.json", campaign)
        if not active and all(job["status"] != "pending" for job in campaign["jobs"]):
            statuses = [job["status"] for job in campaign["jobs"]]
            campaign["status"] = ("partial_failure" if "failed" in statuses else
                                  "complete" if all(x == "complete" for x in statuses) else
                                  "needs_review")
            campaign["finished_utc"] = _utc()
            base._atomic_json(output / "campaign_status.json", campaign)
            return 1 if campaign["status"] == "partial_failure" else 0
        time.sleep(2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--target-adam-step", type=int)
    parser.add_argument("--reason", default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    return resume_campaign(args.output.resolve(), args.workers,
                           args.reason, dry_run=args.dry_run,
                           target_step=args.target_adam_step)


if __name__ == "__main__":
    sys.exit(main())
