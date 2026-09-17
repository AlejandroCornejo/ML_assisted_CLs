"""Run the 15 official Material-B fits with three bounded CPU workers.

The campaign succeeds only when every fit meets both validation stop rules.
Review-boundary pauses remain visible as needs_review, never as complete.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

from protocol.select_features import BASE
from protocol.train_material_b import _atomic_json
from protocol.train_material_b_official import RULE, load_rule

ROOT = BASE.parents[1]
OUTPUT = BASE / "results/neural_training"
NAMES = ("Free", "ICNN-fixed", "ICNN-learned", "ICKAN-fixed", "ICKAN-learned")
SEEDS = (16, 29, 47)


def _sha(path: Path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _slug(name: str, seed: int):
    return f"{name.lower().replace('-', '_')}_seed{seed}"


def _verify_job(folder: Path, job, rule_sha: str):
    report_path, model_path = folder/"run_report.json", folder/"model.pt"
    if report_path.is_file() and model_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if (report.get("status") != "complete"
                or report.get("model") != job["model"]
                or report.get("seed") != job["seed"]
                or report.get("official_rule_sha256") != rule_sha
                or report.get("model_sha256") != _sha(model_path)
                or report.get("adam_stop_reason") != "validation_plateau"
                or report.get("lbfgs_stop_reason") != "validation_plateau"
                or report.get("test_labels_loaded") is not False
                or report.get("path_labels_loaded") is not False):
            raise ValueError("Official final report failed integrity checks")
        return "complete", dict(model_sha256=report["model_sha256"],
            best_validation_score=report["best_validation_score"],
            adam_steps=report["adam_steps"], lbfgs_outer_calls=report["lbfgs_outer_calls"],
            adam_stop_reason=report["adam_stop_reason"],
            lbfgs_stop_reason=report["lbfgs_stop_reason"])
    if report_path.exists() or model_path.exists():
        raise ValueError("Incomplete or mismatched final artifacts")
    path = folder/"run_state.pt"
    if path.is_file():
        state = torch.load(path, map_location="cpu", weights_only=False)
        if (state.get("format_version") == 3 and state.get("review_required") is True
                and state.get("phase") in ("adam", "lbfgs")
                and state.get("identity", {}).get("model") == job["model"]
                and state.get("identity", {}).get("seed") == job["seed"]
                and state.get("identity", {}).get("recipe_sha256") == rule_sha):
            return "needs_review", dict(phase=state["phase"],
                adam_step=state["adam_step"], lbfgs_call=state["lbfgs_call"],
                best_validation_score=state["best_score"])
    raise ValueError("No valid complete model or review-boundary checkpoint")


def run_campaign(output: Path, workers: int):
    if not 1 <= workers <= 4:
        raise ValueError("Choose one to four concurrent fits")
    if output.exists():
        raise FileExistsError(f"Official campaign already exists: {output}")
    load_rule(BASE/"results/feature_selection_v1", RULE)
    rule_sha = _sha(RULE)
    output.mkdir(parents=True, exist_ok=False)
    (output/"logs").mkdir()
    jobs = [dict(model=name, seed=seed, slug=_slug(name, seed), status="pending")
            for seed in SEEDS for name in NAMES]
    campaign = dict(status="running", official_rule_sha256=rule_sha,
        started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        workers=workers, jobs=jobs)
    _atomic_json(output/"campaign_status.json", campaign)
    env = os.environ.copy()
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env[key] = "2"
    env["PYTHONPATH"] = os.pathsep.join((str(ROOT/"coupon_fe2_paper/.pydeps"), str(BASE)))
    active = {}
    while True:
        for job in jobs:
            if len(active) >= workers:
                break
            if job["status"] != "pending":
                continue
            folder = output/job["slug"]
            log_path = output/"logs"/f"{job['slug']}.log"
            command = [sys.executable, "-B", "-m", "protocol.train_material_b_official",
                       "--model", job["model"], "--seed", str(job["seed"]),
                       "--output", str(folder)]
            with log_path.open("xb") as log:
                process = subprocess.Popen(command, cwd=ROOT, env=env,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            job.update(status="running", pid=process.pid,
                started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                log=str(log_path.relative_to(output)))
            active[job["slug"]] = process
            _atomic_json(output/"campaign_status.json", campaign)
        for job in jobs:
            if job["status"] != "running":
                continue
            process = active[job["slug"]]
            code = process.poll()
            if code is None:
                continue
            job["exit_code"] = code
            job["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            try:
                if code != 0:
                    raise RuntimeError(f"Trainer exited with status {code}")
                status, details = _verify_job(output/job["slug"], job, rule_sha)
                job.update(status=status, **details)
            except Exception as error:
                job.update(status="failed", error=str(error))
            del active[job["slug"]]
            _atomic_json(output/"campaign_status.json", campaign)
        if not active and all(job["status"] != "pending" for job in jobs):
            campaign["status"] = ("complete" if all(job["status"] == "complete"
                                              for job in jobs) else "needs_review" if any(
                                              job["status"] == "needs_review" for job in jobs)
                                              else "partial_failure")
            campaign["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            _atomic_json(output/"campaign_status.json", campaign)
            return 0 if campaign["status"] in ("complete", "needs_review") else 1
        time.sleep(2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    return run_campaign(args.output.resolve(), args.workers)


if __name__ == "__main__":
    sys.exit(main())
