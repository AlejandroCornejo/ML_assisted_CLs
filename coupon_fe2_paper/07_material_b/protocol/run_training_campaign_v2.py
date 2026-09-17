"""Launch and track the 15 fresh Material-B v2 fits without opening test labels.

The three-worker default uses six declared compute threads. Each fit has its
own log, checkpoint, and validation report. A failed fit is retained, not
silently retried; independent fits continue and the campaign reports failure.
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

from protocol.select_features import BASE
from protocol.train_material_b import _atomic_json
from protocol.train_material_b_v2 import AMENDMENT, load_recipe

ROOT = BASE.parents[1]
OUTPUT = BASE / "results/neural_training_v2"
NAMES = ("Free", "ICNN-fixed", "ICNN-learned", "ICKAN-fixed", "ICKAN-learned")
SEEDS = (16, 29, 47)


def _sha(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _slug(name: str, seed: int):
    return f"{name.lower().replace('-', '_')}_seed{seed}"


def _verify_run(folder: Path, name: str, seed: int, amendment_sha: str):
    report_path = folder / "run_report.json"
    model_path = folder / "model.pt"
    if not report_path.is_file() or not model_path.is_file():
        raise ValueError("Missing final report or model checkpoint")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (report.get("status") != "complete" or report.get("model") != name
            or report.get("seed") != seed or report.get("recipe_version") != 2
            or report.get("amendment_sha256") != amendment_sha
            or report.get("model_sha256") != _sha(model_path)
            or report.get("adam_stop_reason") not in ("validation_plateau", "safety_cap")
            or report.get("lbfgs_stop_reason") not in ("validation_plateau", "safety_cap")
            or report.get("test_labels_loaded") is not False
            or report.get("path_labels_loaded") is not False):
        raise ValueError("Final report fails v2 integrity checks")
    return report


def run_campaign(output: Path, workers: int):
    if not 1 <= workers <= 4:
        raise ValueError("Choose one to four concurrent fits")
    if output.exists():
        raise FileExistsError(f"Official campaign output already exists: {output}")
    load_recipe(BASE / "results/feature_selection_v1", AMENDMENT)
    amendment_sha = _sha(AMENDMENT)
    output.mkdir(parents=True, exist_ok=False)
    (output / "logs").mkdir()
    jobs = [dict(model=name, seed=seed, slug=_slug(name, seed), status="pending")
            for seed in SEEDS for name in NAMES]
    campaign = dict(recipe_version=2, amendment_sha256=amendment_sha,
                    started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    status="running", workers=workers, jobs=jobs)
    _atomic_json(output / "campaign_status.json", campaign)
    env = os.environ.copy()
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env[key] = "2"
    env["PYTHONPATH"] = os.pathsep.join((str(ROOT / "coupon_fe2_paper/.pydeps"), str(BASE)))
    active = {}
    while True:
        for job in jobs:
            if len(active) >= workers:
                break
            if job["status"] != "pending":
                continue
            folder = output / job["slug"]
            log_path = output / "logs" / f"{job['slug']}.log"
            command = [sys.executable, "-B", "-m", "protocol.train_material_b_v2",
                       "--model", job["model"], "--seed", str(job["seed"]),
                       "--output", str(folder)]
            with log_path.open("xb") as log:
                process = subprocess.Popen(command, cwd=ROOT, env=env,
                                           stdout=log, stderr=subprocess.STDOUT,
                                           start_new_session=True)
            job.update(status="running", pid=process.pid,
                       started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                       log=str(log_path.relative_to(output)))
            active[job["slug"]] = process
            _atomic_json(output / "campaign_status.json", campaign)
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
                report = _verify_run(output / job["slug"], job["model"],
                                     job["seed"], amendment_sha)
                job.update(status="complete", model_sha256=report["model_sha256"],
                           best_validation_score=report["best_validation_score"],
                           adam_steps=report["adam_steps"],
                           adam_stop_reason=report["adam_stop_reason"],
                           lbfgs_outer_calls=report["lbfgs_outer_calls"],
                           lbfgs_stop_reason=report["lbfgs_stop_reason"])
            except Exception as error:
                job.update(status="failed", error=str(error))
            del active[job["slug"]]
            _atomic_json(output / "campaign_status.json", campaign)
        if not active and all(job["status"] != "pending" for job in jobs):
            campaign["status"] = ("complete" if all(job["status"] == "complete"
                                             for job in jobs) else "partial_failure")
            campaign["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            _atomic_json(output / "campaign_status.json", campaign)
            return 0 if campaign["status"] == "complete" else 1
        time.sleep(2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    return run_campaign(args.output.resolve(), args.workers)


if __name__ == "__main__":
    sys.exit(main())
