#!/usr/bin/env python3
"""Run the six-model SC-RVE m=6 learned-feature campaign in parallel."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MATERIAL_B = ROOT / "07_material_b"
sys.path.insert(0, str(MATERIAL_B))
sys.path.insert(0, str(ROOT / "06_pann"))

from protocol.prepare_design import digest
from protocol.train_material_b import _atomic_json
from train_sc_m06_v1 import NAMES, SEEDS, DEFAULT_PREPARATION, load_rule

DEFAULT_OUTPUT = ROOT / "06_pann/results/sc_m06_learned_v1/training"


def slug(model: str, seed: int) -> str:
    return f"m06_{model.lower().replace('-', '_')}_seed{seed}"


def read_report(folder: Path):
    path = folder / "run_report.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def verify_report(job: dict, rule_sha: str) -> dict:
    folder = Path(job["output"])
    report = read_report(folder)
    model = folder / "model.pt"
    if report is None or not model.is_file():
        raise RuntimeError(f"Missing terminal artifacts for {job['slug']}")
    if (report.get("status") != "complete" or report.get("model") != job["model"]
            or report.get("seed") != job["seed"] or report.get("feature_count") != 6
            or report.get("training_rule_sha256") != rule_sha
            or report.get("model_sha256") != digest(model)
            or report.get("adam_stop_reason") not in ("validation_plateau", "safety_cap")
            or report.get("lbfgs_stop_reason") not in ("validation_plateau", "safety_cap")
            or report.get("test_labels_loaded") is not False
            or report.get("probe_labels_loaded") is not False):
        raise RuntimeError(f"Integrity check failed for {job['slug']}")
    return report


def run_campaign(output: Path, preparation: Path, workers: int) -> int:
    if not 1 <= workers <= 6:
        raise ValueError("Choose one to six two-thread workers")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite campaign: {output}")
    recipe, manifest, table, rule, rule_path, recipe_path, labels = load_rule(preparation)
    if table["specs"].shape != (6, 5):
        raise ValueError("Preparation does not contain six features")
    output.mkdir(parents=True)
    (output / "logs").mkdir()
    rule_sha = digest(rule_path)
    jobs = []
    for seed in SEEDS:
        for model in NAMES:
            name = slug(model, seed)
            jobs.append(dict(slug=name, model=model, seed=seed,
                             output=str((output / name).resolve()), status="pending"))
    campaign = dict(status="running", protocol_id=rule["id"], feature_count=6,
        started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        workers=workers, threads_per_fit=2, declared_jobs=len(jobs),
        preparation=str(preparation), preparation_recipe_sha256=digest(recipe_path),
        feature_table_sha256=digest(preparation / "feature_table.npz"),
        labels_sha256=digest(labels), training_rule_sha256=rule_sha,
        test_probe_accessed=False, jobs=jobs)
    _atomic_json(output / "campaign_status.json", campaign)

    env = os.environ.copy()
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env[key] = "2"
    env["PYTHONPATH"] = os.pathsep.join((str(ROOT / ".pydeps"), str(MATERIAL_B),
                                         str(ROOT / "06_pann")))
    active: dict[str, subprocess.Popen] = {}
    while True:
        capacity = workers - len(active)
        for job in jobs:
            if capacity <= 0:
                break
            if job["status"] != "pending":
                continue
            log = output / "logs" / f"{job['slug']}.log"
            command = [sys.executable, "-B", str(ROOT / "06_pann/train_sc_m06_v1.py"),
                       "--model", job["model"], "--seed", str(job["seed"]),
                       "--preparation", str(preparation), "--output", job["output"]]
            with log.open("xb") as stream:
                process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream,
                                           stderr=subprocess.STDOUT, start_new_session=True)
            job.update(status="running", pid=process.pid,
                       started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                       log=str(log.relative_to(output)))
            active[job["slug"]] = process
            capacity -= 1
            _atomic_json(output / "campaign_status.json", campaign)

        for job in jobs:
            if job["status"] != "running":
                continue
            process = active[job["slug"]]
            code = process.poll()
            if code is None:
                continue
            job.update(exit_code=code,
                       finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
            try:
                if code != 0:
                    raise RuntimeError(f"Trainer exited with status {code}")
                report = verify_report(job, rule_sha)
                job.update(status="complete", model_sha256=report["model_sha256"],
                           best_validation_score=report["best_validation_score"],
                           adam_steps=report["adam_steps"],
                           adam_stop_reason=report["adam_stop_reason"],
                           lbfgs_outer_calls=report["lbfgs_outer_calls"],
                           lbfgs_stop_reason=report["lbfgs_stop_reason"],
                           elapsed_seconds=report["elapsed_seconds"])
            except Exception as error:
                job.update(status="failed", error=str(error))
            del active[job["slug"]]
            _atomic_json(output / "campaign_status.json", campaign)

        if not active and all(job["status"] != "pending" for job in jobs):
            failures = [job for job in jobs if job["status"] != "complete"]
            if failures:
                campaign.update(status="partial_failure",
                    finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
                _atomic_json(output / "campaign_status.json", campaign)
                return 1
            selections = []
            for core in ("ICNN", "ICKAN"):
                rows = [job for job in jobs if job["model"].startswith(core)]
                chosen = min(rows, key=lambda row: (row["best_validation_score"], row["seed"]))
                selections.append(dict(core=core, model=chosen["model"], seed=chosen["seed"],
                    slug=chosen["slug"], checkpoint=str(Path(chosen["output"]) / "model.pt"),
                    best_validation_score=chosen["best_validation_score"],
                    checkpoint_sha256=chosen["model_sha256"]))
            selection = dict(status="frozen_before_test_probe", protocol_id=rule["id"],
                created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                criterion="minimum validation stress score within each core; seed number breaks exact ties",
                training_rule_sha256=rule_sha, test_probe_accessed=False,
                candidates=[dict(model=j["model"], seed=j["seed"], slug=j["slug"],
                                 best_validation_score=j["best_validation_score"],
                                 checkpoint_sha256=j["model_sha256"]) for j in jobs],
                selected=selections)
            _atomic_json(output / "validation_selection.json", selection)
            campaign.update(status="complete_validation_selected", test_probe_accessed=False,
                validation_selection_sha256=digest(output / "validation_selection.json"),
                finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
            _atomic_json(output / "campaign_status.json", campaign)
            return 0
        time.sleep(2)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preparation", type=Path, default=DEFAULT_PREPARATION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    return run_campaign(args.output.resolve(), args.preparation.resolve(), args.workers)


if __name__ == "__main__":
    raise SystemExit(main())
