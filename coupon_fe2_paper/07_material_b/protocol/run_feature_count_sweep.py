"""Launch and track the 60 constrained multicavity feature-count fits.

Ten workers use the twenty available CPU compute threads (two per full-batch
fit).  Every fit has its own log and resumable checkpoint.  Historical
Material-B campaigns are never reused or overwritten.
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
from protocol.train_feature_count_sweep import NAMES, RULE, load_rule

ROOT = BASE.parents[1]
FEATURES_ROOT = BASE / "results/feature_count_sweep_v1/feature_tables"
OUTPUT = BASE / "results/feature_count_sweep_v1/training"
COUNTS = (8, 16, 24, 32, 40)
SEEDS = (16, 29, 47)


def _sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _slug(feature_count: int, name: str, seed: int) -> str:
    return f"m{feature_count:02d}_{name.lower().replace("-", "_")}_seed{seed}"


def _verify_features(features_root: Path, rule_path: Path) -> dict:
    if not features_root.is_dir():
        raise FileNotFoundError("Feature tables have not been prepared")
    summary = {}
    for count in COUNTS:
        folder = features_root / f"m{count:02d}"
        _, rule, manifest, _, _, selected_count = load_rule(folder, rule_path)
        check_path = folder / "initialization_checks.json"
        if selected_count != count or tuple(rule["feature_counts"]) != COUNTS:
            raise ValueError("Feature table and sweep rule disagree")
        if not check_path.is_file():
            raise ValueError("Paired initialization checks have not been frozen")
        checks = json.loads(check_path.read_text(encoding="utf-8"))
        if (checks.get("passed") is not True or checks.get("feature_count") != count
                or checks.get("feature_table_sha256") != manifest["feature_table_sha256"]
                or len(checks.get("runs", ())) != 12
                or len(checks.get("paired_initializations", ())) != 6):
            raise ValueError("Initialization checks do not cover this feature table")
        summary[str(count)] = dict(feature_table_sha256=manifest["feature_table_sha256"],
                                   manifest_sha256=_sha(folder / "manifest.json"),
                                   preparation_recipe_sha256=_sha(folder / "preparation_recipe.json"),
                                   initialization_checks_sha256=_sha(check_path))
    frozen = BASE / "results/feature_selection_v1/feature_table.npz"
    if summary["32"]["feature_table_sha256"] != _sha(frozen):
        raise ValueError("m=32 table is not byte-identical to the locked historical table")
    return summary


def _verify_report(folder: Path, job: dict, rule_sha: str) -> dict:
    report_path, model_path = folder / "run_report.json", folder / "model.pt"
    if not report_path.is_file() or not model_path.is_file():
        raise ValueError("Missing final report or model checkpoint")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (report.get("status") != "complete" or report.get("model") != job["model"]
            or report.get("seed") != job["seed"] or report.get("feature_count") != job["feature_count"]
            or report.get("sweep_rule_sha256") != rule_sha
            or report.get("model_sha256") != _sha(model_path)
            or report.get("adam_stop_reason") not in ("validation_plateau", "safety_cap")
            or report.get("lbfgs_stop_reason") not in ("validation_plateau", "safety_cap")
            or report.get("test_labels_loaded") is not False
            or report.get("path_labels_loaded") is not False):
        raise ValueError("Final report fails sweep integrity checks")
    return report


def run_campaign(output: Path, features_root: Path, workers: int, rule_path: Path) -> int:
    if not 1 <= workers <= 10:
        raise ValueError("Choose one to ten concurrent fits (two compute threads each)")
    if output.exists():
        raise FileExistsError(f"Sweep output already exists: {output}")
    if _sha(rule_path) != _sha(RULE):
        raise ValueError("Use the frozen feature-count sweep rule")
    feature_hashes = _verify_features(features_root, rule_path)
    rule_sha = _sha(rule_path)
    output.mkdir(parents=True, exist_ok=False)
    (output / "logs").mkdir()
    jobs = [dict(feature_count=count, model=name, seed=seed,
                 slug=_slug(count, name, seed), status="pending")
            for count in COUNTS for seed in SEEDS for name in NAMES]
    if len(jobs) != 60:
        raise RuntimeError("Sweep design must contain exactly 60 jobs")
    campaign = dict(status="running", protocol_id="material_B_feature_count_sweep_v1",
        sweep_rule_sha256=rule_sha, features=feature_hashes, workers=workers,
        threads_per_fit=2, declared_jobs=len(jobs),
        started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), jobs=jobs)
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
            command = [sys.executable, "-B", "-m", "protocol.train_feature_count_sweep",
                "--model", job["model"], "--seed", str(job["seed"]),
                "--features", str(features_root / f"m{job['feature_count']:02d}"),
                "--rule", str(rule_path), "--output", str(folder)]
            with log_path.open("xb") as log:
                process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                           stderr=subprocess.STDOUT, start_new_session=True)
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
                report = _verify_report(output / job["slug"], job, rule_sha)
                job.update(status="complete", model_sha256=report["model_sha256"],
                    best_validation_score=report["best_validation_score"],
                    adam_steps=report["adam_steps"], adam_stop_reason=report["adam_stop_reason"],
                    lbfgs_outer_calls=report["lbfgs_outer_calls"],
                    lbfgs_stop_reason=report["lbfgs_stop_reason"],
                    elapsed_seconds=report["elapsed_seconds"])
            except Exception as error:
                job.update(status="failed", error=str(error))
            del active[job["slug"]]
            _atomic_json(output / "campaign_status.json", campaign)
        if not active and all(job["status"] != "pending" for job in jobs):
            campaign["status"] = "complete" if all(job["status"] == "complete" for job in jobs) else "partial_failure"
            campaign["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            _atomic_json(output / "campaign_status.json", campaign)
            return 0 if campaign["status"] == "complete" else 1
        time.sleep(2)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--features-root", type=Path, default=FEATURES_ROOT)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--rule", type=Path, default=RULE)
    args = parser.parse_args()
    return run_campaign(args.output.resolve(), args.features_root.resolve(), args.workers, args.rule.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
