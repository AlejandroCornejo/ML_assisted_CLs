"""Complete retained m=32 jobs and run the m=2,4,6 count amendment.

The legacy coordinator was stopped before its m=40 jobs began.  Its ten live
m=24/m=32 child fits remain untouched.  This controller observes those fits,
uses only newly freed two-thread slots, completes the two missing m=32 ICKAN
seeds under the original rule, and then runs the 36 new low-count fits.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

from protocol.select_features import BASE
from protocol.train_material_b import _atomic_json
from protocol.train_feature_count_amendment_v2 import NAMES, RULE as AMENDMENT_RULE, load_rule as load_amendment_rule
from protocol.train_feature_count_sweep import RULE as LEGACY_RULE, load_rule as load_legacy_rule

ROOT = BASE.parents[1]
LEGACY_ROOT = BASE / "results/feature_count_sweep_v1/training"
LOW_FEATURES = BASE / "results/feature_count_amendment_v2/feature_tables"
OUTPUT = BASE / "results/feature_count_amendment_v2/training"
LOW_COUNTS = (2, 4, 6)
SEEDS = (16, 29, 47)
WORKERS = 10


def _sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _slug(feature_count: int, name: str, seed: int) -> str:
    return f"m{feature_count:02d}_{name.lower().replace('-', '_')}_seed{seed}"


def _report(folder: Path) -> dict | None:
    path = folder / "run_report.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _legacy_snapshot() -> tuple[list[dict], list[dict], dict]:
    """Return live retained jobs, queued m=32 completions, and audit receipt."""
    status_path = LEGACY_ROOT / "campaign_status.json"
    campaign = json.loads(status_path.read_text(encoding="utf-8"))
    m40 = [j for j in campaign["jobs"] if j["feature_count"] == 40]
    retained = [j for j in campaign["jobs"] if j["feature_count"] in (8, 16, 24, 32)]
    if len(m40) != 12 or any(j["status"] != "pending" for j in m40):
        raise RuntimeError("m=40 was not wholly unstarted when the amendment was made")
    live, missing = [], []
    for job in retained:
        folder = LEGACY_ROOT / job["slug"]
        report = _report(folder)
        if report is not None:
            if report.get("status") != "complete":
                raise RuntimeError(f"Invalid retained report: {job['slug']}")
            continue
        if (folder / "failure.json").is_file():
            raise RuntimeError(f"Retained job failed: {job['slug']}")
        if job["status"] == "running":
            live.append(job)
        elif job["status"] == "pending" and job["feature_count"] == 32:
            missing.append(job)
        else:
            raise RuntimeError(f"Unexpected retained legacy state: {job['slug']}")
    if len(missing) != 2 or {j["model"] for j in missing} != {"ICKAN-fixed", "ICKAN-learned"} or {j["seed"] for j in missing} != {47}:
        raise RuntimeError("The amendment expects exactly the two queued m=32 ICKAN seed-47 jobs")
    receipt = dict(legacy_campaign_status_sha256=_sha(status_path),
                   legacy_coordinator_pid=1588150,
                   cancelled_unstarted_jobs=[j["slug"] for j in m40],
                   retained_live_jobs=[j["slug"] for j in live],
                   carried_m32_jobs=[j["slug"] for j in missing])
    return live, missing, receipt


def _live_legacy_count(jobs: list[dict]) -> int:
    count = 0
    for job in jobs:
        folder = LEGACY_ROOT / job["slug"]
        if _report(folder) is not None:
            continue
        if (folder / "failure.json").is_file():
            raise RuntimeError(f"Retained job failed: {job['slug']}")
        pid = int(job["pid"])
        try:
            os.kill(pid, 0)
        except ProcessLookupError as error:
            raise RuntimeError(f"Retained job ended without final report: {job['slug']}") from error
        count += 1
    return count


def _verify_low_features(features_root: Path, rule_path: Path) -> dict:
    summary = {}
    for count in LOW_COUNTS:
        folder = features_root / f"m{count:02d}"
        _, rule, manifest, _, _, actual = load_amendment_rule(folder, rule_path)
        check = folder / "initialization_checks.json"
        if actual != count or tuple(rule["feature_counts"]) != LOW_COUNTS or not check.is_file():
            raise RuntimeError("Low-count feature preparation is incomplete")
        receipt = json.loads(check.read_text(encoding="utf-8"))
        if (receipt.get("passed") is not True or receipt.get("feature_count") != count
                or receipt.get("feature_table_sha256") != manifest["feature_table_sha256"]
                or len(receipt.get("runs", ())) != 12 or len(receipt.get("paired_initializations", ())) != 6):
            raise RuntimeError("Low-count initialization receipt is invalid")
        summary[str(count)] = dict(feature_table_sha256=manifest["feature_table_sha256"],
                                   manifest_sha256=_sha(folder / "manifest.json"),
                                   initialization_checks_sha256=_sha(check))
    return summary


def _verify_completed(job: dict, rule_sha: str) -> dict:
    folder = Path(job["output"])
    report = _report(folder)
    if report is None or not (folder / "model.pt").is_file():
        raise RuntimeError(f"Missing final artifacts: {job['slug']}")
    expected_field = "sweep_rule_sha256" if job["kind"] == "carry_m32" else "amendment_rule_sha256"
    if (report.get("status") != "complete" or report.get("model") != job["model"]
            or report.get("seed") != job["seed"] or report.get("feature_count") != job["feature_count"]
            or report.get(expected_field) != rule_sha
            or report.get("model_sha256") != _sha(folder / "model.pt")
            or report.get("adam_stop_reason") not in ("validation_plateau", "safety_cap")
            or report.get("lbfgs_stop_reason") not in ("validation_plateau", "safety_cap")
            or report.get("test_labels_loaded") is not False or report.get("path_labels_loaded") is not False):
        raise RuntimeError(f"Final integrity check failed: {job['slug']}")
    return report


def _command(job: dict, low_features: Path, amendment_rule: Path) -> list[str]:
    if job["kind"] == "carry_m32":
        return [sys.executable, "-B", "-m", "protocol.train_feature_count_sweep",
                "--model", job["model"], "--seed", str(job["seed"]),
                "--features", str(BASE / "results/feature_count_sweep_v1/feature_tables/m32"),
                "--rule", str(LEGACY_RULE), "--output", job["output"]]
    return [sys.executable, "-B", "-m", "protocol.train_feature_count_amendment_v2",
            "--model", job["model"], "--seed", str(job["seed"]),
            "--features", str(low_features / f"m{job['feature_count']:02d}"),
            "--rule", str(amendment_rule), "--output", job["output"]]


def run_campaign(output: Path, *, low_features: Path, amendment_rule: Path, workers: int = WORKERS) -> int:
    if not 1 <= workers <= WORKERS:
        raise ValueError("Choose one to ten two-thread workers")
    if output.exists():
        raise FileExistsError(f"Amendment output already exists: {output}")
    legacy_live, carried, receipt = _legacy_snapshot()
    load_legacy_rule(BASE / "results/feature_count_sweep_v1/feature_tables/m32", LEGACY_RULE)
    feature_receipts = _verify_low_features(low_features, amendment_rule)
    legacy_sha, amendment_sha = _sha(LEGACY_RULE), _sha(amendment_rule)
    output.mkdir(parents=True, exist_ok=False)
    (output / "logs").mkdir()
    notice = dict(status="amended", protocol_id="material_B_feature_count_amendment_v2", **receipt,
                  amended_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  amendment_rule_sha256=amendment_sha,
                  note="The legacy coordinator was stopped while all m=40 jobs were still pending. Its active child fits continue independently.")
    _atomic_json(LEGACY_ROOT / "amendment_notice_v2.json", notice)
    jobs = []
    for old in carried:
        jobs.append(dict(kind="carry_m32", feature_count=32, model=old["model"], seed=old["seed"],
                         slug=old["slug"], output=str((LEGACY_ROOT / old["slug"]).resolve()), status="pending"))
    for count in LOW_COUNTS:
        for seed in SEEDS:
            for name in NAMES:
                slug = _slug(count, name, seed)
                jobs.append(dict(kind="low_count", feature_count=count, model=name, seed=seed,
                                 slug=slug, output=str((output / slug).resolve()), status="pending"))
    if len(jobs) != 38:
        raise RuntimeError("Amendment must contain 2 carried plus 36 low-count jobs")
    campaign = dict(status="running", protocol_id="material_B_feature_count_amendment_v2",
        started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), workers=workers,
        threads_per_fit=2, legacy_receipt=receipt, legacy_rule_sha256=legacy_sha,
        amendment_rule_sha256=amendment_sha, low_feature_tables=feature_receipts,
        declared_jobs=len(jobs), jobs=jobs)
    _atomic_json(output / "campaign_status.json", campaign)
    env = os.environ.copy()
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env[key] = "2"
    env["PYTHONPATH"] = os.pathsep.join((str(ROOT / "coupon_fe2_paper/.pydeps"), str(BASE)))
    active: dict[str, subprocess.Popen] = {}
    while True:
        capacity = workers - _live_legacy_count(legacy_live) - len(active)
        for job in jobs:
            if capacity <= 0:
                break
            if job["status"] != "pending":
                continue
            log = output / "logs" / f"{job['slug']}.log"
            with log.open("xb") as stream:
                process = subprocess.Popen(_command(job, low_features, amendment_rule), cwd=ROOT, env=env,
                                           stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
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
            job.update(exit_code=code, finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
            try:
                if code != 0:
                    raise RuntimeError(f"Trainer exited with status {code}")
                rule_sha = legacy_sha if job["kind"] == "carry_m32" else amendment_sha
                report = _verify_completed(job, rule_sha)
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
        if not active and all(job["status"] != "pending" for job in jobs) and not _live_legacy_count(legacy_live):
            campaign["status"] = "complete" if all(job["status"] == "complete" for job in jobs) else "partial_failure"
            campaign["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            _atomic_json(output / "campaign_status.json", campaign)
            return 0 if campaign["status"] == "complete" else 1
        time.sleep(2)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--features-root", type=Path, default=LOW_FEATURES)
    parser.add_argument("--rule", type=Path, default=AMENDMENT_RULE)
    parser.add_argument("--workers", type=int, default=WORKERS)
    args = parser.parse_args()
    return run_campaign(args.output.resolve(), low_features=args.features_root.resolve(),
                        amendment_rule=args.rule.resolve(), workers=args.workers)


if __name__ == "__main__":
    raise SystemExit(main())
