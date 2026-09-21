"""Close the three Adam-budget-limited Free runs with the frozen L-BFGS rule.

This narrow, audited amendment does not read reserved test or path labels and
does not modify the original official trainer or its hashed training recipe.
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
from protocol.run_training_campaign import OUTPUT, SEEDS, _verify_job
from protocol.select_features import BASE
from protocol.training_setup import build_initial_model

AMENDMENT = BASE / "protocol/FREE_LBFGS_CLOSEOUT.md"
REVIEW_NAME = "adam_200000_to_lbfgs"
BUDGET_STEP = 200_000
FINAL_STATUS = "complete_budget_limited_adam"


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def expected_identity(seed, manifest):
    result = base._identity("Free", seed, official.RULE, base.FEATURES,
                            base.LABELS, manifest)
    result["parent_preparation_sha256"] = digest(official.RECIPE)
    source = Path(official.__file__).resolve()
    result["sources_sha256"][str(source.relative_to(BASE.parents[1]))] = digest(source)
    return result


def preflight(output):
    """Require the exact 12-complete/3-Free-at-200k campaign, without writes."""
    campaign = json.loads((output / "campaign_status.json").read_text(encoding="utf-8"))
    recipe, rule, manifest, table = official.load_rule(base.FEATURES)
    rule_sha = digest(official.RULE)
    if (campaign["status"] != "needs_review"
            or campaign["official_rule_sha256"] != rule_sha
            or len(campaign["jobs"]) != 15):
        raise ValueError("Expected the frozen 15-run campaign at a review boundary")
    free = []
    for job in campaign["jobs"]:
        folder = output / job["slug"]
        verified, _ = _verify_job(folder, job, rule_sha)
        if verified != job["status"] or (folder / "failure.json").exists():
            raise ValueError(f"Invalid campaign artifact: {job['slug']}")
        if job["model"] != "Free":
            if verified != "complete":
                raise ValueError(f"Non-Free run is not complete: {job['slug']}")
            continue
        state = torch.load(folder / official.STATE_NAME, map_location="cpu",
                           weights_only=False)
        if (verified != "needs_review" or state["phase"] != "adam"
                or state["review_required"] is not True
                or state["adam_step"] != BUDGET_STEP
                or state["adam_limit"] != BUDGET_STEP
                or state["lbfgs_call"] != 0
                or state["identity"] != expected_identity(job["seed"], manifest)
                or official.adam_plateau(state, rule)):
            raise ValueError(f"Free run is not an eligible Adam-cap pause: {job['slug']}")
        free.append((job, state))
    if sorted(job["seed"] for job, _ in free) != sorted(SEEDS):
        raise ValueError("Exactly the three declared Free seeds must be paused")
    return campaign, recipe, rule, manifest, table, free


def prepare(output):
    campaign, recipe, rule, manifest, table, free = preflight(output)
    review = output / "reviews" / REVIEW_NAME
    review.mkdir(parents=True, exist_ok=False)
    try:
        shutil.copy2(output / "campaign_status.json", review / "campaign_status_before.json")
        for job, _ in free:
            shutil.copy2(output / job["slug"] / official.STATE_NAME,
                         review / f"{job['slug']}_before.pt")
        arrays = base.load_training_arrays(base.LABELS, recipe)
        audit = dict(approved_utc=utc(), decision="User stopped Adam at 200000 "
                     "without plateau and approved the original L-BFGS phase",
                     official_rule_sha256=digest(official.RULE),
                     amendment_sha256=digest(AMENDMENT),
                     closeout_runner_sha256=digest(Path(__file__)),
                     reserved_test_labels_loaded=False,
                     heldout_path_labels_loaded=False,
                     free_runs=[], untouched_completed=[])
        for job in campaign["jobs"]:
            if job["model"] != "Free":
                audit["untouched_completed"].append(dict(
                    slug=job["slug"], model_sha256=digest(output / job["slug"] / "model.pt")))
        for job, state in free:
            folder = output / job["slug"]
            old_sha = digest(folder / official.STATE_NAME)
            model, _ = build_initial_model("Free", job["seed"], recipe, manifest,
                                            table, arrays["E_fit"], arrays["S_fit"])
            model.load_state_dict(state["best_model_state"], strict=True)
            base._check_model(model)
            lbfgs = base._lbfgs_optimizer(model, recipe)
            amended = copy.deepcopy(state)
            amended.update(phase="lbfgs", review_required=False,
                           model_state=base._copy_state(model),
                           lbfgs_state=lbfgs.state_dict())
            amended["review_events"].append(dict(
                phase="adam_budget_to_lbfgs", approved_utc=audit["approved_utc"],
                adam_step=BUDGET_STEP, adam_plateau_met=False,
                best_adam_origin=state["best_origin"],
                amendment_sha256=audit["amendment_sha256"]))
            base._atomic_torch(folder / official.STATE_NAME, amended)
            audit["free_runs"].append(dict(
                slug=job["slug"], seed=job["seed"],
                original_state_sha256=old_sha,
                transitioned_state_sha256=digest(folder / official.STATE_NAME),
                best_adam_origin=state["best_origin"],
                best_adam_validation_score=state["best_score"]))
        base._atomic_json(review / "approval.json", audit)
        campaign["status"] = "running_lbfgs_closeout"
        campaign["free_closeout_audit"] = str((review / "approval.json").relative_to(output))
        campaign["free_closeout_amendment_sha256"] = audit["amendment_sha256"]
        campaign["free_closeout_started_utc"] = utc()
        for job, _ in free:
            job.update(status="pending_lbfgs", phase="lbfgs", lbfgs_call=0)
        base._atomic_json(output / "campaign_status.json", campaign)
    except Exception:
        for job, _ in free:
            backup = review / f"{job['slug']}_before.pt"
            if backup.is_file():
                shutil.copy2(backup, output / job["slug"] / official.STATE_NAME)
        shutil.copy2(review / "campaign_status_before.json", output / "campaign_status.json")
        raise
    return campaign, audit


def finish_budget_limited(output, state, model, adam, scheduler, lbfgs,
                          scales, rule, rule_path):
    if (digest(rule_path) != state["identity"]["recipe_sha256"]
            or state["identity"]["model"] != "Free"
            or state["identity"]["seed"] not in SEEDS
            or state["adam_step"] != BUDGET_STEP
            or official.adam_plateau(state, rule)
            or not official.lbfgs_plateau(state, rule)
            or not any(event.get("phase") == "adam_budget_to_lbfgs"
                       and event.get("amendment_sha256") == digest(AMENDMENT)
                       for event in state["review_events"])):
        raise RuntimeError("Free closeout conditions were not met")
    model.load_state_dict(state["best_model_state"], strict=True)
    base._check_model(model)
    checkpoint = dict(configuration=state["configuration"],
        state_dict=base._copy_state(model), strain_scale=scales["strain_scale"],
        energy_scale=scales["energy_scale"], name="Free",
        seed=state["identity"]["seed"], best_validation_score=state["best_score"],
        best_origin=state["best_origin"], identity=state["identity"],
        protocol_amendment_sha256=digest(AMENDMENT))
    base._atomic_torch(output / "model.pt", checkpoint)
    optimizer_state = lbfgs.state_dict()["state"]
    counts = next(iter(optimizer_state.values()), {}) if optimizer_state else {}
    report = dict(status=FINAL_STATUS, model="Free", seed=checkpoint["seed"],
        official_rule_sha256=digest(rule_path),
        protocol_amendment_sha256=digest(AMENDMENT),
        closeout_runner_sha256=digest(Path(__file__)),
        best_validation_score=state["best_score"], best_origin=state["best_origin"],
        adam_steps=state["adam_step"],
        adam_stop_reason="budget_cap_200000_without_plateau",
        adam_final_learning_rate=adam.param_groups[0]["lr"],
        adam_scheduler_reductions=state["scheduler_reductions"],
        lbfgs_outer_calls=state["lbfgs_call"],
        lbfgs_stop_reason="validation_plateau",
        lbfgs_internal_iterations=counts.get("n_iter", 0),
        lbfgs_function_evaluations=counts.get("func_evals", 0),
        review_events=state["review_events"], elapsed_seconds=state["elapsed_seconds"],
        validation_history=state["validation_history"],
        accessed_label_arrays=["E_fit", "S_fit", "W_fit", "E_reference",
                               "S_reference", "W_reference", "D_reference",
                               "E_validation", "S_validation"],
        test_labels_loaded=False, path_labels_loaded=False,
        model_sha256=digest(output / "model.pt"), identity=state["identity"])
    base._atomic_json(output / "run_report.json", report)
    state["phase"] = "complete"
    state["review_required"] = False
    official._save_state(output, state, model, adam, scheduler, lbfgs)
    return report


def worker(output, seed):
    official._finish = finish_budget_limited
    result = official.run("Free", seed, output / f"free_seed{seed}", resume=True)
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ("identity", "validation_history")},
                     allow_nan=False), flush=True)
    return 0 if result["status"] in (FINAL_STATUS, "needs_review") else 1


def verify_free(folder, seed, rule_sha):
    report_path, model_path = folder / "run_report.json", folder / "model.pt"
    if report_path.is_file() and model_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if (report.get("status") != FINAL_STATUS or report.get("model") != "Free"
                or report.get("seed") != seed
                or report.get("official_rule_sha256") != rule_sha
                or report.get("protocol_amendment_sha256") != digest(AMENDMENT)
                or report.get("model_sha256") != digest(model_path)
                or report.get("adam_stop_reason") != "budget_cap_200000_without_plateau"
                or report.get("lbfgs_stop_reason") != "validation_plateau"
                or report.get("test_labels_loaded") is not False
                or report.get("path_labels_loaded") is not False):
            raise ValueError(f"Invalid final Free report: {folder.name}")
        return FINAL_STATUS, dict(model_sha256=report["model_sha256"],
            best_validation_score=report["best_validation_score"],
            adam_steps=report["adam_steps"], lbfgs_outer_calls=report["lbfgs_outer_calls"],
            adam_stop_reason=report["adam_stop_reason"],
            lbfgs_stop_reason=report["lbfgs_stop_reason"])
    if report_path.exists() or model_path.exists():
        raise ValueError(f"Incomplete final Free artifacts: {folder.name}")
    state = torch.load(folder / official.STATE_NAME, map_location="cpu", weights_only=False)
    if (state["phase"] == "lbfgs" and state["review_required"] is True
            and state["identity"]["recipe_sha256"] == rule_sha
            and state["identity"]["seed"] == seed):
        return "needs_review", dict(phase="lbfgs", lbfgs_call=state["lbfgs_call"],
                                    best_validation_score=state["best_score"])
    raise ValueError(f"No valid final or review-paused Free state: {folder.name}")


def launch(output, workers):
    if workers != 3:
        raise ValueError("This closeout runs exactly the three declared Free seeds")
    campaign, _ = prepare(output)
    env = os.environ.copy()
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env[key] = "2"
    env["PYTHONPATH"] = os.pathsep.join((str(BASE.parents[0] / ".pydeps"), str(BASE)))
    active = {}
    for job in campaign["jobs"]:
        if job["model"] != "Free":
            continue
        log_path = output / "logs" / f"{job['slug']}_lbfgs_closeout.log"
        command = [sys.executable, "-B", "-m", "protocol.close_free_lbfgs",
                   "--worker-seed", str(job["seed"]), "--output", str(output)]
        with log_path.open("xb") as log:
            process = subprocess.Popen(command, cwd=BASE.parents[1], env=env,
                                       stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
        job.update(status="running_lbfgs", pid=process.pid, started_utc=utc(),
                   log=str(log_path.relative_to(output)))
        active[job["slug"]] = process
        base._atomic_json(output / "campaign_status.json", campaign)
    while active:
        for job in campaign["jobs"]:
            process = active.get(job["slug"])
            if process is None or process.poll() is None:
                continue
            code = process.returncode
            job.update(exit_code=code, finished_utc=utc())
            try:
                if code:
                    raise RuntimeError(f"Closeout worker exited with status {code}")
                status, details = verify_free(output / job["slug"], job["seed"],
                                              campaign["official_rule_sha256"])
                job.update(status=status, **details)
            except Exception as error:
                job.update(status="failed", error=str(error))
            del active[job["slug"]]
            base._atomic_json(output / "campaign_status.json", campaign)
        if active:
            time.sleep(2)
    statuses = [job["status"] for job in campaign["jobs"]]
    campaign["status"] = ("partial_failure" if "failed" in statuses else
                          "needs_review" if "needs_review" in statuses else
                          "complete_with_budget_limited_adam")
    campaign["finished_utc"] = utc()
    base._atomic_json(output / "campaign_status.json", campaign)
    return 1 if campaign["status"] == "partial_failure" else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--worker-seed", type=int, choices=SEEDS)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.worker_seed is not None:
        return worker(output, args.worker_seed)
    if args.dry_run:
        _, _, _, _, _, free = preflight(output)
        print(json.dumps(dict(status="ready", free_seeds=[job["seed"] for job, _ in free],
                              adam_step=BUDGET_STEP, lbfgs_first_review_call=300,
                              test_labels_loaded=False, path_labels_loaded=False)))
        return 0
    return launch(output, args.workers)


if __name__ == "__main__":
    sys.exit(main())
