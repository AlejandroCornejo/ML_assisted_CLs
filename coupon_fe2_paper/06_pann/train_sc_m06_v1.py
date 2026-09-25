#!/usr/bin/env python3
"""Train one SC-RVE learned-feature model at the transferred count m=6.

The runner reads only frozen fit/reference arrays and validation stress. It is
resumable and never opens the independent test or extrapolation-probe arrays.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
MATERIAL_B = ROOT / "07_material_b"
import sys
sys.path.insert(0, str(MATERIAL_B))
sys.path.insert(0, str(ROOT / "06_pann"))

from protocol import train_material_b as base
from protocol.prepare_design import digest
from protocol.select_features import load_fit_reference
from protocol.training_setup import build_initial_model, load_preparation, training_objective

NAMES = ("ICNN-learned", "ICKAN-learned")
SEEDS = (16, 29, 47)
DEFAULT_PREPARATION = ROOT / "06_pann/results/sc_m06_learned_v1/preparation"


def load_rule(preparation: Path):
    recipe_path = preparation / "preparation_recipe.json"
    rule_path = preparation / "training_rule.json"
    labels = preparation / "training_labels.npz"
    recipe, manifest, table = load_preparation(preparation, recipe_path)
    rule = json.loads(rule_path.read_text(encoding="utf-8"))
    if (rule.get("id") != "sc_rve_m06_learned_training_v1"
            or rule.get("feature_count") != 6
            or tuple(rule.get("models", ())) != NAMES
            or tuple(rule.get("seeds", ())) != SEEDS
            or recipe["feature_selection"]["count"] != 6
            or tuple(recipe["models"]["names"]) != NAMES
            or tuple(recipe["models"]["seeds"]) != SEEDS
            or table["specs"].shape != (6, 5)
            or rule["preparation_recipe_sha256"] != digest(recipe_path)
            or rule["feature_table_sha256"] != digest(preparation / "feature_table.npz")
            or rule["labels_sha256"] != digest(labels)
            or manifest["fit_count"] != 4208 or manifest["validation_count"] != 742
            or rule["adam"]["minimum_steps"] != recipe["adam"]["maximum_steps"]
            or rule["adam"]["maximum_steps"] != 200000
            or rule["lbfgs"]["minimum_outer_calls"] != recipe["lbfgs"]["outer_calls"]
            or rule["lbfgs"]["maximum_outer_calls"] != 300):
        raise ValueError("Invalid or inconsistent SC-RVE m=6 preparation")
    return recipe, manifest, table, rule, rule_path, recipe_path, labels


def load_arrays(labels: Path, recipe: dict) -> dict[str, np.ndarray]:
    if digest(labels) != recipe["labels_sha256"]:
        raise ValueError("Frozen SC-RVE label hash changed")
    with np.load(labels, allow_pickle=False) as store:
        arrays = load_fit_reference(store, recipe["selection_allowed_arrays"])
        arrays["E_validation"] = np.asarray(store["E_validation"], dtype=np.float64)
        arrays["S_validation"] = np.asarray(store["S_validation"], dtype=np.float64)
    if (arrays["E_fit"].shape != (4208, 3) or arrays["S_fit"].shape != (4208, 3)
            or arrays["W_fit"].shape != (4208,)
            or arrays["E_validation"].shape != (742, 3)
            or arrays["S_validation"].shape != (742, 3)
            or not all(np.isfinite(value).all() for value in arrays.values())):
        raise ValueError("Invalid frozen SC-RVE fit/reference/validation arrays")
    return arrays


def last_material_gain(history, phase, baseline, relative_threshold):
    anchor = float(baseline)
    last_index = 0
    for event in history:
        if event["phase"] != phase:
            continue
        score = float(event["score"])
        if score < anchor:
            anchor = score
        if anchor > 0 and anchor <= baseline * (1.0 - relative_threshold):
            baseline = anchor
            last_index = int(event["index"])
    return last_index


def adam_stop_reason(state, optimizer, recipe, rule):
    step = state["adam_step"]
    setting = rule["adam"]
    interval = recipe["adam"]["validation_interval_steps"]
    if step < setting["minimum_steps"] or step % interval:
        return None
    baseline = state["validation_history"][0]["score"]
    last = last_material_gain(state["validation_history"], "adam", baseline,
                              setting["material_improvement_relative"])
    at_floor = optimizer.param_groups[0]["lr"] <= recipe["adam"]["scheduler"]["min_lr"] * (1 + 1e-10)
    if step - last >= setting["plateau_patience_validation_calls"] * interval and at_floor:
        return "validation_plateau"
    if step >= setting["maximum_steps"]:
        return "safety_cap"
    return None


def lbfgs_stop_reason(state, rule):
    call = state["lbfgs_call"]
    setting = rule["lbfgs"]
    if call < setting["minimum_outer_calls"]:
        return None
    baseline = min(event["score"] for event in state["validation_history"]
                   if event["phase"] != "lbfgs")
    last = last_material_gain(state["validation_history"], "lbfgs", baseline,
                              setting["material_improvement_relative"])
    if call - last >= setting["plateau_patience_outer_calls"]:
        return "validation_plateau"
    if call >= setting["maximum_outer_calls"]:
        return "safety_cap"
    return None


def finish(output, state, model, adam, scheduler, lbfgs, scales, recipe, rule, rule_path):
    model.load_state_dict(state["best_model_state"], strict=True)
    checkpoint = dict(configuration=state["configuration"], state_dict=base._copy_state(model),
        strain_scale=scales["strain_scale"], energy_scale=scales["energy_scale"],
        name=state["identity"]["model"], seed=state["identity"]["seed"],
        best_validation_score=state["best_score"], best_origin=state["best_origin"],
        identity=state["identity"])
    base._atomic_torch(output / "model.pt", checkpoint)
    optimizer_state = lbfgs.state_dict()["state"]
    counts = next(iter(optimizer_state.values()), {}) if optimizer_state else {}
    report = dict(status="complete", protocol_id=rule["id"], model=checkpoint["name"],
        seed=checkpoint["seed"], feature_count=6, training_rule_sha256=digest(rule_path),
        best_validation_score=state["best_score"], best_origin=state["best_origin"],
        adam_steps=state["adam_step"], adam_stop_reason=adam_stop_reason(state, adam, recipe, rule),
        adam_final_learning_rate=adam.param_groups[0]["lr"],
        lbfgs_outer_calls=state["lbfgs_call"], lbfgs_stop_reason=lbfgs_stop_reason(state, rule),
        lbfgs_internal_iterations=counts.get("n_iter", 0),
        lbfgs_function_evaluations=counts.get("func_evals", 0),
        elapsed_seconds=state["elapsed_seconds"], validation_history=state["validation_history"],
        accessed_label_arrays=["E_fit", "S_fit", "W_fit", "E_reference", "S_reference",
                               "W_reference", "D_reference", "E_validation", "S_validation"],
        test_labels_loaded=False, probe_labels_loaded=False,
        model_sha256=digest(output / "model.pt"), identity=state["identity"])
    if report["adam_stop_reason"] is None or report["lbfgs_stop_reason"] is None:
        raise RuntimeError("Finalized before both stopping rules were met")
    base._atomic_json(output / "run_report.json", report)
    state["phase"] = "complete"
    base._save_progress(output, state, model, adam, scheduler, lbfgs)
    return report


def run(name: str, seed: int, output: Path, preparation: Path, *, resume=False,
        stop_after_adam_step=None, stop_after_lbfgs_call=None):
    recipe, manifest, table, rule, rule_path, recipe_path, labels = load_rule(preparation)
    if name not in NAMES or seed not in SEEDS:
        raise ValueError("Undeclared SC-RVE m=6 model or seed")
    if any(os.environ.get(key) != "2" for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")):
        raise EnvironmentError("Set OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=2")
    arrays = load_arrays(labels, recipe)
    identity = base._identity(name, seed, recipe_path, preparation, labels, manifest)
    identity.update(protocol_id=rule["id"], feature_count=6,
                    training_rule_sha256=digest(rule_path))
    identity["sources_sha256"][str(Path(__file__).resolve().relative_to(ROOT))] = digest(Path(__file__))
    if resume:
        if not (output / base.CHECKPOINT_NAME).is_file() or (output / "failure.json").exists():
            raise RuntimeError("Run is not safely resumable")
    else:
        output.mkdir(parents=True, exist_ok=False)

    e, s, w = (arrays[key] for key in ("E_fit", "S_fit", "W_fit"))
    scales = manifest["scales"]
    model, metadata = build_initial_model(name, seed, recipe, manifest, table, e, s)
    adam, scheduler = base._adam_optimizer(model, recipe)
    if resume:
        state, lbfgs = base._load_progress(output, identity, model, adam, scheduler, recipe)
    else:
        initial = base.validation_score(model, arrays["E_validation"], arrays["S_validation"], scales)
        scheduler.step(initial)
        state = dict(identity=identity, phase="adam", adam_step=0, lbfgs_call=0,
            best_model_state=base._copy_state(model), best_score=initial,
            best_origin=dict(phase="initialization", index=0),
            validation_history=[dict(phase="initialization", index=0, score=initial)],
            elapsed_seconds=0.0, calibration_factor=metadata["calibration_factor"],
            configuration=metadata["configuration"])
        lbfgs = None
        base._save_progress(output, state, model, adam, scheduler, lbfgs)

    started = time.perf_counter()
    try:
        if state["phase"] == "adam":
            reason = adam_stop_reason(state, adam, recipe, rule)
            while reason is None:
                step = state["adam_step"] + 1
                adam.zero_grad(set_to_none=True)
                total, _ = training_objective(model, e, s, w, scales,
                                               table["reference_tangent"], recipe)
                total.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(),
                    recipe["adam"]["gradient_clip_norm"],
                    norm_type=recipe["adam"]["clip_norm_type"], error_if_nonfinite=True)
                if not torch.isfinite(norm):
                    raise RuntimeError("Nonfinite Adam gradient norm")
                adam.step(); base._check_model(model); state["adam_step"] = step
                if step % recipe["adam"]["validation_interval_steps"] == 0:
                    score = base.validation_score(model, arrays["E_validation"],
                                                  arrays["S_validation"], scales)
                    base._record_validation(state, model, score, "adam", step)
                    scheduler.step(score)
                    reason = adam_stop_reason(state, adam, recipe, rule)
                    print(json.dumps(dict(phase="adam", step=step, validation_score=score,
                        best_score=state["best_score"], learning_rate=adam.param_groups[0]["lr"])), flush=True)
                state["elapsed_seconds"] += time.perf_counter() - started
                started = time.perf_counter()
                if step % 200 == 0 or reason is not None or step == stop_after_adam_step:
                    base._save_progress(output, state, model, adam, scheduler, lbfgs)
                if step == stop_after_adam_step:
                    return dict(status="paused", phase="adam", adam_step=step)
            model.load_state_dict(state["best_model_state"], strict=True)
            lbfgs = base._lbfgs_optimizer(model, recipe)
            state["phase"] = "lbfgs"
            base._save_progress(output, state, model, adam, scheduler, lbfgs)

        reason = lbfgs_stop_reason(state, rule)
        while reason is None:
            call = state["lbfgs_call"] + 1
            def closure():
                lbfgs.zero_grad(set_to_none=True)
                total, _ = training_objective(model, e, s, w, scales,
                                               table["reference_tangent"], recipe)
                total.backward()
                for parameter in model.parameters():
                    if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                        raise RuntimeError("Nonfinite L-BFGS gradient")
                return total
            lbfgs.step(closure); base._check_model(model); state["lbfgs_call"] = call
            score = base.validation_score(model, arrays["E_validation"], arrays["S_validation"], scales)
            base._record_validation(state, model, score, "lbfgs", call)
            reason = lbfgs_stop_reason(state, rule)
            print(json.dumps(dict(phase="lbfgs", call=call, validation_score=score,
                                  best_score=state["best_score"])), flush=True)
            state["elapsed_seconds"] += time.perf_counter() - started
            started = time.perf_counter()
            base._save_progress(output, state, model, adam, scheduler, lbfgs)
            if call == stop_after_lbfgs_call:
                return dict(status="paused", phase="lbfgs", lbfgs_call=call)
        return finish(output, state, model, adam, scheduler, lbfgs, scales,
                      recipe, rule, rule_path)
    except Exception as error:
        base._atomic_json(output / "failure.json", dict(status="failed", model=name, seed=seed,
            feature_count=6, last_complete_adam_step=state["adam_step"],
            last_complete_lbfgs_call=state["lbfgs_call"], error=repr(error),
            traceback=traceback.format_exc(), identity=identity))
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=NAMES)
    parser.add_argument("--seed", type=int, required=True, choices=SEEDS)
    parser.add_argument("--preparation", type=Path, default=DEFAULT_PREPARATION)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-adam-step", type=int)
    parser.add_argument("--stop-after-lbfgs-call", type=int)
    args = parser.parse_args()
    result = run(args.model, args.seed, args.output.resolve(), args.preparation.resolve(),
                 resume=args.resume, stop_after_adam_step=args.stop_after_adam_step,
                 stop_after_lbfgs_call=args.stop_after_lbfgs_call)
    print(json.dumps({k: v for k, v in result.items()
                      if k not in ("identity", "validation_history")}, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
