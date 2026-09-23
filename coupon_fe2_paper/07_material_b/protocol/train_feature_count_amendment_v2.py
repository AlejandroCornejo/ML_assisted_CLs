"""Train one constrained low-count multicavity amendment model.

The runner is isolated from the retained m=8,16,24,32 campaign.  It reads frozen
fit/reference labels plus validation stress for stopping, never test or paths.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import time
import traceback
from pathlib import Path

import torch

from protocol import train_material_b as base
from protocol.prepare_design import digest
from protocol.select_features import BASE, RECIPE
from protocol.training_setup import build_initial_model, load_preparation, training_objective

RULE = BASE / "protocol/feature_count_amendment_v2_training_rule.json"
NAMES = ("ICNN-fixed", "ICNN-learned", "ICKAN-fixed", "ICKAN-learned")


def load_rule(features: Path, rule_path: Path = RULE):
    preparation_path = features / "preparation_recipe.json"
    recipe, manifest, table = load_preparation(features, preparation_path)
    rule = json.loads(rule_path.read_text(encoding="utf-8"))
    feature_count = int(manifest.get("feature_count", -1))
    adam, lbfgs = rule["adam"], rule["lbfgs"]
    if (rule.get("id") != "material_B_feature_count_amendment_v2_training"
            or rule.get("parent_preparation_recipe") != RECIPE.name
            or rule.get("parent_preparation_recipe_sha256") != digest(RECIPE)
            or feature_count not in rule["feature_counts"]
            or recipe["feature_selection"]["count"] != feature_count
            or tuple(recipe["models"]["names"]) != tuple(rule["models"])
            or tuple(recipe["models"]["seeds"]) != tuple(rule["seeds"])
            or tuple(rule["models"]) != NAMES
            or table["specs"].shape != (feature_count, 5)
            or adam["minimum_steps"] != recipe["adam"]["maximum_steps"]
            or adam["maximum_steps"] != 200000
            or adam["minimum_steps"] >= adam["maximum_steps"]
            or adam["maximum_steps"] % recipe["adam"]["validation_interval_steps"]
            or not 0 < adam["material_improvement_relative"] < 1
            or not isinstance(adam["plateau_patience_validation_calls"], int)
            or adam["plateau_patience_validation_calls"] < 1
            or adam["require_scheduler_minimum_lr"] is not True
            or lbfgs["minimum_outer_calls"] != recipe["lbfgs"]["outer_calls"]
            or lbfgs["minimum_outer_calls"] >= lbfgs["maximum_outer_calls"]
            or not 0 < lbfgs["material_improvement_relative"] < 1
            or not isinstance(lbfgs["plateau_patience_outer_calls"], int)
            or lbfgs["plateau_patience_outer_calls"] < 1):
        raise ValueError("Invalid low-count amendment rule or preparation")
    active = copy.deepcopy(recipe)
    active["adam"]["maximum_steps"] = adam["maximum_steps"]
    active["adam"]["early_stopping"] = True
    active["lbfgs"]["outer_calls"] = lbfgs["maximum_outer_calls"]
    active["lbfgs"]["maximum_iterations_total"] = (
        lbfgs["maximum_outer_calls"] * active["lbfgs"]["maximum_iterations_per_call"])
    return active, rule, manifest, table, preparation_path, feature_count


def _last_material_gain(history, phase, baseline, relative_threshold):
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
    last = _last_material_gain(state["validation_history"], "adam", baseline,
                               setting["material_improvement_relative"])
    at_floor = optimizer.param_groups[0]["lr"] <= recipe["adam"]["scheduler"]["min_lr"]*(1+1e-10)
    if step-last >= setting["plateau_patience_validation_calls"]*interval and at_floor:
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
    last = _last_material_gain(state["validation_history"], "lbfgs", baseline,
                               setting["material_improvement_relative"])
    if call-last >= setting["plateau_patience_outer_calls"]:
        return "validation_plateau"
    if call >= setting["maximum_outer_calls"]:
        return "safety_cap"
    return None


def _finish(output, state, model, adam, scheduler, lbfgs, scales, recipe, rule, rule_path):
    model.load_state_dict(state["best_model_state"], strict=True)
    checkpoint = dict(configuration=state["configuration"], state_dict=base._copy_state(model),
        strain_scale=scales["strain_scale"], energy_scale=scales["energy_scale"],
        name=state["identity"]["model"], seed=state["identity"]["seed"],
        best_validation_score=state["best_score"], best_origin=state["best_origin"],
        identity=state["identity"])
    base._atomic_torch(output / "model.pt", checkpoint)
    optimizer_state = lbfgs.state_dict()["state"]
    counts = next(iter(optimizer_state.values()), {}) if optimizer_state else {}
    report = dict(status="complete", model=checkpoint["name"], seed=checkpoint["seed"],
        feature_count=state["identity"]["feature_count"],
        amendment_rule_sha256=digest(rule_path),
        best_validation_score=state["best_score"], best_origin=state["best_origin"],
        adam_steps=state["adam_step"], adam_stop_reason=adam_stop_reason(state, adam, recipe, rule),
        adam_final_learning_rate=adam.param_groups[0]["lr"],
        lbfgs_outer_calls=state["lbfgs_call"], lbfgs_stop_reason=lbfgs_stop_reason(state, rule),
        lbfgs_internal_iterations=counts.get("n_iter", 0),
        lbfgs_function_evaluations=counts.get("func_evals", 0),
        elapsed_seconds=state["elapsed_seconds"], validation_history=state["validation_history"],
        accessed_label_arrays=["E_fit", "S_fit", "W_fit", "E_reference", "S_reference",
                               "W_reference", "D_reference", "E_validation", "S_validation"],
        test_labels_loaded=False, path_labels_loaded=False,
        model_sha256=digest(output / "model.pt"), identity=state["identity"])
    if report["adam_stop_reason"] is None or report["lbfgs_stop_reason"] is None:
        raise RuntimeError("Finalized before both stopping rules were met")
    base._atomic_json(output / "run_report.json", report)
    state["phase"] = "complete"
    base._save_progress(output, state, model, adam, scheduler, lbfgs)
    return report


def run(name, seed, output: Path, *, features: Path, rule_path: Path = RULE, resume=False,
        stop_after_adam_step=None, stop_after_lbfgs_call=None, labels=base.LABELS):
    recipe, rule, manifest, table, preparation_path, feature_count = load_rule(features, rule_path)
    if name not in NAMES or seed not in rule["seeds"]:
        raise ValueError("Undeclared amendment model or seed")
    if any(os.environ.get(key) != "2" for key in (
            "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")):
        raise EnvironmentError("Set OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=2")
    if (stop_after_adam_step is not None and not 1 <= stop_after_adam_step <= rule["adam"]["maximum_steps"]
            or stop_after_lbfgs_call is not None and not 1 <= stop_after_lbfgs_call <= rule["lbfgs"]["maximum_outer_calls"]):
        raise ValueError("Invalid smoke pause point")
    arrays = base.load_training_arrays(labels, recipe)
    identity = base._identity(name, seed, rule_path, features, labels, manifest)
    identity.update(feature_count=feature_count, preparation_recipe_sha256=digest(preparation_path),
                    parent_preparation_recipe_sha256=digest(RECIPE))
    identity["sources_sha256"][str(Path(__file__).resolve().relative_to(BASE.parents[1]))] = digest(Path(__file__))
    if resume:
        if not (output / base.CHECKPOINT_NAME).is_file():
            raise FileNotFoundError("No resumable amendment state")
        if (output / "failure.json").exists():
            raise RuntimeError("Failed amendment run requires inspection before resume")
    else:
        output.mkdir(parents=True, exist_ok=False)
    e, s, w = (arrays[key] for key in ("E_fit", "S_fit", "W_fit"))
    scales = manifest["scales"]
    model, metadata = build_initial_model(name, seed, recipe, manifest, table, e, s)
    adam, scheduler = base._adam_optimizer(model, recipe)
    if resume:
        state, lbfgs = base._load_progress(output, identity, model, adam, scheduler, recipe)
        if stop_after_adam_step is not None and (state["phase"] != "adam" or stop_after_adam_step <= state["adam_step"]):
            raise ValueError("Adam smoke pause must follow saved step")
        if stop_after_lbfgs_call is not None and state["phase"] == "lbfgs" and stop_after_lbfgs_call <= state["lbfgs_call"]:
            raise ValueError("LBFGS smoke pause must follow saved call")
    else:
        initial = base.validation_score(model, arrays["E_validation"], arrays["S_validation"], scales)
        scheduler.step(initial)
        state = dict(identity=identity, phase="adam", adam_step=0, lbfgs_call=0,
            best_model_state=base._copy_state(model), best_score=initial,
            best_origin=dict(phase="initialization", index=0),
            validation_history=[dict(phase="initialization", index=0, score=initial)],
            elapsed_seconds=0., calibration_factor=metadata["calibration_factor"],
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
                total, _ = training_objective(model, e, s, w, scales, table["reference_tangent"], recipe)
                total.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), recipe["adam"]["gradient_clip_norm"],
                    norm_type=recipe["adam"]["clip_norm_type"], error_if_nonfinite=True)
                if not torch.isfinite(norm):
                    raise RuntimeError("Nonfinite Adam gradient norm")
                adam.step(); base._check_model(model); state["adam_step"] = step
                if step % recipe["adam"]["validation_interval_steps"] == 0:
                    score = base.validation_score(model, arrays["E_validation"], arrays["S_validation"], scales)
                    base._record_validation(state, model, score, "adam", step); scheduler.step(score)
                    reason = adam_stop_reason(state, adam, recipe, rule)
                    print(json.dumps(dict(phase="adam", step=step, validation_score=score,
                        best_score=state["best_score"], learning_rate=adam.param_groups[0]["lr"])), flush=True)
                state["elapsed_seconds"] += time.perf_counter() - started; started = time.perf_counter()
                if step % 200 == 0 or reason is not None or step == stop_after_adam_step:
                    base._save_progress(output, state, model, adam, scheduler, lbfgs)
                if step == stop_after_adam_step:
                    return dict(status="paused", phase="adam", adam_step=step)
            model.load_state_dict(state["best_model_state"], strict=True)
            lbfgs = base._lbfgs_optimizer(model, recipe); state["phase"] = "lbfgs"
            base._save_progress(output, state, model, adam, scheduler, lbfgs)
        reason = lbfgs_stop_reason(state, rule)
        while reason is None:
            call = state["lbfgs_call"] + 1
            def closure():
                lbfgs.zero_grad(set_to_none=True)
                total, _ = training_objective(model, e, s, w, scales, table["reference_tangent"], recipe)
                total.backward()
                for parameter in model.parameters():
                    if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                        raise RuntimeError("Nonfinite LBFGS gradient")
                return total
            lbfgs.step(closure); base._check_model(model); state["lbfgs_call"] = call
            score = base.validation_score(model, arrays["E_validation"], arrays["S_validation"], scales)
            base._record_validation(state, model, score, "lbfgs", call); reason = lbfgs_stop_reason(state, rule)
            print(json.dumps(dict(phase="lbfgs", call=call, validation_score=score,
                best_score=state["best_score"])), flush=True)
            state["elapsed_seconds"] += time.perf_counter() - started; started = time.perf_counter()
            base._save_progress(output, state, model, adam, scheduler, lbfgs)
            if call == stop_after_lbfgs_call:
                return dict(status="paused", phase="lbfgs", lbfgs_call=call)
        return _finish(output, state, model, adam, scheduler, lbfgs, scales, recipe, rule, rule_path)
    except Exception as error:
        base._atomic_json(output / "failure.json", dict(status="failed", model=name, seed=seed,
            feature_count=feature_count, last_complete_adam_step=state["adam_step"],
            last_complete_lbfgs_call=state["lbfgs_call"], error=repr(error),
            traceback=traceback.format_exc(), identity=identity))
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=NAMES)
    parser.add_argument("--seed", type=int, required=True, choices=(16, 29, 47))
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rule", type=Path, default=RULE)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-adam-step", type=int)
    parser.add_argument("--stop-after-lbfgs-call", type=int)
    args = parser.parse_args()
    result = run(args.model, args.seed, args.output, features=args.features, rule_path=args.rule,
                 resume=args.resume, stop_after_adam_step=args.stop_after_adam_step,
                 stop_after_lbfgs_call=args.stop_after_lbfgs_call)
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ("identity", "validation_history")}, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
