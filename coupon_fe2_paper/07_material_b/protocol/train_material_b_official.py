"""Official Material-B training: validation stops, review boundaries pause.

Only fit/reference and validation arrays are read. A review boundary never
produces a final model. Resume after review preserves the exact current model,
optimizer, scheduler, LBFGS history and RNG states.
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

RULE = BASE / "protocol/training_recipe.json"
STATE_NAME = "run_state.pt"


def load_rule(features: Path, rule_path: Path = RULE):
    """Verify the frozen feature selection before applying the official rule."""
    parent, manifest, table = load_preparation(features, RECIPE)
    rule = json.loads(rule_path.read_text(encoding="utf-8"))
    if (rule.get("id") != "material_B_official_training"
            or rule.get("parent_preparation_recipe") != RECIPE.name
            or rule.get("parent_preparation_sha256") != digest(RECIPE)):
        raise ValueError("Official rule does not match frozen feature preparation")
    adam_rule, lbfgs_rule = rule["adam"], rule["lbfgs"]
    interval = parent["adam"]["validation_interval_steps"]
    if (rule["material_improvement_relative"] <= 0
            or rule["material_improvement_relative"] >= 1
            or adam_rule["minimum_steps"] != parent["adam"]["maximum_steps"]
            or adam_rule["validation_interval_steps"] != interval
            or adam_rule["minimum_steps"] >= adam_rule["first_review_step"]
            or adam_rule["first_review_step"] % interval
            or adam_rule["review_extension_steps"] <= 0
            or adam_rule["review_extension_steps"] % interval
            or adam_rule["plateau_patience_validation_calls"] <= 0
            or adam_rule["minimum_scheduler_reductions"] < 1
            or adam_rule["checks_after_last_reduction"] <= 0
            or lbfgs_rule["minimum_outer_calls"] != parent["lbfgs"]["outer_calls"]
            or lbfgs_rule["minimum_outer_calls"] >= lbfgs_rule["first_review_call"]
            or lbfgs_rule["plateau_patience_outer_calls"] <= 0
            or lbfgs_rule["review_extension_calls"] <= 0
            or adam_rule["scheduler"]["type"] != "ReduceLROnPlateau"
            or adam_rule["scheduler"]["threshold_mode"] != "rel"
            or adam_rule["scheduler"]["threshold"] != rule["material_improvement_relative"]):
        raise ValueError("Invalid official stopping and scheduler rule")
    recipe = copy.deepcopy(parent)
    recipe["adam"]["scheduler"] = copy.deepcopy(adam_rule["scheduler"])
    recipe["adam"]["maximum_steps"] = adam_rule["first_review_step"]
    recipe["adam"]["early_stopping"] = True
    recipe["lbfgs"]["outer_calls"] = lbfgs_rule["first_review_call"]
    recipe["lbfgs"]["maximum_iterations_total"] = (
        lbfgs_rule["first_review_call"] * recipe["lbfgs"]["maximum_iterations_per_call"])
    return recipe, rule, manifest, table


def last_material_gain(history, phase, baseline, relative_threshold):
    anchor = float(baseline)
    best = anchor
    last_index = 0
    for event in history:
        if event["phase"] != phase:
            continue
        best = min(best, float(event["score"]))
        if anchor > 0 and best <= anchor * (1.0-relative_threshold):
            anchor = best
            last_index = int(event["index"])
    return last_index, anchor


def adam_plateau(state, rule):
    settings = rule["adam"]
    step = state["adam_step"]
    interval = settings["validation_interval_steps"]
    if step < settings["minimum_steps"] or step % interval:
        return False
    baseline = state["validation_history"][0]["score"]
    last, _ = last_material_gain(state["validation_history"], "adam", baseline,
                                 rule["material_improvement_relative"])
    return (step-last >= settings["plateau_patience_validation_calls"]*interval
            and state["scheduler_reductions"] >= settings["minimum_scheduler_reductions"]
            and step-state["last_reduction_step"] >=
            settings["checks_after_last_reduction"]*interval)


def lbfgs_plateau(state, rule):
    call = state["lbfgs_call"]
    settings = rule["lbfgs"]
    if call < settings["minimum_outer_calls"]:
        return False
    baseline = min(event["score"] for event in state["validation_history"]
                   if event["phase"] != "lbfgs")
    last, _ = last_material_gain(state["validation_history"], "lbfgs", baseline,
                                 rule["material_improvement_relative"])
    return call-last >= settings["plateau_patience_outer_calls"]


def _save_state(output, state, model, adam, scheduler, lbfgs):
    payload = dict(format_version=3, identity=state["identity"], phase=state["phase"],
        review_required=state["review_required"], review_events=state["review_events"],
        adam_limit=state["adam_limit"], lbfgs_limit=state["lbfgs_limit"],
        scheduler_reductions=state["scheduler_reductions"],
        last_reduction_step=state["last_reduction_step"],
        adam_step=state["adam_step"], lbfgs_call=state["lbfgs_call"],
        model_state=base._copy_state(model), best_model_state=state["best_model_state"],
        best_score=state["best_score"], best_origin=state["best_origin"],
        validation_history=state["validation_history"], adam_state=adam.state_dict(),
        scheduler_state=scheduler.state_dict(),
        lbfgs_state=None if lbfgs is None else lbfgs.state_dict(),
        rng_state=base._rng_state(), elapsed_seconds=state["elapsed_seconds"],
        calibration_factor=state["calibration_factor"],
        configuration=state["configuration"])
    base._atomic_torch(output/STATE_NAME, payload)


def _load_state(output, identity, model, adam, scheduler, recipe):
    payload = torch.load(output/STATE_NAME, map_location="cpu", weights_only=False)
    if payload.get("format_version") != 3 or payload.get("identity") != identity:
        raise ValueError("Resume identity, sources, data or official rule changed")
    if payload["phase"] == "complete":
        raise ValueError("Official run is already complete")
    model.load_state_dict(payload["model_state"], strict=True)
    adam.load_state_dict(payload["adam_state"])
    scheduler.load_state_dict(payload["scheduler_state"])
    lbfgs = None
    if payload["phase"] == "lbfgs":
        lbfgs = base._lbfgs_optimizer(model, recipe)
        lbfgs.load_state_dict(payload["lbfgs_state"])
    base._restore_rng(payload["rng_state"])
    keys = ("identity", "phase", "review_required", "review_events",
            "adam_limit", "lbfgs_limit", "scheduler_reductions",
            "last_reduction_step", "adam_step", "lbfgs_call", "best_model_state",
            "best_score", "best_origin", "validation_history", "elapsed_seconds",
            "calibration_factor", "configuration")
    return {key: payload[key] for key in keys}, lbfgs


def _authorize_extension(state, rule):
    if not state["review_required"]:
        raise ValueError("No review pause is awaiting approval")
    if state["phase"] == "adam":
        old_limit = state["adam_limit"]
        new_limit = old_limit + rule["adam"]["review_extension_steps"]
        state["adam_limit"] = new_limit
    elif state["phase"] == "lbfgs":
        old_limit = state["lbfgs_limit"]
        new_limit = old_limit + rule["lbfgs"]["review_extension_calls"]
        state["lbfgs_limit"] = new_limit
    else:
        raise ValueError("Cannot extend a completed run")
    state["review_events"].append(dict(phase=state["phase"],
        old_limit=old_limit, new_limit=new_limit,
        approved_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())))
    state["review_required"] = False


def _finish(output, state, model, adam, scheduler, lbfgs, scales, rule, rule_path):
    if digest(rule_path) != state["identity"]["recipe_sha256"]:
        raise RuntimeError("Official rule changed during training")
    if not adam_plateau(state, rule):
        raise RuntimeError("Adam never met the official validation-stop rule")
    if not lbfgs_plateau(state, rule):
        raise RuntimeError("LBFGS never met the official validation-stop rule")
    model.load_state_dict(state["best_model_state"], strict=True)
    checkpoint = dict(configuration=state["configuration"],
        state_dict=base._copy_state(model), strain_scale=scales["strain_scale"],
        energy_scale=scales["energy_scale"], name=state["identity"]["model"],
        seed=state["identity"]["seed"], best_validation_score=state["best_score"],
        best_origin=state["best_origin"], identity=state["identity"])
    base._atomic_torch(output/"model.pt", checkpoint)
    optimizer_state = lbfgs.state_dict()["state"]
    counts = next(iter(optimizer_state.values()), {}) if optimizer_state else {}
    report = dict(status="complete", model=checkpoint["name"], seed=checkpoint["seed"],
        official_rule_sha256=digest(rule_path),
        best_validation_score=state["best_score"], best_origin=state["best_origin"],
        adam_steps=state["adam_step"], adam_stop_reason="validation_plateau",
        adam_final_learning_rate=adam.param_groups[0]["lr"],
        adam_scheduler_reductions=state["scheduler_reductions"],
        lbfgs_outer_calls=state["lbfgs_call"], lbfgs_stop_reason="validation_plateau",
        lbfgs_internal_iterations=counts.get("n_iter", 0),
        lbfgs_function_evaluations=counts.get("func_evals", 0),
        review_events=state["review_events"], elapsed_seconds=state["elapsed_seconds"],
        validation_history=state["validation_history"],
        accessed_label_arrays=["E_fit", "S_fit", "W_fit", "E_reference", "S_reference",
                               "W_reference", "D_reference", "E_validation", "S_validation"],
        test_labels_loaded=False, path_labels_loaded=False,
        model_sha256=digest(output/"model.pt"), identity=state["identity"])
    base._atomic_json(output/"run_report.json", report)
    state["phase"] = "complete"
    state["review_required"] = False
    _save_state(output, state, model, adam, scheduler, lbfgs)
    return report


def run(name, seed, output: Path, *, resume=False, approve_next_block=False,
        stop_after_adam_step=None, stop_after_lbfgs_call=None,
        labels=base.LABELS, features=base.FEATURES, rule_path=RULE):
    recipe, rule, manifest, table = load_rule(features, rule_path)
    if name not in recipe["models"]["names"] or seed not in recipe["models"]["seeds"]:
        raise ValueError("Undeclared model or seed")
    if (not resume and approve_next_block) or (resume and not (output/STATE_NAME).is_file()):
        raise ValueError("Review extension requires an existing resumable run")
    if any(os.environ.get(key) != "2" for key in (
            "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")):
        raise EnvironmentError("Set OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=2")
    if (stop_after_adam_step is not None and stop_after_adam_step < 1
            or stop_after_lbfgs_call is not None and stop_after_lbfgs_call < 1):
        raise ValueError("Invalid smoke pause point")
    arrays = base.load_training_arrays(labels, recipe)
    identity = base._identity(name, seed, rule_path, features, labels, manifest)
    identity["parent_preparation_sha256"] = digest(RECIPE)
    identity["sources_sha256"][str(Path(__file__).resolve().relative_to(BASE.parents[1]))] = digest(Path(__file__))
    if resume:
        if (output/"failure.json").exists():
            raise RuntimeError("A failed run requires inspection before restart")
    else:
        output.mkdir(parents=True, exist_ok=False)

    e, s, w = (arrays[key] for key in ("E_fit", "S_fit", "W_fit"))
    scales = manifest["scales"]
    model, metadata = build_initial_model(name, seed, recipe, manifest, table, e, s)
    adam, scheduler = base._adam_optimizer(model, recipe)
    if resume:
        state, lbfgs = _load_state(output, identity, model, adam, scheduler, recipe)
        if state["review_required"]:
            if not approve_next_block:
                return dict(status="needs_review", phase=state["phase"],
                            adam_step=state["adam_step"], lbfgs_call=state["lbfgs_call"])
            _authorize_extension(state, rule)
            _save_state(output, state, model, adam, scheduler, lbfgs)
        elif approve_next_block:
            raise ValueError("Extension requested without a review pause")
        if (stop_after_adam_step is not None and
                (state["phase"] != "adam" or stop_after_adam_step <= state["adam_step"])):
            raise ValueError("Adam smoke pause must follow the saved step")
        if (stop_after_lbfgs_call is not None and state["phase"] == "lbfgs" and
                stop_after_lbfgs_call <= state["lbfgs_call"]):
            raise ValueError("LBFGS smoke pause must follow the saved call")
    else:
        initial_score = base.validation_score(model, arrays["E_validation"],
                                               arrays["S_validation"], scales)
        scheduler.step(initial_score)
        state = dict(identity=identity, phase="adam", review_required=False,
            review_events=[], adam_limit=rule["adam"]["first_review_step"],
            lbfgs_limit=rule["lbfgs"]["first_review_call"],
            scheduler_reductions=0, last_reduction_step=0,
            adam_step=0, lbfgs_call=0, best_model_state=base._copy_state(model),
            best_score=initial_score, best_origin=dict(phase="initialization", index=0),
            validation_history=[dict(phase="initialization", index=0, score=initial_score)],
            elapsed_seconds=0., calibration_factor=metadata["calibration_factor"],
            configuration=metadata["configuration"])
        lbfgs = None
        _save_state(output, state, model, adam, scheduler, lbfgs)
    started = time.perf_counter()
    try:
        if state["phase"] == "adam":
            stopped = adam_plateau(state, rule)
            if not stopped:
                for step in range(state["adam_step"]+1, state["adam_limit"]+1):
                    adam.zero_grad(set_to_none=True)
                    total, _ = training_objective(model, e, s, w, scales,
                                                  table["reference_tangent"], recipe)
                    total.backward()
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(),
                        recipe["adam"]["gradient_clip_norm"],
                        norm_type=recipe["adam"]["clip_norm_type"], error_if_nonfinite=True)
                    if not torch.isfinite(norm):
                        raise RuntimeError("Nonfinite Adam gradient norm")
                    adam.step()
                    base._check_model(model)
                    state["adam_step"] = step
                    if step % rule["adam"]["validation_interval_steps"] == 0:
                        score = base.validation_score(model, arrays["E_validation"],
                                                      arrays["S_validation"], scales)
                        base._record_validation(state, model, score, "adam", step)
                        old_lr = adam.param_groups[0]["lr"]
                        scheduler.step(score)
                        new_lr = adam.param_groups[0]["lr"]
                        if new_lr < old_lr:
                            state["scheduler_reductions"] += 1
                            state["last_reduction_step"] = step
                        stopped = adam_plateau(state, rule)
                        print(json.dumps(dict(phase="adam", step=step,
                            validation_score=score, best_score=state["best_score"],
                            learning_rate=new_lr)), flush=True)
                    state["elapsed_seconds"] += time.perf_counter()-started
                    started = time.perf_counter()
                    if step % 200 == 0 or stopped or step == stop_after_adam_step:
                        _save_state(output, state, model, adam, scheduler, lbfgs)
                    if step == stop_after_adam_step:
                        return dict(status="smoke_paused", phase="adam", adam_step=step)
                    if stopped:
                        break
            if not stopped:
                state["review_required"] = True
                _save_state(output, state, model, adam, scheduler, lbfgs)
                return dict(status="needs_review", phase="adam", adam_step=state["adam_step"],
                            best_validation_score=state["best_score"])
            model.load_state_dict(state["best_model_state"], strict=True)
            lbfgs = base._lbfgs_optimizer(model, recipe)
            state["phase"] = "lbfgs"
            _save_state(output, state, model, adam, scheduler, lbfgs)
        stopped = lbfgs_plateau(state, rule)
        if not stopped:
            for call in range(state["lbfgs_call"]+1, state["lbfgs_limit"]+1):
                def closure():
                    lbfgs.zero_grad(set_to_none=True)
                    total, _ = training_objective(model, e, s, w, scales,
                                                  table["reference_tangent"], recipe)
                    total.backward()
                    for parameter in model.parameters():
                        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                            raise RuntimeError("Nonfinite LBFGS gradient")
                    return total

                lbfgs.step(closure)
                base._check_model(model)
                state["lbfgs_call"] = call
                score = base.validation_score(model, arrays["E_validation"],
                                              arrays["S_validation"], scales)
                base._record_validation(state, model, score, "lbfgs", call)
                stopped = lbfgs_plateau(state, rule)
                print(json.dumps(dict(phase="lbfgs", call=call,
                    validation_score=score, best_score=state["best_score"])), flush=True)
                state["elapsed_seconds"] += time.perf_counter()-started
                started = time.perf_counter()
                _save_state(output, state, model, adam, scheduler, lbfgs)
                if call == stop_after_lbfgs_call:
                    return dict(status="smoke_paused", phase="lbfgs", lbfgs_call=call)
                if stopped:
                    break
        if not stopped:
            state["review_required"] = True
            _save_state(output, state, model, adam, scheduler, lbfgs)
            return dict(status="needs_review", phase="lbfgs", lbfgs_call=state["lbfgs_call"],
                        best_validation_score=state["best_score"])
        return _finish(output, state, model, adam, scheduler, lbfgs, scales,
                       rule, rule_path)
    except Exception as error:
        base._atomic_json(output/"failure.json", dict(status="failed", model=name, seed=seed,
            adam_step=state["adam_step"], lbfgs_call=state["lbfgs_call"],
            error=repr(error), traceback=traceback.format_exc(), identity=identity))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True,
                        choices=["Free", "ICNN-fixed", "ICNN-learned",
                                 "ICKAN-fixed", "ICKAN-learned"])
    parser.add_argument("--seed", required=True, type=int, choices=[16, 29, 47])
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--approve-next-block", action="store_true")
    parser.add_argument("--stop-after-adam-step", type=int)
    parser.add_argument("--stop-after-lbfgs-call", type=int)
    args = parser.parse_args()
    result = run(args.model, args.seed, args.output, resume=args.resume,
        approve_next_block=args.approve_next_block,
        stop_after_adam_step=args.stop_after_adam_step,
        stop_after_lbfgs_call=args.stop_after_lbfgs_call)
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ("identity", "validation_history")}, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
