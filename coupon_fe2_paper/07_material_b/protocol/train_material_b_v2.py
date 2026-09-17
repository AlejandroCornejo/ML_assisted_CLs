"""Material-B official runner with prospective validation-plateau stopping.

The v1 runner and its checkpoints remain untouched. This runner reads only
fit/reference and validation arrays; reserved test/path labels stay closed.
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

AMENDMENT = BASE / "protocol/training_recipe_v2.json"


def load_recipe(features: Path, amendment_path: Path = AMENDMENT):
    """Use the v1-verified feature table; change only declared optimizer budgets."""
    parent, manifest, table = load_preparation(features, RECIPE)
    amendment = json.loads(amendment_path.read_text(encoding="utf-8"))
    if (amendment.get("id") != "material_B_training_amendment_v2"
            or amendment.get("parent_recipe") != RECIPE.name
            or amendment.get("parent_recipe_sha256") != digest(RECIPE)):
        raise ValueError("Amendment does not match the immutable v1 recipe")
    adam = amendment["adam"]
    lbfgs = amendment["lbfgs"]
    if (adam["minimum_steps"] != parent["adam"]["maximum_steps"]
            or lbfgs["minimum_outer_calls"] != parent["lbfgs"]["outer_calls"]
            or adam["minimum_steps"] >= adam["maximum_steps"]
            or lbfgs["minimum_outer_calls"] >= lbfgs["maximum_outer_calls"]
            or not 0 < adam["material_improvement_relative"] < 1
            or not 0 < lbfgs["material_improvement_relative"] < 1
            or not isinstance(adam["plateau_patience_validation_calls"], int)
            or adam["plateau_patience_validation_calls"] < 1
            or not isinstance(lbfgs["plateau_patience_outer_calls"], int)
            or lbfgs["plateau_patience_outer_calls"] < 1
            or adam["maximum_steps"] % parent["adam"]["validation_interval_steps"]
            or adam["minimum_steps"] % parent["adam"]["validation_interval_steps"]
            or adam["require_scheduler_minimum_lr"] is not True):
        raise ValueError("Invalid prospective stopping amendment")
    recipe = copy.deepcopy(parent)
    recipe["adam"]["maximum_steps"] = adam["maximum_steps"]
    recipe["adam"]["early_stopping"] = True
    recipe["lbfgs"]["outer_calls"] = lbfgs["maximum_outer_calls"]
    recipe["lbfgs"]["maximum_iterations_total"] = (
        lbfgs["maximum_outer_calls"] * recipe["lbfgs"]["maximum_iterations_per_call"])
    return recipe, amendment, manifest, table


def last_material_gain(history, phase, baseline, relative_threshold):
    """Return last cumulative material-gain index and its best-score anchor.

    A sequence of individually small gains may jointly cross the threshold.
    Recomputing this from saved validation history makes resume deterministic.
    """
    anchor = float(baseline)
    best = anchor
    last_index = 0
    for event in history:
        if event["phase"] != phase:
            continue
        best = min(best, float(event["score"]))
        if anchor > 0 and best <= anchor * (1.0 - relative_threshold):
            anchor = best
            last_index = int(event["index"])
    return last_index, anchor


def adam_stop_reason(state, optimizer, recipe, amendment):
    step = state["adam_step"]
    rule = amendment["adam"]
    interval = recipe["adam"]["validation_interval_steps"]
    if step < rule["minimum_steps"] or step % interval:
        return None
    baseline = state["validation_history"][0]["score"]
    last, _ = last_material_gain(state["validation_history"], "adam", baseline,
                                 rule["material_improvement_relative"])
    lr = optimizer.param_groups[0]["lr"]
    min_lr = recipe["adam"]["scheduler"]["min_lr"]
    if (step-last >= rule["plateau_patience_validation_calls"]*interval
            and lr <= min_lr*(1.0+1e-10)):
        return "validation_plateau"
    if step >= rule["maximum_steps"]:
        return "safety_cap"
    return None


def lbfgs_stop_reason(state, amendment):
    call = state["lbfgs_call"]
    rule = amendment["lbfgs"]
    if call < rule["minimum_outer_calls"]:
        return None
    adam_scores = [event["score"] for event in state["validation_history"]
                   if event["phase"] != "lbfgs"]
    baseline = min(adam_scores)
    last, _ = last_material_gain(state["validation_history"], "lbfgs", baseline,
                                 rule["material_improvement_relative"])
    if call-last >= rule["plateau_patience_outer_calls"]:
        return "validation_plateau"
    if call >= rule["maximum_outer_calls"]:
        return "safety_cap"
    return None


def _finish(output, state, model, adam, scheduler, lbfgs, scales,
            recipe, amendment, amendment_path):
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
        recipe_version=2, amendment_sha256=digest(amendment_path),
        best_validation_score=state["best_score"], best_origin=state["best_origin"],
        adam_steps=state["adam_step"], adam_stop_reason=adam_stop_reason(
            state, adam, recipe, amendment), adam_final_learning_rate=adam.param_groups[0]["lr"],
        lbfgs_outer_calls=state["lbfgs_call"], lbfgs_stop_reason=lbfgs_stop_reason(
            state, amendment), lbfgs_internal_iterations=counts.get("n_iter", 0),
        lbfgs_function_evaluations=counts.get("func_evals", 0),
        elapsed_seconds=state["elapsed_seconds"], validation_history=state["validation_history"],
        accessed_label_arrays=["E_fit", "S_fit", "W_fit", "E_reference", "S_reference",
                               "W_reference", "D_reference", "E_validation", "S_validation"],
        test_labels_loaded=False, path_labels_loaded=False,
        model_sha256=digest(output/"model.pt"), identity=state["identity"])
    if report["adam_stop_reason"] is None or report["lbfgs_stop_reason"] is None:
        raise RuntimeError("Finalized before both prospective stopping rules were met")
    base._atomic_json(output/"run_report.json", report)
    state["phase"] = "complete"
    base._save_progress(output, state, model, adam, scheduler, lbfgs)
    return report


def run(name, seed, output: Path, *, resume=False, stop_after_adam_step=None,
        stop_after_lbfgs_call=None, labels=base.LABELS, features=base.FEATURES,
        amendment_path=AMENDMENT):
    recipe, amendment, manifest, table = load_recipe(features, amendment_path)
    if name not in recipe["models"]["names"] or seed not in recipe["models"]["seeds"]:
        raise ValueError("Undeclared model or seed")
    if output.resolve().is_relative_to((BASE/"results/neural_training_v1").resolve()):
        raise ValueError("v2 runs must not overwrite the v1 pilot directory")
    if any(os.environ.get(key) != "2" for key in (
            "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")):
        raise EnvironmentError("Set OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=2")
    if (stop_after_adam_step is not None and not 1 <= stop_after_adam_step <=
            amendment["adam"]["maximum_steps"]):
        raise ValueError("Invalid Adam pause point")
    if (stop_after_lbfgs_call is not None and not 1 <= stop_after_lbfgs_call <=
            amendment["lbfgs"]["maximum_outer_calls"]):
        raise ValueError("Invalid LBFGS pause point")
    arrays = base.load_training_arrays(labels, recipe)
    identity = base._identity(name, seed, amendment_path, features, labels, manifest)
    identity["parent_recipe_sha256"] = digest(RECIPE)
    identity["sources_sha256"][str(Path(__file__).resolve().relative_to(BASE.parents[1]))] = digest(Path(__file__))
    if resume:
        if not (output/base.CHECKPOINT_NAME).is_file():
            raise FileNotFoundError("No resumable v2 run state in output directory")
        if (output/"failure.json").exists():
            raise RuntimeError("A failed run cannot be silently restarted; inspect failure.json")
    else:
        output.mkdir(parents=True, exist_ok=False)

    e, s, w = (arrays[key] for key in ("E_fit", "S_fit", "W_fit"))
    scales = manifest["scales"]
    model, metadata = build_initial_model(name, seed, recipe, manifest, table, e, s)
    adam, scheduler = base._adam_optimizer(model, recipe)
    if resume:
        state, lbfgs = base._load_progress(output, identity, model, adam, scheduler, recipe)
        if (stop_after_adam_step is not None and
                (state["phase"] != "adam" or stop_after_adam_step <= state["adam_step"])):
            raise ValueError("Adam pause point must follow the saved Adam step")
        if (stop_after_lbfgs_call is not None and state["phase"] == "lbfgs" and
                stop_after_lbfgs_call <= state["lbfgs_call"]):
            raise ValueError("LBFGS pause point must follow the saved outer call")
    else:
        initial_score = base.validation_score(model, arrays["E_validation"],
                                               arrays["S_validation"], scales)
        scheduler.step(initial_score)
        state = dict(identity=identity, phase="adam", adam_step=0, lbfgs_call=0,
                     best_model_state=base._copy_state(model), best_score=initial_score,
                     best_origin=dict(phase="initialization", index=0),
                     validation_history=[dict(phase="initialization", index=0,
                                              score=initial_score)], elapsed_seconds=0.,
                     calibration_factor=metadata["calibration_factor"],
                     configuration=metadata["configuration"])
        lbfgs = None
        base._save_progress(output, state, model, adam, scheduler, lbfgs)
    started = time.perf_counter()
    try:
        if state["phase"] == "adam":
            reason = adam_stop_reason(state, adam, recipe, amendment)
            if reason is None:
                for step in range(state["adam_step"]+1, amendment["adam"]["maximum_steps"]+1):
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
                    if step % recipe["adam"]["validation_interval_steps"] == 0:
                        score = base.validation_score(model, arrays["E_validation"],
                                                      arrays["S_validation"], scales)
                        base._record_validation(state, model, score, "adam", step)
                        scheduler.step(score)
                        reason = adam_stop_reason(state, adam, recipe, amendment)
                        print(json.dumps(dict(phase="adam", step=step,
                            validation_score=score, best_score=state["best_score"],
                            learning_rate=adam.param_groups[0]["lr"])), flush=True)
                    state["elapsed_seconds"] += time.perf_counter()-started
                    started = time.perf_counter()
                    if step % 200 == 0 or reason is not None or step == stop_after_adam_step:
                        base._save_progress(output, state, model, adam, scheduler, lbfgs)
                    if step == stop_after_adam_step:
                        return dict(status="paused", phase="adam", adam_step=step,
                                    best_validation_score=state["best_score"])
                    if reason is not None:
                        break
            model.load_state_dict(state["best_model_state"], strict=True)
            lbfgs = base._lbfgs_optimizer(model, recipe)
            state["phase"] = "lbfgs"
            base._save_progress(output, state, model, adam, scheduler, lbfgs)
        reason = lbfgs_stop_reason(state, amendment)
        if reason is None:
            for call in range(state["lbfgs_call"]+1, amendment["lbfgs"]["maximum_outer_calls"]+1):
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
                reason = lbfgs_stop_reason(state, amendment)
                print(json.dumps(dict(phase="lbfgs", call=call,
                    validation_score=score, best_score=state["best_score"])), flush=True)
                state["elapsed_seconds"] += time.perf_counter()-started
                started = time.perf_counter()
                base._save_progress(output, state, model, adam, scheduler, lbfgs)
                if call == stop_after_lbfgs_call:
                    return dict(status="paused", phase="lbfgs", lbfgs_call=call,
                                best_validation_score=state["best_score"])
                if reason is not None:
                    break
        return _finish(output, state, model, adam, scheduler, lbfgs, scales,
                       recipe, amendment, amendment_path)
    except Exception as error:
        base._atomic_json(output/"failure.json", dict(status="failed", model=name, seed=seed,
            last_complete_adam_step=state["adam_step"],
            last_complete_lbfgs_call=state["lbfgs_call"], error=repr(error),
            traceback=traceback.format_exc(), identity=identity))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True,
                        choices=["Free", "ICNN-fixed", "ICNN-learned",
                                 "ICKAN-fixed", "ICKAN-learned"])
    parser.add_argument("--seed", required=True, type=int, choices=[16, 29, 47])
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-adam-step", type=int,
                        help="Pause for verification; never makes a final checkpoint")
    parser.add_argument("--stop-after-lbfgs-call", type=int,
                        help="Pause for verification; never makes a final checkpoint")
    args = parser.parse_args()
    result = run(args.model, args.seed, args.output, resume=args.resume,
                 stop_after_adam_step=args.stop_after_adam_step,
                 stop_after_lbfgs_call=args.stop_after_lbfgs_call)
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ("identity", "validation_history")}, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
