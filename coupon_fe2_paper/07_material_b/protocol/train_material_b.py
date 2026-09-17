"""Resumable Material-B optimization under the frozen training recipe.

This runner only reads fit/reference/validation labels. A --stop-after-* limit
pauses a run for a smoke check; it does not produce a final model or amend the
declared optimization budget. Resume the same output directory to continue.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from protocol.prepare_design import digest
from protocol.select_features import BASE, RECIPE, load_fit_reference
from protocol.training_setup import (build_initial_model, load_preparation,
                                     normalized_response, source_paths,
                                     training_objective)

LABELS = BASE / "results/data_labels_v1.npz"
FEATURES = BASE / "results/feature_selection_v1"
CHECKPOINT_NAME = "run_state.pt"


def _atomic_torch(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".saving_", suffix=".pt",
                                     delete=False) as stream:
        temporary = Path(stream.name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=".saving_", suffix=".json", delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _copy_state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state())


def _restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])


def load_training_arrays(labels: Path, recipe):
    """Read only approved fit/reference arrays and validation strain/stress."""
    if digest(labels) != recipe["labels_sha256"]:
        raise ValueError("Unapproved or changed label file")
    with np.load(labels, allow_pickle=False) as store:
        arrays = load_fit_reference(store, recipe["selection_allowed_arrays"])
        arrays["E_validation"] = np.asarray(store["E_validation"], dtype=np.float64)
        arrays["S_validation"] = np.asarray(store["S_validation"], dtype=np.float64)
    nv = len(arrays["E_validation"])
    if (arrays["E_validation"].shape != (nv, 3)
            or arrays["S_validation"].shape != (nv, 3)
            or nv != 512 or len(arrays["E_fit"]) != 4200
            or not np.isfinite(arrays["E_validation"]).all()
            or not np.isfinite(arrays["S_validation"]).all()):
        raise ValueError("Invalid frozen fit/validation shapes or values")
    return arrays


def _identity(name, seed, recipe_path, features, labels, manifest):
    sources = source_paths()+[Path(__file__)]
    return dict(model=name, seed=seed, recipe_sha256=digest(recipe_path),
                labels_sha256=digest(labels), feature_table_sha256=digest(features/"feature_table.npz"),
                feature_manifest_sha256=digest(features/"manifest.json"),
                scales=manifest["scales"],
                sources_sha256={str(path.resolve().relative_to(BASE.parents[1])): digest(path)
                                for path in sources})


def _check_model(model):
    for name, parameter in model.named_parameters():
        if not torch.isfinite(parameter).all():
            raise RuntimeError(f"Nonfinite parameter: {name}")
    if hasattr(model, "effective_specs"):
        specs = model.effective_specs().detach()
        if not torch.isfinite(specs).all():
            raise RuntimeError("Nonfinite learned feature specification")
        _, p, q, b, c = specs.T
        if torch.any((p < .5) | (q < .5) | (b < 0) | (c < 0)
                     | (b > 2*p-1) | (c > 2*q-1)):
            raise RuntimeError("Feature convexity bounds violated")


def validation_score(model, e, s, scales):
    """Validation stress MSE divided by the frozen FIT stress denominator."""
    with torch.enable_grad():
        _, predicted = normalized_response(model, e, scales, create_graph=False)
    target = torch.as_tensor(s*scales["strain_scale"]/scales["energy_scale"],
                             dtype=torch.float64)
    score = (predicted-target).square().mean()/scales["stress_denominator"]
    if not torch.isfinite(score):
        raise RuntimeError("Nonfinite validation stress score")
    return float(score.detach())


def _record_validation(state, model, score, phase, index):
    state["validation_history"].append(dict(phase=phase, index=index, score=score))
    if score < state["best_score"]:  # Strict inequality: earliest equal score wins.
        state["best_score"] = score
        state["best_origin"] = dict(phase=phase, index=index)
        state["best_model_state"] = _copy_state(model)


def _adam_optimizer(model, recipe):
    settings = recipe["adam"]
    family = "Free" if model.__class__.__name__ == "AnisotropicFreeEnergy" else model.core_kind.upper()
    optimizer = torch.optim.Adam(model.parameters(),
        lr=settings["learning_rates"][family], betas=tuple(settings["betas"]),
        eps=settings["eps"], weight_decay=settings["weight_decay"])
    schedule = settings["scheduler"]
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
        mode=schedule["mode"], factor=schedule["factor"],
        patience=schedule["patience_validation_calls"], threshold=schedule["threshold"],
        threshold_mode=schedule["threshold_mode"], cooldown=schedule["cooldown"],
        min_lr=schedule["min_lr"], eps=schedule["eps"])
    return optimizer, scheduler


def _lbfgs_optimizer(model, recipe):
    settings = recipe["lbfgs"]
    return torch.optim.LBFGS(model.parameters(), lr=settings["learning_rate"],
        max_iter=settings["maximum_iterations_per_call"],
        max_eval=settings["maximum_evaluations_per_call"],
        history_size=settings["history_size"], tolerance_grad=settings["tolerance_grad"],
        tolerance_change=settings["tolerance_change"], line_search_fn=settings["line_search"])


def _save_progress(output, state, model, adam, scheduler, lbfgs):
    payload = dict(format_version=1, identity=state["identity"],
        phase=state["phase"], adam_step=state["adam_step"], lbfgs_call=state["lbfgs_call"],
        model_state=_copy_state(model), best_model_state=state["best_model_state"],
        best_score=state["best_score"], best_origin=state["best_origin"],
        validation_history=state["validation_history"],
        adam_state=adam.state_dict(), scheduler_state=scheduler.state_dict(),
        lbfgs_state=None if lbfgs is None else lbfgs.state_dict(),
        rng_state=_rng_state(), elapsed_seconds=state["elapsed_seconds"],
        calibration_factor=state["calibration_factor"], configuration=state["configuration"])
    _atomic_torch(output/CHECKPOINT_NAME, payload)


def _load_progress(output, expected, model, adam, scheduler, recipe):
    payload = torch.load(output/CHECKPOINT_NAME, map_location="cpu", weights_only=False)
    if payload.get("format_version") != 1 or payload.get("identity") != expected:
        raise ValueError("Resume identity, sources, data or recipe changed")
    if payload["phase"] == "complete":
        raise ValueError("Run is already complete")
    model.load_state_dict(payload["model_state"], strict=True)
    adam.load_state_dict(payload["adam_state"])
    scheduler.load_state_dict(payload["scheduler_state"])
    lbfgs = None
    if payload["phase"] == "lbfgs":
        lbfgs = _lbfgs_optimizer(model, recipe)
        lbfgs.load_state_dict(payload["lbfgs_state"])
    _restore_rng(payload["rng_state"])
    state = {key: payload[key] for key in ("identity", "phase", "adam_step", "lbfgs_call",
             "best_model_state", "best_score", "best_origin", "validation_history",
             "elapsed_seconds", "calibration_factor", "configuration")}
    return state, lbfgs


def _finish(output, state, model, adam, scheduler, lbfgs, scales):
    model.load_state_dict(state["best_model_state"], strict=True)
    checkpoint = dict(configuration=state["configuration"], state_dict=_copy_state(model),
        strain_scale=scales["strain_scale"], energy_scale=scales["energy_scale"],
        name=state["identity"]["model"], seed=state["identity"]["seed"],
        best_validation_score=state["best_score"], best_origin=state["best_origin"],
        identity=state["identity"])
    _atomic_torch(output/"model.pt", checkpoint)
    lbfgs_state = None if lbfgs is None else lbfgs.state_dict()["state"]
    lbfgs_counts = next(iter(lbfgs_state.values()), {}) if lbfgs_state else {}
    report = dict(status="complete", model=checkpoint["name"], seed=checkpoint["seed"],
        best_validation_score=state["best_score"], best_origin=state["best_origin"],
        adam_steps=state["adam_step"], lbfgs_outer_calls=state["lbfgs_call"],
        lbfgs_internal_iterations=lbfgs_counts.get("n_iter", 0),
        lbfgs_function_evaluations=lbfgs_counts.get("func_evals", 0),
        elapsed_seconds=state["elapsed_seconds"], validation_history=state["validation_history"],
        accessed_label_arrays=["E_fit", "S_fit", "W_fit", "E_reference", "S_reference",
                               "W_reference", "D_reference", "E_validation", "S_validation"],
        test_labels_loaded=False, path_labels_loaded=False,
        model_sha256=digest(output/"model.pt"), identity=state["identity"])
    _atomic_json(output/"run_report.json", report)
    # Mark complete only after both final artifacts exist. A crash beforehand
    # leaves the last resumable LBFGS state and can safely repeat this step.
    state["phase"] = "complete"
    _save_progress(output, state, model, adam, scheduler, lbfgs)
    return report


def run(name, seed, output: Path, *, resume=False, stop_after_adam_step=None,
        stop_after_lbfgs_call=None, labels=LABELS, features=FEATURES,
        recipe_path=RECIPE):
    recipe, manifest, table = load_preparation(features, recipe_path)
    if name not in recipe["models"]["names"] or seed not in recipe["models"]["seeds"]:
        raise ValueError("Undeclared model or seed")
    if any(os.environ.get(key) != "2" for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")):
        raise EnvironmentError("Set OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=2")
    max_adam = recipe["adam"]["maximum_steps"]
    max_lbfgs = recipe["lbfgs"]["outer_calls"]
    if (stop_after_adam_step is not None and not 1 <= stop_after_adam_step <= max_adam):
        raise ValueError("Invalid Adam pause point")
    if (stop_after_lbfgs_call is not None and not 1 <= stop_after_lbfgs_call <= max_lbfgs):
        raise ValueError("Invalid LBFGS pause point")
    arrays = load_training_arrays(labels, recipe)
    identity = _identity(name, seed, recipe_path, features, labels, manifest)
    if resume:
        if not (output/CHECKPOINT_NAME).is_file():
            raise FileNotFoundError("No resumable run state in output directory")
        if (output/"failure.json").exists():
            raise RuntimeError("A failed run cannot be silently restarted; inspect failure.json")
    else:
        output.mkdir(parents=True, exist_ok=False)

    e, s, w = (arrays[key] for key in ("E_fit", "S_fit", "W_fit"))
    scales = manifest["scales"]
    model, metadata = build_initial_model(name, seed, recipe, manifest, table, e, s)
    adam, scheduler = _adam_optimizer(model, recipe)
    if resume:
        state, lbfgs = _load_progress(output, identity, model, adam, scheduler, recipe)
        if (stop_after_adam_step is not None
                and (state["phase"] != "adam" or stop_after_adam_step <= state["adam_step"])):
            raise ValueError("Adam pause point must follow the saved Adam step")
        if (stop_after_lbfgs_call is not None and state["phase"] == "lbfgs"
                and stop_after_lbfgs_call <= state["lbfgs_call"]):
            raise ValueError("LBFGS pause point must follow the saved outer call")
    else:
        initial_score = validation_score(model, arrays["E_validation"],
                                         arrays["S_validation"], scales)
        scheduler.step(initial_score)
        state = dict(identity=identity, phase="adam", adam_step=0, lbfgs_call=0,
                     best_model_state=_copy_state(model), best_score=initial_score,
                     best_origin=dict(phase="initialization", index=0),
                     validation_history=[dict(phase="initialization", index=0,
                                              score=initial_score)], elapsed_seconds=0.,
                     calibration_factor=metadata["calibration_factor"],
                     configuration=metadata["configuration"])
        lbfgs = None
        _save_progress(output, state, model, adam, scheduler, lbfgs)
    started = time.perf_counter()
    try:
        if state["phase"] == "adam":
            for step in range(state["adam_step"]+1, max_adam+1):
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
                _check_model(model)
                state["adam_step"] = step
                if step % recipe["adam"]["validation_interval_steps"] == 0:
                    score = validation_score(model, arrays["E_validation"],
                                             arrays["S_validation"], scales)
                    _record_validation(state, model, score, "adam", step)
                    scheduler.step(score)
                    print(json.dumps(dict(phase="adam", step=step, validation_score=score,
                                          best_score=state["best_score"])), flush=True)
                state["elapsed_seconds"] += time.perf_counter()-started
                started = time.perf_counter()
                if step % 200 == 0 or step == stop_after_adam_step:
                    _save_progress(output, state, model, adam, scheduler, lbfgs)
                if step == stop_after_adam_step:
                    return dict(status="paused", phase="adam", adam_step=step,
                                best_validation_score=state["best_score"])
            model.load_state_dict(state["best_model_state"], strict=True)
            lbfgs = _lbfgs_optimizer(model, recipe)
            state["phase"] = "lbfgs"
            _save_progress(output, state, model, adam, scheduler, lbfgs)
        for call in range(state["lbfgs_call"]+1, max_lbfgs+1):
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
            _check_model(model)
            state["lbfgs_call"] = call
            score = validation_score(model, arrays["E_validation"],
                                     arrays["S_validation"], scales)
            _record_validation(state, model, score, "lbfgs", call)
            print(json.dumps(dict(phase="lbfgs", call=call, validation_score=score,
                                  best_score=state["best_score"])), flush=True)
            state["elapsed_seconds"] += time.perf_counter()-started
            started = time.perf_counter()
            _save_progress(output, state, model, adam, scheduler, lbfgs)
            if call == stop_after_lbfgs_call:
                return dict(status="paused", phase="lbfgs", lbfgs_call=call,
                            best_validation_score=state["best_score"])
        return _finish(output, state, model, adam, scheduler, lbfgs, scales)
    except Exception as error:
        _atomic_json(output/"failure.json", dict(status="failed", model=name, seed=seed,
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
