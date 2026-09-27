#!/usr/bin/env python3
"""Train one model of a new cell of the SC-RVE constraint-by-capacity study.

Labels, split, initial features, scaling, objective, optimizer settings and
stopping rules are those of the frozen SC-RVE m=6 campaign; only the model
class and its hidden widths change, as declared in the frozen 2x2 rule. The
runner reads fit/reference arrays and validation stress only.
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

ROOT = Path(__file__).resolve().parents[1]
MATERIAL_B = ROOT / "07_material_b"
import sys
sys.path.insert(0, str(MATERIAL_B))
sys.path.insert(0, str(ROOT / "06_pann"))

from protocol import train_material_b as base
from protocol.prepare_design import digest
from protocol.training_setup import build_initial_model, load_preparation, training_objective
import train_sc_m06_v1 as m06

SEEDS = (16, 29, 47)
DEFAULT_RULE = ROOT / "06_pann/results/sc_capacity_2x2_v1/training_rule.json"


def load_rule(rule_path: Path, cell: str):
    rule = json.loads(rule_path.read_text(encoding="utf-8"))
    preparation = ROOT / rule["preparation"]
    recipe_path = preparation / "preparation_recipe.json"
    labels = preparation / "training_labels.npz"
    parent = ROOT / rule["parent_training_rule"]
    recipe, manifest, table = load_preparation(preparation, recipe_path)
    parent_rule = json.loads(parent.read_text(encoding="utf-8"))
    if (rule.get("id") != "sc_rve_capacity_2x2_v1"
            or cell not in rule["design"]["new_cells"]
            or tuple(rule.get("seeds", ())) != SEEDS
            or rule["preparation_recipe_sha256"] != digest(recipe_path)
            or rule["preparation_manifest_sha256"] != digest(preparation / "manifest.json")
            or rule["feature_table_sha256"] != digest(preparation / "feature_table.npz")
            or rule["labels_sha256"] != digest(labels)
            or rule["parent_training_rule_sha256"] != digest(parent)
            or rule["adam"] != parent_rule["adam"] or rule["lbfgs"] != parent_rule["lbfgs"]
            or rule["objective"] != recipe["objective"]
            or rule["learning_rates"] != recipe["adam"]["learning_rates"]):
        raise ValueError("Invalid or inconsistent SC-RVE capacity rule")
    spec = rule["design"]["new_cells"][cell]
    recipe_cell = copy.deepcopy(recipe)
    recipe_cell["models"]["names"] = [spec["model"]]
    recipe_cell["models"]["widths"][spec["family"]] = list(spec["widths"])
    return recipe, recipe_cell, manifest, table, rule, spec, preparation, recipe_path, labels


def finish(output, state, model, adam, scheduler, lbfgs, scales, recipe, rule, rule_path, cell):
    model.load_state_dict(state["best_model_state"], strict=True)
    checkpoint = dict(configuration=state["configuration"], state_dict=base._copy_state(model),
        strain_scale=scales["strain_scale"], energy_scale=scales["energy_scale"],
        name=state["identity"]["model"], seed=state["identity"]["seed"], cell=cell,
        best_validation_score=state["best_score"], best_origin=state["best_origin"],
        identity=state["identity"])
    base._atomic_torch(output / "model.pt", checkpoint)
    optimizer_state = lbfgs.state_dict()["state"]
    counts = next(iter(optimizer_state.values()), {}) if optimizer_state else {}
    report = dict(status="complete", protocol_id=rule["id"], cell=cell, model=checkpoint["name"],
        seed=checkpoint["seed"], training_rule_sha256=digest(rule_path),
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        best_validation_score=state["best_score"], best_origin=state["best_origin"],
        adam_steps=state["adam_step"],
        adam_stop_reason=m06.adam_stop_reason(state, adam, recipe, rule),
        adam_final_learning_rate=adam.param_groups[0]["lr"],
        lbfgs_outer_calls=state["lbfgs_call"], lbfgs_stop_reason=m06.lbfgs_stop_reason(state, rule),
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


def run(cell: str, seed: int, output: Path, rule_path: Path, *, resume=False,
        stop_after_adam_step=None, stop_after_lbfgs_call=None):
    (recipe, recipe_cell, manifest, table, rule, spec, preparation, recipe_path,
     labels) = load_rule(rule_path, cell)
    if seed not in SEEDS:
        raise ValueError("Undeclared seed")
    if any(os.environ.get(key) != "2" for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")):
        raise EnvironmentError("Set OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=2")
    arrays = m06.load_arrays(labels, recipe)
    name = spec["model"]
    identity = base._identity(name, seed, recipe_path, preparation, labels, manifest)
    identity.update(protocol_id=rule["id"], cell=cell, widths=list(spec["widths"]),
                    training_rule_sha256=digest(rule_path))
    for source in (Path(__file__), Path(m06.__file__)):
        identity["sources_sha256"][str(source.resolve().relative_to(ROOT))] = digest(source)
    if resume:
        if not (output / base.CHECKPOINT_NAME).is_file() or (output / "failure.json").exists():
            raise RuntimeError("Run is not safely resumable")
    else:
        output.mkdir(parents=True, exist_ok=False)

    e, s, w = (arrays[key] for key in ("E_fit", "S_fit", "W_fit"))
    scales = manifest["scales"]
    model, metadata = build_initial_model(name, seed, recipe_cell, manifest, table, e, s)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if trainable != spec["trainable_parameters"]:
        raise RuntimeError(f"Unexpected parameter count {trainable} for {cell}")
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
            reason = m06.adam_stop_reason(state, adam, recipe, rule)
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
                    reason = m06.adam_stop_reason(state, adam, recipe, rule)
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

        reason = m06.lbfgs_stop_reason(state, rule)
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
            reason = m06.lbfgs_stop_reason(state, rule)
            print(json.dumps(dict(phase="lbfgs", call=call, validation_score=score,
                                  best_score=state["best_score"])), flush=True)
            state["elapsed_seconds"] += time.perf_counter() - started
            started = time.perf_counter()
            base._save_progress(output, state, model, adam, scheduler, lbfgs)
            if call == stop_after_lbfgs_call:
                return dict(status="paused", phase="lbfgs", lbfgs_call=call)
        return finish(output, state, model, adam, scheduler, lbfgs, scales,
                      recipe, rule, rule_path, cell)
    except Exception as error:
        base._atomic_json(output / "failure.json", dict(status="failed", cell=cell, seed=seed,
            last_complete_adam_step=state["adam_step"],
            last_complete_lbfgs_call=state["lbfgs_call"], error=repr(error),
            traceback=traceback.format_exc(), identity=identity))
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", required=True, choices=("icnn_large", "free_small", "ickan_large"))
    parser.add_argument("--seed", type=int, required=True, choices=SEEDS)
    parser.add_argument("--rule", type=Path, default=DEFAULT_RULE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-adam-step", type=int)
    parser.add_argument("--stop-after-lbfgs-call", type=int)
    args = parser.parse_args()
    result = run(args.cell, args.seed, args.output.resolve(), args.rule.resolve(), resume=args.resume,
                 stop_after_adam_step=args.stop_after_adam_step,
                 stop_after_lbfgs_call=args.stop_after_lbfgs_call)
    print(json.dumps({k: v for k, v in result.items()
                      if k not in ("identity", "validation_history")}, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
