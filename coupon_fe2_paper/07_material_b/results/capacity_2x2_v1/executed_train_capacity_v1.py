#!/usr/bin/env python3
"""Train one fit of a new cell of the MC-RVE constraint-by-capacity study.

The preparation, labels, active Adam/L-BFGS settings and both stopping rules are
loaded through the unchanged amendment-sweep functions of
protocol/train_feature_count_amendment_v2.py, so every cell follows the exact
protocol of the m=6 constrained fits. Only the model class and its hidden
widths change, as declared in training_rule.json next to this file.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
import traceback
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
MATERIAL_B = HERE.parents[1]
sys.path.insert(0, str(MATERIAL_B))

from protocol import train_material_b as base
from protocol.prepare_design import digest
from protocol.select_features import BASE
from protocol.train_feature_count_amendment_v2 import (RULE as AMENDMENT_RULE, adam_stop_reason,
                                                       lbfgs_stop_reason, load_rule as amendment_rule)
from protocol.training_setup import build_initial_model, training_objective

CAPACITY_RULE = HERE / "training_rule.json"
FEATURES = BASE / "results/feature_count_amendment_v2/feature_tables/m06"
SEEDS = (16, 29, 47)


def load_cell(cell: str):
    capacity = json.loads(CAPACITY_RULE.read_text(encoding="utf-8"))
    if (capacity.get("id") != "material_B_capacity_2x2_v1" or cell not in capacity["new_cells"]
            or capacity["amendment_rule_sha256"] != digest(AMENDMENT_RULE)
            or capacity["features_manifest_sha256"] != digest(FEATURES / "manifest.json")
            or capacity["feature_table_sha256"] != digest(FEATURES / "feature_table.npz")
            or capacity["preparation_recipe_sha256"] != digest(FEATURES / "preparation_recipe.json")
            or capacity["labels_sha256"] != digest(base.LABELS)):
        raise ValueError("Invalid or inconsistent MC-RVE capacity rule")
    active, rule, manifest, table, _, count = amendment_rule(FEATURES, AMENDMENT_RULE)
    if count != 6 or capacity["adam"] != rule["adam"] or capacity["lbfgs"] != rule["lbfgs"]:
        raise ValueError("Capacity rule does not match the m=6 amendment protocol")
    spec = capacity["new_cells"][cell]
    cell_recipe = copy.deepcopy(active)
    cell_recipe["models"]["names"] = [spec["model"]]
    cell_recipe["models"]["widths"][spec["family"]] = list(spec["widths"])
    return active, cell_recipe, rule, capacity, manifest, table, spec


def finish(output, state, model, adam, scheduler, lbfgs, scales, active, rule, cell):
    model.load_state_dict(state["best_model_state"], strict=True)
    checkpoint = dict(configuration=state["configuration"], state_dict=base._copy_state(model),
        strain_scale=scales["strain_scale"], energy_scale=scales["energy_scale"],
        name=state["identity"]["model"], seed=state["identity"]["seed"], cell=cell,
        best_validation_score=state["best_score"], best_origin=state["best_origin"],
        identity=state["identity"])
    base._atomic_torch(output / "model.pt", checkpoint)
    optimizer_state = lbfgs.state_dict()["state"]
    counts = next(iter(optimizer_state.values()), {}) if optimizer_state else {}
    report = dict(status="complete", protocol_id="material_B_capacity_2x2_v1", cell=cell,
        model=checkpoint["name"], seed=checkpoint["seed"], capacity_rule_sha256=digest(CAPACITY_RULE),
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        best_validation_score=state["best_score"], best_origin=state["best_origin"],
        adam_steps=state["adam_step"], adam_stop_reason=adam_stop_reason(state, adam, active, rule),
        adam_final_learning_rate=adam.param_groups[0]["lr"],
        lbfgs_outer_calls=state["lbfgs_call"], lbfgs_stop_reason=lbfgs_stop_reason(state, rule),
        lbfgs_internal_iterations=counts.get("n_iter", 0),
        lbfgs_function_evaluations=counts.get("func_evals", 0),
        elapsed_seconds=state["elapsed_seconds"], validation_history=state["validation_history"],
        test_labels_loaded=False, path_labels_loaded=False,
        model_sha256=digest(output / "model.pt"), identity=state["identity"])
    if report["adam_stop_reason"] is None or report["lbfgs_stop_reason"] is None:
        raise RuntimeError("Finalized before both stopping rules were met")
    base._atomic_json(output / "run_report.json", report)
    state["phase"] = "complete"
    base._save_progress(output, state, model, adam, scheduler, lbfgs)
    return report


def run(cell: str, seed: int, output: Path, *, resume=False, stop_after_adam_step=None):
    active, cell_recipe, rule, capacity, manifest, table, spec = load_cell(cell)
    if seed not in SEEDS:
        raise ValueError("Undeclared seed")
    if any(os.environ.get(key) != "2" for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")):
        raise EnvironmentError("Set OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=2")
    arrays = base.load_training_arrays(base.LABELS, active)
    identity = base._identity(spec["model"], seed, CAPACITY_RULE, FEATURES, base.LABELS, manifest)
    identity.update(protocol_id=capacity["id"], cell=cell, widths=list(spec["widths"]),
                    amendment_rule_sha256=digest(AMENDMENT_RULE))
    identity["sources_sha256"][str(Path(__file__).resolve().relative_to(BASE.parents[1]))] = digest(Path(__file__))
    if resume:
        if not (output / base.CHECKPOINT_NAME).is_file() or (output / "failure.json").exists():
            raise RuntimeError("Run is not safely resumable")
    else:
        output.mkdir(parents=True, exist_ok=False)
    e, s, w = (arrays[key] for key in ("E_fit", "S_fit", "W_fit"))
    scales = manifest["scales"]
    model, metadata = build_initial_model(spec["model"], seed, cell_recipe, manifest, table, e, s)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if trainable != spec["trainable_parameters"]:
        raise RuntimeError(f"Unexpected parameter count {trainable} for {cell}")
    adam, scheduler = base._adam_optimizer(model, active)
    if resume:
        state, lbfgs = base._load_progress(output, identity, model, adam, scheduler, active)
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
            reason = adam_stop_reason(state, adam, active, rule)
            while reason is None:
                step = state["adam_step"] + 1
                adam.zero_grad(set_to_none=True)
                total, _ = training_objective(model, e, s, w, scales, table["reference_tangent"], active)
                total.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), active["adam"]["gradient_clip_norm"],
                    norm_type=active["adam"]["clip_norm_type"], error_if_nonfinite=True)
                if not torch.isfinite(norm):
                    raise RuntimeError("Nonfinite Adam gradient norm")
                adam.step(); base._check_model(model); state["adam_step"] = step
                if step % active["adam"]["validation_interval_steps"] == 0:
                    score = base.validation_score(model, arrays["E_validation"], arrays["S_validation"], scales)
                    base._record_validation(state, model, score, "adam", step); scheduler.step(score)
                    reason = adam_stop_reason(state, adam, active, rule)
                    print(json.dumps(dict(phase="adam", step=step, validation_score=score,
                        best_score=state["best_score"], learning_rate=adam.param_groups[0]["lr"])), flush=True)
                state["elapsed_seconds"] += time.perf_counter() - started; started = time.perf_counter()
                if step % 200 == 0 or reason is not None or step == stop_after_adam_step:
                    base._save_progress(output, state, model, adam, scheduler, lbfgs)
                if step == stop_after_adam_step:
                    return dict(status="paused", phase="adam", adam_step=step)
            model.load_state_dict(state["best_model_state"], strict=True)
            lbfgs = base._lbfgs_optimizer(model, active); state["phase"] = "lbfgs"
            base._save_progress(output, state, model, adam, scheduler, lbfgs)
        reason = lbfgs_stop_reason(state, rule)
        while reason is None:
            call = state["lbfgs_call"] + 1
            def closure():
                lbfgs.zero_grad(set_to_none=True)
                total, _ = training_objective(model, e, s, w, scales, table["reference_tangent"], active)
                total.backward()
                for parameter in model.parameters():
                    if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                        raise RuntimeError("Nonfinite L-BFGS gradient")
                return total
            lbfgs.step(closure); base._check_model(model); state["lbfgs_call"] = call
            score = base.validation_score(model, arrays["E_validation"], arrays["S_validation"], scales)
            base._record_validation(state, model, score, "lbfgs", call); reason = lbfgs_stop_reason(state, rule)
            print(json.dumps(dict(phase="lbfgs", call=call, validation_score=score,
                                  best_score=state["best_score"])), flush=True)
            state["elapsed_seconds"] += time.perf_counter() - started; started = time.perf_counter()
            base._save_progress(output, state, model, adam, scheduler, lbfgs)
        return finish(output, state, model, adam, scheduler, lbfgs, scales, active, rule, cell)
    except Exception as error:
        base._atomic_json(output / "failure.json", dict(status="failed", cell=cell, seed=seed,
            last_complete_adam_step=state["adam_step"], last_complete_lbfgs_call=state["lbfgs_call"],
            error=repr(error), traceback=traceback.format_exc(), identity=identity))
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", required=True, choices=("icnn_large", "free_small", "free_large", "ickan_large"))
    parser.add_argument("--seed", type=int, required=True, choices=SEEDS)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-adam-step", type=int)
    args = parser.parse_args()
    result = run(args.cell, args.seed, args.output.resolve(), resume=args.resume,
                 stop_after_adam_step=args.stop_after_adam_step)
    print(json.dumps({k: v for k, v in result.items() if k not in ("identity", "validation_history")},
                     allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
