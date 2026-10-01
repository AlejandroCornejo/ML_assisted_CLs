#!/usr/bin/env python3
"""Open one gate of the MC-RVE capacity study (test states and held-out paths).

Every seed of every cell in the gate is evaluated with the metric, prediction and
reserved-label routines of protocol/evaluate_final_models.py, after its saved
validation score is reproduced. Nothing is fitted or selected here.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
MATERIAL_B = HERE.parents[1]
sys.path.insert(0, str(MATERIAL_B))
sys.path.insert(0, str(MATERIAL_B.parent / "06_pann"))
sys.path.insert(0, str(MATERIAL_B.parents[1] / "RVE_NeoHookean_Homogenization" / "pann" / "anisotropic"))

from protocol import train_material_b as base
from protocol.evaluate_final_models import metrics, predict, read_reserved
from protocol.prepare_design import digest
from protocol.training_setup import FlexibleEnergy
from anisotropic_pann_model import AnisotropicFreeEnergy
import executed_train_capacity_v1 as trainer

TRAINING = HERE / "training"


def load(folder: Path):
    report = json.loads((folder / "run_report.json").read_text(encoding="utf-8"))
    checkpoint = torch.load(folder / "model.pt", map_location="cpu", weights_only=False)
    if (report["status"] != "complete" or report["model_sha256"] != digest(folder / "model.pt")
            or report["capacity_rule_sha256"] != digest(trainer.CAPACITY_RULE)
            or report["test_labels_loaded"] is not False or report["path_labels_loaded"] is not False):
        raise ValueError(f"Invalid terminal report: {folder.name}")
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        constructor = AnisotropicFreeEnergy if checkpoint["name"] == "Free" else FlexibleEnergy
        model = constructor(**checkpoint["configuration"]).double()
    finally:
        torch.set_default_dtype(previous)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return report, model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", required=True, choices=("A", "B"))
    args = parser.parse_args()
    output = HERE / f"evaluation_gate{args.gate}.json"
    if output.exists():
        raise FileExistsError("Gate already opened")
    capacity = json.loads(trainer.CAPACITY_RULE.read_text(encoding="utf-8"))
    cells = [cell for cell, spec in capacity["new_cells"].items() if spec["gate"] == args.gate]
    folders = [TRAINING / f"{cell}_seed{seed}" for cell in cells for seed in trainer.SEEDS]
    active, _, _, _, manifest, _, _ = trainer.load_cell(cells[0])
    arrays = base.load_training_arrays(base.LABELS, active)
    scales = manifest["scales"]
    torch.set_num_threads(2)
    models, checks = {}, []
    for folder in folders:
        report, model = load(folder)
        score = base.validation_score(model, arrays["E_validation"], arrays["S_validation"], scales)
        if not np.isclose(score, report["best_validation_score"], rtol=1e-8, atol=1e-14):
            raise ValueError(f"Validation score not reproducible: {folder.name}")
        models[folder.name] = (report, model)
        checks.append(dict(slug=folder.name, reproduced_validation_score=float(score)))
    gate = dict(status="opened_once", gate=args.gate, opened_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                capacity_rule_sha256=digest(trainer.CAPACITY_RULE), evaluator_sha256=digest(Path(__file__)),
                labels_sha256=digest(base.LABELS), validation_preflight=checks)
    base._atomic_json(HERE / f"test_path_gate{args.gate}.json", gate)
    reserved = read_reserved()
    test_target = tuple(reserved[name] for name in ("W_test", "S_test", "D_test"))
    path_target = tuple(reserved[name] for name in ("W_paths", "S_paths", "D_paths"))
    rows = []
    for slug, (report, model) in models.items():
        test = predict(model, reserved["E_test"], scales)
        paths = predict(model, reserved["E_paths"], scales)
        per_path = {str(name): metrics(tuple(v[40*i:40*(i+1)] for v in paths),
                                       tuple(v[40*i:40*(i+1)] for v in path_target), scales)
                    for i, name in enumerate(reserved["path_names"])}
        rows.append(dict(slug=slug, cell=report["cell"], seed=report["seed"],
                         trainable_parameters=report["trainable_parameters"],
                         best_validation_score=report["best_validation_score"],
                         adam_steps=report["adam_steps"], adam_stop_reason=report["adam_stop_reason"],
                         test=metrics(test, test_target, scales),
                         paths_aggregate=metrics(paths, path_target, scales), paths=per_path))
        print(json.dumps(dict(slug=slug, test_stress_percent=rows[-1]["test"]["aggregate_percent"]["stress"])), flush=True)
    summary = {}
    for cell in cells:
        group = [row for row in rows if row["cell"] == cell]
        summary[cell] = {split: {q: dict(median=float(np.median([r[key]["aggregate_percent"][q] for r in group])),
                                         minimum=float(min(r[key]["aggregate_percent"][q] for r in group)),
                                         maximum=float(max(r[key]["aggregate_percent"][q] for r in group)))
                                 for q in ("stress", "energy", "tangent")}
                         for split, key in (("test", "test"), ("paths", "paths_aggregate"))}
    base._atomic_json(output, dict(gate=gate, summary=summary, rows=rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
