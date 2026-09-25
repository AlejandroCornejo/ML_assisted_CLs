#!/usr/bin/env python3
"""Freeze the SC-RVE m=6 learned-feature training preparation.

Only the historical SC-RVE fitting split and undeformed reference quantities
enter feature selection. Validation is saved for checkpoint selection; test and
probe arrays are deliberately absent from the training-label archive.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import platform
from pathlib import Path

import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[1]
MATERIAL_B = ROOT / "07_material_b"
import sys
sys.path.insert(0, str(MATERIAL_B))

from protocol.prepare_design import deterministic_npz, digest
from protocol.select_features import load_fit_reference, prepare

SOURCE_DATA = ROOT / "03_data/data.npz"
SOURCE_TANGENT = ROOT / "00_rve/C0_periodic.npz"
PARENT_RECIPE = MATERIAL_B / "protocol/training_recipe_v1.json"
DEFAULT_OUT = ROOT / "06_pann/results/sc_m06_learned_v1/preparation"
SEEDS = (16, 29, 47)
MODELS = ("ICNN-learned", "ICKAN-learned")


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def relative_to_material_b(path: Path) -> str:
    return os.path.relpath(path.resolve(), MATERIAL_B.resolve())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite frozen preparation: {out}")
    out.mkdir(parents=True)

    with np.load(SOURCE_DATA, allow_pickle=False) as store:
        e = np.asarray(store["E_train"], dtype=np.float64)
        s = np.asarray(store["S_train"], dtype=np.float64)
        w = np.asarray(store["W_train"], dtype=np.float64)
        s0 = np.asarray(store["S_zero"], dtype=np.float64)
        w0 = np.asarray(store["W_zero"], dtype=np.float64)
    if e.shape != (4950, 3) or s.shape != (4950, 3) or w.shape != (4950,):
        raise ValueError("Unexpected SC-RVE training-array shapes")
    order = np.random.default_rng(5).permutation(len(e))
    validation_index, fit_index = order[:742], order[742:]
    with np.load(SOURCE_TANGENT, allow_pickle=False) as store:
        d0 = np.asarray(store["C0_periodic"], dtype=np.float64)
    if d0.shape != (3, 3) or s0.shape != (1, 3) or w0.shape != (1,):
        raise ValueError("Unexpected SC-RVE reference-array shapes")

    arrays = {
        "E_fit": e[fit_index], "S_fit": s[fit_index], "W_fit": w[fit_index],
        "E_validation": e[validation_index], "S_validation": s[validation_index],
        "W_validation": w[validation_index],
        "E_reference": np.zeros((1, 3), dtype=np.float64),
        "S_reference": s0, "W_reference": w0, "D_reference": d0[None, :, :],
        "fit_index": fit_index.astype(np.int64),
        "validation_index": validation_index.astype(np.int64),
    }
    labels = out / "training_labels.npz"
    deterministic_npz(labels, arrays)

    recipe = copy.deepcopy(json.loads(PARENT_RECIPE.read_text(encoding="utf-8")))
    recipe.update(
        id="sc_rve_m06_learned_preparation_v1",
        labels_sha256=digest(labels),
        source_data_sha256=digest(SOURCE_DATA),
        source_tangent_sha256=digest(SOURCE_TANGENT),
        timing=("Frozen after the MC-RVE validation sensitivity selected m=6 and before "
                "any SC-RVE m=6 optimization. Test and probe arrays are excluded."),
        scope=("SC-RVE m=6 learned-feature preparation only. Feature selection uses fit and "
               "reference arrays; validation, test, and probe labels do not select features."),
    )
    recipe["feature_selection"]["count"] = 6
    recipe["models"]["names"] = list(MODELS)
    recipe["models"]["seeds"] = list(SEEDS)
    recipe["execution"]["batch"] = "All 4208 fitting states per objective evaluation"
    recipe_path = out / "preparation_recipe.json"
    atomic_json(recipe_path, recipe)

    selection_arrays = load_fit_reference(np.load(labels, allow_pickle=False),
                                           recipe["selection_allowed_arrays"])
    payload, scales, diagnostics = prepare(selection_arrays, recipe)
    table = out / "feature_table.npz"
    deterministic_npz(table, payload)

    source_paths = [Path(__file__), MATERIAL_B / "protocol/select_features.py",
                    MATERIAL_B / "protocol/prepare_design.py"]
    manifest = {
        "status": "complete", "neural_training_started": False,
        "protocol_id": "sc_rve_m06_learned_v1", "feature_count": 6,
        "recipe_sha256": digest(recipe_path), "labels_sha256": digest(labels),
        "feature_table_sha256": digest(table), "source_data_sha256": digest(SOURCE_DATA),
        "source_tangent_sha256": digest(SOURCE_TANGENT), "scales": scales,
        "diagnostics": diagnostics, "fit_count": 4208, "validation_count": 742,
        "split_seed": 5, "models": list(MODELS), "seeds": list(SEEDS),
        "accessed_label_arrays": recipe["selection_allowed_arrays"],
        "validation_used_for_feature_selection": False,
        "test_or_probe_arrays_present": False,
        "sources_sha256": {relative_to_material_b(source): digest(source) for source in source_paths},
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "scipy": scipy.__version__},
        "interpretation": ("The feature count is transferred from the MC-RVE sensitivity result; "
                           "the six initial feature specifications are selected anew from SC-RVE fit/reference data."),
    }
    atomic_json(out / "manifest.json", manifest)

    rule = {
        "id": "sc_rve_m06_learned_training_v1",
        "preparation_recipe_sha256": digest(recipe_path),
        "feature_table_sha256": digest(table), "labels_sha256": digest(labels),
        "source_data_sha256": digest(SOURCE_DATA), "source_tangent_sha256": digest(SOURCE_TANGENT),
        "feature_count": 6, "models": list(MODELS), "seeds": list(SEEDS),
        "validation_metric": ("Mean squared normalized validation stress error divided by the frozen "
                              "fit-stress denominator; strict minimum over initialization, Adam, and L-BFGS."),
        "adam": {"minimum_steps": 2600, "maximum_steps": 200000,
                 "material_improvement_relative": 0.001,
                 "plateau_patience_validation_calls": 100,
                 "require_scheduler_minimum_lr": True},
        "lbfgs": {"minimum_outer_calls": 40, "maximum_outer_calls": 300,
                  "material_improvement_relative": 0.001,
                  "plateau_patience_outer_calls": 30},
        "selection": ("After all six fits are terminal, select one checkpoint per core by minimum validation "
                      "score. No test or probe quantity may enter training or selection."),
        "evaluation_gate": ("Test and probe arrays remain closed until all six reports and checkpoint hashes "
                            "are verified and validation_selection.json has been written."),
    }
    atomic_json(out / "training_rule.json", rule)
    atomic_json(out / "preparation_receipt.json", {
        "status": "complete", "feature_count": 6, "models": list(MODELS), "seeds": list(SEEDS),
        "labels_sha256": digest(labels), "recipe_sha256": digest(recipe_path),
        "feature_table_sha256": digest(table), "rule_sha256": digest(out / "training_rule.json"),
        "validation_used_for_feature_selection": False, "test_probe_used": False,
    })
    print(json.dumps({"status": "complete", "out": str(out), "feature_count": 6,
                      "fit_count": 4208, "validation_count": 742,
                      "rule_sha256": digest(out / "training_rule.json")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
