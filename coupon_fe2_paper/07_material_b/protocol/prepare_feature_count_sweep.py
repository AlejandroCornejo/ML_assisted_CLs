"""Freeze count-specific multicavity feature tables for the constrained sweep.

This post-hoc sensitivity protocol creates every table from the frozen fit and
reference labels only.  It never reads validation, test, or path labels.
The historical 32-feature table is copied byte-for-byte after functional
reproduction checks, so the new campaign does not silently redefine m=32.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import platform
import shutil
from pathlib import Path

import numpy as np
import scipy

from protocol.prepare_design import deterministic_npz, digest
from protocol.select_features import BASE, RECIPE, load_fit_reference, prepare

COUNTS = (8, 16, 24, 32, 40)
FROZEN = BASE / "results/feature_selection_v1"
LABELS = BASE / "results/data_labels_v1.npz"


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    os.replace(temporary, path)


def _recipe(feature_count: int) -> dict:
    parent = json.loads(RECIPE.read_text(encoding="utf-8"))
    recipe = copy.deepcopy(parent)
    recipe["id"] = f"material_B_feature_count_sweep_preparation_v1_m{feature_count:02d}"
    recipe["feature_selection"]["count"] = feature_count
    recipe["models"]["names"] = ["ICNN-fixed", "ICNN-learned",
                                  "ICKAN-fixed", "ICKAN-learned"]
    recipe["timing"] = ("Post-hoc multicavity feature-count sensitivity preparation. "
                        "The parent data split, architectures, initialization, and objective "
                        "are inherited unchanged. Feature selection is recomputed from only "
                        "the frozen fit and reference arrays for this declared count.")
    recipe["scope"] = ("Preparation only for the constrained feature-count sweep: no Adam/LBFGS "
                       "optimization and no validation, test, or path labels are read.")
    return recipe


def _verify_parent() -> None:
    parent = json.loads((BASE / "protocol/data_protocol_v1.json").read_text(encoding="utf-8"))
    original = json.loads(RECIPE.read_text(encoding="utf-8"))
    if digest(LABELS) != original["labels_sha256"]:
        raise ValueError("Frozen label hash changed")
    if digest(BASE / "protocol/data_protocol_v1.json") != original["parent_data_protocol_sha256"]:
        raise ValueError("Frozen data protocol hash changed")
    report_path = LABELS.with_suffix(".json")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "complete" or not report.get("passed"):
        raise ValueError("FOM assembly gate does not pass")
    if len(parent["models"]["initialization_seeds"]) != 3:
        raise ValueError("Unexpected parent seed set")


def _functional_m32_check(payload: dict) -> dict:
    """Check runtime-relevant data exactly; diagnostics may differ at roundoff ties."""
    frozen_path = FROZEN / "feature_table.npz"
    with np.load(frozen_path, allow_pickle=False) as frozen:
        checks = {name: bool(np.array_equal(payload[name], frozen[name])) for name in (
            "candidate_bank", "selected_candidate_indices", "specs",
            "support_candidate_indices", "feature_center", "feature_scale",
            "initialization_coefficients", "reference_tangent")}
    if not all(checks.values()):
        raise RuntimeError("m=32 selection does not reproduce the frozen runtime table: "
                           + json.dumps(checks, sort_keys=True))
    return dict(runtime_arrays_exact=checks,
                frozen_table_sha256=digest(frozen_path),
                byte_identity_obtained_by="copying the locked historical feature table")


def prepare_one(feature_count: int, destination: Path, arrays: dict[str, np.ndarray]) -> dict:
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    destination.mkdir(parents=True)
    recipe = _recipe(feature_count)
    recipe_path = destination / "preparation_recipe.json"
    _atomic_json(recipe_path, recipe)
    payload, scales, diagnostics = prepare(arrays, recipe)
    table_path = destination / "feature_table.npz"
    m32 = None
    if feature_count == 32:
        m32 = _functional_m32_check(payload)
        shutil.copyfile(FROZEN / "feature_table.npz", table_path)
    else:
        deterministic_npz(table_path, payload)
    source_paths = [Path(__file__), Path(__file__).with_name("select_features.py"),
                    Path(__file__).with_name("prepare_design.py")]
    manifest = dict(
        status="complete", neural_training_started=False, feature_count=feature_count,
        protocol_id="material_B_feature_count_sweep_v1",
        parent_preparation_recipe=RECIPE.name, parent_preparation_recipe_sha256=digest(RECIPE),
        accessed_label_arrays=recipe["selection_allowed_arrays"], validation_used=False,
        test_used=False, paths_used=False, recipe_sha256=digest(recipe_path),
        data_protocol_sha256=recipe["parent_data_protocol_sha256"],
        labels_sha256=recipe["labels_sha256"],
        assembly_report_sha256=digest(LABELS.with_suffix(".json")),
        feature_table_sha256=digest(table_path), scales=scales, diagnostics=diagnostics,
        sources_sha256={str(source.relative_to(BASE)): digest(source) for source in source_paths},
        environment=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                         thread_environment={key: os.environ.get(key) for key in
                                             ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}),
        feature_set_rule=("Each count is selected independently by the inherited NNLS-support and "
                          "pivoted-QR rule on the frozen fit/reference information; tables are "
                          "not forced to be nested."),
        limitations=recipe["feature_selection"]["limitations"],
        m32_reproduction=m32)
    _atomic_json(destination / "manifest.json", manifest)
    return dict(feature_count=feature_count, folder=str(destination),
                recipe_sha256=manifest["recipe_sha256"],
                feature_table_sha256=manifest["feature_table_sha256"],
                m32_reproduction=m32)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True,
                        help="New feature-table root; never overwrites.")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Use a new feature-table root")
    _verify_parent()
    args.out.mkdir(parents=True)
    with np.load(LABELS, allow_pickle=False) as store:
        arrays = load_fit_reference(store, _recipe(32)["selection_allowed_arrays"])
    rows = [prepare_one(count, args.out / f"m{count:02d}", arrays) for count in COUNTS]
    _atomic_json(args.out / "sweep_preparation_manifest.json", dict(
        status="complete", protocol_id="material_B_feature_count_sweep_v1",
        counts=list(COUNTS), source_recipe_sha256=digest(RECIPE), labels_sha256=digest(LABELS),
        validation_used=False, test_used=False, paths_used=False, tables=rows,
        interpretation=("Frozen count-specific feature tables for a post-hoc constrained-model "
                        "sensitivity experiment; no trained accuracy or model-selection claim.")))
    print(json.dumps(dict(status="complete", feature_counts=list(COUNTS), tables=rows),
                     indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
