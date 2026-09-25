#!/usr/bin/env python3
"""Augment the frozen common path with fixed/learned m=2 and m=6 predictions.

The FOM path and endpoint fields are copied byte-for-byte at the array level
from common_path_evidence_v1.  Model checkpoints are selected by the median
validation score within each predeclared group; path values never enter the
selection.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
MC = HERE.parent
ROOT = MC.parent
SOURCE = MC / "results/common_path_evidence_v1"
OUTPUT = MC / "results/common_path_comparison_v2"
VALIDATION = (MC / "results/feature_count_analysis_v1/validation_audit_v1"
              / "validation_summary.json")

for path in (ROOT / "00_rve", MC):
    sys.path.insert(0, str(path))

from protocol.evaluate_final_models import predict  # noqa: E402
from protocol.generate_common_path_evidence_v1 import load_model  # noqa: E402


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def selected_rows() -> list[dict]:
    audit = json.loads(VALIDATION.read_text(encoding="utf-8"))
    selected = []
    for core in ("ICNN", "ICKAN"):
        for count in (2, 6):
            for feature_type in ("fixed", "learned"):
                group = sorted(
                    (row for row in audit["rows"]
                     if row["core"] == core
                     and row["feature_type"] == feature_type
                     and row["feature_count"] == count),
                    key=lambda row: row["best_validation_score"],
                )
                if len(group) != 3:
                    raise ValueError(
                        f"Incomplete validation group: {core}, {feature_type}, m={count}")
                selected.append(group[1])
    return selected


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite frozen evidence: {OUTPUT}")
    source_report = json.loads((SOURCE / "report.json").read_text(encoding="utf-8"))
    source_evidence = SOURCE / "evidence.npz"
    if (source_report.get("status") != "complete"
            or source_report.get("selection_used_path_values") is not False
            or source_report.get("evidence_sha256") != digest(source_evidence)):
        raise ValueError("Invalid v1 common-path evidence")

    with np.load(source_evidence, allow_pickle=False) as source:
        arrays = {key: source[key].copy() for key in (
            "path_parameter", "strain", "W_mc_fom", "S_mc_fom",
            "W_sc_fom", "S_sc_fom", "mc_xy", "mc_triangles", "mc_u_nodal",
            "mc_displacement_magnitude", "mc_von_mises_element",
            "mc_minimum_micro_J", "mc_maximum_micro_J", "sc_xy",
            "sc_triangles", "sc_u_nodal", "sc_displacement_magnitude",
            "sc_von_mises_element", "sc_minimum_micro_J", "sc_maximum_micro_J")}

    states = arrays["strain"]
    reference_stress = arrays["S_mc_fom"]
    selected, metrics = [], []
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    for row in selected_rows():
        model, scales, checkpoint = load_model(row)
        energy, stress, _ = predict(model, states, scales)
        key = (f"{row['core'].lower()}_{row['feature_type']}_"
               f"m{int(row['feature_count']):02d}")
        arrays[f"W_{key}"] = energy
        arrays[f"S_{key}"] = stress
        difference = stress - reference_stress
        component_error = [
            float(np.linalg.norm(difference[:, index])
                  / np.linalg.norm(reference_stress[:, index]))
            for index in range(3)
        ]
        metrics.append(dict(
            core=row["core"], feature_type=row["feature_type"],
            feature_count=row["feature_count"], seed=row["seed"],
            relative_stress_error=float(np.linalg.norm(difference)
                                        / np.linalg.norm(reference_stress)),
            component_relative_errors=component_error,
            component_maximum_absolute_errors_pa=[
                float(np.max(np.abs(difference[:, index]))) for index in range(3)]))
        selected.append(dict(
            core=row["core"], feature_type=row["feature_type"],
            feature_count=row["feature_count"], seed=row["seed"],
            slug=row["slug"], best_validation_score=row["best_validation_score"],
            checkpoint_sha256=digest(checkpoint)))

    OUTPUT.mkdir(parents=True)
    np.savez_compressed(OUTPUT / "evidence.npz", **arrays)
    report = dict(
        status="complete",
        purpose=("Interpret the validation-selected feature-count decision along a "
                 "mixed-loading path; no path value enters checkpoint selection."),
        source_v1_evidence_sha256=digest(source_evidence),
        selection_rule=("For each core, feature type, and m in {2,6}, use the run "
                        "with the median validation score over seeds 16, 29, and 47."),
        selection_used_path_values=False,
        selected_checkpoints=selected,
        path_metrics=metrics,
        evidence_sha256=digest(OUTPUT / "evidence.npz"),
        units=source_report["units"])
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
