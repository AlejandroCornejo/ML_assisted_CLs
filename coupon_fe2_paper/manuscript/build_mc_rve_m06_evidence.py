#!/usr/bin/env python3
"""Build compact independent-test evidence for the MC-RVE m=6 comparison."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
TABLES = HERE / "tables"
BASE = PROJECT / "07_material_b"
EVALUATION = BASE / "results/feature_count_analysis_v1/m06_independent_evaluation_v1"
LABELS = BASE / "results/data_labels_v1.npz"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load() -> dict:
    gate = json.loads((EVALUATION / "gate_decision.json").read_text())
    summary = json.loads((EVALUATION / "summary.json").read_text())
    per_run = json.loads((EVALUATION / "per_run.json").read_text())
    if (gate.get("status") != "opened_once" or summary.get("status") != "complete"
            or summary.get("predictions_sha256") != digest(EVALUATION / "predictions.npz")
            or per_run.get("prediction_sha256") != digest(EVALUATION / "predictions.npz")
            or gate.get("labels_sha256") != digest(LABELS)):
        raise ValueError("Independent-evaluation receipt is invalid")
    return summary


def value(cell: dict) -> str:
    return f"{cell['median']:.3g} [{cell['minimum']:.3g}, {cell['maximum']:.3g}]"


def table(summary: dict) -> None:
    rows = []
    for core in ("ICNN", "ICKAN"):
        fixed = summary["by_model"][f"{core}-fixed"]
        learned = summary["by_model"][f"{core}-learned"]
        rows.append((core,
                     value(fixed["test_aggregate_percent"]["stress"]),
                     value(learned["test_aggregate_percent"]["stress"])))
    content = [
        r"\begin{tabular}{lcc}", r"\toprule",
        r"Core & \multicolumn{2}{c}{Independent test}\\",
        r" & Fixed & Learned\\",
        r"\midrule",
    ]
    content.extend(rf"{core} & {fixed}\% & {learned}\%\\"
                   for core, fixed, learned in rows)
    content.extend([r"\bottomrule", r"\end{tabular}", ""])
    TABLES.mkdir(exist_ok=True)
    (TABLES / "mc_rve_m06_stress_errors.tex").write_text("\n".join(content))


def main() -> None:
    table(load())


if __name__ == "__main__":
    main()
