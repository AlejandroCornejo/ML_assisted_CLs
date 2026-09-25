#!/usr/bin/env python3
"""Mechanical audit of the validation-selected SC-RVE m=6 models.

The test/probe gate has already been opened by ``independent_audit.json``.
This script reuses the predeclared cycle and rank-one scans from
``mechanics_witnesses.py`` and changes only the two constrained checkpoints.
Historical Regression and Free checkpoints remain fixed controls.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import mechanics_witnesses as witness


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CAMPAIGN = HERE / "results" / "sc_m06_learned_v1"
TRAINING = CAMPAIGN / "training"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycle-order", type=int, default=64)
    parser.add_argument("--box-states", type=int, default=12000)
    parser.add_argument("--broad-states", type=int, default=24000)
    parser.add_argument("--radial-directions", type=int, default=1800)
    parser.add_argument("--radial-max-ring", type=float, default=4.0)
    parser.add_argument("--radial-ring-step", type=float, default=0.05)
    parser.add_argument("--rank-one-directions", type=int, default=180)
    parser.add_argument("--output", type=Path,
                        default=CAMPAIGN / "mechanics_audit.json")
    args = parser.parse_args()

    selection_path = TRAINING / "validation_selection.json"
    selection = json.loads(selection_path.read_text())
    if selection.get("status") != "frozen_before_test_probe":
        raise RuntimeError("Validation selection is not frozen")
    selected = {row["core"]: Path(row["checkpoint"]) for row in selection["selected"]}
    paths = {
        "Regression": HERE / "pann_regression.pt",
        "Free": HERE / "pann_free.pt",
        "ICNN": selected["ICNN"],
        "ICKAN": selected["ICKAN"],
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing frozen checkpoints: " + ", ".join(missing))

    rings = np.arange(1.0, args.radial_max_ring + 0.5 * args.radial_ring_step,
                      args.radial_ring_step)
    laws = witness.load_laws(paths)
    report = {
        "scope": ("validation-selected SC-RVE m=6 constrained models; fixed historical "
                  "Regression and Free controls; no retraining or model selection"),
        "selection_manifest": str(selection_path),
        "checkpoints": {
            name: {"path": str(path), "sha256": witness.sha256(path)}
            for name, path in paths.items()
        },
        "cycle": witness.cycle_audit(laws, args.cycle_order),
        "rank_one": witness.rank_one_audit(
            laws, args.box_states, args.broad_states,
            args.rank_one_directions, args.radial_directions, rings,
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    clean = witness.as_json(report)
    args.output.write_text(json.dumps(clean, indent=2, allow_nan=False) + "\n")
    summary = {
        "output": str(args.output),
        "ICNN_cycle_work_J_per_m3": clean["cycle"]["models"]["ICNN"]["stress_work_J_per_m3"],
        "ICKAN_cycle_work_J_per_m3": clean["cycle"]["models"]["ICKAN"]["stress_work_J_per_m3"],
        "ICNN_min_broad_rank_one_curvature_Pa": clean["rank_one"]["clouds"][
            "broad_unvalidated_principal_stretch_scan"]["models"]["ICNN"]["curvature_Pa"],
        "ICKAN_min_broad_rank_one_curvature_Pa": clean["rank_one"]["clouds"][
            "broad_unvalidated_principal_stretch_scan"]["models"]["ICKAN"]["curvature_Pa"],
    }
    print(json.dumps(summary, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
