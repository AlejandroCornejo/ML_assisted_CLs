#!/usr/bin/env python3
"""Diagnostic: show that the square RVE is not assumed isotropic.

This script compares each mixed strain state with the same state transformed
by a 45-degree material rotation.  A 45-degree rotation is not an element of
the square group D4.  Therefore equality of energy and transformed stress is
*not* expected; a non-zero discrepancy is the physical signature that the
effective material has square symmetry rather than full isotropy.

Run after enabling the Kratos Eigen environment:
    source /home/kratos/set_up_kratos_eigen.sh
    python3 run_non_d4_rotation_diagnostic.py
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from run_d4_symmetry_test import (
    BASE_STRAINS,
    THIS_DIR,
    relative_error,
    run_one_case,
    strain_transform,
    stress_transform,
)


ROTATION_45 = np.array(
    [
        [np.sqrt(0.5), -np.sqrt(0.5)],
        [np.sqrt(0.5), np.sqrt(0.5)],
    ],
    dtype=float,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-steps", type=int, default=100)
    parser.add_argument("--reference-amplitude", type=float, default=0.15)
    parser.add_argument("--results-dir", type=Path, default=THIS_DIR / "isotropy_diagnostic")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.chdir(THIS_DIR)
    results_dir = args.results_dir.resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for base_name, base_strain in BASE_STRAINS.items():
        print(f"{base_name}: reference orientation", flush=True)
        reference = run_one_case(
            case_name=f"{base_name}__identity",
            applied_strain=base_strain,
            results_dir=results_dir,
            reference_steps=args.reference_steps,
            reference_amplitude=args.reference_amplitude,
        )

        rotated_strain = strain_transform(base_strain, ROTATION_45)
        print(f"{base_name}: 45-degree material rotation, E={rotated_strain.tolist()}", flush=True)
        rotated = run_one_case(
            case_name=f"{base_name}__rotate_45",
            applied_strain=rotated_strain,
            results_dir=results_dir,
            reference_steps=args.reference_steps,
            reference_amplitude=args.reference_amplitude,
        )

        expected_if_isotropic = stress_transform(reference["stress_final"], ROTATION_45)
        stress_difference = relative_error(rotated["stress_final"], expected_if_isotropic)
        energy_difference = abs(rotated["energy_fem"] - reference["energy_fem"]) / max(
            abs(reference["energy_fem"]), 1.0e-14
        )
        rows.append(
            {
                "base_case": base_name,
                "strain_reference": reference["input_strain"].tolist(),
                "strain_rotate_45": rotated["input_strain"].tolist(),
                "stress_relative_difference_from_isotropic_prediction": stress_difference,
                "energy_relative_difference_from_isotropic_prediction": energy_difference,
                "energy_reference": reference["energy_fem"],
                "energy_rotate_45": rotated["energy_fem"],
            }
        )

    summary = {
        "test": "non_D4_45_degree_rotation_diagnostic",
        "meaning": "A non-zero discrepancy is expected for D4 symmetry; it would be zero for an isotropic effective energy.",
        "rows": rows,
        "max_relative_differences": {
            "stress": max(row["stress_relative_difference_from_isotropic_prediction"] for row in rows),
            "energy": max(row["energy_relative_difference_from_isotropic_prediction"] for row in rows),
        },
    }
    (results_dir / "isotropy_diagnostic_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Non-\\(D_4\\) 45-degree rotation diagnostic",
        "",
        r"A \(45^\circ\) material rotation is not a symmetry of a square RVE. The table reports the discrepancy from the response that full isotropy would predict.",
        "",
        "| Base state | Stress difference from isotropy | Energy difference from isotropy |",
        "|---|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['base_case']} | {row['stress_relative_difference_from_isotropic_prediction']:.6e} | "
            f"{row['energy_relative_difference_from_isotropic_prediction']:.6e} |"
        )
    (results_dir / "isotropy_diagnostic_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nReport: {results_dir / 'isotropy_diagnostic_summary.md'}")


if __name__ == "__main__":
    main()
