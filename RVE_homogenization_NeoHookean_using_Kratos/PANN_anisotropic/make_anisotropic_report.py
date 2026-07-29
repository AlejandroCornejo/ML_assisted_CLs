#!/usr/bin/env python3
"""Write result macros and compile the anisotropic-PANN technical memorandum."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
METRICS = RESULTS / "anisotropic_stage10_metrics.json"
MACROS = RESULTS / "anisotropic_numbers.tex"
METRIC_MAP = HERE / "data" / "local_isotropic_metric.json"
FOM_TANGENT = HERE.parent / "fom_tangent_stability_test" / "results" / "h_1.000e-02" / "fom_energy_hessian_summary.json"


def percentage(value: float) -> str:
    return f"{100.0 * value:.3f}\\%"


def main() -> None:
    if not METRICS.exists():
        raise FileNotFoundError("Run evaluate_anisotropic_pann.py first.")
    results = json.loads(METRICS.read_text(encoding="utf-8"))
    free = results["free_anisotropic_c_pann"]
    metric = results["metric_preconditioned_anisotropic_pann"]
    poly = results["polyconvex_anisotropic_pann"]
    hprom = results["compatible_direct_hprom_ann_reference"]["hprom_ann_vs_direct_fom"]
    poly_audit = poly["guarantee_audit"]
    local_metric = json.loads(METRIC_MAP.read_text(encoding="utf-8"))
    tangent_records = json.loads(FOM_TANGENT.read_text(encoding="utf-8"))["records"]
    stage10_tangent = next(record for record in tangent_records if record["name"] == "stage10_mixed")
    names = {
        "FreeEnergyError": percentage(free["energy_relative_l2"]),
        "FreeStressError": percentage(free["stress_relative_l2"]),
        "FreeSxxError": percentage(free["stress_component_relative_l2"][0]),
        "FreeSyyError": percentage(free["stress_component_relative_l2"][1]),
        "FreeSxyError": percentage(free["stress_component_relative_l2"][2]),
        "MetricEnergyError": percentage(metric["energy_relative_l2"]),
        "MetricStressError": percentage(metric["stress_relative_l2"]),
        "MetricSxxError": percentage(metric["stress_component_relative_l2"][0]),
        "MetricSyyError": percentage(metric["stress_component_relative_l2"][1]),
        "MetricSxyError": percentage(metric["stress_component_relative_l2"][2]),
        "PolyEnergyError": percentage(poly["energy_relative_l2"]),
        "PolyStressError": percentage(poly["stress_relative_l2"]),
        "PolySxxError": percentage(poly["stress_component_relative_l2"][0]),
        "PolySyyError": percentage(poly["stress_component_relative_l2"][1]),
        "PolySxyError": percentage(poly["stress_component_relative_l2"][2]),
        "HPROMEnergyError": percentage(hprom["energy_relative_l2"]),
        "HPROMStressError": percentage(hprom["stress_relative_l2"]),
        "HPROMSxxError": percentage(hprom["stress_component_relative_l2"][0]),
        "HPROMSyyError": percentage(hprom["stress_component_relative_l2"][1]),
        "HPROMSxyError": percentage(hprom["stress_component_relative_l2"][2]),
        "RankOneCurvature": f"{poly_audit['rank_one_curvature_audit']['minimum_second_derivative'] / 1.0e9:.3e}\\,\\mathrm{{GPa}}",
        "BarrierCoefficient": f"{poly_audit['analytic_polyconvex_certificate']['barrier_coefficient']:.4e}",
        "QuadraticCoefficient": f"{poly_audit['analytic_polyconvex_certificate']['quadratic_volumetric_coefficient']:.4e}",
        "SampledMinimumEnergy": f"{poly_audit['broad_energy_sampling_audit']['minimum_energy'] / 1.0e9:.3e}\\,\\mathrm{{GPa}}",
        "MetricTangentDistance": percentage(local_metric["checks"]["relative_distance_of_D0_to_D_hat"]),
        "MetricIdentityDistance": percentage(local_metric["checks"]["relative_distance_of_T_to_identity"]),
        "FomStageTenTangentMinimum": f"{stage10_tangent['min_eigenvalue'] / 1.0e9:.4f}\\,\\mathrm{{GPa}}",
    }
    MACROS.write_text("\n".join(f"\\newcommand{{\\{key}}}{{{value}}}" for key, value in names.items()) + "\n", encoding="utf-8")
    subprocess.run(["pdflatex", "-interaction=nonstopmode", "PANN_anisotropic.tex"], cwd=HERE, check=True)
    subprocess.run(["pdflatex", "-interaction=nonstopmode", "PANN_anisotropic.tex"], cwd=HERE, check=True)


if __name__ == "__main__":
    main()
