#!/usr/bin/env python3
"""Write result macros and compile the polyconvex-only anisotropic-PANN memorandum."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
METRICS = RESULTS / "polyconvex_final_claude_metrics.json"
OLD_METRICS = RESULTS / "polyconvex_original_claude_metrics.json"
CONTROL_METRICS = RESULTS / "polyconvex_widthonly_claude_metrics.json"
ICKAN_METRICS = RESULTS / "polyconvex_ickan_final_claude_metrics.json"
FREE_MATCHED_METRICS = RESULTS / "free_compw000_ellipticity_claude_metrics.json"
REGRESSION_METRICS = RESULTS / "regression_claude_metrics.json"
REGRESSION_TRAINING_SUMMARY = RESULTS / "PANN_anisotropic_regression_claude_training_summary.json"
C1_CYCLIC_METRICS = RESULTS / "c1_cyclic_claude_metrics.json"
HPROM_ANN_DIRECT_METRICS = HERE.parent / "data" / "hprom_ann_direct_stage10_metrics.json"
MACROS = RESULTS / "polyconvex_numbers_claude.tex"
FOM_TANGENT = HERE.parent.parent / "studies" / "fom_tangent_stability_test" / "results" / "h_1.000e-02" / "fom_energy_hessian_summary.json"


def percentage(value: float) -> str:
    return f"{100.0 * value:.3f}\\%"


def main() -> None:
    if not METRICS.exists():
        raise FileNotFoundError("Run evaluate_polyconvex_final_claude.py first.")
    if not OLD_METRICS.exists():
        raise FileNotFoundError(
            "Run evaluate_polyconvex_final_claude.py --checkpoint PANN_anisotropic_polyconvex.pt "
            "--output-prefix polyconvex_original_claude first."
        )
    result = json.loads(METRICS.read_text(encoding="utf-8"))
    old_result = json.loads(OLD_METRICS.read_text(encoding="utf-8"))
    control_result = json.loads(CONTROL_METRICS.read_text(encoding="utf-8"))
    ickan_result = json.loads(ICKAN_METRICS.read_text(encoding="utf-8")) if ICKAN_METRICS.exists() else None
    free_matched_result = json.loads(FREE_MATCHED_METRICS.read_text(encoding="utf-8")) if FREE_MATCHED_METRICS.exists() else None
    audit = result["guarantee_audit"]
    tangent_records = json.loads(FOM_TANGENT.read_text(encoding="utf-8"))["records"]
    stage10_tangent = next(record for record in tangent_records if record["name"] == "stage10_mixed")

    names = {
        "FinalEnergyError": percentage(result["energy_relative_l2"]),
        "FinalStressError": percentage(result["stress_relative_l2"]),
        "FinalSxxError": percentage(result["stress_component_relative_l2"][0]),
        "FinalSyyError": percentage(result["stress_component_relative_l2"][1]),
        "FinalSxyError": percentage(result["stress_component_relative_l2"][2]),
        "FinalVonMisesError": percentage(result["von_mises_relative_l2"]),
        "RankOneCurvature": f"{audit['rank_one_curvature_audit']['minimum_second_derivative'] / 1.0e9:.3e}\\,\\mathrm{{GPa}}",
        "BarrierCoefficient": f"{audit['analytic_polyconvex_certificate']['barrier_coefficient']:.4e}",
        "QuadraticCoefficient": f"{audit['analytic_polyconvex_certificate']['quadratic_volumetric_coefficient']:.4e}",
        "SampledMinimumEnergy": f"{audit['broad_energy_sampling_audit']['minimum_energy'] / 1.0e9:.3e}\\,\\mathrm{{GPa}}",
        "FomStageTenTangentMinimum": f"{stage10_tangent['min_eigenvalue'] / 1.0e9:.4f}\\,\\mathrm{{GPa}}",
        "BestEpoch": str(result["best_epoch"]),
        "IcnnWidths": ",".join(str(w) for w in result["icnn_widths"]),
        "ComponentStressWeight": f"{result['loss_weights']['component_stress_weight']:.2f}",
        "EnergyWeight": f"{result['loss_weights']['energy_weight']:.2f}",
        "StressWeight": f"{result['loss_weights']['stress_weight']:.2f}",
        "PolyEnergyErrorOld": percentage(old_result["energy_relative_l2"]),
        "PolyStressErrorOld": percentage(old_result["stress_relative_l2"]),
        "PolySxyErrorOld": percentage(old_result["stress_component_relative_l2"][2]),
        "PolyVonMisesErrorOld": percentage(old_result["von_mises_relative_l2"]),
        "ComponentStressWeightOld": f"{old_result['loss_weights']['component_stress_weight']:.2f}",
        "IcnnWidthsOld": ",".join(str(w) for w in old_result["icnn_widths"]),
        "ControlEnergyError": percentage(control_result["energy_relative_l2"]),
        "ControlStressError": percentage(control_result["stress_relative_l2"]),
        "ControlVonMisesError": percentage(control_result["von_mises_relative_l2"]),
    }

    if ickan_result is not None:
        ickan_audit = ickan_result["guarantee_audit"]
        names.update({
            "IckanEnergyError": percentage(ickan_result["energy_relative_l2"]),
            "IckanStressError": percentage(ickan_result["stress_relative_l2"]),
            "IckanSxxError": percentage(ickan_result["stress_component_relative_l2"][0]),
            "IckanSyyError": percentage(ickan_result["stress_component_relative_l2"][1]),
            "IckanSxyError": percentage(ickan_result["stress_component_relative_l2"][2]),
            "IckanVonMisesError": percentage(ickan_result["von_mises_relative_l2"]),
            "IckanHiddenWidths": ",".join(str(w) for w in ickan_result["ickan_hidden_widths"]),
            "IckanGrid": str(ickan_result["ickan_grid"]),
            "IckanSplineOrder": str(ickan_result["ickan_spline_order"]),
            "IckanBestEpoch": str(ickan_result["best_epoch"]),
            "IckanRankOneCurvature": f"{ickan_audit['rank_one_curvature_audit']['minimum_second_derivative'] / 1.0e9:.3e}\\,\\mathrm{{GPa}}",
            "IckanReferenceStressNorm": f"{ickan_audit['reference']['stress_norm']:.3e}",
            "IckanSampledMinimumEnergy": f"{ickan_audit['broad_energy_sampling_audit']['minimum_energy'] / 1.0e9:.3e}\\,\\mathrm{{GPa}}",
        })

    if free_matched_result is not None:
        free_audit = free_matched_result["rank_one_curvature_audit"]
        names.update({
            "FreeMatchedEnergyError": percentage(free_matched_result["energy_relative_l2"]),
            "FreeMatchedStressError": percentage(free_matched_result["stress_relative_l2"]),
            "FreeMatchedSxxError": percentage(free_matched_result["stress_component_relative_l2"][0]),
            "FreeMatchedSyyError": percentage(free_matched_result["stress_component_relative_l2"][1]),
            "FreeMatchedSxyError": percentage(free_matched_result["stress_component_relative_l2"][2]),
            "FreeMatchedVonMisesError": percentage(free_matched_result["von_mises_relative_l2"]),
            "FreeMatchedWidths": ",".join(str(w) for w in free_matched_result["widths"]),
            "FreeRankOneSamples": str(free_audit["n_valid_paths"]),
            "FreeRankOneViolations": str(free_audit["n_violations"]),
            "FreeRankOneViolationFraction": percentage(free_audit["fraction_violations"]),
            "FreeRankOneMinimumCurvature": f"{free_audit['minimum_curvature'] / 1.0e9:.3e}\\,\\mathrm{{GPa}}",
        })

    regression_result = json.loads(REGRESSION_METRICS.read_text(encoding="utf-8")) if REGRESSION_METRICS.exists() else None
    regression_training = (
        json.loads(REGRESSION_TRAINING_SUMMARY.read_text(encoding="utf-8")) if REGRESSION_TRAINING_SUMMARY.exists() else None
    )
    c1_result = json.loads(C1_CYCLIC_METRICS.read_text(encoding="utf-8")) if C1_CYCLIC_METRICS.exists() else None

    if regression_result is not None:
        names.update({
            "RegressionStressError": percentage(regression_result["stress_relative_l2"]),
            "RegressionSxxError": percentage(regression_result["stress_component_relative_l2"][0]),
            "RegressionSyyError": percentage(regression_result["stress_component_relative_l2"][1]),
            "RegressionSxyError": percentage(regression_result["stress_component_relative_l2"][2]),
            "RegressionVonMisesError": percentage(regression_result["von_mises_relative_l2"]),
            "RegressionWidths": ",".join(str(w) for w in regression_result["widths"]),
            "RegressionBestEpoch": str(regression_result["best_epoch"]),
        })
    if regression_training is not None:
        names["RegressionTrainStressError"] = percentage(regression_training["training_metrics"]["training_stress_relative_l2"])

    def pascals(value: float) -> str:
        return f"{value:.3e}\\,\\mathrm{{Pa}}"

    if c1_result is not None:
        loops = c1_result["cyclic_work_per_model_per_loop"]
        loop_key_to_macro = {
            "small_normal_square": "SmallSquare",
            "large_normal_square": "LargeSquare",
            "shear_loop": "Shear",
            "mixed_triangle": "Triangle",
        }
        model_key_to_macro = {
            "regression_baseline_tier1": "Regression",
            "free_hyperelastic_tier2": "Free",
            "polyconvex_icnn_tier3a": "Icnn",
            "polyconvex_ickan_tier3b": "Ickan",
        }
        for loop_key, loop_macro in loop_key_to_macro.items():
            for model_key, model_macro in model_key_to_macro.items():
                names[f"Cyclic{loop_macro}{model_macro}"] = pascals(loops[model_key][loop_key])
        names["CyclicPointsPerEdge"] = str(c1_result["points_per_edge"])
        names["CyclicTypicalEnergyScale"] = pascals(c1_result["typical_energy_scale_for_context"])
        names["CyclicRegressionWorstFraction"] = percentage(c1_result["regression_worst_case_fraction_of_typical_energy"])

        refinement = c1_result["discretization_refinement_check"]
        levels = refinement["points_per_edge_levels"]
        names["CyclicRefinementLoop"] = refinement["loop"].replace("_", " ")
        names["CyclicRefinementLevels"] = ", ".join(str(v) for v in levels)
        for model_key, model_macro in model_key_to_macro.items():
            values = refinement["cyclic_work_per_model_per_level"][model_key]
            names[f"CyclicRefinement{model_macro}"] = ", ".join(pascals(v) for v in values)

    if HPROM_ANN_DIRECT_METRICS.exists():
        hprom_direct = json.loads(HPROM_ANN_DIRECT_METRICS.read_text(encoding="utf-8"))
        hprom_iter = hprom_direct["modes"]["hprom_ann_iterative"]
        dhprom = hprom_direct["modes"]["dhprom_ann_direct"]
        cross_check = hprom_direct["old_reference_cross_check"]
        names.update({
            "HpromIterStressError": percentage(hprom_iter["stress_relative_l2"]),
            "HpromIterSxxError": percentage(hprom_iter["stress_component_relative_l2"][0]),
            "HpromIterSyyError": percentage(hprom_iter["stress_component_relative_l2"][1]),
            "HpromIterSxyError": percentage(hprom_iter["stress_component_relative_l2"][2]),
            "HpromIterVonMisesError": percentage(hprom_iter["von_mises_relative_l2"]),
            "DHpromStressError": percentage(dhprom["stress_relative_l2"]),
            "DHpromSxxError": percentage(dhprom["stress_component_relative_l2"][0]),
            "DHpromSyyError": percentage(dhprom["stress_component_relative_l2"][1]),
            "DHpromSxyError": percentage(dhprom["stress_component_relative_l2"][2]),
            "DHpromVonMisesError": percentage(dhprom["von_mises_relative_l2"]),
            "DHpromOldStressError": percentage(cross_check["old_reference_stress_relative_l2"]),
            "DHpromOldSxyError": percentage(cross_check["old_reference_stress_component_relative_l2"][2]),
            "DHpromNewOverOldRatio": f"{cross_check['new_over_old_ratio']:.2f}",
            "HpromControlFomStressError": f"{hprom_direct['control_full_fom_single_run_vs_stage10_stress_relative_l2']:.2e}",
        })

    MACROS.write_text("\n".join(f"\\newcommand{{\\{key}}}{{{value}}}" for key, value in names.items()) + "\n", encoding="utf-8")
    subprocess.run(["pdflatex", "-interaction=nonstopmode", "PANN_anisotropic_claude.tex"], cwd=HERE, check=True)
    subprocess.run(["pdflatex", "-interaction=nonstopmode", "PANN_anisotropic_claude.tex"], cwd=HERE, check=True)


if __name__ == "__main__":
    main()
