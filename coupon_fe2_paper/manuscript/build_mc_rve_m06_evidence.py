#!/usr/bin/env python3
"""Build the MC-RVE m=6 evidence: Table 3 and the beyond-the-data figure."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
TABLES = HERE / "tables"
BASE = PROJECT / "07_material_b"
EVALUATION = BASE / "results/feature_count_analysis_v1/m06_independent_evaluation_v1"
# The Unconstrained energy (internal name Free) has no paired-feature count. It is the
# free_large cell of the capacity study, trained under the same rule as the m=6 fits.
UNCONSTRAINED = BASE / "results/capacity_2x2_v1"
LABELS = BASE / "results/data_labels_v1.npz"
METRICS = ("stress", "energy", "tangent")
BEYOND = BASE / "results/feature_count_analysis_v1/m06_beyond_data_v1"
FIGURES = HERE / "figures"
# Model colors shared by every figure (build_evidence.py MODEL_COLORS): ICNN green, ICKAN blue,
# Unconstrained red, checked for color-vision deficiency. The box ends at E11=E22=-0.04, i.e. J=0.92.
CURVES = (("ICNN-learned", "ICNN, learned", "#2AA780"),
          ("ICKAN-learned", "ICKAN, learned", "#2B5DAA"),
          ("Unconstrained", "Unconstrained energy", "#C44939"))
BOX_J = 0.92


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


def load_unconstrained() -> dict:
    receipt = json.loads((UNCONSTRAINED / "evaluation_gateA.json").read_text())
    gate = receipt["gate"]
    if (gate.get("status") != "opened_once" or gate.get("labels_sha256") != digest(LABELS)
            or gate.get("capacity_rule_sha256") != digest(UNCONSTRAINED / "training_rule.json")
            or gate.get("evaluator_sha256")
            != digest(UNCONSTRAINED / "executed_evaluate_capacity_v1.py")):
        raise ValueError("Unconstrained-energy evaluation receipt is invalid")
    seeds = sorted(int(row["seed"]) for row in receipt["rows"] if row["cell"] == "free_large")
    if seeds != [16, 29, 47]:
        raise ValueError("Unexpected Unconstrained-energy seeds")
    return {metric: receipt["summary"]["free_large"]["test"][metric] for metric in METRICS}


def value(cell: dict) -> str:
    return f"{cell['median']:#.3g} [{cell['minimum']:#.3g}, {cell['maximum']:#.3g}]"


def median(cell: dict) -> str:
    return f"{cell['median']:.4f}"


def table(summary: dict, unconstrained: dict) -> None:
    # The caption states that every learned seed beats every fixed seed.
    for core in ("ICNN", "ICKAN"):
        fixed = summary["by_model"][f"{core}-fixed"]["test_aggregate_percent"]
        learned = summary["by_model"][f"{core}-learned"]["test_aggregate_percent"]
        if any(learned[metric]["maximum"] >= fixed[metric]["minimum"] for metric in METRICS):
            raise ValueError("Seed ranges overlap; revise the Table 3 caption")
    content = [
        r"\begin{tabular}{lccc}", r"\toprule",
        r"Model & Stress [\%] & Energy [\%] & Tangent [\%]\\",
        r"\midrule",
    ]
    for core in ("ICNN", "ICKAN"):
        for variant in ("fixed", "learned"):
            cells = summary["by_model"][f"{core}-{variant}"]["test_aggregate_percent"]
            content.append(rf"{core}, {variant} & "
                           + " & ".join(median(cells[metric]) for metric in METRICS) + r"\\")
        if core == "ICNN":
            content.append(r"\addlinespace")
    content.extend([r"\midrule",
                    r"Unconstrained energy & "
                    + " & ".join(median(unconstrained[metric]) for metric in METRICS) + r"\\",
                    r"\bottomrule", r"\end{tabular}", ""])
    TABLES.mkdir(exist_ok=True)
    (TABLES / "mc_rve_m06_stress_errors.tex").write_text("\n".join(content))


def beyond_data_figure() -> None:
    summary = json.loads((BEYOND / "summary.json").read_text())
    if summary.get("status") != "complete" or len(summary["rows"]) != 15:
        raise ValueError("Beyond-the-data audit is incomplete")
    plt.rcParams.update({
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{lmodern}",
        "font.family": "serif",
        "font.size": 9,
        "axes.linewidth": 0.7,
    })
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 2.85))
    for model, label, color in CURVES:
        rows = [row for row in summary["rows"] if row["model"] == model]
        if len(rows) != 3:
            raise ValueError(f"Expected three seeds for {model}")
        # One line per energy (median over the seeds) and a band spanning the three seeds.
        for ax, key, value, first in ((axes[0], "equibiaxial_path", "minimum_curvature_Pa", 0),
                                      (axes[1], "collapse", "energy_Pa", 1)):
            J = np.asarray(rows[0][key]["J"])
            assert all(np.array_equal(J, row[key]["J"]) for row in rows)
            curves = np.array([row[key][value] for row in rows])[:, first:] / 1e6
            ax.fill_between(J[first:], curves.min(axis=0), curves.max(axis=0), color=color,
                            alpha=0.16, linewidth=0, zorder=1)
            ax.plot(J[first:], np.median(curves, axis=0), color=color, linewidth=1.45,
                    label=label if ax is axes[0] else None, zorder=2)
    axes[0].axvspan(BOX_J, 1.0, color="#E3E6EA", linewidth=0, zorder=0)
    axes[0].axhline(0.0, color="#3F4650", linewidth=0.8, zorder=1)
    axes[0].set_xlim(1.0, 0.3025)
    axes[0].set_ylabel(r"Minimum rank-one curvature [MPa]")
    axes[0].set_title(r"(a) Rank-one curvature", fontsize=10, fontweight="bold", pad=5)
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlim(1.0, 1e-8)
    axes[1].set_ylim(1e-1, 1e5)
    axes[1].set_ylabel(r"Energy density [MPa]")
    axes[1].set_title(r"(b) Energy toward volume collapse", fontsize=10, fontweight="bold", pad=5)
    for ax in axes:
        ax.set_xlabel(r"$J$")
        ax.grid(which="major", color="#D7DBE0", linewidth=0.55, alpha=0.75)
        ax.set_axisbelow(True)
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(Patch(facecolor="#E3E6EA", edgecolor="none"))
    labels.append("Inside the sampled box")
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 1.01), handlelength=2.0)
    fig.subplots_adjust(left=0.085, right=0.965, bottom=0.17, top=0.79, wspace=0.28)
    FIGURES.mkdir(exist_ok=True)
    pdf = FIGURES / "mc_beyond_data.pdf"
    fig.savefig(pdf, facecolor="white")
    plt.close(fig)
    subprocess.run(["pdftoppm", "-png", "-singlefile", "-r", "220", str(pdf),
                    str(FIGURES / "mc_beyond_data")], check=True, stdout=subprocess.DEVNULL)


def main() -> None:
    table(load(), load_unconstrained())
    beyond_data_figure()


if __name__ == "__main__":
    main()
