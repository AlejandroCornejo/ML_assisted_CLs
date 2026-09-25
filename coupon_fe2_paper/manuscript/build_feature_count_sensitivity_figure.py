#!/usr/bin/env python3
"""Build the validation-only MC-RVE feature-count sensitivity figure."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
FIGURES = HERE / "figures"
SUMMARY = (PROJECT / "07_material_b/results/feature_count_analysis_v1"
           / "validation_audit_v1/validation_summary.json")
COUNTS = np.array((2, 4, 6, 8, 16, 24, 32), dtype=float)
COLORS = {"fixed": "#7C8794", "learned": "#006B6B"}


def collect(summary: dict, core: str, feature_type: str):
    rows = [row for row in summary["rows"]
            if row["core"] == core and row["feature_type"] == feature_type]
    values = []
    for count in COUNTS.astype(int):
        group = sorted((row for row in rows if row["feature_count"] == count),
                       key=lambda row: row["seed"])
        if [row["seed"] for row in group] != [16, 29, 47]:
            raise ValueError(f"Incomplete {core}-{feature_type} group at m={count}")
        values.append([row["best_validation_score"] for row in group])
    array = np.asarray(values)
    return np.median(array, axis=1), np.min(array, axis=1), np.max(array, axis=1)


def main():
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    if len(summary["rows"]) != 84 or summary["counts"] != COUNTS.astype(int).tolist():
        raise ValueError("Unexpected validation-audit contents")

    plt.rcParams.update({
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{lmodern}",
        "font.family": "serif",
        "font.size": 9,
        "axes.linewidth": 0.7,
    })
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 2.85), sharey=True)
    for ax, core, panel in zip(axes, ("ICNN", "ICKAN"), ("(a)", "(b)")):
        for feature_type, linestyle in (("fixed", "--"), ("learned", "-")):
            center, lower, upper = collect(summary, core, feature_type)
            color = COLORS[feature_type]
            ax.fill_between(COUNTS, lower, upper, color=color, alpha=0.16,
                            linewidth=0, zorder=1)
            ax.plot(COUNTS, center, linestyle=linestyle, marker="o", markersize=4,
                    linewidth=1.45, color=color, label=feature_type.capitalize(),
                    zorder=2)
            index = int(np.flatnonzero(COUNTS == 6)[0])
            ax.scatter([6], [center[index]], s=38, facecolor="white", edgecolor=color,
                       linewidth=1.2, zorder=3)
        ax.axvline(6, color="#3F4650", linestyle=":", linewidth=0.8, zorder=0)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(COUNTS, [str(int(value)) for value in COUNTS])
        ax.set_xlim(1.75, 36)
        ax.set_ylim(7e-8, 8e-3)
        ax.grid(which="major", color="#D7DBE0", linewidth=0.55, alpha=0.75)
        ax.grid(which="minor", axis="y", color="#E8EAED", linewidth=0.35, alpha=0.55)
        ax.set_axisbelow(True)
        ax.set_xlabel(r"Number of paired features $m$")
        ax.set_title(rf"{panel} {core} core", fontsize=10, fontweight="bold", pad=5)
    axes[0].set_ylabel("Normalized validation stress MSE")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 1.01), handlelength=2.4)
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.19, top=0.79, wspace=0.08)

    FIGURES.mkdir(exist_ok=True)
    pdf = FIGURES / "feature_count_sensitivity.pdf"
    fig.savefig(pdf, facecolor="white")
    plt.close(fig)
    subprocess.run(["pdfinfo", str(pdf)], check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["pdftoppm", "-png", "-singlefile", "-r", "220", str(pdf),
                    str(FIGURES / "feature_count_sensitivity")], check=True,
                   stdout=subprocess.DEVNULL)


if __name__ == "__main__":
    main()
