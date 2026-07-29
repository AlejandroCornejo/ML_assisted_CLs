#!/usr/bin/env python3
"""Build publication-style figures and LaTeX numeric macros from FOM results."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
import numpy as np


ROOT = Path(__file__).resolve().parent
FIGURES = ROOT / "figures"


def scientific_tex(value: float, digits: int = 3) -> str:
    """Format a positive scalar as a compact LaTeX scientific number."""
    if value == 0.0:
        return "0"
    exponent = int(np.floor(np.log10(abs(value))))
    mantissa = value / 10.0**exponent
    return rf"{mantissa:.{digits}f}\times 10^{{{exponent}}}"


def load_results() -> tuple[dict, dict]:
    d4_path = ROOT / "results" / "d4_symmetry_summary.json"
    iso_path = ROOT / "isotropy_diagnostic" / "isotropy_diagnostic_summary.json"
    if not d4_path.exists() or not iso_path.exists():
        raise FileNotFoundError(
            "Missing FOM summaries. Run run_d4_symmetry_test.py and "
            "run_non_d4_rotation_diagnostic.py before building the report."
        )
    return json.loads(d4_path.read_text(encoding="utf-8")), json.loads(iso_path.read_text(encoding="utf-8"))


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.grid": True,
            "grid.alpha": 0.28,
            "grid.linestyle": ":",
            "figure.dpi": 160,
            "savefig.bbox": "tight",
        }
    )


def make_rve_schematic() -> None:
    fig, ax = plt.subplots(figsize=(5.7, 4.6))
    ax.add_patch(Rectangle((-1.0, -1.0), 2.0, 2.0, fill=False, linewidth=2.4, edgecolor="#173F5F"))
    ax.add_patch(Circle((0.0, 0.0), 0.50, facecolor="#F6C85F", edgecolor="#9A6B00", linewidth=1.6, alpha=0.92))
    ax.text(0.0, 0.0, "hueco", ha="center", va="center", fontsize=11, weight="bold")

    ax.add_patch(FancyArrowPatch((-0.82, -0.80), (-0.22, -0.80), arrowstyle="->", mutation_scale=15, linewidth=1.7, color="#20639B"))
    ax.add_patch(FancyArrowPatch((-0.82, -0.80), (-0.82, -0.20), arrowstyle="->", mutation_scale=15, linewidth=1.7, color="#20639B"))
    ax.text(-0.18, -0.86, r"$\mathbf{e}_x$", color="#20639B", fontsize=11)
    ax.text(-0.95, -0.15, r"$\mathbf{e}_y$", color="#20639B", fontsize=11)

    for x, y, dx, dy in ((0.52, 0.73, -0.28, 0.0), (0.73, -0.52, 0.0, 0.28), (-0.52, -0.73, 0.28, 0.0), (-0.73, 0.52, 0.0, -0.28)):
        ax.add_patch(FancyArrowPatch((x, y), (x + dx, y + dy), arrowstyle="->", mutation_scale=13, linewidth=1.4, color="#ED553B"))
    ax.text(0.0, 1.16, r"Operaciones permitidas: $90^\circ$ y reflexiones", ha="center", fontsize=10)
    ax.text(0.0, -1.23, r"$D_4$: simetría cuadrada; $45^\circ$ no pertenece a $D_4$", ha="center", fontsize=10)

    ax.set_aspect("equal")
    ax.set_xlim(-1.32, 1.32)
    ax.set_ylim(-1.36, 1.34)
    ax.axis("off")
    fig.savefig(FIGURES / "rve_square_symmetry_schematic.pdf")
    plt.close(fig)


def make_d4_error_plot(d4: dict) -> None:
    actions = ["reflect_x", "rotate_90", "reflect_diagonal"]
    action_labels = [r"reflexión $x$", r"rotación $90^\circ$", r"reflexión diagonal"]
    base_names = ["mixed_A", "mixed_B", "mixed_C"]
    comparisons = {(row["base_case"], row["action"]): row for row in d4["comparisons"]}
    colors = ["#20639B", "#3CAEA3", "#ED553B"]
    x = np.arange(len(base_names))
    width = 0.24

    fig, axes = plt.subplots(1, 2, figsize=(8.7, 3.7), sharex=True)
    for index, (action, label, color) in enumerate(zip(actions, action_labels, colors)):
        stress = [comparisons[(base, action)]["stress_rel_error"] for base in base_names]
        energy = [comparisons[(base, action)]["energy_rel_error"] for base in base_names]
        axes[0].bar(x + (index - 1) * width, stress, width, label=label, color=color)
        axes[1].bar(x + (index - 1) * width, energy, width, label=label, color=color)

    for axis, title, ylabel in zip(
        axes,
        ["Tensión PK2", "Energía microscópica"],
        [r"error relativo $e_S$", r"error relativo $e_{\Psi}$"],
    ):
        axis.set_yscale("log")
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.set_xticks(x, ["A", "B", "C"])
        axis.set_xlabel("estado base de deformación")
        axis.set_ylim(1.0e-8, 1.0e-4)
        axis.grid(True, axis="y")
    axes[1].legend(loc="upper left", frameon=False)
    fig.suptitle(r"Verificación FOM de las simetrías $D_4$", y=1.03, fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGURES / "d4_symmetry_errors.pdf")
    plt.close(fig)


def make_isotropy_plot(isotropy: dict) -> None:
    rows = isotropy["rows"]
    labels = [row["base_case"].replace("mixed_", "") for row in rows]
    stress = np.array([row["stress_relative_difference_from_isotropic_prediction"] for row in rows])
    energy = np.array([row["energy_relative_difference_from_isotropic_prediction"] for row in rows])
    x = np.arange(len(rows))

    fig, axes = plt.subplots(1, 2, figsize=(8.7, 3.7))
    values = [100.0 * stress, 100.0 * energy]
    titles = [r"Tensión: discrepancia a $45^\circ$", r"Energía: discrepancia a $45^\circ$"]
    colors = ["#ED553B", "#F6C85F"]
    for axis, data, title, color in zip(axes, values, titles, colors):
        bars = axis.bar(x, data, color=color, edgecolor="#3A3A3A", linewidth=0.6)
        axis.set_xticks(x, labels)
        axis.set_xlabel("estado base de deformación")
        axis.set_title(title)
        axis.set_ylabel("diferencia relativa [%]")
        axis.set_ylim(0.0, 1.20 * max(data))
        for bar, value in zip(bars, data):
            axis.text(bar.get_x() + 0.5 * bar.get_width(), value, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    fig.suptitle(r"Una rotación material de $45^\circ$ no satisface la igualdad isotrópica", y=1.03, fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGURES / "isotropy_45_degree_mismatch.pdf")
    plt.close(fig)


def write_latex_macros(d4: dict, isotropy: dict) -> None:
    d4_errors = d4["max_relative_errors"]
    iso_errors = isotropy["max_relative_differences"]
    rows = isotropy["rows"]
    lines = [
        "% Generated by make_report_figures.py; do not edit manually.",
        rf"\newcommand{{\MaxDfourStressError}}{{{scientific_tex(d4_errors['stress'])}}}",
        rf"\newcommand{{\MaxDfourStrainError}}{{{scientific_tex(d4_errors['homogenized_strain'])}}}",
        rf"\newcommand{{\MaxDfourEnergyError}}{{{scientific_tex(d4_errors['energy'])}}}",
        rf"\newcommand{{\MaxIsoStressMismatch}}{{{scientific_tex(iso_errors['stress'])}}}",
        rf"\newcommand{{\MaxIsoEnergyMismatch}}{{{scientific_tex(iso_errors['energy'])}}}",
    ]
    for row in rows:
        suffix = row["base_case"].split("_")[-1].upper()
        lines.append(rf"\newcommand{{\IsoStress{suffix}}}{{{scientific_tex(row['stress_relative_difference_from_isotropic_prediction'])}}}")
        lines.append(rf"\newcommand{{\IsoEnergy{suffix}}}{{{scientific_tex(row['energy_relative_difference_from_isotropic_prediction'])}}}")
    (FIGURES / "results_macros.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    set_style()
    d4, isotropy = load_results()
    make_rve_schematic()
    make_d4_error_plot(d4)
    make_isotropy_plot(isotropy)
    write_latex_macros(d4, isotropy)
    print(f"Wrote figures and macros in {FIGURES}")


if __name__ == "__main__":
    main()
