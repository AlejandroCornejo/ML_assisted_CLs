import argparse
import os

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplcfg")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np


REFERENCE_COLOR = "#000000"
ICKAN_COLOR = "#d62728"
ICNN_COLOR = "#1f77b4"
EXTRA_COLORS = ("#006400", "#ff7f0e")
COMPONENT_TEX = {
    "xx": r"xx",
    "yy": r"yy",
    "xy": r"xy",
}
INVARIANT_LABELS = {
    "K1_I1bar_minus_3": r"$K_1 = \bar{I}_1 - 3 = I_1 I_3^{-1/3} - 3$",
    "K2_I2bar_power_minus_3sqrt3": (
        r"$K_2 = \bar{I}_2^{3/2} - 3\sqrt{3}"
        r" = \left[(I_1+I_3-1)I_3^{-2/3}\right]^{3/2}-3\sqrt{3}$"
    ),
    "K3_Jminus1_squared": r"$K_3 = (J - 1)^2 = (\sqrt{I_3}-1)^2$",
    "J": r"$J = \sqrt{I_3}$",
    "C12_signed": r"$C_{12}$",
}
INVARIANT_CONTEXT_LINE_1 = (
    r"$C = I + 2E = "
    r"\begin{bmatrix}"
    r"1+2E_{xx} & \gamma_{xy} & 0\\"
    r"\gamma_{xy} & 1+2E_{yy} & 0\\"
    r"0 & 0 & 1"
    r"\end{bmatrix},\quad "
    r"I_1=\mathrm{tr}(C),\quad I_3=\det(C),\quad J=\sqrt{I_3}$"
)
INVARIANT_CONTEXT_LINE_2 = (
    r"$\bar{I}_1=I_1I_3^{-1/3},\quad "
    r"\bar{I}_2=(I_1+I_3-1)I_3^{-2/3}$"
)


def configure_latex_matplotlib():
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "text.latex.preamble": r"\usepackage{amsmath}",
            "axes.labelsize": 14,
            "axes.titlesize": 16,
            "legend.fontsize": 12,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
        }
    )


def latex_text(text):
    return str(text).replace("_", r"\_")


def invariant_label(name):
    return INVARIANT_LABELS.get(name, latex_text(name))


def stress_label(component_name):
    return rf"$S_{{{COMPONENT_TEX.get(component_name, component_name)}}}$"


def normalized_stress_label(component_name):
    return rf"$\widehat{{S}}_{{{COMPONENT_TEX.get(component_name, component_name)}}}$"


def add_invariant_formula_footer(fig):
    fig.text(0.5, 0.050, INVARIANT_CONTEXT_LINE_1, ha="center", va="center", fontsize=9)
    fig.text(0.5, 0.020, INVARIANT_CONTEXT_LINE_2, ha="center", va="center", fontsize=9)


def model_color(label, model_index=0):
    lower_label = str(label).lower()
    if "ickan" in lower_label:
        return ICKAN_COLOR
    if "balanced" in lower_label:
        return EXTRA_COLORS[0]
    if "icnn" in lower_label:
        return ICNN_COLOR
    return EXTRA_COLORS[min(model_index, len(EXTRA_COLORS) - 1)]


def reference_style(alpha=0.42):
    return {
        "s": 9,
        "alpha": alpha,
        "marker": "o",
        "linewidths": 0.0,
        "color": REFERENCE_COLOR,
    }


def model_style(color):
    return {
        "s": 8,
        "alpha": 0.68,
        "marker": "x",
        "linewidths": 0.6,
        "color": color,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot stress and energy predictions as functions of C-based invariants."
    )
    parser.add_argument(
        "--prediction-dirs",
        nargs="+",
        required=True,
        help="Prediction directories containing predictions.npz.",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Optional labels matching --prediction-dirs.",
    )
    parser.add_argument(
        "--axis-source",
        default="applied_strain",
        choices=("applied_strain", "strain"),
        help="Use imposed applied strain or homogenized strain to compute invariants.",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Directory where invariant plots are written.",
    )
    return parser.parse_args()


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def compute_invariants(strain):
    """Compute plane-strain 3D C-based invariant features from [E_xx, E_yy, gamma_xy]."""
    flat = strain.reshape(-1, 3)
    c11 = 1.0 + 2.0 * flat[:, 0]
    c22 = 1.0 + 2.0 * flat[:, 1]
    c12 = flat[:, 2]
    i1 = c11 + c22 + 1.0
    i3 = np.maximum(c11 * c22 - c12 * c12, 1.0e-12)
    j = np.sqrt(i3)
    i1_bar_minus_3 = i1 * np.power(i3, -1.0 / 3.0) - 3.0
    i2_bar = (i1 + i3 - 1.0) * np.power(i3, -2.0 / 3.0)
    k2 = np.power(np.maximum(i2_bar, 1.0e-12), 1.5) - 3.0 * np.sqrt(3.0)
    k3 = (j - 1.0) ** 2
    return {
        "K1_I1bar_minus_3": i1_bar_minus_3,
        "K2_I2bar_power_minus_3sqrt3": k2,
        "K3_Jminus1_squared": k3,
        "J": j,
        "C12_signed": c12,
    }


def load_prediction(prediction_dir, axis_source):
    path = os.path.join(prediction_dir, "predictions.npz")
    data = np.load(path)
    if axis_source == "applied_strain":
        strain_axis = data["applied_strain_plot_axis"]
    else:
        strain_axis = data["strain_plot_axis"]

    return {
        "prediction_dir": prediction_dir,
        "invariants": compute_invariants(strain_axis),
        "stress_reference": data["stress_reference"].reshape(-1, 3),
        "stress_predicted": data["stress_predicted"].reshape(-1, 3),
        "stress_reference_normalized": data["stress_reference_normalized"].reshape(-1, 3),
        "stress_predicted_normalized": data["stress_predicted_normalized"].reshape(-1, 3),
        "W_reference_normalized": data["W_reference_normalized"].reshape(-1),
        "W_predicted_normalized": data["W_predicted_normalized"].reshape(-1),
    }


def plot_case(case, label, out_dir):
    component_names = ("xx", "yy", "xy")
    color = model_color(label)
    display_label = latex_text(label)
    for invariant_name, invariant_values in case["invariants"].items():
        x = invariant_values
        x_label = invariant_label(invariant_name)

        fig, axes = plt.subplots(2, 2, figsize=(13, 9))
        axes = axes.ravel()
        for component, component_name in enumerate(component_names):
            ax = axes[component]
            ax.scatter(
                x,
                case["stress_reference_normalized"][:, component],
                label="Reference",
                **reference_style(alpha=0.48),
            )
            ax.scatter(
                x,
                case["stress_predicted_normalized"][:, component],
                label=display_label,
                **model_style(color),
            )
            ax.set_xlabel(x_label)
            ax.set_ylabel(rf"Normalized stress {normalized_stress_label(component_name)}")
            ax.grid(True)
            ax.legend()

        ax = axes[3]
        ax.scatter(
            x,
            case["W_reference_normalized"],
            label="Reference",
            **reference_style(alpha=0.48),
        )
        ax.scatter(
            x,
            case["W_predicted_normalized"],
            label=display_label,
            **model_style(color),
        )
        ax.set_xlabel(x_label)
        ax.set_ylabel(r"Normalized energy $\widehat{W}$")
        ax.grid(True)
        ax.legend()

        fig.suptitle(rf"{display_label}: stresses and energy vs {x_label}")
        add_invariant_formula_footer(fig)
        fig.tight_layout(rect=[0.0, 0.085, 1.0, 0.93])
        fig.savefig(os.path.join(out_dir, f"{label}_stress_energy_vs_{invariant_name}.png"), dpi=200)
        plt.close(fig)


def plot_energy_comparison(cases, labels, out_dir):
    for invariant_name in cases[0]["invariants"]:
        fig, ax = plt.subplots(figsize=(9, 6))
        x_label = invariant_label(invariant_name)
        reference_labeled = False
        for model_index, (case, label) in enumerate(zip(cases, labels)):
            invariant_values = case["invariants"][invariant_name]
            x = invariant_values
            color = model_color(label, model_index)
            display_label = latex_text(label)
            ax.scatter(
                x,
                case["W_reference_normalized"],
                label="Reference" if not reference_labeled else None,
                **reference_style(alpha=0.25),
            )
            reference_labeled = True
            ax.scatter(
                x,
                case["W_predicted_normalized"],
                label=display_label,
                **model_style(color),
            )
        ax.set_xlabel(x_label)
        ax.set_ylabel(r"Normalized energy $\widehat{W}$")
        ax.set_title(rf"Energy vs {x_label}")
        ax.grid(True)
        ax.legend()
        add_invariant_formula_footer(fig)
        fig.tight_layout(rect=[0.0, 0.13, 1.0, 0.94])
        fig.savefig(os.path.join(out_dir, f"comparison_energy_vs_{invariant_name}.png"), dpi=200)
        plt.close(fig)


def main():
    args = parse_args()
    configure_latex_matplotlib()
    if args.labels is not None and len(args.labels) != len(args.prediction_dirs):
        raise ValueError("--labels must have the same length as --prediction-dirs.")

    out_dir = ensure_dir(os.path.abspath(args.out_dir))
    labels = args.labels
    if labels is None:
        labels = [os.path.basename(os.path.normpath(path)) for path in args.prediction_dirs]

    cases = [
        load_prediction(os.path.abspath(path), args.axis_source)
        for path in args.prediction_dirs
    ]

    for case, label in zip(cases, labels):
        plot_case(case, label, out_dir)
    if len(cases) > 1:
        plot_energy_comparison(cases, labels, out_dir)

    print(f"Invariant plots saved to: {out_dir}")


if __name__ == "__main__":
    main()
