import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np

from ickan_workflow import plot_prediction_results


BEST_CASES = (
    ("ICKAN_best_prediction", "ICKAN"),
    ("ICNN_best_prediction", "ICNN"),
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Regenerate the standard best prediction plots from predictions.npz."
    )
    parser.add_argument(
        "--best-dir",
        default="Sebastian_ICKAN_Tests/best_ICKAN_vs_ICNN_predictions",
        help="Directory containing the copied best prediction folders.",
    )
    return parser.parse_args()


def load_axis_variants(data):
    axis_variants = []
    if "applied_strain_plot_axis_normalized" in data.files:
        axis_variants.append(
            (
                "applied_strain",
                "applied strain",
                data["applied_strain_plot_axis_normalized"],
            )
        )
    if "strain_plot_axis_normalized" in data.files:
        axis_variants.append(
            (
                "strain",
                "homogenized strain",
                data["strain_plot_axis_normalized"],
            )
        )
    return axis_variants


def main():
    args = parse_args()
    best_dir = Path(args.best_dir).resolve()

    regenerated = []
    for folder_name, label in BEST_CASES:
        prediction_dir = best_dir / folder_name
        prediction_file = prediction_dir / "predictions.npz"
        if not prediction_file.exists():
            continue

        data = np.load(prediction_file)
        plot_prediction_results(
            str(prediction_dir),
            data["trajectory_ids"],
            data["strain_normalized"],
            data["stress_reference_normalized"],
            data["stress_predicted_normalized"],
            data["W_reference_normalized"],
            data["W_predicted_normalized"],
            strain_axis_variants=load_axis_variants(data),
            prediction_label=label,
        )
        regenerated.append(folder_name)

    print("Regenerated best prediction plots:")
    for folder_name in regenerated:
        print(f"  {folder_name}")


if __name__ == "__main__":
    main()
