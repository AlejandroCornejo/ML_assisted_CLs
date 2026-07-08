import argparse
import os

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ["MPLCONFIGDIR"] = os.path.join(_SCRIPT_DIR, ".mplconfig")
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np

from ickan_workflow import (
    apply_normalization,
    compute_metrics,
    compute_reference_energy,
    create_model,
    ensure_dir,
    load_checkpoint,
    load_external_history,
    load_fom_dataset,
    parse_trajectory_ids,
    plot_prediction_results,
    predict_normalized,
    save_json,
)


REPO_ROOT = os.path.dirname(_SCRIPT_DIR)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Load a trained Sebastian ICKAN/KAN checkpoint and predict FOM trajectories."
    )
    parser.add_argument(
        "--checkpoint",
        default=os.path.join(_SCRIPT_DIR, "ICKAN_training", "ICKAN_model_checkpoint.pth"),
        help="Checkpoint produced by train_ICKAN_surrogate.py.",
    )
    parser.add_argument(
        "--fom-dir",
        default=os.path.join(REPO_ROOT, "stage_1_training_set_fom"),
        help="Directory containing trajectory_i FOM folders.",
    )
    parser.add_argument(
        "--out-dir",
        default=os.path.join(_SCRIPT_DIR, "ICKAN_prediction"),
        help="Directory where prediction arrays and plots are written.",
    )
    parser.add_argument(
        "--trajectories",
        default="9-10",
        help="Trajectory ids to predict. Examples: '9-10', '1-8', 'all'.",
    )
    parser.add_argument(
        "--strain-npy",
        default=None,
        help="External strain history with shape [steps, 3], for example Stage12 fom_strain.npy.",
    )
    parser.add_argument(
        "--stress-npy",
        default=None,
        help="External reference stress history with shape [steps, 3], for example Stage12 fom_stress.npy.",
    )
    parser.add_argument(
        "--stage12-results-dir",
        default=None,
        help="Stage12 output directory. Uses fom_strain.npy and fom_stress.npy from this folder.",
    )
    parser.add_argument(
        "--trajectory-label",
        default="external",
        help="Label used in plots when --strain-npy/--stress-npy are provided.",
    )
    parser.add_argument(
        "--strain-source",
        default=None,
        choices=("strain", "applied_strain"),
        help="Override strain source. By default this is read from the checkpoint.",
    )
    parser.add_argument(
        "--samples-per-trajectory",
        type=int,
        default=None,
        help="Override number of samples. By default this is read from the checkpoint.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint_path = os.path.abspath(args.checkpoint)
    out_dir = ensure_dir(os.path.abspath(args.out_dir))
    fom_dir = os.path.abspath(args.fom_dir)

    checkpoint = load_checkpoint(checkpoint_path)
    model_config = checkpoint["model_config"]
    normalization = checkpoint["normalization"]
    training_info = checkpoint.get("training", {})

    strain_source = args.strain_source or training_info.get("strain_source", "strain")
    if args.stage12_results_dir is not None:
        stage12_dir = os.path.abspath(args.stage12_results_dir)
        if args.strain_npy is None:
            args.strain_npy = os.path.join(stage12_dir, "fom_strain.npy")
        if args.stress_npy is None:
            args.stress_npy = os.path.join(stage12_dir, "fom_stress.npy")
        if args.trajectory_label == "external":
            args.trajectory_label = "stage12_fom"

    using_external_history = args.strain_npy is not None or args.stress_npy is not None
    if using_external_history and (args.strain_npy is None or args.stress_npy is None):
        raise ValueError("--strain-npy and --stress-npy must be provided together.")

    if using_external_history:
        samples_per_trajectory = args.samples_per_trajectory
        trajectory_ids = [str(args.trajectory_label)]
    else:
        samples_per_trajectory = (
            args.samples_per_trajectory
            if args.samples_per_trajectory is not None
            else training_info.get("samples_per_trajectory", 500)
        )
        trajectory_ids = parse_trajectory_ids(args.trajectories)

    print("[SEBASTIAN-ICKAN][PREDICT]")
    print(f"Checkpoint             : {checkpoint_path}")
    print(f"FOM input dir          : {fom_dir}")
    print(f"Output dir             : {out_dir}")
    print(f"Input mode             : {'external .npy history' if using_external_history else 'stage_1 trajectory folders'}")
    print(f"Predicted trajectories : {trajectory_ids}")
    print(f"Strain source          : {strain_source}")
    print(f"Samples per trajectory : {samples_per_trajectory}")

    model, _ = create_model(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    kan_backend = getattr(model, "kan_backend", checkpoint.get("kan_backend", "unknown"))
    print(f"KAN backend             : {kan_backend}")

    if using_external_history:
        dataset = load_external_history(
            strain_path=os.path.abspath(args.strain_npy),
            stress_path=os.path.abspath(args.stress_npy),
            label=args.trajectory_label,
            samples_per_trajectory=samples_per_trajectory,
        )
    else:
        dataset = load_fom_dataset(
            fom_dir=fom_dir,
            trajectory_ids=trajectory_ids,
            strain_source=strain_source,
            samples_per_trajectory=samples_per_trajectory,
        )

    strain_normalized, stress_reference_normalized = apply_normalization(
        dataset["strain"],
        dataset["stress"],
        normalization,
    )
    energy_reference_raw = compute_reference_energy(
        strain_normalized,
        stress_reference_normalized,
    )
    energy_reference_normalized = energy_reference_raw / float(normalization["max_w"])

    prediction = predict_normalized(
        model=model,
        strain_normalized=strain_normalized,
        max_w=normalization["max_w"],
    )
    stress_predicted_normalized = prediction["stress_normalized"]
    energy_predicted_normalized = prediction["W_normalized"]

    stress_predicted = stress_predicted_normalized * float(normalization["stress_scale"])
    metrics_normalized = compute_metrics(
        stress_reference_normalized,
        stress_predicted_normalized,
    )
    metrics_physical = compute_metrics(dataset["stress"], stress_predicted)

    np.savez(
        os.path.join(out_dir, "predictions.npz"),
        trajectory_ids=np.asarray(trajectory_ids),
        strain=dataset["strain"],
        stress_reference=dataset["stress"],
        stress_predicted=stress_predicted,
        strain_normalized=strain_normalized,
        stress_reference_normalized=stress_reference_normalized,
        stress_predicted_normalized=stress_predicted_normalized,
        W_reference_normalized=energy_reference_normalized,
        W_predicted_normalized=energy_predicted_normalized,
    )

    metadata = {
        "checkpoint": checkpoint_path,
        "kan_backend": kan_backend,
        "fom_dir": fom_dir,
        "trajectory_ids": trajectory_ids,
        "input_mode": "external_npy" if using_external_history else "stage_1_fom",
        "stage12_results_dir": os.path.abspath(args.stage12_results_dir) if args.stage12_results_dir else None,
        "strain_npy": os.path.abspath(args.strain_npy) if args.strain_npy else None,
        "stress_npy": os.path.abspath(args.stress_npy) if args.stress_npy else None,
        "strain_source": strain_source,
        "samples_per_trajectory": dataset["samples_per_trajectory"],
        "normalization": normalization,
        "training_info": training_info,
        "metrics_normalized": metrics_normalized,
        "metrics_physical": metrics_physical,
    }
    save_json(os.path.join(out_dir, "prediction_metrics.json"), metadata)

    plot_prediction_results(
        out_dir=out_dir,
        trajectory_ids=trajectory_ids,
        strain_normalized=strain_normalized,
        stress_reference_normalized=stress_reference_normalized,
        stress_predicted_normalized=stress_predicted_normalized,
        energy_reference_normalized=energy_reference_normalized,
        energy_predicted_normalized=energy_predicted_normalized,
    )

    print("\nPrediction finished.")
    print(f"Normalized relative MSE: {metrics_normalized['relative_mse_global']:.8E}")
    print(f"Normalized relative L2 : {metrics_normalized['relative_l2_global']:.8E}")
    print(f"Physical relative MSE  : {metrics_physical['relative_mse_global']:.8E}")
    print(f"Physical relative L2   : {metrics_physical['relative_l2_global']:.8E}")
    print(f"Plots and arrays saved to: {out_dir}")


if __name__ == "__main__":
    main()
