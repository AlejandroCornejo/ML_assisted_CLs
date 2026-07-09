import json
import os

import numpy as np
import torch


COMPONENT_NAMES = ("xx", "yy", "xy")
DEFAULT_MODEL_CONFIG = {
    "model_type": "ickan",
    "order_stretches": 1,
    "grid_size": 100,
    "k": 3,
    "W_width": [3, 5, 4, 1, 1],
    "input_mode": "principal",
    "base_fun": "silu",
    "noise_scale": 0.0,
    "grid_eps": 0.01,
    "icnn_activation": "softplus",
    "icnn_softplus_beta": 5.0,
    "icnn_quadratic": True,
}


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def parse_trajectory_ids(spec):
    if spec is None:
        return list(range(1, 11))

    text = str(spec).strip()
    if text.lower() in ("", "all"):
        return list(range(1, 11))

    ids = []
    for chunk in text.replace(" ", "").split(","):
        if not chunk:
            continue
        if "-" in chunk:
            start, end = chunk.split("-", 1)
            start = int(start)
            end = int(end)
            step = 1 if end >= start else -1
            ids.extend(range(start, end + step, step))
        else:
            ids.append(int(chunk))

    clean_ids = []
    for trajectory_id in ids:
        if trajectory_id <= 0:
            raise ValueError(f"Trajectory ids must be positive. Got {trajectory_id}.")
        if trajectory_id not in clean_ids:
            clean_ids.append(trajectory_id)
    return clean_ids


def sample_equally_spaced(data, n_samples):
    original_n = data.shape[0]
    if n_samples is None or n_samples >= original_n:
        return data.copy()
    if n_samples <= 1:
        raise ValueError("n_samples must be greater than 1.")
    indices = np.linspace(0, original_n - 1, n_samples, dtype=int)
    return data[indices].copy()


def load_fom_dataset(fom_dir, trajectory_ids, strain_source="strain", samples_per_trajectory=500):
    raw_strains = []
    raw_stresses = []
    original_lengths = []

    for trajectory_id in trajectory_ids:
        trajectory_dir = os.path.join(fom_dir, f"trajectory_{trajectory_id}")
        strain_path = os.path.join(
            trajectory_dir,
            f"trajectory_{trajectory_id}_{strain_source}.npy",
        )
        stress_path = os.path.join(
            trajectory_dir,
            f"trajectory_{trajectory_id}_stress.npy",
        )

        if not os.path.exists(strain_path):
            raise FileNotFoundError(strain_path)
        if not os.path.exists(stress_path):
            raise FileNotFoundError(stress_path)

        strain = np.load(strain_path)
        stress = np.load(stress_path)

        if strain.shape != stress.shape:
            raise ValueError(
                f"Shape mismatch for trajectory_{trajectory_id}: "
                f"strain={strain.shape}, stress={stress.shape}"
            )
        if strain.ndim != 2 or strain.shape[1] != 3:
            raise ValueError(
                f"Expected trajectory_{trajectory_id} arrays with shape [steps, 3]. "
                f"Got {strain.shape}."
            )

        raw_strains.append(strain)
        raw_stresses.append(stress)
        original_lengths.append(strain.shape[0])

    available_min_steps = min(original_lengths)
    if samples_per_trajectory is None:
        n_samples = available_min_steps
    else:
        n_samples = min(int(samples_per_trajectory), available_min_steps)

    strains = [sample_equally_spaced(array, n_samples) for array in raw_strains]
    stresses = [sample_equally_spaced(array, n_samples) for array in raw_stresses]

    return {
        "trajectory_ids": list(trajectory_ids),
        "strain": np.stack(strains, axis=0).astype(np.float32),
        "stress": np.stack(stresses, axis=0).astype(np.float32),
        "original_lengths": original_lengths,
        "samples_per_trajectory": n_samples,
        "available_min_steps": available_min_steps,
        "strain_source": strain_source,
    }


def load_external_history(strain_path, stress_path, label="external", samples_per_trajectory=None):
    strain = np.load(strain_path)
    stress = np.load(stress_path)

    if strain.shape != stress.shape:
        raise ValueError(
            f"Shape mismatch for external history: strain={strain.shape}, stress={stress.shape}"
        )
    if strain.ndim != 2 or strain.shape[1] != 3:
        raise ValueError(
            f"Expected external arrays with shape [steps, 3]. Got {strain.shape}."
        )

    original_length = int(strain.shape[0])
    if samples_per_trajectory is not None:
        n_samples = min(int(samples_per_trajectory), original_length)
        strain = sample_equally_spaced(strain, n_samples)
        stress = sample_equally_spaced(stress, n_samples)
    else:
        n_samples = original_length

    return {
        "trajectory_ids": [str(label)],
        "strain": strain[None, :, :].astype(np.float32),
        "stress": stress[None, :, :].astype(np.float32),
        "original_lengths": [original_length],
        "samples_per_trajectory": n_samples,
        "available_min_steps": original_length,
        "strain_source": "external_npy",
    }


def compute_normalization(strain, stress):
    strain_scale = float(np.max(np.abs(strain)))
    stress_scale = float(np.max(np.abs(stress)))
    if strain_scale <= 0.0:
        raise ValueError("Cannot normalize strain: max(abs(strain)) is zero.")
    if stress_scale <= 0.0:
        raise ValueError("Cannot normalize stress: max(abs(stress)) is zero.")
    return {
        "strain_scale": strain_scale,
        "stress_scale": stress_scale,
    }


def apply_normalization(strain, stress, normalization):
    return (
        strain / float(normalization["strain_scale"]),
        stress / float(normalization["stress_scale"]),
    )


def compute_reference_energy(strain_normalized, stress_normalized):
    delta_strain = np.zeros_like(strain_normalized)
    delta_strain[:, 1:, :] = strain_normalized[:, 1:, :] - strain_normalized[:, :-1, :]

    stress_average = 0.5 * (stress_normalized[:, 1:, :] + stress_normalized[:, :-1, :])
    energy_increment = np.sum(
        stress_average * delta_strain[:, 1:, :],
        axis=2,
        keepdims=True,
    )
    energy_cumulative = np.cumsum(energy_increment, axis=1)
    energy_zero = np.zeros(
        (strain_normalized.shape[0], 1, 1),
        dtype=strain_normalized.dtype,
    )
    return np.concatenate([energy_zero, energy_cumulative], axis=1)


def flatten_history(array):
    return array.reshape(-1, array.shape[-1])


def create_model(model_config=None):
    import ICKAN_surrogate as surrogate

    config = dict(DEFAULT_MODEL_CONFIG)
    if model_config is not None:
        config.update(model_config)
        # Backward compatibility with old checkpoints saved before input_mode existed.
        if "input_mode" not in model_config:
            config["input_mode"] = "principal"
        if "model_type" not in model_config:
            config["model_type"] = "ickan"

    model_type = config.get("model_type", "ickan")
    if model_type == "icnn":
        model_kwargs = {
            "order_stretches": config["order_stretches"],
            "W_width": config["W_width"],
            "input_mode": config["input_mode"],
            "icnn_activation": config.get("icnn_activation", "softplus"),
            "icnn_softplus_beta": config.get("icnn_softplus_beta", 5.0),
            "icnn_quadratic": config.get("icnn_quadratic", True),
        }
        model = surrogate.ICNN_W_Surrogate(**model_kwargs)
        model.kan_backend = "icnn"
    elif model_type == "ickan":
        model_kwargs = {
            "order_stretches": config["order_stretches"],
            "grid_size": config["grid_size"],
            "k": config["k"],
            "W_width": config["W_width"],
            "input_mode": config["input_mode"],
            "base_fun": config.get("base_fun", "silu"),
            "noise_scale": config.get("noise_scale", 0.0),
            "grid_eps": config.get("grid_eps", 0.01),
        }
        model = surrogate.ICKAN_W_Surrogate(**model_kwargs)
        model.kan_backend = getattr(surrogate, "KAN_BACKEND", "unknown")
    else:
        raise ValueError(f"Unsupported model_type '{model_type}'.")
    return model, config


def l2_relative_error(prediction, target):
    return torch.mean((prediction - target) ** 2) / (torch.mean(target**2) + 1.0e-12)


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def load_checkpoint(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def predict_normalized(model, strain_normalized, max_w):
    strain_flat = torch.tensor(
        flatten_history(strain_normalized),
        dtype=torch.float32,
    )
    predicted_w = model.CalculateCorrectedW(strain_flat).detach().cpu().numpy()
    predicted_stress = (
        float(max_w) * model.CalculateNormalizedStress(strain_flat)
    ).detach().cpu().numpy()

    n_trajectories, n_steps, _ = strain_normalized.shape
    return {
        "W_normalized": predicted_w.reshape(n_trajectories, n_steps, 1),
        "stress_normalized": predicted_stress.reshape(n_trajectories, n_steps, 3),
    }


def compute_metrics(reference, prediction):
    error = prediction - reference
    mse_components = np.mean(error**2, axis=(0, 1))
    reference_mse_components = np.mean(reference**2, axis=(0, 1))
    relative_mse_components = mse_components / (reference_mse_components + 1.0e-12)

    mse_global = float(np.mean(error**2))
    reference_mse_global = float(np.mean(reference**2))
    relative_mse_global = mse_global / (reference_mse_global + 1.0e-12)
    relative_l2_components = np.sqrt(relative_mse_components)

    return {
        "relative_mse_global": float(relative_mse_global),
        "relative_l2_global": float(np.sqrt(relative_mse_global)),
        "rmse_global": float(np.sqrt(mse_global)),
        "relative_mse_by_component": {
            name: float(value)
            for name, value in zip(COMPONENT_NAMES, relative_mse_components)
        },
        "relative_l2_by_component": {
            name: float(value)
            for name, value in zip(COMPONENT_NAMES, relative_l2_components)
        },
        "rmse_by_component": {
            name: float(np.sqrt(value))
            for name, value in zip(COMPONENT_NAMES, mse_components)
        },
    }


def plot_loss_history(loss_history, out_dir):
    if not loss_history:
        return

    import matplotlib.pyplot as plt

    ensure_dir(out_dir)
    epochs = [item.get("global_epoch", item["epoch"]) for item in loss_history]
    losses = [item["loss"] for item in loss_history]
    best_losses = [item["best_loss"] for item in loss_history]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(epochs, losses, label="current loss")
    ax.semilogy(epochs, best_losses, label="best loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Relative MSE loss")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "loss_history.png"), dpi=180)
    fig.savefig(os.path.join(out_dir, "loss_history.eps"))
    plt.close(fig)


def plot_prediction_results(
    out_dir,
    trajectory_ids,
    strain_normalized,
    stress_reference_normalized,
    stress_predicted_normalized,
    energy_reference_normalized,
    energy_predicted_normalized,
    strain_axis_variants=None,
    prediction_label="ICKAN",
):
    import matplotlib.pyplot as plt

    ensure_dir(out_dir)
    trajectory_dir = ensure_dir(os.path.join(out_dir, "trajectories"))

    steps = np.arange(strain_normalized.shape[1])
    for local_index, trajectory_id in enumerate(trajectory_ids):
        fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
        for component, name in enumerate(COMPONENT_NAMES):
            axes[component].plot(
                steps,
                stress_reference_normalized[local_index, :, component],
                "--",
                label=f"Reference S_{name}",
            )
            axes[component].plot(
                steps,
                stress_predicted_normalized[local_index, :, component],
                "-",
                label=f"{prediction_label} S_{name}",
            )
            axes[component].set_ylabel(f"S_{name}")
            axes[component].grid(True)
            axes[component].legend()
        axes[-1].set_xlabel("Trajectory sample index")
        fig.suptitle(f"trajectory_{trajectory_id}: stress over path")
        fig.tight_layout()
        fig.savefig(
            os.path.join(trajectory_dir, f"trajectory_{trajectory_id}_stress_vs_step.png"),
            dpi=180,
        )
        plt.close(fig)

        fig, axes = plt.subplots(3, 1, figsize=(8, 8))
        for component, name in enumerate(COMPONENT_NAMES):
            axes[component].plot(
                strain_normalized[local_index, :, component],
                stress_reference_normalized[local_index, :, component],
                "--",
                label=f"Reference S_{name}",
            )
            axes[component].plot(
                strain_normalized[local_index, :, component],
                stress_predicted_normalized[local_index, :, component],
                "-",
                label=f"{prediction_label} S_{name}",
            )
            axes[component].set_xlabel(f"Normalized strain {name}")
            axes[component].set_ylabel(f"Normalized stress {name}")
            axes[component].grid(True)
            axes[component].legend()
        fig.suptitle(f"trajectory_{trajectory_id}: stress-strain path")
        fig.tight_layout()
        fig.savefig(
            os.path.join(trajectory_dir, f"trajectory_{trajectory_id}_stress_vs_strain.png"),
            dpi=180,
        )
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(
            steps,
            energy_reference_normalized[local_index, :, 0],
            "--",
            label="Reference W",
        )
        ax.plot(
            steps,
            energy_predicted_normalized[local_index, :, 0],
            "-",
            label=f"{prediction_label} W",
        )
        ax.set_xlabel("Trajectory sample index")
        ax.set_ylabel("Normalized energy")
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        fig.savefig(
            os.path.join(trajectory_dir, f"trajectory_{trajectory_id}_energy_vs_step.png"),
            dpi=180,
        )
        plt.close(fig)

    for component, name in enumerate(COMPONENT_NAMES):
        fig, ax = plt.subplots(figsize=(8, 5))
        for local_index, _ in enumerate(trajectory_ids):
            label_ref = "Reference" if local_index == 0 else None
            label_pred = prediction_label if local_index == 0 else None
            ax.plot(
                strain_normalized[local_index, :, component],
                stress_reference_normalized[local_index, :, component],
                "--",
                label=label_ref,
                alpha=0.85,
            )
            ax.plot(
                strain_normalized[local_index, :, component],
                stress_predicted_normalized[local_index, :, component],
                "-",
                label=label_pred,
                alpha=0.85,
            )
        ax.set_xlabel(f"Normalized strain {name}")
        ax.set_ylabel(f"Normalized stress {name}")
        ax.set_title(f"S_{name} over each trajectory")
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"S_{name}_by_trajectory.png"), dpi=180)
        fig.savefig(os.path.join(out_dir, f"S_{name}_by_trajectory.eps"))
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for local_index, _ in enumerate(trajectory_ids):
        label_ref = "Reference" if local_index == 0 else None
        label_pred = prediction_label if local_index == 0 else None
        ax.plot(
            steps,
            energy_reference_normalized[local_index, :, 0],
            "--",
            label=label_ref,
            alpha=0.85,
        )
        ax.plot(
            steps,
            energy_predicted_normalized[local_index, :, 0],
            "-",
            label=label_pred,
            alpha=0.85,
        )
    ax.set_xlabel("Trajectory sample index")
    ax.set_ylabel("Normalized energy")
    ax.set_title("Energy over each trajectory")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "W_by_trajectory.png"), dpi=180)
    fig.savefig(os.path.join(out_dir, "W_by_trajectory.eps"))
    plt.close(fig)

    if strain_axis_variants is None:
        return

    for axis_name, axis_label, axis_strain_normalized in strain_axis_variants:
        safe_axis_name = (
            str(axis_name)
            .replace(" ", "_")
            .replace("/", "_")
            .replace(":", "_")
        )

        for local_index, trajectory_id in enumerate(trajectory_ids):
            fig, axes = plt.subplots(3, 1, figsize=(8, 8))
            for component, name in enumerate(COMPONENT_NAMES):
                axes[component].plot(
                    axis_strain_normalized[local_index, :, component],
                    stress_reference_normalized[local_index, :, component],
                    "--",
                    label=f"Reference S_{name}",
                )
                axes[component].plot(
                    axis_strain_normalized[local_index, :, component],
                    stress_predicted_normalized[local_index, :, component],
                    "-",
                    label=f"{prediction_label} S_{name}",
                )
                axes[component].set_xlabel(f"Normalized {axis_label} {name}")
                axes[component].set_ylabel(f"Normalized stress {name}")
                axes[component].grid(True)
                axes[component].legend()
            fig.suptitle(f"trajectory_{trajectory_id}: stress path vs {axis_label}")
            fig.tight_layout()
            fig.savefig(
                os.path.join(
                    trajectory_dir,
                    f"trajectory_{trajectory_id}_stress_vs_{safe_axis_name}_axis.png",
                ),
                dpi=180,
            )
            plt.close(fig)

        for component, name in enumerate(COMPONENT_NAMES):
            fig, ax = plt.subplots(figsize=(8, 5))
            for local_index, _ in enumerate(trajectory_ids):
                label_ref = "Reference" if local_index == 0 else None
                label_pred = prediction_label if local_index == 0 else None
                ax.plot(
                    axis_strain_normalized[local_index, :, component],
                    stress_reference_normalized[local_index, :, component],
                    "--",
                    label=label_ref,
                    alpha=0.85,
                )
                ax.plot(
                    axis_strain_normalized[local_index, :, component],
                    stress_predicted_normalized[local_index, :, component],
                    "-",
                    label=label_pred,
                    alpha=0.85,
                )
            ax.set_xlabel(f"Normalized {axis_label} {name}")
            ax.set_ylabel(f"Normalized stress {name}")
            ax.set_title(f"S_{name} over each trajectory vs {axis_label}")
            ax.grid(True)
            ax.legend()
            fig.tight_layout()
            fig.savefig(
                os.path.join(out_dir, f"S_{name}_by_trajectory_vs_{safe_axis_name}.png"),
                dpi=180,
            )
            fig.savefig(
                os.path.join(out_dir, f"S_{name}_by_trajectory_vs_{safe_axis_name}.eps")
            )
            plt.close(fig)
