import argparse
import os
import random

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ["MPLCONFIGDIR"] = os.path.join(_SCRIPT_DIR, ".mplconfig")
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ickan_workflow import (
    COMPONENT_NAMES,
    apply_normalization,
    compute_metrics,
    compute_normalization,
    ensure_dir,
    flatten_history,
    load_fom_dataset,
    parse_trajectory_ids,
    save_json,
)


REPO_ROOT = os.path.dirname(_SCRIPT_DIR)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Direct stress MLP baseline: learn normalized strain -> normalized stress."
    )
    parser.add_argument(
        "--fom-dir",
        default=os.path.join(REPO_ROOT, "stage_1_training_set_fom"),
    )
    parser.add_argument(
        "--out-dir",
        default=os.path.join(_SCRIPT_DIR, "MLP_stress_baseline"),
    )
    parser.add_argument("--train-trajectories", default="3")
    parser.add_argument(
        "--strain-source",
        default="strain",
        choices=("strain", "applied_strain"),
    )
    parser.add_argument("--samples-per-trajectory", type=int, default=300)
    parser.add_argument("--hidden-widths", default="64,64,64")
    parser.add_argument("--activation", default="silu", choices=("silu", "tanh", "relu"))
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--verbose-interval", type=int, default=250)
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="Use CUDA only if available. Current machine may still be CPU-only.",
    )
    return parser.parse_args()


def parse_hidden_widths(text):
    widths = []
    for chunk in str(text).replace(" ", "").split(","):
        if chunk:
            value = int(chunk)
            if value <= 0:
                raise ValueError(f"Hidden widths must be positive. Got {value}.")
            widths.append(value)
    if not widths:
        raise ValueError("At least one hidden width is required.")
    return widths


def make_activation(name):
    if name == "silu":
        return nn.SiLU()
    if name == "tanh":
        return nn.Tanh()
    return nn.ReLU()


class DirectStressMLP(nn.Module):
    def __init__(self, hidden_widths, activation):
        super().__init__()
        layers = []
        in_dim = 3
        for width in hidden_widths:
            layers.append(nn.Linear(in_dim, width))
            layers.append(make_activation(activation))
            in_dim = width
        layers.append(nn.Linear(in_dim, 3))
        self.net = nn.Sequential(*layers)

    def forward(self, strain):
        return self.net(strain)


def relative_component_mse(prediction, target):
    numerator = torch.mean((prediction - target) ** 2, dim=0)
    denominator = torch.mean(target**2, dim=0) + 1.0e-12
    return torch.mean(numerator / denominator)


def plot_loss_history(loss_history, out_dir):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(loss_history[:, 0], loss_history[:, 1], label="loss")
    ax.semilogy(loss_history[:, 0], loss_history[:, 2], label="best loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Relative component MSE")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "loss_history.png"), dpi=180)
    plt.close(fig)


def plot_stress_fit(
    out_dir,
    trajectory_ids,
    strain_normalized,
    stress_reference_normalized,
    stress_predicted_normalized,
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
                label=f"MLP S_{name}",
            )
            axes[component].set_ylabel(f"S_{name}")
            axes[component].grid(True)
            axes[component].legend()
        axes[-1].set_xlabel("Trajectory sample index")
        fig.suptitle(f"trajectory_{trajectory_id}: direct MLP stress over path")
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
                label=f"MLP S_{name}",
            )
            axes[component].set_xlabel(f"Normalized strain {name}")
            axes[component].set_ylabel(f"Normalized stress {name}")
            axes[component].grid(True)
            axes[component].legend()
        fig.suptitle(f"trajectory_{trajectory_id}: direct MLP stress-strain path")
        fig.tight_layout()
        fig.savefig(
            os.path.join(trajectory_dir, f"trajectory_{trajectory_id}_stress_vs_strain.png"),
            dpi=180,
        )
        plt.close(fig)

    for component, name in enumerate(COMPONENT_NAMES):
        fig, ax = plt.subplots(figsize=(8, 5))
        for local_index, _ in enumerate(trajectory_ids):
            label_ref = "Reference" if local_index == 0 else None
            label_pred = "MLP" if local_index == 0 else None
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
        plt.close(fig)


def main():
    args = parse_args()
    out_dir = ensure_dir(os.path.abspath(args.out_dir))
    fom_dir = os.path.abspath(args.fom_dir)
    trajectory_ids = parse_trajectory_ids(args.train_trajectories)
    hidden_widths = parse_hidden_widths(args.hidden_widths)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")

    print("[SEBASTIAN-MLP-STRESS][TRAIN]")
    print(f"FOM input dir          : {fom_dir}")
    print(f"Output dir             : {out_dir}")
    print(f"Train trajectories     : {trajectory_ids}")
    print(f"Strain source          : {args.strain_source}")
    print(f"Samples per trajectory : {args.samples_per_trajectory}")
    print(f"Hidden widths          : {hidden_widths}")
    print(f"Activation             : {args.activation}")
    print(f"Epochs                 : {args.epochs}")
    print(f"Learning rate          : {args.learning_rate:.8E}")
    print(f"Weight decay           : {args.weight_decay:.8E}")
    print(f"Device                 : {device}")

    dataset = load_fom_dataset(
        fom_dir=fom_dir,
        trajectory_ids=trajectory_ids,
        strain_source=args.strain_source,
        samples_per_trajectory=args.samples_per_trajectory,
    )
    normalization = compute_normalization(dataset["strain"], dataset["stress"])
    strain_normalized, stress_normalized = apply_normalization(
        dataset["strain"],
        dataset["stress"],
        normalization,
    )

    train_strain = torch.tensor(
        flatten_history(strain_normalized),
        dtype=torch.float32,
        device=device,
    )
    train_stress = torch.tensor(
        flatten_history(stress_normalized),
        dtype=torch.float32,
        device=device,
    )

    model = DirectStressMLP(hidden_widths, args.activation).to(device)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    best_loss = float("inf")
    best_state = None
    loss_history = []
    for epoch in range(args.epochs):
        optimizer.zero_grad()
        prediction = model(train_strain)
        loss = relative_component_mse(prediction, train_stress)
        loss.backward()
        optimizer.step()

        loss_value = float(loss.detach().cpu().item())
        if loss_value < best_loss:
            best_loss = loss_value
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        loss_history.append([epoch, loss_value, best_loss])

        if epoch % args.verbose_interval == 0 or epoch == args.epochs - 1:
            print(
                f"Epoch {epoch:05d}, "
                f"Loss: {loss_value:.8E}, "
                f"Best Loss: {best_loss:.8E}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        predicted_stress = model(train_strain).detach().cpu().numpy()

    n_trajectories, n_steps, _ = strain_normalized.shape
    stress_predicted_normalized = predicted_stress.reshape(n_trajectories, n_steps, 3)
    metrics_normalized = compute_metrics(stress_normalized, stress_predicted_normalized)
    stress_predicted_physical = stress_predicted_normalized * normalization["stress_scale"]
    metrics_physical = compute_metrics(dataset["stress"], stress_predicted_physical)

    loss_history_array = np.array(loss_history, dtype=np.float64)
    np.savetxt(
        os.path.join(out_dir, "loss_history.csv"),
        loss_history_array,
        delimiter=",",
        header="epoch,loss,best_loss",
        comments="",
    )
    plot_loss_history(loss_history_array, out_dir)

    np.savez(
        os.path.join(out_dir, "predictions.npz"),
        trajectory_ids=np.array(trajectory_ids),
        strain=dataset["strain"],
        stress_reference=dataset["stress"],
        stress_predicted=stress_predicted_physical,
        strain_normalized=strain_normalized,
        stress_reference_normalized=stress_normalized,
        stress_predicted_normalized=stress_predicted_normalized,
    )
    plot_stress_fit(
        out_dir,
        trajectory_ids,
        strain_normalized,
        stress_normalized,
        stress_predicted_normalized,
    )

    checkpoint_path = os.path.join(out_dir, "direct_stress_mlp_checkpoint.pth")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "hidden_widths": hidden_widths,
            "activation": args.activation,
            "normalization": normalization,
            "training": {
                "fom_dir": fom_dir,
                "strain_source": args.strain_source,
                "train_trajectories": trajectory_ids,
                "samples_per_trajectory": dataset["samples_per_trajectory"],
                "original_lengths": dataset["original_lengths"],
                "epochs_requested": args.epochs,
                "epochs_completed": args.epochs,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "best_loss": best_loss,
                "device": str(device),
            },
        },
        checkpoint_path,
    )

    metadata = {
        "checkpoint": checkpoint_path,
        "model": {
            "type": "DirectStressMLP",
            "hidden_widths": hidden_widths,
            "activation": args.activation,
            "input": "normalized [E_xx, E_yy, G_xy]",
            "output": "normalized [S_xx, S_yy, S_xy]",
            "energy_potential": False,
        },
        "normalization": normalization,
        "training": {
            "fom_dir": fom_dir,
            "strain_source": args.strain_source,
            "train_trajectories": trajectory_ids,
            "samples_per_trajectory": dataset["samples_per_trajectory"],
            "original_lengths": dataset["original_lengths"],
            "epochs_requested": args.epochs,
            "epochs_completed": args.epochs,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "best_loss": best_loss,
            "device": str(device),
        },
        "metrics_normalized": metrics_normalized,
        "metrics_physical": metrics_physical,
    }
    save_json(os.path.join(out_dir, "training_and_fit_metrics.json"), metadata)

    print("\nDirect stress MLP finished.")
    print(f"Best training loss     : {best_loss:.8E}")
    print(f"Normalized relative L2 : {metrics_normalized['relative_l2_global']:.8E}")
    print(f"Physical relative L2   : {metrics_physical['relative_l2_global']:.8E}")
    print(f"Outputs saved to       : {out_dir}")


if __name__ == "__main__":
    main()
