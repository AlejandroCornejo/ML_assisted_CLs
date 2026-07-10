import argparse
import os

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ["MPLCONFIGDIR"] = os.path.join(_SCRIPT_DIR, ".mplconfig")
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import torch
import torch.optim as optim

from ickan_workflow import (
    DEFAULT_MODEL_CONFIG,
    apply_normalization,
    compute_normalization,
    compute_reference_energy,
    create_model,
    ensure_dir,
    flatten_history,
    l2_relative_error,
    load_fom_dataset,
    parse_trajectory_ids,
    plot_loss_history,
    save_json,
)


REPO_ROOT = os.path.dirname(_SCRIPT_DIR)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Sebastian's ICKAN/KAN energy surrogate and save a reusable checkpoint."
    )
    parser.add_argument(
        "--fom-dir",
        default=os.path.join(REPO_ROOT, "stage_1_training_set_fom"),
        help="Directory containing trajectory_i FOM folders.",
    )
    parser.add_argument(
        "--out-dir",
        default=os.path.join(_SCRIPT_DIR, "ICKAN_training"),
        help="Directory where the checkpoint and training diagnostics are written.",
    )
    parser.add_argument(
        "--resume-checkpoint",
        default=None,
        help=(
            "Optional Sebastian checkpoint used to initialize the model and reuse "
            "its model configuration and normalization before continuing training."
        ),
    )
    parser.add_argument(
        "--train-trajectories",
        default="1-8",
        help="Trajectory ids used for training. Examples: '1-8', '1,3,5', or 'all'.",
    )
    parser.add_argument(
        "--strain-source",
        default="strain",
        choices=("strain", "applied_strain"),
        help="Use homogenized strain or imposed applied strain as model input.",
    )
    parser.add_argument(
        "--model-type",
        default="ickan",
        choices=("ickan", "icnn"),
        help="Energy surrogate family. 'icnn' uses a smooth input-convex neural network.",
    )
    parser.add_argument(
        "--samples-per-trajectory",
        type=int,
        default=500,
        help="Equally spaced samples taken from each selected trajectory.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument(
        "--optimizer",
        default="lbfgs",
        choices=("adam", "lbfgs"),
        help="Optimizer. LBFGS is the default; Adam is available but usually needs very conservative tuning here.",
    )
    parser.add_argument(
        "--lbfgs-max-iter",
        type=int,
        default=20,
        help="Maximum inner LBFGS iterations per reported epoch.",
    )
    parser.add_argument(
        "--lbfgs-history-size",
        type=int,
        default=100,
        help="LBFGS curvature history size.",
    )
    parser.add_argument(
        "--adam-warmup-epochs",
        type=int,
        default=0,
        help="Optional Adam warmup epochs before the selected optimizer. Useful before LBFGS plateaus.",
    )
    parser.add_argument(
        "--adam-warmup-lr",
        type=float,
        default=1.0e-3,
        help="Initial Adam learning rate used during warmup.",
    )
    parser.add_argument(
        "--adam-warmup-max-lr",
        type=float,
        default=1.0e-2,
        help="Maximum learning rate for cyclic Adam warmup.",
    )
    parser.add_argument(
        "--adam-warmup-step-size",
        type=int,
        default=50,
        help="Half-cycle length for cyclic Adam warmup.",
    )
    parser.add_argument(
        "--adam-warmup-scheduler",
        default="cyclic",
        choices=("none", "cyclic"),
        help="Learning-rate scheduler used during Adam warmup.",
    )
    parser.add_argument(
        "--input-mode",
        default="principal",
        choices=(
            "direct_strain",
            "principal",
            "hybrid",
            "orthotropic_invariants",
            "orthotropic_invariants_signed",
            "ickan_invariants",
            "ickan_invariants_linear",
        ),
        help=(
            "'direct_strain' learns W(E_xx,E_yy,G_xy). "
            "'principal' uses the original principal-stretch/logJ features. "
            "'hybrid' appends [E_xx,E_yy,G_xy] to the principal features for diagnostics. "
            "'orthotropic_invariants' learns W(C_xx-1,C_yy-1,C_xy^2,logJ). "
            "'orthotropic_invariants_signed' learns W(C_xx-1,C_yy-1,C_xy,C_xy^2,logJ). "
            "'ickan_invariants' uses the transformed invariant inputs from the upstream ICKAN driver. "
            "'ickan_invariants_linear' uses the upstream core/features.py variant."
        ),
    )
    parser.add_argument(
        "--order-stretches",
        type=int,
        default=3,
        help=(
            "Number of principal-stretch powers used in principal mode. "
            "order=3 gives [lambda1^1,lambda2^1,lambda1^2,lambda2^2,lambda1^3,lambda2^3,logJ]."
        ),
    )
    parser.add_argument(
        "--freeze-feature-powers",
        action="store_true",
        help=(
            "Keep the principal-stretch powers fixed instead of training them. "
            "This is a more conservative diagnostic when using principal/hybrid inputs."
        ),
    )
    parser.add_argument(
        "--hidden-widths",
        default="16,16,8",
        help="Comma-separated hidden layer widths for the KAN energy network.",
    )
    parser.add_argument("--grid-size", type=int, default=100)
    parser.add_argument("--spline-degree", type=int, default=3)
    parser.add_argument(
        "--kan-base-fun",
        default="silu",
        choices=("silu", "zero", "identity", "softplus"),
        help=(
            "Base branch used inside each KAN spline layer. "
            "The original ICKAN hyperelasticity code uses 'zero' for a purer convex spline model."
        ),
    )
    parser.add_argument(
        "--kan-noise-scale",
        type=float,
        default=0.0,
        help="Initial spline noise scale. A small positive value can avoid degenerate flat starts with --kan-base-fun zero.",
    )
    parser.add_argument(
        "--kan-grid-eps",
        type=float,
        default=0.01,
        help="KAN grid interpolation: 1 is uniform, 0 is fully sample-adaptive.",
    )
    parser.add_argument(
        "--icnn-activation",
        default="softplus",
        choices=("softplus", "relu"),
        help="Convex non-decreasing activation used by --model-type icnn.",
    )
    parser.add_argument(
        "--icnn-softplus-beta",
        type=float,
        default=5.0,
        help="Softplus beta used by --model-type icnn when --icnn-activation softplus.",
    )
    parser.add_argument(
        "--no-icnn-quadratic",
        action="store_true",
        help="Disable the non-negative diagonal quadratic term in the ICNN energy.",
    )
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=1.0,
        help="Gradient clipping norm for Adam. Ignored by LBFGS.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="Use CUDA if available. 'auto' selects CUDA when torch can see it.",
    )
    parser.add_argument(
        "--stress-loss",
        default="component",
        choices=("global", "component", "blended"),
        help=(
            "'global' is one relative MSE over all stress components. "
            "'component' averages per-component relative MSE so S_xy is not ignored. "
            "'blended' combines both losses."
        ),
    )
    parser.add_argument(
        "--component-loss-weight",
        type=float,
        default=0.25,
        help=(
            "Weight of the component-balanced term when --stress-loss blended is used. "
            "0 gives global loss, 1 gives pure component loss."
        ),
    )
    parser.add_argument(
        "--energy-loss-weight",
        type=float,
        default=0.0,
        help="Weight for an auxiliary normalized W loss. 0 means stress-only training.",
    )
    parser.add_argument("--early-stopping-threshold", type=float, default=1.0e-3)
    parser.add_argument(
        "--update-grid-during-training",
        action="store_true",
        help="Move the KAN spline grid during optimization. Disabled by default because it can destabilize LBFGS.",
    )
    parser.add_argument("--grid-update-interval", type=int, default=10)
    parser.add_argument("--initial-grid-update-step", type=int, default=10)
    parser.add_argument("--final-grid-update-step", type=int, default=50)
    parser.add_argument("--verbose-interval", type=int, default=1)
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


def make_model_config(args):
    order_stretches = int(args.order_stretches)
    if args.input_mode == "direct_strain":
        input_size = 3
    elif args.input_mode in ("ickan_invariants", "ickan_invariants_linear"):
        input_size = 3
    elif args.input_mode == "orthotropic_invariants":
        input_size = 4
    elif args.input_mode == "orthotropic_invariants_signed":
        input_size = 5
    else:
        if order_stretches <= 0:
            raise ValueError("--order-stretches must be positive in principal/hybrid mode.")
        principal_input_size = 2 * order_stretches + 1
        if args.input_mode == "hybrid":
            input_size = principal_input_size + 3
        else:
            input_size = principal_input_size

    hidden_widths = parse_hidden_widths(args.hidden_widths)
    return {
        "order_stretches": order_stretches,
        "grid_size": int(args.grid_size),
        "k": int(args.spline_degree),
        "W_width": [input_size] + hidden_widths + [1],
        "input_mode": args.input_mode,
        "base_fun": args.kan_base_fun,
        "noise_scale": float(args.kan_noise_scale),
        "grid_eps": float(args.kan_grid_eps),
        "model_type": args.model_type,
        "icnn_activation": args.icnn_activation,
        "icnn_softplus_beta": float(args.icnn_softplus_beta),
        "icnn_quadratic": not bool(args.no_icnn_quadratic),
        "train_feature_powers": not bool(args.freeze_feature_powers),
    }


def relative_component_mse(prediction, target):
    numerator = torch.mean((prediction - target) ** 2, dim=0)
    denominator = torch.mean(target**2, dim=0) + 1.0e-12
    return torch.mean(numerator / denominator)


def compute_stress_loss(
    model,
    strain_tensor,
    stress_tensor,
    max_w,
    stress_loss_mode,
    component_loss_weight,
):
    predicted_stress = model.CalculateNormalizedStress(strain_tensor) * float(max_w)
    global_loss = l2_relative_error(predicted_stress, stress_tensor)
    component_loss = relative_component_mse(predicted_stress, stress_tensor)
    if stress_loss_mode == "component":
        return component_loss
    if stress_loss_mode == "blended":
        alpha = float(component_loss_weight)
        return (1.0 - alpha) * global_loss + alpha * component_loss
    return global_loss


def compute_total_loss(
    model,
    strain_tensor,
    stress_tensor,
    energy_tensor,
    max_w,
    stress_loss_mode,
    component_loss_weight,
    energy_loss_weight,
):
    stress_loss = compute_stress_loss(
        model,
        strain_tensor,
        stress_tensor,
        max_w,
        stress_loss_mode,
        component_loss_weight,
    )
    if energy_loss_weight <= 0.0:
        return stress_loss

    predicted_energy = model.CalculateCorrectedW(strain_tensor)
    energy_loss = l2_relative_error(predicted_energy, energy_tensor)
    return stress_loss + float(energy_loss_weight) * energy_loss


def train_model(
    model,
    optimizer,
    optimizer_name,
    scheduler,
    strain_tensor,
    stress_tensor,
    energy_tensor,
    max_w,
    n_epochs,
    early_stopping_threshold,
    update_grid_during_training,
    grid_update_interval,
    initial_grid_update_step,
    final_grid_update_step,
    verbose_interval,
    grad_clip,
    stress_loss_mode,
    component_loss_weight,
    energy_loss_weight,
    epoch_offset=0,
    phase="train",
):
    best_loss = float("inf")
    best_state = None
    grid_update_counter = 0
    loss_history = []

    for epoch in range(n_epochs):
        if optimizer_name == "lbfgs":
            def closure():
                optimizer.zero_grad()
                loss_inner = compute_total_loss(
                    model,
                    strain_tensor,
                    stress_tensor,
                    energy_tensor,
                    max_w,
                    stress_loss_mode,
                    component_loss_weight,
                    energy_loss_weight,
                )
                loss_inner.backward()
                return loss_inner

            optimizer.step(closure)
        else:
            optimizer.zero_grad()
            loss = compute_total_loss(
                model,
                strain_tensor,
                stress_tensor,
                energy_tensor,
                max_w,
                stress_loss_mode,
                component_loss_weight,
                energy_loss_weight,
            )
            loss.backward()
            if grad_clip is not None and grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        # LBFGS and Adam both report a loss evaluated before the parameter
        # update. Re-evaluate after the step so the printed loss and the saved
        # best state describe the same model parameters.
        loss = compute_total_loss(
            model,
            strain_tensor,
            stress_tensor,
            energy_tensor,
            max_w,
            stress_loss_mode,
            component_loss_weight,
            energy_loss_weight,
        )
        loss_value = float(loss.detach().item())
        global_epoch = epoch_offset + epoch
        grid_update_counter += 1

        if not np.isfinite(loss_value):
            print(
                f"Non-finite loss detected at epoch {epoch}. "
                "Restoring the best state found so far and stopping."
            )
            if best_state is not None:
                model.load_state_dict(best_state)
            break

        if loss_value < best_loss:
            best_loss = loss_value
            best_state = {
                key: value.detach().clone()
                for key, value in model.state_dict().items()
            }

        loss_history.append(
            {
                "epoch": int(epoch),
                "global_epoch": int(global_epoch),
                "phase": phase,
                "loss": loss_value,
                "best_loss": float(best_loss),
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
        )

        if epoch % verbose_interval == 0:
            print(
                f"{phase} Epoch {epoch:04d} "
                f"(global {global_epoch:04d}), "
                f"Loss: {loss_value:.8E}, "
                f"Best Loss: {best_loss:.8E}, "
                f"LR: {optimizer.param_groups[0]['lr']:.3E}"
            )

        if loss_value < early_stopping_threshold:
            print(f"Early stopping at epoch {epoch} with loss {loss_value:.8E}")
            break

        if (
            update_grid_during_training
            and epoch >= initial_grid_update_step
            and epoch <= final_grid_update_step
            and grid_update_counter >= grid_update_interval
        ):
            model.UpdateGridFromSamples(strain_tensor)
            post_update_loss = compute_total_loss(
                model,
                strain_tensor,
                stress_tensor,
                energy_tensor,
                max_w,
                stress_loss_mode,
                component_loss_weight,
                energy_loss_weight,
            )
            post_update_loss_value = float(post_update_loss.detach().item())
            if best_state is not None and not np.isfinite(post_update_loss_value):
                model.load_state_dict(best_state)
                update_grid_during_training = False
                print(
                    f"\tGrid update rejected at epoch {epoch}: "
                    f"post-update loss {post_update_loss_value:.8E}. "
                    "Continuing from best state."
                )
            else:
                print(
                    f"\tGrid updated from samples at epoch {epoch}; "
                    f"post-update loss {post_update_loss_value:.8E}"
                )
            if optimizer_name == "lbfgs":
                optimizer.state.clear()
            grid_update_counter = 0

    if best_state is not None:
        model.load_state_dict(best_state)

    return loss_history, best_loss


def main():
    args = parse_args()
    out_dir = ensure_dir(os.path.abspath(args.out_dir))
    fom_dir = os.path.abspath(args.fom_dir)
    train_trajectory_ids = parse_trajectory_ids(args.train_trajectories)
    resume_checkpoint = None
    resume_checkpoint_path = None
    if args.resume_checkpoint:
        resume_checkpoint_path = os.path.abspath(args.resume_checkpoint)
        resume_checkpoint = torch.load(resume_checkpoint_path, map_location="cpu")

    print("[SEBASTIAN-ICKAN][TRAIN]")
    print(f"FOM input dir          : {fom_dir}")
    print(f"Output dir             : {out_dir}")
    if resume_checkpoint_path is not None:
        print(f"Resume checkpoint      : {resume_checkpoint_path}")
    print(f"Train trajectories     : {train_trajectory_ids}")
    print(f"Strain source          : {args.strain_source}")
    print(f"Model type             : {args.model_type}")
    print(f"Samples per trajectory : {args.samples_per_trajectory}")
    print(f"Update grid in training: {args.update_grid_during_training}")
    print(f"Input mode             : {args.input_mode}")
    print(f"Order stretches        : {args.order_stretches}")
    print(f"Train feature powers   : {not bool(args.freeze_feature_powers)}")
    print(f"Hidden widths          : {parse_hidden_widths(args.hidden_widths)}")
    print(f"Grid size              : {args.grid_size}")
    print(f"Spline degree          : {args.spline_degree}")
    print(f"KAN base function      : {args.kan_base_fun}")
    print(f"KAN noise scale        : {args.kan_noise_scale:.8E}")
    print(f"KAN grid eps           : {args.kan_grid_eps:.8E}")
    if args.model_type == "icnn":
        print(f"ICNN activation        : {args.icnn_activation}")
        print(f"ICNN softplus beta     : {args.icnn_softplus_beta:.8E}")
        print(f"ICNN quadratic term    : {not bool(args.no_icnn_quadratic)}")
    print(f"Optimizer              : {args.optimizer}")
    if args.optimizer == "lbfgs":
        print(f"LBFGS max iter/epoch   : {args.lbfgs_max_iter}")
        print(f"LBFGS history size     : {args.lbfgs_history_size}")
    print(f"Adam warmup epochs     : {args.adam_warmup_epochs}")
    print(f"Adam warmup LR/max LR  : {args.adam_warmup_lr:.8E} / {args.adam_warmup_max_lr:.8E}")
    print(f"Adam warmup scheduler  : {args.adam_warmup_scheduler}")
    print(f"Stress loss            : {args.stress_loss}")
    if args.stress_loss == "blended":
        print(f"Component loss weight  : {args.component_loss_weight:.8E}")
    print(f"Energy loss weight     : {args.energy_loss_weight:.8E}")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
    if args.update_grid_during_training and device.type == "cuda":
        raise ValueError(
            "--update-grid-during-training is disabled for CUDA runs in this script. "
            "The initial grid update is done on CPU before moving the model to CUDA."
        )
    if args.update_grid_during_training and args.model_type == "icnn":
        raise ValueError("--update-grid-during-training is only meaningful for --model-type ickan.")
    print(f"Device                 : {device}")

    if args.learning_rate is None:
        learning_rate = 1.0e-5 if args.optimizer == "adam" else 1.0e-3
    else:
        learning_rate = float(args.learning_rate)
    print(f"Learning rate          : {learning_rate:.8E}")

    dataset = load_fom_dataset(
        fom_dir=fom_dir,
        trajectory_ids=train_trajectory_ids,
        strain_source=args.strain_source,
        samples_per_trajectory=args.samples_per_trajectory,
    )

    if resume_checkpoint is not None:
        checkpoint_normalization = resume_checkpoint.get("normalization", {})
        if "strain_scale" not in checkpoint_normalization or "stress_scale" not in checkpoint_normalization:
            raise ValueError(
                "The resume checkpoint does not contain strain_scale and stress_scale normalization."
            )
        normalization = {
            "strain_scale": float(checkpoint_normalization["strain_scale"]),
            "stress_scale": float(checkpoint_normalization["stress_scale"]),
        }
    else:
        normalization = compute_normalization(dataset["strain"], dataset["stress"])
    strain_normalized, stress_normalized = apply_normalization(
        dataset["strain"],
        dataset["stress"],
        normalization,
    )

    energy_raw = compute_reference_energy(strain_normalized, stress_normalized)
    if resume_checkpoint is not None and "max_w" in resume_checkpoint.get("normalization", {}):
        max_w = float(resume_checkpoint["normalization"]["max_w"])
    else:
        max_w = float(np.max(np.abs(energy_raw)))
    if max_w <= 0.0:
        raise ValueError("Cannot normalize W: max(abs(W)) is zero.")
    energy_normalized = energy_raw / max_w

    train_strain_cpu = torch.tensor(
        flatten_history(strain_normalized),
        dtype=torch.float32,
    )
    train_stress_cpu = torch.tensor(
        flatten_history(stress_normalized),
        dtype=torch.float32,
    )
    train_energy_cpu = torch.tensor(
        flatten_history(energy_normalized),
        dtype=torch.float32,
    )

    print(f"Original trajectory lengths : {dataset['original_lengths']}")
    print(f"Used samples per trajectory : {dataset['samples_per_trajectory']}")
    print(f"Training samples            : {train_strain_cpu.shape[0]}")
    print(f"Strain scale                : {normalization['strain_scale']:.8E}")
    print(f"Stress scale                : {normalization['stress_scale']:.8E}")
    print(f"Energy normalization max_W  : {max_w:.8E}")

    if resume_checkpoint is not None:
        if "model_config" not in resume_checkpoint:
            raise ValueError("The resume checkpoint does not contain model_config.")
        model_config = dict(resume_checkpoint["model_config"])
    else:
        model_config = make_model_config(args)
    model, model_config = create_model(model_config)
    kan_backend = getattr(model, "kan_backend", "unknown")
    print(f"Model backend           : {kan_backend}")
    print(
        "Effective model config  : "
        f"type={model_config.get('model_type')}, "
        f"input={model_config.get('input_mode')}, "
        f"order={model_config.get('order_stretches')}, "
        f"train_feature_powers={model_config.get('train_feature_powers', True)}, "
        f"W_width={model_config.get('W_width')}"
    )
    if resume_checkpoint is not None:
        model.load_state_dict(resume_checkpoint["model_state_dict"])
        print("Loaded model weights from resume checkpoint.")
    elif getattr(model, "uses_grid", False):
        print("\nInitial KAN grid update from training samples on CPU:")
        model.UpdateGridFromSamples(train_strain_cpu)
    else:
        print("\nInitial KAN grid update skipped: ICNN does not use spline grids.")
    model.to(device)
    train_strain = train_strain_cpu.to(device)
    train_stress = train_stress_cpu.to(device)
    train_energy = train_energy_cpu.to(device)

    print("Check null W at null strain: ", model.CalculateCorrectedW(torch.zeros(1, 3, device=device)))
    print("Check null S at null strain: ", model.CalculateNormalizedStress(torch.zeros(1, 3, device=device)))

    print("\nStarting stress-based optimization...")
    loss_history = []
    best_loss = float("inf")
    best_state_global = None
    epoch_offset = 0
    if resume_checkpoint is not None:
        resume_initial_loss = compute_total_loss(
            model=model,
            strain_tensor=train_strain,
            stress_tensor=train_stress,
            energy_tensor=train_energy,
            max_w=max_w,
            stress_loss_mode=args.stress_loss,
            component_loss_weight=args.component_loss_weight,
            energy_loss_weight=args.energy_loss_weight,
        )
        best_loss = float(resume_initial_loss.detach().item())
        best_state_global = {
            key: value.detach().clone()
            for key, value in model.state_dict().items()
        }
        print(f"Initial resumed loss    : {best_loss:.8E}")

    if args.adam_warmup_epochs > 0:
        print("\nStarting Adam warmup...")
        warmup_optimizer = optim.Adam(
            model.parameters(),
            lr=float(args.adam_warmup_lr),
        )
        if args.adam_warmup_scheduler == "cyclic":
            warmup_scheduler = optim.lr_scheduler.CyclicLR(
                warmup_optimizer,
                base_lr=float(args.adam_warmup_lr),
                max_lr=float(args.adam_warmup_max_lr),
                step_size_up=int(args.adam_warmup_step_size),
                step_size_down=int(args.adam_warmup_step_size),
                cycle_momentum=False,
            )
        else:
            warmup_scheduler = None

        warmup_history, warmup_best_loss = train_model(
            model=model,
            optimizer=warmup_optimizer,
            optimizer_name="adam",
            scheduler=warmup_scheduler,
            strain_tensor=train_strain,
            stress_tensor=train_stress,
            energy_tensor=train_energy,
            max_w=max_w,
            n_epochs=args.adam_warmup_epochs,
            early_stopping_threshold=args.early_stopping_threshold,
            update_grid_during_training=False,
            grid_update_interval=args.grid_update_interval,
            initial_grid_update_step=args.initial_grid_update_step,
            final_grid_update_step=args.final_grid_update_step,
            verbose_interval=args.verbose_interval,
            grad_clip=args.grad_clip,
            stress_loss_mode=args.stress_loss,
            component_loss_weight=args.component_loss_weight,
            energy_loss_weight=args.energy_loss_weight,
            epoch_offset=epoch_offset,
            phase="adam",
        )
        loss_history.extend(warmup_history)
        if warmup_best_loss < best_loss:
            best_loss = warmup_best_loss
            best_state_global = {
                key: value.detach().clone()
                for key, value in model.state_dict().items()
            }
        epoch_offset += len(warmup_history)

    if args.optimizer == "lbfgs":
        optimizer = optim.LBFGS(
            model.parameters(),
            lr=learning_rate,
            line_search_fn="strong_wolfe",
            max_iter=args.lbfgs_max_iter,
            history_size=args.lbfgs_history_size,
        )
    else:
        optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
        )

    main_history, main_best_loss = train_model(
        model=model,
        optimizer=optimizer,
        optimizer_name=args.optimizer,
        scheduler=None,
        strain_tensor=train_strain,
        stress_tensor=train_stress,
        energy_tensor=train_energy,
        max_w=max_w,
        n_epochs=args.epochs,
        early_stopping_threshold=args.early_stopping_threshold,
        update_grid_during_training=args.update_grid_during_training,
        grid_update_interval=args.grid_update_interval,
        initial_grid_update_step=args.initial_grid_update_step,
        final_grid_update_step=args.final_grid_update_step,
        verbose_interval=args.verbose_interval,
        grad_clip=args.grad_clip,
        stress_loss_mode=args.stress_loss,
        component_loss_weight=args.component_loss_weight,
        energy_loss_weight=args.energy_loss_weight,
        epoch_offset=epoch_offset,
        phase=args.optimizer,
    )
    loss_history.extend(main_history)
    if main_best_loss < best_loss:
        best_loss = main_best_loss
        best_state_global = {
            key: value.detach().clone()
            for key, value in model.state_dict().items()
        }
    elif best_state_global is not None:
        model.load_state_dict(best_state_global)
    best_loss_for_metadata = float(best_loss) if np.isfinite(best_loss) else None

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_config": model_config,
        "kan_backend": kan_backend,
        "normalization": {
            **normalization,
            "max_w": max_w,
        },
        "training": {
            "fom_dir": fom_dir,
            "strain_source": args.strain_source,
            "train_trajectories": train_trajectory_ids,
            "samples_per_trajectory": dataset["samples_per_trajectory"],
            "original_lengths": dataset["original_lengths"],
            "epochs_requested": args.epochs,
            "epochs_completed": len(loss_history),
            "learning_rate": learning_rate,
            "optimizer": args.optimizer,
            "lbfgs_max_iter": args.lbfgs_max_iter if args.optimizer == "lbfgs" else None,
            "lbfgs_history_size": args.lbfgs_history_size if args.optimizer == "lbfgs" else None,
            "adam_warmup_epochs": args.adam_warmup_epochs,
            "adam_warmup_lr": args.adam_warmup_lr,
            "adam_warmup_max_lr": args.adam_warmup_max_lr,
            "adam_warmup_step_size": args.adam_warmup_step_size,
            "adam_warmup_scheduler": args.adam_warmup_scheduler,
            "device": str(device),
            "best_loss": best_loss_for_metadata,
            "loss_target": "normalized stress plus optional normalized energy",
            "stress_loss": args.stress_loss,
            "component_loss_weight": args.component_loss_weight,
            "energy_loss_weight": args.energy_loss_weight,
            "update_grid_during_training": args.update_grid_during_training,
            "grid_update_interval": args.grid_update_interval,
            "initial_grid_update_step": args.initial_grid_update_step,
            "final_grid_update_step": args.final_grid_update_step,
        },
        "loss_history": loss_history,
    }

    checkpoint_path = os.path.join(out_dir, "ICKAN_model_checkpoint.pth")
    torch.save(checkpoint, checkpoint_path)

    np.savez(
        os.path.join(out_dir, "training_reference_data_normalized.npz"),
        trajectory_ids=np.array(train_trajectory_ids, dtype=np.int64),
        strain_normalized=strain_normalized,
        stress_normalized=stress_normalized,
        W_normalized=energy_normalized,
    )

    if loss_history:
        np.savetxt(
            os.path.join(out_dir, "loss_history.csv"),
            np.array(
                [
                    [
                        item.get("global_epoch", item["epoch"]),
                        item["epoch"],
                        item.get("phase", "train"),
                        item["loss"],
                        item["best_loss"],
                        item.get("lr", np.nan),
                    ]
                    for item in loss_history
                ],
                dtype=object,
            ),
            delimiter=",",
            fmt="%s",
            header="global_epoch,phase_epoch,phase,loss,best_loss,lr",
            comments="",
        )
    plot_loss_history(loss_history, out_dir)

    metadata = {
        "checkpoint": checkpoint_path,
        "kan_backend": kan_backend,
        "model_config": model_config,
        "normalization": checkpoint["normalization"],
        "training": checkpoint["training"],
    }
    save_json(os.path.join(out_dir, "training_metadata.json"), metadata)

    print(f"\nTraining finished. Checkpoint saved to: {checkpoint_path}")
    if best_loss_for_metadata is None:
        print("Best training loss: not available because no training epochs were run.")
    else:
        print(f"Best training loss: {best_loss:.8E}")


if __name__ == "__main__":
    main()
