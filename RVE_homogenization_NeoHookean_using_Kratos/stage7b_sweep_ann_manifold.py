#!/usr/bin/env python3
import os
import csv
import json
import shutil
import argparse
import traceback
import numpy as np

from stage7b_train_ann_manifold import train_ann


def _sample_configs_random(n_trials, seed):
    rng = np.random.default_rng(int(seed))

    hidden_choices = [
        (64, 64, 64, 64),
        (96, 96, 96, 96),
        (128, 128, 128),
        (128, 128, 128, 128),
        (64, 128, 128, 64),
        (128, 192, 192, 128),
        (128, 256, 256, 128),
        (256, 256, 128),
    ]
    activation_choices = ["elu", "silu", "gelu", "relu"]
    dropout_choices = [0.0, 0.01, 0.02, 0.03, 0.05, 0.08]
    use_bn_choices = [True, False]
    loss_choices = ["mse", "smoothl1"]
    smoothl1_beta_choices = [0.3, 0.5, 0.8, 1.0]
    lr_choices = [1.2e-3, 1.0e-3, 8.0e-4, 6.0e-4, 4.0e-4, 2.0e-4]
    wd_choices = [0.0, 1.0e-6, 1.0e-5, 5.0e-5, 1.0e-4]
    batch_choices = [1024, 1536, 2048, 3072, 4096]
    patience_choices = [80, 100, 120, 140, 160]
    lr_patience_choices = [20, 25, 30, 35, 45]
    lr_factor_choices = [0.4, 0.5, 0.6, 0.7]
    grad_clip_choices = [0.0, 0.5, 1.0, 2.0]

    configs = []
    seen = set()
    attempts = 0
    max_attempts = int(max(200, n_trials * 30))

    while len(configs) < int(n_trials) and attempts < max_attempts:
        attempts += 1
        loss_name = str(rng.choice(loss_choices))
        beta = float(rng.choice(smoothl1_beta_choices)) if loss_name == "smoothl1" else 1.0
        hidden = hidden_choices[int(rng.integers(0, len(hidden_choices)))]
        cfg = {
            "hidden_layers": tuple(int(v) for v in hidden),
            "activation": str(rng.choice(activation_choices)),
            "dropout": float(rng.choice(dropout_choices)),
            "use_batchnorm": bool(rng.choice(use_bn_choices)),
            "loss": loss_name,
            "smoothl1_beta": float(beta),
            "lr": float(rng.choice(lr_choices)),
            "weight_decay": float(rng.choice(wd_choices)),
            "batch_size": int(rng.choice(batch_choices)),
            "patience": int(rng.choice(patience_choices)),
            "lr_patience": int(rng.choice(lr_patience_choices)),
            "lr_factor": float(rng.choice(lr_factor_choices)),
            "grad_clip_norm": float(rng.choice(grad_clip_choices)),
        }
        key = (
            cfg["hidden_layers"],
            cfg["activation"],
            cfg["dropout"],
            int(cfg["use_batchnorm"]),
            cfg["loss"],
            cfg["smoothl1_beta"],
            cfg["lr"],
            cfg["weight_decay"],
            cfg["batch_size"],
            cfg["patience"],
            cfg["lr_patience"],
            cfg["lr_factor"],
            cfg["grad_clip_norm"],
        )
        if key in seen:
            continue
        seen.add(key)
        configs.append(cfg)

    if len(configs) < int(n_trials):
        raise RuntimeError(
            f"Could only generate {len(configs)} unique configs out of requested {int(n_trials)}."
        )
    return configs


def _sample_configs_progressive_fast(n_trials, seed):
    """
    Build progressive-width ANN candidates oriented to faster Jacobian evaluation:
      - compact/progressive hidden widths
      - no batchnorm
      - zero dropout
      - avoid GELU by default (ELU/ReLU/SiLU only)
    """
    rng = np.random.default_rng(int(seed))

    # Progressive families (input is 3; output is 18 in ANN-LS dataset).
    hidden_choices = [
        (5, 10, 15, 20, 23),
        (4, 8, 12, 16, 20),
        (5, 10, 15, 20),
        (6, 9, 12, 15, 18),
        (6, 12, 18, 24),
        (8, 12, 16, 20),
        (8, 12, 16, 20, 24),
        (10, 14, 18, 22),
        (10, 15, 20),
        (12, 16, 20),
    ]
    activation_choices = ["elu", "relu", "silu"]
    # Fixed for speed:
    dropout_choices = [0.0]
    use_bn_choices = [False]

    # Still keep optimization diversity.
    loss_choices = ["mse", "smoothl1"]
    smoothl1_beta_choices = [0.5, 0.8, 1.0]
    lr_choices = [1.2e-3, 1.0e-3, 8.0e-4, 6.0e-4, 4.0e-4]
    wd_choices = [0.0, 1.0e-6, 1.0e-5, 5.0e-5]
    batch_choices = [2048, 3072, 4096]
    patience_choices = [60, 80, 100]
    lr_patience_choices = [20, 30, 35]
    lr_factor_choices = [0.5, 0.7]
    grad_clip_choices = [0.5, 1.0, 2.0]

    configs = []
    seen = set()
    attempts = 0
    max_attempts = int(max(300, n_trials * 80))

    while len(configs) < int(n_trials) and attempts < max_attempts:
        attempts += 1
        loss_name = str(rng.choice(loss_choices))
        beta = float(rng.choice(smoothl1_beta_choices)) if loss_name == "smoothl1" else 1.0
        hidden = hidden_choices[int(rng.integers(0, len(hidden_choices)))]
        cfg = {
            "hidden_layers": tuple(int(v) for v in hidden),
            "activation": str(rng.choice(activation_choices)),
            "dropout": float(rng.choice(dropout_choices)),
            "use_batchnorm": bool(rng.choice(use_bn_choices)),
            "loss": loss_name,
            "smoothl1_beta": float(beta),
            "lr": float(rng.choice(lr_choices)),
            "weight_decay": float(rng.choice(wd_choices)),
            "batch_size": int(rng.choice(batch_choices)),
            "patience": int(rng.choice(patience_choices)),
            "lr_patience": int(rng.choice(lr_patience_choices)),
            "lr_factor": float(rng.choice(lr_factor_choices)),
            "grad_clip_norm": float(rng.choice(grad_clip_choices)),
        }
        key = (
            cfg["hidden_layers"],
            cfg["activation"],
            cfg["dropout"],
            int(cfg["use_batchnorm"]),
            cfg["loss"],
            cfg["smoothl1_beta"],
            cfg["lr"],
            cfg["weight_decay"],
            cfg["batch_size"],
            cfg["patience"],
            cfg["lr_patience"],
            cfg["lr_factor"],
            cfg["grad_clip_norm"],
        )
        if key in seen:
            continue
        seen.add(key)
        configs.append(cfg)

    if len(configs) < int(n_trials):
        raise RuntimeError(
            f"Could only generate {len(configs)} unique progressive-fast configs out of requested {int(n_trials)}."
        )
    return configs


def _copy_best_artifacts(best_dir, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    files = [
        "manifold_ann.pt",
        "manifold_ann_metadata.npz",
        "training_history.png",
        "training_summary.txt",
    ]
    copied = []
    for name in files:
        src = os.path.join(best_dir, name)
        if os.path.exists(src):
            dst = os.path.join(dst_dir, name)
            shutil.copy2(src, dst)
            copied.append(name)
    return copied


def run_sweep(
    data_dir="stage_7_ann_data",
    out_dir="stage_7_ann_sweep",
    best_out_dir="",
    n_trials=20,
    seed=42,
    val_fraction=0.1,
    epochs=2000,
    base_seed=1000,
    continue_on_error=True,
    config_mode="random",
):
    os.makedirs(out_dir, exist_ok=True)
    if not best_out_dir:
        best_out_dir = data_dir

    mode = str(config_mode).strip().lower()
    if mode == "random":
        configs = _sample_configs_random(n_trials=n_trials, seed=seed)
    elif mode == "progressive_fast":
        configs = _sample_configs_progressive_fast(n_trials=n_trials, seed=seed)
    else:
        raise ValueError(f"Unsupported config_mode='{config_mode}'. Use one of: random, progressive_fast.")

    print("=" * 72)
    print("Stage 7b-ANN sweep")
    print("=" * 72)
    print(f"data_dir      : {data_dir}")
    print(f"out_dir       : {out_dir}")
    print(f"best_out_dir  : {best_out_dir}")
    print(f"n_trials      : {int(n_trials)}")
    print(f"seed          : {int(seed)}")
    print(f"epochs        : {int(epochs)}")
    print(f"val_fraction  : {float(val_fraction):.3f}")
    print(f"config_mode   : {mode}")

    results = []
    for i, cfg in enumerate(configs, start=1):
        trial_name = f"trial_{i:02d}"
        trial_dir = os.path.join(out_dir, trial_name)
        os.makedirs(trial_dir, exist_ok=True)
        trial_seed = int(base_seed) + int(i)

        print("-" * 72)
        print(f"[{i:02d}/{int(n_trials):02d}] {trial_name}")
        print(
            f"  hidden={cfg['hidden_layers']} act={cfg['activation']} drop={cfg['dropout']:.3f} "
            f"bn={int(cfg['use_batchnorm'])} loss={cfg['loss']} lr={cfg['lr']:.2e} wd={cfg['weight_decay']:.1e}"
        )

        row = {
            "trial": trial_name,
            "status": "ok",
            "error": "",
            "trial_seed": int(trial_seed),
            **cfg,
        }
        try:
            train_info = train_ann(
                data_dir=data_dir,
                out_dir=trial_dir,
                seed=trial_seed,
                hidden_layers=cfg["hidden_layers"],
                activation=cfg["activation"],
                dropout=cfg["dropout"],
                use_batchnorm=cfg["use_batchnorm"],
                val_fraction=val_fraction,
                epochs=epochs,
                patience=cfg["patience"],
                batch_size=cfg["batch_size"],
                lr=cfg["lr"],
                weight_decay=cfg["weight_decay"],
                lr_patience=cfg["lr_patience"],
                lr_factor=cfg["lr_factor"],
                loss=cfg["loss"],
                smoothl1_beta=cfg["smoothl1_beta"],
                grad_clip_norm=cfg["grad_clip_norm"],
            )
            row.update(train_info)
            print(
                f"  -> best_val_scaled_mse={float(train_info['best_val_scaled_mse']):.6e} "
                f"(best_epoch={int(train_info['best_epoch'])})"
            )
        except Exception as exc:
            row["status"] = "fail"
            row["error"] = f"{type(exc).__name__}: {exc}"
            print(f"  [ERROR] {row['error']}")
            if not continue_on_error:
                raise
            with open(os.path.join(trial_dir, "error_traceback.txt"), "w", encoding="utf-8") as f:
                f.write(traceback.format_exc())

        results.append(row)

    csv_path = os.path.join(out_dir, "sweep_results.csv")
    json_path = os.path.join(out_dir, "sweep_results.json")
    keys = [
        "trial",
        "status",
        "error",
        "trial_seed",
        "hidden_layers",
        "activation",
        "dropout",
        "use_batchnorm",
        "loss",
        "smoothl1_beta",
        "lr",
        "weight_decay",
        "batch_size",
        "patience",
        "lr_patience",
        "lr_factor",
        "grad_clip_norm",
        "best_val_scaled_mse",
        "best_epoch",
        "epochs_ran",
        "last_val_scaled_mse",
        "last_train_scaled_mse",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in results:
            rr = dict(r)
            if isinstance(rr.get("hidden_layers"), (tuple, list)):
                rr["hidden_layers"] = ",".join(str(int(v)) for v in rr["hidden_layers"])
            w.writerow({k: rr.get(k, "") for k in keys})

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    ok_rows = [r for r in results if r.get("status") == "ok" and np.isfinite(float(r.get("best_val_scaled_mse", np.inf)))]
    if not ok_rows:
        raise RuntimeError("Sweep finished with no successful trial.")

    best = min(ok_rows, key=lambda r: float(r["best_val_scaled_mse"]))
    best_dir = os.path.join(out_dir, str(best["trial"]))
    copied = _copy_best_artifacts(best_dir=best_dir, dst_dir=best_out_dir)

    summary_path = os.path.join(out_dir, "sweep_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Stage 7b ANN sweep summary\n")
        f.write(f"data_dir={data_dir}\n")
        f.write(f"out_dir={out_dir}\n")
        f.write(f"best_out_dir={best_out_dir}\n")
        f.write(f"n_trials={int(n_trials)}\n")
        f.write(f"n_success={len(ok_rows)}\n")
        f.write(f"n_fail={len(results)-len(ok_rows)}\n")
        f.write(f"best_trial={best['trial']}\n")
        f.write(f"best_val_scaled_mse={float(best['best_val_scaled_mse']):.16e}\n")
        f.write(f"best_epoch={int(best['best_epoch'])}\n")
        f.write(f"copied_artifacts={copied}\n")
        f.write(f"best_config={json.dumps(best, default=str)}\n")

    best_cfg_path = os.path.join(best_out_dir, "manifold_ann_best_config_from_sweep.json")
    with open(best_cfg_path, "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2, default=str)

    print("=" * 72)
    print("Sweep complete")
    print("=" * 72)
    print(f"Successful trials: {len(ok_rows)}/{len(results)}")
    print(f"Best trial: {best['trial']}")
    print(f"Best val scaled MSE: {float(best['best_val_scaled_mse']):.6e}")
    print(f"Best artifacts copied to: {best_out_dir}")
    print(f"Copied files: {copied}")
    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Stage 7b ANN sweep over multiple architectures/hyperparameters.")
    p.add_argument("--data-dir", type=str, default="stage_7_ann_data", help="Dataset directory (contains ANN LS/POD inputs).")
    p.add_argument("--out-dir", type=str, default="stage_7_ann_sweep", help="Directory where all trial subfolders and ranking are saved.")
    p.add_argument("--best-out-dir", type=str, default="", help="Where best model artifacts are copied. Default: data-dir.")
    p.add_argument("--n-trials", type=int, default=20, help="Number of sweep trials.")
    p.add_argument("--seed", type=int, default=42, help="Sweep sampling seed.")
    p.add_argument("--base-seed", type=int, default=1000, help="Base seed for trial-specific training seeds.")
    p.add_argument("--val-fraction", type=float, default=0.10, help="Validation split fraction.")
    p.add_argument("--epochs", type=int, default=2500, help="Max epochs per trial.")
    p.add_argument(
        "--config-mode",
        type=str,
        default="random",
        choices=["random", "progressive_fast"],
        help="Config sampler: 'random' (legacy broad sweep) or 'progressive_fast' (progressive-width, fast-Jacobian oriented).",
    )
    p.add_argument("--stop-on-error", action="store_true", help="Stop the sweep if one trial fails.")
    args = p.parse_args()

    run_sweep(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        best_out_dir=args.best_out_dir,
        n_trials=args.n_trials,
        seed=args.seed,
        val_fraction=args.val_fraction,
        epochs=args.epochs,
        base_seed=args.base_seed,
        continue_on_error=(not args.stop_on_error),
        config_mode=args.config_mode,
    )
