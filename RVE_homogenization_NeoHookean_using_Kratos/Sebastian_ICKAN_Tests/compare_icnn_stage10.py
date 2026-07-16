#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplcfg")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np


COMPONENTS = (
    (0, r"$\sigma_{xx}$", r"$\varepsilon_{xx}$", "sigma_xx"),
    (1, r"$\sigma_{yy}$", r"$\varepsilon_{yy}$", "sigma_yy"),
    (2, r"$\sigma_{xy}$", r"$\gamma_{xy}$", "sigma_xy"),
)


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def _load_array(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return np.asarray(np.load(path), dtype=float)


def _maybe_load_array(path):
    if not os.path.exists(path):
        return None
    return np.asarray(np.load(path), dtype=float)


def _as_history(array):
    array = np.asarray(array, dtype=float)
    if array.ndim == 3:
        if array.shape[0] != 1:
            raise ValueError(f"Expected one external trajectory, got shape {array.shape}.")
        array = array[0]
    if array.ndim != 2 or array.shape[1] < 3:
        raise ValueError(f"Expected history with shape [steps, >=3], got {array.shape}.")
    return array[:, :3]


def _relative_error(pred, ref):
    pred = np.asarray(pred, dtype=float)
    ref = np.asarray(ref, dtype=float)
    return float(np.linalg.norm(pred - ref) / max(np.linalg.norm(ref), 1.0e-30))


def _component_errors(pred, ref):
    labels = ("sigma_xx", "sigma_yy", "sigma_xy")
    return {labels[i]: _relative_error(pred[:, i], ref[:, i]) for i in range(3)}


def _format_component_errors(errors):
    return ", ".join(f"{key}={value:.4e}" for key, value in errors.items())


def _compute_equivalent(eps, sig):
    eps = np.asarray(eps, dtype=float)
    sig = np.asarray(sig, dtype=float)
    exx = eps[:, 0]
    eyy = eps[:, 1]
    gxy = eps[:, 2]
    sxx = sig[:, 0]
    syy = sig[:, 1]
    sxy = sig[:, 2]
    sigma_eq = np.sqrt(np.maximum(sxx * sxx - sxx * syy + syy * syy + 3.0 * sxy * sxy, 0.0))
    eps_eq = (2.0 / 3.0) * np.sqrt(np.maximum(exx * exx + eyy * eyy - exx * eyy + 0.75 * gxy * gxy, 0.0))
    return eps_eq, sigma_eq


def _path_work(eps, sig):
    eps = np.asarray(eps, dtype=float)
    sig = np.asarray(sig, dtype=float)
    increments = np.zeros((eps.shape[0],), dtype=float)
    if eps.shape[0] > 1:
        de = eps[1:] - eps[:-1]
        sig_avg = 0.5 * (sig[1:] + sig[:-1])
        increments[1:] = np.sum(sig_avg * de, axis=1)
    return np.cumsum(increments)


def _configure_plots():
    plt.rcParams.update(
        {
            "font.size": 12,
            "mathtext.fontset": "stix",
            "font.family": "STIXGeneral",
        }
    )


def _plot_component_strain_paths(out_dir, histories, n, dpi):
    for i, label_sig, label_eps, suffix in COMPONENTS:
        fig, ax = plt.subplots(figsize=(7, 6))
        for name, style in histories.items():
            eps = style["eps"][:n]
            sig = style["sig"][:n]
            ax.plot(
                eps[:, i],
                sig[:, i],
                style["line"],
                label=name,
                linewidth=style.get("lw", 1.5),
            )
        ax.set_title(f"Stage 10 comparison: {label_sig}")
        ax.set_xlabel(f"{label_eps} [-]")
        ax.set_ylabel(f"{label_sig} [Pa]")
        ax.grid(True, linestyle="--", alpha=0.7)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"stage10_icnn_comp_{suffix}.png"), dpi=dpi)
        plt.close(fig)


def _plot_equivalent(out_dir, histories, n, dpi):
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, style in histories.items():
        eps_eq, sig_eq = _compute_equivalent(style["eps"][:n], style["sig"][:n])
        ax.plot(eps_eq, sig_eq, style["line"], label=name, linewidth=style.get("lw", 1.5))
    ax.set_title(r"Stage 10 comparison: $\sigma_{eq}$ vs $\varepsilon_{eq}$")
    ax.set_xlabel(r"$\varepsilon_{eq}$ [-]")
    ax.set_ylabel(r"$\sigma_{eq}$ [Pa]")
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "stage10_icnn_comp_sigma_eq.png"), dpi=dpi)
    plt.close(fig)


def _plot_stress_vs_step(out_dir, histories, n, dpi):
    steps = np.arange(n)
    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    for i, label_sig, _, _ in COMPONENTS:
        for name, style in histories.items():
            axes[i].plot(
                steps,
                style["sig"][:n, i],
                style["line"],
                label=name if i == 0 else None,
                linewidth=style.get("lw", 1.5),
            )
        axes[i].set_ylabel(f"{label_sig} [Pa]")
        axes[i].grid(True, linestyle="--", alpha=0.7)
    axes[-1].set_xlabel("Step")
    axes[0].legend()
    fig.suptitle("Stage 10 comparison: stress vs step")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "stage10_icnn_stress_vs_step.png"), dpi=dpi)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for name, style in histories.items():
        _, sig_eq = _compute_equivalent(style["eps"][:n], style["sig"][:n])
        ax.plot(steps, sig_eq, style["line"], label=name, linewidth=style.get("lw", 1.5))
    ax.set_title(r"Stage 10 comparison: $\sigma_{eq}$ vs step")
    ax.set_xlabel("Step")
    ax.set_ylabel(r"$\sigma_{eq}$ [Pa]")
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "stage10_icnn_sigma_eq_vs_step.png"), dpi=dpi)
    plt.close(fig)


def _plot_error_history(out_dir, histories, n, dpi):
    fom_sig = histories["FOM"]["sig"][:n]
    fom_norm = np.linalg.norm(fom_sig, axis=1)
    floor = max(1.0e-12 * float(np.max(fom_norm)), 1.0e-30)
    denom = np.maximum(fom_norm, floor)

    fig, ax = plt.subplots(figsize=(8, 5))
    for name, style in histories.items():
        if name == "FOM":
            continue
        err = np.linalg.norm(style["sig"][:n] - fom_sig, axis=1) / denom
        ax.plot(np.arange(n), err, style["line"], label=name, linewidth=style.get("lw", 1.5))
    ax.set_title("Stage 10 comparison: pointwise stress error")
    ax.set_xlabel("Step")
    ax.set_ylabel("Relative error [-]")
    ax.set_yscale("log")
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "stage10_icnn_error_history.png"), dpi=dpi)
    plt.close(fig)


def _plot_energy(out_dir, histories, pred, n, dpi):
    steps = np.arange(n)

    if "W_reference_normalized" in pred and "W_predicted_normalized" in pred:
        w_ref = _as_scalar_history(pred["W_reference_normalized"])[:n]
        w_pred = _as_scalar_history(pred["W_predicted_normalized"])[:n]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(steps, w_ref, "k-", label="FOM path energy", linewidth=2.0)
        ax.plot(steps, w_pred, "c--", label="ICNN potential", linewidth=1.6)
        ax.set_title("Stage 10 ICNN energy")
        ax.set_xlabel("Step")
        ax.set_ylabel(r"Normalized energy $\widehat{W}$")
        ax.grid(True, linestyle="--", alpha=0.7)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "stage10_icnn_energy_vs_step.png"), dpi=dpi)
        plt.close(fig)

    work = {name: _path_work(style["eps"][:n], style["sig"][:n]) for name, style in histories.items()}
    scale = max(float(np.max(np.abs(work["FOM"]))), 1.0e-30)
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, style in histories.items():
        ax.plot(
            steps,
            work[name] / scale,
            style["line"],
            label=name,
            linewidth=style.get("lw", 1.5),
        )
    ax.set_title("Stage 10 comparison: accumulated path work")
    ax.set_xlabel("Step")
    ax.set_ylabel("Path work / max |FOM path work| [-]")
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "stage10_icnn_path_work_vs_step.png"), dpi=dpi)
    plt.close(fig)


def _as_scalar_history(array):
    array = np.asarray(array, dtype=float)
    if array.ndim == 3:
        array = array[0, :, 0]
    elif array.ndim == 2:
        array = array[:, 0]
    elif array.ndim != 1:
        raise ValueError(f"Expected scalar history, got shape {array.shape}.")
    return array


def _write_summary(out_dir, histories, pred, n):
    fom_sig = histories["FOM"]["sig"][:n]
    fom_eps = histories["FOM"]["eps"][:n]
    lines = [
        "Stage 10 ICNN comparison",
        "========================",
        f"compared steps: {n}",
        "",
        "Stress relative errors vs FOM:",
    ]

    summary = {"compared_steps": int(n), "stress_vs_fom": {}, "strain_vs_fom": {}}
    for name, style in histories.items():
        if name == "FOM":
            continue
        sig = style["sig"][:n]
        eps = style["eps"][:n]
        stress_rel = _relative_error(sig, fom_sig)
        strain_rel = _relative_error(eps, fom_eps)
        stress_comp = _component_errors(sig, fom_sig)
        strain_comp = {
            ("eps_xx", "eps_yy", "gamma_xy")[i]: _relative_error(eps[:, i], fom_eps[:, i])
            for i in range(3)
        }
        summary["stress_vs_fom"][name] = {
            "relative": stress_rel,
            "components": stress_comp,
        }
        summary["strain_vs_fom"][name] = {
            "relative": strain_rel,
            "components": strain_comp,
        }
        lines.append(f"  {name:<12}: {stress_rel:.4e} ({_format_component_errors(stress_comp)})")

    if "W_reference_normalized" in pred and "W_predicted_normalized" in pred:
        w_ref = _as_scalar_history(pred["W_reference_normalized"])[:n]
        w_pred = _as_scalar_history(pred["W_predicted_normalized"])[:n]
        energy_rel = _relative_error(w_pred, w_ref)
        summary["icnn_energy_relative_error"] = energy_rel
        lines.extend(["", f"ICNN normalized energy vs FOM path energy: {energy_rel:.4e}"])

    path = os.path.join(out_dir, "stage10_icnn_comparison_summary.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(out_dir, "stage10_icnn_comparison_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    for line in lines:
        print(line)
    print(f"Saved summary: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare an ICNN/ICKAN Stage-10 prediction against FOM/PROM/HPROM arrays."
    )
    parser.add_argument("--prediction-npz", required=True, help="predictions.npz from predict_ICKAN_surrogate.py")
    parser.add_argument(
        "--stage10-dir",
        default="../stage_10_hprom_ann_ls_results_mawecm_res_eps_sig_phase1to40_phase2to10_sum990_ann_hrom",
        help="Stage 10 output directory containing FOM/PROM/HPROM .npy arrays.",
    )
    parser.add_argument("--out-dir", required=True, help="Directory where comparison plots are written.")
    parser.add_argument("--model-label", default="ICNN", help="Label for the predicted energy surrogate.")
    parser.add_argument("--include-direct-hprom", type=int, default=1, choices=[0, 1])
    parser.add_argument(
        "--icnn-plot-strain-source",
        default="stage10_fom",
        choices=("stage10_fom", "prediction_input"),
        help=(
            "Strain history used as x-axis for the ICNN stress plots. Use "
            "'stage10_fom' to match Stage 10 HPROM plots even when the ICNN "
            "input is applied_strain."
        ),
    )
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()

    _configure_plots()
    out_dir = _ensure_dir(os.path.abspath(args.out_dir))
    stage10_dir = os.path.abspath(args.stage10_dir)
    pred = np.load(os.path.abspath(args.prediction_npz), allow_pickle=True)

    fom_eps = _load_array(os.path.join(stage10_dir, "fom_strain.npy"))
    fom_sig = _load_array(os.path.join(stage10_dir, "fom_stress.npy"))
    prom_eps = _load_array(os.path.join(stage10_dir, "prom_ann_strain.npy"))
    prom_sig = _load_array(os.path.join(stage10_dir, "prom_ann_stress.npy"))
    hprom_eps = _load_array(os.path.join(stage10_dir, "hprom_ann_strain.npy"))
    hprom_sig = _load_array(os.path.join(stage10_dir, "hprom_ann_stress.npy"))
    icnn_input = _as_history(pred["strain"])
    if args.icnn_plot_strain_source == "stage10_fom":
        icnn_eps = fom_eps
    else:
        icnn_eps = icnn_input
    icnn_sig = _as_history(pred["stress_predicted"])

    histories = {
        "FOM": {"eps": fom_eps, "sig": fom_sig, "line": "k-", "lw": 2.0},
        "PROM-ANN": {"eps": prom_eps, "sig": prom_sig, "line": "r--", "lw": 1.5},
        "HPROM-ANN": {"eps": hprom_eps, "sig": hprom_sig, "line": "b:", "lw": 1.5},
    }
    if int(args.include_direct_hprom):
        d_eps = _maybe_load_array(os.path.join(stage10_dir, "dhprom_ann_strain.npy"))
        d_sig = _maybe_load_array(os.path.join(stage10_dir, "dhprom_ann_stress.npy"))
        if d_eps is not None and d_sig is not None:
            histories["D-HPROM-ANN"] = {"eps": d_eps, "sig": d_sig, "line": "g-.", "lw": 1.5}
    histories[str(args.model_label)] = {"eps": icnn_eps, "sig": icnn_sig, "line": "c--", "lw": 1.6}

    n = min(len(style["sig"]) for style in histories.values())
    n = min(n, min(len(style["eps"]) for style in histories.values()))

    _plot_component_strain_paths(out_dir, histories, n, args.dpi)
    _plot_equivalent(out_dir, histories, n, args.dpi)
    _plot_stress_vs_step(out_dir, histories, n, args.dpi)
    _plot_error_history(out_dir, histories, n, args.dpi)
    _plot_energy(out_dir, histories, pred, n, args.dpi)
    _write_summary(out_dir, histories, pred, n)


if __name__ == "__main__":
    main()
