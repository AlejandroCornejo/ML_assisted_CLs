#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage 10: FOM vs PROM-GPR vs HPROM-GPR benchmark.
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from plot_style_utils import apply_latex_plot_style
apply_latex_plot_style()

from stage6_test_hprom import generate_safe_test_path
from stage4_test_rve import plot_path_in_domain
from fom_solver_rve import (
    setup_kratos_parameters,
    RunFomBatchSimulation,
    BuildDynamicSegmentSteps,
    REFERENCE_STEPS_FOR_UNIT_AMPLITUDE,
    MIN_STEPS_PER_SEGMENT,
)
from prom_gpr_solver_rve import LoadPromGprModel, RunPromGprBatchSimulation
from hprom_gpr_solver_rve import LoadHpromGprModel, RunHpromGprBatchSimulation


def _load_hrom_mesh_base_from_ecm_dir(hprom_gpr_dir):
    ecm_file = os.path.join(str(hprom_gpr_dir), "ecm_weights_all.npz")
    if not os.path.exists(ecm_file):
        return None
    try:
        ecm = np.load(ecm_file, allow_pickle=True)
    except Exception:
        return None
    if "hrom_mesh_base" not in ecm:
        return None
    try:
        return str(np.ravel(ecm["hrom_mesh_base"])[0])
    except Exception:
        return None


def _compute_equivalent_stress_strain(eps, sig):
    eps = np.asarray(eps, dtype=float)
    sig = np.asarray(sig, dtype=float)
    if eps.ndim != 2 or sig.ndim != 2 or eps.shape[1] < 3 or sig.shape[1] < 3:
        raise ValueError("Expected eps and sig arrays with shape [n_steps, >=3].")

    exx = eps[:, 0]
    eyy = eps[:, 1]
    gxy = eps[:, 2]  # engineering shear strain
    sxx = sig[:, 0]
    syy = sig[:, 1]
    sxy = sig[:, 2]

    sigma_eq = np.sqrt(np.maximum(sxx * sxx - sxx * syy + syy * syy + 3.0 * sxy * sxy, 0.0))
    eps_eq = (2.0 / 3.0) * np.sqrt(np.maximum(exx * exx + eyy * eyy - exx * eyy + 0.75 * gxy * gxy, 0.0))
    return eps_eq, sigma_eq


def _rel_fro(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def _aligned_error_metrics(reference, approximation):
    ref = np.asarray(reference, dtype=float)
    approx = np.asarray(approximation, dtype=float)
    if ref.ndim != 2 or approx.ndim != 2:
        raise ValueError("Expected two-dimensional histories.")

    n = int(min(ref.shape[0], approx.shape[0]))
    k = int(min(ref.shape[1], approx.shape[1]))
    if n <= 0 or k <= 0:
        raise ValueError("Cannot compare empty histories.")

    ref = ref[:n, :k]
    approx = approx[:n, :k]
    diff = approx - ref
    return {
        "n": n,
        "k": k,
        "rel_fro": float(np.linalg.norm(diff) / max(np.linalg.norm(ref), 1e-30)),
        "rmse": float(np.sqrt(np.mean(diff * diff))),
        "max_abs": float(np.max(np.abs(diff))),
        "per_component_rel": np.array(
            [
                np.linalg.norm(diff[:, j]) / max(np.linalg.norm(ref[:, j]), 1e-30)
                for j in range(k)
            ],
            dtype=float,
        ),
    }


def _format_component_errors(values, labels):
    vals = np.asarray(values, dtype=float).reshape(-1)
    return ", ".join(
        f"{label}={vals[j]:.4e}"
        for j, label in enumerate(labels[: vals.size])
    )


def _build_step_macro_strain_series(strain_path, seg_steps):
    e_wp = np.asarray(strain_path, dtype=float)
    if e_wp.ndim != 2 or e_wp.shape[1] < 3:
        raise ValueError("strain_path must have shape [n_pts, >=3].")
    seg_steps = np.asarray(seg_steps, dtype=int).reshape(-1)
    n_steps = int(np.sum(seg_steps))
    step_offsets = np.concatenate(([0], np.cumsum(seg_steps)))
    e_hist = np.zeros((n_steps + 1, 3), dtype=float)
    n_seg = int(len(seg_steps))
    for step in range(1, n_steps + 1):
        s = int(np.searchsorted(step_offsets, step, side="left") - 1)
        s = max(0, min(s, n_seg - 1))
        xi = float(step - step_offsets[s]) / float(max(seg_steps[s], 1))
        e_hist[step, :] = (1.0 - xi) * e_wp[s, :3] + xi * e_wp[s + 1, :3]
    return e_hist


def _column_corr(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    c = np.zeros(x.shape[1], dtype=float)
    for j in range(x.shape[1]):
        sx = float(np.std(x[:, j]))
        sy = float(np.std(y[:, j]))
        if sx <= 1e-30 or sy <= 1e-30:
            c[j] = 0.0
        else:
            c[j] = float(np.corrcoef(x[:, j], y[:, j])[0, 1])
    return c


def _analyze_mu_vs_qp(mu_hist, qp_hist):
    mu = np.asarray(mu_hist, dtype=float)
    qp = np.asarray(qp_hist, dtype=float)
    if mu.ndim != 2 or qp.ndim != 2:
        raise ValueError("mu_hist and qp_hist must be 2D arrays.")

    k = int(min(3, mu.shape[1], qp.shape[1]))
    if k <= 0:
        raise ValueError("No overlapping dimensions to compare (need at least 1).")

    n = int(min(mu.shape[0], qp.shape[0]))
    x = mu[:n, :k]
    y = qp[:n, :k]
    mask = np.all(np.isfinite(x), axis=1) & np.all(np.isfinite(y), axis=1)
    x = x[mask, :]
    y = y[mask, :]

    if x.shape[0] < max(10, 2 * k):
        raise RuntimeError("Not enough valid samples for mu-vs-q_p diagnostics.")

    # 1) Direct comparison (identity)
    rel_direct = _rel_fro(x, y)
    corr_direct = _column_corr(x, y)

    # 2) Per-axis scale/sign map: y_i ≈ a_i x_i
    den = np.sum(x * x, axis=0)
    den = np.where(den <= 1e-30, 1.0, den)
    a_diag = np.sum(x * y, axis=0) / den
    y_hat_diag = x * a_diag[None, :]
    rel_diag = _rel_fro(y_hat_diag, y)
    corr_diag = _column_corr(y_hat_diag, y)

    # 3) Full linear map (scaling/rotation/shear, no translation): y ≈ x A
    a_lin, *_ = np.linalg.lstsq(x, y, rcond=None)
    y_hat_lin = x @ a_lin
    rel_lin = _rel_fro(y_hat_lin, y)
    corr_lin = _column_corr(y_hat_lin, y)

    # 4) Affine map (linear + translation): y ≈ [x,1] B
    x_aug = np.hstack([x, np.ones((x.shape[0], 1), dtype=float)])
    b_aff, *_ = np.linalg.lstsq(x_aug, y, rcond=None)
    y_hat_aff = x_aug @ b_aff
    rel_aff = _rel_fro(y_hat_aff, y)
    corr_aff = _column_corr(y_hat_aff, y)

    # 5) Similarity Procrustes (rotation/reflection + uniform scale + translation)
    x_mean = np.mean(x, axis=0, keepdims=True)
    y_mean = np.mean(y, axis=0, keepdims=True)
    xc = x - x_mean
    yc = y - y_mean
    u, svals, vt = np.linalg.svd(xc.T @ yc, full_matrices=False)
    r = u @ vt
    beta = float(np.sum(svals) / max(np.sum(xc * xc), 1e-30))
    y_hat_sim = beta * (xc @ r) + y_mean
    rel_sim = _rel_fro(y_hat_sim, y)
    corr_sim = _column_corr(y_hat_sim, y)

    return {
        "n_samples": int(x.shape[0]),
        "k_compare": int(k),
        "x": x,
        "y": y,
        "y_hat_diag": y_hat_diag,
        "y_hat_lin": y_hat_lin,
        "y_hat_aff": y_hat_aff,
        "y_hat_sim": y_hat_sim,
        "a_diag": a_diag,
        "a_lin": a_lin,
        "b_aff": b_aff,
        "r_sim": r,
        "beta_sim": beta,
        "rel_direct": rel_direct,
        "rel_diag": rel_diag,
        "rel_lin": rel_lin,
        "rel_aff": rel_aff,
        "rel_sim": rel_sim,
        "corr_direct": corr_direct,
        "corr_diag": corr_diag,
        "corr_lin": corr_lin,
        "corr_aff": corr_aff,
        "corr_sim": corr_sim,
    }


def _save_mu_qp_diagnostics(diag, out_dir, label):
    x = diag["x"]
    y = diag["y"]
    y_hat_aff = diag["y_hat_aff"]
    k = int(diag["k_compare"])
    safe = str(label).strip().lower().replace(" ", "_")

    summary_path = os.path.join(out_dir, f"mu_qp_alignment_{safe}.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"label={label}\n")
        f.write(f"n_samples={diag['n_samples']}\n")
        f.write(f"k_compare={k}\n")
        f.write(f"rel_direct={diag['rel_direct']:.16e}\n")
        f.write(f"rel_diag={diag['rel_diag']:.16e}\n")
        f.write(f"rel_linear={diag['rel_lin']:.16e}\n")
        f.write(f"rel_affine={diag['rel_aff']:.16e}\n")
        f.write(f"rel_similarity={diag['rel_sim']:.16e}\n")
        f.write("a_diag=" + ",".join([f"{float(v):.16e}" for v in diag["a_diag"]]) + "\n")
        f.write(
            "corr_direct=" + ",".join([f"{float(v):.16e}" for v in diag["corr_direct"]]) + "\n"
        )
        f.write("corr_affine=" + ",".join([f"{float(v):.16e}" for v in diag["corr_aff"]]) + "\n")

    np.savez(
        os.path.join(out_dir, f"mu_qp_alignment_{safe}.npz"),
        x=x,
        y=y,
        y_hat_diag=diag["y_hat_diag"],
        y_hat_lin=diag["y_hat_lin"],
        y_hat_aff=y_hat_aff,
        y_hat_sim=diag["y_hat_sim"],
        a_diag=diag["a_diag"],
        a_lin=diag["a_lin"],
        b_aff=diag["b_aff"],
        r_sim=diag["r_sim"],
        beta_sim=np.array([diag["beta_sim"]], dtype=float),
    )

    fig, axs = plt.subplots(1, k, figsize=(5.2 * k, 4.6))
    if k == 1:
        axs = [axs]
    for j in range(k):
        ax = axs[j]
        ax.scatter(x[:, j], y[:, j], s=8, alpha=0.35, label=r"$q_p$ vs $\mu$")
        ax.scatter(x[:, j], y_hat_aff[:, j], s=8, alpha=0.35, label="Affine fit")
        xmin = float(min(np.min(x[:, j]), np.min(y[:, j])))
        xmax = float(max(np.max(x[:, j]), np.max(y[:, j])))
        if xmax > xmin:
            ax.plot([xmin, xmax], [xmin, xmax], "k--", lw=1.0, alpha=0.8, label="y=x")
        ax.set_xlabel(fr"$\mu_{j+1}$")
        ax.set_ylabel(fr"$q_{{p,{j+1}}}$")
        ax.grid(True, linestyle="--", alpha=0.35)
    axs[0].legend(loc="best")
    fig.suptitle(f"mu vs q_p Alignment ({label})")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"mu_qp_alignment_{safe}_scatter.png"), dpi=180)
    plt.close(fig)

    fig, axs = plt.subplots(k, 1, figsize=(9.5, 2.4 * k), sharex=True)
    if k == 1:
        axs = [axs]
    t = np.arange(x.shape[0], dtype=int)
    for j in range(k):
        ax = axs[j]
        ax.plot(t, y[:, j], "k-", lw=1.2, label=fr"$q_{{p,{j+1}}}$")
        ax.plot(t, x[:, j], "r--", lw=1.0, label=fr"$\mu_{j+1}$")
        ax.plot(t, y_hat_aff[:, j], "b:", lw=1.0, label="Affine fit")
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.set_ylabel(fr"comp {j+1}")
    axs[0].legend(loc="best")
    axs[-1].set_xlabel("Step")
    fig.suptitle(f"mu/q_p Time Series ({label})")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"mu_qp_alignment_{safe}_timeseries.png"), dpi=180)
    plt.close(fig)

    print(f"  [{label}] mu-vs-q_p diagnostics saved: {summary_path}")
    print(
        f"  [{label}] rel errors: direct={diag['rel_direct']:.3e}, "
        f"diag={diag['rel_diag']:.3e}, linear={diag['rel_lin']:.3e}, "
        f"affine={diag['rel_aff']:.3e}, similarity={diag['rel_sim']:.3e}"
    )

def plot_hprom_gpr_comparison(f_eps, f_sig, p_eps, p_sig, h_eps, h_sig, out_dir, timings=None):
    n = min(len(f_sig), len(p_sig), len(h_sig), len(f_eps), len(p_eps), len(h_eps))

    plt.rcParams.update(
        {
            "font.size": 12,
            "mathtext.fontset": "stix",
            "font.family": "STIXGeneral",
        }
    )

    comp_labels = [
        (0, r"$\sigma_{xx}$", r"$\varepsilon_{xx}$", "sigma_xx"),
        (1, r"$\sigma_{yy}$", r"$\varepsilon_{yy}$", "sigma_yy"),
        (2, r"$\sigma_{xy}$", r"$\gamma_{xy}$", "sigma_xy"),
    ]

    for i, label_sig, label_eps, suffix in comp_labels:
        plt.figure(figsize=(7, 6))
        plt.plot(f_eps[:n, i], f_sig[:n, i], "k-", label="FOM", linewidth=2.0)
        plt.plot(p_eps[:n, i], p_sig[:n, i], "r--", label="PROM-GPR", linewidth=1.5)
        plt.plot(h_eps[:n, i], h_sig[:n, i], "b:", label="HPROM-GPR", linewidth=1.5)
        plt.title(f"HPROM-GPR Benchmark: {label_sig}")
        plt.xlabel(f"{label_eps} [-]")
        plt.ylabel(f"{label_sig} [Pa]")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"hprom_gpr_comp_{suffix}.png"), dpi=150)
        plt.close()

    f_eps_eq, f_sig_eq = _compute_equivalent_stress_strain(f_eps[:n], f_sig[:n])
    p_eps_eq, p_sig_eq = _compute_equivalent_stress_strain(p_eps[:n], p_sig[:n])
    h_eps_eq, h_sig_eq = _compute_equivalent_stress_strain(h_eps[:n], h_sig[:n])

    plt.figure(figsize=(7, 6))
    plt.plot(f_eps_eq, f_sig_eq, "k-", label="FOM", linewidth=2.0)
    plt.plot(p_eps_eq, p_sig_eq, "r--", label="PROM-GPR", linewidth=1.5)
    plt.plot(h_eps_eq, h_sig_eq, "b:", label="HPROM-GPR", linewidth=1.5)
    plt.title(r"HPROM-GPR Benchmark: $\sigma_{eq}$ vs $\varepsilon_{eq}$")
    plt.xlabel(r"$\varepsilon_{eq}$ [-]")
    plt.ylabel(r"$\sigma_{eq}$ [Pa]")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "hprom_gpr_comp_sigma_eq.png"), dpi=150)
    plt.close()

    fom_norm = np.linalg.norm(f_sig[:n], axis=1) + 1e-30
    err_prom_rbf = np.linalg.norm(f_sig[:n] - p_sig[:n], axis=1) / fom_norm
    err_hprom_rbf = np.linalg.norm(f_sig[:n] - h_sig[:n], axis=1) / fom_norm

    plt.figure(figsize=(7, 6))
    plt.plot(err_prom_rbf, "r-", label="PROM-GPR Error", linewidth=1.5)
    plt.plot(err_hprom_rbf, "b-", label="HPROM-GPR Error", linewidth=1.5)
    plt.title("Relative Stress Error Comparison")
    plt.xlabel("Step")
    plt.ylabel("Relative Error [-]")
    plt.yscale("log")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "hprom_gpr_comp_error.png"), dpi=150)
    plt.close()

    if timings:
        plt.figure(figsize=(7, 6))
        methods = list(timings.keys())
        values = [timings[m] for m in methods]
        colors = {"FOM": "gray", "PROM-GPR": "red", "HPROM-GPR": "blue"}
        bars = plt.bar(methods, values, color=[colors.get(m, "green") for m in methods], alpha=0.8)
        for bar, t in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(), f"{t:.1f}s", ha="center", va="bottom")
        plt.title("Computational Performance")
        plt.ylabel("Wall-Clock Time [s]")
        plt.grid(True, axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "hprom_gpr_performance.png"), dpi=150)
        plt.close()

    print(f"Stage 10 plots saved to: {out_dir}")


def run_stage10(
    run_fom=False,
    run_prom_gpr=False,
    run_hprom_gpr=False,
    gpr_data_dir="stage_7_gpr_data",
    hprom_gpr_dir="stage_9_hprom_gpr_data",
    out_dir="stage_10_hprom_gpr_results",
    qp_init_mode="previous",
    max_its=25,
    hprom_max_its=None,
    relnorm_cutoff=1.0e-5,
    abs_res_cutoff=1.0e-8,
    dq_abs_cutoff=1.0e-6,
    max_res_for_rel_convergence=1.0e-1,
    min_rel_drop_stop=1.0e-2,
    stagnation_relnorm_gate=1.0e-4,
    max_dq_norm=0.5,
    damping_after_iter=10,
    damping_factor=0.5,
    old_stiffness_residual_cutoff=1.0e5,
    regularization=1.0e-10,
    no_old_stiffness_first_it=False,
    verbose_iterations=False,
):
    os.makedirs(out_dir, exist_ok=True)
    hprom_max_its_eff = int(max_its) if hprom_max_its is None else int(hprom_max_its)

    # Load domain parameters from stage0 bundle
    emax = 2.0
    rel6 = [1.0, 0.05, 1.0, 0.05, 0.05, 0.05]
    domain_type = "box"

    bundle_path = "stage_0_trajectory/stage_0_trajectories.npz"
    if os.path.exists(bundle_path):
        data = np.load(bundle_path, allow_pickle=True)
        rel6 = list(data["relative_boundary"])
        if "emax" in data:
            emax = float(np.ravel(data["emax"])[0])
        else:
            emax = float(np.ravel(data["reference_amplitude"])[0])
        if "domain_type" in data:
            domain_type = str(data["domain_type"][0])

    control_points, waypoints = generate_safe_test_path(emax, rel6, domain_type)
    strain_path = np.array(waypoints, dtype=float)

    # Visualization of the test path in the current domain
    _plot_file = os.path.join(out_dir, "hprom_gpr_test_path.png")
    plot_path_in_domain(control_points, waypoints, emax, rel6, domain_type, _plot_file)

    seg_steps, _ = BuildDynamicSegmentSteps(
        strain_path,
        reference_steps=REFERENCE_STEPS_FOR_UNIT_AMPLITUDE,
        min_steps=MIN_STEPS_PER_SEGMENT,
        reference_amplitude=emax,
    )
    total_steps = int(np.sum(seg_steps))

    print("=" * 60)
    print("  Stage 10: HPROM-GPR Benchmark (FOM vs PROM-GPR vs HPROM-GPR)")
    print("=" * 60)
    print(f"  Waypoints: {len(control_points)}")
    print(f"  Strain path points: {len(strain_path)}")
    print(f"  Dynamic steps: {total_steps} (+1 initial = {total_steps + 1})")
    print(f"  Segments: {seg_steps}")
    print(f"  GPR data dir: {gpr_data_dir}")
    print(f"  HPROM-GPR dir: {hprom_gpr_dir}")
    print(f"  Output dir: {out_dir}")
    print(
        "  Newton knobs: "
        f"max_its={int(max_its)}, rel_tol={float(relnorm_cutoff):.1e}, "
        f"hprom_max_its={int(hprom_max_its_eff)}, "
        f"abs_tol={float(abs_res_cutoff):.1e}, dq_abs_tol={float(dq_abs_cutoff):.1e}, "
        f"res_floor={float(max_res_for_rel_convergence):.1e}, "
        f"damping_after_iter={int(damping_after_iter)}, damping_factor={float(damping_factor):.3f}"
    )

    mesh_fom_prom = "rve_geometry"
    mesh_hprom = "rve_geometry"
    auto_hrom_mesh = _load_hrom_mesh_base_from_ecm_dir(hprom_gpr_dir)
    if auto_hrom_mesh:
        mesh_hprom = str(auto_hrom_mesh)
    print(f"  FOM/PROM mesh: {mesh_fom_prom}")
    print(f"  HPROM mesh:    {mesh_hprom}")

    parameters = setup_kratos_parameters(mesh_fom_prom)
    parameters_hprom = setup_kratos_parameters(mesh_hprom)
    timings = {}

    fom_eps_file = os.path.join(out_dir, "fom_strain.npy")
    fom_sig_file = os.path.join(out_dir, "fom_stress.npy")
    if run_fom or not (os.path.exists(fom_eps_file) and os.path.exists(fom_sig_file)):
        print("\n[Stage 10] Running FOM...")
        t0 = time.perf_counter()
        f_eps, f_sig = RunFomBatchSimulation(
            parameters,
            out_dir=out_dir,
            strain_path=strain_path,
            trajectory_index=None,
            save_plot=False,
            reference_amplitude=emax,
            reference_steps=REFERENCE_STEPS_FOR_UNIT_AMPLITUDE,
        )
        timings["FOM"] = time.perf_counter() - t0
        f_eps = np.asarray(f_eps, dtype=float)
        f_sig = np.asarray(f_sig, dtype=float)
        np.save(fom_eps_file, f_eps)
        np.save(fom_sig_file, f_sig)
    else:
        print("\n[Stage 10] Loading cached FOM.")
        f_eps = np.load(fom_eps_file)
        f_sig = np.load(fom_sig_file)

    prom_eps_file = os.path.join(out_dir, "prom_gpr_strain.npy")
    prom_sig_file = os.path.join(out_dir, "prom_gpr_stress.npy")
    if run_prom_gpr or not (os.path.exists(prom_eps_file) and os.path.exists(prom_sig_file)):
        print("\n[Stage 10] Running PROM-GPR...")
        phi_p, phi_s, free_dofs, _, _, gpr_model, _include_macro = LoadPromGprModel(
            basis_dir="stage_2_pod_rve", gpr_data_dir=gpr_data_dir
        )
        t0 = time.perf_counter()
        p_eps, p_sig = RunPromGprBatchSimulation(
            parameters,
            phi_p,
            phi_s,
            free_dofs,
            gpr_model,
            strain_path,
            out_dir=out_dir,
            reference_amplitude=emax,
            reference_steps=REFERENCE_STEPS_FOR_UNIT_AMPLITUDE,
            qp_init_mode=qp_init_mode,
            max_its=max_its,
            relnorm_cutoff=relnorm_cutoff,
            abs_res_cutoff=abs_res_cutoff,
            dq_abs_cutoff=dq_abs_cutoff,
            max_res_for_rel_convergence=max_res_for_rel_convergence,
            min_rel_drop_stop=min_rel_drop_stop,
            stagnation_relnorm_gate=stagnation_relnorm_gate,
            max_dq_norm=max_dq_norm,
            damping_after_iter=damping_after_iter,
            damping_factor=damping_factor,
            old_stiffness_residual_cutoff=old_stiffness_residual_cutoff,
            regularization=regularization,
            use_old_stiffness_in_first_iteration=not bool(no_old_stiffness_first_it),
            verbose_iterations=bool(verbose_iterations),
        )
        timings["PROM-GPR"] = time.perf_counter() - t0
        p_eps = np.asarray(p_eps, dtype=float)
        p_sig = np.asarray(p_sig, dtype=float)
        np.save(prom_eps_file, p_eps)
        np.save(prom_sig_file, p_sig)
    else:
        print("\n[Stage 10] Loading cached PROM-GPR.")
        p_eps = np.load(prom_eps_file)
        p_sig = np.load(prom_sig_file)

    hprom_eps_file = os.path.join(out_dir, "hprom_gpr_strain.npy")
    hprom_sig_file = os.path.join(out_dir, "hprom_gpr_stress.npy")
    if run_hprom_gpr or not (os.path.exists(hprom_eps_file) and os.path.exists(hprom_sig_file)):
        print("\n[Stage 10] Running HPROM-GPR...")
        (
            phi_p_h,
            phi_s_h,
            free_dofs_h,
            _dir_dofs_h,
            eq_map_h,
            Xc_h,
            Yc_h,
            gpr_model_h,
            ecm_data_h,
            _include_macro_h,
        ) = LoadHpromGprModel(
            basis_dir="stage_2_pod_rve",
            gpr_data_dir=gpr_data_dir,
            hprom_gpr_dir=hprom_gpr_dir,
        )
        t0 = time.perf_counter()
        h_eps, h_sig = RunHpromGprBatchSimulation(
            parameters_hprom,
            phi_p_h,
            phi_s_h,
            free_dofs_h,
            gpr_model_h,
            ecm_data_h,
            out_dir=out_dir,
            strain_path=strain_path,
            trajectory_index=None,
            reference_amplitude=emax,
            reference_steps=REFERENCE_STEPS_FOR_UNIT_AMPLITUDE,
            eq_map_full=eq_map_h,
            Xc=Xc_h,
            Yc=Yc_h,
            qp_init_mode=qp_init_mode,
            max_its=hprom_max_its_eff,
            relnorm_cutoff=relnorm_cutoff,
            abs_res_cutoff=abs_res_cutoff,
            dq_abs_cutoff=dq_abs_cutoff,
            max_res_for_rel_convergence=max_res_for_rel_convergence,
            min_rel_drop_stop=min_rel_drop_stop,
            stagnation_relnorm_gate=stagnation_relnorm_gate,
            max_dq_norm=max_dq_norm,
            damping_after_iter=damping_after_iter,
            damping_factor=damping_factor,
            old_stiffness_residual_cutoff=old_stiffness_residual_cutoff,
            regularization=regularization,
            use_old_stiffness_in_first_iteration=not bool(no_old_stiffness_first_it),
            verbose_iterations=bool(verbose_iterations),
        )
        timings["HPROM-GPR"] = time.perf_counter() - t0
        h_eps = np.asarray(h_eps, dtype=float)
        h_sig = np.asarray(h_sig, dtype=float)
        np.save(hprom_eps_file, h_eps)
        np.save(hprom_sig_file, h_sig)
    else:
        print("\n[Stage 10] Loading cached HPROM-GPR.")
        h_eps = np.load(hprom_eps_file)
        h_sig = np.load(hprom_sig_file)

    stress_prom_fom = _aligned_error_metrics(f_sig, p_sig)
    stress_hprom_fom = _aligned_error_metrics(f_sig, h_sig)
    stress_hprom_prom = _aligned_error_metrics(p_sig, h_sig)
    strain_prom_fom = _aligned_error_metrics(f_eps, p_eps)
    strain_hprom_fom = _aligned_error_metrics(f_eps, h_eps)
    strain_hprom_prom = _aligned_error_metrics(p_eps, h_eps)

    prom_q_file = os.path.join(out_dir, "prom_gpr_run_q_p.npy")
    hprom_q_file = os.path.join(out_dir, "hprom_gpr_run_q_p.npy")
    q_hprom_prom = None
    if os.path.exists(prom_q_file) and os.path.exists(hprom_q_file):
        q_hprom_prom = _aligned_error_metrics(
            np.load(prom_q_file),
            np.load(hprom_q_file),
        )

    print("\n" + "=" * 60)
    print("  Stage 10 Summary")
    print("=" * 60)
    print(
        "  PROM-GPR  vs FOM: Rel. Stress Error = "
        f"{stress_prom_fom['rel_fro']:.4e}"
    )
    print(
        "  HPROM-GPR vs FOM: Rel. Stress Error = "
        f"{stress_hprom_fom['rel_fro']:.4e}"
    )
    print(
        "  HPROM-GPR vs PROM-GPR: Rel. Stress Error = "
        f"{stress_hprom_prom['rel_fro']:.4e}"
    )
    print("  Stress component relative errors [reference in denominator]:")
    print(
        "    PROM/FOM       : "
        + _format_component_errors(
            stress_prom_fom["per_component_rel"],
            ("sigma_xx", "sigma_yy", "sigma_xy"),
        )
    )
    print(
        "    HPROM/FOM      : "
        + _format_component_errors(
            stress_hprom_fom["per_component_rel"],
            ("sigma_xx", "sigma_yy", "sigma_xy"),
        )
    )
    print(
        "    HPROM/PROM     : "
        + _format_component_errors(
            stress_hprom_prom["per_component_rel"],
            ("sigma_xx", "sigma_yy", "sigma_xy"),
        )
    )
    print("  Strain component relative errors [reference in denominator]:")
    print(
        "    PROM/FOM       : "
        + _format_component_errors(
            strain_prom_fom["per_component_rel"],
            ("eps_xx", "eps_yy", "gamma_xy"),
        )
    )
    print(
        "    HPROM/FOM      : "
        + _format_component_errors(
            strain_hprom_fom["per_component_rel"],
            ("eps_xx", "eps_yy", "gamma_xy"),
        )
    )
    print(
        "    HPROM/PROM     : "
        + _format_component_errors(
            strain_hprom_prom["per_component_rel"],
            ("eps_xx", "eps_yy", "gamma_xy"),
        )
    )
    if q_hprom_prom is not None:
        q_labels = tuple(f"q_{j + 1}" for j in range(q_hprom_prom["k"]))
        print(
            "  HPROM-GPR vs PROM-GPR: Rel. q_m Error = "
            f"{q_hprom_prom['rel_fro']:.4e}"
        )
        print(
            "    q_m component relative errors: "
            + _format_component_errors(
                q_hprom_prom["per_component_rel"],
                q_labels,
            )
        )
        print(
            "    q_m RMSE / max |difference|: "
            f"{q_hprom_prom['rmse']:.4e} / "
            f"{q_hprom_prom['max_abs']:.4e}"
        )
    else:
        print("  HPROM/PROM q_m error unavailable: q_m history file missing.")
    for method, t in timings.items():
        print(f"  {method} time: {t:.2f}s")

    error_summary_file = os.path.join(out_dir, "stage10_error_summary.txt")
    with open(error_summary_file, "w", encoding="utf-8") as f:
        f.write(f"stress_rel_prom_vs_fom={stress_prom_fom['rel_fro']:.16e}\n")
        f.write(f"stress_rel_hprom_vs_fom={stress_hprom_fom['rel_fro']:.16e}\n")
        f.write(f"stress_rel_hprom_vs_prom={stress_hprom_prom['rel_fro']:.16e}\n")
        f.write(
            "stress_component_rel_prom_vs_fom="
            + ",".join(f"{float(v):.16e}" for v in stress_prom_fom["per_component_rel"])
            + "\n"
        )
        f.write(
            "stress_component_rel_hprom_vs_fom="
            + ",".join(f"{float(v):.16e}" for v in stress_hprom_fom["per_component_rel"])
            + "\n"
        )
        f.write(
            "stress_component_rel_hprom_vs_prom="
            + ",".join(f"{float(v):.16e}" for v in stress_hprom_prom["per_component_rel"])
            + "\n"
        )
        f.write(f"strain_rel_prom_vs_fom={strain_prom_fom['rel_fro']:.16e}\n")
        f.write(f"strain_rel_hprom_vs_fom={strain_hprom_fom['rel_fro']:.16e}\n")
        f.write(f"strain_rel_hprom_vs_prom={strain_hprom_prom['rel_fro']:.16e}\n")
        f.write(
            "strain_component_rel_prom_vs_fom="
            + ",".join(f"{float(v):.16e}" for v in strain_prom_fom["per_component_rel"])
            + "\n"
        )
        f.write(
            "strain_component_rel_hprom_vs_fom="
            + ",".join(f"{float(v):.16e}" for v in strain_hprom_fom["per_component_rel"])
            + "\n"
        )
        f.write(
            "strain_component_rel_hprom_vs_prom="
            + ",".join(f"{float(v):.16e}" for v in strain_hprom_prom["per_component_rel"])
            + "\n"
        )
        if q_hprom_prom is not None:
            f.write(f"q_m_rel_hprom_vs_prom={q_hprom_prom['rel_fro']:.16e}\n")
            f.write(f"q_m_rmse_hprom_vs_prom={q_hprom_prom['rmse']:.16e}\n")
            f.write(f"q_m_max_abs_hprom_vs_prom={q_hprom_prom['max_abs']:.16e}\n")
            f.write(
                "q_m_component_rel_hprom_vs_prom="
                + ",".join(f"{float(v):.16e}" for v in q_hprom_prom["per_component_rel"])
                + "\n"
            )
    print(f"  Error summary saved: {error_summary_file}")

    # mu vs q_p diagnostics (important for LS interpretation and initialization quality)
    try:
        mu_hist = _build_step_macro_strain_series(strain_path, seg_steps)
        if os.path.exists(prom_q_file):
            d_prom = _analyze_mu_vs_qp(mu_hist, np.load(prom_q_file))
            _save_mu_qp_diagnostics(d_prom, out_dir, label="PROM-GPR")
        else:
            print(f"  [PROM-GPR] mu-vs-q_p diagnostics skipped (missing {prom_q_file}).")
        if os.path.exists(hprom_q_file):
            d_hprom = _analyze_mu_vs_qp(mu_hist, np.load(hprom_q_file))
            _save_mu_qp_diagnostics(d_hprom, out_dir, label="HPROM-GPR")
        else:
            print(f"  [HPROM-GPR] mu-vs-q_p diagnostics skipped (missing {hprom_q_file}).")
    except Exception as ex:
        print(f"  [Stage 10] WARNING: mu-vs-q_p diagnostics failed: {ex}")

    plot_hprom_gpr_comparison(f_eps, f_sig, p_eps, p_sig, h_eps, h_sig, out_dir, timings)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Stage 10: HPROM-GPR benchmark")
    p.add_argument("--run-fom", action="store_true", help="Force FOM recompute.")
    p.add_argument("--run-prom-gpr", action="store_true", help="Force PROM-GPR recompute.")
    p.add_argument("--run-hprom-gpr", action="store_true", help="Force HPROM-GPR recompute.")
    p.add_argument(
        "--gpr-data-dir",
        type=str,
        default="stage_7_gpr_data",
        help="Directory with sparse-GP model files (sparse_gp_model.npz, phi_p.npy, phi_s.npy).",
    )
    p.add_argument(
        "--hprom-gpr-dir",
        type=str,
        default="stage_9_hprom_gpr_data",
        help="Directory with HPROM-GPR ECM file (ecm_weights_all.npz).",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="stage_10_hprom_gpr_results",
        help="Output directory for Stage 10 results.",
    )
    p.add_argument(
        "--qp-init-mode",
        type=str,
        default="previous",
        choices=["previous", "zero", "mu_affine"],
        help="Initializer for q_p at each increment in PROM/HPROM-GPR.",
    )
    p.add_argument("--max-its", type=int, default=25, help="Max reduced Newton corrector iterations.")
    p.add_argument(
        "--hprom-max-its",
        type=int,
        default=None,
        help=(
            "Max HPROM-GPR corrector iterations. If omitted, uses --max-its. "
            "Set to 0 for HPROM predictor-only/direct mode while keeping PROM-GPR iterative."
        ),
    )
    p.add_argument("--relnorm-cutoff", type=float, default=1e-5, help="Relative reduced residual tolerance.")
    p.add_argument("--abs-res-cutoff", type=float, default=1e-8, help="Absolute reduced residual tolerance.")
    p.add_argument("--dq-abs-cutoff", type=float, default=1e-6, help="Absolute reduced increment tolerance.")
    p.add_argument(
        "--max-res-for-rel-convergence",
        type=float,
        default=1e-1,
        help="Residual floor used with relative/stagnation criteria.",
    )
    p.add_argument("--min-rel-drop-stop", type=float, default=1e-2, help="Stagnation relative-drop threshold.")
    p.add_argument(
        "--stagnation-relnorm-gate",
        type=float,
        default=1e-4,
        help="Relative residual gate for stagnation acceptance.",
    )
    p.add_argument("--max-dq-norm", type=float, default=0.5, help="Clip threshold for ||dq||.")
    p.add_argument(
        "--damping-after-iter",
        type=int,
        default=10,
        help="Apply damping to reduced update for it > this iteration.",
    )
    p.add_argument(
        "--damping-factor",
        type=float,
        default=0.5,
        help="Reduced update damping factor after damping-after-iter.",
    )
    p.add_argument(
        "--old-stiffness-residual-cutoff",
        type=float,
        default=1e5,
        help="Use K_old on first iteration only when ||R_r|| is below this value.",
    )
    p.add_argument("--regularization", type=float, default=1e-10, help="L2 regularization used in reduced solve fallback.")
    p.add_argument(
        "--no-old-stiffness-first-it",
        action="store_true",
        help="Disable K_old acceleration in first corrector iteration.",
    )
    p.add_argument(
        "--verbose-iterations",
        action="store_true",
        help="Print extra per-iteration reduced-state diagnostics.",
    )
    args = p.parse_args()

    run_stage10(
        run_fom=args.run_fom,
        run_prom_gpr=args.run_prom_gpr,
        run_hprom_gpr=args.run_hprom_gpr,
        gpr_data_dir=args.gpr_data_dir,
        hprom_gpr_dir=args.hprom_gpr_dir,
        out_dir=args.out_dir,
        qp_init_mode=args.qp_init_mode,
        max_its=args.max_its,
        hprom_max_its=args.hprom_max_its,
        relnorm_cutoff=args.relnorm_cutoff,
        abs_res_cutoff=args.abs_res_cutoff,
        dq_abs_cutoff=args.dq_abs_cutoff,
        max_res_for_rel_convergence=args.max_res_for_rel_convergence,
        min_rel_drop_stop=args.min_rel_drop_stop,
        stagnation_relnorm_gate=args.stagnation_relnorm_gate,
        max_dq_norm=args.max_dq_norm,
        damping_after_iter=args.damping_after_iter,
        damping_factor=args.damping_factor,
        old_stiffness_residual_cutoff=args.old_stiffness_residual_cutoff,
        regularization=args.regularization,
        no_old_stiffness_first_it=args.no_old_stiffness_first_it,
        verbose_iterations=args.verbose_iterations,
    )
