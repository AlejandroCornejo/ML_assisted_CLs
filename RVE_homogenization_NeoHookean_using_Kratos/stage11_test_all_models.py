#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage 11: unified benchmark on one trajectory with selectable solvers.
Default compares all available families:
  KRATOS, FOM, PROM, HPROM, PROM-DL, PROM-ANN, HPROM-ANN, PROM-RBF, HPROM-RBF
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt

from stage6_test_hprom import generate_safe_test_path
from stage4_test_rve import plot_path_in_domain
from fom_solver_rve import (
    setup_kratos_parameters,
    RunFomBatchSimulation,
    BuildDynamicSegmentSteps,
    REFERENCE_STEPS_FOR_UNIT_AMPLITUDE,
    MIN_STEPS_PER_SEGMENT,
)
from kratos_solver_rve import RunKratosBatchSimulation
from prom_solver_rve import RunPromBatchSimulation
from hprom_solver_rve import RunHpromBatchSimulation
from prom_dl_solver_rve import LoadPromDlModel, RunPromDlBatchSimulation
from prom_ann_solver_rve import LoadPromAnnModel, RunPromAnnBatchSimulation
from hprom_ann_solver_rve import LoadHpromAnnModel, RunHpromAnnBatchSimulation
from prom_rbf_solver_rve import LoadPromRbfModel, RunPromRbfBatchSimulation
from hprom_rbf_solver_rve import LoadHpromRbfModel, RunHpromRbfBatchSimulation


AVAILABLE_METHODS = [
    "KRATOS",
    "FOM",
    "PROM",
    "HPROM",
    "PROM-DL",
    "PROM-ANN",
    "HPROM-ANN",
    "PROM-RBF",
    "HPROM-RBF",
]


def _ordered_methods(keys):
    keys_set = set(keys)
    return [m for m in AVAILABLE_METHODS if m in keys_set]


def _load_linear_rom_model(model_dir="stage_2_pod_rve"):
    phi_f = np.load(os.path.join(model_dir, "pod_basis_free.npy"))
    free_dofs = np.load(os.path.join(model_dir, "free_dofs.npy"))
    dir_dofs = np.load(os.path.join(model_dir, "dirichlet_dofs.npy"))
    eq_map = np.load(os.path.join(model_dir, "eq_map.npy"))
    Xc, Yc = np.load(os.path.join(model_dir, "domain_center.npy"))
    return phi_f, free_dofs, dir_dofs, eq_map, Xc, Yc


def _load_linear_hprom_data(hprom_dir="stage_5_hprom_data"):
    ecm = np.load(os.path.join(hprom_dir, "ecm_weights_all.npz"))
    return {k: ecm[k] for k in ecm.files}


def _rel_l2_vs_ref(sig_ref, sig_pred):
    n = min(len(sig_ref), len(sig_pred))
    num = np.linalg.norm(sig_ref[:n] - sig_pred[:n])
    den = np.linalg.norm(sig_ref[:n]) + 1e-30
    return num / den, n


def _save_timing_cache(path, timings):
    if not timings:
        return
    np.savez(path, **{k: np.array([float(v)], dtype=float) for k, v in timings.items()})


def _load_timing_cache(path):
    if not os.path.exists(path):
        return {}
    data = np.load(path)
    out = {}
    for k in data.files:
        out[k] = float(np.ravel(data[k])[0])
    return out


def _plot_stage11(all_eps, all_sig, out_dir, timings, reference_method):
    plt.rcParams.update(
        {
            "font.size": 12,
            "mathtext.fontset": "stix",
            "font.family": "STIXGeneral",
        }
    )

    methods_order = _ordered_methods(all_sig.keys())
    if not methods_order:
        print("[Stage 11] No methods available to plot.")
        return

    styles = {
        "KRATOS": ("-", 1.7, "c"),
        "FOM": ("-", 2.2, "k"),
        "PROM": ("--", 1.5, "r"),
        "HPROM": (":", 1.8, "b"),
        "PROM-DL": ("-", 1.4, "y"),
        "PROM-ANN": ("-", 1.6, "tab:orange"),
        "HPROM-ANN": ("-", 1.6, "tab:brown"),
        "PROM-RBF": ("-.", 1.6, "g"),
        "HPROM-RBF": ("-", 1.2, "m"),
    }

    comp_labels = [
        (0, r"$\sigma_{xx}$", r"$\varepsilon_{xx}$", "sigma_xx"),
        (1, r"$\sigma_{yy}$", r"$\varepsilon_{yy}$", "sigma_yy"),
        (2, r"$\sigma_{xy}$", r"$\gamma_{xy}$", "sigma_xy"),
    ]

    n_common = min(len(all_sig[m]) for m in methods_order)
    n_common = min(n_common, min(len(all_eps[m]) for m in methods_order))

    for i, label_sig, label_eps, suffix in comp_labels:
        plt.figure(figsize=(7.2, 6.0))
        for m in methods_order:
            ls, lw, color = styles.get(m, ("-", 1.4, None))
            plt.plot(
                all_eps[m][:n_common, i],
                all_sig[m][:n_common, i],
                linestyle=ls,
                color=color,
                linewidth=lw,
                label=m,
            )
        plt.title(f"Stage 11 Benchmark: {label_sig}")
        plt.xlabel(f"{label_eps} [-]")
        plt.ylabel(f"{label_sig} [Pa]")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"stage11_comp_{suffix}.png"), dpi=150)
        plt.close()

    if reference_method in all_sig and len(methods_order) > 1:
        ref_norm = np.linalg.norm(all_sig[reference_method][:n_common], axis=1) + 1e-30
        plt.figure(figsize=(7.2, 6.0))
        for m in methods_order:
            if m == reference_method:
                continue
            err = np.linalg.norm(all_sig[reference_method][:n_common] - all_sig[m][:n_common], axis=1) / ref_norm
            ls, lw, color = styles.get(m, ("-", 1.4, None))
            plt.plot(err, linestyle=ls, color=color, linewidth=lw, label=f"{m} error")
        plt.title(f"Stage 11: Relative Stress Error vs {reference_method}")
        plt.xlabel("Step")
        plt.ylabel("Relative Error [-]")
        plt.yscale("log")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "stage11_error_vs_step.png"), dpi=150)
        plt.close()

    timing_methods = [m for m in methods_order if m in timings]
    if timing_methods:
        plt.figure(figsize=(7.2, 6.0))
        vals = [timings[m] for m in timing_methods]
        colors = {
            "KRATOS": "c",
            "FOM": "gray",
            "PROM": "red",
            "HPROM": "blue",
            "PROM-DL": "gold",
            "PROM-ANN": "tab:orange",
            "HPROM-ANN": "tab:brown",
            "PROM-RBF": "green",
            "HPROM-RBF": "magenta",
        }
        bars = plt.bar(
            timing_methods,
            vals,
            color=[colors.get(m, "tab:gray") for m in timing_methods],
            alpha=0.85,
        )
        for bar, t in zip(bars, vals):
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height(),
                f"{t:.1f}s",
                ha="center",
                va="bottom",
            )
        plt.title("Stage 11: Wall-Clock Time")
        plt.ylabel("Time [s]")
        plt.grid(True, axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "stage11_timing.png"), dpi=150)
        plt.close()

    print(f"[Stage 11] Plots saved to: {out_dir}")


def _attempt_model_run(method, run_callable, strict=False):
    try:
        return True, run_callable()
    except Exception as exc:
        print(f"[Stage 11] WARNING: {method} failed and will be skipped: {exc}")
        if strict:
            raise
        return False, None


def run_stage11(
    methods=None,
    run_kratos=False,
    run_fom=False,
    run_prom=False,
    run_hprom=False,
    run_prom_dl=False,
    run_prom_ann=False,
    run_hprom_ann=False,
    run_prom_rbf=False,
    run_hprom_rbf=False,
    step_reference_amplitude=None,
    use_bundle_reference_amplitude=False,
    strict=False,
):
    out_dir = "stage_11_all_models_results"
    os.makedirs(out_dir, exist_ok=True)

    requested_methods = AVAILABLE_METHODS if methods is None else list(methods)
    unknown = [m for m in requested_methods if m not in AVAILABLE_METHODS]
    if unknown:
        raise ValueError(f"Unknown methods requested: {unknown}")

    # Load domain parameters from stage0 bundle
    emax = 2.0
    reference_amplitude_bundle = emax
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
        if "reference_amplitude" in data:
            reference_amplitude_bundle = float(np.ravel(data["reference_amplitude"])[0])
        else:
            reference_amplitude_bundle = emax
        if "domain_type" in data:
            domain_type = str(np.ravel(data["domain_type"])[0])

    # Keep Stage 11 consistent with Stage 8/10 by default:
    # use emax for dynamic-step scaling unless explicitly overridden.
    if step_reference_amplitude is not None:
        reference_amplitude = float(step_reference_amplitude)
    elif use_bundle_reference_amplitude:
        reference_amplitude = float(reference_amplitude_bundle)
    else:
        reference_amplitude = float(emax)

    control_points, waypoints = generate_safe_test_path(emax, rel6, domain_type)
    strain_path = np.array(waypoints, dtype=float)

    # Visualization of the test path in the current domain
    _plot_file = os.path.join(out_dir, "stage11_test_path.png")
    plot_path_in_domain(control_points, waypoints, emax, rel6, domain_type, _plot_file)

    seg_steps, _ = BuildDynamicSegmentSteps(
        strain_path,
        reference_steps=REFERENCE_STEPS_FOR_UNIT_AMPLITUDE,
        min_steps=MIN_STEPS_PER_SEGMENT,
        reference_amplitude=reference_amplitude,
    )
    total_steps = int(np.sum(seg_steps))

    print("=" * 70)
    print("  Stage 11: Unified Benchmark (selectable solver families)")
    print("=" * 70)
    print(f"  Methods requested: {requested_methods}")
    print(f"  Waypoints: {len(control_points)}")
    print(f"  Path points: {len(strain_path)}")
    print(f"  Dynamic steps: {total_steps} (+1 initial = {total_steps + 1})")
    print(f"  emax: {emax}")
    print(f"  Bundle reference amplitude: {reference_amplitude_bundle}")
    print(f"  Step-control reference amplitude (used): {reference_amplitude}")
    print(f"  Segments: {seg_steps}")

    parameters = setup_kratos_parameters("rve_geometry")

    # Cache files
    cache = {
        "KRATOS": (
            os.path.join(out_dir, "kratos_strain.npy"),
            os.path.join(out_dir, "kratos_stress.npy"),
        ),
        "FOM": (
            os.path.join(out_dir, "fom_strain.npy"),
            os.path.join(out_dir, "fom_stress.npy"),
        ),
        "PROM": (
            os.path.join(out_dir, "prom_strain.npy"),
            os.path.join(out_dir, "prom_stress.npy"),
        ),
        "HPROM": (
            os.path.join(out_dir, "hprom_strain.npy"),
            os.path.join(out_dir, "hprom_stress.npy"),
        ),
        "PROM-DL": (
            os.path.join(out_dir, "prom_dl_strain.npy"),
            os.path.join(out_dir, "prom_dl_stress.npy"),
        ),
        "PROM-ANN": (
            os.path.join(out_dir, "prom_ann_strain.npy"),
            os.path.join(out_dir, "prom_ann_stress.npy"),
        ),
        "HPROM-ANN": (
            os.path.join(out_dir, "hprom_ann_strain.npy"),
            os.path.join(out_dir, "hprom_ann_stress.npy"),
        ),
        "PROM-RBF": (
            os.path.join(out_dir, "prom_rbf_strain.npy"),
            os.path.join(out_dir, "prom_rbf_stress.npy"),
        ),
        "HPROM-RBF": (
            os.path.join(out_dir, "hprom_rbf_strain.npy"),
            os.path.join(out_dir, "hprom_rbf_stress.npy"),
        ),
    }

    force_map = {
        "KRATOS": bool(run_kratos),
        "FOM": bool(run_fom),
        "PROM": bool(run_prom),
        "HPROM": bool(run_hprom),
        "PROM-DL": bool(run_prom_dl),
        "PROM-ANN": bool(run_prom_ann),
        "HPROM-ANN": bool(run_hprom_ann),
        "PROM-RBF": bool(run_prom_rbf),
        "HPROM-RBF": bool(run_hprom_rbf),
    }

    timings_new = {}
    timings_file = os.path.join(out_dir, "stage11_timings.npz")
    timings_all = _load_timing_cache(timings_file)

    all_eps = {}
    all_sig = {}

    def _run_or_load(method, run_block):
        eps_file, sig_file = cache[method]
        force = force_map.get(method, False)
        if force or not (os.path.exists(eps_file) and os.path.exists(sig_file)):
            ok, result = _attempt_model_run(method, run_block, strict=strict)
            if not ok:
                return
            eps_arr, sig_arr, elapsed = result
            timings_new[method] = elapsed
            np.save(eps_file, eps_arr)
            np.save(sig_file, sig_arr)
            all_eps[method], all_sig[method] = eps_arr, sig_arr
        else:
            print(f"\n[Stage 11] Loading cached {method}.")
            all_eps[method] = np.asarray(np.load(eps_file), dtype=float)
            all_sig[method] = np.asarray(np.load(sig_file), dtype=float)

    if "KRATOS" in requested_methods:
        def _run_kratos():
            print("\n[Stage 11] Running KRATOS...")
            t0 = time.perf_counter()
            eps, sig, _, _ = RunKratosBatchSimulation(
                parameters,
                out_dir=out_dir,
                save_plot=False,
                save_data=False,
                strain_path=strain_path,
                trajectory_index=None,
                reference_amplitude=reference_amplitude,
                reference_steps=REFERENCE_STEPS_FOR_UNIT_AMPLITUDE,
            )
            return np.asarray(eps, dtype=float), np.asarray(sig, dtype=float), time.perf_counter() - t0

        _run_or_load("KRATOS", _run_kratos)

    if "FOM" in requested_methods:
        def _run_fom():
            print("\n[Stage 11] Running FOM...")
            t0 = time.perf_counter()
            eps, sig = RunFomBatchSimulation(
                parameters,
                out_dir=out_dir,
                strain_path=strain_path,
                trajectory_index=None,
                save_plot=False,
                reference_amplitude=reference_amplitude,
                reference_steps=REFERENCE_STEPS_FOR_UNIT_AMPLITUDE,
            )
            return np.asarray(eps, dtype=float), np.asarray(sig, dtype=float), time.perf_counter() - t0

        _run_or_load("FOM", _run_fom)

    if "PROM" in requested_methods:
        def _run_prom():
            print("\n[Stage 11] Running PROM...")
            phi_f, free_dofs, dir_dofs, eq_map, Xc, Yc = _load_linear_rom_model("stage_2_pod_rve")
            t0 = time.perf_counter()
            eps, sig = RunPromBatchSimulation(
                parameters,
                phi_f,
                free_dofs,
                dir_dofs,
                eq_map,
                Xc,
                Yc,
                out_dir=out_dir,
                save_plot=False,
                strain_path=strain_path,
                trajectory_index=None,
                reference_amplitude=reference_amplitude,
                reference_steps=REFERENCE_STEPS_FOR_UNIT_AMPLITUDE,
            )
            return np.asarray(eps, dtype=float), np.asarray(sig, dtype=float), time.perf_counter() - t0

        _run_or_load("PROM", _run_prom)

    if "HPROM" in requested_methods:
        def _run_hprom():
            print("\n[Stage 11] Running HPROM...")
            phi_f, free_dofs, dir_dofs, eq_map, Xc, Yc = _load_linear_rom_model("stage_2_pod_rve")
            ecm_data = _load_linear_hprom_data("stage_5_hprom_data")
            t0 = time.perf_counter()
            eps, sig = RunHpromBatchSimulation(
                parameters,
                phi_f,
                free_dofs,
                dir_dofs,
                eq_map,
                Xc,
                Yc,
                ecm_data=ecm_data,
                out_dir=out_dir,
                save_plot=False,
                strain_path=strain_path,
                trajectory_index=None,
                reference_amplitude=reference_amplitude,
                reference_steps=REFERENCE_STEPS_FOR_UNIT_AMPLITUDE,
            )
            return np.asarray(eps, dtype=float), np.asarray(sig, dtype=float), time.perf_counter() - t0

        _run_or_load("HPROM", _run_hprom)

    if "PROM-DL" in requested_methods:
        def _run_prom_dl():
            print("\n[Stage 11] Running PROM-DL...")
            phi_q, free_dofs, _, _, pod_dl_model, device, _ = LoadPromDlModel(
                basis_dir="stage_2_pod_rve",
                pod_dl_data_dir="stage_7_pod_dl_data",
            )
            t0 = time.perf_counter()
            eps, sig = RunPromDlBatchSimulation(
                parameters,
                phi_q,
                free_dofs,
                pod_dl_model,
                device,
                strain_path,
                reference_amplitude=reference_amplitude,
                reference_steps=REFERENCE_STEPS_FOR_UNIT_AMPLITUDE,
            )
            return np.asarray(eps, dtype=float), np.asarray(sig, dtype=float), time.perf_counter() - t0

        _run_or_load("PROM-DL", _run_prom_dl)

    if "PROM-ANN" in requested_methods:
        def _run_prom_ann():
            print("\n[Stage 11] Running PROM-ANN...")
            phi_p, phi_s, free_dofs, _, _, ann_model, device, include_macro = LoadPromAnnModel(
                basis_dir="stage_2_pod_rve",
                ann_data_dir="stage_7_ann_data",
            )
            t0 = time.perf_counter()
            eps, sig = RunPromAnnBatchSimulation(
                parameters,
                phi_p,
                phi_s,
                free_dofs,
                ann_model,
                device,
                strain_path,
                include_macro_strain_input=include_macro,
                reference_amplitude=reference_amplitude,
                reference_steps=REFERENCE_STEPS_FOR_UNIT_AMPLITUDE,
            )
            return np.asarray(eps, dtype=float), np.asarray(sig, dtype=float), time.perf_counter() - t0

        _run_or_load("PROM-ANN", _run_prom_ann)

    if "HPROM-ANN" in requested_methods:
        def _run_hprom_ann():
            print("\n[Stage 11] Running HPROM-ANN...")
            (
                phi_p,
                phi_s,
                free_dofs,
                _,
                _,
                ann_model,
                device,
                ecm_data,
                include_macro,
            ) = LoadHpromAnnModel(
                basis_dir="stage_2_pod_rve",
                ann_data_dir="stage_7_ann_data",
                hprom_ann_dir="stage_9_hprom_ann_data",
            )
            t0 = time.perf_counter()
            eps, sig = RunHpromAnnBatchSimulation(
                parameters,
                phi_p,
                phi_s,
                free_dofs,
                ann_model,
                device,
                ecm_data,
                out_dir=out_dir,
                strain_path=strain_path,
                trajectory_index=None,
                include_macro_strain_input=include_macro,
                reference_amplitude=reference_amplitude,
                reference_steps=REFERENCE_STEPS_FOR_UNIT_AMPLITUDE,
            )
            return np.asarray(eps, dtype=float), np.asarray(sig, dtype=float), time.perf_counter() - t0

        _run_or_load("HPROM-ANN", _run_hprom_ann)

    if "PROM-RBF" in requested_methods:
        def _run_prom_rbf():
            print("\n[Stage 11] Running PROM-RBF...")
            phi_p, phi_s, free_dofs, _, _, rbf_model, include_macro = LoadPromRbfModel(
                basis_dir="stage_2_pod_rve",
                rbf_data_dir="stage_7_rbf_data",
            )
            t0 = time.perf_counter()
            eps, sig = RunPromRbfBatchSimulation(
                parameters,
                phi_p,
                phi_s,
                free_dofs,
                rbf_model,
                strain_path,
                include_macro_strain_input=include_macro,
                reference_amplitude=reference_amplitude,
                reference_steps=REFERENCE_STEPS_FOR_UNIT_AMPLITUDE,
            )
            return np.asarray(eps, dtype=float), np.asarray(sig, dtype=float), time.perf_counter() - t0

        _run_or_load("PROM-RBF", _run_prom_rbf)

    if "HPROM-RBF" in requested_methods:
        def _run_hprom_rbf():
            print("\n[Stage 11] Running HPROM-RBF...")
            (
                phi_p,
                phi_s,
                free_dofs,
                _,
                _,
                _,
                _,
                rbf_model,
                ecm_data,
                include_macro,
            ) = LoadHpromRbfModel(
                basis_dir="stage_2_pod_rve",
                rbf_data_dir="stage_7_rbf_data",
                hprom_rbf_dir="stage_9_hprom_rbf_data",
            )
            t0 = time.perf_counter()
            eps, sig = RunHpromRbfBatchSimulation(
                parameters,
                phi_p,
                phi_s,
                free_dofs,
                rbf_model,
                ecm_data,
                out_dir=out_dir,
                strain_path=strain_path,
                trajectory_index=None,
                include_macro_strain_input=include_macro,
                reference_amplitude=reference_amplitude,
                reference_steps=REFERENCE_STEPS_FOR_UNIT_AMPLITUDE,
            )
            return np.asarray(eps, dtype=float), np.asarray(sig, dtype=float), time.perf_counter() - t0

        _run_or_load("HPROM-RBF", _run_hprom_rbf)

    if not all_sig:
        raise RuntimeError("Stage 11 did not produce any solver results.")

    timings_all.update(timings_new)
    _save_timing_cache(timings_file, timings_all)

    ordered = _ordered_methods(all_sig.keys())
    if "FOM" in ordered:
        reference_method = "FOM"
    elif "KRATOS" in ordered:
        reference_method = "KRATOS"
    else:
        reference_method = ordered[0]

    errors = {}
    for m in ordered:
        if m == reference_method:
            continue
        errors[m] = _rel_l2_vs_ref(all_sig[reference_method], all_sig[m])

    print("\n" + "=" * 70)
    print("  Stage 11 Summary")
    print("=" * 70)
    print(f"  Reference method: {reference_method}")
    for m in ordered:
        if m == reference_method:
            continue
        err_val, n_val = errors[m]
        print(f"  {m:10s} vs {reference_method}: Rel. Stress Error = {err_val:.4e}  (n={n_val})")
    for m in ordered:
        if m in timings_all:
            src = "new" if m in timings_new else "cached"
            print(f"  {m:10s} time: {timings_all[m]:.2f}s  [{src}]")

    summary_path = os.path.join(out_dir, "stage11_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Stage 11 Unified Benchmark Summary\n")
        f.write("=" * 48 + "\n")
        f.write(f"reference_method: {reference_method}\n")
        f.write(f"methods_requested: {requested_methods}\n")
        f.write(f"methods_available: {ordered}\n")
        f.write(f"steps_total: {total_steps}\n")
        f.write(f"entries_total: {total_steps + 1}\n")
        for m in ordered:
            if m == reference_method:
                continue
            err_val, n_val = errors[m]
            f.write(f"error_{m}_vs_{reference_method}: {err_val:.12e}\n")
            f.write(f"n_{m}_vs_{reference_method}: {n_val}\n")
        for m in ordered:
            if m in timings_all:
                f.write(f"time_{m}: {timings_all[m]:.12e}\n")

    _plot_stage11(all_eps, all_sig, out_dir, timings_all, reference_method)
    print(f"[Stage 11] Summary saved to: {summary_path}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Stage 11: unified all-model benchmark")
    p.add_argument(
        "--methods",
        nargs="+",
        choices=AVAILABLE_METHODS,
        default=AVAILABLE_METHODS,
        help="Subset of methods to include in Stage 11 compare-all run.",
    )
    p.add_argument("--run-all", action="store_true", help="Force recomputation of all selected methods.")
    p.add_argument("--run-kratos", action="store_true", help="Force recomputation of KRATOS.")
    p.add_argument("--run-fom", action="store_true", help="Force recomputation of FOM.")
    p.add_argument("--run-prom", action="store_true", help="Force recomputation of PROM.")
    p.add_argument("--run-hprom", action="store_true", help="Force recomputation of HPROM.")
    p.add_argument("--run-prom-dl", action="store_true", help="Force recomputation of PROM-DL.")
    p.add_argument("--run-prom-ann", action="store_true", help="Force recomputation of PROM-ANN.")
    p.add_argument("--run-hprom-ann", action="store_true", help="Force recomputation of HPROM-ANN.")
    p.add_argument("--run-prom-rbf", action="store_true", help="Force recomputation of PROM-RBF.")
    p.add_argument("--run-hprom-rbf", action="store_true", help="Force recomputation of HPROM-RBF.")
    p.add_argument("--strict", action="store_true", help="Fail fast if any selected solver errors.")
    args = p.parse_args()

    if args.run_all:
        if "KRATOS" in args.methods:
            args.run_kratos = True
        if "FOM" in args.methods:
            args.run_fom = True
        if "PROM" in args.methods:
            args.run_prom = True
        if "HPROM" in args.methods:
            args.run_hprom = True
        if "PROM-DL" in args.methods:
            args.run_prom_dl = True
        if "PROM-ANN" in args.methods:
            args.run_prom_ann = True
        if "HPROM-ANN" in args.methods:
            args.run_hprom_ann = True
        if "PROM-RBF" in args.methods:
            args.run_prom_rbf = True
        if "HPROM-RBF" in args.methods:
            args.run_hprom_rbf = True

    run_stage11(
        methods=args.methods,
        run_kratos=args.run_kratos,
        run_fom=args.run_fom,
        run_prom=args.run_prom,
        run_hprom=args.run_hprom,
        run_prom_dl=args.run_prom_dl,
        run_prom_ann=args.run_prom_ann,
        run_hprom_ann=args.run_hprom_ann,
        run_prom_rbf=args.run_prom_rbf,
        run_hprom_rbf=args.run_hprom_rbf,
        strict=args.strict,
    )
