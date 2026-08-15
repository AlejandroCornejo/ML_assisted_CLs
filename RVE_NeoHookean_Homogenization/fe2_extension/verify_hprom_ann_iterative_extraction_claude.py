#!/usr/bin/env python3
"""Stage 2 verification: does HpromAnnIterativeLaw reproduce the
ORIGINAL, already-saved iterative HPROM-ANN online output on the
held-out test trajectory?

Primary (pass/fail) target: hprom/ann/stage_10_results_maw_dynamic/
hprom_ann_strain.npy and hprom_ann_stress.npy, the raw saved h_eps/h_sig
history from RunHpromAnnBatchSimulation's iterative (Newton-corrected)
mode. That reference run used qp_init_mode="continuation" (run_stage10's
own default), so this verification also uses "continuation" and warm-
starts q_p sequentially across steps, exactly like the original loop.

Secondary, informational: also runs the SAME held-out path with
qp_init_mode="mu_affine" (no warm-starting, no path dependency across
Gauss points) -- not expected to hit the primary tolerance, but should
converge to physically close values, since this is the config we
actually want for Stage 3 (each Cook's-membrane Gauss point queried
independently, no per-point state to carry).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for sub in ("core", "prom/ann", "hprom/ann"):
    p = str(ROOT / sub)
    if p not in sys.path:
        sys.path.insert(0, p)

from fom_solver_rve import BuildDynamicSegmentSteps, REFERENCE_STEPS_FOR_UNIT_AMPLITUDE, MIN_STEPS_PER_SEGMENT
from stage6_test_hprom import generate_safe_test_path

from hprom_ann_iterative_law_claude import HpromAnnIterativeLaw

REFERENCE_DIR = ROOT / "hprom" / "ann" / "stage_10_results_maw_dynamic"


def build_held_out_strain_path():
    emax = 2.0
    rel6 = [1.0, 0.05, 1.0, 0.05, 0.05, 0.05]
    domain_type = "box"
    bundle_path = ROOT / "trajectories" / "stage_0_trajectory" / "stage_0_trajectories.npz"
    if bundle_path.exists():
        data = np.load(bundle_path, allow_pickle=True)
        rel6 = list(data["relative_boundary"])
        emax = float(np.ravel(data["emax"])[0]) if "emax" in data else float(np.ravel(data["reference_amplitude"])[0])
        if "domain_type" in data:
            domain_type = str(data["domain_type"][0])
    _control_points, full_path = generate_safe_test_path(emax, rel6, domain_type)
    return np.array(full_path, dtype=float), emax


def densified_per_step_strains(strain_path, emax):
    E_wp = np.asarray(strain_path, dtype=float)
    n_seg = len(E_wp) - 1
    seg_steps, _ = BuildDynamicSegmentSteps(
        E_wp, reference_steps=REFERENCE_STEPS_FOR_UNIT_AMPLITUDE,
        min_steps=MIN_STEPS_PER_SEGMENT, reference_amplitude=emax,
    )
    step_offsets = np.concatenate(([0], np.cumsum(seg_steps)))
    n_steps_total = int(step_offsets[-1])
    E_per_step = np.zeros((n_steps_total, 3), dtype=float)
    for step in range(1, n_steps_total + 1):
        s = int(np.searchsorted(step_offsets, step, side="left") - 1)
        s = max(0, min(s, n_seg - 1))
        xi = float(step - step_offsets[s]) / float(max(seg_steps[s], 1))
        E_per_step[step - 1] = (1.0 - xi) * E_wp[s, :] + xi * E_wp[s + 1, :]
    return E_per_step


def relative_l2(pred, ref):
    pred, ref = np.asarray(pred, dtype=float), np.asarray(ref, dtype=float)
    return float(np.linalg.norm(pred - ref) / max(np.linalg.norm(ref), 1e-30))


def run_sequential(law, E_per_step, qp_init_mode_label):
    n = E_per_step.shape[0]
    eps_out = np.zeros_like(E_per_step)
    sig_out = np.zeros_like(E_per_step)
    q_prev = np.zeros(law.n_primary, dtype=float)
    iters_hist = np.zeros(n, dtype=int)
    n_nonconverged = 0
    for i, E in enumerate(E_per_step):
        eps_out[i], sig_out[i], q_prev, n_it, converged = law.evaluate(E, q_prev=q_prev, step_index=i + 1)
        iters_hist[i] = n_it
        if not converged:
            n_nonconverged += 1
        if (i + 1) % 200 == 0:
            print(f"    [{qp_init_mode_label}] ... {i + 1}/{n} (mean iters so far={iters_hist[:i + 1].mean():.2f})")
    print(f"    [{qp_init_mode_label}] done: mean iters={iters_hist.mean():.2f}, "
          f"max iters={iters_hist.max()}, non-converged steps={n_nonconverged}/{n}")
    return eps_out, sig_out


def main():
    ref_strain = np.load(REFERENCE_DIR / "hprom_ann_strain.npy")
    ref_stress = np.load(REFERENCE_DIR / "hprom_ann_stress.npy")
    print(f"[verify] loaded reference: strain={ref_strain.shape}, stress={ref_stress.shape}")

    strain_path, emax = build_held_out_strain_path()
    E_per_step = densified_per_step_strains(strain_path, emax)
    print(f"[verify] rebuilt {E_per_step.shape[0]} densified per-step strains from the held-out path")

    print("\n[verify] PRIMARY: building HpromAnnIterativeLaw with qp_init_mode='continuation' ...")
    law_cont = HpromAnnIterativeLaw(qp_init_mode="continuation")
    eps_cont, sig_cont = run_sequential(law_cont, E_per_step, "continuation")

    n = eps_cont.shape[0]
    rel_eps = relative_l2(eps_cont, ref_strain[1:n + 1])
    rel_sig = relative_l2(sig_cont, ref_stress[1:n + 1])
    print(f"\n[verify] PRIMARY comparison (vs. real online hprom_ann_*.npy, continuation mode):")
    print(f"[verify]   relative L2: eps={rel_eps:.3e}, sig={rel_sig:.3e}")
    tol = 1e-4
    primary_pass = rel_eps < tol and rel_sig < tol
    print(f"[verify] {'PASS' if primary_pass else 'FAIL'}: {'within' if primary_pass else 'exceeds'} {tol:.0e}.")

    np.save(HERE / "stage2_extracted_eps_continuation.npy", eps_cont)
    np.save(HERE / "stage2_extracted_sig_continuation.npy", sig_cont)

    print("\n[verify] SECONDARY (informational): qp_init_mode='mu_affine', no warm-starting ...")
    law_aff = HpromAnnIterativeLaw(qp_init_mode="mu_affine")
    eps_aff, sig_aff = run_sequential(law_aff, E_per_step, "mu_affine")
    rel_eps_aff = relative_l2(eps_aff, ref_strain[1:n + 1])
    rel_sig_aff = relative_l2(sig_aff, ref_stress[1:n + 1])
    rel_eps_vs_cont = relative_l2(eps_aff, eps_cont)
    rel_sig_vs_cont = relative_l2(sig_aff, sig_cont)
    print(f"[verify]   mu_affine vs. real online reference: eps={rel_eps_aff:.3e}, sig={rel_sig_aff:.3e}")
    print(f"[verify]   mu_affine vs. THIS RUN's continuation-mode result: eps={rel_eps_vs_cont:.3e}, sig={rel_sig_vs_cont:.3e}")
    print("[verify]   (informational only -- mu_affine is the memoryless config Stage 3 will actually use)")

    np.save(HERE / "stage2_extracted_eps_mu_affine.npy", eps_aff)
    np.save(HERE / "stage2_extracted_sig_mu_affine.npy", sig_aff)


if __name__ == "__main__":
    main()
