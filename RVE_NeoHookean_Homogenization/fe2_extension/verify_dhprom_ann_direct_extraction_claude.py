#!/usr/bin/env python3
"""Stage 1 verification: does DHpromAnnDirectLaw (dhprom_ann_direct_law_claude.py)
reproduce the ORIGINAL, already-saved D-HPROM-ANN online output on the
held-out test trajectory?

Primary (pass/fail) target: hprom/ann/stage_10_results_maw_dynamic/
dhprom_ann_stress.npy and dhprom_ann_strain.npy -- the actual saved
online hom_sig/hom_eps history from RunHpromAnnBatchSimulation itself.
This is a pure refactor, so agreement should be near machine precision.

Secondary (informational only) comparison: pann/data/
hprom_ann_direct_stage10_metrics.npz's dhprom_ann_direct_stress -- a
DIFFERENT, reaction-force-recomputed stress convention (see
pann/direct_energy/hprom_ann_direct_stress.py), expected to differ at
the paper's usual ~1e-3-1e-5 level. Not a pass/fail gate.

Read-only: no file outside fe2_extension/ is written.
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

from dhprom_ann_direct_law_claude import DHpromAnnDirectLaw

REFERENCE_DIR = ROOT / "hprom" / "ann" / "stage_10_results_maw_dynamic"
SECONDARY_METRICS_NPZ = ROOT / "pann" / "data" / "hprom_ann_direct_stage10_metrics.npz"


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


def main():
    ref_strain = np.load(REFERENCE_DIR / "dhprom_ann_strain.npy")
    ref_stress = np.load(REFERENCE_DIR / "dhprom_ann_stress.npy")
    print(f"[verify] loaded reference: strain={ref_strain.shape}, stress={ref_stress.shape}")

    strain_path, emax = build_held_out_strain_path()
    E_per_step = densified_per_step_strains(strain_path, emax)
    print(f"[verify] rebuilt {E_per_step.shape[0]} densified per-step strains from the held-out path")
    if E_per_step.shape[0] != ref_strain.shape[0] - 1:
        print(
            f"[verify] WARNING: step-count mismatch ({E_per_step.shape[0]} vs "
            f"{ref_strain.shape[0] - 1}) -- held-out path may not match exactly."
        )

    rel_strain_path = relative_l2(E_per_step, ref_strain[1:E_per_step.shape[0] + 1])
    print(f"[verify] sanity check, rebuilt applied strain vs reference applied strain: rel L2={rel_strain_path:.3e}")

    print("[verify] building DHpromAnnDirectLaw ...")
    law = DHpromAnnDirectLaw()

    print(f"[verify] evaluating extracted law at {E_per_step.shape[0]} independent strains ...")
    eps_ext = np.zeros_like(E_per_step)
    sig_ext = np.zeros_like(E_per_step)
    for i, E in enumerate(E_per_step):
        eps_ext[i], sig_ext[i] = law.evaluate(E)
        if (i + 1) % 200 == 0:
            print(f"    ... {i + 1}/{E_per_step.shape[0]}")

    np.save(HERE / "stage1_extracted_eps.npy", eps_ext)
    np.save(HERE / "stage1_extracted_sig.npy", sig_ext)

    n = eps_ext.shape[0]
    rel_eps = relative_l2(eps_ext, ref_strain[1:n + 1])
    rel_sig = relative_l2(sig_ext, ref_stress[1:n + 1])
    print(f"\n[verify] PRIMARY comparison (vs. real online dhprom_ann_*.npy):")
    print(f"[verify]   relative L2: eps={rel_eps:.3e}, sig={rel_sig:.3e}")

    tol = 1e-5
    if rel_eps < tol and rel_sig < tol:
        print(f"[verify] PASS: extraction reproduces the original online output to within {tol:.0e}.")
    else:
        print(f"[verify] FAIL: mismatch exceeds {tol:.0e} -- do not trust this extraction for new queries yet.")

    if SECONDARY_METRICS_NPZ.exists():
        sec = np.load(SECONDARY_METRICS_NPZ)
        if "dhprom_ann_direct_stress" in sec.files and "stage10_strain" in sec.files:
            sec_stress = sec["dhprom_ann_direct_stress"]
            m = min(n, sec_stress.shape[0] - 1)
            rel_sig_sec = relative_l2(sig_ext[:m], sec_stress[1:m + 1])
            print(f"\n[verify] SECONDARY, informational only (vs. reaction-force-recomputed stress):")
            print(f"[verify]   relative L2 sig={rel_sig_sec:.3e} (expected ~1e-3-1e-5, NOT a pass/fail gate)")


if __name__ == "__main__":
    main()
