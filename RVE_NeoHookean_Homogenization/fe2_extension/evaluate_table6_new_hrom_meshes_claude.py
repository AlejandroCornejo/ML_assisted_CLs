#!/usr/bin/env python3
"""Real, honest Table 6 numbers for the new, unified reaction-force stress
rule, evaluated through the ACTUAL online classes (DHpromAnnDirectLawFloat64
/ HpromAnnIterativeLawFloat64) wired to the newly-built, genuinely
hyper-reduced meshes (fe2_extension/maw_dynamic_reaction_force_dhpromann/,
.../maw_dynamic_reaction_force_hpromann/), NOT the oracle-displacement
proxy used earlier in Track B (fit_final_10point_rule_claude.py) -- this
end-to-end path also includes the POD-ANN manifold's own displacement-
reconstruction error, so honest numbers here are expected to be somewhat
worse than that oracle check's 0.5579%/1.3673%, not identical to it.

D-HPROM-ANN is memoryless (each state evaluated independently); HPROM-ANN-
iterative is warm-started sequentially across the trajectory (q_prev
threading), exactly mirroring verify_hprom_ann_iterative_extraction_claude.
py's own already-established run_sequential pattern.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for sub in ("core", "prom/ann", "hprom/ann"):
    p = str(ROOT / sub)
    if p not in sys.path:
        sys.path.insert(0, p)

from dhprom_ann_direct_law_float64_claude import DHpromAnnDirectLawFloat64  # noqa: E402
from hprom_ann_iterative_law_float64_claude import HpromAnnIterativeLawFloat64  # noqa: E402

DHPROMANN_DIR = HERE / "maw_dynamic_reaction_force_dhpromann"
HPROMANN_DIR = HERE / "maw_dynamic_reaction_force_hpromann"

STAGE10_DIR = ROOT / "hprom" / "ann" / "stage_10_results_maw_dynamic"
PANN_DATA = ROOT / "pann" / "data" / "alltraj_stage10_direct_energy.npz"


def relative_l2(pred, ref):
    pred, ref = np.asarray(pred, dtype=float), np.asarray(ref, dtype=float)
    return float(np.linalg.norm(pred - ref) / max(np.linalg.norm(ref), 1e-30))


def run_dhpromann(E_all):
    law = DHpromAnnDirectLawFloat64(hprom_ann_dir=str(DHPROMANN_DIR))
    eps0, sig0 = law.evaluate(np.zeros(3))
    print(f"[dhpromann] zero-strain sanity: |hom_eps|={np.linalg.norm(eps0):.3e} (unused, expect ~small), "
          f"|hom_sig|={np.linalg.norm(sig0):.3e} (expect ~0)")

    n = E_all.shape[0]
    sig_out = np.zeros((n, 3), dtype=float)
    t0 = time.time()
    for i in range(n):
        _, sig_out[i] = law.evaluate(E_all[i])
        if (i + 1) % 200 == 0:
            print(f"    [dhpromann] ... {i + 1}/{n}")
    print(f"[dhpromann] {n} states in {time.time() - t0:.1f}s")
    return sig_out


def run_hpromann(E_all):
    law = HpromAnnIterativeLawFloat64(hprom_ann_dir=str(HPROMANN_DIR))
    eps0, sig0, _, it0, conv0 = law.evaluate(np.zeros(3), q_prev=np.zeros(law.n_primary), step_index=1)
    print(f"[hpromann] zero-strain sanity: |hom_eps|={np.linalg.norm(eps0):.3e} (unused, expect ~small), "
          f"|hom_sig|={np.linalg.norm(sig0):.3e} (expect ~0), iters={it0}, converged={conv0}")

    n = E_all.shape[0]
    sig_out = np.zeros((n, 3), dtype=float)
    iters_hist = np.zeros(n, dtype=int)
    n_nonconverged = 0
    q_prev = np.zeros(law.n_primary, dtype=float)
    t0 = time.time()
    for i in range(n):
        _, sig_out[i], q_prev, n_it, converged = law.evaluate(E_all[i], q_prev=q_prev, step_index=i + 1)
        iters_hist[i] = n_it
        if not converged:
            n_nonconverged += 1
        if (i + 1) % 200 == 0:
            print(f"    [hpromann] ... {i + 1}/{n} (mean iters so far={iters_hist[:i + 1].mean():.2f})")
    print(f"[hpromann] {n} states in {time.time() - t0:.1f}s, mean iters={iters_hist.mean():.2f}, "
          f"max iters={iters_hist.max()}, non-converged={n_nonconverged}/{n}")
    return sig_out


def main():
    E_all = np.load(STAGE10_DIR / "single_run_applied_strain.npy")
    pann = np.load(PANN_DATA)
    stress_true = np.asarray(pann["stage10_stress"], dtype=float)
    print(f"[table6] test trajectory: {E_all.shape[0]} states")
    assert E_all.shape[0] == stress_true.shape[0], (E_all.shape, stress_true.shape)

    print("\n=== D-HPROM-ANN-FE2 (memoryless, new mesh+rule) ===")
    sig_dhpromann = run_dhpromann(E_all)
    err_dhpromann = relative_l2(sig_dhpromann, stress_true)
    print(f"[table6] D-HPROM-ANN-FE2 S-err vs TRUE reaction-force stress: {err_dhpromann:.4%}")

    print("\n=== HPROM-ANN-FE2 (iterative, new mesh+rule) ===")
    sig_hpromann = run_hpromann(E_all)
    err_hpromann = relative_l2(sig_hpromann, stress_true)
    print(f"[table6] HPROM-ANN-FE2 S-err vs TRUE reaction-force stress: {err_hpromann:.4%}")

    print("\n=== SUMMARY (Table 6, new unified reaction-force rule + super-reduced mesh) ===")
    print(f"  D-HPROM-ANN-FE2: {err_dhpromann:.4%}  (mesh: 10 elements)")
    print(f"  HPROM-ANN-FE2  : {err_hpromann:.4%}  (mesh: 20 elements)")
    print("  (for reference, Track B's oracle-displacement check on this same rule: "
          "0.5433%/0.5579% depending on ANN width -- honest end-to-end numbers above "
          "also include POD-ANN displacement-reconstruction error)")

    np.savez(
        HERE / "table6_new_hrom_meshes_result_claude.npz",
        E_all=E_all, stress_true=stress_true,
        sig_dhpromann=sig_dhpromann, sig_hpromann=sig_hpromann,
        err_dhpromann=err_dhpromann, err_hpromann=err_hpromann,
    )
    print(f"\n[table6] saved {HERE / 'table6_new_hrom_meshes_result_claude.npz'}")


if __name__ == "__main__":
    main()
