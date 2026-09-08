#!/usr/bin/env python3
"""Stage A of the Cook's-membrane reaction-force unification: extract q_p
(reduced primary coordinate) for D-HPROM-ANN at each of Cook's 384 already
-converged final-state macro strains (e_gp, saved in
cook_results_dhprom_f64_consistent_claude.npz).

Why q_p at all, instead of reusing the saved s_gp directly: s_gp is the
native MAW-ECM (Z_sigma) stress, exactly the quantity being replaced. To
apply the reaction-force correction we need the FULL (all-990-element)
reconstructed displacement field, which requires q_p as input to the LS
decoder (Phi_m, Phi_s, A_m, ann_model) used by
pann/direct_energy/hprom_ann_direct_stress.py's reconstruct_free_fluctuation
-- exactly the same, already-validated pattern, just fed Cook's own strains
instead of the Stage-10 test trajectory.

For D-HPROM-ANN, q_p is a pure closed-form affine function of E (no Newton
correction, no history dependence -- confirmed by this project's own
"direct mode is provably memoryless" finding), so it is computed directly
here rather than by adding a new return value to the existing, already-
verified DHpromAnnDirectLawFloat64 class (left completely unmodified).

Cross-check performed here (not just assumed): DHpromAnnDirectLawFloat64's
own evaluate_with_tangent(E) is also called at every one of the 384 states,
and its hom_sig is compared against the already-saved s_gp -- these should
match to near machine precision, since both are the same native
computation; a mismatch would mean e_gp/s_gp do not actually correspond to
this exact law/config, which must be resolved before trusting anything
downstream.

Run in its own process (not sharing a Python process with any other script
that also builds its own Kratos model), matching this project's established
caution around monkeypatched-global material-law state
(_material_law_guard_claude.true_neo_hookean_active) and multiple
simultaneous Kratos AnalysisStage instances.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from dhprom_ann_direct_law_float64_claude import DHpromAnnDirectLawFloat64  # noqa: E402

COOK_NPZ = HERE / "cook_results_dhprom_f64_consistent_claude.npz"
OUT_NPZ = HERE / "stress_correction_stage_a_dhprom_result_claude.npz"


def main() -> None:
    d = np.load(COOK_NPZ)
    e_gp = np.asarray(d["e_gp"], dtype=float)
    s_gp_saved = np.asarray(d["s_gp"], dtype=float)
    n = e_gp.shape[0]
    print(f"[stage-a-dhprom] loaded e_gp {e_gp.shape}, s_gp {s_gp_saved.shape} from {COOK_NPZ.name}")

    law = DHpromAnnDirectLawFloat64()
    mu_dim = int(law.qp_aff["mu_dim"])
    b_aff = np.asarray(law.qp_aff["b_aff"], dtype=float)

    q_p_all = np.zeros((n, law.n_primary), dtype=float)
    hom_sig_check = np.zeros((n, 3), dtype=float)

    t0 = time.perf_counter()
    for i in range(n):
        E = e_gp[i]
        mu = E[:mu_dim]
        q_p_all[i] = np.concatenate([mu, [1.0]]) @ b_aff
        _hom_eps, hom_sig, _dEps, _dSig = law.evaluate_with_tangent(E)
        hom_sig_check[i] = hom_sig
        if (i + 1) % 64 == 0:
            dt = time.perf_counter() - t0
            print(f"  [{i + 1}/{n}] {dt:.1f}s elapsed ({dt / (i + 1):.4f}s/call)", flush=True)

    dt = time.perf_counter() - t0
    print(f"[stage-a-dhprom] DONE: {n} calls, {dt:.1f}s total ({dt / n:.4f}s/call)")

    err = np.linalg.norm(hom_sig_check - s_gp_saved) / max(np.linalg.norm(s_gp_saved), 1.0e-30)
    print(f"[stage-a-dhprom] cross-check: fresh evaluate_with_tangent hom_sig vs saved s_gp, "
          f"relative L2 = {err:.6e}")
    if err > 1.0e-6:
        print("[stage-a-dhprom] WARNING: cross-check error is larger than expected for a pure "
              "re-evaluation at the same E -- investigate before trusting q_p_all downstream.")

    np.savez(OUT_NPZ, e_gp=e_gp, q_p_all=q_p_all, hom_sig_check=hom_sig_check,
             s_gp_saved=s_gp_saved, cross_check_relative_l2=err)
    print(f"[stage-a-dhprom] saved to {OUT_NPZ}")


if __name__ == "__main__":
    main()
