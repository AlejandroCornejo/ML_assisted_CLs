#!/usr/bin/env python3
"""Diagnose why the new reaction-force rule's Cook tip_uy error (2.87%/
1.80%) is notably worse than the old naive-average rule's (0.55%/0.28%)
even though the raw final-state S err. is comparable or better.

Step 1: isolate whether the NEW material law's OWN prediction is accurate
in isolation (no macro Newton-path compounding) by evaluating it DIRECTLY
at FOM-nested's own 384 already-converged macro states
(stress_correction_stage_a_fom_result_claude.npz's e_gp) and comparing
against the true reaction-force reference (stage_b's stress_rf) -- if
this is small (~Table 6's own 0.5-0.7% ballpark), the material law itself
is fine and the tip_uy gap is a path-sensitivity/compounding effect, not
a flaw in the new rule.

Step 2: directly compare NEW (reaction-force, 10-elem) vs OLD (naive-
average, 29-elem) hom_sig at the SAME macro states, to see how much the
two conventions actually disagree pointwise at Cook's own operating
range (expected to be small, given the already-established 0.0066%
naive-vs-true gap there) -- if they mostly agree, the tip_uy gap is not
explained by "which convention," pointing instead at mesh-size/support-
count approximation noise.
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

from dhprom_ann_direct_law_float64_claude import DHpromAnnDirectLawFloat64  # noqa: E402
from fom_solver_rve import CalculateHomogenizedFromAssemblerWithElementWeights  # noqa: E402

NEW_DIR = HERE / "maw_dynamic_reaction_force_dhpromann"
OLD_DIR = ROOT / "hprom" / "ann" / "maw_dynamic"

STAGE_A_NPZ = HERE / "stress_correction_stage_a_fom_result_claude.npz"
STAGE_B_NPZ = HERE / "stress_correction_stage_b_fom_result_claude.npz"


def relative_l2(pred, ref):
    return float(np.linalg.norm(pred - ref) / max(np.linalg.norm(ref), 1.0e-30))


def naive_average_hom_sig(law, E):
    """Reproduces the OLD (pre-fix) evaluate()'s naive-average hom_sig,
    reusing whatever internal vec_assembler state law.evaluate(E) just
    left behind (ComputeLocalArrays already called for this E/disp)."""
    mu_dim = int(law.qp_aff["mu_dim"])
    mu = np.asarray(E, dtype=float).reshape(-1)[:mu_dim]
    q_p = np.concatenate([mu, [1.0]]) @ np.asarray(law.qp_aff["b_aff"], dtype=float)
    w_sig = law._hom_weights(law.maw_sig_hom, q_p, E)
    _, hom_sig = CalculateHomogenizedFromAssemblerWithElementWeights(
        law.vec_assembler, w_eps=None, w_sig=w_sig, reference_measure=law.hom_reference_measure,
    )
    return hom_sig


def main():
    stage_a = np.load(STAGE_A_NPZ)
    stage_b = np.load(STAGE_B_NPZ)
    e_gp = np.asarray(stage_a["e_gp"], dtype=float)
    stress_rf = np.asarray(stage_b["stress_rf"], dtype=float)
    n = e_gp.shape[0]
    print(f"[diagnose] {n} FOM-nested converged macro states loaded")

    law_new = DHpromAnnDirectLawFloat64(hprom_ann_dir=str(NEW_DIR))
    law_old = DHpromAnnDirectLawFloat64(hprom_ann_dir=str(OLD_DIR))

    pred_new = np.zeros((n, 3), dtype=float)
    pred_old_naive = np.zeros((n, 3), dtype=float)
    for i in range(n):
        E = e_gp[i]
        _, pred_new[i] = law_new.evaluate(E)
        law_old.evaluate(E)  # populate law_old.vec_assembler state for this E; ignore its (meaningless) reaction-force return
        pred_old_naive[i] = naive_average_hom_sig(law_old, E)

    err_new_vs_true = relative_l2(pred_new, stress_rf)
    err_old_vs_true = relative_l2(pred_old_naive, stress_rf)
    err_new_vs_old = relative_l2(pred_new, pred_old_naive)

    print(f"\n[diagnose] Step 1 -- material law accuracy in isolation (no macro Newton compounding):")
    print(f"  NEW (reaction-force, 10-elem) vs TRUE stress_rf : {err_new_vs_true:.4%}")
    print(f"  OLD (naive-average, 29-elem)  vs TRUE stress_rf : {err_old_vs_true:.4%}")

    print(f"\n[diagnose] Step 2 -- how much do the two conventions disagree pointwise at these states:")
    print(f"  NEW vs OLD (both evaluated at the SAME 384 macro states): {err_new_vs_old:.4%}")

    per_state_err_new = np.linalg.norm(pred_new - stress_rf, axis=1) / np.maximum(np.linalg.norm(stress_rf, axis=1), 1e-30)
    per_state_err_old = np.linalg.norm(pred_old_naive - stress_rf, axis=1) / np.maximum(np.linalg.norm(stress_rf, axis=1), 1e-30)
    print(f"\n[diagnose] per-state error distribution (relative, per macro Gauss point):")
    print(f"  NEW: mean={per_state_err_new.mean():.4%}, median={np.median(per_state_err_new):.4%}, "
          f"max={per_state_err_new.max():.4%}, states>5%={int(np.sum(per_state_err_new > 0.05))}/{n}")
    print(f"  OLD: mean={per_state_err_old.mean():.4%}, median={np.median(per_state_err_old):.4%}, "
          f"max={per_state_err_old.max():.4%}, states>5%={int(np.sum(per_state_err_old > 0.05))}/{n}")


if __name__ == "__main__":
    main()
