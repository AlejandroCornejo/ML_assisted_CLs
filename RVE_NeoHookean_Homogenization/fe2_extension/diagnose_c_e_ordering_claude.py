#!/usr/bin/env python3
"""Focused check: does DHpromAnnDirectLawFloat64's own reduced-mesh c_e
(via _reaction_force_c_e, using the NEW _dirichlet_sensitivity method)
match the FULL-mesh, already-validated c_e from
reaction_force_ecm_target_claude.py's per_element_reaction_force_
contribution (via DirectStressGenerator), at the SAME Z_support elements
and the SAME TRUE displacement u -- isolating whether the Cook-state
7.69% error is an indexing/formula bug in the reduced-mesh path, or a
genuine POD-ANN displacement-reconstruction-quality issue specific to
Cook's strain combinations.
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
from reaction_force_ecm_target_claude import (  # noqa: E402
    DirectStressGenerator, build_dof_to_dirpos, per_element_reaction_force_contribution,
)

NEW_DIR = HERE / "maw_dynamic_reaction_force_dhpromann"
STAGE_A_NPZ = HERE / "stress_correction_stage_a_fom_result_claude.npz"
STAGE_B_NPZ = HERE / "stress_correction_stage_b_fom_result_claude.npz"


def main():
    stage_a = np.load(STAGE_A_NPZ)
    stage_b = np.load(STAGE_B_NPZ)
    e_gp = np.asarray(stage_a["e_gp"], dtype=float)
    U_final_all = np.asarray(stage_a["U_final_all"], dtype=float)
    stress_rf = np.asarray(stage_b["stress_rf"], dtype=float)

    law = DHpromAnnDirectLawFloat64(hprom_ann_dir=str(NEW_DIR))
    Z_support = np.load(HERE / "reaction_force_ecm_ann_model_FINAL_claude.npz")["Z_support"]
    print(f"[diag-ce] Z_support (training order) = {Z_support}")

    full_to_local = law.full_to_local_hom  # dict: full_idx -> local_idx, built in __init__
    local_pos_for_Z_support = np.array([full_to_local[int(z)] for z in Z_support])
    print(f"[diag-ce] Z_support's LOCAL positions in the reduced mesh: {local_pos_for_Z_support}")

    gen = DirectStressGenerator()
    dof_to_dirpos = build_dof_to_dirpos(gen)

    idxs = [0, 50, 100, 150, 200, 250, 300, 350]
    print(f"\n{'i':>4}{'|c_e full (Z_support order)|':>30}{'|c_e local (via online recon)|':>32}{'rel diff':>12}")
    hom_sig_true_u = np.zeros((len(idxs), 3))
    hom_sig_online = np.zeros((len(idxs), 3))
    for j, i in enumerate(idxs):
        E = e_gp[i]
        u_true = U_final_all[i]

        # (a) TRUE u, full-mesh c_e (already-validated Stage 1 path), restricted to Z_support.
        c_e_full = per_element_reaction_force_contribution(gen, dof_to_dirpos, u_true, E)
        c_e_full_at_support = c_e_full[Z_support, :]

        # (b) online: reconstructed u via the POD-ANN manifold, reduced-mesh c_e,
        # gathered at the SAME elements via full_to_local.
        _, hom_sig_online[j] = law.evaluate(E)
        c_e_local_full = law._reaction_force_c_e(E)  # (n_current_elements, 3), populated by the evaluate() call just above
        c_e_local_at_support = c_e_local_full[local_pos_for_Z_support, :]

        # (c) TRUE-u oracle prediction at Z_support order, using the SAME w_sig the online path used.
        mu_dim = int(law.qp_aff["mu_dim"])
        mu = E[:mu_dim]
        q_p = np.concatenate([mu, [1.0]]) @ np.asarray(law.qp_aff["b_aff"], dtype=float)
        w_online_local = law._hom_weights(law.maw_sig_hom, q_p, E)  # local-mesh order
        w_online_at_support = w_online_local[local_pos_for_Z_support]
        hom_sig_true_u[j] = c_e_full_at_support.T @ w_online_at_support

        rel = np.linalg.norm(c_e_full_at_support - c_e_local_at_support) / max(np.linalg.norm(c_e_full_at_support), 1e-30)
        print(f"{i:>4}{np.linalg.norm(c_e_full_at_support):>30.6e}{np.linalg.norm(c_e_local_at_support):>32.6e}{rel:>12.4%}")

    gen.close()

    def relative_l2(pred, ref):
        return float(np.linalg.norm(pred - ref) / max(np.linalg.norm(ref), 1e-30))

    print(f"\n[diag-ce] hom_sig, TRUE-u oracle (this w_sig, sampled states)  vs stress_rf: "
          f"{relative_l2(hom_sig_true_u, stress_rf[idxs]):.4%}")
    print(f"[diag-ce] hom_sig, ONLINE (reconstructed u, sampled states)     vs stress_rf: "
          f"{relative_l2(hom_sig_online, stress_rf[idxs]):.4%}")


if __name__ == "__main__":
    main()
