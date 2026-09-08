#!/usr/bin/env python3
"""Stage B of the Cook's-membrane reaction-force unification, D-HPROM-ANN.

Takes Stage A's saved q_p_all (384, 3) -- the reduced primary coordinate at
each of Cook's 384 already-converged final-state macro strains -- and:
  1. reconstructs the FULL (all-990-element, all-2122-node) displacement
     field via the exact same LS-decoder pattern already validated in
     pann/direct_energy/hprom_ann_direct_stress.py (reused, unmodified,
     via direct import): u_fluc = Phi_m @ A_m @ q_m + Phi_s @ q_s(q_m),
     plus the affine baseline at both free and Dirichlet dofs;
  2. applies pann/direct_energy/reaction_force_direct_stress.py's
     DirectStressGenerator (also reused, unmodified) to get the reaction-
     force-based, exactly energy-conjugate stress at all 384 states.

Run in its own process, separate from Stage A (which builds its own,
different Kratos model/mesh) -- this project's established discipline
around not mixing multiple simultaneously-alive Kratos AnalysisStage
instances / monkeypatched material-law globals in one process.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
DIRECT_ENERGY_DIR = REPO_ROOT / "pann" / "direct_energy"
PROM_ANN_DIR = REPO_ROOT / "prom" / "ann"
POD_DIR = REPO_ROOT / "pod" / "stage_2_pod_rve"
ANN_MODEL_DIR = PROM_ANN_DIR / "stage_7_ann_model_ls"

for p in (DIRECT_ENERGY_DIR, PROM_ANN_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

KRATOS_PATH = "/home/kratos/Kratos_Eigen_Check/bin/Release"
if KRATOS_PATH not in sys.path:
    sys.path.append(KRATOS_PATH)

from reaction_force_direct_stress import DirectStressGenerator  # noqa: E402
from prom_ann_solver_rve import LoadPromAnnModel  # noqa: E402
from hprom_ann_direct_stress import (  # noqa: E402
    reconstruct_free_fluctuation,
    free_dof_reference_coordinates,
    assemble_full_displacement,
)

STAGE_A_NPZ = HERE / "stress_correction_stage_a_dhprom_result_claude.npz"
OUT_NPZ = HERE / "stress_correction_stage_b_dhprom_result_claude.npz"


def relative_l2(prediction: np.ndarray, reference: np.ndarray) -> float:
    return float(np.linalg.norm(prediction - reference) / max(np.linalg.norm(reference), 1.0e-30))


def main() -> None:
    stage_a = np.load(STAGE_A_NPZ)
    e_gp = np.asarray(stage_a["e_gp"], dtype=float)
    q_p_all = np.asarray(stage_a["q_p_all"], dtype=float)
    s_gp_native = np.asarray(stage_a["s_gp_saved"], dtype=float)
    n = e_gp.shape[0]
    print(f"[stage-b-dhprom] loaded q_p_all {q_p_all.shape}, e_gp {e_gp.shape} from stage A")

    gen = DirectStressGenerator()
    free_dofs_pod = np.load(POD_DIR / "free_dofs.npy")
    dir_dofs_pod = np.load(POD_DIR / "dirichlet_dofs.npy")
    if not np.array_equal(np.sort(free_dofs_pod), np.sort(gen.free_dofs)):
        raise RuntimeError("free-dof partition mismatch between POD basis and full-mesh assembler.")
    if not np.array_equal(np.sort(dir_dofs_pod), np.sort(gen.dir_dofs)):
        raise RuntimeError("Dirichlet-dof partition mismatch between POD basis and full-mesh assembler.")
    print(f"[stage-b-dhprom] dof partition check OK: n_free={free_dofs_pod.size}, n_dir={dir_dofs_pod.size}")

    phi_p, phi_s, free_dofs_ref, _dir_dofs_ref, _eq_map_ref, ann_model, device, include_macro = LoadPromAnnModel(
        basis_dir=str(POD_DIR), ann_data_dir=str(ANN_MODEL_DIR)
    )
    if include_macro:
        raise RuntimeError("Unexpected: loaded ANN model expects a macro-strain input.")
    if not np.array_equal(np.sort(free_dofs_ref), np.sort(free_dofs_pod)):
        raise RuntimeError("LoadPromAnnModel's free_dofs differs from pod/stage_2_pod_rve/free_dofs.npy.")
    a_m = np.asarray(ann_model.a_m_np, dtype=float)
    print(f"[stage-b-dhprom] ls decoder loaded: phi_p={phi_p.shape}, phi_s={phi_s.shape}, a_m={a_m.shape}")

    x_free, y_free, is_x_free = free_dof_reference_coordinates(free_dofs_pod, gen)

    t0 = time.perf_counter()
    u_fluc_hist = reconstruct_free_fluctuation(q_p_all, phi_p, phi_s, a_m, ann_model, device)
    U_full = assemble_full_displacement(u_fluc_hist, e_gp, free_dofs_pod, x_free, y_free, is_x_free, gen)
    print(f"[stage-b-dhprom] full-field reconstruction: {time.perf_counter() - t0:.2f}s for {n} states")

    t0 = time.perf_counter()
    stress_rf = gen.direct_stress_history(U_full, e_gp)
    dt = time.perf_counter() - t0
    print(f"[stage-b-dhprom] reaction-force correction: {dt:.2f}s total ({dt / n:.4f}s/state)")

    # Diagnostic sanity check, same spirit as hprom_ann_direct_stress.py's
    # own: the reconstructed field's plain full-mesh (unweighted) strain at
    # a couple of states should be in the same ballpark as e_gp itself
    # (not identical -- e_gp is the imposed macro strain, this is the
    # computed volume-averaged strain of the reconstructed RVE state).
    import fom_solver_rve as fom  # noqa: E402  (already on sys.path via reaction_force_direct_stress's CORE_DIR insert)
    gen.assembler.Assemble(U_full[0])
    eps_h0, _ = fom.CalculateHomogenizedFromAssemblerWithElementWeights(gen.assembler)
    print(f"[stage-b-dhprom] diagnostic state 0: reconstructed full-mesh strain={eps_h0}, imposed e_gp={e_gp[0]}")

    err_vs_native = relative_l2(stress_rf, s_gp_native)
    print(f"[stage-b-dhprom] reaction-force-corrected stress vs native (Z_sigma) stress: "
          f"relative L2 = {err_vs_native:.6e} (expected large-ish, same ~naive-vs-conjugate "
          f"gap discussed in the paper -- NOT a bug if large)")

    gen.close()
    np.savez(OUT_NPZ, e_gp=e_gp, stress_rf=stress_rf, s_gp_native=s_gp_native,
             relative_l2_rf_vs_native=err_vs_native)
    print(f"[stage-b-dhprom] saved to {OUT_NPZ}")


if __name__ == "__main__":
    main()
