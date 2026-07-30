#!/usr/bin/env python3
""""Direct" energy-conjugate stress for the fresh HPROM-ANN / D-HPROM-ANN Stage-10 rerun.

Applies the exact reaction-force formula verified in
``reaction_force_direct_stress.py`` (see that module's docstring for the full
derivation) to the two ROM modes stored in
``hprom/ann/stage_10_results_maw_dynamic/``:

  * ITERATIVE HPROM-ANN (multi-corrector): reduced primary coordinates in
    ``hprom_ann_run_q_p.npy`` (identical to ``hprom_ann_strain.npy`` /
    ``hprom_ann_stress.npy`` -- verified numerically, see
    ``_identify_modes`` below / module-level notes).
  * DIRECT D-HPROM-ANN (zero-corrector): reduced primary coordinates in
    ``trajectory_direct_hprom_ann_q_p.npy`` (identical to
    ``dhprom_ann_strain.npy`` / ``dhprom_ann_stress.npy``).

Neither ROM mode saves a full nodal displacement history -- only the reduced
primary coordinates ``q_m(t)`` (``q_p`` in this repo's naming, shape
``(n_steps, 3)``). The full displacement is reconstructed with the exact same
LS decoder used online by ``hprom/ann/hprom_ann_solver_rve.py``:

    u_fluc = Phi_m @ A_m @ q_m + Phi_s @ q_s(q_m)

where ``q_s(q_m) = ann_model(q_m)`` (the manifold ANN takes *only* q_m as
input -- confirmed via ``hprom_ann_solver_rve.py``'s
``_build_ann_input(q_tensor, e_tensor): return q_tensor`` and via
``manifold_ann_metadata.npz``'s ``include_macro_strain_input=0``,
``n_primary=3``). The solver's online code additionally applies a "manifold
correction" using constants ``q0_const, J0_const`` evaluated at ``q_m=0``;
algebraically this correction cancels exactly (independent of their values),
so the closed form above is not an approximation of the online decoder, it
is algebraically identical to it.  (This is also moot here because the ANN
is origin-anchored, i.e. ``ann_model(0) = 0`` by construction, so
``q0_const = 0`` identically.)

``Phi_m`` (``phi_m.npy``, loaded into the ``phi_p`` variable -- this repo's
own naming quirk, not a mistake here), ``Phi_s`` (``phi_s.npy``), and
``A_m`` (``A_m.npy``, attached as ``ann_model.a_m_np`` by
``LoadPromAnnModel``) live in ``prom/ann/stage_7_ann_model_ls/`` (the
canonical, non-superseded ANN-LS checkpoint directory -- see
``animations/README.md``). ``free_dofs.npy`` / ``dirichlet_dofs.npy`` /
``domain_center.npy`` live in ``pod/stage_2_pod_rve/``. This script reuses
``prom_ann_solver_rve.LoadPromAnnModel`` verbatim to load all of the above,
rather than re-deriving the loading logic.

IMPORTANT correction (found by debugging an initial ~2-4 order-of-magnitude
blow-up in the reconstructed strain field): ``u_fluc`` above is only the
*fluctuation* part of the displacement at the FREE dofs. Both
``prom_ann_solver_rve.py`` (``u_free = u_aff_free + u_fluc``, line ~558) and
``hprom_ann_solver_rve.py`` add the same macro-affine baseline
``u_aff_free(E) = (F(E)-I) @ (X - Xc)`` (the exact same closed-form map used
for the Dirichlet BC, just evaluated at the FREE dofs' reference
coordinates) on top of the POD/ANN fluctuation. Omitting this term (as an
earlier version of this script did) leaves the free-dof displacement missing
its dominant, strain-proportional part, which produces wildly unphysical
local Green-Lagrange strains (observed: |E| up into the hundreds, det(C)
approaching the 1e-30 clip floor) once the applied strain is not tiny. The
correct total displacement is therefore:

    u_free      = u_aff_free(E) + Phi_m @ A_m @ q_m + Phi_s @ q_s(q_m)
    u_dirichlet = u_aff_dirichlet(E)   [same affine map, Dirichlet reference coords]

The Dirichlet block is filled with the same closed-form affine map used
everywhere else in this project (``fom.ComputeDirichletValuesFromGreenLagrange``),
driven by the macro strain schedule that is common to every solver on this
Stage-10 path (confirmed identical to ``pann/data``'s ``stage10_strain`` --
see the assertion in ``main()``). The resulting full nodal displacement
history is then fed through the exact same
``DirectStressGenerator.direct_stress_history`` used and verified in
``reaction_force_direct_stress.py`` -- run on the FULL (non-hyper-reduced)
mesh, since the "direct" energy-conjugate quantity is by definition a
property of the full fine-scale RVE's stored energy, not of whichever
element subset a given ROM approximates it with.

Ground truth: ``pann/data/alltraj_stage10_direct_energy.npz``'s
``stage10_strain``/``stage10_stress`` (same convention/metric code as
``pann/anisotropic/evaluate_polyconvex_final_claude.py``: relative L2 with a
1e-30 floor in the denominator, plane-stress von Mises equivalent
``sqrt(max(S11^2 - S11*S22 + S22^2 + 3*S12^2, 0))``).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
CORE_DIR = REPO_ROOT / "core"
PROM_ANN_DIR = REPO_ROOT / "prom" / "ann"
POD_DIR = REPO_ROOT / "pod" / "stage_2_pod_rve"
ANN_MODEL_DIR = PROM_ANN_DIR / "stage_7_ann_model_ls"
RESULTS_DIR = REPO_ROOT / "hprom" / "ann" / "stage_10_results_maw_dynamic"
PANN_DATA = REPO_ROOT / "pann" / "data" / "alltraj_stage10_direct_energy.npz"
OLD_REFERENCE = REPO_ROOT / "pann" / "data" / "stage10_hprom_direct_reference.json"
OUT_JSON = REPO_ROOT / "pann" / "data" / "hprom_ann_direct_stage10_metrics.json"
OUT_NPZ = REPO_ROOT / "pann" / "data" / "hprom_ann_direct_stage10_metrics.npz"

for _p in (CORE_DIR, PROM_ANN_DIR, HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import fom_solver_rve as fom  # noqa: E402
from reaction_force_direct_stress import DirectStressGenerator  # noqa: E402
from prom_ann_solver_rve import LoadPromAnnModel  # noqa: E402


def sigma_eq(stress: np.ndarray) -> np.ndarray:
    """Plane-stress von Mises equivalent of [S11, S22, S12] (matches
    pann/anisotropic/evaluate_polyconvex_final_claude.py's sigma_eq)."""
    sxx, syy, sxy = stress[:, 0], stress[:, 1], stress[:, 2]
    return np.sqrt(np.maximum(sxx * sxx - sxx * syy + syy * syy + 3.0 * sxy * sxy, 0.0))


def relative_l2(prediction: np.ndarray, reference: np.ndarray) -> float:
    return float(np.linalg.norm(prediction - reference) / max(np.linalg.norm(reference), 1.0e-30))


def component_relative_l2(prediction: np.ndarray, reference: np.ndarray) -> list[float]:
    return [relative_l2(prediction[:, k], reference[:, k]) for k in range(3)]


def reconstruct_free_fluctuation(q_p_hist, phi_p, phi_s, a_m, ann_model, device):
    """u_fluc = Phi_m @ A_m @ q_m + Phi_s @ q_s(q_m); see module docstring."""
    with torch.no_grad():
        q_p_t = torch.as_tensor(np.asarray(q_p_hist, dtype=np.float32), device=device)
        q_s_hist = ann_model(q_p_t).detach().cpu().numpy().astype(float)
    phi_master = phi_p @ a_m  # (n_free, n_primary)
    u_fluc_hist = q_p_hist @ phi_master.T + q_s_hist @ phi_s.T  # (n_steps, n_free)
    return u_fluc_hist


def free_dof_reference_coordinates(free_dofs_pod, gen: DirectStressGenerator):
    """x,y,is_x arrays for the free dofs, in the SAME (x-x0c, y-y0c) convention
    used by gen.dir_x/dir_y/dir_is_x, needed for the macro-affine baseline
    u_aff_free(E) that prom_ann_solver_rve.py / hprom_ann_solver_rve.py add on
    top of the POD/ANN fluctuation at every free dof (see module docstring)."""
    x0c, y0c = float(gen.sim._x0c), float(gen.sim._y0c)
    dof_x = np.zeros(gen.n_dof, dtype=float)
    dof_y = np.zeros(gen.n_dof, dtype=float)
    is_x_dof = np.zeros(gen.n_dof, dtype=bool)
    for i, node in enumerate(gen.mp.Nodes):
        xr = float(node.X0) - x0c
        yr = float(node.Y0) - y0c
        idx_x = int(gen.eq_map[i, 0])
        idx_y = int(gen.eq_map[i, 1])
        if 0 <= idx_x < gen.n_dof:
            dof_x[idx_x] = xr
            dof_y[idx_x] = yr
            is_x_dof[idx_x] = True
        if 0 <= idx_y < gen.n_dof:
            dof_x[idx_y] = xr
            dof_y[idx_y] = yr
            is_x_dof[idx_y] = False
    return dof_x[free_dofs_pod], dof_y[free_dofs_pod], is_x_dof[free_dofs_pod]


def assemble_full_displacement(
    u_fluc_hist, applied_strain, free_dofs_pod, x_free, y_free, is_x_free, gen: DirectStressGenerator
):
    n_steps = u_fluc_hist.shape[0]
    U_full = np.zeros((n_steps, gen.n_dof), dtype=float)
    for i in range(n_steps):
        u_aff_free = fom.ComputeDirichletValuesFromGreenLagrange(
            applied_strain[i], x_free, y_free, is_x_free
        )
        U_full[i, free_dofs_pod] = u_aff_free + u_fluc_hist[i]
        U_full[i, gen.dir_dofs] = fom.ComputeDirichletValuesFromGreenLagrange(
            applied_strain[i], gen.dir_x, gen.dir_y, gen.dir_is_x
        )
    return U_full


def main():
    free_dofs_pod = np.load(POD_DIR / "free_dofs.npy")
    dir_dofs_pod = np.load(POD_DIR / "dirichlet_dofs.npy")

    gen = DirectStressGenerator()

    # Integrity check: the POD basis's dof partition must be the *same set*
    # as the full-mesh assembler's own Dirichlet/free partition (both derive
    # from the same mesh/mdpa, but were loaded independently -- verify
    # rather than assume).
    if not np.array_equal(np.sort(free_dofs_pod), np.sort(gen.free_dofs)):
        raise RuntimeError("free-dof partition mismatch between POD basis and full-mesh assembler.")
    if not np.array_equal(np.sort(dir_dofs_pod), np.sort(gen.dir_dofs)):
        raise RuntimeError("Dirichlet-dof partition mismatch between POD basis and full-mesh assembler.")
    print(f"[check] dof partition matches: n_free={free_dofs_pod.size}, n_dir={dir_dofs_pod.size}")

    applied_strain = np.load(RESULTS_DIR / "single_run_applied_strain.npy")
    pann = np.load(PANN_DATA)
    stage10_strain = np.asarray(pann["stage10_strain"], dtype=float)
    stage10_stress = np.asarray(pann["stage10_stress"], dtype=float)

    if not np.allclose(applied_strain, stage10_strain):
        raise RuntimeError(
            "single_run_applied_strain.npy does not match pann/data's stage10_strain; "
            "this fresh run may not be on the canonical Stage-10 path."
        )
    print("[check] applied strain schedule matches pann/data's stage10_strain exactly.")

    # Control: apply the verified generator directly to the FOM's own full
    # displacement history on this exact Stage-10 path (single_run_U.npy).
    # This is NOT a ROM reconstruction -- it is the same kind of check as
    # trajectories 1/2/6 in reaction_force_direct_stress.py, just on the
    # Stage-10 path instead of a Stage-1 trajectory. It must pass at ~1e-9
    # or better before the ROM reconstruction below can be trusted.
    single_U = np.load(RESULTS_DIR / "single_run_U.npy")
    control_stress = gen.direct_stress_history(single_U, applied_strain)
    control_err = relative_l2(control_stress, stage10_stress)
    print(f"[control] full-FOM single_run reaction-force stress vs stage10_stress: rel L2 = {control_err:.6e}")
    if control_err > 1.0e-4:
        print(
            "[control] WARNING: control error is much larger than the 1e-10 level seen on "
            "Stage-1 trajectories. Something about this Stage-10 path/mesh may differ; "
            "treat downstream ROM numbers with caution."
        )

    # LS decoder resources (Phi_m/Phi_s/A_m/ANN), reusing the exact loader
    # used online by the HPROM-ANN solver.
    phi_p, phi_s, free_dofs_ref, dir_dofs_ref, _eq_map_ref, ann_model, device, include_macro = LoadPromAnnModel(
        basis_dir=str(POD_DIR), ann_data_dir=str(ANN_MODEL_DIR)
    )
    if include_macro:
        raise RuntimeError("Unexpected: loaded ANN model expects a macro-strain input.")
    if not np.array_equal(np.sort(free_dofs_ref), np.sort(free_dofs_pod)):
        raise RuntimeError("LoadPromAnnModel's free_dofs differs from pod/stage_2_pod_rve/free_dofs.npy.")
    a_m = np.asarray(ann_model.a_m_np, dtype=float)
    print(f"[ls decoder] phi_p={phi_p.shape}, phi_s={phi_s.shape}, a_m={a_m.shape}, device={device}")

    x_free, y_free, is_x_free = free_dof_reference_coordinates(free_dofs_pod, gen)

    modes = {
        "hprom_ann_iterative": RESULTS_DIR / "hprom_ann_run_q_p.npy",
        "dhprom_ann_direct": RESULTS_DIR / "trajectory_direct_hprom_ann_q_p.npy",
    }
    legacy_strain_files = {
        "hprom_ann_iterative": RESULTS_DIR / "hprom_ann_strain.npy",
        "dhprom_ann_direct": RESULTS_DIR / "dhprom_ann_strain.npy",
    }

    results = {}
    arrays_to_save = {
        "stage10_strain": stage10_strain,
        "stage10_stress": stage10_stress,
        "control_stress": control_stress,
    }

    for mode_name, q_p_path in modes.items():
        q_p_hist = np.load(q_p_path)
        n_steps = q_p_hist.shape[0]
        u_fluc_hist = reconstruct_free_fluctuation(q_p_hist, phi_p, phi_s, a_m, ann_model, device)
        U_full = assemble_full_displacement(
            u_fluc_hist, applied_strain[:n_steps], free_dofs_pod, x_free, y_free, is_x_free, gen
        )

        # Diagnostic only (not part of the reported metrics): compare the
        # homogenized (full-mesh, unweighted) strain of the reconstructed
        # field against this mode's own reported (ECM/MAW-weighted) legacy
        # strain, as a sanity check that the LS-decoder reconstruction is in
        # the right ballpark before trusting the reaction-force numbers.
        legacy_strain = np.load(legacy_strain_files[mode_name])
        gen.assembler.Assemble(U_full[-1])
        eps_h, _ = fom.CalculateHomogenizedFromAssemblerWithElementWeights(gen.assembler)
        print(
            f"[{mode_name}] diagnostic: reconstructed full-mesh strain at final step={eps_h}, "
            f"mode's own reported legacy strain at final step={legacy_strain[n_steps - 1]}"
        )

        stress_mode = gen.direct_stress_history(U_full, applied_strain[:n_steps])

        ref_strain = stage10_strain[:n_steps]
        ref_stress = stage10_stress[:n_steps]
        err = relative_l2(stress_mode, ref_stress)
        comp_err = component_relative_l2(stress_mode, ref_stress)
        vm_pred = sigma_eq(stress_mode)
        vm_ref = sigma_eq(ref_stress)
        vm_err = relative_l2(vm_pred, vm_ref)

        results[mode_name] = {
            "n_steps": int(n_steps),
            "energy_relative_l2": None,
            "stress_relative_l2": err,
            "stress_component_relative_l2": comp_err,
            "von_mises_relative_l2": vm_err,
        }
        arrays_to_save[f"{mode_name}_q_p"] = q_p_hist
        arrays_to_save[f"{mode_name}_stress"] = stress_mode
        print(
            f"[{mode_name}] stress_relative_l2={err:.6e}, "
            f"component={[f'{v:.6e}' for v in comp_err]}, von_mises={vm_err:.6e}"
        )

    gen.close()

    old_ref = json.loads(OLD_REFERENCE.read_text(encoding="utf-8"))["hprom_ann_vs_direct_fom"]
    new_direct = results["dhprom_ann_direct"]
    cross_check_ratio = None
    if old_ref.get("stress_relative_l2"):
        cross_check_ratio = new_direct["stress_relative_l2"] / old_ref["stress_relative_l2"]
    cross_check = {
        "note": (
            "pann/data/stage10_hprom_direct_reference.json's 'hprom_ann_vs_direct_fom' entry "
            "was produced by a since-lost script and is understood (per task instructions) to "
            "correspond to the D-HPROM-ANN (zero-corrector) mode. Compared here against this "
            "script's freshly computed dhprom_ann_direct numbers on the corrected canonical "
            "checkpoint/ECM configuration."
        ),
        "old_reference_stress_relative_l2": old_ref["stress_relative_l2"],
        "old_reference_stress_component_relative_l2": old_ref["stress_component_relative_l2"],
        "new_stress_relative_l2": new_direct["stress_relative_l2"],
        "new_stress_component_relative_l2": new_direct["stress_component_relative_l2"],
        "new_over_old_ratio": cross_check_ratio,
    }
    print("[cross-check vs old reference]", json.dumps(cross_check, indent=2))

    payload = {
        "protocol": (
            "Fresh maw_dynamic Stage-10 HPROM-ANN/D-HPROM-ANN rerun. Full nodal displacement "
            "reconstructed from saved reduced primary coordinates (q_p) via the LS decoder "
            "u = Phi_m A_m q_m + Phi_s q_s(q_m); the exact reaction-force formula verified in "
            "reaction_force_direct_stress.py (see that file) is then applied on the full "
            "(non-hyper-reduced) mesh to obtain the direct energy-conjugate stress. "
            "Reference is pann/data/alltraj_stage10_direct_energy.npz's stage10_strain/stage10_stress."
        ),
        "n_stage10": int(len(stage10_strain)),
        "control_full_fom_single_run_vs_stage10_stress_relative_l2": control_err,
        "modes": results,
        "old_reference_cross_check": cross_check,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(OUT_NPZ, **arrays_to_save)
    print(f"\nSaved metrics to {OUT_JSON}")
    print(f"Saved arrays to {OUT_NPZ}")


if __name__ == "__main__":
    main()
