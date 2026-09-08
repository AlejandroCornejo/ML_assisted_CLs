#!/usr/bin/env python3
"""Analytic Jacobian of hprom_ann_solver_rve.py's
_evaluate_maw_hom_weights_current, restricted to the one branch this
project's trained sig/eps homogenization-weight models actually use
(confirmed by direct inspection of hprom/ann/maw_dynamic/ecm_weights_all.npz:
maw_eps_coord_label=maw_sig_coord_label='mu', maw_eps_regressor_type=
maw_sig_regressor_type='ann', not componentwise, eval_mode='model'):

  q_query = E                                    (coord_label == "mu")
  w_support = eval_mawecm_ann(q_query, ann_model)  (regressor_type == "ann")
  w_full = scatter(w_support at z_support_full into an n_elem_reference vector)
  w_current = gather(w_full at the HROM mesh's full_to_local map)

The scatter and gather steps are both constant (E-independent) linear index
maps, so the only genuinely new derivative is the ANN's own Jacobian
(mawecm_ann_jacobian_claude.eval_mawecm_ann_with_jacobian); this module just
threads it through the same two index operations
_evaluate_maw_hom_weights_current itself performs, so that the returned
weight VALUE matches it exactly (checked below against the real function,
not just against eval_mawecm_ann in isolation).

Any other configuration (coord_label=='q', regressor_type in ('rbf',
'fixed_classic'), eval_mode=='nearest', or componentwise models) raises
NotImplementedError rather than silently computing the wrong Jacobian --
those paths are not exercised by this project's deployed model and have not
been derived here.
"""
from __future__ import annotations

import numpy as np

from mawecm_ann_jacobian_claude import eval_mawecm_ann_with_jacobian


def maw_hom_weight_and_jacobian_single_model(
    E, target_model, n_elem_reference, n_current_elements, full_to_local,
):
    """Returns (w_current (n_current_elements,), dw_current_dE (n_current_elements,3)).
    Mirrors hprom_ann_solver_rve.py's _evaluate_maw_hom_weights_current for
    the ann/mu/model branch only."""
    regressor_type = str(target_model.get("regressor_type", "rbf")).strip().lower()
    if regressor_type == "fixed_classic":
        # w_support is a constant vector (no E-dependence at all -- classic
        # ECM's own fixed weight, not a regressor evaluated at E), so its
        # Jacobian is exactly zero. Scatter/gather below are unchanged.
        w_support = np.asarray(target_model["w_fixed"], dtype=float).reshape(-1)
        dw_support_dE = np.zeros((w_support.size, 3), dtype=float)
    else:
        coord_label = str(target_model.get("coord_label", "q")).strip().lower()
        if coord_label != "mu":
            raise NotImplementedError(
                f"Consistent tangent only implemented for coord_label='mu' (got '{coord_label}')."
            )
        if regressor_type != "ann":
            raise NotImplementedError(
                f"Consistent tangent only implemented for regressor_type='ann' (got '{regressor_type}')."
            )
        q_query = np.asarray(E, dtype=float).reshape(1, -1)
        w_support, dw_support_dE = eval_mawecm_ann_with_jacobian(q_query, target_model["ann_model"])

    z_full = np.asarray(target_model["z_support_full"], dtype=np.int64).reshape(-1)
    if z_full.size != w_support.size:
        raise RuntimeError(
            f"MAW {target_model.get('target')} support/weight mismatch: "
            f"{z_full.size} vs {w_support.size}."
        )
    n_ref = int(n_elem_reference)
    w_full = np.zeros(n_ref, dtype=float)
    dw_full_dE = np.zeros((n_ref, 3), dtype=float)
    w_full[z_full] = w_support
    dw_full_dE[z_full, :] = dw_support_dE

    if full_to_local is None:
        if int(n_current_elements) != w_full.size:
            raise RuntimeError(
                f"Current mesh has {n_current_elements} elements but "
                f"MAW weights are full-mesh length {w_full.size}."
            )
        return w_full, dw_full_dE

    n_cur = int(n_current_elements)
    w_current = np.zeros(n_cur, dtype=float)
    dw_current_dE = np.zeros((n_cur, 3), dtype=float)
    for full_idx, local_idx in full_to_local.items():
        fi, li = int(full_idx), int(local_idx)
        if 0 <= fi < w_full.size:
            w_current[li] = w_full[fi]
            dw_current_dE[li, :] = dw_full_dE[fi, :]
    return w_current, dw_current_dE


if __name__ == "__main__":
    import os
    import sys
    from pathlib import Path

    HERE = Path(__file__).resolve().parent
    ROOT = HERE.parent
    for sub in ("core", "prom/ann", "hprom/ann"):
        p = str(ROOT / sub)
        if p not in sys.path:
            sys.path.insert(0, p)
    KRATOS_PATH = "/home/kratos/Kratos_Eigen_Check/bin/Release"
    if KRATOS_PATH not in sys.path:
        sys.path.append(KRATOS_PATH)
    os.chdir(str(HERE))

    from hprom_ann_solver_rve import (
        LoadHpromAnnModel,
        _build_full_to_local_map,
        _has_maw_hom_component_models,
        _build_maw_hom_target_model,
        _evaluate_maw_hom_weights_current,
    )

    ann_data_dir = str(ROOT / "prom" / "ann" / "stage_7_ann_model_ls")
    hprom_ann_dir = str(ROOT / "hprom" / "ann" / "maw_dynamic")
    basis_dir = str(ROOT / "pod" / "stage_2_pod_rve")
    (_phi_p, _phi_s, _free_dofs, _dir_dofs, _eq_map, _Xc, _Yc, _ann_model, _device, ecm_data, _inc) = (
        LoadHpromAnnModel(basis_dir=basis_dir, ann_data_dir=ann_data_dir, hprom_ann_dir=hprom_ann_dir)
    )
    assert not _has_maw_hom_component_models(ecm_data), "componentwise models not covered by this self-test"
    maw_eps_hom = _build_maw_hom_target_model(ecm_data, "eps")
    maw_sig_hom = _build_maw_hom_target_model(ecm_data, "sig")

    n_elem_reference = int(np.ravel(ecm_data["n_elem"])[0])
    hrom_full_indices = np.asarray(ecm_data["hrom_element_full_indices"], dtype=np.int64).reshape(-1)
    n_current_elements = hrom_full_indices.size
    full_to_local = _build_full_to_local_map(
        ecm_data, n_elem_reference=n_elem_reference, n_current_elements=n_current_elements,
    )

    test_states = [
        np.array([0.05, 0.0, 0.0]),
        np.array([0.3, -0.1, 0.05]),
        np.array([0.8, 0.4, -0.05]),
        np.array([1.2, 0.6, 0.06]),
    ]
    h = 1.0e-6
    all_ok = True
    for label, model in (("eps", maw_eps_hom), ("sig", maw_sig_hom)):
        for E in test_states:
            w, dw_dE = maw_hom_weight_and_jacobian_single_model(
                E, model, n_elem_reference, n_current_elements, full_to_local,
            )
            w_check = _evaluate_maw_hom_weights_current(
                q_m=np.zeros(3), e_vec=E, target_model=model,
                n_elem_reference=n_elem_reference, n_current_elements=n_current_elements,
                full_to_local=full_to_local, eval_mode="model",
            )
            val_err = np.linalg.norm(w - w_check) / max(np.linalg.norm(w_check), 1e-30)

            dw_dE_fd = np.zeros_like(dw_dE)
            for k in range(3):
                Ep, Em = E.copy(), E.copy()
                Ep[k] += h
                Em[k] -= h
                wp = _evaluate_maw_hom_weights_current(
                    q_m=np.zeros(3), e_vec=Ep, target_model=model,
                    n_elem_reference=n_elem_reference, n_current_elements=n_current_elements,
                    full_to_local=full_to_local, eval_mode="model",
                )
                wm = _evaluate_maw_hom_weights_current(
                    q_m=np.zeros(3), e_vec=Em, target_model=model,
                    n_elem_reference=n_elem_reference, n_current_elements=n_current_elements,
                    full_to_local=full_to_local, eval_mode="model",
                )
                dw_dE_fd[:, k] = (wp - wm) / (2 * h)
            rel_err = np.linalg.norm(dw_dE - dw_dE_fd) / max(np.linalg.norm(dw_dE_fd), 1e-30)
            ok = val_err < 1e-8 and rel_err < 1e-5
            all_ok = all_ok and ok
            print(
                f"[{label}] E={E}: value_rel_err={val_err:.3e}, "
                f"Jacobian_rel_err={rel_err:.3e} [{'ok' if ok else 'FAIL'}]"
            )
    print("PASS" if all_ok else "FAIL")
