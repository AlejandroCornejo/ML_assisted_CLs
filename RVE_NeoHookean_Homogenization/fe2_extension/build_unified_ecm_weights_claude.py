#!/usr/bin/env python3
"""Track B, Stage 5: build a new ecm_weights_all.npz for online deployment,
copying the currently-deployed hprom/ann/maw_dynamic/ecm_weights_all.npz
verbatim (res/eps targets, mesh metadata, everything untouched) except the
"sig" (homogenized-stress) target's own fields, which are replaced with
this session's new, reaction-force-targeting 10-point rule
(reaction_force_ecm_ann_model_FINAL_claude.npz: 0.54% Table-6, 1.37% Cook,
vs the existing naive-average-targeting rule's own accuracy against the
WRONG quantity).

The mesh (hrom/ann/maw_dynamic/rve_geometry_hrom.mdpa) is copied unchanged
-- confirmed this session (direct Kratos console output when running
DHpromAnnDirectLawFloat64: "990 Elements, 2122 Nodes") that this "hrom"
mesh already contains the FULL, non-reduced mesh, so the new rule's 10
elements (a different subset than the old Z_sig) already exist in it; no
mesh regeneration is needed.

Original hprom/ann/maw_dynamic/ecm_weights_all.npz is never modified --
every already-reported Table 6 number stays backed by it unchanged. This
script's output goes to a new, separate directory.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
ORIGINAL_DIR = REPO_ROOT / "hprom" / "ann" / "maw_dynamic"
NEW_MODEL_NPZ = HERE / "reaction_force_ecm_ann_model_FINAL_claude.npz"
OUT_DIR = HERE / "maw_dynamic_reaction_force"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    original = dict(np.load(ORIGINAL_DIR / "ecm_weights_all.npz"))
    new_model = np.load(NEW_MODEL_NPZ)

    Z_support = np.asarray(new_model["Z_support"], dtype=np.int64)
    n_layers = int(new_model["scalar_n_layers"])
    print(f"[build-unified] replacing sig target: old Z_sig={original['Z_sig']} "
          f"-> new Z_support={Z_support}")
    print(f"[build-unified] old accuracy context (this session, native Z_sigma vs true "
          f"conjugate at Cook states): ~1.8-3.6%. New rule: "
          f"Table-6={float(new_model['err_stage10']):.4%}, Cook={float(new_model['err_cook']):.4%}")

    replaced = dict(original)
    replaced["Z_sig"] = Z_support
    replaced["maw_sig_regressor_type"] = np.array(["ann"])
    replaced["maw_sig_ann_x_mean"] = np.asarray(new_model["x_mean"], dtype=float)
    replaced["maw_sig_ann_x_std"] = np.asarray(new_model["x_std"], dtype=float)
    replaced["maw_sig_ann_activation"] = np.array([str(new_model["scalar_activation"])])
    replaced["maw_sig_ann_hidden_dims"] = np.asarray(new_model["hidden_dims"], dtype=np.int64)
    replaced["maw_sig_ann_n_layers"] = np.array([n_layers], dtype=np.int64)
    replaced["maw_sig_ann_target_sum"] = np.array([float(new_model["scalar_target_sum"])])
    replaced["maw_sig_ann_best_epoch"] = np.array([int(new_model["scalar_best_epoch"])], dtype=np.int64)
    replaced["maw_sig_ann_train_rel_error"] = np.array([float(new_model["scalar_train_rel_error"])])
    replaced["maw_sig_ann_val_rel_error"] = np.array([float(new_model["scalar_val_rel_error"])])
    for i in range(n_layers):
        replaced[f"maw_sig_ann_W_{i}"] = np.asarray(new_model[f"W_{i}"], dtype=float)
        replaced[f"maw_sig_ann_b_{i}"] = np.asarray(new_model[f"b_{i}"], dtype=float)
    # Any leftover old-architecture layers beyond the new n_layers would be
    # stale/inconsistent with the new n_layers metadata -- remove them so
    # nothing accidentally reads a mismatched old W_4/b_4 etc.
    stale = [k for k in replaced if k.startswith("maw_sig_ann_W_") or k.startswith("maw_sig_ann_b_")]
    for k in stale:
        idx = int(k.rsplit("_", 1)[1])
        if idx >= n_layers:
            del replaced[k]

    out_npz = OUT_DIR / "ecm_weights_all.npz"
    np.savez(out_npz, **replaced)
    print(f"[build-unified] saved {out_npz}")

    mesh_src = ORIGINAL_DIR / "rve_geometry_hrom.mdpa"
    mesh_dst = OUT_DIR / "rve_geometry_hrom.mdpa"
    shutil.copy2(mesh_src, mesh_dst)
    print(f"[build-unified] copied mesh {mesh_src} -> {mesh_dst}")

    print(f"[build-unified] original {ORIGINAL_DIR}/ecm_weights_all.npz left untouched.")


if __name__ == "__main__":
    main()
