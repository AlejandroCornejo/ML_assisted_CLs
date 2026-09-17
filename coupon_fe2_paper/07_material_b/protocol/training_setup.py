"""Construct the frozen B initialization and objective, without optimization.

Historical model classes are reused unchanged. Never invoke their A trainers.
Only fit/reference arrays are accepted here; validation/evaluation belong to
the future runner and cannot influence this preparation.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

from protocol.prepare_design import digest
from protocol.select_features import BASE, RECIPE, affine_design, load_fit_reference

PANN = BASE.parent/"06_pann"
if str(PANN) not in sys.path:
    sys.path.insert(0, str(PANN))
from flexible_pann import FlexibleEnergy
from enriched_pann import inverse_positive
from anisotropic_pann_model import AnisotropicFreeEnergy


def load_preparation(folder: Path, recipe_path: Path = RECIPE):
    recipe = json.loads(recipe_path.read_text())
    manifest = json.loads((folder/"manifest.json").read_text())
    if (manifest["status"] != "complete" or manifest["recipe_sha256"] != digest(recipe_path)
            or manifest["feature_table_sha256"] != digest(folder/"feature_table.npz")
            or manifest["labels_sha256"] != recipe["labels_sha256"]):
        raise ValueError("Feature preparation integrity mismatch")
    for name, expected in manifest["sources_sha256"].items():
        if digest(BASE/name) != expected:
            raise ValueError(f"Feature selection source changed: {name}")
    with np.load(folder/"feature_table.npz", allow_pickle=False) as store:
        table = {key: store[key] for key in ("specs", "initialization_coefficients",
                    "feature_center", "feature_scale", "reference_tangent")}
    if table["specs"].shape != (recipe["feature_selection"]["count"], 5):
        raise ValueError("Incorrect shared feature count")
    return recipe, manifest, table


def build_initial_model(name, seed, recipe, manifest, table, fit_e, fit_s):
    if name not in recipe["models"]["names"] or seed not in recipe["models"]["seeds"]:
        raise ValueError("Undeclared model or seed")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(recipe["execution"]["torch_threads"])
    torch.use_deterministic_algorithms(recipe["execution"]["deterministic_algorithms"])
    scales = manifest["scales"]
    ss, es = scales["strain_scale"], scales["energy_scale"]
    original_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        if name == "Free":
            config = dict(strain_scale=ss, feature_scale=scales["free_feature_scale"],
                          widths=recipe["models"]["widths"]["Free"])
            model = AnisotropicFreeEnergy(**config).double()
            # This historical constructor explicitly casts buffers to float32.
            # Restore exact frozen float64 values; do not modify the shared class.
            model.strain_scale.fill_(ss)
            model.feature_scale.copy_(torch.tensor(config["feature_scale"], dtype=torch.float64))
            return model, dict(configuration=config, calibration_factor=None)
        family = name.split("-")[0]
        cfg = recipe["models"]["constrained"]
        config = dict(strain_scale=ss, specs=table["specs"].tolist(), core=family.lower(),
            widths=recipe["models"]["widths"][family], seed=seed,
            learn_features=name.endswith("-learned"), dynamic_center=cfg["dynamic_center"],
            analytic_stress=cfg["analytic_stress"], spline_basis=cfg["spline_basis"])
        model = FlexibleEnergy(**config).double()
        np.testing.assert_allclose(model.effective_specs().detach().numpy(), table["specs"], rtol=2e-14, atol=2e-15)
        model.volumetric_floor.fill_(cfg["epsilon_and_volume_floor"])
        model.feature_center.copy_(torch.tensor(table["feature_center"], dtype=torch.float64))
        model.feature_scale.copy_(torch.tensor(table["feature_scale"], dtype=torch.float64))
        coef = table["initialization_coefficients"]
        model.initialize(coef, "nonlinear")
        count = min(500, len(fit_e))
        _, jac = affine_design(fit_e[:count], table["specs"])
        x = torch.tensor(fit_e[:count]/ss, dtype=torch.float64)
        _, initial = model.energy_and_stress(x, create_graph=False)
        nonlinear = initial.detach().numpy()-jac@coef*ss
        target = fit_s[:count]*ss/es
        factor = min(1., .05*np.linalg.norm(target)/max(np.linalg.norm(nonlinear), 1e-12))
        with torch.no_grad():
            params = ([model.base_icnn.raw_output_hidden] if family == "ICNN" else
                      [model.base_icnn.layers[-1].raw_linear, model.base_icnn.layers[-1].raw_cubic])
            for param in params:
                param.copy_(inverse_positive(torch.nn.functional.softplus(param)*factor))
        return model, dict(configuration=config, calibration_factor=float(factor))
    finally:
        torch.set_default_dtype(original_dtype)


def normalized_response(model, physical_e, scales, *, create_graph):
    x = torch.as_tensor(physical_e, dtype=torch.float64)/scales["strain_scale"]
    return model.energy_and_stress(x.detach().clone().requires_grad_(True), create_graph=create_graph)


def normalized_reference_tangent(model, *, create_graph):
    x = torch.zeros((1, 3), dtype=torch.float64, requires_grad=True)
    _, s = model.energy_and_stress(x, create_graph=True)
    return torch.stack([torch.autograd.grad(s[0, k], x, retain_graph=True,
                       create_graph=create_graph)[0][0] for k in range(3)])


def training_objective(model, e, s, w, scales, d0, recipe):
    """Full-batch normalized loss, retaining all reference/feature dependencies."""
    wp, sp = normalized_response(model, e, scales, create_graph=True)
    ss, es = scales["strain_scale"], scales["energy_scale"]
    st = torch.as_tensor(s*ss/es, dtype=torch.float64)
    wt = torch.as_tensor(w/es, dtype=torch.float64)
    dt = torch.as_tensor(d0*ss**2/es, dtype=torch.float64)
    ls = (sp-st).square().mean()/scales["stress_denominator"]
    lw = (wp[:, 0]-wt).square().mean()/scales["energy_denominator"]
    ld = (normalized_reference_tangent(model, create_graph=True)-dt).square().mean()/scales["tangent_denominator"]
    weights = recipe["objective"]
    total = weights["stress_weight"]*ls+weights["energy_weight"]*lw+weights["reference_tangent_weight"]*ld
    if not torch.isfinite(total):
        raise RuntimeError("Nonfinite initial objective")
    return total, dict(stress=ls, energy=lw, reference_tangent=ld)


def source_paths():
    return [Path(__file__), PANN/"flexible_pann.py", PANN/"enriched_pann.py",
            BASE.parents[1]/"RVE_NeoHookean_Homogenization/pann/anisotropic/anisotropic_pann_model.py"]


def check_initializations(folder, labels, recipe_path=RECIPE):
    recipe, manifest, table = load_preparation(folder, recipe_path)
    if digest(labels) != recipe["labels_sha256"]:
        raise ValueError("Unapproved labels")
    with np.load(labels, allow_pickle=False) as store:
        arrays = load_fit_reference(store, recipe["selection_allowed_arrays"])
    e, s, w = (arrays[key] for key in ("E_fit", "S_fit", "W_fit"))
    scales = manifest["scales"]
    rows, comparisons = [], []
    for seed in recipe["models"]["seeds"]:
        pair = {}
        for name in recipe["models"]["names"]:
            print(json.dumps(dict(stage="initialization_only", name=name, seed=seed)), flush=True)
            model, metadata = build_initial_model(name, seed, recipe, manifest, table, e, s)
            total, parts = training_objective(model, e, s, w, scales, table["reference_tangent"], recipe)
            total.backward()  # Diagnostic only: never construct or step an optimizer.
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise RuntimeError("Nonfinite initialization gradient")
            wp, sp = normalized_response(model, e, scales, create_graph=False)
            tangent = normalized_reference_tangent(model, create_graph=False).detach().numpy()
            wz, sz = normalized_response(model, np.zeros((1, 3)), scales, create_graph=False)
            initial = dict(energy=wp.detach().numpy(), stress=sp.detach().numpy(), tangent=tangent,
                           core={k: v.detach().clone() for k, v in
                                 (model.base_icnn.state_dict().items() if name != "Free" else [])})
            if name != "Free":
                family = name.split("-")[0]
                if name.endswith("-fixed"):
                    pair[family] = initial
                else:
                    fixed = pair[family]
                    for key in initial["core"]:
                        torch.testing.assert_close(initial["core"][key], fixed["core"][key], rtol=2e-13, atol=2e-14)
                    for key in ("energy", "stress", "tangent"):
                        np.testing.assert_allclose(initial[key], fixed[key], rtol=2e-12, atol=2e-13)
                    comparisons.append(dict(core=family, seed=seed, passed=True,
                        max_normalized_energy_difference=float(np.abs(initial["energy"]-fixed["energy"]).max()),
                        max_normalized_stress_difference=float(np.abs(initial["stress"]-fixed["stress"]).max()),
                        max_normalized_tangent_difference=float(np.abs(tangent-fixed["tangent"]).max())))
            ref_stress = float(torch.linalg.vector_norm(sz).detach())*scales["energy_scale"]/scales["strain_scale"]
            if abs(float(wz.detach())) > 1e-12 or ref_stress > 1e-4:
                raise RuntimeError("Reference normalization check failed")
            rows.append(dict(name=name, seed=seed, **metadata,
                trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                initial_fit_objective=float(total.detach()),
                initial_loss_terms={key: float(value.detach()) for key, value in parts.items()},
                reference_energy_Pa=float(wz.detach())*scales["energy_scale"],
                reference_stress_norm_Pa=ref_stress, finite_gradients=True))
            del model, total, parts, wp, sp, initial
    result = dict(status="complete", passed=True, optimizer_steps=0,
        recipe_sha256=digest(recipe_path), feature_table_sha256=manifest["feature_table_sha256"],
        labels_sha256=digest(labels), selection_manifest_sha256=digest(folder/"manifest.json"),
        accessed_label_arrays=recipe["selection_allowed_arrays"], validation_used=False,
        test_used=False, paths_used=False, runs=rows, paired_initializations=comparisons,
        sources_sha256={str(p.resolve().relative_to(BASE.parents[1])): digest(p) for p in source_paths()},
        environment=dict(torch=torch.__version__, dtype="float64", device="cpu"),
        interpretation="Initialization and differentiability checks only; no trained accuracy, model selection or stability theorem. Calibration is a heuristic, not a bound on curvature or final error.")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--labels", type=Path, default=BASE/"results/data_labels_v1.npz")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Use a new initialization report path")
    result = check_initializations(args.features, args.labels)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps(dict(status="complete", runs=len(result["runs"]),
                         pairs=len(result["paired_initializations"]), optimizer_steps=0)), flush=True)


if __name__ == "__main__":
    main()
