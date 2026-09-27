#!/usr/bin/env python3
"""Open one evaluation gate of the SC-RVE constraint-by-capacity study.

Only the validation-selected models of the gate are evaluated. Metrics reuse
``audit_flexible.evaluate`` and the broad-energy, derivative and rank-one
sampling sequence of the SC-RVE m=6 audit. Constrained models also receive the
saved-weight nonnegative-energy certificate of Appendix C.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "07_material_b"))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT.parent / "RVE_NeoHookean_Homogenization" / "pann" / "anisotropic"))

from protocol.prepare_design import digest
from protocol.train_material_b import _atomic_json
from audit_flexible import energy_lower_bound_certificate, evaluate
from anisotropic_pann_model import AnisotropicFreeEnergy
from flexible_pann import FlexibleEnergy

CAMPAIGN = HERE / "results" / "sc_capacity_2x2_v1"
DATA = ROOT / "03_data" / "data.npz"


def load_model(path: Path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint["configuration"]
    if checkpoint["name"] == "Free":
        model = AnisotropicFreeEnergy(strain_scale=config["strain_scale"],
                                      feature_scale=torch.tensor(config["feature_scale"]),
                                      widths=tuple(config["widths"])).double()
    else:
        model = FlexibleEnergy(**config).double()
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return model, checkpoint


def audit(model, checkpoint, data) -> dict:
    ss, es = checkpoint["strain_scale"], checkpoint["energy_scale"]
    record = {}
    if checkpoint["name"] != "Free":
        record["nonnegative_energy_certificate"] = energy_lower_bound_certificate(model)
    rng = np.random.default_rng(101)
    angle = rng.uniform(-np.pi, np.pi, 6000)
    lam = np.exp(rng.uniform(np.log(.25), np.log(3.), (6000, 2)))
    co, si = np.cos(angle), np.sin(angle)
    c11 = lam[:, 0]**2*co**2+lam[:, 1]**2*si**2
    c22 = lam[:, 0]**2*si**2+lam[:, 1]**2*co**2
    c12 = (lam[:, 0]**2-lam[:, 1]**2)*co*si
    e = np.column_stack(((c11-1)/2, (c22-1)/2, c12))
    sample = model.energy(torch.tensor(e/ss, dtype=torch.float64)).detach().numpy()
    record["broad_energy_sample"] = dict(count=len(e), stretch_range=[.25, 3.],
        finite=bool(np.isfinite(sample).all()), minimum=float(sample.min())*es,
        negative_count=int((sample < -1e-8).sum()))
    tiny = torch.tensor([[.02, -.01, .03], [.1, -.04, -.08]], dtype=torch.float64,
                        requires_grad=True)/ss
    record["gradcheck"] = torch.autograd.gradcheck(model.energy, (tiny,), atol=1e-5, rtol=1e-4)
    record["gradgradcheck"] = torch.autograd.gradgradcheck(model.energy, (tiny,), atol=1e-5, rtol=1e-4)
    minimum, symmetry = float("inf"), 0.
    for _ in range(24):
        f = torch.eye(2, dtype=torch.float64)+.12*torch.tensor(rng.normal(size=(2, 2)), dtype=torch.float64)
        def energy_F(flat):
            ff = flat.reshape(2, 2); c = ff.T@ff
            ee = torch.stack(((c[0, 0]-1)/2, (c[1, 1]-1)/2, c[0, 1]))
            return model.energy(ee[None, :]/ss)[0, 0]
        h = torch.autograd.functional.hessian(energy_F, f.flatten()).detach().numpy()*es
        symmetry = max(symmetry, float(abs(h-h.T).max()))
        aa = rng.normal(size=(100, 2)); bb = rng.normal(size=(100, 2))
        aa /= np.linalg.norm(aa, axis=1)[:, None]; bb /= np.linalg.norm(bb, axis=1)[:, None]
        rank = np.einsum("ni,nj->nij", aa, bb).reshape(-1, 4)
        minimum = min(minimum, float(np.einsum("ni,ij,nj->n", rank, h, rank).min()))
    record["rank_one_sample"] = dict(count=2400, minimum_Pa=minimum, hessian_max_asymmetry_Pa=symmetry)
    ck = dict(strain_scale=ss, energy_scale=es)
    record["metrics"] = {split: evaluate(model, ck, *[data[f"{key}_{split}"] for key in ("E", "S", "W")])
                         for split in ("test", "probe")}
    labels = data["labels_probe"].astype(str)
    rings = np.array([float(label.rsplit("_", 1)[1][:-1]) for label in labels])
    record["probe_by_ring"] = {str(ring): evaluate(model, ck, *[data[f"{key}_probe"][rings == ring]
                                                                for key in ("E", "S", "W")])
                               for ring in sorted(set(rings))}
    record["trainable_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", required=True, choices=("A", "B"))
    args = parser.parse_args()
    output = CAMPAIGN / f"independent_audit_gate{args.gate}.json"
    if output.exists():
        raise FileExistsError("Gate already opened; refusing to reopen it")
    selection_path = CAMPAIGN / "training" / f"validation_selection_gate{args.gate}.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("status") != "frozen_before_test_probe" or selection.get("test_probe_accessed") is not False:
        raise RuntimeError("Gate selection is not frozen before test/probe")
    for row in selection["selected"]:
        if digest(Path(row["checkpoint"])) != row["checkpoint_sha256"]:
            raise RuntimeError(f"Checkpoint hash changed for {row['slug']}")
    torch.set_num_threads(2)
    gate = dict(status="opened_once", gate=args.gate,
                opened_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                validation_selection_sha256=digest(selection_path), data_sha256=digest(DATA),
                auditor_sha256=digest(Path(__file__)))
    _atomic_json(CAMPAIGN / f"test_probe_gate{args.gate}.json", gate)
    data = np.load(DATA)
    report = {"gate": gate}
    for row in selection["selected"]:
        model, checkpoint = load_model(Path(row["checkpoint"]))
        record = audit(model, checkpoint, data)
        record.update(cell=row["cell"], seed=row["seed"], validation_score=row["best_validation_score"])
        if checkpoint["name"] == "Free":
            # Unchanged tensors in the coupon-law format, with the cell widths.
            fe2 = CAMPAIGN / f"fe2_{row['cell']}_checkpoint.pt"
            torch.save(dict(kind="free", state_dict=checkpoint["state_dict"],
                            strain_scale=checkpoint["strain_scale"], energy_scale=checkpoint["energy_scale"],
                            widths=list(checkpoint["configuration"]["widths"]),
                            err_test=record["metrics"]["test"]["stress"],
                            err_probe=record["metrics"]["probe"]["stress"],
                            source_checkpoint=str(Path(row["checkpoint"]).relative_to(ROOT)),
                            source_checkpoint_sha256=row["checkpoint_sha256"]), fe2)
            record["fe2_checkpoint"] = dict(path=str(fe2.relative_to(ROOT)), sha256=digest(fe2))
        report[str(Path(row["checkpoint"]).relative_to(ROOT))] = record
        print(json.dumps(dict(cell=row["cell"], seed=row["seed"],
                              test_stress=record["metrics"]["test"]["stress"],
                              probe_stress=record["metrics"]["probe"]["stress"])), flush=True)
    _atomic_json(output, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
