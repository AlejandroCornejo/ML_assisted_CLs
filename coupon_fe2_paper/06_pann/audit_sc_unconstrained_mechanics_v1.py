#!/usr/bin/env python3
"""Cycle and rank-one audit of the same-protocol SC-RVE Unconstrained energy.

Only the Unconstrained-energy ("Free") entries are recomputed. The states,
directions and cycle are those of ``audit_sc_m06_mechanics_v1.py``: the same
random seed and call order regenerate identical clouds, which is verified
against the stored state counts. Regression, ICNN and ICKAN entries are copied
unchanged from the m=6 mechanics audit; as a control, ICNN and ICKAN are
re-evaluated on the held-out test cloud and must reproduce their stored minima.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

import mechanics_witnesses as witness
from free_pann_law import CouponFreePANNLaw
from flexible_pann_law import FlexiblePANNLaw

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CAMPAIGN = HERE / "results" / "sc_unconstrained_v1"
BASE_AUDIT = HERE / "results" / "sc_m06_learned_v1" / "mechanics_audit.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=CAMPAIGN / "mechanics_audit.json")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    independent = json.loads((CAMPAIGN / "independent_audit.json").read_text())
    if independent["gate"]["status"] != "opened_once":
        raise RuntimeError("Independent audit has not been completed")
    record = next(value for key, value in independent.items() if key != "gate")
    fe2_path = ROOT / record["fe2_checkpoint"]["path"]
    if witness.sha256(fe2_path) != record["fe2_checkpoint"]["sha256"]:
        raise RuntimeError("Repackaged Unconstrained-energy checkpoint changed")
    base = json.loads(BASE_AUDIT.read_text())
    free = CouponFreePANNLaw(fe2_path)

    # Control: the constrained laws must reproduce their stored test minima.
    data = np.load(ROOT / "03_data" / "data.npz")
    control = {}
    for name in ("ICNN", "ICKAN"):
        law = FlexiblePANNLaw(Path(base["checkpoints"][name]["path"]))
        value = witness.min_rank_one_curvature(law, data["E_test"], 180)["curvature_Pa"]
        stored = base["rank_one"]["clouds"]["held_out_test"]["models"][name]["curvature_Pa"]
        if abs(value - stored) > 1e-9 * abs(stored):
            raise RuntimeError(f"{name} control does not reproduce the m=6 audit")
        control[name] = dict(recomputed_Pa=value, stored_Pa=stored)

    cycle = witness.cycle_audit({"Free": free}, base["cycle"]["quadrature_order_per_edge"])

    # Same seed and call order as mechanics_witnesses.rank_one_audit.
    valid_probe = np.isfinite(data["S_probe"]).all(axis=1) & np.isfinite(data["W_probe"])
    rng = np.random.default_rng(base["rank_one"]["random_seed"])
    rings = np.asarray(base["rank_one"]["radial_box_expansion_rings"], dtype=float)
    n_radial = 1800
    radial_states, radial_tags = witness.radial_box_states(rng, n_radial, rings)
    clouds = {
        "held_out_test": (data["E_test"], None),
        "converged_probe": (data["E_probe"][valid_probe], None),
        "uniform_training_box": (witness.uniform_box_states(rng, 12000), None),
        "radial_box_expansion": (radial_states, radial_tags),
        "broad_unvalidated_principal_stretch_scan": (witness.broad_F_states(rng, 24000), None),
    }
    free_rank_one = {}
    for cloud_name, (states, tags) in clouds.items():
        stored = base["rank_one"]["clouds"][cloud_name]["models"]["Free"]
        if len(states) != stored["n_states"]:
            raise RuntimeError(f"Cloud {cloud_name} differs from the m=6 audit")
        entry = witness.min_rank_one_curvature(
            free, states, stored["n_b_directions"], tags=tags,
            tag_name="ring_factor" if tags is not None else None)
        entry["finite_difference_energy_check"] = witness.finite_difference_candidate(free, entry)
        free_rank_one[cloud_name] = entry

    report = copy.deepcopy(base)
    report["scope"] = ("same-protocol SC-RVE Unconstrained energy recomputed on the states, "
                       "directions and cycle of the m=6 mechanics audit; Regression, ICNN and "
                       "ICKAN entries copied unchanged from that audit")
    report["base_audit"] = dict(path=str(BASE_AUDIT.relative_to(ROOT)), sha256=witness.sha256(BASE_AUDIT))
    report["checkpoints"]["Free"] = dict(path=str(fe2_path), sha256=witness.sha256(fe2_path),
                                         source_checkpoint=record["fe2_checkpoint"])
    report["checkpoints"]["Free_legacy_replaced"] = base["checkpoints"]["Free"]
    report["cycle"]["models"]["Free"] = cycle["models"]["Free"]
    for order, values in cycle["quadrature_work_convergence_J_per_m3"].items():
        report["cycle"]["quadrature_work_convergence_J_per_m3"][order]["Free"] = values["Free"]
    for cloud_name, entry in free_rank_one.items():
        report["rank_one"]["clouds"][cloud_name]["models"]["Free"] = entry
    report["control_constrained_reproduction"] = control
    clean = witness.as_json(report)
    args.output.write_text(json.dumps(clean, indent=2, allow_nan=False) + "\n")
    radial = clean["rank_one"]["clouds"]["radial_box_expansion"]["models"]["Free"]
    print(json.dumps(dict(
        output=str(args.output),
        cycle_work_64_J_per_m3=clean["cycle"]["models"]["Free"]["stress_work_J_per_m3"],
        min_curvature_MPa={name: clean["rank_one"]["clouds"][name]["models"]["Free"]["curvature_Pa"] / 1e6
                           for name in clouds},
        first_negative_radial=radial.get("first_negative_by_ring_factor")), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
