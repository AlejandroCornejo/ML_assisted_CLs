"""Behavior beyond the data of the fifteen fits reported in Table 3.

The two earlier exploratory rules are applied unchanged to the twelve m=6 fits and
the same-rule Unconstrained energy, which replace the 15-model-campaign checkpoints:
ROBUSTNESS_AUDIT_INTERNAL.md (volume collapse and the bounded search in the approved
and double-size boxes) and DIRECTED_SEARCH_INTERNAL.md (search in principal stretches
0.55--1.50, nearest witness, verification, matched comparisons).  The executed search
routines are those of audit_robustness.py and directed_search.py; only the model
loader and the output folder are rebound here.  The directed search now runs on every
fit, not only on the Unconstrained energy, so that its outcome is compared on equal
terms.  The FOM follow-up of the earlier rule is not repeated.  For the figure, the
minimum rank-one curvature is also recorded along equibiaxial compression F=sqrt(J) I,
down to the stretch bound 0.55 of the directed search.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

import protocol.directed_search as directed
from protocol.audit_mc_m06_mechanics_v1 import FREE_SEEDS, load_free
from protocol.audit_robustness import acoustic, collapse, search, sqrt_c
from protocol.evaluate_feature_count_m06_v1 import (FEATURES, OUTPUT as EVALUATION,
                                                    TRAINING as M06_TRAINING, _load_model)
from protocol.evaluate_final_models import predict
from protocol.prepare_design import digest
from protocol.select_features import BASE
from protocol.train_material_b import _atomic_json


OUTPUT = BASE / "results/feature_count_analysis_v1/m06_beyond_data_v1"
HERE = Path(__file__).resolve().parent
EQUIBIAXIAL_J = np.linspace(1., .55**2, 141)


def equibiaxial_path(model, scales):
    """Minimum rank-one curvature along F=sqrt(J) I: 180 unit b, exact minimum over a."""
    e = np.column_stack(((EQUIBIAXIAL_J-1)/2, (EQUIBIAXIAL_J-1)/2, np.zeros_like(EQUIBIAXIAL_J)))
    _, s, d = predict(model, e, scales)
    f = sqrt_c(e)
    minimum = np.full(len(e), np.inf)
    for theta in np.arange(180)*np.pi/180:
        minimum = np.minimum(minimum, acoustic(f, s, d, theta)[0])
    return dict(J=EQUIBIAXIAL_J.tolist(), minimum_curvature_Pa=minimum.tolist(), b_directions=180)


def matched_curvature(model, scales, witness):
    """Rank-one curvature in the witness direction, as in directed_search.main."""
    e = np.array([witness["strain"]])
    _, s, d = predict(model, e, scales)
    f = sqrt_c(e)
    h = np.outer(witness["a"], witness["b"])
    ed = (f[0].T@h+h.T@f[0])/2
    ev = np.array([ed[0, 0], ed[1, 1], 2*ed[0, 1]])
    stress = np.array([[s[0, 0], s[0, 2]], [s[0, 2], s[0, 1]]])
    return float(ev@d[0]@ev+np.sum(stress*(h.T@h)))


def main() -> int:
    torch.set_num_threads(2)
    OUTPUT.mkdir(parents=True, exist_ok=False)
    scales = json.loads((FEATURES / "manifest.json").read_text(encoding="utf-8"))["scales"]
    evaluation = json.loads((EVALUATION / "per_run.json").read_text(encoding="utf-8"))
    models = {}
    for row in evaluation["rows"]:
        if digest(M06_TRAINING / row["slug"] / "model.pt") != row["model_sha256"]:
            raise ValueError(f"Checkpoint differs from the evaluated model: {row['slug']}")
        models[row["slug"]] = (dict(model=row["model"], seed=row["seed"], slug=row["slug"],
                                    model_sha256=row["model_sha256"]), _load_model(row))
    for seed in FREE_SEEDS:
        identity, model = load_free(seed, scales)
        models[identity["slug"]] = (identity, model)
    _atomic_json(OUTPUT / "specification.json", dict(
        status="specified_before_execution", unix_time=time.time(),
        source_sha256=digest(Path(__file__)),
        robustness_rule_sha256=digest(HERE / "ROBUSTNESS_AUDIT_INTERNAL.md"),
        directed_rule_sha256=digest(HERE / "DIRECTED_SEARCH_INTERNAL.md"),
        robustness_source_sha256=digest(HERE / "audit_robustness.py"),
        directed_source_sha256=digest(HERE / "directed_search.py"),
        checkpoints={slug: identity["model_sha256"] for slug, (identity, _) in models.items()}))
    directed.OUTPUT = OUTPUT
    directed.load_model = lambda row: models[row["slug"]][1]
    rows = []
    for slug, (identity, model) in models.items():
        start = time.monotonic()
        result = dict(identity)
        result["collapse"] = collapse(model, scales, identity["model"] == "Unconstrained")
        result["box_search"] = [search(model, scales, factor) for factor in (1, 2)]
        result["directed_search"] = directed.run_search(identity, scales)
        result["equibiaxial_path"] = equibiaxial_path(model, scales)
        result["seconds"] = time.monotonic()-start
        rows.append(result)
        _atomic_json(OUTPUT / f"{slug}.json", result)
        print(slug, [round(s["minimum_Pa"]/1e6, 2) for s in result["box_search"]],
              round(result["directed_search"]["most_negative"]["curvature_Pa"]/1e6, 2),
              f"MPa, {result['seconds']:.0f}s", flush=True)
    comparisons = []
    for row in rows:
        witness = row["directed_search"]["nearest"]
        if witness and witness["negative_verified"]:
            comparisons.append(dict(witness_slug=row["slug"], strain=witness["strain"],
                                    det_F=witness["det_F"],
                                    models={slug: matched_curvature(model, scales, witness)
                                            for slug, (_, model) in models.items()}))
    _atomic_json(OUTPUT / "summary.json", dict(status="complete", rows=rows,
                                               matched_comparisons=comparisons))
    print(OUTPUT / "summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
