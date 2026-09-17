"""Quantify saved A/B departure from reference linear stress; no FOM or training."""
import argparse
import json
from pathlib import Path
import numpy as np
from summarize_box_extension import digest

HERE = Path(__file__).resolve().parent


def diagnose(e, stress, tangent):
    e, stress, tangent = np.asarray(e), np.asarray(stress), np.asarray(tangent)
    finite = np.isfinite(e).all(axis=1) & np.isfinite(stress).all(axis=1)
    strains, values = e[finite], stress[finite]
    if not finite.any():
        raise ValueError("No finite stress/strain pairs")
    prediction = strains@tangent.T
    error = values-prediction
    deviations = np.linalg.norm(error, axis=1)/np.linalg.norm(values, axis=1).clip(1.)
    worst = int(np.argmax(deviations))
    return dict(total=len(e), finite=int(finite.sum()), excluded_nonfinite=int((~finite).sum()),
        stress_norm_below_one_Pa=int((np.linalg.norm(values, axis=1) < 1.).sum()),
        sampled_strain_min=strains.min(axis=0).tolist(), sampled_strain_max=strains.max(axis=0).tolist(),
        aggregate_relative_L2=float(np.linalg.norm(error)/max(np.linalg.norm(values), 1.)),
        per_state_quantiles=dict(zip(("median", "p95", "maximum"), np.quantile(deviations, [.5, .95, 1]).tolist())),
        worst_strain=strains[worst].tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a-data", type=Path, default=HERE.parent / "03_data/data.npz")
    parser.add_argument("--a-c0", type=Path, default=HERE.parent / "00_rve/C0_periodic.npz")
    parser.add_argument("--b-check", type=Path, default=HERE / "nonlinear_check_v1/report.json")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.a_c0) as data:
        D0a = data["C0_periodic"]
    with np.load(args.a_data) as data:
        a = {split:diagnose(data["E_"+split], data["S_"+split], D0a) for split in ("train", "test")}
        elements = int(data["n_elements"])
    b = json.loads(args.b_check.read_text())
    expected = {(path, i+1):strain for path, targets in b["spec"]["paths"].items()
                for i, strain in enumerate(targets)}
    if (b["status"] != "complete" or len(b["states"]) != len(expected)
            or {(s["path"], s["level"]) for s in b["states"]} != set(expected)
            or not all(s["ok"] and s["strain"] == expected[s["path"], s["level"]] for s in b["states"])):
        raise ValueError("Require completed B ray exploration")
    endpoints = [s for s in b["states"] if s["level"] == 2]
    result = dict(status="complete", script_sha256=digest(__file__),
        source_sha256={str(p.resolve()):digest(p) for p in (args.a_data, args.a_c0, args.b_check)},
        definitions=dict(strain="e=(E11,E22,2E12)", stress="s=(S11,S22,S12)",
            linear_prediction="D0 e", aggregate="||s-D0 e||_all_states / ||s||_all_states",
            per_state="||s-D0 e|| / max(||s||,1 Pa); Euclidean engineering-vector norm"),
        A=dict(n_elements=elements, baseline="Saved independent periodic linear-elastic C0; six-point quadrature versus three in finite-strain FOM. "
            "This is a saved-reference comparison, not an assertion of bitwise-identical tangent assemblies.", partitions=a),
        B=dict(baseline="Same-mesh FOM tangent evaluated at zero before the exploration",
               endpoints=[dict(path=s["path"], strain=s["strain"], nonlinearity=s["nonlinearity"]) for s in endpoints]),
        scope="Observed departure from a fixed reference linear law, not a lower error bound for every fitted linear regression, "
        "proof of inelasticity or approval of larger strain combinations. No learned models, model selection or training. "
        "A sample extrema are not a specification of its domain. A source artifacts are read only.")
    with args.out.open("x") as f:
        f.write(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps(a, indent=2))


if __name__ == "__main__":
    main()
