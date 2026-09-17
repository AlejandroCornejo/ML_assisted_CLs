"""Fit-only constant-tangent diagnostic for A; no neural or FE campaign."""
import argparse
import json
from pathlib import Path
import numpy as np
from audit_reference_linearity import diagnose
from summarize_box_extension import digest

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=HERE.parent / "03_data/data.npz")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.data) as data:
        e, s = data["E_train"], data["S_train"]
        et, st = data["E_test"], data["S_test"]
    if not all(np.isfinite(a).all() for a in (e, s, et, st)):
        raise ValueError("Nonfinite input; do not silently change the existing partitions")
    order = np.random.default_rng(5).permutation(len(e))
    vi, fi = order[:742], order[742:]
    coefficients, _residuals, rank, singular_values = np.linalg.lstsq(e[fi], s[fi], rcond=None)
    if rank != 3:
        raise ValueError("Fit strain matrix is not full rank")
    Dfit = coefficients.T
    partitions = {name:diagnose(ex, sx, Dfit) for name, ex, sx in
                  (("fit", e[fi], s[fi]), ("validation", e[vi], s[vi]), ("test", et, st))}
    result = dict(status="complete", script_sha256=digest(__file__),
        source_sha256={str(p.resolve()):digest(p) for p in
                      (args.data, HERE / "audit_reference_linearity.py", HERE / "summarize_box_extension.py")},
        existing_partition_recipe=dict(seed=5, validation_count=742, fit_count=len(fi), independent_test_count=len(et)),
        diagnostic="Single least-squares stress map s=Dfit e through the origin, fit partition only; no intercept or tuning",
        definitions=dict(strain="e=(E11,E22,2E12)", stress="s=(S11,S22,S12)",
                         aggregate="||s-Dfit e||_all_states / ||s||_all_states"),
        Dfit_Pa=Dfit.tolist(), fit_rank=int(rank), strain_singular_values=singular_values.tolist(), partitions=partitions,
        scope="Empirical linear approximation diagnostic on saved A data, not a universal best-linear-error bound. "
        "Dfit is unconstrained: stress-potential integrability, symmetry and stability are not enforced. It is not the existing "
        "neural Regression baseline, a trained energy, or a compute-matched constitutive-model comparison. No validation/test "
        "labels select its coefficients; no neural checkpoint selection, FOM solves or A source/output edits.")
    with args.out.open("x") as f:
        f.write(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({k:dict(count=v["total"], relative_L2=v["aggregate_relative_L2"]) for k, v in partitions.items()}, indent=2))


if __name__ == "__main__":
    main()
