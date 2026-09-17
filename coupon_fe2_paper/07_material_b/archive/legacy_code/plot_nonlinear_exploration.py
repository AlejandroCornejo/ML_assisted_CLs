#!/usr/bin/env python3
"""Consolidate completed nonlinear exploration and plot saved stresses only."""
import os
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/coupon-b-nonlinear-mpl")
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(a, b):
    return float(np.linalg.norm(np.array(a)-b)/max(np.linalg.norm(b), 1.))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=HERE / "nonlinear_reference_v2")
    parser.add_argument("--check", type=Path, default=HERE / "nonlinear_check_v1")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    reference, check = [json.loads((folder / "report.json").read_text()) for folder in (args.reference, args.check)]
    if any(r["status"] != "complete" for r in (reference, check)):
        raise ValueError("Wait for both complete exploration reports")
    if reference["spec_sha256"] != check["spec_sha256"]:
        raise ValueError("Different exploration specifications")
    spec = check["spec"]
    expected = {(p, i+1) for p, targets in spec["paths"].items() for i in range(len(targets))}
    def exact_targets(report):
        return (len(report["states"]) == len(expected)
                and {(s["path"], s["level"]) for s in report["states"]} == expected
                and all(s["ok"] and s["strain"] == spec["paths"][s["path"]][s["level"]-1]
                        and s["physical_screen_passed"] for s in report["states"]))
    checks = [dict(check="all predeclared targets reached and screened", passed=all(map(exact_targets, (reference, check))))]
    refs = {(s["path"], s["level"]):s for s in reference["states"] if s["ok"]}
    comparisons = []
    for row in check["states"]:
        ref = refs.get((row["path"], row["level"]))
        if row["ok"] and ref is not None:
            comparisons.append(dict(path=row["path"], level=row["level"],
                **{k:relative(ref[k], row[k]) for k in ("stress", "tangent", "energy")},
                **{k:relative(ref["fields"][k], row["fields"][k]) for k in ("pk1_l2", "pk1_max")}))
    checks.append(dict(check="all cross-mesh comparisons present", passed=len(comparisons) == len(expected)))
    for key, threshold in dict(spec["reference_output_tolerances"], **spec["field_statistic_tolerances"]).items():
        worst = max(comparisons, key=lambda s:s[key], default=None)
        checks.append(dict(check="cross-mesh "+key, value=None if worst is None else worst[key],
            threshold=threshold, worst=None if worst is None else [worst["path"], worst["level"]],
            passed=worst is not None and worst[key] <= threshold))
    states = [s for r in (reference, check) for s in r["states"] if s["ok"]]
    derivatives = [d for s in states for d in s.get("derivatives", [])]
    checks.append(dict(check="all selected derivative checks present", passed=len(derivatives) == 2*len(spec["fd_steps"])))
    for key in ("energy_gradient_relative_error", "tangent_relative_error", "fd_tangent_relative_asymmetry"):
        value = max((d[key] for d in derivatives), default=None)
        checks.append(dict(check=key, value=value, threshold=1e-4, passed=value is not None and value <= 1e-4))
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=False)
    result = dict(status="complete", passed=all(c["passed"] for c in checks), checks=checks,
        comparisons=comparisons, spec=spec, source_sha256={str(folder / "report.json"):digest(folder / "report.json")
                                                       for folder in (args.reference, args.check)},
        script_sha256=digest(Path(__file__)), baseline=check["baseline"], extended_states=check["states"],
        minimum_micro_J=min((s["fields"]["min_micro_J"] for s in states), default=None),
        minimum_polygon_gap=min((s["boundary"]["min_polygon_gap"] for s in states), default=None),
        minimum_sampled_rank_one_curvature=min((s["sampled_min_rank_one_curvature"] for s in states), default=None),
        rejected_increments=sum(not a["ok"] for s in states for a in s["attempts"]),
        scope="Finite exploratory ray endpoints, not approval of a larger training box, full stability "
              "or uniqueness. Linear deviation is ||s-D0 e||/||s|| with same-mesh D0. Plot lines are guides; "
              "only saved states and the analytic zero reference are shown. No learned models evaluated.")
    (output / "decision.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({"text.usetex":False, "font.family":"DejaVu Sans", "font.size":11,
                        "axes.spines.top":False, "axes.spines.right":False})
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.6))
    colors = ("#0072B2", "#D55E00", "#009E73")
    D0 = np.array(check["reference"]["tangent"])
    names = ("Tracción X", "Tracción Y", "Green-cortante positivo")
    for axis, (path, targets), index, name in zip(axes, spec["paths"].items(), range(3), names):
        old = [s for s in check["baseline"] if s["path"] == path]
        new = [s for s in check["states"] if s["path"] == path and s["ok"]]
        rows = sorted(old+new, key=lambda s:s["strain"][index])
        amplitude = np.array([0.]+[s["strain"][index] for s in rows])
        values = np.vstack([np.zeros(3), [s["stress"] for s in rows]])/1e6
        direction = np.eye(3)[index]
        axis.axvspan(0, 100*max(s["strain"][index] for s in old), color="0.94", zorder=0)
        for j, color in enumerate(colors):
            axis.plot(100*amplitude, values[:, j], color=color, lw=1.8)
            axis.plot(100*amplitude, amplitude*float((D0@direction)[j])/1e6,
                      color=color, ls="--", lw=1.4, alpha=.8)
            axis.scatter([100*s["strain"][index] for s in old], [s["stress"][j]/1e6 for s in old],
                         color=color, s=25, zorder=3)
            axis.scatter([100*s["strain"][index] for s in new], [s["stress"][j]/1e6 for s in new],
                         color=color, marker="s", s=42, edgecolors="white", linewidths=.5, zorder=4)
        first = max(old, key=lambda s:s["strain"][index])["nonlinearity"]["stress_linear_deviation"]
        last = max(new, key=lambda s:s["strain"][index])["nonlinearity"]["stress_linear_deviation"] if new else first
        axis.set_title(f"{name}\nDesviación vectorial: {first:.1%} → {last:.1%}", fontsize=12)
        axis.set(xlabel=(r"$E_{11}$ [%]", r"$E_{22}$ [%]", r"$2E_{12}$ [%]")[index],
                 ylabel="Tensión material [MPa]")
        axis.axhline(0, color="0.7", lw=.7)
        axis.grid(alpha=.2)
    handles = [Line2D([], [], color=color, label=label) for color, label in zip(colors, (r"$S_{11}$", r"$S_{22}$", r"$S_{12}$"))]
    handles += [Line2D([], [], color="0.3", lw=1.8, label="FOM: 8,961 elementos"),
                Line2D([], [], color="0.3", ls="--", label=r"Linealización en reposo: $D_0 e$")]
    fig.legend(handles=handles, ncol=5, loc="lower center", bbox_to_anchor=(.5, .10), frameon=False)
    fig.suptitle("Material B · comparación física con la predicción lineal", fontsize=15, y=.96)
    fig.text(.5, .025, "Sombreado: rango anterior de cada trayectoria · Círculos: estados previos · Cuadrados: nuevos estados exploratorios\n"
             "Desviación = ||s − D₀e|| / ||s||; porcentajes en el extremo anterior → nuevo · Líneas FOM: guías, no estados adicionales",
             ha="center", fontsize=9, color="0.35")
    fig.subplots_adjust(left=.065, right=.99, top=.78, bottom=.25, wspace=.32)
    for suffix in ("png", "pdf"):
        fig.savefig(output / ("nonlinear_response."+suffix), dpi=160)
    plt.close(fig)
    print("Exploratory screen passed:", result["passed"])
    print(checks)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
