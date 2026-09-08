#!/usr/bin/env python3
"""Cruciform analog of compare_true_fom_speedup_accuracy_claude.py
(Cook's membrane): once all 8 models' n_body=6 results exist, compute,
against FOM-FE2 as the true multiscale reference:
  - speedup = true_FOM_wall_time / row_wall_time
  - tip reaction-force relative error (the natural QoI here, since this
    problem is DISPLACEMENT-controlled, not force-controlled like Cook --
    there is no "tip u_y range" to report; the reaction force at the
    pulled tip is the physically meaningful output scalar instead)
  - full 600-Gauss-point final-state stress relative L2 error (identical
    convention to Cook's own S-err column: all rows already report the
    energy-conjugate/reaction-force macro stress, by this project's own
    prior unification work)

Wall times are parsed from this run's own log files (never hardcoded --
unlike Cook's script, which hardcoded previously-verified numbers; here
everything was just computed in this same session, so parse it directly).
Prints a plain summary table AND writes ready-to-paste LaTeX table source
(cruciform_fe2_table_claude.tex), matching Table~\\ref{tab:cook-fe2}'s
own column structure and footnote style.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

FE2_DIR = Path("/home/kratos/ML_assisted_CLs_clean/RVE_NeoHookean_Homogenization/fe2_extension")
PAPER_DIR = Path("/home/kratos/ML_assisted_CLs_clean/RVE_NeoHookean_Homogenization/pann/anisotropic")
SCRATCH = Path("/tmp/claude-1000/-home-kratos-ML-assisted-CLs-clean/4a3da423-8dfc-40e5-a7fa-8fa40d3c8b2f/scratchpad")

FAST_BATCH_LOG = SCRATCH / "nbody6_full_suite.log"
FOM_LOG = SCRATCH / "fom_nbody6_full_claude.log"

ROW_ORDER = ("fom_nested_consistent_parallel", "pann_regression", "pann_free", "pann_certified",
             "pann_ickan", "linear_hprom_parallel_continuation", "hprom_ann_parallel_continuation",
             "dhprom_ann_parallel")
LABELS = {
    "fom_nested_consistent_parallel": "FOM-FE$^2$",
    "pann_regression": "Regression (tier 1)",
    "pann_free": "Free hyperelastic (tier 2)",
    "pann_certified": "Polyconvex ICNN (tier 3a)",
    "pann_ickan": "Polyconvex ICKAN (tier 3b)",
    "linear_hprom_parallel_continuation": "Linear-HPROM-FE$^2$",
    "hprom_ann_parallel_continuation": "HPROM--ANN-FE$^2$",
    "dhprom_ann_parallel": "D-HPROM--ANN-FE$^2$",
}

WALL_RE = re.compile(r"^\[(?P<which>[\w.]+)\].*?wall=(?P<wall>[0-9.]+)s")


def parse_wall_times() -> dict:
    walls = {}
    for log_path in (FAST_BATCH_LOG, FOM_LOG):
        if not log_path.exists():
            continue
        for line in log_path.read_text().splitlines():
            m = WALL_RE.match(line)
            if m:
                walls[m.group("which")] = float(m.group("wall"))
    return walls


def relative_l2(prediction: np.ndarray, reference: np.ndarray) -> float:
    return float(np.linalg.norm(prediction - reference) / max(np.linalg.norm(reference), 1.0e-30))


def newton_behavior(d) -> str:
    iters = np.asarray(d["iters_per_step"])
    status = np.asarray(d["status_per_step"])
    all_converged = bool(np.all(status == "converged"))
    if all_converged:
        lo, hi = int(iters.min()), int(iters.max())
        return f"converged, {lo} it./step" if lo == hi else f"converged, {lo}-{hi} it./step"
    n_bad = int(np.sum(status != "converged"))
    return f"{n_bad}/{len(status)} steps not converged"


def main() -> None:
    walls = parse_wall_times()
    missing = [w for w in ROW_ORDER if w not in walls]
    if missing:
        print(f"[warning] no parsed wall time yet for: {missing}")

    data = {}
    for which in ROW_ORDER:
        path = FE2_DIR / f"cruciform_results_{which}_claude.npz"
        if not path.exists():
            print(f"  [skip] {which}: {path.name} not yet available")
            continue
        d = np.load(path)
        assert len(d["iters_per_step"]) == 20, f"{which}: {path.name} has {len(d['iters_per_step'])} steps, not 20"
        data[which] = d

    if "fom_nested_consistent_parallel" not in data:
        raise SystemExit("FOM-FE2 result not available yet -- cannot compute errors/speedup without the reference.")

    true = data["fom_nested_consistent_parallel"]
    true_coords, true_tris = np.asarray(true["coords"]), np.asarray(true["tris"])
    true_s_gp = np.asarray(true["s_gp"], dtype=np.float64)
    true_reaction_px = float(true["reaction_px"])
    true_wall = walls.get("fom_nested_consistent_parallel")

    rows = []
    for which in ROW_ORDER:
        if which not in data:
            continue
        d = data[which]
        assert np.allclose(d["coords"], true_coords), f"{which}: mesh coords mismatch vs FOM"
        assert np.array_equal(d["tris"], true_tris), f"{which}: mesh connectivity mismatch vs FOM"
        wall = walls.get(which)
        reaction_px = float(d["reaction_px"])
        s_err = relative_l2(np.asarray(d["s_gp"], dtype=np.float64), true_s_gp)
        r_err = abs(reaction_px - true_reaction_px) / max(abs(true_reaction_px), 1e-30)
        is_fom = which == "fom_nested_consistent_parallel"
        rows.append({
            "which": which, "label": LABELS[which], "behavior": newton_behavior(d),
            "reaction_px": reaction_px, "wall": wall,
            "speedup": None if is_fom or wall is None or true_wall is None else true_wall / wall,
            "r_err": None if is_fom else r_err, "s_err": None if is_fom else s_err,
        })

    print(f"\n{'Model':<26}{'Newton behavior':<22}{'Reaction F_x [MN]':>18}{'Wall [s]':>12}"
          f"{'Speedup':>10}{'F_x err':>10}{'S err':>10}")
    for r in rows:
        wall_str = f"{r['wall']:.1f}" if r["wall"] is not None else "?"
        sp_str = f"{r['speedup']:.1f}x" if r["speedup"] is not None else "--"
        re_str = f"{r['r_err']:.2%}" if r["r_err"] is not None else "--"
        se_str = f"{r['s_err']:.2%}" if r["s_err"] is not None else "--"
        print(f"{r['label']:<26}{r['behavior']:<22}{r['reaction_px'] / 1e6:>18.1f}{wall_str:>12}"
              f"{sp_str:>10}{re_str:>10}{se_str:>10}")

    # ready-to-paste LaTeX table, mirroring tab:cook-fe2's own structure
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Cruciform specimen at $n_{\rm body}=6$ ($200$ elements, $600$ Gauss",
        r"points), full $20$-step ramp, every model this paper has trained or built",
        r"as the macroscopic material law, plus the true, non-reduced FE$^2$ solve",
        r"(first row, FOM-FE$^2$) nested the same way at every Gauss point. Tip",
        r"reaction-force $F_x$ is the total force at the pulled tip (this problem is",
        r"displacement-controlled, so this is the physically meaningful output",
        r"scalar, unlike Cook's own tip-displacement range); $S$ err.\ is the",
        r"relative $L^2$ error of the full $600$-Gauss-point final-state stress",
        r"field, the energy-conjugate (reaction-force) convention throughout.}",
        r"\label{tab:cruciform-fe2}",
        r"\footnotesize",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{lcccrrr}",
        r"\toprule",
        r"Model & Newton behavior & Tip $F_x$ [MN] & Wall time & Speedup & Tip $F_x$ err.\ & $S$ err.\ \\",
        r"\midrule",
    ]
    for r in rows:
        wall_str = f"{r['wall']:.1f}~s" if r["wall"] is not None else "?"
        sp_str = f"{r['speedup']:.1f}" if r["speedup"] is not None else "--"
        re_str = f"{r['r_err']:.2%}".replace("%", r"\%") if r["r_err"] is not None else "--"
        se_str = f"{r['s_err']:.2%}".replace("%", r"\%") if r["s_err"] is not None else "--"
        lines.append(f"{r['label']} & {r['behavior']} & {r['reaction_px'] / 1e6:.1f} & {wall_str} & "
                     f"{sp_str} & {re_str} & {se_str} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}"]

    out_path = PAPER_DIR / "cruciform_fe2_table_claude.tex"
    out_path.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
