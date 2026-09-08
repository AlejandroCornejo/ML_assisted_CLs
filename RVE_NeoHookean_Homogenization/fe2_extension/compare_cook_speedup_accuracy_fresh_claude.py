#!/usr/bin/env python3
"""Refreshed version of compare_true_fom_speedup_accuracy_claude.py: once
all 8 models' Cook nx=ny=8 results exist (this session's re-run, with
the optimized/16-worker-parallel infrastructure), compute against
FOM-FE2 as the true multiscale reference:
  - speedup = true_FOM_wall_time / row_wall_time
  - tip u_y relative error, using the midpoint of the (min, max) tip-edge
    range at the FINAL step (matching Table 7's own convention)
  - full 384-Gauss-point final-state stress relative L2 error

Unlike the original script (which hardcoded previously-verified wall
times/tip_uy values into a ROWS list), this one PARSES wall times
directly from this session's own run logs, and reads tip_uy/s_gp
directly from each row's own saved .npz -- everything here was computed
fresh in this session, so there is nothing to hardcode.

Prints a summary table AND writes ready-to-paste LaTeX table source to
pann/anisotropic/cook_fe2_table_fresh_claude.tex, matching Table 7's own
column structure and footnote style.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

FE2_DIR = Path("/home/kratos/ML_assisted_CLs_clean/RVE_NeoHookean_Homogenization/fe2_extension")
PAPER_DIR = Path("/home/kratos/ML_assisted_CLs_clean/RVE_NeoHookean_Homogenization/pann/anisotropic")
SCRATCH = Path("/tmp/claude-1000/-home-kratos-ML-assisted-CLs-clean/4a3da423-8dfc-40e5-a7fa-8fa40d3c8b2f/scratchpad")

LOG_FILES = (
    SCRATCH / "cook_nx8_full_suite.log",
    SCRATCH / "cook_pann_w5_refresh.log",
    SCRATCH / "cook_fom_nx8_full.log",
    # last, so its mean-of-10-reps wall times override the single-run ones
    # above for the 4 PANN keys specifically (matching Table 7's own
    # established convention for these fast rows).
    SCRATCH / "pann_10x_timing.log",
)

ROW_ORDER = ("fom_nested_consistent_parallel", "pann_regression", "pann_free", "pann_certified_w5",
             "pann_ickan_w5", "linear_hprom_parallel_continuation", "hprom_ann_parallel_continuation",
             "dhprom_ann_parallel")
LABELS = {
    "fom_nested_consistent_parallel": "FOM-FE$^2$",
    "pann_regression": "Regression (tier 1)",
    "pann_free": "Free hyperelastic (tier 2)",
    "pann_certified_w5": "Polyconvex ICNN (tier 3a)",
    "pann_ickan_w5": "Polyconvex ICKAN (tier 3b)",
    "linear_hprom_parallel_continuation": "Linear-HPROM-FE$^2$",
    "hprom_ann_parallel_continuation": "HPROM--ANN-FE$^2$",
    "dhprom_ann_parallel": "D-HPROM--ANN-FE$^2$",
}

WALL_RE = re.compile(r"^\[(?P<which>[\w.]+)\].*?wall=(?P<wall>[0-9.]+)s")


def parse_wall_times() -> dict:
    walls = {}
    for log_path in LOG_FILES:
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
    status = np.asarray(d["status_per_step"])
    iters = np.asarray(d["iters_per_step"])
    if bool(np.all(status == "converged")):
        lo, hi = int(iters.min()), int(iters.max())
        return f"converged, {lo} it./step" if lo == hi else f"converged, {lo}-{hi} it./step"
    if bool(np.all(status != "converged")):
        return "stalled, every step"
    n_bad = int(np.sum(status != "converged"))
    return f"{n_bad}/{len(status)} steps not converged"


def main() -> None:
    walls = parse_wall_times()
    missing = [w for w in ROW_ORDER if w not in walls]
    if missing:
        print(f"[warning] no parsed wall time yet for: {missing}")

    data = {}
    for which in ROW_ORDER:
        path = FE2_DIR / f"cook_results_{which}_claude.npz"
        if not path.exists():
            print(f"  [skip] {which}: {path.name} not yet available")
            continue
        d = np.load(path)
        assert len(d["load_per_step"]) == 20, f"{which}: {path.name} has {len(d['load_per_step'])} steps, not 20"
        data[which] = d

    if "fom_nested_consistent_parallel" not in data:
        raise SystemExit("FOM-FE2 result not available yet -- cannot compute errors/speedup without the reference.")

    true = data["fom_nested_consistent_parallel"]
    true_coords, true_tris = np.asarray(true["coords"]), np.asarray(true["tris"])
    true_s_gp = np.asarray(true["s_gp"], dtype=np.float64)
    true_tip = 0.5 * (float(true["tip_uy_min_per_step"][-1]) + float(true["tip_uy_max_per_step"][-1]))
    true_wall = walls.get("fom_nested_consistent_parallel")

    rows = []
    for which in ROW_ORDER:
        if which not in data:
            continue
        d = data[which]
        assert np.allclose(d["coords"], true_coords), f"{which}: mesh coords mismatch vs FOM"
        assert np.array_equal(d["tris"], true_tris), f"{which}: mesh connectivity mismatch vs FOM"
        wall = walls.get(which)
        tip = 0.5 * (float(d["tip_uy_min_per_step"][-1]) + float(d["tip_uy_max_per_step"][-1]))
        s_err = relative_l2(np.asarray(d["s_gp"], dtype=np.float64), true_s_gp)
        tip_err = abs(tip - true_tip) / max(abs(true_tip), 1e-30)
        is_fom = which == "fom_nested_consistent_parallel"
        rows.append({
            "which": which, "label": LABELS[which], "behavior": newton_behavior(d),
            "tip_lo": float(d["tip_uy_min_per_step"][-1]), "tip_hi": float(d["tip_uy_max_per_step"][-1]),
            "wall": wall,
            "speedup": None if is_fom or wall is None or true_wall is None else true_wall / wall,
            "tip_err": None if is_fom else tip_err, "s_err": None if is_fom else s_err,
        })

    print(f"\n{'Model':<26}{'Newton behavior':<24}{'Tip u_y range [m]':>20}{'Wall [s]':>12}"
          f"{'Speedup':>10}{'u_y err':>10}{'S err':>10}")
    for r in rows:
        wall_str = f"{r['wall']:.1f}" if r["wall"] is not None else "?"
        sp_str = f"{r['speedup']:.1f}x" if r["speedup"] is not None else "--"
        te_str = f"{r['tip_err']:.2%}" if r["tip_err"] is not None else "--"
        se_str = f"{r['s_err']:.2%}" if r["s_err"] is not None else "--"
        rng_str = f"{r['tip_lo']:.4f}-{r['tip_hi']:.4f}"
        print(f"{r['label']:<26}{r['behavior']:<24}{rng_str:>20}{wall_str:>12}"
              f"{sp_str:>10}{te_str:>10}{se_str:>10}")

    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Cook's membrane at $n_x=n_y=8$ ($128$ elements, $384$ Gauss",
        r"points), full $20$-step ramp, every model this paper has trained or built",
        r"as the macroscopic material law, plus the true, non-reduced FE$^2$ solve",
        r"(first row, FOM-FE$^2$) nested the same way at every Gauss point. Tip",
        r"$u_y$ err.\ is the relative error of the tip-edge range's midpoint; $S$",
        r"err.\ is the relative $L^2$ error of the full $384$-Gauss-point",
        r"final-state stress field, the energy-conjugate (reaction-force)",
        r"convention throughout. Redone with this session's analytic FOM tangent",
        r"and 16-worker parallel infrastructure (previously reported numbers used",
        r"a central-finite-difference FOM tangent and serial ROM/HPROM evaluation).}",
        r"\label{tab:cook-fe2}",
        r"\footnotesize",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{lcccrrr}",
        r"\toprule",
        r"Model & Newton behavior & Tip $u_y$ range [m] & Wall time & Speedup & Tip $u_y$ err.\ & $S$ err.\ \\",
        r"\midrule",
    ]
    PANN_KEYS = {"pann_regression", "pann_free", "pann_certified_w5", "pann_ickan_w5"}
    STALLED_KEYS = {"pann_regression", "pann_free"}
    for r in rows:
        wall_str = f"{r['wall']:.1f}~s" if r["wall"] is not None else "?"
        sp_str = f"{r['speedup']:.1f}" if r["speedup"] is not None else "--"
        te_str = f"{r['tip_err']:.2%}".replace("%", r"\%") if r["tip_err"] is not None else "--"
        se_str = f"{r['s_err']:.2%}".replace("%", r"\%") if r["s_err"] is not None else "--"
        rng_str = f"{r['tip_lo']:.4f}--{r['tip_hi']:.4f}"
        label = r["label"]
        behavior = r["behavior"]
        if r["which"] == "fom_nested_consistent_parallel":
            label += r"$^\ddagger$"
        if r["which"] in STALLED_KEYS:
            behavior += r"$^\dagger$"
            wall_str += r"$^{\dagger\S}$"
        elif r["which"] in PANN_KEYS:
            wall_str += r"$^\S$"
        lines.append(f"{label} & {behavior} & {rng_str} & {wall_str} & "
                     f"{sp_str} & {te_str} & {se_str} \\\\")
    lines += [
        r"\bottomrule", r"\end{tabular}%", r"}",
        r"", r"\vspace{4pt}",
        r"\footnotesize $^\dagger$Pushed through the full ramp regardless of",
        r"per-step convergence; the short wall time reflects failing fast, not",
        r"efficient convergence, and is not comparable to the converged rows.",
        r"$^\ddagger$The full, non-reduced $990$-element RVE, nested at every one of",
        r"the $384$ Gauss points with no surrogate or reduction of any kind, its",
        r"tangent obtained analytically via the implicit function theorem on its own",
        r"already-assembled tangent stiffness; wall time is the actual, measured",
        r"total for the complete $20$-step ramp, not an extrapolation; being the",
        r"ground truth itself, it has no speedup or error entries of its own.",
        r"$^\S$Mean of $10$ independent repetitions, not a single run, for a more",
        r"representative wall time.",
        r"\end{table}",
    ]

    out_path = PAPER_DIR / "cook_fe2_table_fresh_claude.tex"
    out_path.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
