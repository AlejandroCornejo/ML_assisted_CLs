"""Freeze a validation-only audit of the 84-fit multicavity count sweep.

This utility reads only final reports, rule/feature receipts, and checkpoint
bytes.  It deliberately does not load model weights, constitutive labels, test
labels, or path labels.  It therefore creates the evidence needed to discuss
the feature-count sensitivity without using held-out predictions to choose a
count or a representative comparison point.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import tempfile
from collections import Counter
from pathlib import Path

from protocol.select_features import BASE
from protocol.train_feature_count_amendment_v2 import RULE as AMENDMENT_RULE
from protocol.train_feature_count_sweep import RULE as LEGACY_RULE

LEGACY_ROOT = BASE / "results/feature_count_sweep_v1/training"
AMENDMENT_ROOT = BASE / "results/feature_count_amendment_v2/training"
DEFAULT_OUTPUT = BASE / "results/feature_count_analysis_v1/validation_audit_v1"
COUNTS = (2, 4, 6, 8, 16, 24, 32)
LOW_COUNTS = frozenset((2, 4, 6))
SEEDS = (16, 29, 47)
MODELS = ("ICNN-fixed", "ICNN-learned", "ICKAN-fixed", "ICKAN-learned")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def slug(count: int, model: str, seed: int) -> str:
    return f"m{count:02d}_{model.lower().replace('-', '_')}_seed{seed}"


def atomic_text(path: Path, text: str) -> None:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False,
                                     dir=path.parent, prefix=f".{path.name}.") as stream:
        stream.write(text)
        temporary = Path(stream.name)
    os.replace(temporary, path)


def median(values: list[float]) -> float:
    return float(statistics.median(values))


def _validate_campaign_receipts() -> dict:
    legacy = json.loads((LEGACY_ROOT / "campaign_status.json").read_text(encoding="utf-8"))
    amendment = json.loads((AMENDMENT_ROOT / "campaign_status.json").read_text(encoding="utf-8"))
    m40 = [job for job in legacy.get("jobs", ()) if job.get("feature_count") == 40]
    if len(m40) != 12 or any(job.get("status") != "pending" for job in m40):
        raise ValueError("The cancellation receipt does not show all m=40 fits unstarted")
    if amendment.get("status") != "complete" or amendment.get("declared_jobs") != 38:
        raise ValueError("The amendment campaign is not terminal")
    if any(job.get("status") != "complete" for job in amendment.get("jobs", ())):
        raise ValueError("The amendment receipt contains non-complete jobs")
    notice = json.loads((LEGACY_ROOT / "amendment_notice_v2.json").read_text(encoding="utf-8"))
    if notice.get("cancelled_unstarted_jobs") != [job["slug"] for job in m40]:
        raise ValueError("The m=40 cancellation receipt disagrees with the legacy campaign")
    return dict(
        legacy_campaign_status_sha256=digest(LEGACY_ROOT / "campaign_status.json"),
        amendment_campaign_status_sha256=digest(AMENDMENT_ROOT / "campaign_status.json"),
        amendment_notice_sha256=digest(LEGACY_ROOT / "amendment_notice_v2.json"),
        unstarted_m40_jobs=[job["slug"] for job in m40],
    )


def _read_row(count: int, model: str, seed: int, *, legacy_sha: str, amendment_sha: str) -> dict:
    root = AMENDMENT_ROOT if count in LOW_COUNTS else LEGACY_ROOT
    folder = root / slug(count, model, seed)
    report_path, model_path = folder / "run_report.json", folder / "model.pt"
    if not report_path.is_file() or not model_path.is_file():
        raise FileNotFoundError(f"Missing terminal artifacts for {folder.name}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    expected_rule_field = "amendment_rule_sha256" if count in LOW_COUNTS else "sweep_rule_sha256"
    expected_rule_sha = amendment_sha if count in LOW_COUNTS else legacy_sha
    if (report.get("status") != "complete" or report.get("model") != model
            or report.get("seed") != seed or report.get("feature_count") != count
            or report.get(expected_rule_field) != expected_rule_sha
            or report.get("model_sha256") != digest(model_path)
            or report.get("test_labels_loaded") is not False
            or report.get("path_labels_loaded") is not False
            or report.get("adam_stop_reason") not in ("validation_plateau", "safety_cap")
            or report.get("lbfgs_stop_reason") not in ("validation_plateau", "safety_cap")):
        raise ValueError(f"Invalid final report for {folder.name}")
    history = report.get("validation_history", ())
    if not history or any(event.get("phase") not in ("initialization", "adam", "lbfgs")
                          for event in history):
        raise ValueError(f"Invalid validation history for {folder.name}")
    history_minimum = min(float(event["score"]) for event in history)
    score = float(report["best_validation_score"])
    if not math.isclose(score, history_minimum, rel_tol=0.0, abs_tol=1e-18):
        raise ValueError(f"Checkpoint score is not the validation-history minimum for {folder.name}")
    return dict(
        feature_count=count,
        model=model,
        core=model.split("-", maxsplit=1)[0],
        feature_type=model.split("-", maxsplit=1)[1],
        seed=seed,
        slug=folder.name,
        campaign="amendment_v2" if count in LOW_COUNTS else "legacy_v1",
        rule_sha256=expected_rule_sha,
        report_sha256=digest(report_path),
        model_sha256=report["model_sha256"],
        best_validation_score=score,
        best_origin_phase=report["best_origin"]["phase"],
        best_origin_index=int(report["best_origin"]["index"]),
        adam_steps=int(report["adam_steps"]),
        adam_stop_reason=report["adam_stop_reason"],
        lbfgs_outer_calls=int(report["lbfgs_outer_calls"]),
        lbfgs_stop_reason=report["lbfgs_stop_reason"],
        elapsed_seconds=float(report["elapsed_seconds"]),
        validation_history_events=len(history),
    )


def _aggregate(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    by_group: dict[tuple[str, int], list[dict]] = {}
    by_pair: dict[tuple[str, int, int], dict[str, dict]] = {}
    for row in rows:
        by_group.setdefault((row["model"], row["feature_count"]), []).append(row)
        by_pair.setdefault((row["core"], row["feature_count"], row["seed"]), {})[row["feature_type"]] = row
    groups = []
    for model in MODELS:
        for count in COUNTS:
            group = by_group.get((model, count), [])
            if len(group) != len(SEEDS):
                raise ValueError(f"Incomplete group: {model}, m={count}")
            scores = [row["best_validation_score"] for row in group]
            groups.append(dict(
                model=model, core=group[0]["core"], feature_type=group[0]["feature_type"],
                feature_count=count, seeds=list(SEEDS), runs=len(group),
                median_validation_score=median(scores), minimum_validation_score=min(scores),
                maximum_validation_score=max(scores),
                median_adam_steps=median([row["adam_steps"] for row in group]),
                median_elapsed_seconds=median([row["elapsed_seconds"] for row in group]),
                adam_stop_reasons=dict(sorted(Counter(row["adam_stop_reason"] for row in group).items())),
                lbfgs_stop_reasons=dict(sorted(Counter(row["lbfgs_stop_reason"] for row in group).items())),
            ))
    ratios = []
    for core in ("ICNN", "ICKAN"):
        for count in COUNTS:
            values = []
            for seed in SEEDS:
                pair = by_pair[(core, count, seed)]
                if set(pair) != {"fixed", "learned"}:
                    raise ValueError(f"Unpaired fixed/learned run: {core}, m={count}, seed={seed}")
                values.append(pair["learned"]["best_validation_score"] / pair["fixed"]["best_validation_score"])
            ratios.append(dict(core=core, feature_count=count, seeds=list(SEEDS),
                               learned_over_fixed_ratios=values,
                               median_learned_over_fixed_ratio=median(values),
                               minimum_learned_over_fixed_ratio=min(values),
                               maximum_learned_over_fixed_ratio=max(values)))
    return groups, ratios


def _markdown(receipts: dict, rows: list[dict], groups: list[dict], ratios: list[dict]) -> str:
    lines = [
        "# Validation-only audit: multicavity feature-count sweep",
        "",
        "This frozen audit consolidates the 84 constrained fits at `m = 2, 4, 6, 8, 16, 24, 32`. It reads final training reports and checkpoint hashes only; it does not load test or reserved-path labels, model weights, or FOM predictions.",
        "",
        "The original unstarted `m=40` block was replaced prospectively by the low-count amendment (`m=2,4,6`) using validation-only observations. The retained `m=8,16,24,32` fits use the legacy frozen rule; all low-count fits use the amendment rule. This distinction is retained below.",
        "",
        f"- Validated terminal fits: **{len(rows)} / 84**",
        f"- Unstarted/cancelled `m=40` fits: **{len(receipts['unstarted_m40_jobs'])}**",
        "- Test labels loaded by any fit: **no**",
        "- Reserved path labels loaded by any fit: **no**",
        "",
        "## Median validation stress error by model and feature count",
        "",
        "Lower is better. Ranges span the three declared seeds.",
        "",
        "| Model | `m` | Median | Min--max | Median Adam steps | Median wall time (h) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for group in groups:
        lines.append("| {model} | {feature_count} | {median_validation_score:.3e} | {minimum_validation_score:.3e}--{maximum_validation_score:.3e} | {median_adam_steps:.0f} | {hours:.2f} |".format(
            **group, hours=group["median_elapsed_seconds"]/3600))
    lines += [
        "",
        "## Paired learned/fixed validation ratio",
        "",
        "Each ratio pairs the same core and seed. A value below one favors learned directional features.",
        "",
        "| Core | `m` | Median learned/fixed | Min--max |",
        "|---|---:|---:|---:|",
    ]
    for ratio in ratios:
        lines.append("| {core} | {feature_count} | {median_learned_over_fixed_ratio:.3g} | {minimum_learned_over_fixed_ratio:.3g}--{maximum_learned_over_fixed_ratio:.3g} |".format(**ratio))
    lines += [
        "",
        "## Scope",
        "",
        "This document is a training/validation audit, not the independent-test or loading-path analysis. It must not be used to claim generalization or to select a final reported model from test outcomes.",
        "",
    ]
    return "\n".join(lines)


def audit(output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite frozen audit output: {output}")
    receipts = _validate_campaign_receipts()
    legacy_sha, amendment_sha = digest(LEGACY_RULE), digest(AMENDMENT_RULE)
    rows = [_read_row(count, model, seed, legacy_sha=legacy_sha, amendment_sha=amendment_sha)
            for count in COUNTS for model in MODELS for seed in SEEDS]
    if len(rows) != 84 or len({(row["feature_count"], row["model"], row["seed"]) for row in rows}) != 84:
        raise ValueError("The completed sweep does not contain exactly 84 unique declared fits")
    groups, ratios = _aggregate(rows)
    summary = dict(
        audit_id="multicavity_feature_count_validation_audit_v1",
        scope="Validation-only consolidation of the 84 constrained feature-count fits; no test/path labels or predictions are loaded.",
        counts=list(COUNTS), seeds=list(SEEDS), models=list(MODELS),
        legacy_rule_sha256=legacy_sha, amendment_rule_sha256=amendment_sha,
        receipts=receipts, rows=rows, grouped_statistics=groups,
        paired_learned_over_fixed_ratios=ratios,
    )
    output.mkdir(parents=True)
    csv_columns = list(rows[0])
    with (output / "validation_rows.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=csv_columns)
        writer.writeheader(); writer.writerows(rows)
    atomic_text(output / "validation_summary.json", json.dumps(summary, indent=2, sort_keys=True) + "\n")
    atomic_text(output / "VALIDATION_AUDIT.md", _markdown(receipts, rows, groups, ratios))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    summary = audit(args.output.resolve())
    print(json.dumps(dict(audit_id=summary["audit_id"], fits=len(summary["rows"]), output=str(args.output.resolve()))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
