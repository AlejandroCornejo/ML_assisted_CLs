"""One-time independent evaluation of every constrained MC-RVE m=6 fit.

The feature count was frozen from the full validation-only sensitivity audit in
``m06_reporting_decision_v1``. This evaluator first reproduces all twelve saved
validation scores without accessing reserved labels. Only then does it open the
independent test states and the ten prescribed loading paths. It does not fit,
select, or overwrite a model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from protocol import train_material_b as base
from protocol.evaluate_final_models import metrics, predict, read_reserved
from protocol.prepare_design import digest
from protocol.select_features import BASE
from protocol.train_feature_count_amendment_v2 import RULE, load_rule
from protocol.training_setup import FlexibleEnergy


FEATURES = BASE / "results/feature_count_amendment_v2/feature_tables/m06"
TRAINING = BASE / "results/feature_count_amendment_v2/training"
DECISION = BASE / "results/feature_count_analysis_v1/m06_reporting_decision_v1/decision.json"
OUTPUT = BASE / "results/feature_count_analysis_v1/m06_independent_evaluation_v1"
LABELS = BASE / "results/data_labels_v1.npz"
ASSEMBLY = BASE / "results/data_labels_v1.json"
MODELS = ("ICNN-fixed", "ICNN-learned", "ICKAN-fixed", "ICKAN-learned")
SEEDS = (16, 29, 47)


def _sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _atomic_json(path: Path, value: dict) -> None:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False,
                                     dir=path.parent, prefix=f".{path.name}.") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def _slug(model: str, seed: int) -> str:
    return f"m06_{model.lower().replace('-', '_')}_seed{seed}"


def _load_model(row: dict):
    checkpoint = torch.load(TRAINING / row["slug"] / "model.pt",
                            map_location="cpu", weights_only=False)
    if (checkpoint.get("name") != row["model"] or checkpoint.get("seed") != row["seed"]
            or checkpoint.get("best_validation_score") != row["best_validation_score"]):
        raise ValueError(f"Checkpoint metadata differs from report: {row['slug']}")
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        model = FlexibleEnergy(**checkpoint["configuration"]).double()
    finally:
        torch.set_default_dtype(previous)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    base._check_model(model)
    return model


def _expected_rows(recipe: dict, manifest: dict) -> list[dict]:
    decision = json.loads(DECISION.read_text(encoding="utf-8"))
    if (decision.get("status") != "frozen_before_independent_evaluation"
            or decision.get("selected_feature_count") != 6
            or decision.get("test_or_path_labels_loaded") is not False
            or decision.get("test_or_path_labels_used_for_selection") is not False):
        raise ValueError("The validation-only m=6 reporting decision is invalid")
    campaign = json.loads((TRAINING / "campaign_status.json").read_text(encoding="utf-8"))
    if campaign.get("status") != "complete" or campaign.get("declared_jobs") != 38:
        raise ValueError("The low-count amendment campaign is not terminal")
    if _sha(LABELS) != manifest["labels_sha256"] or _sha(ASSEMBLY) == "":
        raise ValueError("The approved labels do not match the m=6 feature table")
    rule_sha = _sha(RULE)
    rows = []
    for model in MODELS:
        for seed in SEEDS:
            slug = _slug(model, seed)
            folder = TRAINING / slug
            report_path, model_path = folder / "run_report.json", folder / "model.pt"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            identity = report.get("identity", {})
            if (report.get("status") != "complete" or report.get("model") != model
                    or report.get("seed") != seed or report.get("feature_count") != 6
                    or report.get("amendment_rule_sha256") != rule_sha
                    or report.get("model_sha256") != _sha(model_path)
                    or report.get("test_labels_loaded") is not False
                    or report.get("path_labels_loaded") is not False
                    or identity.get("labels_sha256") != _sha(LABELS)
                    or identity.get("feature_table_sha256") != manifest["feature_table_sha256"]
                    or identity.get("feature_count") != 6):
                raise ValueError(f"Invalid terminal m=6 report: {slug}")
            for relative, expected in identity.get("sources_sha256", {}).items():
                if digest(BASE.parents[1] / relative) != expected:
                    raise ValueError(f"Training source changed since {slug}: {relative}")
            rows.append(dict(model=model, seed=seed, slug=slug,
                             model_sha256=report["model_sha256"],
                             report_sha256=_sha(report_path),
                             best_validation_score=report["best_validation_score"],
                             best_origin=report["best_origin"]))
    if len(rows) != 12:
        raise ValueError("Expected exactly twelve m=6 fits")
    return rows


def preflight() -> tuple[list[dict], dict, list[dict]]:
    recipe, _, manifest, _, _, count = load_rule(FEATURES, RULE)
    if count != 6:
        raise ValueError("m=6 preparation did not load")
    rows = _expected_rows(recipe, manifest)
    arrays = base.load_training_arrays(LABELS, recipe)
    scales = manifest["scales"]
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    checks = []
    for row in rows:
        model = _load_model(row)
        score = base.validation_score(model, arrays["E_validation"], arrays["S_validation"], scales)
        if not np.isclose(score, row["best_validation_score"], rtol=1e-8, atol=1e-14):
            raise ValueError(f"Validation score is not reproducible for {row['slug']}")
        tangent = predict(model, arrays["E_validation"][:1], scales)[2]
        asymmetry = float(np.max(np.abs(tangent-tangent.transpose(0, 2, 1))))
        if asymmetry > 1e-5 * max(1., float(np.max(np.abs(tangent)))):
            raise ValueError(f"Nonsymmetric tangent in preflight: {row['slug']}")
        checks.append(dict(slug=row["slug"], reproduced_validation_score=float(score),
                           tangent_max_asymmetry_Pa=asymmetry))
    return rows, scales, checks


def _summary(rows: list[dict]) -> dict:
    answer = {}
    for model in MODELS:
        group = [row for row in rows if row["model"] == model]
        if len(group) != 3:
            raise ValueError(f"Missing seed for {model}")
        def statistics(selector):
            result = {}
            for quantity in ("energy", "stress", "tangent"):
                values = np.asarray([selector(row)[quantity] for row in group], dtype=float)
                result[quantity] = dict(median=float(np.median(values)), minimum=float(values.min()),
                                        maximum=float(values.max()),
                                        by_seed={str(row["seed"]): float(value)
                                                 for row, value in zip(group, values)})
            return result
        answer[model] = dict(
            test_aggregate_percent=statistics(lambda row: row["test"]["aggregate_percent"]),
            paths_aggregate_percent=statistics(lambda row: row["paths_aggregate"]["aggregate_percent"]),
            individual_paths_percent={name: statistics(lambda row, name=name: row["paths"][name]["aggregate_percent"])
                                      for name in group[0]["paths"]},
        )
    return answer


def run_final(output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"Independent evaluation already exists: {output}")
    rows, scales, checks = preflight()
    output.mkdir(parents=True)
    gate = dict(status="opened_once", opened_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                reporting_decision_sha256=_sha(DECISION), evaluator_sha256=_sha(Path(__file__)),
                labels_sha256=_sha(LABELS), labels_assembly_sha256=_sha(ASSEMBLY),
                feature_manifest_sha256=_sha(FEATURES / "manifest.json"),
                feature_table_sha256=_sha(FEATURES / "feature_table.npz"),
                validation_preflight=checks,
                scope="All twelve constrained m=6 fits; no checkpoint or seed selection during evaluation.")
    _atomic_json(output / "gate_decision.json", gate)
    try:
        arrays = read_reserved()
        test_target = tuple(arrays[name] for name in ("W_test", "S_test", "D_test"))
        path_target = tuple(arrays[name] for name in ("W_paths", "S_paths", "D_paths"))
        all_test, all_paths, evaluated = [], [], []
        for row in rows:
            started = time.perf_counter()
            model = _load_model(row)
            test = predict(model, arrays["E_test"], scales)
            paths = predict(model, arrays["E_paths"], scales)
            per_path = {}
            for index, raw_name in enumerate(arrays["path_names"]):
                name = str(raw_name)
                section = slice(40*index, 40*(index+1))
                per_path[name] = metrics(tuple(value[section] for value in paths),
                                         tuple(value[section] for value in path_target), scales)
            tangent = test[2]
            evaluated.append(dict(**row, test=metrics(test, test_target, scales),
                                  paths_aggregate=metrics(paths, path_target, scales), paths=per_path,
                                  test_tangent_max_asymmetry_Pa=float(np.max(np.abs(
                                      tangent-tangent.transpose(0, 2, 1)))),
                                  inference_seconds=time.perf_counter()-started))
            all_test.append(test); all_paths.append(paths)
            print(json.dumps(dict(slug=row["slug"], stress_error_percent=
                                  evaluated[-1]["test"]["aggregate_percent"]["stress"])), flush=True)
        predictions = dict(model_names=np.asarray([row["model"] for row in evaluated]),
                           seeds=np.asarray([row["seed"] for row in evaluated]),
                           path_names=arrays["path_names"], path_parameter=arrays["path_parameter"])
        for prefix, values in (("test", all_test), ("paths", all_paths)):
            for index, name in enumerate(("W", "S", "D")):
                predictions[f"{name}_{prefix}_pred"] = np.stack([value[index] for value in values])
        np.savez_compressed(output / "predictions.npz", **predictions)
        _atomic_json(output / "per_run.json", dict(status="complete", rows=evaluated,
                    units=dict(energy="Pa", stress="Pa", tangent="Pa", percentages="percent"),
                    test_states=512, path_nonreference_states=400,
                    prediction_sha256=_sha(output / "predictions.npz")))
        result = dict(status="complete", by_model=_summary(evaluated),
                      per_run_sha256=_sha(output / "per_run.json"),
                      predictions_sha256=_sha(output / "predictions.npz"),
                      gate_decision_sha256=_sha(output / "gate_decision.json"),
                      test_labels_loaded=True, path_labels_loaded=True,
                      test_or_path_labels_used_for_selection=False,
                      scope="Independent evaluation of every constrained m=6 fit after validation-only reporting decision.")
        _atomic_json(output / "summary.json", result)
        return result
    except Exception as error:
        _atomic_json(output / "failure.json", dict(status="failed", error=repr(error),
                    traceback=traceback.format_exc(), gate_decision_sha256=_sha(output / "gate_decision.json")))
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--run-final", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.preflight == args.run_final:
        parser.error("Choose exactly one of --preflight or --run-final")
    rows, _, checks = preflight()
    if args.preflight:
        print(json.dumps(dict(status="ready", fits=len(rows), checks=len(checks),
                              test_labels_loaded=False, path_labels_loaded=False)))
        return 0
    run_final(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
