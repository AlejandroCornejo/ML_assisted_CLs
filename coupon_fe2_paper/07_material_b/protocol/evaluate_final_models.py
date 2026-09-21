"""One-time, hash-gated Material-B neural test and held-out-path evaluation.

Use --preflight first: it reads fit/reference/validation labels, never test or
path targets. --run-final verifies the same gate and then opens reserved labels.
No model fitting, checkpoint choice, or test-driven model selection occurs.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from protocol import train_material_b as base
from protocol import train_material_b_official as official
from protocol.close_free_lbfgs import AMENDMENT as FREE_AMENDMENT
from protocol.prepare_design import digest
from protocol.run_training_campaign import NAMES, OUTPUT as TRAINING, SEEDS, _slug
from protocol.select_features import BASE
from protocol.training_setup import (AnisotropicFreeEnergy, FlexibleEnergy,
                                     normalized_response)

HERE = Path(__file__).resolve().parent
INTERNAL_RULE = HERE / "FINAL_EVALUATION_INTERNAL.md"
DATA_RULE = HERE / "data_protocol_v1.json"
LOCK = TRAINING / "final_checkpoint_manifest.json"
LABELS = BASE / "results/data_labels_v1.npz"
ASSEMBLY = BASE / "results/data_labels_v1.json"
FEATURE_MANIFEST = BASE / "results/feature_selection_v1/manifest.json"
FEATURE_TABLE = BASE / "results/feature_selection_v1/feature_table.npz"
RESULTS = BASE / "results/neural_evaluation"
LOCK_SHA256 = "d42387b9c6b84cf810b0fc0c5872380da90ffa2af3cca1bb4d891ecd58ec2282"
BATCH = 64


def verify_gate():
    """Hash-check the exact mixed-status lock before accessing reserved arrays."""
    if digest(LOCK) != LOCK_SHA256:
        raise ValueError("Final checkpoint manifest changed after the internal gate amendment")
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    campaign_path = TRAINING / "campaign_status.json"
    campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
    data_rule = json.loads(DATA_RULE.read_text(encoding="utf-8"))
    assembly = json.loads(ASSEMBLY.read_text(encoding="utf-8"))
    features = json.loads(FEATURE_MANIFEST.read_text(encoding="utf-8"))
    expected = {(name, seed, _slug(name, seed)) for seed in SEEDS for name in NAMES}
    actual = {(row["model"], row["seed"], row["slug"]) for row in lock["entries"]}
    if (lock["status"] != "locked_mixed_plateau_and_adam_budget"
            or lock["checkpoint_count"] != 15 or actual != expected
            or campaign["status"] != "complete_with_budget_limited_adam"
            or lock["campaign_status_sha256"] != digest(campaign_path)
            or lock["official_rule_sha256"] != digest(official.RULE)
            or lock["free_amendment_sha256"] != digest(FREE_AMENDMENT)
            or lock["free_closeout_audit_sha256"] != digest(
                TRAINING / campaign["free_closeout_audit"])
            or data_rule["models"]["primary"] != list(NAMES)
            or data_rule["models"]["initialization_seeds"] != list(SEEDS)
            or assembly["status"] != "complete" or assembly["passed"] is not True
            or assembly["protocol_sha256"] != digest(DATA_RULE)
            or assembly["assembled_data_sha256"] != digest(LABELS)
            or features["status"] != "complete"
            or features["labels_sha256"] != digest(LABELS)
            or features["feature_table_sha256"] != digest(FEATURE_TABLE)):
        raise ValueError("Frozen models, feature preparation or approved labels failed the gate")
    for row in lock["entries"]:
        folder = TRAINING / row["slug"]
        checkpoint_path = folder / "model.pt"
        report_path = folder / "run_report.json"
        state_path = folder / official.STATE_NAME
        report = json.loads(report_path.read_text(encoding="utf-8"))
        expected_status = ("complete_budget_limited_adam" if row["model"] == "Free"
                           else "complete")
        if (row["status"] != expected_status or report["status"] != expected_status
                or row["model_sha256"] != digest(checkpoint_path)
                or row["report_sha256"] != digest(report_path)
                or row["run_state_sha256"] != digest(state_path)
                or report["model_sha256"] != row["model_sha256"]
                or report["best_validation_score"] != row["best_validation_score"]
                or report["best_origin"] != row["best_origin"]
                or report["test_labels_loaded"] is not False
                or report["path_labels_loaded"] is not False):
            raise ValueError(f"Locked model/report failed integrity check: {row['slug']}")
        identity = report["identity"]
        if (identity["model"] != row["model"] or identity["seed"] != row["seed"]
                or identity["labels_sha256"] != digest(LABELS)
                or identity["recipe_sha256"] != digest(official.RULE)
                or identity["feature_manifest_sha256"] != digest(FEATURE_MANIFEST)
                or identity["feature_table_sha256"] != digest(FEATURE_TABLE)):
            raise ValueError(f"Saved model identity changed: {row['slug']}")
        for relative_path, source_hash in identity["sources_sha256"].items():
            if digest(BASE.parents[1] / relative_path) != source_hash:
                raise ValueError(f"Model source changed: {relative_path}")
    return lock, features["scales"]


def load_model(row):
    checkpoint = torch.load(TRAINING / row["slug"] / "model.pt",
                            map_location="cpu", weights_only=False)
    if (checkpoint["name"] != row["model"] or checkpoint["seed"] != row["seed"]
            or checkpoint["best_validation_score"] != row["best_validation_score"]):
        raise ValueError(f"Checkpoint metadata differs from lock: {row['slug']}")
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        constructor = AnisotropicFreeEnergy if row["model"] == "Free" else FlexibleEnergy
        model = constructor(**checkpoint["configuration"]).double()
    finally:
        torch.set_default_dtype(previous)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    base._check_model(model)
    return model


def predict(model, strain, scales, batch=BATCH):
    """Return physical W, S, D; D differentiates S w.r.t. engineering strain."""
    e = np.asarray(strain, dtype=np.float64)
    if e.ndim != 2 or e.shape[1] != 3 or not np.isfinite(e).all():
        raise ValueError("Expected finite engineering strains with shape (n,3)")
    ss, es = scales["strain_scale"], scales["energy_scale"]
    energy, stress, tangent = [], [], []
    for first in range(0, len(e), batch):
        x = torch.as_tensor(e[first:first+batch] / ss,
                            dtype=torch.float64).clone().detach().requires_grad_(True)
        w, s = model.energy_and_stress(x, create_graph=True)
        d = torch.stack([
            torch.autograd.grad(s[:, component].sum(), x,
                                retain_graph=component < 2)[0]
            for component in range(3)], dim=1)
        energy.append((w[:, 0] * es).detach().cpu().numpy())
        stress.append((s * (es/ss)).detach().cpu().numpy())
        tangent.append((d * (es/ss**2)).detach().cpu().numpy())
    result = (np.concatenate(energy), np.concatenate(stress),
              np.concatenate(tangent))
    if not all(np.isfinite(values).all() for values in result):
        raise FloatingPointError("Nonfinite neural prediction")
    return result


def aggregate_percent(pred, target):
    numerator = np.linalg.norm(np.asarray(pred)-np.asarray(target))
    denominator = np.linalg.norm(target)
    if not np.isfinite(denominator) or denominator <= 0:
        raise ValueError("Zero or nonfinite aggregate target norm")
    return float(100 * numerator / denominator)


def distribution_percent(pred, target, floor):
    p, t = np.asarray(pred), np.asarray(target)
    if p.ndim == 1:
        difference, magnitude = np.abs(p-t), np.abs(t)
    else:
        difference = np.linalg.norm((p-t).reshape(len(p), -1), axis=1)
        magnitude = np.linalg.norm(t.reshape(len(t), -1), axis=1)
    if floor <= 0 or not np.isfinite(floor):
        raise ValueError("Invalid frozen fit-derived floor")
    values = 100 * difference / np.maximum(magnitude, floor)
    return dict(median=float(np.percentile(values, 50)),
                p95=float(np.percentile(values, 95)),
                maximum=float(np.max(values)))


def metrics(prediction, target, scales):
    w, s, d = prediction
    wr, sr, dr = target
    if any(p.shape != t.shape for p, t in zip(prediction, target)):
        raise ValueError("Prediction and FOM shapes disagree")
    result = dict(
        aggregate_percent=dict(energy=aggregate_percent(w, wr),
                               stress=aggregate_percent(s, sr),
                               tangent=aggregate_percent(d, dr)),
        per_state_percent=dict(
            energy=distribution_percent(w, wr, scales["energy_metric_floor"]),
            stress=distribution_percent(s, sr, scales["stress_metric_floor"]),
            tangent=distribution_percent(d, dr, scales["tangent_metric_floor"])),
        stress_component_rmse_percent=(100*np.sqrt(np.mean((s-sr)**2, axis=0)) /
            np.asarray(scales["stress_component_metric_scale"])).tolist(),
        tangent_component_rmse_Pa=np.sqrt(np.mean((d-dr)**2, axis=0)).tolist())
    return result


def preflight(lock, scales):
    """Run every saved model on validation only, checking its recorded score."""
    recipe, _, _, _ = official.load_rule(base.FEATURES)
    arrays = base.load_training_arrays(LABELS, recipe)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    checks = []
    for row in lock["entries"]:
        model = load_model(row)
        score = base.validation_score(model, arrays["E_validation"],
                                      arrays["S_validation"], scales)
        saved = row["best_validation_score"]
        if not np.isclose(score, saved, rtol=1e-8, atol=1e-14):
            raise ValueError(f"Saved validation score not reproducible: {row['slug']}: {score} != {saved}")
        probe = predict(model, arrays["E_validation"][:1], scales)
        asymmetry = np.max(np.abs(probe[2]-probe[2].transpose(0, 2, 1)))
        if asymmetry > 1e-5 * max(1., np.max(np.abs(probe[2]))):
            raise ValueError(f"Nonsymmetric tangent in preflight: {row['slug']}")
        checks.append(dict(slug=row["slug"], saved_validation_score=saved,
                           reproduced_validation_score=score,
                           probe_tangent_max_asymmetry_Pa=float(asymmetry)))
    return checks


def read_reserved():
    """This is the only routine that accesses reserved test/path target arrays."""
    with np.load(LABELS, allow_pickle=False) as data:
        arrays = {name: np.asarray(data[name]) for name in (
            "E_test", "W_test", "S_test", "D_test",
            "E_paths", "W_paths", "S_paths", "D_paths",
            "E_reference", "W_reference", "S_reference", "D_reference",
            "path_names", "path_parameter")}
    expected_shapes = dict(E_test=(512, 3), W_test=(512,), S_test=(512, 3),
                           D_test=(512, 3, 3), E_paths=(400, 3), W_paths=(400,),
                           S_paths=(400, 3), D_paths=(400, 3, 3),
                           E_reference=(1, 3), W_reference=(1,),
                           S_reference=(1, 3), D_reference=(1, 3, 3))
    for name, shape in expected_shapes.items():
        if arrays[name].shape != shape or not np.isfinite(arrays[name]).all():
            raise ValueError(f"Invalid reserved array {name}: {arrays[name].shape}")
    if len(arrays["path_names"]) != 10 or len(arrays["path_parameter"]) != 41:
        raise ValueError("Unexpected held-out path design")
    return arrays


def summarize(rows, lock):
    def seed_statistics(group, select):
        answer = {}
        for category in ("energy", "stress", "tangent"):
            values = np.asarray([select(row)[category] for row in group],
                                dtype=np.float64)
            answer[category] = dict(mean=float(values.mean()),
                                    sample_sd=float(values.std(ddof=1)),
                                    by_seed={str(row["seed"]): float(value)
                                             for row, value in zip(group, values)})
        return answer

    result = {}
    for name in NAMES:
        group = [row for row in rows if row["model"] == name]
        if len(group) != 3:
            raise ValueError(f"Missing seed in {name}")
        scores = {row["seed"]: row["validation_score"] for row in group}
        median_seed = sorted(scores, key=lambda seed: (scores[seed], seed))[1]
        result[name] = dict(median_validation_seed=median_seed,
            test_aggregate_percent=seed_statistics(
                group, lambda row: row["test"]["aggregate_percent"]),
            paths_aggregate_percent=seed_statistics(
                group, lambda row: row["paths_aggregate"]["aggregate_percent"]),
            individual_paths_percent={path: seed_statistics(
                group, lambda row, path=path: row["paths"][path]["aggregate_percent"])
                for path in group[0]["paths"]})
    return result


def run_final(output, lock, scales, preflight_checks):
    if output.exists():
        raise FileExistsError(f"Final evaluation already exists; refusing overwrite: {output}")
    output.mkdir(parents=True, exist_ok=False)
    gate = dict(status="opened_once", opened_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        final_checkpoint_manifest_sha256=digest(LOCK),
        internal_gate_amendment_sha256=digest(INTERNAL_RULE),
        evaluator_sha256=digest(Path(__file__)),
        approved_labels_sha256=digest(LABELS),
        assembly_report_sha256=digest(ASSEMBLY),
        validation_preflight=preflight_checks)
    base._atomic_json(output / "gate_decision.json", gate)
    try:
        arrays = read_reserved()
        t_target = tuple(arrays[name] for name in ("W_test", "S_test", "D_test"))
        p_target = tuple(arrays[name] for name in ("W_paths", "S_paths", "D_paths"))
        r_target = tuple(arrays[name] for name in ("W_reference", "S_reference", "D_reference"))
        all_test, all_paths, all_reference = [], [], []
        rows = []
        for row in lock["entries"]:
            started = time.perf_counter()
            model = load_model(row)
            test_pred = predict(model, arrays["E_test"], scales)
            path_pred = predict(model, arrays["E_paths"], scales)
            ref_pred = predict(model, arrays["E_reference"], scales)
            all_test.append(test_pred)
            all_paths.append(path_pred)
            all_reference.append(ref_pred)
            path_rows = {}
            for i, raw_name in enumerate(arrays["path_names"]):
                path_name = str(raw_name)
                section = slice(40*i, 40*(i+1))
                path_rows[path_name] = metrics(
                    tuple(value[section] for value in path_pred),
                    tuple(value[section] for value in p_target), scales)
            tangent = test_pred[2]
            rows.append(dict(model=row["model"], seed=row["seed"], slug=row["slug"],
                status=row["status"], model_sha256=row["model_sha256"],
                validation_score=row["best_validation_score"],
                best_origin=row["best_origin"],
                test=metrics(test_pred, t_target, scales),
                paths_aggregate=metrics(path_pred, p_target, scales),
                paths=path_rows,
                reference=dict(energy_error_Pa=float(ref_pred[0][0]-r_target[0][0]),
                    stress_error_Pa=(ref_pred[1][0]-r_target[1][0]).tolist(),
                    tangent_error_Pa=(ref_pred[2][0]-r_target[2][0]).tolist()),
                test_tangent_max_asymmetry_Pa=float(np.max(np.abs(
                    tangent-tangent.transpose(0, 2, 1)))),
                inference_seconds=time.perf_counter()-started))
            print(json.dumps(dict(slug=row["slug"],
                test_aggregate_percent=rows[-1]["test"]["aggregate_percent"])), flush=True)
        predictions = dict(model_names=np.asarray([r["model"] for r in rows]),
            seeds=np.asarray([r["seed"] for r in rows]),
            path_names=arrays["path_names"], path_parameter=arrays["path_parameter"])
        for prefix, values in (("test", all_test), ("paths", all_paths),
                               ("reference", all_reference)):
            for i, kind in enumerate(("W", "S", "D")):
                predictions[f"{kind}_{prefix}_pred"] = np.stack([row[i] for row in values])
        np.savez_compressed(output / "predictions.npz", **predictions)
        base._atomic_json(output / "per_run.json", dict(
            status="complete", rows=rows, units=dict(energy="Pa", stress="Pa",
            tangent="Pa", percentages="percent"),
            test_states=512, path_nonreference_states=400,
            prediction_sha256=digest(output / "predictions.npz")))
        summary = dict(status="complete", by_model=summarize(rows, lock),
            per_run_sha256=digest(output / "per_run.json"),
            predictions_sha256=digest(output / "predictions.npz"),
            gate_decision_sha256=digest(output / "gate_decision.json"),
            test_labels_loaded=True, path_labels_loaded=True,
            test_or_paths_used_for_selection=False,
            limitation="Three Free runs stopped Adam at a 200000-step budget while validation still improved; all three subsequently met the original LBFGS plateau rule, but their selected checkpoints remained Adam snapshots.")
        base._atomic_json(output / "summary.json", summary)
        return summary
    except Exception as error:
        base._atomic_json(output / "failure.json", dict(status="failed",
            error=repr(error), traceback=traceback.format_exc(),
            original_gate_decision_sha256=digest(output / "gate_decision.json")))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--run-final", action="store_true")
    parser.add_argument("--output", type=Path, default=RESULTS)
    args = parser.parse_args()
    if args.preflight == args.run_final:
        parser.error("Choose exactly one of --preflight or --run-final")
    if not INTERNAL_RULE.is_file():
        raise FileNotFoundError("Internal evaluation-gate amendment is missing")
    lock, scales = verify_gate()
    checks = preflight(lock, scales)
    if args.preflight:
        print(json.dumps(dict(status="ready", checks=len(checks),
            locked_models=len(lock["entries"]),
            reserved_test_labels_loaded=False,
            heldout_path_labels_loaded=False)), flush=True)
        return
    run_final(args.output.resolve(), lock, scales, checks)


if __name__ == "__main__":
    main()
