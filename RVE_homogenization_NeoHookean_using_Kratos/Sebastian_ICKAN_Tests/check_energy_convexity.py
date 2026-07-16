import argparse
import json
import os

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplcfg")

import numpy as np
import torch

from ickan_workflow import create_model, ensure_dir, load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Check Jensen convexity for a trained ICKAN/ICNN energy model. "
            "Convexity requires E((u+v)/2) <= 0.5*(E(u)+E(v))."
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prediction-npz", required=True)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--n-random-pairs", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--tol", type=float, default=1.0e-8)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    return parser.parse_args()


def _model_input(model, strain_normalized, device):
    x = torch.as_tensor(strain_normalized, dtype=torch.float32, device=device)
    with torch.no_grad():
        return model._compute_kan_input_for_strain(x).detach()


def _raw_energy_from_model_input(model, x):
    if hasattr(model, "energy_net"):
        return model.energy_net(x)
    if hasattr(model, "KAN_W"):
        return model.KAN_W.forward(x)
    raise TypeError("Unsupported model: expected energy_net or KAN_W.")


def _corrected_energy_from_strain(model, x):
    return model.CalculateCorrectedW(x)


def _evaluate_jensen(coords, energy_fn, pair_indices, batch_size, tol):
    max_gap = -np.inf
    max_rel_gap = -np.inf
    worst_pair = None
    n_violations = 0
    n_pairs = int(pair_indices.shape[0])
    gaps_for_quantiles = []

    for start in range(0, n_pairs, batch_size):
        stop = min(start + batch_size, n_pairs)
        ids = pair_indices[start:stop]
        u = coords[ids[:, 0]]
        v = coords[ids[:, 1]]
        m = 0.5 * (u + v)

        eu = energy_fn(u).detach().reshape(-1)
        ev = energy_fn(v).detach().reshape(-1)
        em = energy_fn(m).detach().reshape(-1)

        rhs = 0.5 * (eu + ev)
        gap = (em - rhs).detach().cpu().numpy()
        rhs_np = rhs.detach().cpu().numpy()
        rel_gap = gap / (np.abs(rhs_np) + 1.0e-12)

        gaps_for_quantiles.append(gap)
        n_violations += int(np.count_nonzero(gap > tol))

        local_argmax = int(np.argmax(gap))
        local_max = float(gap[local_argmax])
        if local_max > max_gap:
            max_gap = local_max
            max_rel_gap = float(rel_gap[local_argmax])
            worst_pair = (
                int(ids[local_argmax, 0]),
                int(ids[local_argmax, 1]),
            )

    all_gaps = np.concatenate(gaps_for_quantiles) if gaps_for_quantiles else np.array([])
    return {
        "n_pairs": n_pairs,
        "n_violations": int(n_violations),
        "violation_fraction": float(n_violations / max(n_pairs, 1)),
        "tol": float(tol),
        "max_gap": float(max_gap),
        "max_relative_gap": float(max_rel_gap),
        "gap_p50": float(np.quantile(all_gaps, 0.50)) if all_gaps.size else None,
        "gap_p95": float(np.quantile(all_gaps, 0.95)) if all_gaps.size else None,
        "gap_p99": float(np.quantile(all_gaps, 0.99)) if all_gaps.size else None,
        "worst_pair_indices": worst_pair,
    }


def _make_pair_sets(n_samples, n_random_pairs, seed):
    adjacent = np.column_stack(
        [
            np.arange(max(n_samples - 1, 0), dtype=np.int64),
            np.arange(1, n_samples, dtype=np.int64),
        ]
    )
    rng = np.random.default_rng(seed)
    random_pairs = np.column_stack(
        [
            rng.integers(0, n_samples, size=n_random_pairs, dtype=np.int64),
            rng.integers(0, n_samples, size=n_random_pairs, dtype=np.int64),
        ]
    )
    return {
        "adjacent_history_pairs": adjacent,
        "random_history_pairs": random_pairs,
    }


def main():
    args = parse_args()
    device = torch.device(args.device)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")

    checkpoint_path = os.path.abspath(args.checkpoint)
    prediction_path = os.path.abspath(args.prediction_npz)
    out_dir = args.out_dir
    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(prediction_path), "convexity_check")
    out_dir = ensure_dir(os.path.abspath(out_dir))

    checkpoint = load_checkpoint(checkpoint_path)
    model_config = checkpoint["model_config"]
    model, _ = create_model(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model.to(device)

    data = np.load(prediction_path)
    strain = data["strain_normalized"].reshape(-1, 3).astype(np.float32)
    n_samples = int(strain.shape[0])
    pair_sets = _make_pair_sets(n_samples, args.n_random_pairs, args.seed)

    strain_coords = torch.as_tensor(strain, dtype=torch.float32, device=device)
    model_coords = _model_input(model, strain, device)

    results = {
        "checkpoint": checkpoint_path,
        "prediction_npz": prediction_path,
        "model_config": model_config,
        "n_samples": n_samples,
        "definition": "convex iff E((u+v)/2) <= 0.5*(E(u)+E(v)); positive Jensen gap is a violation",
        "spaces": {},
    }

    spaces = {
        "model_input_raw_energy": (
            model_coords,
            lambda x: _raw_energy_from_model_input(model, x),
            "Checks convexity with respect to the actual NN input variables.",
        ),
        "normalized_strain_corrected_energy": (
            strain_coords,
            lambda x: _corrected_energy_from_strain(model, x),
            "Diagnostic only: checks convexity with respect to normalized strain.",
        ),
    }

    for space_name, (coords, energy_fn, description) in spaces.items():
        results["spaces"][space_name] = {"description": description, "pair_sets": {}}
        for pair_name, pair_indices in pair_sets.items():
            results["spaces"][space_name]["pair_sets"][pair_name] = _evaluate_jensen(
                coords=coords,
                energy_fn=energy_fn,
                pair_indices=pair_indices,
                batch_size=args.batch_size,
                tol=args.tol,
            )

    json_path = os.path.join(out_dir, "convexity_check_summary.json")
    txt_path = os.path.join(out_dir, "convexity_check_summary.txt")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    lines = [
        "Energy convexity check",
        "======================",
        f"checkpoint     : {checkpoint_path}",
        f"prediction_npz : {prediction_path}",
        f"n_samples      : {n_samples}",
        "",
        "Convexity requires: E((u+v)/2) <= 0.5*(E(u)+E(v))",
        "Violation means: positive Jensen gap = E(mid) - average > tol",
        "",
    ]
    for space_name, space_data in results["spaces"].items():
        lines.append(space_name)
        lines.append("-" * len(space_name))
        lines.append(space_data["description"])
        for pair_name, rec in space_data["pair_sets"].items():
            lines.append(
                f"  {pair_name}: violations={rec['n_violations']}/{rec['n_pairs']} "
                f"({rec['violation_fraction']:.3e}), max_gap={rec['max_gap']:.6e}, "
                f"max_rel_gap={rec['max_relative_gap']:.6e}, p99={rec['gap_p99']:.6e}"
            )
        lines.append("")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\n".join(lines))
    print(f"Saved: {txt_path}")
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()
