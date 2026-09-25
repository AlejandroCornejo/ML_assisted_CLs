#!/usr/bin/env python3
"""Generate a common mixed-loading witness for the SC- and MC-RVEs.

The path and the displayed checkpoints are fixed without reference to the
resulting path errors.  It is explanatory evidence, not another selection set.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
MC = HERE.parent
ROOT = MC.parent
OUTPUT = MC / "results/common_path_evidence_v1"
VALIDATION = (MC / "results/feature_count_analysis_v1/validation_audit_v1"
              / "validation_summary.json")
TRAINING = MC / "results/feature_count_amendment_v2/training"
FEATURES = MC / "results/feature_count_amendment_v2/feature_tables"
RULE = MC / "protocol/feature_count_amendment_v2_training_rule.json"

for path in (ROOT / "00_rve", MC):
    sys.path.insert(0, str(path))

from periodic_fom import PeriodicRVE  # noqa: E402
from _material_law_guard_claude import true_neo_hookean_active  # noqa: E402
from adaptive_continuation import solve as continuation_solve  # noqa: E402
from protocol import train_material_b as base  # noqa: E402
from protocol.evaluate_final_models import predict  # noqa: E402
from protocol.train_feature_count_amendment_v2 import load_rule  # noqa: E402
from protocol.training_setup import FlexibleEnergy  # noqa: E402


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mixed_path(points_per_stage: int = 12) -> tuple[np.ndarray, np.ndarray]:
    """Axial extension, then transverse compression, then positive shear."""
    if points_per_stage < 2:
        raise ValueError("At least two increments per stage are required")
    states = [np.zeros(3)]
    parameter = [0.0]
    for stage, target in enumerate((np.array([0.18, 0.0, 0.0]),
                                    np.array([0.18, -0.04, 0.0]),
                                    np.array([0.18, -0.04, 0.08])), start=1):
        start = states[-1].copy()
        for index in range(1, points_per_stage + 1):
            fraction = index / points_per_stage
            states.append(start + fraction * (target - start))
            parameter.append(stage - 1 + fraction)
    return np.asarray(parameter), np.asarray(states)


def mesh_and_fields(rve: PeriodicRVE, strain: np.ndarray, q: np.ndarray) -> dict:
    displacement = rve.T @ q + rve._g(strain)
    rve.assembler.Assemble(displacement)
    assembler = rve.assembler
    F = np.asarray(assembler._F, dtype=float)
    S = np.empty(F.shape, dtype=float)
    S[..., 0, 0] = assembler._S_voigt[..., 0]
    S[..., 1, 1] = assembler._S_voigt[..., 1]
    S[..., 0, 1] = S[..., 1, 0] = assembler._S_voigt[..., 2]
    P = F @ S
    J = np.linalg.det(F)
    sigma = np.einsum("...ij,...kj->...ik", P, F) / J[..., None, None]
    young = np.asarray(assembler.young, dtype=float)[:, None]
    poisson = np.asarray(assembler.poisson, dtype=float)[:, None]
    lame = young * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
    sigma33 = lame * np.log(J) / J
    s11, s22, s12 = sigma[..., 0, 0], sigma[..., 1, 1], sigma[..., 0, 1]
    von_mises = np.sqrt(0.5 * ((s11 - s22) ** 2 + (s22 - sigma33) ** 2
                              + (sigma33 - s11) ** 2) + 3.0 * s12 ** 2)
    weights = np.asarray(assembler.w_detJ, dtype=float)
    von_mises_element = np.sum(weights * von_mises, axis=1) / np.sum(weights, axis=1)

    nodes = list(rve._mp.Nodes)
    node_index = {node.Id: index for index, node in enumerate(nodes)}
    xy = np.asarray([[node.X0, node.Y0] for node in nodes], dtype=float)
    triangles = np.asarray([[node_index[node.Id] for node in list(element.GetGeometry())[:3]]
                            for element in rve._mp.Elements], dtype=int)
    u_nodal = displacement[rve._eq_map]
    return dict(xy=xy, triangles=triangles, u_nodal=u_nodal,
                displacement_magnitude=np.linalg.norm(u_nodal, axis=1),
                von_mises_element=von_mises_element,
                minimum_micro_J=np.asarray(J.min()), maximum_micro_J=np.asarray(J.max()))


def solve_path(mesh: Path, area: float, states: np.ndarray, label: str) -> tuple[dict, dict]:
    rve = PeriodicRVE(mesh, cell_area=area)
    stress = np.empty((len(states), 3))
    energy = np.empty(len(states))
    previous_e = None
    previous_q = None
    q = None
    started = time.perf_counter()
    for index, strain in enumerate(states):
        stress[index], q, _attempts = continuation_solve(
            rve, strain, max_increment=0.01, min_increment=1.0e-6,
            start=previous_e, q_start=previous_q)
        energy[index] = rve.homogenized_energy()
        previous_e, previous_q = strain.copy(), q.copy()
        print(f"{label}: {index + 1}/{len(states)}", flush=True)
    fields = mesh_and_fields(rve, states[-1], q)
    return dict(stress=stress, energy=energy,
                elapsed_seconds=np.asarray(time.perf_counter() - started)), fields


def validation_median_rows() -> list[dict]:
    audit = json.loads(VALIDATION.read_text(encoding="utf-8"))
    selected = []
    for core in ("ICNN", "ICKAN"):
        for count in (2, 4, 6):
            group = sorted((row for row in audit["rows"]
                            if row["core"] == core and row["feature_type"] == "learned"
                            and row["feature_count"] == count),
                           key=lambda row: row["best_validation_score"])
            if len(group) != 3:
                raise ValueError(f"Incomplete validation group: {core}, m={count}")
            selected.append(group[1])
    return selected


def load_model(row: dict) -> tuple[FlexibleEnergy, dict, Path]:
    count = int(row["feature_count"])
    _, _, manifest, _, _, loaded_count = load_rule(FEATURES / f"m{count:02d}", RULE)
    if loaded_count != count:
        raise ValueError("Feature-table count mismatch")
    path = TRAINING / row["slug"] / "model.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if (checkpoint.get("name") != row["model"] or checkpoint.get("seed") != row["seed"]
            or checkpoint.get("best_validation_score") != row["best_validation_score"]):
        raise ValueError(f"Checkpoint metadata mismatch: {row['slug']}")
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        model = FlexibleEnergy(**checkpoint["configuration"]).double()
    finally:
        torch.set_default_dtype(previous)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    base._check_model(model)
    return model, manifest["scales"], path


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite frozen evidence: {OUTPUT}")
    OUTPUT.mkdir(parents=True)
    parameter, states = mixed_path()
    sc_area = float(np.load(ROOT / "03_data/data.npz")["cell_area"])
    with true_neo_hookean_active():
        mc_response, mc_fields = solve_path(
            MC / "preflight_reference_v1/reference", 4.0, states, "MC-RVE")
        sc_response, sc_fields = solve_path(ROOT / "03_data/rve_mesh", sc_area, states, "SC-RVE")

    predictions = {}
    selected = []
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    for row in validation_median_rows():
        model, scales, checkpoint = load_model(row)
        W, S, _ = predict(model, states, scales)
        key = f"{row['core'].lower()}_m{int(row['feature_count']):02d}"
        predictions[f"W_{key}"] = W
        predictions[f"S_{key}"] = S
        selected.append(dict(core=row["core"], feature_count=row["feature_count"],
                             seed=row["seed"], slug=row["slug"],
                             best_validation_score=row["best_validation_score"],
                             checkpoint_sha256=digest(checkpoint)))

    arrays = dict(path_parameter=parameter, strain=states,
                  W_mc_fom=mc_response["energy"], S_mc_fom=mc_response["stress"],
                  W_sc_fom=sc_response["energy"], S_sc_fom=sc_response["stress"],
                  **predictions)
    for prefix, fields in (("mc", mc_fields), ("sc", sc_fields)):
        arrays.update({f"{prefix}_{key}": value for key, value in fields.items()})
    np.savez_compressed(OUTPUT / "evidence.npz", **arrays)
    report = dict(status="complete", purpose="Explanatory common-path witness; not model selection.",
                  path_definition=dict(stages=["axial extension", "transverse compression", "positive shear"],
                                       endpoints=[[0.18, 0.0, 0.0], [0.18, -0.04, 0.0],
                                                  [0.18, -0.04, 0.08]],
                                       strain_coordinates=["E11", "E22", "2E12"]),
                  selected_checkpoints=selected,
                  selection_rule="For each core and m, use the run with the median validation score over seeds 16, 29, and 47.",
                  selection_used_path_values=False,
                  mc_elapsed_seconds=float(mc_response["elapsed_seconds"]),
                  sc_elapsed_seconds=float(sc_response["elapsed_seconds"]),
                  evidence_sha256=digest(OUTPUT / "evidence.npz"),
                  units=dict(energy="Pa", stress="Pa", coordinates="cell units",
                             displacement="cell units", von_mises="Pa"),
                  von_mises_definition="Local 3D Cauchy von Mises stress. In-plane Cauchy stress is P F^T/J; sigma33=lambda log(J)/J for the plane-strain compressible Neo-Hookean matrix.")
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
