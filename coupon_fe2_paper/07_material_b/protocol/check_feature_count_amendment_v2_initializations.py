"""Verify paired fixed/learned initial states for every sweep feature count."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from protocol.prepare_design import digest
from protocol.select_features import BASE
from protocol.train_material_b import LABELS, _atomic_json
from protocol.training_setup import check_initializations

COUNTS = (2, 4, 6)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-root", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for count in COUNTS:
        folder = args.features_root / f"m{count:02d}"
        output = folder / "initialization_checks.json"
        if output.exists():
            raise FileExistsError(f"Do not overwrite {output}")
        result = check_initializations(folder, LABELS, folder / "preparation_recipe.json")
        result.update(protocol_id="material_B_feature_count_amendment_v2", feature_count=count,
                      checker_sha256=digest(Path(__file__)))
        if not result.get("passed") or len(result.get("runs", ())) != 12 or len(result.get("paired_initializations", ())) != 6:
            raise RuntimeError(f"Initialization check failed for m={count}")
        _atomic_json(output, result)
        rows.append(dict(feature_count=count, checks_sha256=digest(output),
                         feature_table_sha256=result["feature_table_sha256"],
                         pairs=len(result["paired_initializations"])))
    _atomic_json(args.features_root / "initialization_checks_manifest.json",
                 dict(status="complete", protocol_id="material_B_feature_count_amendment_v2",
                      labels_sha256=digest(LABELS), rows=rows,
                      interpretation="Initialization-only checks; no optimizer step, validation, test, or path label use."))
    print(json.dumps(dict(status="complete", rows=rows), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
