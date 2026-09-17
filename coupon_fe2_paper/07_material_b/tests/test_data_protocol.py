import json
import unittest
from pathlib import Path

import numpy as np

from protocol.prepare_design import build


BASE = Path(__file__).resolve().parents[1]


class DataProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = json.loads((BASE / "protocol/data_protocol_v1.json").read_text())
        cls.arrays, cls.report = build(cls.spec)

    def test_declared_counts_and_checks(self):
        self.assertEqual(self.arrays["E_fit"].shape, (4200, 3))
        self.assertEqual(self.arrays["E_validation"].shape, (512, 3))
        self.assertEqual(self.arrays["E_test"].shape, (512, 3))
        self.assertTrue(all(self.report["checks"].values()))

    def test_fit_contains_volume_faces_and_corners(self):
        kinds, counts = np.unique(self.arrays["fit_kind"], return_counts=True)
        self.assertEqual(dict(zip(kinds.tolist(), counts.tolist())),
                         {"corner": 8, "face": 96, "volume": 4096})

    def test_paths_are_distinct_from_selection_sets_except_reference(self):
        scored = self.arrays["path_states"][:, 1:, :].reshape(-1, 3)
        selected = np.vstack((self.arrays["E_fit"], self.arrays["E_validation"]))
        distances = np.linalg.norm(scored[:, None, :] - selected[None, :, :], axis=2)
        self.assertGreater(float(distances.min()), 1e-12)

    def test_audits_use_only_frozen_statistical_states(self):
        sources = {"fit_volume": self.arrays["E_fit"][:4096],
                   "validation": self.arrays["E_validation"], "test": self.arrays["E_test"],
                   "fit_boundary": self.arrays["E_fit"][4096:]}
        for state, split, index in zip(self.arrays["mesh_audit_states"],
                                       self.arrays["mesh_audit_split"],
                                       self.arrays["mesh_audit_index"]):
            np.testing.assert_array_equal(state, sources[str(split)][int(index)])


if __name__ == "__main__":
    unittest.main()
