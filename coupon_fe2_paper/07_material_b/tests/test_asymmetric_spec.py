"""Frozen-domain integrity checks; not physical validation of interior states."""
import json
import unittest
from pathlib import Path
import numpy as np


class AsymmetricSpecTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base = Path(__file__).parent.parent
        cls.spec = json.loads((base / "asymmetric_box_spec.json").read_text())
        cls.original = json.loads((base / "preflight_spec.json").read_text())

    def test_asymmetric_bounds_are_the_declared_candidate(self):
        self.assertEqual(self.spec["candidate_box"],
                         {"E11":[-.04, .20], "E22":[-.04, .20], "2E12":[-.08, .08]})

    def test_all_eight_corners_are_present_once(self):
        limits = self.spec["candidate_box"]
        corners = {(x, y, g) for x in limits["E11"] for y in limits["E22"] for g in limits["2E12"]}
        recorded = {tuple(v) for k, v in self.spec["states"].items() if k.startswith("corner_")}
        self.assertEqual(recorded, corners)
        self.assertEqual(sum(k.startswith("corner_") for k in self.spec["states"]), 8)

    def test_all_targets_are_inside_and_macro_C_is_positive_definite(self):
        bounds = np.array([self.spec["candidate_box"][k] for k in ("E11", "E22", "2E12")])
        for e in self.spec["states"].values():
            self.assertTrue(np.all(bounds[:, 0] <= e) and np.all(e <= bounds[:, 1]))
            C = np.array([[1+2*e[0], e[2]], [e[2], 1+2*e[1]]])
            self.assertGreater(np.linalg.eigvalsh(C).min(), 0.)

    def test_only_normal_tension_is_extended(self):
        old, new = self.original["candidate_box"], self.spec["candidate_box"]
        self.assertEqual(new["E11"][0], old["E11"][0])
        self.assertEqual(new["E22"][0], old["E22"][0])
        self.assertEqual(new["2E12"], old["2E12"])
        self.assertGreater(new["E11"][1], old["E11"][1])
        self.assertGreater(new["E22"][1], old["E22"][1])

    def test_reused_states_are_exact_original_targets(self):
        old = self.original["domain_states"]
        for name in self.spec["reuse_states"]:
            self.assertEqual(self.spec["states"][name], old[name])


if __name__ == "__main__":
    unittest.main()
