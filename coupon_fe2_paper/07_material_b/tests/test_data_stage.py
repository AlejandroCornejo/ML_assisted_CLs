import json
import unittest
from pathlib import Path

import numpy as np

from protocol.run_data_stage import chunk_plan, enforce_physical_screen, selection


BASE = Path(__file__).resolve().parents[1]


class DataStagePlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = json.loads((BASE / "protocol/data_protocol_v1.json").read_text())
        cls.data = np.load(BASE / "results/data_protocol_design_v1.npz", allow_pickle=False)
        cls.lower, cls.upper = cls.data["lower"], cls.data["upper"]

    @classmethod
    def tearDownClass(cls):
        cls.data.close()

    def test_test_chunks_cover_every_state_once(self):
        plans = [chunk_plan(self.data, "test", i, 64, self.lower, self.upper) for i in range(64)]
        indices = [row["global_index"] for plan in plans for row in plan]
        self.assertEqual(len(indices), 512)
        self.assertEqual(sorted(indices), list(range(512)))
        self.assertTrue(all(len(plan) == 8 for plan in plans))

    def test_path_chunks_preserve_each_path_order(self):
        names = self.data["path_names"]
        for index, name in enumerate(names):
            plan = chunk_plan(self.data, "paths", index, len(names), self.lower, self.upper)
            self.assertEqual(len(plan), 40)
            self.assertTrue(all(row["path_name"] == str(name) for row in plan))
            self.assertEqual([row["path_step"] for row in plan], list(range(1, 41)))

    def test_selection_roles_have_declared_lengths(self):
        expected = {"fit": 4200, "validation": 512, "test": 512, "paths": 400,
                    "audit": 64, "cold": 24, "reference": 1}
        for split, count in expected.items():
            points, metadata = selection(self.data, split)
            self.assertEqual(len(points), count)
            self.assertEqual(len(metadata), count)

    def test_physical_screen_accepts_strictly_admissible_state(self):
        fields = {"min_micro_J": 0.8}
        boundary = {"min_polygon_gap": 0.1, "max_periodic_jump_error": 1e-14,
                    "self_intersection": False}
        enforce_physical_screen(fields, boundary, self.spec)

    def test_physical_screen_rejects_each_failure_mode(self):
        admissible_fields = {"min_micro_J": 0.8}
        admissible_boundary = {"min_polygon_gap": 0.1,
                               "max_periodic_jump_error": 1e-14,
                               "self_intersection": False}
        cases = [
            ({"min_micro_J": 0.0}, admissible_boundary),
            (admissible_fields, dict(admissible_boundary, min_polygon_gap=0.0)),
            (admissible_fields, dict(admissible_boundary, max_periodic_jump_error=1e-8)),
            (admissible_fields, dict(admissible_boundary, self_intersection=True)),
            (admissible_fields, dict(admissible_boundary, max_periodic_jump_error=np.nan)),
        ]
        for fields, boundary in cases:
            with self.subTest(fields=fields, boundary=boundary):
                with self.assertRaises(RuntimeError):
                    enforce_physical_screen(fields, boundary, self.spec)


if __name__ == "__main__":
    unittest.main()
