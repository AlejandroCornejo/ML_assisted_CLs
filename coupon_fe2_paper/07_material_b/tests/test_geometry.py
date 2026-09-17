"""Small deterministic tests for geometry checks and polygon screening."""
import copy
import json
import unittest
from pathlib import Path
import numpy as np
from geometry import cavity_parameters, geometry_checks
from audit_pilot import polygon_distance, self_intersection, rank_one_screen


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.spec = json.loads((Path(__file__).parent.parent / "pilot_spec.json").read_text())
        self.square = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])

    def test_area_and_aspect(self):
        for hole, raw in zip(cavity_parameters(self.spec), self.spec["cavities"]):
            self.assertAlmostEqual(np.pi*hole["a"]*hole["b"],
                                   raw["area_over_cell"]*self.spec["cell_side"]**2)
            self.assertAlmostEqual(hole["a"]/hole["b"], raw["aspect"])

    def test_positive_circle_separation(self):
        self.assertGreater(geometry_checks(self.spec)["periodic_ligament_lower_bound"], 0)

    def test_overlapping_centers_rejected(self):
        invalid = copy.deepcopy(self.spec)
        invalid["cavities"][1]["center_over_L"] = invalid["cavities"][0]["center_over_L"]
        with self.assertRaises(ValueError):
            geometry_checks(invalid)

    def test_boundary_crossing_rejected(self):
        invalid = copy.deepcopy(self.spec)
        invalid["cavities"][0]["center_over_L"] = [.49, 0]
        with self.assertRaises(ValueError):
            geometry_checks(invalid)

    def test_wrong_porosity_rejected(self):
        self.spec["porosity"] = .19
        with self.assertRaises(ValueError):
            geometry_checks(self.spec)

    def test_segment_gap(self):
        self.assertAlmostEqual(polygon_distance(self.square, self.square+[2., 0.]), 1.)

    def test_intersection_and_containment(self):
        self.assertEqual(polygon_distance(self.square, self.square+[.5, .5]), 0)
        self.assertEqual(polygon_distance(self.square, .2*self.square+[.3, .3]), 0)

    def test_self_intersection(self):
        self.assertFalse(self_intersection(self.square))
        self.assertTrue(self_intersection(self.square[[0, 2, 1, 3]]))

    def test_rank_one_stress_term_not_omitted(self):
        # W(E)=E:E: at F=.5 I, S=-.75 I and D=diag(2,2,1).
        # A unit simple-shear rank-one H has curvature .25-.75=-.5.
        D = np.diag([2., 2., 1.])
        self.assertAlmostEqual(rank_one_screen([-.375, -.375, 0], [-.75, -.75, 0], D), -.5)
        self.assertAlmostEqual(rank_one_screen([0, 0, 0], [0, 0, 0], D), 1.)


if __name__ == "__main__":
    unittest.main()
