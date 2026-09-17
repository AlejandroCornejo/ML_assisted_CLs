"""Exact/synthetic diagnostics tests; not a physical RVE validation."""
import unittest
import numpy as np
from audit_expanded_fields import crossings, curved_rim, rim_topology, sampled_map


class FieldAuditTests(unittest.TestCase):
    def setUp(self):
        self.xy = np.array([[0., 0.], [1., 0.], [0., 1.], [.5, 0.], [.5, .5], [0., .5]])
        self.triangles = np.arange(6)[None, :]

    def test_affine_mapping_determinant(self):
        F = np.array([[1.2, .3], [0., .8]])
        result = sampled_map(self.xy, self.xy@F.T+[2., 3.], self.triangles)
        self.assertAlmostEqual(result["minimum_sampled_det_F"], np.linalg.det(F))
        self.assertEqual(result["nonpositive_elements"], 0)

    def test_inverted_mapping_is_identified(self):
        result = sampled_map(self.xy, self.xy@np.diag([-1., 1.]), self.triangles)
        self.assertAlmostEqual(result["minimum_sampled_det_F"], -1.)
        self.assertEqual(result["nonpositive_elements"], 1)

    def test_crossing_is_localized_inside_segments(self):
        result = crossings(np.array([[0., 0.], [1., 1.], [0., 1.], [1., 0.]]))
        self.assertEqual(len(result), 1)
        np.testing.assert_allclose(result[0]["point"], [.5, .5])
        np.testing.assert_allclose(result[0]["fractions"], [.5, .5])

    def test_topology_and_curved_order_with_midnode_start(self):
        ids = np.array([3, 1, 4, 2, 5, 0])
        valid, edges = rim_topology(ids, self.triangles)
        self.assertTrue(valid)
        curved = curved_rim(ids, edges, self.xy)
        self.assertEqual(len(curved), 24)
        self.assertFalse(crossings(curved))
        np.testing.assert_allclose(curved[[0, 8, 16]], self.xy[[1, 2, 0]])
        reverse = curved_rim(ids[::-1], edges, self.xy)
        self.assertFalse(crossings(reverse))

    def test_invalid_rim_order_is_rejected(self):
        valid, _edges = rim_topology(np.arange(6), self.triangles)
        self.assertFalse(valid)


if __name__ == "__main__":
    unittest.main()
