"""Exact arithmetic tests, including a caution against equating baseline error with nonlinearity."""
import unittest
import numpy as np
from audit_reference_linearity import diagnose


class ReferenceLinearityTests(unittest.TestCase):
    def setUp(self):
        self.e = np.array([[.1, 0., 0.], [0., .2, 0.], [0., 0., .3]])
        self.D0 = np.diag([10., 20., 30.])

    def test_exact_linear_reference_has_zero_error(self):
        result = diagnose(self.e, self.e@self.D0.T, self.D0)
        self.assertEqual(result["aggregate_relative_L2"], 0.)

    def test_wrong_reference_can_have_large_error_for_linear_response(self):
        stress = 2*self.e@self.D0.T
        self.assertAlmostEqual(diagnose(self.e, stress, self.D0)["aggregate_relative_L2"], .5)
        fitted, _residual, rank, _sv = np.linalg.lstsq(self.e, stress, rcond=None)
        self.assertEqual(rank, 3)
        self.assertLess(diagnose(self.e, stress, fitted.T)["aggregate_relative_L2"], 1e-15)

    def test_nonfinite_mask_is_explicit(self):
        stress = self.e@self.D0.T
        stress[0] = np.nan
        result = diagnose(self.e, stress, self.D0)
        self.assertEqual(result["finite"], 2)
        self.assertEqual(result["excluded_nonfinite"], 1)

    def test_zero_reference_uses_recorded_floor(self):
        result = diagnose(np.zeros((1, 3)), np.zeros((1, 3)), self.D0)
        self.assertEqual(result["aggregate_relative_L2"], 0.)
        self.assertEqual(result["stress_norm_below_one_Pa"], 1)


if __name__ == "__main__":
    unittest.main()
