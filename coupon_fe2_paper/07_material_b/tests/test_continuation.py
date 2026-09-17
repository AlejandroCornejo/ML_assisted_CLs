"""Control-flow tests, distinct from the actual FOM comparison."""
import sys
import types
import unittest
from unittest.mock import patch
import numpy as np
import adaptive_continuation as continuation


class FakeRVE:
    n_ind = n_dof = 1
    ind_xy = dof_xy = np.array([[1., 0.]])
    ind_comp = dof_comp = np.array([0])
    T = np.eye(1)

    def _g(self, strain):
        return np.zeros(1)

    def solve(self, strain, u_ind_init, E_start):
        if np.linalg.norm(strain-E_start) > .006:
            raise RuntimeError("Synthetic increment failure")
        return np.zeros(3), u_ind_init.copy()


class ContinuationTests(unittest.TestCase):
    def test_halving_keeps_last_successful_state_and_restores_density(self):
        law = types.SimpleNamespace(DeformationGradientFromGreenLagrange2D=lambda e: np.diag([1+e[0], 1.]))
        stats = (dict(min_micro_J=1., relative_reduced_residual=0.), None, None)
        old_density = continuation.pf.SUBSTEPS_PER_UNIT_STRAIN
        with patch.dict(sys.modules, {"fom_solver_rve": law}), patch.object(continuation, "field_stats", return_value=stats):
            _s, q, attempts = continuation.solve(FakeRVE(), [.02, 0, 0])
        self.assertAlmostEqual(q[0], .02)
        self.assertTrue(any(not row["ok"] for row in attempts))
        self.assertTrue(all(row["increment_norm"] <= .006 for row in attempts if row["ok"]))
        self.assertEqual(continuation.pf.SUBSTEPS_PER_UNIT_STRAIN, old_density)

    def test_failure_at_minimum_increment_is_not_silently_dropped(self):
        law = types.SimpleNamespace(DeformationGradientFromGreenLagrange2D=lambda e: np.diag([1+e[0], 1.]))
        old_density = continuation.pf.SUBSTEPS_PER_UNIT_STRAIN
        with patch.dict(sys.modules, {"fom_solver_rve": law}), self.assertRaises(continuation.ContinuationFailure) as context:
            continuation.solve(FakeRVE(), [.02, 0, 0], max_increment=.01, min_increment=.006)
        self.assertEqual(len(context.exception.attempts), 1)
        self.assertFalse(context.exception.attempts[0]["ok"])
        self.assertEqual(continuation.pf.SUBSTEPS_PER_UNIT_STRAIN, old_density)


if __name__ == "__main__":
    unittest.main()
