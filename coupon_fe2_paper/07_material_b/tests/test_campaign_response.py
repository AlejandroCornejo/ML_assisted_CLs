import unittest

import numpy as np

from protocol.plot_campaign_response import linearization, path_metrics, verify_model_lock


class ReferencePathDiagnosticTests(unittest.TestCase):
    def test_work_conjugate_shear_and_energy(self):
        e = np.array([[0., 0., .2]])
        d0 = np.diag([3., 4., 5.])
        s, w = linearization(e, np.zeros(3), 0., d0)
        np.testing.assert_allclose(s, [[0., 0., 1.]])
        np.testing.assert_allclose(w, [.1])

    def test_linear_response_has_zero_nonlinearity(self):
        e = np.array([[.1, 0, 0], [.2, 0, 0]])
        d0 = np.eye(3)
        s, _ = linearization(e, np.zeros(3), 0., d0)
        metrics = path_metrics(e, s, np.tile(d0, (2, 1, 1)), np.zeros(3), 0., d0)
        self.assertTrue(all(value == 0 for value in metrics.values()))

    def test_endpoint_metric_uses_fom_norm_not_reference_prediction(self):
        e = np.array([[1., 0, 0]])
        s = np.array([[2., 0, 0]])
        metrics = path_metrics(e, s, np.eye(3)[None], np.zeros(3), 0., np.eye(3))
        self.assertAlmostEqual(metrics["stress_endpoint_relative_deviation"], .5)

    def test_zero_denominator_is_rejected(self):
        with self.assertRaises(ValueError):
            path_metrics(np.zeros((1, 3)), np.zeros((1, 3)), np.eye(3)[None],
                         np.zeros(3), 0., np.eye(3))

    def test_paths_cannot_be_opened_with_missing_model_checkpoints(self):
        spec = {"models": {"primary": ["Free", "ICNN-fixed"],
                           "initialization_seeds": [16, 29]}}
        with self.assertRaises(RuntimeError):
            verify_model_lock(spec, {"status": "locked", "checkpoints": []})

    def test_paths_cannot_be_opened_before_checkpoint_lock(self):
        spec = {"models": {"primary": ["Free"], "initialization_seeds": [16]}}
        with self.assertRaises(RuntimeError):
            verify_model_lock(spec, {"status": "training", "checkpoints": [
                {"model": "Free", "seed": 16, "path": "unused", "sha256": "unused"}]})
