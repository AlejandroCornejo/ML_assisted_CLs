"""Synthetic final-evaluator checks; never open reserved Material-B labels."""
from __future__ import annotations

import unittest

import numpy as np
import torch

from protocol import evaluate_final_models as final


class QuadraticEnergy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("matrix", torch.tensor(
            [[2., .3, 0.], [.3, 4., .2], [0., .2, 3.]], dtype=torch.float64))

    def energy_and_stress(self, x, *, create_graph):
        stress = x @ self.matrix
        energy = .5 * (x * stress).sum(dim=1, keepdim=True)
        return energy, stress


class FinalEvaluationTests(unittest.TestCase):
    def test_physical_stress_and_engineering_tangent_scaling(self):
        scales = dict(strain_scale=.2, energy_scale=5.)
        strain = np.array([[.01, -.02, .03], [.07, .04, -.01]])
        energy, stress, tangent = final.predict(QuadraticEnergy(), strain, scales,
                                                 batch=1)
        matrix = QuadraticEnergy().matrix.numpy()
        expected_stress = (strain / .2) @ matrix * (5 / .2)
        expected_energy = .5 * np.einsum("ni,ij,nj->n", strain/.2, matrix,
                                         strain/.2) * 5
        np.testing.assert_allclose(energy, expected_energy)
        np.testing.assert_allclose(stress, expected_stress)
        np.testing.assert_allclose(tangent, np.broadcast_to(matrix*5/.2**2,
                                                            (2, 3, 3)))

    def test_fit_derived_floor_and_aggregate_denominator_are_distinct(self):
        target = np.array([0., 2.])
        predicted = np.array([1., 3.])
        self.assertAlmostEqual(final.aggregate_percent(predicted, target),
                               100*np.sqrt(2)/2)
        distribution = final.distribution_percent(predicted, target, floor=2.)
        self.assertEqual(distribution["maximum"], 50.)
        self.assertEqual(distribution["median"], 50.)

    def test_median_validation_seed_not_best_test_seed(self):
        rows = []
        for seed, validation, test_error in ((16, .3, 1.), (29, .1, 100.),
                                             (47, .2, 2.)):
            error = dict(energy=test_error, stress=test_error,
                         tangent=test_error)
            rows.append(dict(model="Free", seed=seed,
                validation_score=validation,
                test=dict(aggregate_percent=error),
                paths_aggregate=dict(aggregate_percent=error),
                paths={"axial_x": dict(aggregate_percent=error)}))
        original = final.NAMES
        try:
            final.NAMES = ("Free",)
            result = final.summarize(rows, None)["Free"]
        finally:
            final.NAMES = original
        self.assertEqual(result["median_validation_seed"], 47)
        self.assertAlmostEqual(result["test_aggregate_percent"]["stress"]["mean"],
                               103/3)
        self.assertEqual(result["individual_paths_percent"]["axial_x"]["energy"]
                         ["by_seed"]["29"], 100.)


if __name__ == "__main__":
    unittest.main()
