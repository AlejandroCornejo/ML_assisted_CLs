"""Synthetic checks: never consult scientific validation/test/path labels."""
import json
import unittest

import numpy as np

from protocol.select_features import (RECIPE, affine_design, candidate_bank,
    fit_scales, kinematics, load_fit_reference, raw_features, select_indices,
    tangent_dictionary, weighted_system)


class FeaturePreparationTests(unittest.TestCase):
    def setUp(self):
        self.recipe = json.loads(RECIPE.read_text())
        self.setting = self.recipe["feature_selection"]
        self.specs = candidate_bank(self.setting)[::100]
        self.e = np.array([[0., 0., 0.], [.2, -.04, .08], [.05, .03, -.07]])

    def test_candidate_bank_is_fixed_size_and_strictly_startable(self):
        bank = candidate_bank(self.setting)
        self.assertEqual(bank.shape, (2028, 5))
        _, p, q, b, c = bank.T
        self.assertTrue(np.all((p > .5) & (q > .5)))
        self.assertTrue(np.all((b > 0) & (c > 0) & (b < 2*p-1) & (c < 2*q-1)))
        np.testing.assert_array_equal(bank, candidate_bank(self.setting))

    def test_selection_loader_cannot_open_reserved_arrays(self):
        n = 4
        allowed = dict(E_fit=np.zeros((n, 3)), S_fit=np.ones((n, 3)),
            W_fit=np.ones(n), E_reference=np.zeros((1, 3)),
            S_reference=np.zeros((1, 3)), W_reference=np.zeros(1),
            D_reference=np.eye(3)[None])
        accesses = []
        class GuardedStore:
            def __getitem__(self, key):
                accesses.append(key)
                if key not in allowed:
                    raise AssertionError(f"Reserved array accessed: {key}")
                return allowed[key]
        loaded = load_fit_reference(GuardedStore(), self.recipe["selection_allowed_arrays"])
        self.assertEqual(accesses, self.recipe["selection_allowed_arrays"])
        self.assertEqual(set(loaded), set(allowed))

    def test_selection_loader_rejects_expanded_access_policy(self):
        with self.assertRaises(ValueError):
            load_fit_reference({}, self.recipe["selection_allowed_arrays"]+["E_test"])

    def test_reference_values_and_affine_normalization(self):
        z = raw_features(self.e[:1], self.specs)
        np.testing.assert_allclose(z[0], (1/self.specs[:, 1:3]).sum(axis=1), atol=2e-15)
        a, jac = affine_design(self.e[:1], self.specs)
        np.testing.assert_allclose(a, 0, atol=2e-15)
        np.testing.assert_allclose(jac, 0, atol=2e-15)

    def test_energy_gradient_by_central_differences(self):
        _, jac = affine_design(self.e, self.specs)
        for k in range(3):
            h = np.eye(3)[k]*1e-6
            fd = (affine_design(self.e+h, self.specs)[0]-affine_design(self.e-h, self.specs)[0])/2e-6
            np.testing.assert_allclose(fd, jac[:, k, :], rtol=2e-6, atol=2e-9)

    def test_reference_hessian_by_central_differences(self):
        tangent = tangent_dictionary(self.specs)
        for k in range(3):
            h = np.eye(3)[k:k+1]*1e-6
            fd = (affine_design(h, self.specs)[1][0]-affine_design(-h, self.specs)[1][0])/2e-6
            np.testing.assert_allclose(fd, tangent[:, k, :], rtol=2e-6, atol=4e-9)
        np.testing.assert_allclose(tangent, tangent.swapaxes(0, 1), atol=1e-14)

    def test_physical_reference_raw_feature_gradient_is_isotropic(self):
        rho = 2-self.specs[:, 3]/self.specs[:, 1]-self.specs[:, 4]/self.specs[:, 2]
        for k in range(3):
            h = np.eye(3)[k:k+1]*1e-6
            fd = (raw_features(h, self.specs)-raw_features(-h, self.specs))/2e-6
            np.testing.assert_allclose(fd[0], rho if k < 2 else 0, rtol=1e-6, atol=2e-9)

    def test_weighted_system_equals_fit_loss(self):
        e = self.e[1:]
        s = np.array([[3., 2., 1.], [4., 1., -2.]])*1e6
        w = np.array([4., 2.])*1e5
        scales = fit_scales(e, s, w, np.eye(3)*1e9, 1e-8)
        a, jac = affine_design(e, self.specs)
        matrix, target = weighted_system(a, jac, w, s, scales, .2)
        coef = np.linspace(0, .05, a.shape[1])
        ss, es = scales["strain_scale"], scales["energy_scale"]
        loss = (.2*np.mean((a@coef-w/es)**2)/scales["energy_denominator"]
                + np.mean((jac@coef*ss-s*ss/es)**2)/scales["stress_denominator"])
        self.assertAlmostEqual(float(np.linalg.norm(matrix@coef-target)**2), float(loss), places=12)

    def test_overflow_is_deterministic_and_never_exceeds_count(self):
        # Six feature columns plus two analytic columns; all feature supports active.
        matrix = np.eye(8)
        setting = dict(self.setting, count=3, qr_rows=8)
        coef = np.array([1., 4., 4., 2., 3., 2., 0., 0.])
        selected, support, _, _, _ = select_indices(matrix, np.ones(8), matrix,
            np.ones(8), coef, np.zeros(8), setting)
        np.testing.assert_array_equal(selected, [1, 2, 4])
        self.assertEqual(len(support), 6)

    def test_qr_fills_missing_support_without_volume_columns(self):
        rng = np.random.default_rng(7)
        matrix = rng.normal(size=(20, 8))
        coef = np.zeros(8)
        coef[4] = 1.
        setting = dict(self.setting, count=4, qr_rows=12)
        selected, _, _, pivot, _ = select_indices(matrix, np.ones(20), matrix,
            np.ones(20), coef, np.zeros(8), setting)
        self.assertEqual(selected[0], 4)
        self.assertEqual(len(set(selected)), 4)
        self.assertTrue(np.all(selected < 6))
        self.assertEqual(len(pivot), 6)

    def test_kinematics_rejects_negative_definite_c(self):
        with self.assertRaises(ValueError):
            kinematics(np.array([[-1., -1., 0.]]))

    def test_degenerate_scales_are_rejected(self):
        with self.assertRaises(ValueError):
            fit_scales(np.zeros((2, 3)), np.ones((2, 3)), np.ones(2), np.eye(3), 1e-8)


try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "Use the existing torch environment for neural setup tests")
class InitialModelTests(unittest.TestCase):
    def setUp(self):
        from protocol.training_setup import build_initial_model
        self.build = build_initial_model
        self.recipe = json.loads(RECIPE.read_text())
        self.specs = candidate_bank(self.recipe["feature_selection"])[::60][:32]
        self.e = np.array([[.1, .03, .04], [.03, .15, -.05], [-.02, .01, .02]])
        self.s = np.array([[3., 2., 1.], [4., 1., -2.], [1., 1., 1.]])*1e6
        self.w = np.array([4., 2., 1.])*1e5
        self.d0 = np.eye(3)*1e9
        scales = fit_scales(self.e, self.s, self.w, self.d0, 1e-8)
        z0 = (1/self.specs[:, 1:3]).sum(axis=1)
        self.table = dict(specs=self.specs, feature_center=z0,
            feature_scale=np.maximum(abs(raw_features(self.e, self.specs)-z0).max(axis=0), 1e-4),
            initialization_coefficients=np.r_[np.full(32, .01), .02, .05])
        self.manifest = dict(scales=scales)

    def test_free_buffers_retain_exact_saved_float64_scales(self):
        model, _ = self.build("Free", 16, self.recipe, self.manifest, self.table, self.e, self.s)
        self.assertEqual(model.strain_scale.dtype, torch.float64)
        self.assertEqual(model.strain_scale.item(), self.manifest["scales"]["strain_scale"])
        np.testing.assert_array_equal(model.feature_scale.numpy(), self.manifest["scales"]["free_feature_scale"])

    def test_both_cores_have_equal_fixed_learned_initial_responses(self):
        from protocol.training_setup import normalized_reference_tangent, normalized_response
        for core in ("ICNN", "ICKAN"):
            fixed, fm = self.build(core+"-fixed", 16, self.recipe, self.manifest, self.table, self.e, self.s)
            learned, lm = self.build(core+"-learned", 16, self.recipe, self.manifest, self.table, self.e, self.s)
            self.assertAlmostEqual(fm["calibration_factor"], lm["calibration_factor"], places=12)
            wf, sf = normalized_response(fixed, self.e, self.manifest["scales"], create_graph=False)
            wl, sl = normalized_response(learned, self.e, self.manifest["scales"], create_graph=False)
            torch.testing.assert_close(wf, wl, rtol=2e-12, atol=2e-13)
            torch.testing.assert_close(sf, sl, rtol=2e-12, atol=2e-13)
            torch.testing.assert_close(normalized_reference_tangent(fixed, create_graph=False),
                normalized_reference_tangent(learned, create_graph=False), rtol=2e-12, atol=2e-13)

    def test_energy_gradient_matches_analytic_stress_with_learned_features(self):
        for core in ("ICNN", "ICKAN"):
            model, _ = self.build(core+"-learned", 16, self.recipe, self.manifest, self.table, self.e, self.s)
            x = torch.tensor(self.e/self.manifest["scales"]["strain_scale"], dtype=torch.float64, requires_grad=True)
            energy = model.energy(x)
            derivative = torch.autograd.grad(energy.sum(), x)[0]
            _, analytic = model.energy_and_stress(x, create_graph=False)
            torch.testing.assert_close(derivative, analytic, rtol=2e-12, atol=2e-13)

    def test_anchor_loss_backpropagates_to_learned_feature_parameters(self):
        from protocol.training_setup import training_objective
        for core in ("ICNN", "ICKAN"):
            model, _ = self.build(core+"-learned", 16, self.recipe, self.manifest, self.table, self.e, self.s)
            _, parts = training_objective(model, self.e, self.s, self.w, self.manifest["scales"], self.d0, self.recipe)
            parts["reference_tangent"].backward()
            for param in (model.angles, model.raw_powers, model.raw_ratios):
                self.assertIsNotNone(param.grad)
                self.assertTrue(torch.isfinite(param.grad).all())
                self.assertGreater(float(torch.linalg.vector_norm(param.grad)), 0)


if __name__ == "__main__":
    unittest.main()
