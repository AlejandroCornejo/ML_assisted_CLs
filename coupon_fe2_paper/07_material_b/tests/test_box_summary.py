"""Synthetic integrity/gate tests; these do not validate the physical RVE."""
import copy
import unittest
from summarize_box_extension import assess, differences


class BoxSummaryTests(unittest.TestCase):
    def setUp(self):
        self.spec = dict(states={"corner":[0.2, 0.2, 0.16], "reuse":[-0.04, -0.04, 0.]},
            reuse_states=["reuse"], derivative_states=["corner"], fd_steps=[1e-4, 5e-5],
            cold_check_states=["corner"], derivative_tolerance=1e-4, cold_agreement_tolerance=1e-7,
            reference_output_tolerances=dict(stress=.001, tangent=.001, energy=.001),
            field_statistic_tolerances=dict(pk1_l2=.02, pk1_max=.05),
            relative_residual_tolerance=1e-7, stop_polygon_gap=1e-6,
            periodic_jump_tolerance=1e-8, minimum_sampled_rank_one_curvature=0.)
        base = dict(ok=True, stress=[10., 2., 1.], tangent=[[10., 0., 0.], [0., 10., 0.], [0., 0., 5.]],
            energy=5., fields=dict(pk1_l2=10., pk1_max=20., min_micro_J=.8, relative_reduced_residual=1e-12),
            boundary=dict(min_polygon_gap=.2, self_intersection=False, max_periodic_jump_error=1e-15),
            physical_screen_passed=True, sampled_min_rank_one_curvature=1., attempts=[])
        corner = dict(copy.deepcopy(base), name="corner", strain=self.spec["states"]["corner"], origin="new",
            derivatives=[dict(step=h, energy_gradient_relative_error=1e-8, tangent_relative_error=1e-8,
                              fd_tangent_relative_asymmetry=1e-8) for h in self.spec["fd_steps"]])
        reused = dict(copy.deepcopy(base), name="reuse", strain=self.spec["states"]["reuse"], origin="reused")
        cold = dict(copy.deepcopy(base), name="corner", strain=self.spec["states"]["corner"],
                    differences=dict(stress=1e-12, tangent=1e-12, energy=1e-12, node_displacement=1e-12))
        common = dict(status="complete", all_targets_reached=True, spec=self.spec,
            spec_sha256="fixed", helper_sha256="fixed_helper", failures=[], screen_failures=[],
            states=[corner, reused], cold_checks=[cold])
        self.reference = dict(copy.deepcopy(common), mesh_role="reference", n_elements=10)
        self.check = dict(copy.deepcopy(common), mesh_role="check", n_elements=20)

    def result(self):
        return assess(self.reference, self.check)

    def test_complete_pass(self):
        result = self.result()
        self.assertTrue(result["passed"])
        self.assertEqual(result["comparison_count"], 2)
        self.assertEqual(result["derivative_count"], 4)
        self.assertEqual(result["new_target_count"], 2)
        self.assertEqual(result["reused_target_count"], 2)

    def test_running_is_rejected(self):
        self.reference["status"] = "running"
        with self.assertRaises(RuntimeError):
            self.result()

    def test_spec_change_is_rejected(self):
        self.check["spec"]["derivative_tolerance"] = .5
        with self.assertRaises(ValueError):
            self.result()

    def test_missing_or_duplicate_target_fails(self):
        self.check["states"][1] = copy.deepcopy(self.check["states"][0])
        self.assertFalse(self.result()["passed"])

    def test_wrong_strain_or_origin_fails(self):
        self.check["states"][0]["strain"] = [.1, .1, .08]
        self.assertFalse(self.result()["passed"])
        self.check["states"][0]["strain"] = self.spec["states"]["corner"]
        self.check["states"][1]["origin"] = "new"
        self.assertFalse(self.result()["passed"])

    def test_duplicate_fd_step_cannot_replace_missing_step(self):
        self.check["states"][0]["derivatives"][1]["step"] = 1e-4
        self.assertFalse(self.result()["passed"])

    def test_missing_cold_check_fails(self):
        self.check["cold_checks"] = []
        self.assertFalse(self.result()["passed"])

    def test_negative_curvature_and_failed_cold_agreement_fail(self):
        self.check["states"][0]["sampled_min_rank_one_curvature"] = -1.
        self.assertFalse(self.result()["passed"])
        self.check["states"][0]["sampled_min_rank_one_curvature"] = 1.
        self.check["cold_checks"][0]["differences"]["node_displacement"] = 1e-5
        self.assertFalse(self.result()["passed"])

    def test_mesh_threshold_is_not_relaxed(self):
        self.check["states"][0]["fields"]["pk1_max"] = 22.
        result = self.result()
        gate = next(c for c in result["checks"] if c["check"] == "reference/check: pk1_max")
        self.assertEqual(gate["threshold"], .05)
        self.assertFalse(gate["passed"])

    def test_denominator_is_denser_check(self):
        older, denser = self.reference["states"][0], self.check["states"][0]
        older["stress"], denser["stress"] = [10.], [12.]
        self.assertAlmostEqual(differences(older, denser)["stress"], 2./12.)


if __name__ == "__main__":
    unittest.main()
