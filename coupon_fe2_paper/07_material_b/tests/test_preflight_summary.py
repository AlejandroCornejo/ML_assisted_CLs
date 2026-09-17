"""Synthetic record-integrity tests, not physical FOM validation."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from summarize_preflight import differences, summarize


class SummaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.reference = self.root / "reference"
        self.check = self.root / "check"
        self.reference.mkdir()
        self.check.mkdir()
        (self.root / "report.json").write_text("{}")
        self.spec = dict(domain_states={"corner": [0.1, 0.1, 0.08]},
                         reference_output_tolerances=dict(stress=0.001, tangent=0.001, energy=0.001),
                         derivative_states=["corner"], fd_steps=[0.00005])
        fields = dict(pk1_l2=10., pk1_max=20., min_micro_J=0.8, relative_reduced_residual=1e-12)
        self.row = dict(ok=True, strain=[0.01, 0., 0.], stress=[10., 2., 1.],
                        tangent=[[10., 0., 0.], [0., 10., 0.], [0., 0., 5.]], energy=5.,
                        fields=fields, attempts=[], sampled_min_rank_one_curvature=1.,
                        boundary=dict(min_polygon_gap=0.2, self_intersection=False,
                                      max_periodic_jump_error=1e-15))
        self.original = {(f"path{i}", 1.):copy.deepcopy(self.row) for i in range(32)}
        domain = dict(copy.deepcopy(self.row), name="domain__corner", strain=self.spec["domain_states"]["corner"],
                      derivatives=[dict(energy_gradient_relative_error=1e-8,
                                        tangent_relative_error=1e-8, fd_tangent_relative_asymmetry=1e-8)])
        common = dict(status="complete", numerical_states_complete=True,
                      spec_sha256="fixed", spec=self.spec)
        self.ref_report = dict(common, states=[domain], geometry=dict(n_elements=10))
        rows = [dict(copy.deepcopy(row), name=f"pilot__{path}__1")
                for (path, _fraction), row in self.original.items()]
        self.check_report = dict(common, states=rows+[copy.deepcopy(domain)], geometry=dict(n_elements=20))
        tolerances = dict(mesh_pk1_l2_relative_difference=0.02, mesh_pk1_max_relative_difference=0.05,
                          energy_gradient_relative_error=1e-4, tangent_relative_error=1e-4,
                          fd_tangent_relative_asymmetry=1e-4)
        self.parent = dict(parent_spec=dict(screening_tolerances=tolerances))

    def result(self):
        (self.reference / "report.json").write_text(json.dumps(self.ref_report))
        (self.check / "report.json").write_text(json.dumps(self.check_report))
        with patch("summarize_preflight.old_states", return_value=(self.original, self.parent, self.root, self.root)):
            return summarize(self.reference, self.check)

    def test_complete_records_pass(self):
        self.assertTrue(self.result()["passed"])

    def test_missing_target_cannot_pass(self):
        self.check_report["states"].pop(0)
        self.assertFalse(self.result()["passed"])

    def test_duplicate_cannot_replace_target(self):
        self.check_report["states"][0] = copy.deepcopy(self.check_report["states"][1])
        self.assertFalse(self.result()["passed"])

    def test_wrong_strain_cannot_pass(self):
        self.check_report["states"][0]["strain"] = [0.02, 0., 0.]
        self.assertFalse(self.result()["passed"])

    def test_running_stage_is_rejected(self):
        self.ref_report["status"] = "running"
        with self.assertRaises(RuntimeError):
            self.result()

    def test_denominator_is_second_argument(self):
        older, denser = copy.deepcopy(self.row), copy.deepcopy(self.row)
        older["stress"], denser["stress"] = [10.], [12.]
        self.assertAlmostEqual(differences(older, denser)["stress"], 2./12.)


if __name__ == "__main__":
    unittest.main()
