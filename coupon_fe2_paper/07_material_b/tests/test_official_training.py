"""Official stopping and review-pause tests; no reserved labels are read."""
from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "Use the existing torch environment")
class OfficialTrainingTests(unittest.TestCase):
    def setUp(self):
        from protocol import train_material_b_official as official
        self.official = official
        self.recipe, self.rule, _, _ = official.load_rule(official.base.FEATURES)

    @staticmethod
    def history(adam_scores=(), lbfgs_scores=()):
        result = [dict(phase="initialization", index=0, score=1.)]
        result += [dict(phase="adam", index=i*10, score=value)
                   for i, value in enumerate(adam_scores, 1)]
        result += [dict(phase="lbfgs", index=i, score=value)
                   for i, value in enumerate(lbfgs_scores, 1)]
        return result

    def test_only_scheduler_and_stop_budgets_change_preparation_recipe(self):
        from protocol.training_setup import load_preparation
        parent, _, _ = load_preparation(self.official.base.FEATURES,
                                        self.official.RECIPE)
        new = copy.deepcopy(self.recipe)
        old = copy.deepcopy(parent)
        for recipe in (new, old):
            recipe["adam"].pop("scheduler")
            recipe["adam"].pop("maximum_steps")
            recipe["adam"].pop("early_stopping")
            recipe["lbfgs"].pop("outer_calls")
            recipe["lbfgs"].pop("maximum_iterations_total")
        self.assertEqual(new, old)
        self.assertEqual(self.recipe["adam"]["scheduler"]["threshold"], .001)

    def test_wrong_parent_recipe_hash_is_rejected(self):
        wrong = copy.deepcopy(self.rule)
        wrong["parent_preparation_sha256"] = "0"*64
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/"rule.json"
            path.write_text(json.dumps(wrong), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "frozen feature preparation"):
                self.official.load_rule(self.official.base.FEATURES, path)

    def test_adam_requires_patience_two_lr_drops_and_time_after_last_drop(self):
        state = dict(adam_step=20000, scheduler_reductions=1,
                     last_reduction_step=10000, validation_history=self.history())
        self.assertFalse(self.official.adam_plateau(state, self.rule))
        state["scheduler_reductions"] = 2
        state["last_reduction_step"] = 19800
        self.assertFalse(self.official.adam_plateau(state, self.rule))
        state["last_reduction_step"] = 19600
        self.assertTrue(self.official.adam_plateau(state, self.rule))
        state["adam_step"] = 2590
        self.assertFalse(self.official.adam_plateau(state, self.rule))

    def test_small_gains_accumulate_and_reset_patience(self):
        history = self.history(adam_scores=[.9996, .9992, .9988])
        self.assertEqual(self.official.last_material_gain(history, "adam", 1., .001),
                         (30, .9988))
        self.assertEqual(self.official.last_material_gain(
            [dict(phase="adam", index=10, score=0.)], "adam", 0., .001), (0, 0.))

    def test_lbfgs_plateau_requires_minimum_work_and_patience(self):
        state = dict(lbfgs_call=39, validation_history=self.history())
        self.assertFalse(self.official.lbfgs_plateau(state, self.rule))
        state["lbfgs_call"] = 40
        self.assertTrue(self.official.lbfgs_plateau(state, self.rule))
        state["validation_history"] = self.history(lbfgs_scores=[1.]*39+[.99])
        self.assertFalse(self.official.lbfgs_plateau(state, self.rule))

    def test_review_pause_preserves_exact_optimizer_state_and_is_extendable(self):
        runner = self.official
        model = torch.nn.Linear(1, 1, dtype=torch.float64)
        adam = torch.optim.Adam(model.parameters(), lr=.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(adam)
        state = dict(identity={"model": "synthetic"}, phase="adam",
            review_required=True, review_events=[], adam_limit=20000,
            lbfgs_limit=300, scheduler_reductions=0, last_reduction_step=0,
            adam_step=20000, lbfgs_call=0, best_model_state=runner.base._copy_state(model),
            best_score=1., best_origin=dict(phase="initialization", index=0),
            validation_history=self.history(), elapsed_seconds=1.,
            calibration_factor=None, configuration={})
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            runner._save_state(output, state, model, adam, scheduler, None)
            saved = torch.load(output/"run_state.pt", weights_only=False)
            loaded, lbfgs = runner._load_state(output, state["identity"], model,
                                               adam, scheduler, self.recipe)
            self.assertIsNone(lbfgs)
            self.assertTrue(loaded["review_required"])
            self.assertEqual(saved["phase"], "adam")
            self.assertFalse((output/"model.pt").exists())
            runner._authorize_extension(loaded, self.rule)
            self.assertEqual(loaded["adam_limit"], 25000)
            self.assertFalse(loaded["review_required"])
            self.assertEqual(len(loaded["review_events"]), 1)

    def test_finalization_refuses_a_budget_cutoff(self):
        runner = self.official
        model = torch.nn.Linear(1, 1, dtype=torch.float64)
        adam = torch.optim.Adam(model.parameters(), lr=.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(adam)
        lbfgs = runner.base._lbfgs_optimizer(model, self.recipe)
        state = dict(identity={"model": "synthetic", "seed": 16,
                               "recipe_sha256": runner.digest(runner.RULE)},
            phase="lbfgs", review_required=False, review_events=[],
            adam_limit=20000, lbfgs_limit=300, scheduler_reductions=0,
            last_reduction_step=0, adam_step=20000, lbfgs_call=300,
            best_model_state=runner.base._copy_state(model), best_score=1.,
            best_origin=dict(phase="initialization", index=0),
            validation_history=self.history(), elapsed_seconds=1.,
            calibration_factor=None, configuration={})
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            with self.assertRaisesRegex(RuntimeError, "Adam never met"):
                runner._finish(output, state, model, adam, scheduler, lbfgs,
                    dict(strain_scale=1., energy_scale=1.), self.rule, runner.RULE)
            self.assertFalse((output/"model.pt").exists())

    def test_campaign_marks_review_boundary_incomplete(self):
        from protocol.run_training_campaign import _sha, _verify_job
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            job = dict(model="Free", seed=16)
            torch.save(dict(format_version=3, review_required=True,
                phase="adam", adam_step=20000, lbfgs_call=0, best_score=.01,
                identity=dict(model="Free", seed=16, recipe_sha256=_sha(self.official.RULE))),
                output/"run_state.pt")
            status, details = _verify_job(output, job, _sha(self.official.RULE))
            self.assertEqual(status, "needs_review")
            self.assertEqual(details["adam_step"], 20000)
            self.assertFalse((output/"model.pt").exists())


if __name__ == "__main__":
    unittest.main()
