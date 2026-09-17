"""Prospective stopping-policy checks; reserved prediction labels stay closed."""
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
class TrainingAmendmentTests(unittest.TestCase):
    def setUp(self):
        from protocol import train_material_b_v2 as v2
        self.v2 = v2
        self.recipe, self.amendment, _, _ = v2.load_recipe(v2.base.FEATURES)

    @staticmethod
    def history(adam_scores=(), lbfgs_scores=()):
        events = [dict(phase="initialization", index=0, score=1.)]
        events += [dict(phase="adam", index=i*10, score=score)
                   for i, score in enumerate(adam_scores, 1)]
        events += [dict(phase="lbfgs", index=i, score=score)
                   for i, score in enumerate(lbfgs_scores, 1)]
        return events

    def test_amendment_retains_frozen_scientific_recipe_and_scheduler(self):
        from protocol.training_setup import load_preparation
        parent, _, _ = load_preparation(self.v2.base.FEATURES, self.v2.RECIPE)
        modified = copy.deepcopy(self.recipe)
        modified["adam"].pop("maximum_steps")
        modified["adam"].pop("early_stopping")
        modified["lbfgs"].pop("outer_calls")
        modified["lbfgs"].pop("maximum_iterations_total")
        frozen = copy.deepcopy(parent)
        frozen["adam"].pop("maximum_steps")
        frozen["adam"].pop("early_stopping")
        frozen["lbfgs"].pop("outer_calls")
        frozen["lbfgs"].pop("maximum_iterations_total")
        self.assertEqual(modified, frozen)
        self.assertEqual(self.recipe["adam"]["scheduler"], parent["adam"]["scheduler"])
        self.assertEqual(self.recipe["adam"]["maximum_steps"], 7800)
        self.assertEqual(self.recipe["lbfgs"]["outer_calls"], 120)

    def test_wrong_parent_hash_cannot_be_used(self):
        changed = copy.deepcopy(self.amendment)
        changed["parent_recipe_sha256"] = "0"*64
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/"amendment.json"
            path.write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "immutable v1 recipe"):
                self.v2.load_recipe(self.v2.base.FEATURES, path)

    def test_small_gains_accumulate_before_patience_resets(self):
        history = self.history(adam_scores=[.9996, .9992, .9988])
        index, anchor = self.v2.last_material_gain(history, "adam", 1., .001)
        self.assertEqual(index, 30)
        self.assertEqual(anchor, .9988)
        zero_history = [dict(phase="adam", index=10, score=0.)]
        self.assertEqual(self.v2.last_material_gain(zero_history, "adam", 0., .001),
                         (0, 0.))

    def test_adam_needs_minimum_steps_patience_and_minimum_lr(self):
        model = torch.nn.Linear(1, 1, dtype=torch.float64)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
        state = dict(adam_step=2590, validation_history=self.history())
        self.assertIsNone(self.v2.adam_stop_reason(
            state, optimizer, self.recipe, self.amendment))
        state["adam_step"] = 2600
        self.assertEqual(self.v2.adam_stop_reason(
            state, optimizer, self.recipe, self.amendment), "validation_plateau")
        optimizer.param_groups[0]["lr"] = 2e-5
        self.assertIsNone(self.v2.adam_stop_reason(
            state, optimizer, self.recipe, self.amendment))
        state["adam_step"] = 7800
        self.assertEqual(self.v2.adam_stop_reason(
            state, optimizer, self.recipe, self.amendment), "safety_cap")

    def test_lbfgs_plateau_and_cap_do_not_mean_same_thing(self):
        state = dict(lbfgs_call=39, validation_history=self.history())
        self.assertIsNone(self.v2.lbfgs_stop_reason(state, self.amendment))
        state["lbfgs_call"] = 40
        self.assertEqual(self.v2.lbfgs_stop_reason(state, self.amendment),
                         "validation_plateau")
        state["validation_history"] = self.history(lbfgs_scores=[1.]*38+[.99])
        self.assertIsNone(self.v2.lbfgs_stop_reason(state, self.amendment))
        state["lbfgs_call"] = 120
        state["validation_history"] = self.history(lbfgs_scores=[.99]*119)
        state["validation_history"].append(dict(phase="lbfgs", index=120, score=.98))
        self.assertEqual(self.v2.lbfgs_stop_reason(state, self.amendment), "safety_cap")

    def test_adam_plateau_state_reconstructed_after_resume(self):
        base = self.v2.base
        model = torch.nn.Linear(1, 1, dtype=torch.float64)
        adam = torch.optim.Adam(model.parameters(), lr=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(adam)
        history = self.history(adam_scores=[.9996, .9992, .9988])
        state = dict(identity={"model": "synthetic_v2"}, phase="adam", adam_step=2600,
                     lbfgs_call=0, best_model_state=base._copy_state(model),
                     best_score=.9988, best_origin=dict(phase="adam", index=30),
                     validation_history=history, elapsed_seconds=1.,
                     calibration_factor=None, configuration={})
        before = self.v2.adam_stop_reason(state, adam, self.recipe, self.amendment)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            base._save_progress(output, state, model, adam, scheduler, None)
            restored, _ = base._load_progress(output, state["identity"], model,
                                              adam, scheduler, self.recipe)
            after = self.v2.adam_stop_reason(restored, adam, self.recipe, self.amendment)
        self.assertEqual(before, after)
        self.assertEqual(before, "validation_plateau")

    def test_final_report_distinguishes_both_stopping_reasons(self):
        base = self.v2.base
        model = torch.nn.Linear(1, 1, dtype=torch.float64)
        adam = torch.optim.Adam(model.parameters(), lr=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(adam)
        lbfgs = base._lbfgs_optimizer(model, self.recipe)
        state = dict(identity={"model": "synthetic_v2", "seed": 16},
                     phase="lbfgs", adam_step=2600, lbfgs_call=40,
                     best_model_state=base._copy_state(model), best_score=1.,
                     best_origin=dict(phase="initialization", index=0),
                     validation_history=self.history(), elapsed_seconds=1.,
                     calibration_factor=None, configuration={})
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            report = self.v2._finish(output, state, model, adam, scheduler, lbfgs,
                dict(strain_scale=1., energy_scale=1.), self.recipe, self.amendment,
                self.v2.AMENDMENT)
            recorded = json.loads((output/"run_report.json").read_text())
            checkpoint = torch.load(output/"run_state.pt", map_location="cpu",
                                    weights_only=False)
        self.assertEqual(report["adam_stop_reason"], "validation_plateau")
        self.assertEqual(recorded["lbfgs_stop_reason"], "validation_plateau")
        self.assertEqual(checkpoint["phase"], "complete")


if __name__ == "__main__":
    unittest.main()
