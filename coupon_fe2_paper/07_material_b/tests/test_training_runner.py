"""Synthetic runner checks; no scientific test/path targets are accessed."""
from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "Use the existing torch environment for runner tests")
class RunnerTests(unittest.TestCase):
    def test_label_loader_requests_only_fit_reference_and_validation(self):
        from protocol import train_material_b as runner
        allowed = ("E_fit", "S_fit", "W_fit", "E_reference", "S_reference",
                   "W_reference", "D_reference", "E_validation", "S_validation")
        arrays = dict(E_fit=np.zeros((4200, 3)), S_fit=np.zeros((4200, 3)),
                      W_fit=np.zeros(4200), E_reference=np.zeros((1, 3)),
                      S_reference=np.zeros((1, 3)), W_reference=np.zeros(1),
                      D_reference=np.zeros((1, 3, 3)),
                      E_validation=np.zeros((512, 3)), S_validation=np.zeros((512, 3)))
        calls = []

        class GuardedStore:
            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

            def __getitem__(self, key):
                calls.append(key)
                if key not in allowed:
                    raise AssertionError(f"Reserved array accessed: {key}")
                return arrays[key]

        recipe = dict(labels_sha256="approved", selection_allowed_arrays=list(allowed[:7]))
        with mock.patch.object(runner, "digest", return_value="approved"), \
             mock.patch.object(runner.np, "load", return_value=GuardedStore()):
            loaded = runner.load_training_arrays(Path("synthetic.npz"), recipe)
        self.assertEqual(calls, list(allowed))
        self.assertEqual(set(loaded), set(allowed))

    def test_adam_checkpoint_restores_model_optimizer_scheduler_and_rng(self):
        from protocol import train_material_b as runner
        torch.manual_seed(16)
        model = torch.nn.Linear(1, 1, dtype=torch.float64)
        adam = torch.optim.Adam(model.parameters(), lr=.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(adam, patience=2)
        loss = model(torch.ones((2, 1), dtype=torch.float64)).square().mean()
        loss.backward()
        adam.step()
        scheduler.step(float(loss.detach()))
        saved = runner._copy_state(model)
        state = dict(identity={"model": "synthetic"}, phase="adam", adam_step=1,
                     lbfgs_call=0, best_model_state=saved, best_score=.4,
                     best_origin=dict(phase="initialization", index=0),
                     validation_history=[], elapsed_seconds=1.,
                     calibration_factor=None, configuration={})
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            runner._save_progress(output, state, model, adam, scheduler, None)
            saved_adam = copy.deepcopy(adam.state_dict())
            saved_scheduler = copy.deepcopy(scheduler.state_dict())
            saved_rng = torch.get_rng_state().clone()
            with torch.no_grad():
                model.weight.add_(12.)
            torch.manual_seed(99)
            loaded, lbfgs = runner._load_progress(output, state["identity"], model,
                                                  adam, scheduler, {})
            self.assertIsNone(lbfgs)
            self.assertEqual(loaded["adam_step"], 1)
            self.assertTrue(torch.equal(model.weight, saved["weight"]))
            self.assertEqual(scheduler.state_dict(), saved_scheduler)
            self.assertTrue(torch.equal(torch.get_rng_state(), saved_rng))
            for key, value in saved_adam["state"][0].items():
                actual = adam.state_dict()["state"][0][key]
                if isinstance(value, torch.Tensor):
                    self.assertTrue(torch.equal(actual, value))
                else:
                    self.assertEqual(actual, value)

    def test_lbfgs_optimizer_history_survives_checkpoint(self):
        from protocol import train_material_b as runner
        recipe = dict(lbfgs=dict(learning_rate=.2, maximum_iterations_per_call=1,
                     maximum_evaluations_per_call=3, history_size=5,
                     tolerance_grad=1e-12, tolerance_change=1e-14,
                     line_search="strong_wolfe"))
        torch.manual_seed(16)
        model = torch.nn.Linear(1, 1, dtype=torch.float64)
        adam = torch.optim.Adam(model.parameters(), lr=.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(adam)
        lbfgs = runner._lbfgs_optimizer(model, recipe)

        def one_call(this_model, optimizer):
            def closure():
                optimizer.zero_grad()
                loss = (this_model(torch.ones((1, 1), dtype=torch.float64))-3).square().sum()
                loss.backward()
                return loss
            optimizer.step(closure)

        one_call(model, lbfgs)
        state = dict(identity={"model": "synthetic"}, phase="lbfgs", adam_step=2,
                     lbfgs_call=1, best_model_state=runner._copy_state(model),
                     best_score=.4, best_origin=dict(phase="adam", index=2),
                     validation_history=[], elapsed_seconds=1.,
                     calibration_factor=None, configuration={})
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            runner._save_progress(output, state, model, adam, scheduler, lbfgs)
            restored = torch.nn.Linear(1, 1, dtype=torch.float64)
            restored_adam = torch.optim.Adam(restored.parameters(), lr=.01)
            restored_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(restored_adam)
            loaded, restored_lbfgs = runner._load_progress(output, state["identity"],
                restored, restored_adam, restored_scheduler, recipe)
            self.assertEqual(loaded["lbfgs_call"], 1)
            self.assertEqual(lbfgs.state_dict()["state"].keys(),
                             restored_lbfgs.state_dict()["state"].keys())
            one_call(model, lbfgs)
            one_call(restored, restored_lbfgs)
            for key, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, restored.state_dict()[key]))


if __name__ == "__main__":
    unittest.main()
