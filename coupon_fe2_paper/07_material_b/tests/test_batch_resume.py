"""Batch review approval must preserve trained state and leave an audit trail."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "Use the existing torch environment")
class BatchResumeTests(unittest.TestCase):
    def test_two_blocks_preserve_optimizer_model_scheduler_and_rng(self):
        from protocol import resume_training_campaign as batch

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            batch.base._atomic_json(output / "campaign_status.json",
                                    dict(status="needs_review"))
            job = dict(model="Free", seed=16, slug="free_seed16",
                       status="needs_review")
            folder = output / job["slug"]
            folder.mkdir()
            original = dict(format_version=3, phase="adam", review_required=True,
                adam_step=20000, adam_limit=20000, review_events=[],
                model_state={"weight": torch.tensor([1., 2.])},
                adam_state={"step": torch.tensor(20000)},
                scheduler_state={"best": 0.2},
                rng_state={"torch": torch.tensor([3, 4], dtype=torch.uint8)})
            batch.base._atomic_torch(folder / "run_state.pt", original)
            rule = dict(adam=dict(review_extension_steps=5000))
            review_dir, audit = batch.approve_batch(
                output, dict(status="needs_review", jobs=[job]), rule,
                [(job, original)], 20000, "validation still improving")
            saved = torch.load(folder / "run_state.pt", weights_only=False)
            backup = torch.load(review_dir / "free_seed16_before.pt", weights_only=False)
            self.assertEqual(saved["adam_limit"], 30000)
            self.assertFalse(saved["review_required"])
            self.assertEqual([(e["old_limit"], e["new_limit"])
                              for e in saved["review_events"]],
                             [(20000, 25000), (25000, 30000)])
            self.assertTrue(torch.equal(saved["model_state"]["weight"],
                                        original["model_state"]["weight"]))
            self.assertTrue(torch.equal(saved["adam_state"]["step"],
                                        original["adam_state"]["step"]))
            self.assertEqual(saved["scheduler_state"], original["scheduler_state"])
            self.assertTrue(torch.equal(saved["rng_state"]["torch"],
                                        original["rng_state"]["torch"]))
            self.assertTrue(backup["review_required"])
            self.assertEqual(len(audit["runs"]), 1)
            self.assertEqual(json.loads((review_dir / "approval.json").read_text())
                             ["new_limit"], 30000)

    def test_200000_is_an_approved_ceiling_not_a_completion_rule(self):
        from protocol import resume_training_campaign as batch

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            batch.base._atomic_json(output / "campaign_status.json",
                                    dict(status="needs_review"))
            job = dict(model="Free", seed=16, slug="free_seed16",
                       status="needs_review")
            folder = output / job["slug"]
            folder.mkdir()
            original = dict(format_version=3, phase="adam", review_required=True,
                            adam_step=100000, adam_limit=100000, review_events=[])
            batch.base._atomic_torch(folder / "run_state.pt", original)
            rule = dict(adam=dict(review_extension_steps=5000))
            with self.assertRaisesRegex(ValueError, "at most 200,000"):
                batch.approve_batch(output, {}, rule, [(job, original)], 100000,
                                    "reviewed", target_step=205000)
            review_dir, audit = batch.approve_batch(
                output, dict(jobs=[job]), rule, [(job, original)], 100000,
                "reviewed", target_step=200000)
            saved = torch.load(folder / "run_state.pt", weights_only=False)
            self.assertEqual(saved["adam_limit"], 200000)
            self.assertEqual(len(saved["review_events"]), 20)
            self.assertFalse(saved["review_required"])
            self.assertEqual(audit["blocks"], 20)
            self.assertFalse(audit["ceiling_is_completion_criterion"])
            self.assertTrue((review_dir / "free_seed16_before.pt").is_file())


if __name__ == "__main__":
    unittest.main()
