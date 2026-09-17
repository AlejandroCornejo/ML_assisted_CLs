import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from protocol.assemble_campaign import load_group, relative
from protocol.prepare_design import deterministic_npz, digest
from protocol.run_campaign import jobs


class CampaignAssemblyTests(unittest.TestCase):
    def test_relative_uses_second_argument_and_floor(self):
        self.assertAlmostEqual(relative([2.0, 0.0], [1.0, 0.0]), 1.0)
        self.assertAlmostEqual(relative([0.2], [0.0]), 0.2)

    def test_relative_accepts_tensors(self):
        first = np.eye(3)*2
        second = np.eye(3)
        self.assertAlmostEqual(relative(first, second), 1.0)

    def test_campaign_plan_is_unique_and_has_declared_size(self):
        plan = jobs()
        self.assertEqual(len(plan), 103)
        self.assertEqual(len({row["name"] for row in plan}), len(plan))
        self.assertEqual(sum(row["split"] == "fit" for row in plan), 64)
        self.assertEqual(sum(row["split"] == "paths" for row in plan), 10)
        self.assertEqual(sum(row["mesh"] == "audit" for row in plan), 4)


class ChunkAssemblyIntegrityTests(unittest.TestCase):
    """Exercise integrity rejection without solving or reading scientific test labels."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="material_b_assembly_test_")
        self.campaign = Path(self.temporary.name)
        self.expected = np.array([[0.01, 0.02, 0.03], [0.02, 0.03, 0.04],
                                  [0.03, 0.04, 0.05]])
        for chunk, indices in enumerate(([2, 0], [1])):
            self.write_chunk(chunk, indices)

    def tearDown(self):
        self.temporary.cleanup()

    def write_chunk(self, chunk, indices):
        folder = self.campaign / "chunks" / f"synthetic_{chunk:03d}"
        folder.mkdir(parents=True, exist_ok=True)
        labels = folder / "labels.npz"
        if labels.exists():
            labels.unlink()
        count = len(indices)
        arrays = dict(strain=self.expected[indices], stress=np.ones((count, 3)),
            tangent=np.tile(np.eye(3), (count, 1, 1)), energy=np.ones(count),
            minimum_micro_J=np.full(count, 0.9), relative_reduced_residual=np.zeros(count),
            minimum_deformed_polygon_gap=np.full(count, 0.2),
            deformed_polygon_self_intersection=np.zeros(count, dtype=bool),
            max_periodic_jump_error=np.zeros(count), pk1_l2=np.ones(count),
            pk1_max=np.ones(count), available=np.ones(count, dtype=bool),
            global_index=np.array(indices), warm_audit_index=np.array([], dtype=int),
            warm_audit_q=np.empty((0, 0)))
        deterministic_npz(labels, arrays)
        report = dict(status="complete", all_available=True, labels_sha256=digest(labels),
                      sources_sha256={str(Path(__file__)): digest(Path(__file__))})
        (folder / "report.json").write_text(json.dumps(report))

    def change_report(self, chunk, **changes):
        path = self.campaign / "chunks" / f"synthetic_{chunk:03d}" / "report.json"
        report = json.loads(path.read_text())
        report.update(changes)
        path.write_text(json.dumps(report))

    def test_merge_restores_global_order_and_preserves_physical_fields(self):
        merged, provenance = load_group(self.campaign, "synthetic", 2, self.expected)
        np.testing.assert_array_equal(merged["strain"], self.expected)
        np.testing.assert_array_equal(merged["global_index"], np.arange(3))
        np.testing.assert_array_equal(merged["minimum_deformed_polygon_gap"], [.2, .2, .2])
        self.assertEqual(len(provenance["labels_sha256"]), 2)

    def test_unfinished_or_unavailable_chunk_is_rejected(self):
        self.change_report(0, status="paused")
        with self.assertRaises(RuntimeError):
            load_group(self.campaign, "synthetic", 2, self.expected)
        self.change_report(0, status="complete", all_available=False)
        with self.assertRaises(RuntimeError):
            load_group(self.campaign, "synthetic", 2, self.expected)

    def test_modified_label_hash_is_rejected(self):
        self.change_report(0, labels_sha256="0"*64)
        with self.assertRaises(ValueError):
            load_group(self.campaign, "synthetic", 2, self.expected)

    def test_modified_source_hash_is_rejected(self):
        self.change_report(0, sources_sha256={str(Path(__file__)): "0"*64})
        with self.assertRaises(ValueError):
            load_group(self.campaign, "synthetic", 2, self.expected)

    def test_duplicate_index_cannot_replace_missing_target(self):
        self.write_chunk(1, [0])
        with self.assertRaises(ValueError):
            load_group(self.campaign, "synthetic", 2, self.expected)

    def test_wrong_requested_strain_is_rejected(self):
        expected = self.expected.copy()
        expected[1, 0] += 0.001
        with self.assertRaises(ValueError):
            load_group(self.campaign, "synthetic", 2, expected)


if __name__ == "__main__":
    unittest.main()
