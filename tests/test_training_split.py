"""Verify split provenance against real CSV rows, without training a model."""

import csv
import hashlib
import json
import tempfile
import unittest
from copy import copy
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image
from torch.utils.data import Subset

import data
import train


class TrainingSplitTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.images = self.root / "images"
        self.images.mkdir()
        self.labels = self.root / "labels.csv"
        self.rows = [
            {"id": 100 - 3 * index, "filename": f"arbitrary_{index}.png", "label": index % 10}
            for index in range(20)
        ]
        for index, row in enumerate(self.rows):
            Image.new("RGB", (32, 32), (index, index, index)).save(self.images / row["filename"])
        with self.labels.open("w", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(file, fieldnames=("id", "filename", "label"))
            writer.writeheader()
            writer.writerows(self.rows)
        self.original_bytes = self.labels.read_bytes()
        self.training, self.validation = data.make_loaders(
            self.images, self.labels, 4, 0, 42, torch.device("cpu"),
        )

    def test_records_actual_csv_ids_class_counts_and_exact_hashes(self):
        metadata = train.record_split(self.root, self.training, self.validation, 42)

        manifest_path = self.root / "split.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        training_indices = list(self.training.dataset.indices)
        validation_indices = list(self.validation.dataset.indices)
        expected_training_ids = [self.rows[index]["id"] for index in training_indices]
        expected_validation_ids = [self.rows[index]["id"] for index in validation_indices]
        source_hash = hashlib.sha256(self.original_bytes).hexdigest()
        self.assertEqual(manifest["seed"], 42)
        self.assertEqual(manifest["train_ids"], expected_training_ids)
        self.assertEqual(manifest["val_ids"], expected_validation_ids)
        self.assertTrue(set(manifest["train_ids"]).isdisjoint(manifest["val_ids"]))
        self.assertEqual(set(manifest["train_ids"] + manifest["val_ids"]), {row["id"] for row in self.rows})
        self.assertEqual(manifest["labels_sha256"], source_hash)
        self.assertEqual(metadata["labels_sha256"], source_hash)
        self.assertEqual(manifest["samples_sha256"], self.training.dataset.dataset.samples_sha256)
        self.assertEqual(metadata["samples_sha256"], manifest["samples_sha256"])
        self.assertEqual(Path(metadata["split_manifest"]), manifest_path.resolve())
        self.assertEqual(metadata["split_sha256"], hashlib.sha256(manifest_path.read_bytes()).hexdigest())
        self.assertEqual(metadata["train_class_counts"], [
            sum(self.rows[index]["label"] == label for index in training_indices) for label in range(10)
        ])
        self.assertEqual(metadata["val_class_counts"], [
            sum(self.rows[index]["label"] == label for index in validation_indices) for label in range(10)
        ])
        self.assertEqual(self.labels.read_bytes(), self.original_bytes)

    def test_rejects_overlapping_train_validation_ids(self):
        overlapping_validation = SimpleNamespace(dataset=Subset(
            self.validation.dataset.dataset, [self.training.dataset.indices[0]],
        ))

        with self.assertRaisesRegex(ValueError, "overlap"):
            train.record_split(self.root, self.training, overlapping_validation, 42)
        self.assertFalse((self.root / "split.json").exists())

    def test_rejects_inconsistent_sample_mappings(self):
        different_mapping = copy(self.validation.dataset.dataset)
        different_mapping.samples_sha256 = "0" * 64
        inconsistent_validation = SimpleNamespace(dataset=Subset(
            different_mapping, self.validation.dataset.indices,
        ))

        with self.assertRaisesRegex(ValueError, "inconsistent"):
            train.record_split(self.root, self.training, inconsistent_validation, 42)
        self.assertFalse((self.root / "split.json").exists())


if __name__ == "__main__":
    unittest.main()
