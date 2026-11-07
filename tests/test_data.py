"""Regression checks for CSV/image alignment and honest validation splits."""

import csv
import hashlib
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image

import data


class ImageCsvTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.images = self.root / "images"
        self.images.mkdir()
        self.labels = self.root / "labels.csv"

    def write_image(self, name, color):
        Image.new("RGB", (32, 32), color).save(self.images / name)

    def write_csv(self, rows, fields=("id", "filename", "label"), bom=False):
        encoding = "utf-8-sig" if bom else "utf-8"
        with self.labels.open("w", newline="", encoding=encoding) as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    def load(self):
        return data.LabeledImages(self.images, self.labels, lambda image: image.getpixel((0, 0)))

    def test_original_csv_binds_filename_and_label_in_row_order_without_writing(self):
        self.write_image("z_arbitrary.png", (255, 0, 0))
        self.write_image("a_arbitrary.png", (0, 255, 0))
        # Existing ID-derived decoys make ignoring filename observable.
        self.write_image("train_00007.png", (0, 0, 255))
        self.write_image("train_00002.png", (0, 0, 255))
        rows = [
            {"id": 7, "filename": "z_arbitrary.png", "label": 4},
            {"id": 2, "filename": "a_arbitrary.png", "label": 8},
        ]
        self.write_csv(rows, bom=True)
        original_bytes = self.labels.read_bytes()
        original_hash = hashlib.sha256(original_bytes).hexdigest()

        dataset = self.load()

        self.assertEqual(dataset.ids, [7, 2])
        self.assertEqual([path.name for path, _ in dataset.samples], ["z_arbitrary.png", "a_arbitrary.png"])
        self.assertEqual(dataset[0], ((255, 0, 0), 4))
        self.assertEqual(dataset[1], ((0, 255, 0), 8))
        self.assertEqual(dataset.labels_sha256, original_hash)
        self.assertRegex(dataset.samples_sha256, r"^[0-9a-f]{64}$")
        self.assertEqual(self.load().samples_sha256, dataset.samples_sha256)
        self.assertEqual(self.labels.read_bytes(), original_bytes)
        self.assertEqual(hashlib.sha256(self.labels.read_bytes()).hexdigest(), original_hash)

        self.write_csv(list(reversed(rows)), bom=True)
        reordered = self.load()
        self.assertEqual(reordered[0], ((0, 255, 0), 8))
        self.assertNotEqual(reordered.samples_sha256, dataset.samples_sha256)

    def test_legacy_id_label_csv_remains_supported(self):
        self.write_image("train_00009.png", (17, 18, 19))
        self.write_image("train_00001.png", (21, 22, 23))
        self.write_csv([{"id": 9, "label": 6}, {"id": 1, "label": 3}], fields=("id", "label"))

        dataset = self.load()

        self.assertEqual(dataset.ids, [9, 1])
        self.assertEqual(dataset[0], ((17, 18, 19), 6))
        self.assertEqual(dataset[1], ((21, 22, 23), 3))

    @unittest.skipUnless(os.name == "nt", "Windows filenames are case insensitive")
    def test_rejects_case_aliases_of_the_same_windows_image(self):
        self.write_image("shared.png", (1, 2, 3))
        self.assertTrue((self.images / "shared.png").samefile(self.images / "SHARED.PNG"))
        self.write_csv([
            {"id": 0, "filename": "shared.png", "label": 1},
            {"id": 1, "filename": "SHARED.PNG", "label": 1},
        ])

        with self.assertRaises(ValueError):
            self.load()

    def test_rejects_ambiguous_or_invalid_rows(self):
        self.write_image("a.png", (1, 2, 3))
        self.write_image("b.png", (4, 5, 6))
        Image.new("RGB", (32, 32), (7, 8, 9)).save(self.root / "outside.png")
        valid = {"id": 0, "filename": "a.png", "label": 1}
        second = {"id": 1, "filename": "b.png", "label": 2}
        cases = {
            "duplicate ID": [valid, {**second, "id": 0}],
            "duplicate filename": [valid, {**second, "filename": "a.png"}],
            "negative ID": [{**valid, "id": -1}, second],
            "noninteger ID": [{**valid, "id": "unknown"}, second],
            "negative label": [{**valid, "label": -1}, second],
            "label above range": [{**valid, "label": 10}, second],
            "nonnumeric label": [{**valid, "label": "cat"}, second],
            "empty filename": [{**valid, "filename": ""}, second],
            "parent traversal": [{**valid, "filename": "../outside.png"}, second],
            "Windows parent traversal": [{**valid, "filename": "..\\outside.png"}, second],
            "nested path": [{**valid, "filename": "nested/a.png"}, second],
            "Windows nested path": [{**valid, "filename": "nested\\a.png"}, second],
            "absolute path": [{**valid, "filename": (self.root / "outside.png").as_posix()}, second],
            "Windows absolute path": [{**valid, "filename": "C:\\outside.png"}, second],
        }
        for description, rows in cases.items():
            with self.subTest(description=description):
                self.write_csv(rows)
                with self.assertRaises(ValueError):
                    self.load()

    def test_missing_required_columns_and_missing_image_fail_early(self):
        for fields, rows in (
            (("filename", "label"), [{"filename": "a.png", "label": 1}]),
            (("id", "filename"), [{"id": 0, "filename": "a.png"}]),
        ):
            with self.subTest(fields=fields):
                self.write_csv(rows, fields=fields)
                with self.assertRaises(ValueError):
                    self.load()

        self.write_image("a.png", (1, 2, 3))
        self.write_csv([
            {"id": 0, "filename": "a.png", "label": 1},
            {"id": 1, "filename": "missing.png", "label": 2},
        ])
        with self.assertRaises(FileNotFoundError):
            self.load()

    def test_split_is_disjoint_reproducible_and_reads_one_csv(self):
        rows = []
        for index in range(20):
            filename = f"arbitrary_{index:02d}.png"
            self.write_image(filename, (index, 2 * index, 3 * index))
            rows.append({"id": 100 - index, "filename": filename, "label": index % 10})
        self.write_csv(rows)
        original_bytes = self.labels.read_bytes()
        device = torch.device("cpu")

        with patch.object(data.csv, "DictReader", wraps=csv.DictReader) as reader:
            train, validation = data.make_loaders(self.images, self.labels, 4, 0, 42, device)
        self.assertEqual(reader.call_count, 1)
        repeated_train, repeated_validation = data.make_loaders(self.images, self.labels, 4, 0, 42, device)

        train_indices = list(train.dataset.indices)
        val_indices = list(validation.dataset.indices)
        self.assertEqual(len(train_indices), 18)
        self.assertEqual(len(val_indices), 2)
        self.assertTrue(set(train_indices).isdisjoint(val_indices))
        self.assertEqual(set(train_indices) | set(val_indices), set(range(len(rows))))
        self.assertEqual(train_indices, list(repeated_train.dataset.indices))
        self.assertEqual(val_indices, list(repeated_validation.dataset.indices))
        train_full, validation_full = train.dataset.dataset, validation.dataset.dataset
        self.assertIsNot(train_full, validation_full)
        self.assertIs(train_full.samples, validation_full.samples)
        self.assertEqual(train_full.ids, validation_full.ids)
        self.assertEqual(train_full.labels_sha256, validation_full.labels_sha256)
        self.assertEqual(train_full.samples_sha256, validation_full.samples_sha256)

        # Validation must retain the same deterministic transform as inference.
        sample_index = val_indices[0]
        with Image.open(validation_full.samples[sample_index][0]) as image:
            expected = data.make_eval_transform()(image.convert("RGB"))
        first, label = validation_full[sample_index]
        second, repeated_label = validation_full[sample_index]
        self.assertTrue(torch.equal(first, expected))
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(label, rows[sample_index]["label"])
        self.assertEqual(label, repeated_label)
        self.assertEqual(self.labels.read_bytes(), original_bytes)

    def test_eval_transform_uses_supplied_normalization(self):
        mean, std = (0.1, 0.2, 0.3), (0.5, 0.25, 1.0)
        image = Image.new("RGB", (32, 32), (51, 102, 153))

        result = data.make_eval_transform(mean=mean, std=std)(image)

        expected = (torch.tensor([51, 102, 153]) / 255 - torch.tensor(mean)) / torch.tensor(std)
        self.assertEqual(tuple(result.shape), (3, 32, 32))
        torch.testing.assert_close(result[:, 0, 0], expected)


if __name__ == "__main__":
    unittest.main()
