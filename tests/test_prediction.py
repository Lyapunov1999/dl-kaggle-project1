"""Submission alignment, preprocessing provenance, and input-file protection."""

import contextlib
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image
from torch import nn

import predict


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.received = []

    def forward(self, images):
        self.received.append(images.detach().cpu())
        labels = (images[:, 0].mean(dim=(1, 2)) + 2).round().long().clamp(0, 9)
        return torch.zeros(len(images), 10, device=images.device).scatter_(1, labels[:, None], 1)


class PredictionTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.images = self.directory / "images"
        self.images.mkdir()
        for image_id, value in ((7, 0), (42, 128), (105, 255)):
            Image.new("RGB", (4, 4), (value, value, value)).save(self.images / f"test_{image_id:04d}.png")
        self.template = self.directory / "sample.csv"
        self.write_template([105, 42, 7])
        self.checkpoint = self.directory / "model.pt"
        torch.save({
            "model": {}, "epoch": 12, "val_acc": 0.937,
            "config": {"model": {"widths": [1], "blocks": [1]},
                       "data": {"mean": [0.5, 0.5, 0.5], "std": [0.25, 0.25, 0.25],
                                "class_names": [f"class_{index}" for index in range(10)]}},
        }, self.checkpoint)
        self.output = self.directory / "prediction.csv"

    def write_template(self, ids, columns=("id", "label")):
        with self.template.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(columns)
            writer.writerows((image_id, 0) for image_id in ids)

    def run_prediction(self, extra_args=(), output=None):
        model = TinyModel()
        args = ["--checkpoint", str(self.checkpoint), "--images", str(self.images),
                "--sample-submission", str(self.template), "--output", str(output or self.output),
                "--workers", "0", "--batch-size", "2", *extra_args]
        with patch.object(predict, "SmallResNet", return_value=model), \
                patch.object(torch.cuda, "is_available", return_value=False), \
                patch.object(torch.backends.mps, "is_available", return_value=False), \
                contextlib.redirect_stdout(io.StringIO()):
            predict.main(args)
        return model

    def test_checkpoint_must_be_explicit(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            predict.parse_args([])
        self.assertEqual(error.exception.code, 2)

    def test_unordered_noncontiguous_ids_follow_template_and_record_provenance(self):
        model = self.run_prediction()
        with self.output.open(newline="", encoding="utf-8") as file:
            self.assertEqual(list(csv.reader(file)), [["id", "label"], ["105", "4"], ["42", "2"], ["7", "0"]])
        received = torch.cat(model.received)
        self.assertTrue(torch.allclose(received[:, 0, 0, 0],
                                       torch.tensor([-2.0, (128 / 255 - 0.5) / 0.25, 2.0]), atol=1e-6))
        metadata = json.loads(self.output.with_name(self.output.name + ".metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["checkpoint"]["epoch"], 12)
        self.assertEqual(metadata["checkpoint"]["val_acc"], 0.937)
        self.assertEqual(metadata["checkpoint"]["sha256"], predict.file_sha256(self.checkpoint))
        self.assertEqual(metadata["submission"]["sha256"], predict.file_sha256(self.output))
        self.assertEqual(metadata["sample_submission"]["sha256"], predict.file_sha256(self.template))
        self.assertEqual(metadata["normalization"], {"mean": [0.5] * 3, "std": [0.25] * 3})
        self.assertEqual(metadata["prediction_count"], 3)
        self.assertEqual(metadata["label_format"], "index")

    def test_custom_columns_and_checkpoint_class_names(self):
        with self.template.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(("category", "image_id"))
            writer.writerows(((0, "00105"), (0, "00007"), (0, "00042")))
        self.run_prediction(["--id-column", "image_id", "--label-column", "category", "--label-format", "name"])
        with self.output.open(newline="", encoding="utf-8") as file:
            self.assertEqual(list(csv.reader(file)), [["category", "image_id"],
                                                     ["class_4", "00105"], ["class_0", "00007"], ["class_2", "00042"]])

    def test_template_missing_image_is_rejected(self):
        self.write_template([105, 42, 7, 999])
        with self.assertRaisesRegex(ValueError, "missing images: \\[999\\]"):
            self.run_prediction()
        self.assertFalse(self.output.exists())

    def test_extra_image_is_rejected(self):
        self.write_template([105, 42])
        with self.assertRaisesRegex(ValueError, "extra images: \\[7\\]"):
            self.run_prediction()
        self.assertFalse(self.output.exists())

    def test_duplicate_template_id_is_rejected(self):
        self.write_template([105, 42, 7, "0007"])
        with self.assertRaisesRegex(ValueError, "duplicate template image ID"):
            self.run_prediction()

    def test_duplicate_image_id_is_rejected(self):
        Image.new("RGB", (4, 4)).save(self.images / "test_7.png")
        with self.assertRaisesRegex(ValueError, "Duplicate image IDs"):
            self.run_prediction()

    def test_template_requires_exact_columns_and_valid_rows(self):
        for content in ("id,label,extra\n7,0,x\n", "id,id\n7,0\n", "id,label\n7,0,extra\n",
                        "id,label\nno_id,0\n", "id,label\n", "id,label\n7\n"):
            with self.subTest(content=content):
                self.template.write_text(content, encoding="utf-8")
                with self.assertRaises(ValueError):
                    predict.read_template(self.template)

    def test_input_files_cannot_be_overwritten(self):
        for path in (self.checkpoint, self.template, self.images / "test_0007.png"):
            with self.subTest(path=path):
                original = path.read_bytes()
                with self.assertRaisesRegex(ValueError, "overwrite an input file"):
                    self.run_prediction(output=path)
                self.assertEqual(path.read_bytes(), original)
        with self.assertRaisesRegex(ValueError, "overwrite an input file"):
            predict.validate_output_paths(self.output, self.checkpoint, [self.checkpoint])

    def test_original_training_labels_cannot_be_overwritten(self):
        labels = self.directory / "train_labels.csv"
        labels.write_text("id,filename,label\n0,train_00000.png,6\n", encoding="utf-8")
        original = labels.read_bytes()
        with patch.object(predict.Path, "cwd", return_value=self.directory), \
                self.assertRaisesRegex(ValueError, "overwrite an input file"):
            self.run_prediction(output=labels)
        self.assertEqual(labels.read_bytes(), original)

    def test_checkpoint_recorded_training_labels_cannot_be_overwritten(self):
        labels = self.directory / "original-competition-labels.csv"
        labels.write_text("id,filename,label\n0,train_00000.png,6\n", encoding="utf-8")
        original = labels.read_bytes()
        checkpoint = torch.load(self.checkpoint, map_location="cpu", weights_only=True)
        checkpoint["config"]["labels"] = str(labels)
        torch.save(checkpoint, self.checkpoint)
        with self.assertRaisesRegex(ValueError, "overwrite an input file"):
            self.run_prediction(output=labels)
        self.assertEqual(labels.read_bytes(), original)

    def test_normalization_falls_back_for_legacy_checkpoints_and_rejects_invalid_metadata(self):
        self.assertEqual(predict.checkpoint_normalization({}), (predict.MEAN, predict.STD))
        for mean, std in (([0, 0], [1, 1, 1]), ([0, 0, float("nan")], [1, 1, 1]),
                          ([0, 0, 0], [1, 0, 1]), ([0, 0, 0], [1, -1, 1])):
            with self.subTest(mean=mean, std=std), self.assertRaises(ValueError):
                predict.checkpoint_normalization({"config": {"data": {"mean": mean, "std": std}}})

    def test_legacy_checkpoint_uses_default_preprocessing_and_class_names(self):
        torch.save({"model": {}, "epoch": 2, "val_acc": 0.8}, self.checkpoint)
        model = self.run_prediction(["--label-format", "name"])
        expected_black = -torch.tensor(predict.MEAN) / torch.tensor(predict.STD)
        self.assertTrue(torch.allclose(torch.cat(model.received)[0, :, 0, 0], expected_black))
        metadata = json.loads(self.output.with_name(self.output.name + ".metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["class_names"], list(predict.CLASS_NAMES))


if __name__ == "__main__":
    unittest.main()
