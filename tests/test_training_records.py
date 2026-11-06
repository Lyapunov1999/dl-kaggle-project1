"""Check that a complete training run leaves usable report data."""

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import train


class TrainingRecordTest(unittest.TestCase):
    def test_one_epoch_writes_config_metrics_summary_and_checkpoint(self):
        samples = TensorDataset(torch.randn(4, 3, 32, 32), torch.tensor([0, 1, 0, 1]))
        loader = DataLoader(samples, batch_size=2)
        tiny_model = lambda: nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 10))
        with tempfile.TemporaryDirectory() as directory:
            args = ["train.py", "--epochs", "1", "--workers", "0", "--runs-dir", directory]
            with patch.object(sys, "argv", args), patch.object(train, "make_loaders", return_value=(loader, loader)), \
                    patch.object(train, "SmallResNet", tiny_model):
                train.main()

            run_dirs = list(Path(directory).iterdir())
            self.assertEqual(len(run_dirs), 1)
            run_dir = run_dirs[0]
            config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
            summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
            with (run_dir / "metrics.csv").open(newline="", encoding="utf-8") as file:
                metrics = list(csv.DictReader(file))
            self.assertEqual(config["epochs"], 1)
            self.assertEqual(len(metrics), 1)
            self.assertEqual(metrics[0]["epoch"], "1")
            self.assertEqual(summary["status"], "completed")
            self.assertTrue((run_dir / "best_resnet.pt").is_file())
            self.assertIn("Epoch 001/1", (run_dir / "train.log").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
