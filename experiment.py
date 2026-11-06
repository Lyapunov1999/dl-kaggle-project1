"""Durable, per-run experiment metadata and epoch metrics."""

import csv
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

METRIC_FIELDS = ("epoch", "lr", "train_loss", "train_acc", "val_loss", "val_acc", "epoch_seconds", "is_best")


def make_run_dir(runs_dir):
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]
    run_dir = Path(runs_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def write_json(path, data):
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)
        file.write("\n")


class ExperimentLog:
    def __init__(self, run_dir):
        self.run_dir = run_dir
        self.logger = logging.getLogger(f"experiment.{run_dir.name}")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        for handler in (logging.StreamHandler(), logging.FileHandler(run_dir / "train.log", encoding="utf-8")):
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        self.metrics_file = (run_dir / "metrics.csv").open("w", newline="", encoding="utf-8")
        self.metrics = csv.DictWriter(self.metrics_file, fieldnames=METRIC_FIELDS)
        self.metrics.writeheader()
        self.metrics_file.flush()

    def record_epoch(self, metrics):
        self.metrics.writerow(metrics)
        self.metrics_file.flush()

    def close(self):
        self.metrics_file.close()
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)
