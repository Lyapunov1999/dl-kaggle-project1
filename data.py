"""Image/CSV dataset and reproducible train/validation split."""

import csv
import hashlib
import io
import json
import math
from copy import copy
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import transforms

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)
CLASS_NAMES = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")


def make_eval_transform(mean=MEAN, std=STD):
    """Use exactly the same preprocessing for validation and prediction."""
    mean, std = tuple(map(float, mean)), tuple(map(float, std))
    if len(mean) != 3 or len(std) != 3 or not all(math.isfinite(x) for x in mean + std) or any(x <= 0 for x in std):
        raise ValueError("RGB normalization requires three finite means and three positive standard deviations")
    return transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])


class LabeledImages(Dataset):
    def __init__(self, image_dir, labels_csv, transform):
        image_dir, labels_csv = Path(image_dir), Path(labels_csv)
        if not image_dir.is_dir():
            raise FileNotFoundError(f"Training image directory not found: {image_dir}")
        if not labels_csv.is_file():
            raise FileNotFoundError(f"Training labels CSV not found: {labels_csv}")
        self.transform = transform
        self.samples = []
        self.ids = []
        seen_ids, seen_paths = set(), set()
        # Hash and parse the same bytes; never rewrite the source CSV.
        contents = labels_csv.read_bytes()
        self.labels_sha256 = hashlib.sha256(contents).hexdigest()
        with io.StringIO(contents.decode("utf-8-sig"), newline="") as file:
            reader = csv.DictReader(file)
            if not reader.fieldnames or not {"id", "label"}.issubset(reader.fieldnames):
                raise ValueError(f"Expected id,label columns in {labels_csv}")
            if len(reader.fieldnames) != len(set(reader.fieldnames)):
                raise ValueError(f"Duplicate column names in {labels_csv}")
            for row in reader:
                if None in row or any(value is None for value in row.values()):
                    raise ValueError(f"Malformed CSV row at line {reader.line_num} in {labels_csv}")
                try:
                    image_id, label = int(row["id"]), int(row["label"])
                except (TypeError, ValueError) as error:
                    raise ValueError(f"Invalid id or label at line {reader.line_num} in {labels_csv}") from error
                if image_id < 0:
                    raise ValueError(f"Image ID cannot be negative: {image_id}")
                if image_id in seen_ids:
                    raise ValueError(f"Duplicate image ID in {labels_csv}: {image_id}")
                if not 0 <= label <= 9:
                    raise ValueError(f"Label must be in 0..9 for image ID {image_id}: {label}")
                # The original competition CSV explicitly binds filename to label.
                # Keep support for older id,label CSVs without changing either file.
                filename = row["filename"] if "filename" in row else f"train_{image_id:05d}.png"
                if not filename or filename in (".", "..") or any(char in filename for char in ("/", "\\", ":")):
                    raise ValueError(f"filename must be a nonempty file name, without directories: {filename!r}")
                path = image_dir / filename
                if not path.is_file():
                    raise FileNotFoundError(f"Missing labeled training image: {path}")
                resolved_path = path.resolve()
                if resolved_path in seen_paths:
                    raise ValueError(f"Duplicate filename in {labels_csv}: {filename}")
                seen_ids.add(image_id)
                seen_paths.add(resolved_path)
                self.ids.append(image_id)
                self.samples.append((path, label))
        if len(self.samples) < 2:
            raise ValueError("At least two labeled images are required")
        mapping = [(image_id, path.name, label) for image_id, (path, label) in zip(self.ids, self.samples, strict=True)]
        self.samples_sha256 = hashlib.sha256(json.dumps(mapping, separators=(",", ":")).encode("utf-8")).hexdigest()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        with Image.open(path) as image:
            if image.size != (32, 32):
                raise ValueError(f"Expected a 32x32 image: {path}, got {image.size}")
            image = image.convert("RGB")
            return self.transform(image) if self.transform else image, label


def make_loaders(image_dir, labels_csv, batch_size, workers, seed, device):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    eval_transform = make_eval_transform()
    train_full = LabeledImages(image_dir, labels_csv, train_transform)
    # Read the CSV once so both subsets share the same sample mapping.
    eval_full = copy(train_full)
    eval_full.transform = eval_transform
    val_size = max(1, round(len(train_full) * 0.1))
    train_size = len(train_full) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_idx, val_idx = random_split(range(len(train_full)), [train_size, val_size], generator=generator)
    train_data = Subset(train_full, train_idx.indices)
    val_data = Subset(eval_full, val_idx.indices)
    loader_kwargs = dict(batch_size=batch_size, num_workers=workers,
                         pin_memory=device.type == "cuda", persistent_workers=workers > 0)
    train_loader = DataLoader(train_data, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_data, shuffle=False, **loader_kwargs)
    return train_loader, val_loader
