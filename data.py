"""Image/CSV dataset and reproducible train/validation split."""

import csv
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import transforms

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)

class LabeledImages(Dataset):
    def __init__(self, image_dir, labels_csv, transform):
        image_dir, labels_csv = Path(image_dir), Path(labels_csv)
        if not image_dir.is_dir():
            raise FileNotFoundError(f"Training image directory not found: {image_dir}")
        if not labels_csv.is_file():
            raise FileNotFoundError(f"Training labels CSV not found: {labels_csv}")
        self.transform = transform
        self.samples = []
        seen = set()
        with labels_csv.open(newline="", encoding="utf-8-sig") as file:
            reader = csv.DictReader(file)
            if not reader.fieldnames or not {"id", "label"}.issubset(reader.fieldnames):
                raise ValueError(f"Expected id,label columns in {labels_csv}")
            for row in reader:
                image_id, label = int(row["id"]), int(row["label"])
                if image_id in seen:
                    raise ValueError(f"Duplicate image ID in {labels_csv}: {image_id}")
                if not 0 <= label <= 9:
                    raise ValueError(f"Label must be in 0..9 for image ID {image_id}: {label}")
                seen.add(image_id)
                path = image_dir / f"train_{image_id:05d}.png"
                if not path.is_file():
                    raise FileNotFoundError(f"Missing labeled training image: {path}")
                self.samples.append((path, label))
        if len(self.samples) < 2:
            raise ValueError("At least two labeled images are required")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        with Image.open(path) as image:
            return self.transform(image.convert("RGB")), label


def make_loaders(image_dir, labels_csv, batch_size, workers, seed, device):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    eval_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    train_full = LabeledImages(image_dir, labels_csv, train_transform)
    eval_full = LabeledImages(image_dir, labels_csv, eval_transform)
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
