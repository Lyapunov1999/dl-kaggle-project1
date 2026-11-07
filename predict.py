"""Predict test_images with a trained SmallResNet and write a Kaggle CSV."""

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.torch_version import TorchVersion

from data import CLASS_NAMES, MEAN, STD, make_eval_transform
from model import SmallResNet


IMAGE_NAME = re.compile(r"test_(\d+)\.png$", re.IGNORECASE)


class TestImages(Dataset):
    def __init__(self, directory, transform=None):
        paths = list(Path(directory).glob("test_*.png"))
        if not paths or any(IMAGE_NAME.fullmatch(path.name) is None for path in paths):
            raise ValueError(f"No valid test_XXXX.png images found in {directory}")
        self.paths = sorted(paths, key=lambda path: int(IMAGE_NAME.fullmatch(path.name).group(1)))
        self.ids = [int(IMAGE_NAME.fullmatch(path.name).group(1)) for path in self.paths]
        if len(set(self.ids)) != len(self.ids):
            raise ValueError("Duplicate image IDs found")
        self.transform = make_eval_transform() if transform is None else transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            return self.transform(image.convert("RGB"))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Explicit checkpoint from the training run to submit")
    parser.add_argument("--images", type=Path, default=Path("test_images"))
    parser.add_argument("--output", type=Path, default=Path("submission.csv"))
    parser.add_argument("--sample-submission", type=Path, default=Path("sample_submission.csv"),
                        help="Template defining the exact test IDs and their output order")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--label-format", choices=("index", "name"), default="index",
                        help="Use CIFAR-10 class indices (0-9) or class names")
    parser.add_argument("--id-column", default="id")
    parser.add_argument("--label-column", default="label")
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.workers < 0:
        parser.error("batch-size must be positive and workers cannot be negative")
    if not args.id_column or not args.label_column or args.id_column == args.label_column:
        parser.error("id-column and label-column must be distinct, nonempty names")
    return args


def read_template(path, id_column="id", label_column="label"):
    """Keep the template's row order and literal IDs while checking numeric identity."""
    with Path(path).open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if (not reader.fieldnames or len(reader.fieldnames) != 2
                or set(reader.fieldnames) != {id_column, label_column}):
            raise ValueError(f"Template must contain exactly {id_column},{label_column} columns: {path}")
        columns = reader.fieldnames
        rows, seen = [], set()
        for line_number, row in enumerate(reader, start=2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"Malformed template row {line_number}: {path}")
            try:
                image_id = int(row[id_column])
            except (TypeError, ValueError) as error:
                raise ValueError(f"Invalid template image ID at row {line_number}: {row[id_column]!r}") from error
            if image_id < 0 or image_id in seen:
                raise ValueError(f"Negative or duplicate template image ID: {image_id}")
            seen.add(image_id)
            rows.append((image_id, row[id_column]))
    if not rows:
        raise ValueError(f"Template has no image IDs: {path}")
    return columns, rows


def validate_image_ids(images, template_rows):
    image_ids = set(images.ids)
    template_ids = {image_id for image_id, _ in template_rows}
    missing, extra = template_ids - image_ids, image_ids - template_ids
    if missing or extra:
        raise ValueError(
            f"Template/image ID mismatch; missing images: {sorted(missing)[:10]}; "
            f"extra images: {sorted(extra)[:10]}"
        )


def checkpoint_normalization(checkpoint):
    data_config = checkpoint.get("config", {}).get("data", {})
    mean, std = data_config.get("mean", MEAN), data_config.get("std", STD)
    try:
        mean, std = tuple(float(value) for value in mean), tuple(float(value) for value in std)
    except (TypeError, ValueError) as error:
        raise ValueError("Checkpoint mean/std must contain three finite numbers") from error
    if (len(mean) != 3 or len(std) != 3 or not all(math.isfinite(value) for value in mean + std)
            or any(value <= 0 for value in std)):
        raise ValueError("Checkpoint mean/std must contain three finite numbers and std must be positive")
    return mean, std


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_output_paths(output, metadata_path, protected_paths):
    for destination in (output, metadata_path):
        resolved = destination.resolve()
        for source in protected_paths:
            if resolved == source.resolve() or (
                    destination.exists() and source.exists() and destination.samefile(source)):
                raise ValueError(f"Output would overwrite an input file: {destination} -> {source}")


def main(argv=None):
    args = parse_args(argv)
    columns, template_rows = read_template(args.sample_submission, args.id_column, args.label_column)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    # train.py records torch.__version__ in the checkpoint config.
    with torch.serialization.safe_globals([TorchVersion]):
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    mean, std = checkpoint_normalization(checkpoint)
    images = TestImages(args.images, transform=make_eval_transform(mean=mean, std=std))
    validate_image_ids(images, template_rows)
    metadata_path = args.output.with_name(args.output.name + ".metadata.json")
    protected_paths = [args.checkpoint, args.sample_submission, Path.cwd() / "train_labels.csv", *images.paths]
    recorded_labels = checkpoint.get("config", {}).get("labels")
    if recorded_labels:
        protected_paths.append(Path(recorded_labels))
    validate_output_paths(args.output, metadata_path, protected_paths)
    class_names = checkpoint.get("config", {}).get("data", {}).get("class_names", CLASS_NAMES)
    if (not isinstance(class_names, (list, tuple)) or len(class_names) != 10
            or any(not isinstance(name, str) for name in class_names) or len(set(class_names)) != 10):
        raise ValueError("Checkpoint class_names must contain ten unique class names")
    print(f"Checkpoint: {args.checkpoint.resolve()} | epoch: {checkpoint.get('epoch', 'unknown')} | "
          f"saved validation accuracy: {checkpoint.get('val_acc', 'unknown')}")
    model_config = checkpoint.get("config", {}).get("model", {})
    # Checkpoints predating architecture metadata use the original smaller model.
    model = SmallResNet(widths=model_config.get("widths", (32, 64, 128, 256)),
                        blocks=model_config.get("blocks", (2, 2, 2, 2))).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    loader = DataLoader(images, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                        pin_memory=device.type == "cuda")

    predictions = []
    with torch.inference_mode():
        for batch in loader:
            labels = model(batch.to(device, non_blocking=True)).argmax(dim=1).cpu().tolist()
            predictions.extend(labels)

    predictions_by_id = dict(zip(images.ids, predictions, strict=True))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for image_id, original_id in template_rows:
            label = predictions_by_id[image_id]
            writer.writerow({args.id_column: original_id,
                             args.label_column: class_names[label] if args.label_format == "name" else label})
    metadata = {
        "checkpoint": {
            "path": str(args.checkpoint.resolve()), "sha256": file_sha256(args.checkpoint),
            "epoch": checkpoint.get("epoch"), "val_acc": checkpoint.get("val_acc"),
            "config": checkpoint.get("config", {}),
        },
        "submission": {"path": str(args.output.resolve()), "sha256": file_sha256(args.output)},
        "sample_submission": {"path": str(args.sample_submission.resolve()),
                              "sha256": file_sha256(args.sample_submission)},
        "images": str(args.images.resolve()),
        "normalization": {"mean": mean, "std": std},
        "class_names": class_names, "prediction_count": len(predictions),
        "label_format": args.label_format, "id_column": args.id_column, "label_column": args.label_column,
    }
    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)
        file.write("\n")
    print(f"Wrote {len(predictions)} predictions to {args.output} using {args.checkpoint} on {device}")
    print(f"Prediction metadata: {metadata_path}")


if __name__ == "__main__":
    main()
