"""Audit local labels and RGB image identity without changing any dataset files.

Only use trusted local CIFAR pickle files. This command never downloads data,
never uses official test labels, and does not run model training or prediction.
"""

import argparse
import csv
import hashlib
import json
import os
import pickle
import tempfile
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from data import LabeledImages


EXAMPLE_LIMIT = 5


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pixel_sha256(pixels):
    """Include shape so differently shaped RGB images cannot count as equal."""
    digest = hashlib.sha256()
    digest.update(f"RGB:{pixels.shape[1]}x{pixels.shape[0]}:".encode("ascii"))
    digest.update(pixels.tobytes())
    return digest.hexdigest()


def inspect_image(path):
    with Image.open(path) as image:
        mode, size = image.mode, image.size
        pixels = np.asarray(image.convert("RGB"))
    return pixel_sha256(pixels), mode, f"{size[0]}x{size[1]}"


def check_output_path(output, labels, image_dirs, cifar_dir, input_files):
    """Reject outputs that could replace labels, images, or CIFAR source files."""
    output = output.resolve()
    if output.suffix.lower() != ".json":
        raise ValueError("Audit output must have a .json extension")
    if output == labels.resolve():
        raise ValueError("Audit output cannot overwrite the labels CSV")
    for directory in (*image_dirs, cifar_dir):
        directory = directory.resolve()
        if output == directory or directory in output.parents:
            raise ValueError(f"Audit output cannot be inside an input directory: {directory}")
    # Resolving paths covers symlinks. samefile also covers an existing hard link.
    if output.exists():
        if not output.is_file():
            raise ValueError(f"Audit output is not a regular file: {output}")
        for source in input_files:
            if source.exists() and os.path.samefile(output, source):
                raise ValueError(f"Audit output refers to an input file: {source}")
    return output


def load_official_images(directory):
    """Return official pixel hashes; access targets only for training batches."""
    train_by_hash = defaultdict(list)
    train_hashes = []
    test_hashes = set()
    test_count = 0
    for index in range(1, 6):
        path = directory / f"data_batch_{index}"
        with path.open("rb") as source:
            batch = pickle.load(source, encoding="bytes")
        pixels = np.asarray(batch[b"data"], dtype=np.uint8)
        targets = batch[b"labels"]
        if pixels.ndim != 2 or pixels.shape[1] != 3072 or len(pixels) != len(targets):
            raise ValueError(f"Invalid local CIFAR training batch: {path}")
        pixels = pixels.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        for image, label in zip(pixels, targets):
            digest = pixel_sha256(image)
            train_by_hash[digest].append((len(train_hashes), int(label)))
            train_hashes.append(digest)
    with (directory / "test_batch").open("rb") as source:
        batch = pickle.load(source, encoding="bytes")
    # Deliberately do not access batch[b"labels"] from the official test batch.
    pixels = np.asarray(batch[b"data"], dtype=np.uint8)
    if pixels.ndim != 2 or pixels.shape[1] != 3072:
        raise ValueError("Invalid local CIFAR test image array")
    for image in pixels.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1):
        test_hashes.add(pixel_sha256(image))
        test_count += 1
    return train_by_hash, train_hashes, test_hashes, test_count


def duplicate_summary(hashes, names):
    groups = defaultdict(list)
    for digest, name in zip(hashes, names):
        groups[digest].append(name)
    duplicates = [members for members in groups.values() if len(members) > 1]
    return {
        "extra_images": sum(len(members) - 1 for members in duplicates),
        "groups": len(duplicates),
        "examples": [members[:EXAMPLE_LIMIT] for members in duplicates[:EXAMPLE_LIMIT]],
    }


def audit(args):
    started = time.perf_counter()
    images, labels = args.images.resolve(), args.labels.resolve()
    test_images, cifar_dir = args.test_images.resolve(), args.cifar_dir.resolve()
    labels_hash = file_sha256(labels)
    dataset = LabeledImages(images, labels, transform=None)
    with labels.open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        columns = reader.fieldnames
        rows = list(reader)
    # Dataset IDs and samples retain CSV row order, including shuffled CSVs.
    ids = dataset.ids
    train_paths = [path for path, _ in dataset.samples]
    train_targets = [int(label) for _, label in dataset.samples]
    if len(rows) != len(dataset) or len(ids) != len(dataset):
        raise ValueError("CSV row count and dataset sample/ID counts disagree")
    if not test_images.is_dir():
        raise FileNotFoundError(f"Test image directory not found: {test_images}")
    test_paths = sorted(test_images.glob("*.png"))
    if not test_paths:
        raise ValueError(f"No PNG test images found in {test_images}")
    official_files = [cifar_dir / f"data_batch_{index}" for index in range(1, 6)]
    official_files.append(cifar_dir / "test_batch")
    output = check_output_path(
        args.output, labels, (images, test_images), cifar_dir,
        [labels, *train_paths, *test_paths, *official_files],
    )

    print(f"Auditing {len(train_paths):,} training and {len(test_paths):,} test images...", flush=True)
    with ThreadPoolExecutor(max_workers=8) as executor:
        train_scan = list(executor.map(inspect_image, train_paths))
        test_scan = list(executor.map(inspect_image, test_paths))
    train_hashes = [item[0] for item in train_scan]
    test_hashes = [item[0] for item in test_scan]
    train_set, test_set = set(train_hashes), set(test_hashes)
    official_train, official_order, official_test, official_test_count = load_official_images(cifar_dir)
    official_all = set(official_train) | official_test

    filename_present = "filename" in columns
    convention_mismatches = []
    filename_paths_used = 0
    for row, image_id, path in zip(rows, ids, train_paths):
        expected = f"train_{image_id:05d}.png"
        if filename_present:
            filename = row["filename"]
            filename_paths_used += int(path.resolve() == (images / filename).resolve())
            if filename != expected:
                convention_mismatches.append({"id": image_id, "filename": filename, "id_based_filename": expected})
    matched = label_matches = aligned = 0
    label_mismatches = []
    unmatched_train = []
    for image_id, path, label, digest in zip(ids, train_paths, train_targets, train_hashes):
        matches = official_train.get(digest, ())
        matched += int(bool(matches))
        aligned += int(0 <= image_id < len(official_order) and digest == official_order[image_id])
        if matches:
            same_label = any(official_label == label for _, official_label in matches)
            label_matches += int(same_label)
            if not same_label and len(label_mismatches) < EXAMPLE_LIMIT:
                label_mismatches.append({"id": image_id, "filename": path.name, "csv_label": label,
                                         "official_train_labels": sorted({item[1] for item in matches})})
        elif len(unmatched_train) < EXAMPLE_LIMIT:
            unmatched_train.append({"id": image_id, "filename": path.name})
    overlap_examples = [path.name for path, digest in zip(test_paths, test_hashes) if digest in train_set]
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {"images": str(images), "labels": str(labels), "test_images": str(test_images),
                   "cifar_dir": str(cifar_dir)},
        "labels": {
            "sha256": labels_hash,
            "row_count": len(rows),
            "columns": columns,
            "filename_column_present": filename_present,
            "filename_paths_used": filename_paths_used if filename_present else None,
            "id_filename_convention_mismatches": len(convention_mismatches) if filename_present else None,
            "id_filename_mismatch_examples": convention_mismatches[:EXAMPLE_LIMIT],
            "per_class_count": {str(label): train_targets.count(label) for label in range(10)},
        },
        "training_images": {
            "count": len(train_paths),
            "modes": dict(Counter(item[1] for item in train_scan)),
            "sizes": dict(Counter(item[2] for item in train_scan)),
            "duplicates": duplicate_summary(train_hashes, [path.name for path in train_paths]),
            "matched_official_train_images": matched,
            "matched_official_train_image_and_label": label_matches,
            "matched_official_train_image_at_csv_id": aligned,
            "matched_images_with_inconsistent_label": matched - label_matches,
            "unmatched_official_train_images": len(train_paths) - matched,
            "label_mismatch_examples": label_mismatches,
            "unmatched_examples": unmatched_train,
        },
        "test_images": {
            "count": len(test_paths),
            "modes": dict(Counter(item[1] for item in test_scan)),
            "sizes": dict(Counter(item[2] for item in test_scan)),
            "duplicates": duplicate_summary(test_hashes, [path.name for path in test_paths]),
            "matched_official_train_images": sum(digest in official_train for digest in test_hashes),
            "matched_official_test_images": sum(digest in official_test for digest in test_hashes),
            "matched_any_official_cifar_images": sum(digest in official_all for digest in test_hashes),
            "unmatched_examples": [path.name for path, digest in zip(test_paths, test_hashes)
                                   if digest not in official_all][:EXAMPLE_LIMIT],
        },
        "train_test_overlap": {
            "unique_images": len(train_set & test_set),
            "test_images_in_training": len(overlap_examples),
            "examples": overlap_examples[:EXAMPLE_LIMIT],
        },
        "official_reference": {
            "training_image_count": len(official_order),
            "test_image_count": official_test_count,
            "test_labels_used": False,
            "downloaded": False,
        },
        "comparison": "SHA-256 of decoded RGB dimensions and pixel bytes; exact pixel identity only",
        "limitations": [
            "Exact image matching does not detect near duplicates or certify semantic annotation quality.",
            "Unmatched test images do not by themselves identify their source or quantify distribution shift.",
        ],
    }
    if file_sha256(labels) != labels_hash:
        raise RuntimeError("Labels CSV changed during the audit; no report written")
    report["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    # Recheck immediately before writing, then replace the report atomically.
    output = check_output_path(output, labels, (images, test_images), cifar_dir,
                               [labels, *train_paths, *test_paths, *official_files])
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output.parent,
                                         suffix=".json.tmp", delete=False) as destination:
            temporary = Path(destination.name)
            json.dump(report, destination, ensure_ascii=False, indent=2)
            destination.write("\n")
        temporary.replace(output)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return report, output


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, default=Path("train_images"))
    parser.add_argument("--labels", type=Path, default=Path("train_labels.csv"))
    parser.add_argument("--test-images", type=Path, default=Path("test_images"))
    parser.add_argument("--cifar-dir", type=Path, default=Path("data/cifar-10-batches-py"))
    parser.add_argument("--output", type=Path, default=Path("outputs/data_audit.json"))
    return parser.parse_args()


def main():
    report, output = audit(parse_args())
    print(f"Training image/label matches: {report['training_images']['matched_official_train_image_and_label']:,}"
          f"/{report['labels']['row_count']:,}; test matches in official CIFAR: "
          f"{report['test_images']['matched_any_official_cifar_images']:,}/{report['test_images']['count']:,}")
    print(f"Wrote {output} in {report['elapsed_seconds']:.3f}s; labels SHA-256: {report['labels']['sha256']}")


if __name__ == "__main__":
    main()
