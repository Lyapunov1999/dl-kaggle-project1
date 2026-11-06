"""Predict test_images with a trained SmallResNet and write a Kaggle CSV."""

import argparse
import csv
import re
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.torch_version import TorchVersion
from torchvision import transforms

from data import MEAN, STD
from model import SmallResNet


DEFAULT_CHECKPOINT = Path("runs/20261106T214241Z_885b2bd2/best_resnet.pt")
IMAGE_NAME = re.compile(r"test_(\d+)\.png$", re.IGNORECASE)
CLASS_NAMES = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")


class TestImages(Dataset):
    def __init__(self, directory):
        paths = list(Path(directory).glob("test_*.png"))
        if not paths or any(IMAGE_NAME.fullmatch(path.name) is None for path in paths):
            raise ValueError(f"No valid test_XXXX.png images found in {directory}")
        self.paths = sorted(paths, key=lambda path: int(IMAGE_NAME.fullmatch(path.name).group(1)))
        self.ids = [int(IMAGE_NAME.fullmatch(path.name).group(1)) for path in self.paths]
        if len(set(self.ids)) != len(self.ids):
            raise ValueError("Duplicate image IDs found")
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            return self.transform(image.convert("RGB"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--images", type=Path, default=Path("test_images"))
    parser.add_argument("--output", type=Path, default=Path("submission.csv"))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--label-format", choices=("index", "name"), default="index",
                        help="Use CIFAR-10 class indices (0-9) or class names")
    parser.add_argument("--id-column", default="Id")
    parser.add_argument("--label-column", default="Label")
    args = parser.parse_args()
    if args.batch_size < 1 or args.workers < 0:
        parser.error("batch-size must be positive and workers cannot be negative")

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    # train.py records torch.__version__ in the checkpoint config.
    with torch.serialization.safe_globals([TorchVersion]):
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = SmallResNet().to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    images = TestImages(args.images)
    loader = DataLoader(images, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                        pin_memory=device.type == "cuda")

    predictions = []
    with torch.inference_mode():
        for batch in loader:
            labels = model(batch.to(device, non_blocking=True)).argmax(dim=1).cpu().tolist()
            predictions.extend(labels)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow((args.id_column, args.label_column))
        for image_id, label in zip(images.ids, predictions, strict=True):
            writer.writerow((image_id, CLASS_NAMES[label] if args.label_format == "name" else label))
    print(f"Wrote {len(predictions)} predictions to {args.output} using {args.checkpoint} on {device}")


if __name__ == "__main__":
    main()
