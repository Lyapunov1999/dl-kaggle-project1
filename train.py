"""Train SmallResNet on train_images/train_labels.csv and record every run."""

import argparse
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from platform import python_version

import numpy as np
import torch
import torchvision
from torch import nn

from data import MEAN, STD, make_loaders
from experiment import ExperimentLog, make_run_dir, write_json
from model import BLOCKS, WIDTHS, SmallResNet


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--images", default="./train_images", help="Directory containing train_XXXXX.png")
    parser.add_argument("--labels", default="./train_labels.csv", help="CSV with id,label columns")
    parser.add_argument("--runs-dir", default="./runs")
    parser.add_argument("--output", default=None, help="Checkpoint path; default: <run_dir>/best_resnet.pt")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0 or args.lr <= 0:
        parser.error("epochs, batch-size and lr must be positive; workers cannot be negative")
    if args.weight_decay < 0 or args.seed < 0:
        parser.error("weight-decay and seed cannot be negative")
    return args


def evaluate(model, loader, device, criterion):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    with torch.inference_mode():
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits = model(x)
            loss_sum += criterion(logits, y).item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            total += len(y)
    return loss_sum / total, correct / total


def train_epoch(model, loader, device, criterion, optimizer, scaler):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(x)
            loss = criterion(logits, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        loss_sum += loss.item() * len(y)
        correct += (logits.argmax(1) == y).sum().item()
        total += len(y)
    return loss_sum / total, correct / total


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    model = SmallResNet().to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if params > 5_000_000:
        raise ValueError(f"Model has {params:,} trainable parameters; the limit is 5,000,000")
    if args.smoke_test:
        x = torch.randn(4, 3, 32, 32, device=device)
        print(f"Device: {device}; parameters: {params:,}; output: {tuple(model(x).shape)}")
        return

    run_dir = make_run_dir(args.runs_dir)
    checkpoint = Path(args.output) if args.output else run_dir / "best_resnet.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    experiment = ExperimentLog(run_dir)
    logger = experiment.logger
    started_at = datetime.now(timezone.utc).isoformat()
    try:
        train_loader, val_loader = make_loaders(args.images, args.labels, args.batch_size, args.workers, args.seed, device)
    except Exception:
        logger.exception("Could not load training data")
        experiment.close()
        raise
    config = {
        "started_at_utc": started_at,
        "command": sys.argv,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "optimizer": "Adam",
        "weight_decay": args.weight_decay,
        "workers": args.workers,
        "seed": args.seed,
        "images": str(Path(args.images).resolve()),
        "labels": str(Path(args.labels).resolve()),
        "checkpoint": str(checkpoint.resolve()),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "numpy_version": np.__version__,
        "python_version": python_version(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "model": {
            "name": "SmallResNet", "widths": WIDTHS, "blocks": BLOCKS,
            "trainable_parameters": params,
        },
        "data": {
            "dataset": "train_images/train_labels.csv", "train_size": len(train_loader.dataset),
            "val_size": len(val_loader.dataset),
            "split_seed": args.seed, "mean": MEAN, "std": STD,
            "train_augmentation": ["RandomCrop(32, padding=4)", "RandomHorizontalFlip"],
        },
        "scheduler": "CosineAnnealingLR",
        "mixed_precision": device.type == "cuda",
    }
    write_json(run_dir / "config.json", config)
    logger.info(
        "Run: %s | device: %s | optimizer: Adam | lr: %.5g | parameters: %s",
        run_dir, device, args.lr, f"{params:,}",
    )
    try:
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
        best_acc = -1.0
        best_epoch = 0
        for epoch in range(1, args.epochs + 1):
            epoch_start = time.perf_counter()
            lr = optimizer.param_groups[0]["lr"]
            train_loss, train_acc = train_epoch(model, train_loader, device, criterion, optimizer, scaler)
            val_loss, val_acc = evaluate(model, val_loader, device, criterion)
            is_best = val_acc > best_acc
            if is_best:
                best_acc, best_epoch = val_acc, epoch
            if is_best or epoch % 10 == 0:
                checkpoint_data = {
                    "model": model.state_dict(), "val_acc": val_acc,
                    "params": params, "epoch": epoch, "config": config,
                }
                if is_best:
                    torch.save(checkpoint_data, checkpoint)
                if epoch % 10 == 0:
                    periodic_checkpoint = run_dir / f"resnet_epoch_{epoch:03d}.pt"
                    torch.save(checkpoint_data, periodic_checkpoint)
                    logger.info("Saved epoch %d checkpoint: %s", epoch, periodic_checkpoint)
            metrics = {
                "epoch": epoch, "lr": lr, "train_loss": train_loss,
                "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc,
                "epoch_seconds": time.perf_counter() - epoch_start,
                "is_best": is_best,
            }
            experiment.record_epoch(metrics)
            logger.info(
                "Epoch %03d/%d | lr=%.5g | train_loss=%.4f | train_acc=%.4f | val_loss=%.4f | val_acc=%.4f%s",
                epoch, args.epochs, lr, train_loss, train_acc, val_loss, val_acc,
                " | best" if is_best else "",
            )
            scheduler.step()
        summary = {
            "status": "completed",
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "best_val_acc": best_acc,
            "best_epoch": best_epoch,
            "checkpoint": str(checkpoint.resolve()),
        }
        write_json(run_dir / "summary.json", summary)
        logger.info("Best validation accuracy: %.4f at epoch %d; checkpoint: %s", best_acc, best_epoch, checkpoint)
    except Exception:
        logger.exception("Run failed")
        write_json(run_dir / "summary.json", {"status": "failed", "finished_at_utc": datetime.now(timezone.utc).isoformat()})
        raise
    finally:
        experiment.close()


if __name__ == "__main__":
    main()
