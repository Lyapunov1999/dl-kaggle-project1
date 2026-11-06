import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = (nn.Identity() if stride == 1 and in_channels == out_channels else
                         nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                                       nn.BatchNorm2d(out_channels)))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x))))) + self.shortcut(x))

class SmallResNet(nn.Module):
    def __init__(self, widths=(32, 64, 128, 256), blocks=(2, 2, 2, 2)):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, widths[0], 3, 1, 1, bias=False),
                                  nn.BatchNorm2d(widths[0]), nn.ReLU(inplace=True))
        stages = []
        in_channels = widths[0]
        for i, (width, count) in enumerate(zip(widths, blocks)):
            layers = []
            for j in range(count):
                stride = 2 if i > 0 and j == 0 else 1
                layers.append(BasicBlock(in_channels, width, stride))
                in_channels = width
            stages.append(nn.Sequential(*layers))
        self.stages = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(widths[-1], 10)

    def forward(self, x):
        return self.fc(self.pool(self.stages(self.stem(x))).flatten(1))

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

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--lr', type=float, default=0.1)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--data-dir', default='./data')
    p.add_argument('--output', default='best_resnet.pt')
    p.add_argument('--smoke-test', action='store_true')
    args = p.parse_args()
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    print('Device:', device)
    if device.type == 'cuda':
        print('GPU:', torch.cuda.get_device_name(0))
    model = SmallResNet().to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable parameters: {params:,}')
    assert params <= 5_000_000
    if args.smoke_test:
        x = torch.randn(4, 3, 32, 32, device=device)
        print('Smoke-test output:', tuple(model(x).shape))
        return

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    eval_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    train_full = datasets.CIFAR10(args.data_dir, train=True, download=True, transform=train_transform)
    eval_full = datasets.CIFAR10(args.data_dir, train=True, download=False, transform=eval_transform)
    generator = torch.Generator().manual_seed(42)
    train_idx, val_idx = random_split(range(len(train_full)), [45000, 5000], generator=generator)
    train_data = torch.utils.data.Subset(train_full, train_idx.indices)
    val_data = torch.utils.data.Subset(eval_full, val_idx.indices)
    loader_kwargs = dict(batch_size=args.batch_size, num_workers=args.workers,
                         pin_memory=device.type == 'cuda', persistent_workers=args.workers > 0)
    train_loader = DataLoader(train_data, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_data, shuffle=False, **loader_kwargs)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, correct, loss_sum = 0, 0, 0.0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=device.type == 'cuda'):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_sum += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            total += len(y)
        val_loss, val_acc = evaluate(model, val_loader, device, criterion)
        print(f'Epoch {epoch:03d}/{args.epochs} | train_loss={loss_sum/total:.4f} '
              f'| train_acc={correct/total:.4%} | val_loss={val_loss:.4f} '
              f'| val_acc={val_acc:.4%}', flush=True)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({'model': model.state_dict(), 'val_acc': val_acc,
                        'params': params, 'epoch': epoch}, args.output)
        scheduler.step()
    print(f'Best validation accuracy: {best_acc:.4%}; checkpoint: {args.output}')

if __name__ == '__main__':
    main()
