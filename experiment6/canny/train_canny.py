"""Experiment 6: FFHQ real vs. StyleGAN fake — Canny edge map detection.

Trains a ResNet classifier on 1-channel 256x256 edge maps to test whether
GAN artifacts survive Canny edge detection.

Real edge maps:  /general/omrosa/FFHQ/canny_reals/  (70 folders × 1000 PNGs)
Fake edge maps:  /general/omrosa/FFHQ/canny_fakes/  (70000 flat PNGs)

Usage:
    python train_canny.py
    python train_canny.py --epochs 50 --real-dir /path/to/reals --fake-dir /path/to/fakes
"""

import argparse
import os
import sys
import json
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from resnet_canny import ResNetCanny

IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}


# ── Dataset ──────────────────────────────────────────────────────────────────

class CannyDataset(Dataset):
    """Loads grayscale edge map PNGs from real (nested) and fake (flat) dirs.

    Real dir layout: real_dir/folder_000/img.png  (nested)
    Fake dir layout: fake_dir/img.png              (flat)

    Labels: real=1, fake=0
    """

    def __init__(self, real_paths, fake_paths, transform=None):
        self.paths = [(p, 1.0) for p in real_paths] + [(p, 0.0) for p in fake_paths]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path, label = self.paths[idx]
        img = Image.open(path).convert('L')  # force 1-channel grayscale
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.float32)


def collect_paths(directory, nested=False):
    """Return sorted list of image paths. If nested, recurse one level."""
    d = Path(directory)
    if nested:
        paths = []
        for sub in sorted(d.iterdir()):
            if sub.is_dir():
                paths.extend(sorted(p for p in sub.iterdir()
                                    if p.suffix.lower() in IMAGE_EXTENSIONS))
    else:
        paths = sorted(p for p in d.iterdir()
                       if p.suffix.lower() in IMAGE_EXTENSIONS)
    return paths


# ── Training helpers ──────────────────────────────────────────────────────────

def train_one_epoch(model, device, loader, optimizer, criterion, epoch):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in tqdm(loader, desc=f'Epoch {epoch}', leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        preds = model(imgs).squeeze(1)
        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct += ((preds >= 0.5).float() == labels).sum().item()
        total += imgs.size(0)
    return total_loss / total, 100.0 * correct / total


def evaluate(model, device, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs).squeeze(1)
            loss = criterion(preds, labels)
            total_loss += loss.item() * imgs.size(0)
            correct += ((preds >= 0.5).float() == labels).sum().item()
            total += imgs.size(0)
    return total_loss / total, 100.0 * correct / total


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--real-dir', default='/general/omrosa/FFHQ/canny_reals')
    p.add_argument('--fake-dir', default='/general/omrosa/FFHQ/canny_fakes')
    p.add_argument('--output-dir', default='results/resnet_canny')
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--num-workers', type=int, default=4)
    return p.parse_args()


def main():
    args = parse_args()

    # Seed
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Collect paths ──
    print('Collecting image paths...')
    real_paths = collect_paths(args.real_dir, nested=True)
    fake_paths = collect_paths(args.fake_dir, nested=False)
    print(f'  Real: {len(real_paths)} | Fake: {len(fake_paths)}')

    if len(real_paths) == 0 or len(fake_paths) == 0:
        print('ERROR: No images found. Check --real-dir and --fake-dir paths.')
        sys.exit(1)

    # Balance classes
    n = min(len(real_paths), len(fake_paths))
    rng = random.Random(args.seed)
    real_paths = rng.sample(real_paths, n)
    fake_paths = rng.sample(fake_paths, n)
    print(f'  Using {n} per class (balanced, total {2*n})')

    # ── Train/test split (80/20) ──
    rng2 = random.Random(args.seed + 1)
    rng2.shuffle(real_paths)
    rng2.shuffle(fake_paths)
    n_test = int(0.2 * n)
    n_train = n - n_test

    real_train, real_test = real_paths[:n_train], real_paths[n_train:]
    fake_train, fake_test = fake_paths[:n_train], fake_paths[n_train:]
    print(f'  Train: {2*n_train} | Test: {2*n_test}')

    # ── Transforms ──
    # Grayscale 256x256 → normalize to [-1, 1] (consistent with DCGAN training)
    train_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),                          # [0,1]
        transforms.Normalize((0.5,), (0.5,)),           # [-1,1]
    ])
    test_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    train_ds = CannyDataset(real_train, fake_train, transform=train_tf)
    test_ds  = CannyDataset(real_test,  fake_test,  transform=test_tf)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    # ── Model ──
    model = ResNetCanny().to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model: ResNetCanny | Parameters: {n_params:,}')

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCELoss()

    # ── Training loop ──
    train_losses, train_accs = [], []
    test_losses,  test_accs  = [], []
    best_test_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, device, train_loader,
                                          optimizer, criterion, epoch)
        te_loss, te_acc = evaluate(model, device, test_loader, criterion)

        train_losses.append(tr_loss)
        train_accs.append(tr_acc)
        test_losses.append(te_loss)
        test_accs.append(te_acc)

        print(f'Epoch {epoch:3d}/{args.epochs} | '
              f'Train loss={tr_loss:.4f} acc={tr_acc:.2f}% | '
              f'Test loss={te_loss:.4f} acc={te_acc:.2f}%')

        # Save best model
        if te_acc > best_test_acc:
            best_test_acc = te_acc
            torch.save(model.state_dict(),
                       os.path.join(args.output_dir, 'best_model.pth'))

        # Save log immediately after each epoch (crash-safe)
        log = {
            'architecture': 'resnet_canny',
            'experiment': 'canny_edge_maps',
            'real_dir': args.real_dir,
            'fake_dir': args.fake_dir,
            'n_per_class': n,
            'epochs_completed': epoch,
            'total_epochs': args.epochs,
            'seed': args.seed,
            'train_losses': train_losses,
            'train_accuracies': train_accs,
            'test_losses': test_losses,
            'test_accuracies': test_accs,
            'best_test_accuracy': best_test_acc,
            'final_train_accuracy': tr_acc,
            'final_test_accuracy': te_acc,
        }
        with open(os.path.join(args.output_dir, 'training_log.json'), 'w') as f:
            json.dump(log, f, indent=2)

    print(f'\nDone. Best test accuracy: {best_test_acc:.2f}%')
    print(f'Results saved to: {args.output_dir}')


if __name__ == '__main__':
    main()
