"""Experiment 6: FFHQ real vs. fake — Segmentation map detection.

Trains ResNet or DenseNet on raw 1-channel segmentation maps (values 0-18)
to test whether GAN artifacts survive face parsing.

Raw maps are the *_raw.png files in:
  Real: /general/omrosa/FFHQ/segm_reals/resnet18/
  Fake: /general/omrosa/FFHQ/segm_fakes/resnet18/

Usage:
    python train_segm.py --arch resnet
    python train_segm.py --arch densenet
    python train_segm.py --arch resnet --resume results/resnet_segm/best_model.pth --lr 1e-4
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
from densenet_segm import DenseNetSegm

IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg'}


# ── Dataset ──────────────────────────────────────────────────────────────────

class SegmDataset(Dataset):
    """Loads raw segmentation map PNGs (only *_raw.png files).

    Labels: real=1, fake=0
    """

    def __init__(self, real_paths, fake_paths, transform=None):
        self.paths = [(p, 1.0) for p in real_paths] + [(p, 0.0) for p in fake_paths]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path, label = self.paths[idx]
        img = Image.open(path).convert('L')  # grayscale, values 0-18
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.float32)


def collect_raw_paths(directory):
    """Return sorted list of *_raw.png paths only."""
    d = Path(directory)
    return sorted(p for p in d.iterdir()
                  if p.name.endswith('_raw.png'))


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
    p.add_argument('--arch', required=True, choices=['resnet', 'densenet'],
                   help='Architecture to train')
    p.add_argument('--real-dir',
                   default='/general/omrosa/FFHQ/segm_reals/resnet18')
    p.add_argument('--fake-dir',
                   default='/general/omrosa/FFHQ/segm_fakes/resnet18')
    p.add_argument('--output-base',
                   default='/home/o/omrosa/research/experiment6/face-parsing/results')
    p.add_argument('--resume', type=str, default=None,
                   help='Path to checkpoint to resume from')
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight-decay', type=float, default=0.0)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--num-workers', type=int, default=4)
    return p.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    output_dir = os.path.join(args.output_base, f'{args.arch}_segm')
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    print(f'Arch: {args.arch} | LR: {args.lr} | WD: {args.weight_decay}')

    # ── Collect raw paths ──
    print('Collecting raw segmentation map paths...')
    real_paths = collect_raw_paths(args.real_dir)
    fake_paths = collect_raw_paths(args.fake_dir)
    print(f'  Real: {len(real_paths)} | Fake: {len(fake_paths)}')

    if len(real_paths) == 0 or len(fake_paths) == 0:
        print('ERROR: No *_raw.png files found. Check --real-dir and --fake-dir.')
        sys.exit(1)

    # Balance classes
    n = min(len(real_paths), len(fake_paths))
    rng = random.Random(args.seed)
    real_paths = rng.sample(real_paths, n)
    fake_paths = rng.sample(fake_paths, n)
    print(f'  Using {n} per class (balanced, total {2*n})')

    # 80/20 split
    rng2 = random.Random(args.seed + 1)
    rng2.shuffle(real_paths)
    rng2.shuffle(fake_paths)
    n_test  = int(0.2 * n)
    n_train = n - n_test

    real_train, real_test = real_paths[:n_train], real_paths[n_train:]
    fake_train, fake_test = fake_paths[:n_train], fake_paths[n_train:]
    print(f'  Train: {2*n_train} | Test: {2*n_test}')

    # ── Transforms ──
    # Normalize to [-1,1] — consistent with canny experiment
    tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),           # [0,1]
        transforms.Normalize((0.5,), (0.5,)),  # [-1,1]
    ])

    train_ds = SegmDataset(real_train, fake_train, transform=tf)
    test_ds  = SegmDataset(real_test,  fake_test,  transform=tf)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    # ── Model ──
    if args.arch == 'resnet':
        model = ResNetCanny().to(device)
    else:
        model = DenseNetSegm().to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model: {args.arch} | Parameters: {n_params:,}')

    # ── Resume ──
    if args.resume:
        if os.path.exists(args.resume):
            model.load_state_dict(torch.load(args.resume, map_location=device))
            print(f'Resumed from checkpoint: {args.resume}')
        else:
            print(f'WARNING: Checkpoint not found at {args.resume}, starting fresh')

    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )
    criterion = nn.BCELoss()

    # ── Training loop ──
    train_losses, train_accs = [], []
    test_losses,  test_accs  = [], []
    lr_history = []
    best_test_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        current_lr = optimizer.param_groups[0]['lr']
        lr_history.append(current_lr)

        tr_loss, tr_acc = train_one_epoch(model, device, train_loader,
                                          optimizer, criterion, epoch)
        te_loss, te_acc = evaluate(model, device, test_loader, criterion)

        train_losses.append(tr_loss)
        train_accs.append(tr_acc)
        test_losses.append(te_loss)
        test_accs.append(te_acc)

        scheduler.step(te_acc)

        print(f'Epoch {epoch:3d}/{args.epochs} | LR={current_lr:.2e} | '
              f'Train loss={tr_loss:.4f} acc={tr_acc:.2f}% | '
              f'Test loss={te_loss:.4f} acc={te_acc:.2f}%')

        if te_acc > best_test_acc:
            best_test_acc = te_acc
            torch.save(model.state_dict(),
                       os.path.join(output_dir, 'best_model.pth'))
            print(f'  -> New best: {best_test_acc:.2f}%')

        # Save final model every epoch (overwrite)
        torch.save(model.state_dict(),
                   os.path.join(output_dir, 'final_model.pth'))

        # Crash-safe log
        log = {
            'architecture': args.arch,
            'experiment': 'segmentation_maps',
            'real_dir': args.real_dir,
            'fake_dir': args.fake_dir,
            'n_per_class': n,
            'resumed_from': args.resume,
            'epochs_completed': epoch,
            'total_epochs': args.epochs,
            'seed': args.seed,
            'initial_lr': args.lr,
            'weight_decay': args.weight_decay,
            'lr_history': lr_history,
            'train_losses': train_losses,
            'train_accuracies': train_accs,
            'test_losses': test_losses,
            'test_accuracies': test_accs,
            'best_test_accuracy': best_test_acc,
            'final_train_accuracy': tr_acc,
            'final_test_accuracy': te_acc,
        }
        with open(os.path.join(output_dir, 'training_log.json'), 'w') as f:
            json.dump(log, f, indent=2)

    print(f'\nDone. Best test accuracy: {best_test_acc:.2f}%')
    print(f'Results saved to: {output_dir}')


if __name__ == '__main__':
    main()
