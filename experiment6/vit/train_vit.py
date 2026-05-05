"""Experiment 6: ViT classifier for Canny edge maps and segmentation maps.

Uses lucidrains/vit-pytorch for real vs. fake binary classification.
Bypasses numpy entirely (PIL -> bytes -> torch tensor) due to environment
conflicts on GAIVI.

Usage:
    python train_vit.py --experiment canny \
        --real-dir /data/omrosa/FFHQ/canny_reals \
        --fake-dir /data/omrosa/FFHQ/canny_fakes

    python train_vit.py --experiment segmentation \
        --real-dir /data/omrosa/FFHQ/segm_reals/resnet18 \
        --fake-dir /data/omrosa/FFHQ/segm_fakes/resnet18
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
from PIL import Image
from tqdm import tqdm
from vit_pytorch import ViT

IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}
IMG_SIZE = 256


# ── Dataset ──────────────────────────────────────────────────────────────────

class ImageDataset(Dataset):
    """Loads 1-channel grayscale PNGs as torch tensors directly from PIL bytes
    (no numpy involved). Normalizes to [-1, 1].
    Labels: real=1, fake=0
    """

    def __init__(self, real_paths, fake_paths):
        self.paths = [(p, 1.0) for p in real_paths] + [(p, 0.0) for p in fake_paths]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path, label = self.paths[idx]
        # PIL -> bytes -> torch (no numpy at all)
        img = Image.open(path).convert('L').resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        t = torch.frombuffer(img.tobytes(), dtype=torch.uint8)
        t = t.float().div_(255.0).sub_(0.5).div_(0.5)  # normalize to [-1, 1]
        tensor = t.view(1, IMG_SIZE, IMG_SIZE).clone()  # clone to avoid buffer issues
        return tensor, torch.tensor(label, dtype=torch.float32)


def collect_paths(directory, nested=False):
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


def collect_raw_paths(directory):
    """Only *_raw.png files for segmentation maps."""
    return sorted(p for p in Path(directory).iterdir()
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
    p.add_argument('--experiment', required=True, choices=['canny', 'segmentation'])
    p.add_argument('--real-dir', required=True)
    p.add_argument('--fake-dir', required=True)
    p.add_argument('--output-base', default='/home/o/omrosa/research/experiment6/vit/results')
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--num-workers', type=int, default=0)  # default 0 to avoid env issues
    p.add_argument('--patch-size', type=int, default=16)
    p.add_argument('--dim', type=int, default=512)
    p.add_argument('--depth', type=int, default=6)
    p.add_argument('--heads', type=int, default=8)
    p.add_argument('--mlp-dim', type=int, default=1024)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--resume', type=str, default=None,
               help='Path to checkpoint to resume from')
    p.add_argument('--emb-dropout', type=float, default=0.1)
    return p.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    output_dir = os.path.join(args.output_base, f'vit_{args.experiment}')
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    print(f'Experiment: {args.experiment} | LR: {args.lr} | WD: {args.weight_decay}')

    # ── Collect paths ──
    print('Collecting paths...')
    if args.experiment == 'canny':
        real_paths = collect_paths(args.real_dir, nested=True)
        fake_paths = collect_paths(args.fake_dir, nested=False)
    else:
        real_paths = collect_raw_paths(args.real_dir)
        fake_paths = collect_raw_paths(args.fake_dir)

    print(f'  Real: {len(real_paths)} | Fake: {len(fake_paths)}')

    if len(real_paths) == 0 or len(fake_paths) == 0:
        print('ERROR: No images found.')
        sys.exit(1)

    n = min(len(real_paths), len(fake_paths))
    rng = random.Random(args.seed)
    real_paths = rng.sample(real_paths, n)
    fake_paths = rng.sample(fake_paths, n)

    rng2 = random.Random(args.seed + 1)
    rng2.shuffle(real_paths)
    rng2.shuffle(fake_paths)
    n_test  = int(0.2 * n)
    n_train = n - n_test

    real_train, real_test = real_paths[:n_train], real_paths[n_train:]
    fake_train, fake_test = fake_paths[:n_train], fake_paths[n_train:]
    print(f'  Train: {2*n_train} | Test: {2*n_test}')

    train_ds = ImageDataset(real_train, fake_train)
    test_ds  = ImageDataset(real_test,  fake_test)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    # ── ViT Model ──
    vit = ViT(
        image_size   = IMG_SIZE,
        patch_size   = args.patch_size,
        num_classes  = 1,
        dim          = args.dim,
        depth        = args.depth,
        heads        = args.heads,
        mlp_dim      = args.mlp_dim,
        channels     = 1,
        dropout      = args.dropout,
        emb_dropout  = args.emb_dropout,
    )

    class ViTClassifier(nn.Module):
        def __init__(self, vit):
            super().__init__()
            self.vit = vit
        def forward(self, x):
            return torch.sigmoid(self.vit(x))

    model = ViTClassifier(vit).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model: ViT | patch={args.patch_size} dim={args.dim} depth={args.depth} '
          f'heads={args.heads} | Parameters: {n_params:,}')

    if args.resume:
    	if os.path.exists(args.resume):
            model.load_state_dict(torch.load(args.resume, map_location=device))
            print(f'Resumed from: {args.resume}')
    	else:
            print(f'WARNING: checkpoint not found at {args.resume}, starting fresh')

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    criterion = nn.BCELoss()

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
            torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pth'))
            print(f'  -> New best: {best_test_acc:.2f}%')

        torch.save(model.state_dict(), os.path.join(output_dir, 'final_model.pth'))

        log = {
            'architecture': 'vit',
            'experiment': args.experiment,
            'real_dir': args.real_dir,
            'fake_dir': args.fake_dir,
            'n_per_class': n,
            'epochs_completed': epoch,
            'total_epochs': args.epochs,
            'seed': args.seed,
            'initial_lr': args.lr,
            'weight_decay': args.weight_decay,
            'vit_config': {
                'patch_size': args.patch_size, 'dim': args.dim, 'depth': args.depth,
                'heads': args.heads, 'mlp_dim': args.mlp_dim,
                'dropout': args.dropout, 'emb_dropout': args.emb_dropout,
                'n_params': n_params,
            },
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
