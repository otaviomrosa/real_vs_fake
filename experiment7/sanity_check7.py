"""Sanity check for Experiment 7: two random halves of real FFHQ RGB images.

Expected: ~50% test accuracy. Verifies no dataset artifacts.
Uses DenseNet with 3-channel input.
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SCRIPT_DIR / 'experiment6' / 'face-parsing'))
from densenet_segm import DenseNetSegm

IMG_SIZE = 256
IN_CHANNELS = 3
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg'}


def collect_paths(directory):
    d = Path(directory)
    paths = []
    for sub in sorted(d.iterdir()):
        if sub.is_dir():
            paths.extend(sorted(
                p for p in sub.iterdir()
                if p.suffix.lower() in IMAGE_EXTENSIONS
            ))
    return paths


def pil_to_tensor(pil_img):
    buf = bytearray(pil_img.tobytes())
    t = torch.frombuffer(buf, dtype=torch.uint8).float().div_(255.0)
    return t.view(IMG_SIZE, IMG_SIZE, 3).permute(2, 0, 1).sub_(0.5).div_(0.5).contiguous().clone()


class HalfDataset(Dataset):
    def __init__(self, paths, labels):
        self.samples = list(zip(paths, labels))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB').resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        return pil_to_tensor(img), torch.tensor(label, dtype=torch.float32)


def run_epoch(model, loader, device, criterion, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = correct = total = 0
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            if training:
                optimizer.zero_grad()
            preds = model(imgs).squeeze(1)
            loss = criterion(preds, labels)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            correct += ((preds >= 0.5).float() == labels).sum().item()
            total += imgs.size(0)
    return total_loss / total, 100.0 * correct / total


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--real-dir', default='/data/omrosa/FFHQ/reals_256')
    p.add_argument('--output-dir',
                   default='/home/o/omrosa/research/experiment7/ablations/results/sanity_check')
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--resume', type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Sanity check (Exp 7) | Device: {device} | Expected: ~50%')

    all_paths = collect_paths(args.real_dir)
    print(f'Found: {len(all_paths):,} real images')

    rng = random.Random(args.seed)
    rng.shuffle(all_paths)
    n = len(all_paths)
    half = n // 2
    all_labeled = [(p, 1.0) for p in all_paths[:half]] + [(p, 0.0) for p in all_paths[half:2*half]]
    rng.shuffle(all_labeled)

    n_total = len(all_labeled)
    n_test  = int(0.2 * n_total)
    train_data, test_data = all_labeled[n_test:], all_labeled[:n_test]
    print(f'Train: {len(train_data):,} | Test: {len(test_data):,}')

    train_ds = HalfDataset([p for p, _ in train_data], [l for _, l in train_data])
    test_ds  = HalfDataset([p for p, _ in test_data],  [l for _, l in test_data])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    model = DenseNetSegm().to(device)
    model.stem[0] = nn.Conv2d(IN_CHANNELS, 32, 3, stride=2, padding=1, bias=False)
    model = model.to(device)

    if args.resume and os.path.exists(args.resume):
        model.load_state_dict(torch.load(args.resume, map_location=device))
        print(f'Resumed from: {args.resume}')

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    criterion = nn.BCELoss()

    train_losses, train_accs, test_losses, test_accs, lr_history = [], [], [], [], []
    best_test_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        current_lr = optimizer.param_groups[0]['lr']
        lr_history.append(current_lr)

        tr_loss, tr_acc = run_epoch(model, train_loader, device, criterion, optimizer)
        te_loss, te_acc = run_epoch(model, test_loader,  device, criterion)

        train_losses.append(tr_loss)
        train_accs.append(tr_acc)
        test_losses.append(te_loss)
        test_accs.append(te_acc)
        scheduler.step(te_acc)

        print(f'Epoch {epoch:3d}/{args.epochs} | LR={current_lr:.2e} | '
              f'Train {tr_acc:.2f}% | Test {te_acc:.2f}%')

        if te_acc > best_test_acc:
            best_test_acc = te_acc
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_model.pth'))

        torch.save(model.state_dict(), os.path.join(args.output_dir, 'final_model.pth'))

        log = {
            'experiment': 'experiment7_sanity_check',
            'description': 'Real FFHQ RGB split in half — expects ~50% test accuracy',
            'real_dir': args.real_dir,
            'n_total': n_total,
            'resumed_from': args.resume,
            'epochs_completed': epoch,
            'total_epochs': args.epochs,
            'seed': args.seed,
            'lr_history': lr_history,
            'train_losses': train_losses,
            'train_accuracies': train_accs,
            'test_losses': test_losses,
            'test_accuracies': test_accs,
            'best_test_accuracy': best_test_acc,
        }
        with open(os.path.join(args.output_dir, 'training_log.json'), 'w') as f:
            json.dump(log, f, indent=2)

    print(f'\nDone. Best: {best_test_acc:.2f}% (expected ~50%)')


if __name__ == '__main__':
    main()
