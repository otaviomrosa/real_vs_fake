"""Experiment 7: Raw FFHQ real vs. StyleGAN2 fake — ablation study on original RGB images.

Architectures: resnet, densenet, vit (all adapted to 3-channel 256x256 input).
Ablations:     crop, downsample, pca, blur.

All ablations resize back to 256x256 so all architectures receive consistent input.
Image loading uses PIL -> torch.frombuffer (bytearray) in HWC order, permuted to CHW.
PCA uses torch.pca_lowrank on a subsample of training images (no sklearn needed).

Resume from checkpoint:
    python train_ablation7.py ... --resume path/to/final_model.pth
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
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).parent.parent  # experiment7/ -> experiment6 siblings
sys.path.insert(0, str(SCRIPT_DIR / 'experiment6' / 'canny'))
sys.path.insert(0, str(SCRIPT_DIR / 'experiment6' / 'face-parsing'))
from resnet_canny import ResNetCanny
from densenet_segm import DenseNetSegm

IMG_SIZE = 256
IN_CHANNELS = 3
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg'}


# ── Path collection ──────────────────────────────────────────────────────────

def collect_paths(directory, nested=False):
    d = Path(directory)
    if nested:
        paths = []
        for sub in sorted(d.iterdir()):
            if sub.is_dir():
                paths.extend(sorted(
                    p for p in sub.iterdir()
                    if p.suffix.lower() in IMAGE_EXTENSIONS
                ))
        return paths
    return sorted(p for p in d.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)


# ── Image loading ─────────────────────────────────────────────────────────────

def pil_to_tensor(pil_img):
    """256x256 RGB PIL -> normalized (3, 256, 256) float32 tensor. No numpy."""
    buf = bytearray(pil_img.tobytes())  # HWC bytes: H*W*3
    t = torch.frombuffer(buf, dtype=torch.uint8).float().div_(255.0)
    return t.view(IMG_SIZE, IMG_SIZE, 3).permute(2, 0, 1).sub_(0.5).div_(0.5).contiguous().clone()


def load_rgb(path):
    img = Image.open(path).convert('RGB')
    if img.size != (IMG_SIZE, IMG_SIZE):
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    return img


# ── PCA fitting ──────────────────────────────────────────────────────────────

def fit_pca(train_paths, n_components, n_fit=5000):
    """Fit PCA via torch.pca_lowrank on a subsample of training images.

    RGB images are flattened to HWC vectors (196608-dim).
    Returns (V, mean): V shape (196608, n_components), mean shape (196608,).
    Reconstruction: recon = (centered @ V) @ V.T + mean -> reshape HWC -> permute CHW.
    """
    n_fit = min(n_fit, len(train_paths))
    if n_components >= n_fit:
        print(f'WARNING: n_components ({n_components}) >= n_fit ({n_fit}), clamping.')
        n_components = n_fit - 1

    sample = train_paths[:n_fit]
    p = IMG_SIZE * IMG_SIZE * IN_CHANNELS  # 196608

    print(f'Loading {n_fit} images for PCA (n_components={n_components}, dim={p})...')
    rows = []
    for path in tqdm(sample, desc='PCA load', leave=False):
        img = load_rgb(path)
        buf = bytearray(img.tobytes())
        t = torch.frombuffer(buf, dtype=torch.uint8).float().div_(255.0).clone()
        rows.append(t)  # (196608,) HWC flat

    X = torch.stack(rows)  # (n_fit, 196608)
    print(f'  Fitting pca_lowrank on {X.shape}...')
    _, _, V = torch.pca_lowrank(X, q=n_components, center=True, niter=4)
    mean = X.mean(dim=0)
    print(f'  Done. V shape: {V.shape}')
    return V.float(), mean.float()


# ── Dataset ──────────────────────────────────────────────────────────────────

class AblationDataset(Dataset):
    def __init__(self, paths, labels, ablation, param,
                 pca_V=None, pca_mean=None):
        self.samples = list(zip(paths, labels))
        self.ablation = ablation
        self.param = param
        self.pca_V = pca_V        # (196608, n_comp) or None
        self.pca_mean = pca_mean  # (196608,) or None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        if self.ablation == 'pca':
            img = load_rgb(path)
            buf = bytearray(img.tobytes())
            flat = torch.frombuffer(buf, dtype=torch.uint8).float().div_(255.0).clone()
            centered = flat - self.pca_mean                    # (196608,)
            proj = centered @ self.pca_V                       # (n_comp,)
            recon = proj @ self.pca_V.T + self.pca_mean        # (196608,)
            tensor = (recon.clamp(0, 1).sub_(0.5).div_(0.5)
                      .view(IMG_SIZE, IMG_SIZE, 3).permute(2, 0, 1).contiguous().clone())

        elif self.ablation == 'blur':
            img = load_rgb(path)
            img = img.filter(ImageFilter.GaussianBlur(radius=float(self.param)))
            tensor = pil_to_tensor(img)

        elif self.ablation == 'crop':
            img = load_rgb(path)
            size = int(self.param)
            if size < IMG_SIZE:
                left = (IMG_SIZE - size) // 2
                img = img.crop((left, left, left + size, left + size))
                img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
            tensor = pil_to_tensor(img)

        elif self.ablation == 'downsample':
            img = load_rgb(path)
            size = int(self.param)
            if size < IMG_SIZE:
                img = img.resize((size, size), Image.BILINEAR)
                img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
            tensor = pil_to_tensor(img)

        else:
            raise ValueError(f'Unknown ablation: {self.ablation}')

        return tensor, torch.tensor(label, dtype=torch.float32)


# ── Model ────────────────────────────────────────────────────────────────────

def build_model(arch):
    if arch == 'resnet':
        m = ResNetCanny()
        # Replace 1-channel stem conv with 3-channel
        m.stem[0] = nn.Conv2d(IN_CHANNELS, 16, 3, stride=1, padding=1, bias=False)
        return m
    if arch == 'densenet':
        m = DenseNetSegm()
        m.stem[0] = nn.Conv2d(IN_CHANNELS, 32, 3, stride=2, padding=1, bias=False)
        return m
    if arch == 'vit':
        from vit_pytorch import ViT

        class ViTClassifier(nn.Module):
            def __init__(self):
                super().__init__()
                self.vit = ViT(
                    image_size=IMG_SIZE, patch_size=16, num_classes=1,
                    dim=512, depth=6, heads=8, mlp_dim=1024,
                    channels=IN_CHANNELS, dropout=0.1, emb_dropout=0.1,
                )
            def forward(self, x):
                return torch.sigmoid(self.vit(x))

        return ViTClassifier()
    raise ValueError(f'Unknown arch: {arch}')


# ── Train / eval ─────────────────────────────────────────────────────────────

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


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--arch', required=True, choices=['resnet', 'densenet', 'vit'])
    p.add_argument('--ablation', required=True, choices=['crop', 'downsample', 'pca', 'blur'])
    p.add_argument('--param', required=True, type=float)
    p.add_argument('--real-dir', default='/data/omrosa/FFHQ/reals_256')
    p.add_argument('--fake-dir', default='/data/omrosa/FFHQ/fakes_256')
    p.add_argument('--output-base',
                   default='/home/o/omrosa/research/experiment7/ablations/results')
    p.add_argument('--resume', type=str, default=None)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--lr', type=float, default=None)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--num-workers', type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()

    if args.lr is None:
        args.lr = 1e-4 if args.arch == 'vit' else 1e-3
    if args.num_workers is None:
        args.num_workers = 0 if args.arch == 'vit' else 4

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    param_str = str(int(args.param)) if args.param == int(args.param) else str(args.param)
    job_name = f'{args.arch}_{args.ablation}_{param_str}'
    output_dir = os.path.join(args.output_base, args.arch, f'{args.ablation}_{param_str}')
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Job:    {job_name}')
    print(f'Device: {device} | LR: {args.lr} | WD: {args.weight_decay} | Workers: {args.num_workers}')
    print(f'Output: {output_dir}')

    # ── Collect paths ──
    print('Collecting image paths...')
    real_paths = collect_paths(args.real_dir, nested=True)
    fake_paths = collect_paths(args.fake_dir, nested=False)
    print(f'  Real: {len(real_paths):,} | Fake: {len(fake_paths):,}')
    if not real_paths or not fake_paths:
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

    train_paths_all = real_train + fake_train
    train_labels    = [1.0] * len(real_train) + [0.0] * len(fake_train)
    test_paths_all  = real_test + fake_test
    test_labels     = [1.0] * len(real_test)  + [0.0] * len(fake_test)
    print(f'  Train: {len(train_paths_all):,} | Test: {len(test_paths_all):,}')

    # ── PCA ──
    pca_V = pca_mean = None
    if args.ablation == 'pca':
        pca_cache = os.path.join(output_dir, 'pca_model.pt')
        if os.path.exists(pca_cache):
            print(f'Loading cached PCA: {pca_cache}')
            ckpt = torch.load(pca_cache, map_location='cpu')
            pca_V, pca_mean = ckpt['V'], ckpt['mean']
        else:
            pca_V, pca_mean = fit_pca(real_train, int(args.param), n_fit=20000)
            torch.save({'V': pca_V, 'mean': pca_mean}, pca_cache)
            print(f'PCA cached: {pca_cache}')

    # ── Datasets & loaders ──
    train_ds = AblationDataset(train_paths_all, train_labels,
                               args.ablation, args.param, pca_V, pca_mean)
    test_ds  = AblationDataset(test_paths_all,  test_labels,
                               args.ablation, args.param, pca_V, pca_mean)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    # ── Model ──
    model = build_model(args.arch).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Params: {n_params:,}')

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
            torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pth'))
            print(f'  -> New best: {best_test_acc:.2f}%')

        torch.save(model.state_dict(), os.path.join(output_dir, 'final_model.pth'))

        log = {
            'job': job_name,
            'experiment': 'experiment7',
            'arch': args.arch,
            'ablation': args.ablation,
            'param': args.param,
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
        }
        with open(os.path.join(output_dir, 'training_log.json'), 'w') as f:
            json.dump(log, f, indent=2)

    print(f'\nDone. Best test accuracy: {best_test_acc:.2f}%')
    print(f'Results: {output_dir}')


if __name__ == '__main__':
    main()
