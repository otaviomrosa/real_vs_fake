"""Generate a 2x2 composite of 8x8 grids showing originals and edge maps.

Layout:
  Top-left:     Real original images (64 random samples)
  Top-right:    Canny edge maps of those same real images
  Bottom-left:  Fake original images (64 random samples)
  Bottom-right: Canny edge maps of those same fake images

The edge map shown is the one corresponding to the same image in the original
grid (matched by filename stem), so the pairs line up visually.

Usage:
    python make_canny_grid.py
    python make_canny_grid.py --seed 123 --output-dir ./grids
"""

import argparse
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg'}


def collect_paths(directory, nested=False):
    """Return a dict of {stem: path} for all images in directory."""
    d = Path(directory)
    result = {}
    if nested:
        for sub in sorted(d.iterdir()):
            if sub.is_dir():
                for p in sub.iterdir():
                    if p.suffix.lower() in IMAGE_EXTENSIONS:
                        result[p.stem] = p
    else:
        for p in d.iterdir():
            if p.suffix.lower() in IMAGE_EXTENSIONS:
                result[p.stem] = p
    return result


def make_grid(paths, grid_size=8, img_size=128, padding=4, grayscale=False):
    """Arrange a list of paths into a grid. Returns RGB PIL Image."""
    mode = 'L' if grayscale else 'RGB'
    bg = 255
    cell = img_size + padding
    canvas_size = grid_size * cell + padding
    grid = Image.new(mode, (canvas_size, canvas_size), bg)

    for idx, path in enumerate(paths[:grid_size * grid_size]):
        row = idx // grid_size
        col = idx % grid_size
        x = padding + col * cell
        y = padding + row * cell
        img = Image.open(path).convert(mode).resize(
            (img_size, img_size), Image.BILINEAR if not grayscale else Image.NEAREST
        )
        grid.paste(img, (x, y))

    return grid.convert('RGB')  # always return RGB for compositing


def add_label(img, text, font_size=20):
    """Add a label banner at the top of an image."""
    banner_h = font_size + 10
    new_img = Image.new('RGB', (img.width, img.height + banner_h), (30, 30, 30))
    new_img.paste(img, (0, banner_h))
    draw = ImageDraw.Draw(new_img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()
    draw.text((10, 5), text, fill=(255, 255, 255), font=font)
    return new_img


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--real-orig-dir', default='/general/omrosa/FFHQ/reals_256',
                   help='Original real images (nested subfolders)')
    p.add_argument('--fake-orig-dir', default='/general/omrosa/FFHQ/fakes_256',
                   help='Original fake images (flat)')
    p.add_argument('--real-edge-dir', default='/general/omrosa/FFHQ/canny_reals',
                   help='Real edge maps (nested subfolders)')
    p.add_argument('--fake-edge-dir', default='/general/omrosa/FFHQ/canny_fakes',
                   help='Fake edge maps (flat)')
    p.add_argument('--output-dir', default='./grids')
    p.add_argument('--img-size', type=int, default=128,
                   help='Size of each image cell in the grid')
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print('Collecting paths...')
    real_orig = collect_paths(args.real_orig_dir, nested=True)
    fake_orig = collect_paths(args.fake_orig_dir, nested=False)
    real_edge = collect_paths(args.real_edge_dir, nested=True)
    fake_edge = collect_paths(args.fake_edge_dir, nested=False)

    print(f'  Real originals: {len(real_orig)} | Real edges: {len(real_edge)}')
    print(f'  Fake originals: {len(fake_orig)} | Fake edges: {len(fake_edge)}')

    # Find stems that exist in both original and edge for each class
    real_common = [s for s in real_orig if s in real_edge]
    fake_common = [s for s in fake_orig if s in fake_edge]

    print(f'  Matched real pairs: {len(real_common)}')
    print(f'  Matched fake pairs: {len(fake_common)}')

    # Randomly sample 64 matched pairs from each class
    real_sample = random.sample(real_common, min(64, len(real_common)))
    fake_sample = random.sample(fake_common, min(64, len(fake_common)))

    # Build path lists in the same order so grids align
    real_orig_paths = [real_orig[s] for s in real_sample]
    real_edge_paths = [real_edge[s] for s in real_sample]
    fake_orig_paths = [fake_orig[s] for s in fake_sample]
    fake_edge_paths = [fake_edge[s] for s in fake_sample]

    print('Building grids...')
    g_real_orig = make_grid(real_orig_paths, img_size=args.img_size, grayscale=False)
    g_real_edge = make_grid(real_edge_paths, img_size=args.img_size, grayscale=True)
    g_fake_orig = make_grid(fake_orig_paths, img_size=args.img_size, grayscale=False)
    g_fake_edge = make_grid(fake_edge_paths, img_size=args.img_size, grayscale=True)

    # Add labels
    g_real_orig = add_label(g_real_orig, "Real Images")
    g_real_edge = add_label(g_real_edge, "Real - Canny Edge Maps")
    g_fake_orig = add_label(g_fake_orig, "Fake Images (StyleGAN2)")
    g_fake_edge = add_label(g_fake_edge, "Fake - Canny Edge Maps")

    # Composite: 2x2 layout with a gap between quadrants
    gap = 8
    W = g_real_orig.width
    H = g_real_orig.height
    composite = Image.new('RGB',
                          (W * 2 + gap, H * 2 + gap),
                          (50, 50, 50))  # dark gray gap

    composite.paste(g_real_orig, (0, 0))
    composite.paste(g_real_edge, (W + gap, 0))
    composite.paste(g_fake_orig, (0, H + gap))
    composite.paste(g_fake_edge, (W + gap, H + gap))

    out_path = str(Path(args.output_dir) / 'canny_comparison_grid.png')
    composite.save(out_path)
    print(f'Saved: {out_path}')
    print('Done.')


if __name__ == '__main__':
    main()
