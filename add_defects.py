"""Add more realistic-looking defects to SEM images in a folder.

Usage:
    python add_defects.py --dir output/sem

Defects layered:
    - Particles with soft shadows
    - Smudges/stains (low-frequency blotches)
    - Scratches (thin, aliased lines)
    - Broken lines / cracks (segmented, offset strokes)
    - Charging glow bands
    - Banding and shot noise

Images are overwritten in place; use version control if you need originals.
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageFilter, ImageDraw


def parse_args():
    p = argparse.ArgumentParser(description="Add realistic defects to SEM images in-place.")
    p.add_argument("--dir", type=Path, default=Path("output/sem"), help="Directory containing SEM PNG images.")
    p.add_argument("--seed", type=int, default=None, help="Optional RNG seed for reproducibility.")
    p.add_argument(
        "--fraction",
        type=float,
        default=0.1,
        help="Fraction of images to augment (0–1]. Defaults to 0.1 (10%).",
    )
    return p.parse_args()


def iter_images(folder: Path) -> Iterable[Path]:
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"):
        for path in folder.glob(ext):
            if path.is_file():
                yield path


def add_particles(arr: np.ndarray, count_range=(4, 10)):
    """Add particles with soft shadows and slight edge darkening."""
    h, w = arr.shape
    count = random.randint(*count_range)
    for _ in range(count):
        cx = random.randint(0, w - 1)
        cy = random.randint(0, h - 1)
        sigma = random.uniform(1.8, 4.5)
        size = int(max(8, sigma * 7))
        x0 = max(0, cx - size // 2)
        y0 = max(0, cy - size // 2)
        x1 = min(w, cx + size // 2)
        y1 = min(h, cy + size // 2)
        xs = np.linspace(-(x1 - x0) / 2, (x1 - x0) / 2, x1 - x0)
        ys = np.linspace(-(y1 - y0) / 2, (y1 - y0) / 2, y1 - y0)
        gx, gy = np.meshgrid(xs, ys)
        gauss = np.exp(-(gx ** 2 + gy ** 2) / (2 * sigma ** 2))
        edge = np.exp(-(gx ** 2 + gy ** 2) / (2 * (sigma * 0.6) ** 2))
        shadow = np.exp(-((gx + sigma) ** 2 + (gy + sigma) ** 2) / (2 * (sigma * 1.4) ** 2))
        strength = random.uniform(30, 70) * random.choice([-1, 1])
        patch = gauss * strength - edge * (abs(strength) * 0.15) - shadow * (abs(strength) * 0.1)
        arr[y0:y1, x0:x1] = np.clip(arr[y0:y1, x0:x1] + patch, 0, 255)


def add_scratches(img: Image.Image, count_range=(0, 1)):
    draw = ImageDraw.Draw(img)
    w, h = img.size
    count = random.randint(*count_range)
    for _ in range(count):
        x0 = random.randint(0, w - 1)
        y0 = random.randint(0, h - 1)
        x1 = random.randint(0, w - 1)
        y1 = random.randint(0, h - 1)
        # shorten scratch length by biasing endpoints closer
        x1 = int(x0 + (x1 - x0) * random.uniform(0.1, 0.4))
        y1 = int(y0 + (y1 - y0) * random.uniform(0.1, 0.4))
        width = random.uniform(0.3, 0.9)
        intensity = random.randint(20, 60) * random.choice([-1, 1])
        draw.line((x0, y0, x1, y1), fill=int(128 + intensity), width=math.ceil(width))
    return img


def add_broken_lines(img: Image.Image, count_range=(0, 2)):
    """Add cracked/broken lines with slight misalignment and gaps."""
    draw = ImageDraw.Draw(img)
    w, h = img.size
    count = random.randint(*count_range)
    for _ in range(count):
        segs = random.randint(2, 4)
        x = random.uniform(0, w)
        y = random.uniform(0, h)
        ang = random.uniform(0, 2 * math.pi)
        step = random.uniform(6, 15)
        width = random.uniform(0.4, 1.0)
        intensity = random.randint(20, 60) * random.choice([-1, 1])
        for _ in range(segs):
            dx = step * math.cos(ang) + random.uniform(-4, 4)
            dy = step * math.sin(ang) + random.uniform(-4, 4)
            nx = min(max(x + dx, 0), w - 1)
            ny = min(max(y + dy, 0), h - 1)
            draw.line((x, y, nx, ny), fill=int(128 + intensity), width=math.ceil(width))
            # introduce small gaps / offsets
            x = nx + random.uniform(-3, 3)
            y = ny + random.uniform(-3, 3)
            if random.random() < 0.3:
                ang += random.uniform(-0.4, 0.4)
    return img


def add_stains(arr: np.ndarray, count_range=(1, 3)):
    """Low-frequency blotches / charging stains."""
    h, w = arr.shape
    count = random.randint(*count_range)
    y, x = np.ogrid[:h, :w]
    for _ in range(count):
        cx = random.randint(0, w - 1)
        cy = random.randint(0, h - 1)
        radius = random.uniform(10, 22)
        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        mask = np.exp(-(dist ** 2) / (2 * (radius ** 2)))
        strength = random.uniform(8, 22) * random.choice([-1, 1])
        arr[:] = np.clip(arr + mask * strength, 0, 255)


def add_noise(arr: np.ndarray, sigma=3.0):
    noise = np.random.normal(0, sigma, size=arr.shape)
    return np.clip(arr + noise, 0, 255)


def add_banding(arr: np.ndarray, amp_range=(4, 10)):
    """Subtle horizontal banding / drift."""
    h, w = arr.shape
    amp = random.uniform(*amp_range) * random.choice([-1, 1])
    freq = random.uniform(1 / 200, 1 / 80)
    phase = random.uniform(0, 2 * math.pi)
    y = np.arange(h)
    band = amp * np.sin(2 * math.pi * freq * y + phase)
    arr[:] = np.clip(arr + band[:, None], 0, 255)


def add_charging_glow(arr: np.ndarray):
    """Add a soft gradient glow to mimic charging."""
    h, w = arr.shape
    cx = random.uniform(0.2, 0.8) * w
    cy = random.uniform(0.2, 0.8) * h
    y, x = np.ogrid[:h, :w]
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    radius = random.uniform(0.3, 0.6) * min(w, h)
    mask = np.exp(-(dist ** 2) / (2 * (radius ** 2)))
    strength = random.uniform(5, 12) * random.choice([-1, 1])
    arr[:] = np.clip(arr + mask * strength, 0, 255)


def add_speckle(arr: np.ndarray, amount=0.002):
    """Sparse salt/pepper speckles."""
    h, w = arr.shape
    n = int(h * w * amount)
    ys = np.random.randint(0, h, size=n)
    xs = np.random.randint(0, w, size=n)
    vals = np.random.choice([0, 255], size=n)
    arr[ys, xs] = vals


def process_image(path: Path):
    img = Image.open(path).convert("L")
    arr = np.array(img, dtype=np.float32)

    add_particles(arr)
    add_stains(arr)
    add_banding(arr)
    add_charging_glow(arr)
    arr = add_noise(arr, sigma=3.0)
    add_speckle(arr, amount=0.0015)

    img = Image.fromarray(arr.astype(np.uint8), mode="L")
    img = add_scratches(img)
    img = add_broken_lines(img)
    img = img.filter(ImageFilter.GaussianBlur(radius=0.2))
    img.save(path)
    print(f"Defects added to {path}")


def main():
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    folder = args.dir
    if not folder.exists():
        raise SystemExit(f"Directory not found: {folder}")

    images = list(iter_images(folder))
    if not images:
        raise SystemExit(f"No images found in {folder}")

    frac = max(0.0, min(1.0, args.fraction))
    k = max(1, int(math.ceil(len(images) * frac))) if images else 0
    chosen = set(random.sample(images, k))

    for img_path in images:
        if img_path in chosen:
            process_image(img_path)
        else:
            print(f"Skipped {img_path}")


if __name__ == "__main__":
    main()
