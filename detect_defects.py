"""Compare CAD and SEM images with alignment and SSIM to highlight defects.

Usage:
    python detect_defects.py --config config_detect.json

Steps:
    - Load matching CAD/SEM grayscale images by filename.
    - Align via phase correlation (translation).
    - Compute SSIM map; convert to defect score (1 - SSIM).
    - Threshold and clean with morphology to produce a defect mask.
    - Save mask and overlay images to out_dir.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image, ImageOps
from scipy.ndimage import shift as nd_shift, binary_opening, binary_closing
from skimage.metrics import structural_similarity as ssim
from skimage.registration import phase_cross_correlation

CONFIG_DEFAULTS = {
    "cad_dir": "output/cad",
    "sem_dir": "output/sem",
    "out_dir": "output/diff",
    "threshold": 0.35,
    "std_factor": 0.5,
    "min_size": 8,
}


def parse_args():
    p = argparse.ArgumentParser(description="Detect defects by comparing CAD vs SEM images.")
    p.add_argument("--config", type=Path, default=Path("config_detect.json"), help="Config JSON file.")
    return p.parse_args()


def load_config(path: Path) -> dict:
    cfg = dict(CONFIG_DEFAULTS)
    if path.exists():
        with path.open() as f:
            data = json.load(f)
        for k in CONFIG_DEFAULTS:
            if k in data:
                cfg[k] = data[k]
    # coerce
    cfg["cad_dir"] = Path(cfg["cad_dir"])
    cfg["sem_dir"] = Path(cfg["sem_dir"])
    cfg["out_dir"] = Path(cfg["out_dir"])
    cfg["threshold"] = float(cfg["threshold"])
    cfg["min_size"] = int(cfg["min_size"])
    cfg["std_factor"] = float(cfg.get("std_factor", CONFIG_DEFAULTS["std_factor"]))
    return cfg


def load_gray(path: Path) -> np.ndarray:
    img = Image.open(path)
    img = ImageOps.grayscale(img)
    return np.array(img, dtype=np.float32)


def align_sem_to_cad(cad: np.ndarray, sem: np.ndarray) -> Tuple[np.ndarray, Tuple[float, float]]:
    # Phase correlation for translation.
    shift_estimate, _, _ = phase_cross_correlation(cad, sem, upsample_factor=10)
    aligned = nd_shift(sem, shift=shift_estimate, mode="nearest", order=1)
    return aligned, (float(shift_estimate[1]), float(shift_estimate[0]))  # (dx, dy)


def defect_map(cad: np.ndarray, sem: np.ndarray) -> np.ndarray:
    # Normalize to 0-1
    c = (cad - cad.min()) / (cad.max() - cad.min() + 1e-6)
    s = (sem - sem.min()) / (sem.max() - sem.min() + 1e-6)
    _, full = ssim(c, s, full=True, data_range=1.0)
    return 1.0 - full  # higher = more different


def postprocess_mask(mask: np.ndarray, min_size: int) -> np.ndarray:
    # Simple opening/closing to reduce speckle, then remove small blobs.
    m = mask.astype(bool)
    m = binary_opening(m, structure=np.ones((2, 2), dtype=bool))
    m = binary_closing(m, structure=np.ones((2, 2), dtype=bool))
    # Remove small regions
    labeled = np.zeros_like(m, dtype=np.uint32)
    current = 1
    sizes = []
    h, w = m.shape
    for y in range(h):
        for x in range(w):
            if m[y, x] and labeled[y, x] == 0:
                # flood fill
                stack = [(y, x)]
                labeled[y, x] = current
                count = 0
                while stack:
                    cy, cx = stack.pop()
                    count += 1
                    for ny in (cy - 1, cy, cy + 1):
                        for nx in (cx - 1, cx, cx + 1):
                            if 0 <= ny < h and 0 <= nx < w and m[ny, nx] and labeled[ny, nx] == 0:
                                labeled[ny, nx] = current
                                stack.append((ny, nx))
                sizes.append((current, count))
                current += 1
    keep = {label for label, size in sizes if size >= min_size}
    cleaned = np.isin(labeled, list(keep))
    return cleaned.astype(np.uint8)


def overlay_mask(sem: np.ndarray, mask: np.ndarray) -> Image.Image:
    base = Image.fromarray(np.clip(sem, 0, 255).astype(np.uint8), mode="L").convert("RGB")
    overlay = base.copy()
    pixels = overlay.load()
    h, w = mask.shape
    for y in range(h):
        for x in range(w):
            if mask[y, x]:
                pixels[x, y] = (255, 0, 0)
    return overlay


def main():
    args = parse_args()
    cfg = load_config(args.config)
    cfg["out_dir"].mkdir(parents=True, exist_ok=True)

    cad_files = {p.name: p for p in cfg["cad_dir"].glob("*.png")}
    sem_files = {p.name: p for p in cfg["sem_dir"].glob("*.png")}
    common = sorted(set(cad_files) & set(sem_files))
    if not common:
        raise SystemExit("No matching filenames between CAD and SEM directories.")

    for name in common:
        cad_img = load_gray(cad_files[name])
        sem_img = load_gray(sem_files[name])

        if cad_img.shape != sem_img.shape:
            # resize SEM to CAD size
            sem_img = np.array(Image.fromarray(sem_img.astype(np.uint8)).resize((cad_img.shape[1], cad_img.shape[0]), Image.BILINEAR), dtype=np.float32)

        sem_aligned, shift = align_sem_to_cad(cad_img, sem_img)
        diff = defect_map(cad_img, sem_aligned)
        adaptive_thr = diff.mean() + cfg["std_factor"] * diff.std()
        thr = max(cfg["threshold"], adaptive_thr)
        mask = (diff > thr).astype(np.uint8)
        mask = postprocess_mask(mask, cfg["min_size"])

        mask_img = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
        overlay = overlay_mask(sem_aligned, mask)

        mask_img.save(cfg["out_dir"] / f"{name}_mask.png")
        overlay.save(cfg["out_dir"] / f"{name}_overlay.png")
        print(f"Processed {name} (shift {shift})")


if __name__ == "__main__":
    main()
