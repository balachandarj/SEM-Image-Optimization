"""Rank images by CAD/SEM difference to pick likely defect images.

Usage:
    python detect_defects_rank.py --config config_detect.json --top 10

Outputs a CSV with scores and a text listing of top-N images.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image, ImageOps
from skimage.metrics import structural_similarity as ssim
from skimage.registration import phase_cross_correlation
from scipy.ndimage import shift as nd_shift

CONFIG_DEFAULTS = {
    "cad_dir": "output/cad",
    "sem_dir": "output/sem",
    "out_dir": "output/diff",
}


def parse_args():
    p = argparse.ArgumentParser(description="Rank CAD/SEM pairs by difference.")
    p.add_argument("--config", type=Path, default=Path("config_detect.json"))
    p.add_argument("--top", type=int, default=10, help="Number of top images to report.")
    return p.parse_args()


def load_config(path: Path) -> dict:
    cfg = dict(CONFIG_DEFAULTS)
    if path.exists():
        data = json.load(path.open())
        cfg.update({k: data[k] for k in CONFIG_DEFAULTS if k in data})
    cfg["cad_dir"] = Path(cfg["cad_dir"])
    cfg["sem_dir"] = Path(cfg["sem_dir"])
    cfg["out_dir"] = Path(cfg["out_dir"])
    return cfg


def load_gray(path: Path) -> np.ndarray:
    img = Image.open(path)
    img = ImageOps.grayscale(img)
    return np.array(img, dtype=np.float32)


def align_sem_to_cad(cad: np.ndarray, sem: np.ndarray) -> Tuple[np.ndarray, Tuple[float, float]]:
    shift_estimate, _, _ = phase_cross_correlation(cad, sem, upsample_factor=10)
    aligned = nd_shift(sem, shift=shift_estimate, mode="nearest", order=1)
    return aligned, (float(shift_estimate[1]), float(shift_estimate[0]))  # (dx, dy)


def diff_score(cad: np.ndarray, sem: np.ndarray) -> float:
    c = (cad - cad.min()) / (cad.max() - cad.min() + 1e-6)
    s = (sem - sem.min()) / (sem.max() - sem.min() + 1e-6)
    score, full = ssim(c, s, full=True, data_range=1.0)
    diff = 1.0 - full
    # Aggregate: mean + std to emphasize spread
    return float(diff.mean() + diff.std())


def main():
    args = parse_args()
    cfg = load_config(args.config)
    out_csv = cfg["out_dir"] / "diff_scores.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    cad_files = {p.name: p for p in cfg["cad_dir"].glob("*.png")}
    sem_files = {p.name: p for p in cfg["sem_dir"].glob("*.png")}
    common = sorted(set(cad_files) & set(sem_files))
    if not common:
        raise SystemExit("No matching filenames between CAD and SEM directories.")

    scores = []
    for name in common:
        cad_img = load_gray(cad_files[name])
        sem_img = load_gray(sem_files[name])
        if cad_img.shape != sem_img.shape:
            sem_img = np.array(Image.fromarray(sem_img.astype(np.uint8)).resize((cad_img.shape[1], cad_img.shape[0]), Image.BILINEAR), dtype=np.float32)
        sem_aligned, shift = align_sem_to_cad(cad_img, sem_img)
        score = diff_score(cad_img, sem_aligned)
        scores.append((name, score, shift))

    scores.sort(key=lambda x: x[1], reverse=True)

    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "score", "shift_dx", "shift_dy"])
        for name, score, shift in scores:
            w.writerow([name, f"{score:.6f}", shift[0], shift[1]])

    top_n = scores[: args.top]
    print(f"Top {args.top} most different images:")
    for name, score, shift in top_n:
        print(f"{name}: score={score:.4f}, shift={shift}")


if __name__ == "__main__":
    main()
