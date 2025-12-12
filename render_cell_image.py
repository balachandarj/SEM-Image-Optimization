"""Render CAD and SEM-like images from centers.csv using settings in config.json.

The script takes no arguments. Files in the working directory:
  - centers.csv : rows with cx, cy, w, h, label
  - config.json : global settings (oas, cell, px_size, pad, dpi, generate_cad, generate_sem, use_color)

Outputs per row:
  cad_{label}_cx{cx}_cy{cy}_w{w}_h{h}.png  (if generate_cad)
  sem_{label}_cx{cx}_cy{cy}_w{w}_h{h}.png  (if generate_sem)
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Tuple

import gdstk
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

# Simple color map per layer to aid visualization.
LAYER_COLORS: Dict[Tuple[int, int], str] = {
    (10, 0): "#c9d3ff",  # nwell
    (1, 0): "#74c476",  # active
    (2, 0): "#fdae6b",  # poly
    (3, 0): "#9e9ac8",  # contact
    (4, 0): "#6baed6",  # metal1
    (30, 0): "#bcbddc",  # via1
    (5, 0): "#3182bd",  # metal2
    (31, 0): "#9ecae1",  # via2
    (6, 0): "#08519c",  # metal3
}

# Grayscale palette (different shades to keep layers distinguishable).
LAYER_GRAYS: Dict[Tuple[int, int], str] = {
    (10, 0): "#d9d9d9",
    (1, 0): "#b0b0b0",
    (2, 0): "#969696",
    (3, 0): "#a6a6a6",
    (4, 0): "#7f7f7f",
    (30, 0): "#c0c0c0",
    (5, 0): "#555555",
    (31, 0): "#8c8c8c",
    (6, 0): "#3f3f3f",
}

CONFIG_DEFAULTS = {
    "oas": "synthetic_layout.oas",
    "cell": "TOP",
    "px_size": 115,
    "pad": 0.5,
    "dpi": 300,
    "generate_cad": True,
    "generate_sem": True,
    "use_color": False,
    "output_dir": "output",
}


def load_cell(oas_path: Path, cell_name: str) -> gdstk.Cell:
    lib = gdstk.read_oas(oas_path)
    for cell in lib.cells:
        if cell.name == cell_name:
            return cell
    raise SystemExit(f"Cell '{cell_name}' not found. Available: {[c.name for c in lib.cells]}")


def load_config(config_path: Path) -> dict:
    if not config_path.exists():
        return dict(CONFIG_DEFAULTS)
    with config_path.open() as f:
        data = json.load(f)
    cfg = dict(CONFIG_DEFAULTS)
    cfg.update({k: data[k] for k in data if k in cfg})
    # Coerce numeric types
    cfg["px_size"] = int(cfg["px_size"])
    cfg["pad"] = float(cfg["pad"])
    cfg["dpi"] = int(cfg["dpi"])
    cfg["generate_cad"] = bool(cfg["generate_cad"])
    cfg["generate_sem"] = bool(cfg["generate_sem"])
    cfg["use_color"] = bool(cfg["use_color"])
    cfg["oas"] = str(cfg["oas"])
    cfg["cell"] = str(cfg["cell"])
    return cfg


def render_cell(
    cell: gdstk.Cell,
    out_path: Path,
    pad: float = 0.5,
    dpi: int = 300,
    px_size: int | None = 115,
    window: tuple[float, float, float, float] | None = None,
    use_gray: bool = True,
    use_sem: bool = False,
) -> None:
    bbox = cell.bounding_box()
    if bbox is None:
        raise SystemExit("Cell is empty; nothing to render.")
    (minx, miny), (maxx, maxy) = bbox

    # Determine view window.
    if window:
        x0, y0, x1, y1 = window
    else:
        x0, y0, x1, y1 = minx, miny, maxx, maxy
    width = x1 - x0
    height = y1 - y0

    polys = cell.get_polygons(apply_repetitions=True, include_paths=True, depth=None)

    fig, ax = plt.subplots()
    fig.set_dpi(dpi)
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(x0 - pad, x1 + pad)
    ax.set_ylim(y0 - pad, y1 + pad)
    ax.axis("off")
    fig.subplots_adjust(0, 0, 1, 1)  # remove outer padding

    # Draw polygons grouped by layer for consistent color mapping.
    palette = LAYER_GRAYS if use_gray else LAYER_COLORS
    for poly in polys:
        pts = getattr(poly, "points", None)
        if pts is None or len(pts) < 3:
            continue
        layer = getattr(poly, "layer", 0)
        datatype = getattr(poly, "datatype", 0)
        color = palette.get((layer, datatype), "#aaaaaa")
        patch = MplPolygon(pts, closed=True, facecolor=color, edgecolor="black", linewidth=0.3)
        ax.add_patch(patch)

    if px_size:
        size_in = px_size / dpi
        fig.set_size_inches(size_in, size_in)  # enforce square pixel size
    else:
        fig.set_size_inches(width / 10, height / 10)  # scale roughly to microns/10

    if use_sem:
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        plt.close(fig)
        sem_img = _sem_effect(rgba)
        sem_img.save(out_path, dpi=(dpi, dpi))
    else:
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0)
        plt.close(fig)


def _sem_effect(rgba: np.ndarray) -> Image.Image:
    """Apply a lightweight SEM-like effect to an RGBA array."""
    rgb = rgba[:, :, :3]
    gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    gray = 255.0 - gray  # invert: bright features on dark background

    # Vignette to darken corners slightly.
    h, w = gray.shape
    y, x = np.ogrid[:h, :w]
    cx, cy = w / 2, h / 2
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    r_norm = r / r.max()
    vignette = 1.0 - 0.18 * (r_norm ** 1.5)
    gray = gray * vignette

    # Add mild Gaussian noise.
    noise = np.random.normal(0, 5, size=gray.shape)
    gray = np.clip(gray + noise, 0, 255)

    img = Image.fromarray(gray.astype(np.uint8), mode="L")
    img = img.filter(ImageFilter.GaussianBlur(radius=0.35))
    img = ImageEnhance.Contrast(img).enhance(1.2)
    return img


def main():
    centers_path = Path("centers.csv")
    if not centers_path.exists():
        raise SystemExit("centers.csv not found in current directory.")

    config = load_config(Path("config.json"))
    oas = Path(config["oas"])
    cell_name = config["cell"]
    px_size = config["px_size"]
    pad = config["pad"]
    dpi = config["dpi"]
    generate_cad = config["generate_cad"]
    generate_sem = config["generate_sem"]
    use_color = config["use_color"]
    output_dir = Path(config.get("output_dir", "."))
    output_dir.mkdir(parents=True, exist_ok=True)

    cell = load_cell(oas, cell_name)
    use_gray = not use_color

    with centers_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            try:
                cx = float(row["cx"])
                cy = float(row["cy"])
                w = float(row["w"])
                h = float(row["h"])
            except KeyError:
                raise SystemExit("centers.csv must contain headers: cx, cy, w, h")
            
            window = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)    
            stem = f"cx{cx}_cy{cy}"

            if generate_cad:
                cad_out = output_dir / f"cad_{stem}.png"
                render_cell(
                    cell,
                    cad_out,
                    pad=pad,
                    dpi=dpi,
                    px_size=px_size,
                    window=window,
                    use_gray=use_gray,
                    use_sem=False,
                )
                print(f"Wrote {cad_out.resolve()} from cell '{cell_name}'")

            if generate_sem:
                sem_out = output_dir / f"sem_{stem}.png"
                render_cell(
                    cell,
                    sem_out,
                    pad=pad,
                    dpi=dpi,
                    px_size=px_size,
                    window=window,
                    use_gray=True,
                    use_sem=True,
                )
                print(f"Wrote {sem_out.resolve()} from cell '{cell_name}'")


if __name__ == "__main__":
    main()
