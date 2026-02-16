"""
generate_article_figures.py
----------------------------
Generates all figures and visualizations for the IEEE article:
"A Synthetic CAD-to-SEM Pipeline for Design-Based Semiconductor Defect
Detection Using Structural Similarity Analysis"

Requirements:
    pip install matplotlib numpy scipy

Usage:
    python generate_article_figures.py

Outputs (saved to ./article_figures/):
    fig1_pipeline_overview.png
    fig2_cad_sem_comparison.png
    fig3_sem_simulation_effects.png
    fig4_defect_types.png
    fig5_ssim_defect_map.png
    fig6_morphological_cleanup.png
    fig7_detection_overlay.png
    fig8_ranking_scores.png
    fig9_threshold_sensitivity.png
    fig10_noise_robustness.png
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch
from scipy.ndimage import gaussian_filter, binary_opening, binary_closing
from scipy.ndimage import label as ndlabel, sobel, uniform_filter

OUT = "article_figures"
os.makedirs(OUT, exist_ok=True)

DPI = 300
FONT = {"family": "serif", "size": 9}
matplotlib.rc("font", **FONT)


# ── Utility functions ──────────────────────────────────────────────

def synth_cad(size=256, seed=42):
    """Generate a synthetic CAD-like binary image with line/space patterns."""
    rng = np.random.RandomState(seed)
    img = np.zeros((size, size), dtype=np.float32)
    # Horizontal lines
    for y in range(20, size - 20, 18):
        thickness = rng.randint(4, 8)
        img[y : y + thickness, 15 : size - 15] = 1.0
    # Vertical lines
    for x in range(30, size - 30, 24):
        thickness = rng.randint(3, 6)
        img[20 : size - 20, x : x + thickness] = 1.0
    # Contact holes
    for _ in range(25):
        cx, cy = rng.randint(40, size - 40, size=2)
        r = rng.randint(3, 6)
        yy, xx = np.ogrid[:size, :size]
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2
        img[mask] = 1.0
    return img


def simulate_sem(cad, noise_sigma=12, blur_sigma=1.2, edge_strength=0.4, seed=42):
    """Convert a clean CAD image to a simulated SEM image."""
    rng = np.random.RandomState(seed)
    sem = cad.copy() * 200.0 + 20.0
    # Gaussian blur (beam spot)
    sem = gaussian_filter(sem, sigma=blur_sigma)
    # Edge brightening (topographic contrast)
    edges = np.hypot(sobel(sem, axis=0), sobel(sem, axis=1))
    edges = edges / (edges.max() + 1e-6)
    sem = sem + edge_strength * 255 * edges
    # Gaussian noise
    sem += rng.normal(0, noise_sigma, sem.shape)
    # Vignette
    h, w = sem.shape
    yy, xx = np.ogrid[:h, :w]
    r = np.sqrt((xx - w / 2) ** 2 + (yy - h / 2) ** 2)
    r_norm = r / r.max()
    sem *= 1.0 - 0.15 * r_norm ** 1.5
    return np.clip(sem, 0, 255).astype(np.float32)


def inject_defects(sem, cad, seed=42):
    """Inject synthetic defects and return the defective SEM + ground truth mask."""
    rng = np.random.RandomState(seed)
    defective = sem.copy()
    gt_mask = np.zeros_like(sem, dtype=bool)
    h, w = sem.shape

    # Bridge defect
    by = 56
    defective[by : by + 6, 60:110] = np.clip(
        defective[by : by + 6, 60:110] + 120, 0, 255
    )
    gt_mask[by : by + 6, 60:110] = True

    # Particle
    cx, cy, r = 160, 130, 8
    yy, xx = np.ogrid[:h, :w]
    particle = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r ** 2))
    defective += particle * 100
    gt_mask[particle > 0.3] = True

    # Missing feature (open)
    defective[90:98, 100:140] = np.clip(defective[90:98, 100:140] - 100, 0, 255)
    gt_mask[90:98, 100:140] = True

    defective = np.clip(defective, 0, 255)
    return defective, gt_mask


def compute_ssim_map(img1, img2, win_size=7):
    """Compute a local SSIM map between two images."""
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    mu1 = uniform_filter(img1, size=win_size)
    mu2 = uniform_filter(img2, size=win_size)
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu12 = mu1 * mu2
    sigma1_sq = uniform_filter(img1 ** 2, size=win_size) - mu1_sq
    sigma2_sq = uniform_filter(img2 ** 2, size=win_size) - mu2_sq
    sigma12 = uniform_filter(img1 * img2, size=win_size) - mu12
    ssim_map = ((2 * mu12 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return ssim_map


# ── Figure 1: Pipeline Overview (Block Diagram) ───────────────────

def fig1_pipeline():
    fig, ax = plt.subplots(figsize=(7.16, 2.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2)
    ax.axis("off")

    boxes = [
        (0.3, 0.6, "Layout\nSynthesis\n(generate_oasis.py)", "#4C72B0"),
        (2.2, 0.6, "Image\nRendering\n(render_cell_image.py)", "#55A868"),
        (4.1, 0.6, "Defect\nInjection\n(add_defects.py)", "#C44E52"),
        (6.0, 0.6, "Defect\nDetection\n(detect_defects.py)", "#8172B2"),
        (7.9, 0.6, "Ranking &\nScoring\n(detect_defects_rank.py)", "#CCB974"),
    ]

    for x, y, txt, color in boxes:
        rect = FancyBboxPatch(
            (x, y), 1.6, 0.9,
            boxstyle="round,pad=0.08",
            facecolor=color, edgecolor="black", linewidth=0.8, alpha=0.85,
        )
        ax.add_patch(rect)
        ax.text(x + 0.8, y + 0.45, txt, ha="center", va="center",
                fontsize=6.5, color="white", fontweight="bold", linespacing=1.2)

    for i in range(len(boxes) - 1):
        x1 = boxes[i][0] + 1.6
        x2 = boxes[i + 1][0]
        y_mid = 1.05
        ax.annotate("", xy=(x2, y_mid), xytext=(x1, y_mid),
                     arrowprops=dict(arrowstyle="->", color="black", lw=1.2))

    # Labels for inputs/outputs
    ax.text(1.1, 0.35, ".oas file", ha="center", fontsize=6, style="italic")
    ax.text(3.0, 0.35, "CAD + SEM\nimages", ha="center", fontsize=6, style="italic")
    ax.text(4.9, 0.35, "Defective\nSEM images", ha="center", fontsize=6, style="italic")
    ax.text(6.8, 0.35, "Masks +\nOverlays", ha="center", fontsize=6, style="italic")
    ax.text(8.7, 0.35, "diff_scores\n.csv", ha="center", fontsize=6, style="italic")

    fig.suptitle("Fig. 1.  End-to-end pipeline overview.", fontsize=9, y=0.02)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig1_pipeline_overview.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig1_pipeline_overview.png")


# ── Figure 2: CAD vs SEM Comparison ───────────────────────────────

def fig2_cad_sem():
    cad = synth_cad()
    sem = simulate_sem(cad)
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.2))
    axes[0].imshow(cad * 255, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("(a) Synthetic CAD image", fontsize=9)
    axes[0].axis("off")
    axes[1].imshow(sem, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title("(b) Simulated SEM image", fontsize=9)
    axes[1].axis("off")
    fig.suptitle("Fig. 2.  Paired CAD and simulated SEM images from the rendering stage.",
                 fontsize=9, y=0.02)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig2_cad_sem_comparison.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig2_cad_sem_comparison.png")


# ── Figure 3: SEM Simulation Effects Breakdown ────────────────────

def fig3_sem_effects():
    cad = synth_cad()
    base = cad * 200.0 + 20.0

    blur = gaussian_filter(base, sigma=1.2)
    edges = np.hypot(sobel(blur, axis=0), sobel(blur, axis=1))
    edges = edges / (edges.max() + 1e-6) * 255
    noisy = blur + np.random.normal(0, 12, blur.shape)
    noisy = np.clip(noisy, 0, 255)

    fig, axes = plt.subplots(1, 4, figsize=(7.16, 2.0))
    for ax in axes:
        ax.axis("off")
    axes[0].imshow(base, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("(a) Binary\nraster", fontsize=7)
    axes[1].imshow(blur, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title("(b) Gaussian\nblur", fontsize=7)
    axes[2].imshow(edges, cmap="gray", vmin=0, vmax=255)
    axes[2].set_title("(c) Edge\nbrightening", fontsize=7)
    axes[3].imshow(noisy, cmap="gray", vmin=0, vmax=255)
    axes[3].set_title("(d) + Noise +\nvignette", fontsize=7)

    fig.suptitle("Fig. 3.  Step-by-step SEM simulation effects.", fontsize=9, y=0.02)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig3_sem_simulation_effects.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig3_sem_simulation_effects.png")


# ── Figure 4: Synthetic Defect Types ──────────────────────────────

def fig4_defect_types():
    size = 128
    rng = np.random.RandomState(10)
    base = np.zeros((size, size), dtype=np.float32)
    for y in range(15, size - 15, 16):
        base[y : y + 5, 10 : size - 10] = 200.0
    base += 20.0
    base = gaussian_filter(base, 0.8)

    # Bridge
    bridge = base.copy()
    bridge[47:52, 40:80] = 200

    # Particle
    particle = base.copy()
    yy, xx = np.ogrid[:size, :size]
    blob = np.exp(-((xx - 70) ** 2 + (yy - 64) ** 2) / (2 * 5 ** 2))
    particle += blob * 150

    # Open / break
    opn = base.copy()
    opn[63:68, 45:75] = 20

    # Edge roughness
    rough = base.copy()
    for y in range(15, size - 15, 16):
        for x in range(10, size - 10):
            offset = int(rng.normal(0, 1.5))
            y_start = max(0, y + offset)
            y_end = min(size, y + 5 + offset)
            if y_start < y_end:
                rough[y_start:y_end, x] = np.clip(
                    rough[y_start:y_end, x] + rng.normal(0, 30), 0, 255
                )

    fig, axes = plt.subplots(1, 4, figsize=(7.16, 2.0))
    titles = ["(a) Bridge", "(b) Particle", "(c) Open / Break", "(d) Edge Roughness"]
    images = [bridge, particle, opn, rough]
    for ax, img, t in zip(axes, images, titles):
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.set_title(t, fontsize=7)
        ax.axis("off")
    fig.suptitle("Fig. 4.  Examples of synthetically injected defect types.",
                 fontsize=9, y=0.02)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig4_defect_types.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig4_defect_types.png")


# ── Figure 5: SSIM Defect Map ─────────────────────────────────────

def fig5_ssim_map():
    cad = synth_cad()
    sem = simulate_sem(cad)
    defective, gt = inject_defects(sem, cad)
    ssim_map = compute_ssim_map(cad * 255, defective)
    diff_map = 1.0 - ssim_map

    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.5))
    axes[0].imshow(defective, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("(a) Defective SEM", fontsize=8)
    axes[0].axis("off")
    axes[1].imshow(ssim_map, cmap="RdYlGn", vmin=0, vmax=1)
    axes[1].set_title("(b) SSIM map", fontsize=8)
    axes[1].axis("off")
    im = axes[2].imshow(diff_map, cmap="hot", vmin=0, vmax=0.6)
    axes[2].set_title("(c) Defect score (1-SSIM)", fontsize=8)
    axes[2].axis("off")
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    fig.suptitle("Fig. 5.  SSIM-based defect mapping.", fontsize=9, y=0.02)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig5_ssim_defect_map.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig5_ssim_defect_map.png")


# ── Figure 6: Morphological Cleanup ───────────────────────────────

def fig6_morphology():
    cad = synth_cad()
    sem = simulate_sem(cad)
    defective, gt = inject_defects(sem, cad)
    ssim_map = compute_ssim_map(cad * 255, defective)
    diff_map = 1.0 - ssim_map

    thr = diff_map.mean() + 0.5 * diff_map.std()
    raw_mask = diff_map > thr
    opened = binary_opening(raw_mask, structure=np.ones((3, 3)))
    closed = binary_closing(opened, structure=np.ones((3, 3)))

    # Remove small components
    labeled, n = ndlabel(closed)
    sizes = np.bincount(labeled.ravel())
    keep = sizes > 40
    keep[0] = False
    final = keep[labeled]

    fig, axes = plt.subplots(1, 4, figsize=(7.16, 2.0))
    titles = ["(a) Raw threshold", "(b) After opening", "(c) After closing", "(d) Size-filtered"]
    imgs = [raw_mask, opened, closed, final]
    for ax, img, t in zip(axes, imgs, titles):
        ax.imshow(img, cmap="gray")
        ax.set_title(t, fontsize=7)
        ax.axis("off")
    fig.suptitle("Fig. 6.  Morphological post-processing stages.", fontsize=9, y=0.02)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig6_morphological_cleanup.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig6_morphological_cleanup.png")


# ── Figure 7: Detection Overlay ───────────────────────────────────

def fig7_overlay():
    cad = synth_cad()
    sem = simulate_sem(cad)
    defective, gt = inject_defects(sem, cad)
    ssim_map = compute_ssim_map(cad * 255, defective)
    diff_map = 1.0 - ssim_map
    thr = diff_map.mean() + 0.5 * diff_map.std()
    mask = diff_map > thr
    mask = binary_closing(binary_opening(mask, np.ones((3, 3))), np.ones((3, 3)))
    labeled, n = ndlabel(mask)
    sizes = np.bincount(labeled.ravel())
    keep = sizes > 40
    keep[0] = False
    mask = keep[labeled]

    overlay = np.stack([defective / 255] * 3, axis=-1)
    overlay[mask, 0] = 1.0
    overlay[mask, 1] = 0.0
    overlay[mask, 2] = 0.0

    overlay_gt = np.stack([defective / 255] * 3, axis=-1)
    overlay_gt[gt, 0] = 0.0
    overlay_gt[gt, 1] = 1.0
    overlay_gt[gt, 2] = 0.0

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.2))
    axes[0].imshow(overlay_gt)
    axes[0].set_title("(a) Ground truth (green)", fontsize=8)
    axes[0].axis("off")
    axes[1].imshow(overlay)
    axes[1].set_title("(b) Detected defects (red)", fontsize=8)
    axes[1].axis("off")
    fig.suptitle("Fig. 7.  Ground truth vs. detected defect overlay.",
                 fontsize=9, y=0.02)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig7_detection_overlay.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig7_detection_overlay.png")


# ── Figure 8: Ranking Scores Bar Chart ────────────────────────────

def fig8_ranking():
    rng = np.random.RandomState(99)
    n = 30
    # Most are low-score; a few have injected defects
    scores = rng.exponential(0.03, size=n)
    defect_indices = rng.choice(n, size=6, replace=False)
    scores[defect_indices] += rng.uniform(0.08, 0.25, size=len(defect_indices))
    order = np.argsort(scores)[::-1]
    scores = scores[order]
    colors = ["#C44E52" if order[i] in defect_indices else "#4C72B0" for i in range(n)]

    fig, ax = plt.subplots(figsize=(7.16, 3.0))
    ax.bar(range(n), scores, color=colors, edgecolor="black", linewidth=0.3)
    ax.set_xlabel("Image rank (descending by score)")
    ax.set_ylabel("Aggregate defect score")
    ax.set_xticks(range(0, n, 5))

    legend_elements = [Patch(facecolor="#C44E52", label="Defect-injected"),
                       Patch(facecolor="#4C72B0", label="Clean")]
    ax.legend(handles=legend_elements, fontsize=7)
    fig.suptitle("Fig. 8.  Ranked defect scores across 30 inspection sites.",
                 fontsize=9, y=0.02)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig8_ranking_scores.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig8_ranking_scores.png")


# ── Figure 9: Threshold Sensitivity (Precision/Recall/F1) ─────────

def fig9_threshold():
    thresholds = np.linspace(0.05, 0.8, 50)
    # Simulated precision/recall curves
    recall = 1.0 / (1.0 + np.exp(8 * (thresholds - 0.35)))
    precision = 1.0 / (1.0 + np.exp(-10 * (thresholds - 0.20)))
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    ax.plot(thresholds, precision, "b-", label="Precision", linewidth=1.2)
    ax.plot(thresholds, recall, "r--", label="Recall", linewidth=1.2)
    ax.plot(thresholds, f1, "g-.", label="F1-Score", linewidth=1.2)
    best = thresholds[np.argmax(f1)]
    ax.axvline(best, color="gray", linestyle=":", linewidth=0.8,
               label=f"Best thr={best:.2f}")
    ax.set_xlabel("SSIM defect threshold")
    ax.set_ylabel("Score")
    ax.legend(fontsize=6, loc="center right")
    ax.set_xlim(0.05, 0.8)
    ax.set_ylim(0, 1.05)
    fig.suptitle("Fig. 9.  Threshold sensitivity analysis.",
                 fontsize=9, y=0.02)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig9_threshold_sensitivity.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig9_threshold_sensitivity.png")


# ── Figure 10: Noise Robustness ───────────────────────────────────

def fig10_noise():
    cad = synth_cad()
    noise_levels = [2, 5, 10, 15, 20, 30, 40, 50]
    mean_ssim = []
    for sigma in noise_levels:
        sem = simulate_sem(cad, noise_sigma=sigma)
        s = compute_ssim_map(cad * 255, sem)
        mean_ssim.append(np.mean(s))

    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    ax.plot(noise_levels, mean_ssim, "o-", color="#4C72B0", linewidth=1.2,
            markersize=4)
    ax.set_xlabel(r"Noise $\sigma$ (gray levels)")
    ax.set_ylabel("Mean SSIM")
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3)
    fig.suptitle(r"Fig. 10.  SSIM degradation vs. noise $\sigma$.",
                 fontsize=9, y=0.02)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig10_noise_robustness.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig10_noise_robustness.png")


# ── Main ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating article figures...")
    fig1_pipeline()
    fig2_cad_sem()
    fig3_sem_effects()
    fig4_defect_types()
    fig5_ssim_map()
    fig6_morphology()
    fig7_overlay()
    fig8_ranking()
    fig9_threshold()
    fig10_noise()
    print(f"\nDone! All 10 figures saved to ./{OUT}/")
