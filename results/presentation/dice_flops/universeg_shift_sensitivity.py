"""
Experiment: Dice vs. centroid distance between GT target and context masks.
Compares UniverSeg (position-aligned CrossConv) vs PatchICL v3 (patch attention).

Setup (S=1):
  - Target image and GT mask are fixed.
  - Context image+mask are circularly shifted on a 2D (dy, dx) grid.
  - For each shift: Euclidean distance between target and context mask centroids,
    and Dice score of the model's prediction vs. target GT.

Dataset: promise12_256 (prostate MRI, binary label, 256×256) for both models.
"""

import os
import sys
import random

import numpy as np
import torch
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

sys.path.insert(0, "/home/dpxuser/repos/UniverSeg")
sys.path.insert(0, "/home/dpxuser/ic_segmentation")

DATA_ROOT = "/nfs/data/nii/data1/Analysis/camaret___in_context_segmentation/ANALYSIS_20251122/data/medsegbench"
CKPT_PATH = "/nfs/data/nii/data1/Analysis/camaret___in_context_segmentation/ANALYSIS_20251122/results/checkpoints/2026-03-20_driven-wind-609/best_model.pt"
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_PAIRS   = 40
SEED      = 42

DATASETS  = ["cystoidfluid", "uwaterlooskincancer", "brifiseg", "ultrasoundnerve"]

_STEPS    = list(range(-64, 65, 16))              # ±64 px, step 16
SHIFT_GRID = [(dy, dx) for dy in _STEPS for dx in _STEPS]   # 81 combinations


# ── model loading ─────────────────────────────────────────────────────────────

def load_universeg():
    from src.models.universeg_baseline import UniverSegBaseline
    model = UniverSegBaseline(pretrained=True, input_size=128).to(DEVICE).eval()
    print(f"UniverSeg loaded ({sum(p.numel() for p in model.parameters()):,} params)")
    return model


def load_patchicl():
    from src.model_builder import build_patch_icl_model

    # Load stored config from checkpoint and override the checkpoint path
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    cfg  = OmegaConf.create(ckpt["config"])
    cfg.checkpoint = CKPT_PATH       # use this checkpoint, not the one stored inside it
    cfg.feature_mode = "on_the_fly"

    model = build_patch_icl_model(cfg, DEVICE, context_size=1).to(DEVICE).eval()
    print(f"PatchICL v3 loaded ({sum(p.numel() for p in model.parameters()):,} params)")
    return model


# ── helpers ───────────────────────────────────────────────────────────────────

def to_gray(img: np.ndarray) -> np.ndarray:
    """Convert any image (H,W), (H,W,C), or (C,H,W) to (H,W) grayscale."""
    if img.ndim == 2:
        return img
    if img.ndim == 3 and img.shape[-1] in (1, 3):   # H,W,C
        return img.mean(axis=-1) if img.shape[-1] == 3 else img[..., 0]
    if img.ndim == 3 and img.shape[0] in (1, 3):     # C,H,W
        return img.mean(axis=0) if img.shape[0] == 3 else img[0]
    return img


def normalize(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    lo, hi = img.min(), img.max()
    return (img - lo) / (hi - lo + 1e-8)


def load_dataset(name: str):
    """Load test images and masks from a medsegbench 256-px npz file."""
    npz  = np.load(os.path.join(DATA_ROOT, f"{name}_256.npz"))
    keys = list(npz.keys())
    img_key  = next(k for k in keys if "test" in k and ("image" in k or "img" in k))
    mask_key = next(k for k in keys if "test" in k and ("label" in k or "mask" in k))
    imgs  = npz[img_key]
    masks = npz[mask_key]
    # Convert to (N, H, W) grayscale
    imgs  = np.stack([to_gray(imgs[i]) for i in range(len(imgs))])
    return imgs, masks


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    p, g = pred.flatten(), gt.flatten()
    return float(2 * (p * g).sum() / (p.sum() + g.sum() + 1e-8))


def centroid(mask: np.ndarray):
    coords = np.argwhere(mask > 0)
    return coords.mean(axis=0) if len(coords) > 0 else None


def predict(model, tgt_img, ctx_img, ctx_mask):
    """Return binary prediction (H, W). Both models share the same forward interface."""
    def t(a):  # (H,W) → (1, 1, H, W)
        return torch.from_numpy(a).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = model(
            t(tgt_img),
            context_in=t(ctx_img).unsqueeze(1),    # (1, S=1, 1, H, W)
            context_out=t(ctx_mask).unsqueeze(1),  # (1, S=1, 1, H, W)
            mode="val",
        )
    return (out["final_logit"].squeeze() > 0).cpu().numpy().astype(np.float32)


# ── experiment ────────────────────────────────────────────────────────────────

def run(model, imgs, masks, pairs):
    all_dists, all_dices = [], []

    for pair_i, (ti, ci) in enumerate(pairs):
        tgt_img   = normalize(imgs[ti])
        tgt_mask  = masks[ti].astype(np.float32)
        ctx_img0  = normalize(imgs[ci])
        ctx_mask0 = masks[ci].astype(np.float32)

        tgt_c = centroid(tgt_mask)
        if tgt_c is None:
            continue

        for dy, dx in SHIFT_GRID:
            ctx_img  = np.roll(np.roll(ctx_img0,  dy, axis=0), dx, axis=1)
            ctx_mask = np.roll(np.roll(ctx_mask0, dy, axis=0), dx, axis=1)

            ctx_c = centroid(ctx_mask)
            if ctx_c is None:
                continue

            dist = float(np.linalg.norm(tgt_c - ctx_c))
            pred = predict(model, tgt_img, ctx_img, ctx_mask)
            all_dists.append(dist)
            all_dices.append(dice(pred, tgt_mask))

        if (pair_i + 1) % 10 == 0:
            print(f"  {pair_i + 1}/{N_PAIRS} pairs done")

    return np.array(all_dists), np.array(all_dices)


def bin_results(dists, dices, bin_width=10):
    edges   = np.arange(0, dists.max() + bin_width, bin_width)
    centers = (edges[:-1] + edges[1:]) / 2
    means, stds = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        vals = dices[(dists >= lo) & (dists < hi)]
        means.append(vals.mean() if len(vals) > 0 else np.nan)
        stds.append(vals.std()   if len(vals) > 1 else 0.0)
    return centers, np.array(means), np.array(stds)


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_dataset(ax, results, title):
    """Plot both model curves on a single axes."""
    colors = {"UniverSeg": "steelblue", "PatchICL v3": "coral"}
    for name, (dists, dices) in results.items():
        centers, means, stds = bin_results(dists, dices)
        valid = ~np.isnan(means)
        rng = np.random.default_rng(0)
        idx = rng.choice(len(dists), size=min(800, len(dists)), replace=False)
        ax.scatter(dists[idx], dices[idx], s=3, alpha=0.12,
                   color=colors[name], linewidths=0)
        ax.fill_between(centers[valid],
                        (means - stds)[valid], (means + stds)[valid],
                        alpha=0.20, color=colors[name])
        ax.plot(centers[valid], means[valid], "o-", color=colors[name],
                linewidth=1.8, markersize=4, label=name)
    ax.set_title(title, fontsize=10)
    ax.set_ylim(-0.05, 1.0)
    ax.set_xlim(left=0)
    ax.set_xlabel("Centroid distance (px)", fontsize=8)
    ax.set_ylabel("Dice", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7)


def plot_all(all_results, save_dir="results/presentation"):
    n = len(all_results)
    ncols = 2
    nrows = (n + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 4 * nrows))
    axes = axes.flatten()

    for ax, (ds_name, results) in zip(axes, all_results.items()):
        plot_dataset(ax, results, ds_name)

    # Hide any unused axes
    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle(
        "Position sensitivity: UniverSeg vs PatchICL v3\n"
        f"S=1, {N_PAIRS} pairs × {len(SHIFT_GRID)} 2D circular shifts, 256×256",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "universeg_shift_sensitivity.pdf")
    plt.savefig(path, bbox_inches="tight")
    print(f"\nFigure saved → {path}")


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    print(f"Datasets: {DATASETS}\n")

    # Load models once, reuse across datasets
    print("Loading UniverSeg...")
    useg = load_universeg()
    print("Loading PatchICL v3...")
    picl = load_patchicl()

    all_results = {}

    for ds_name in DATASETS:
        print(f"\n{'='*50}")
        print(f"Dataset: {ds_name}")
        imgs, masks = load_dataset(ds_name)
        fg = [i for i in range(len(masks)) if masks[i].max() > 0]
        n_pairs = min(N_PAIRS, len(fg) - 1)
        print(f"  fg samples: {len(fg)}, pairs: {n_pairs}, "
              f"evals per model: {n_pairs * len(SHIFT_GRID)}")

        random.seed(SEED)
        pairs = [tuple(random.sample(fg, 2)) for _ in range(n_pairs)]

        results = {}

        print("  → UniverSeg")
        results["UniverSeg"] = run(useg, imgs, masks, pairs)

        print("  → PatchICL v3")
        results["PatchICL v3"] = run(picl, imgs, masks, pairs)

        all_results[ds_name] = results

        # Print summary table
        for mname, (dists, dices) in results.items():
            centers, means, _ = bin_results(dists, dices)
            valid = ~np.isnan(means)
            print(f"\n  {mname}")
            for c, m in zip(centers[valid], means[valid]):
                print(f"    dist~{c:>4.0f}px  Dice={m:.3f}")

    plot_all(all_results)
