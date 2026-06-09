"""
Visualise target/context pairs with mask overlays and model predictions
for uwaterlooskincancer, at four centroid-distance ranges.

Grid layout (rows = centroid-distance bins, columns = target|context|UniverSeg|PatchICL).
"""

import os
import sys
import random

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from omegaconf import OmegaConf

sys.path.insert(0, "/home/dpxuser/repos/UniverSeg")
sys.path.insert(0, "/home/dpxuser/ic_segmentation")

DATA_ROOT = "/nfs/data/nii/data1/Analysis/camaret___in_context_segmentation/ANALYSIS_20251122/data/medsegbench"
CKPT_PATH = "/nfs/data/nii/data1/Analysis/camaret___in_context_segmentation/ANALYSIS_20251122/results/checkpoints/2026-03-20_driven-wind-609/best_model.pt"
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED      = 0

# Centroid-distance ranges to illustrate [lo, hi)
DIST_BINS  = [(0, 20), (40, 60), (80, 100), (110, 140)]
BIN_LABELS = ["~10 px (aligned)", "~50 px", "~90 px", "~120 px (misaligned)"]
_STEPS     = list(range(-64, 65, 8))
SHIFT_GRID = [(dy, dx) for dy in _STEPS for dx in _STEPS]


# ── model loading ─────────────────────────────────────────────────────────────

def load_universeg():
    from src.models.universeg_baseline import UniverSegBaseline
    m = UniverSegBaseline(pretrained=True, input_size=128).to(DEVICE).eval()
    return m


def load_patchicl():
    from src.model_builder import build_patch_icl_model
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    cfg  = OmegaConf.create(ckpt["config"])
    cfg.checkpoint  = CKPT_PATH
    cfg.feature_mode = "on_the_fly"
    m = build_patch_icl_model(cfg, DEVICE, context_size=1, verbose=False)
    return m.to(DEVICE).eval()


# ── helpers ───────────────────────────────────────────────────────────────────

def to_gray(img):
    if img.ndim == 2:
        return img
    if img.ndim == 3 and img.shape[-1] in (1, 3):
        return img.mean(axis=-1) if img.shape[-1] == 3 else img[..., 0]
    return img.mean(axis=0)


def normalize(img):
    img = img.astype(np.float32)
    lo, hi = img.min(), img.max()
    return (img - lo) / (hi - lo + 1e-8)


def centroid(mask):
    coords = np.argwhere(mask > 0)
    return coords.mean(axis=0) if len(coords) > 0 else None


def predict(model, tgt_img, ctx_img, ctx_mask):
    def t(a):
        return torch.from_numpy(a).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = model(
            t(tgt_img),
            context_in=t(ctx_img).unsqueeze(1),
            context_out=t(ctx_mask).unsqueeze(1),
            mode="val",
        )
    return (out["final_logit"].squeeze() > 0).cpu().numpy().astype(np.float32)


def overlay(ax, img, mask, mask_color, alpha=0.45, label=None):
    """Show grayscale img with a semi-transparent mask overlay."""
    ax.imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    rgba = np.zeros((*mask.shape, 4), dtype=np.float32)
    rgba[..., :3] = np.array(mask_color)
    rgba[..., 3]  = mask * alpha
    ax.imshow(rgba, interpolation="nearest")
    ax.axis("off")
    if label:
        ax.set_title(label, fontsize=8, pad=2)


# ── collect examples ──────────────────────────────────────────────────────────

def prefilter_pairs(imgs, masks, fg, picl_model, min_dice=0.5):
    """Return set of (ti, ci) pairs where PatchICL dice > min_dice at zero shift."""
    def t(a):
        return torch.from_numpy(a).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
    good = set()
    for ti in fg:
        for ci in fg:
            if ti == ci:
                continue
            tgt_img  = normalize(to_gray(imgs[ti]))
            tgt_mask = masks[ti].astype(np.float32)
            ctx_img  = normalize(to_gray(imgs[ci]))
            ctx_mask = masks[ci].astype(np.float32)
            with torch.no_grad():
                out = picl_model(
                    t(tgt_img),
                    context_in=t(ctx_img).unsqueeze(1),
                    context_out=t(ctx_mask).unsqueeze(1),
                    mode="val",
                )
            pred = (out["final_logit"].squeeze() > 0).cpu().numpy().astype(np.float32)
            d = 2*(pred*tgt_mask).sum() / (pred.sum()+tgt_mask.sum()+1e-8)
            if d >= min_dice:
                good.add((ti, ci))
    print(f"  prefilter: {len(good)} / {len(fg)*(len(fg)-1)} pairs with PatchICL Dice ≥ {min_dice}")
    return good


def collect_examples(imgs, masks, fg, good_pairs, picl_model, n_examples=1, min_picl_dice=0.5):
    """For each distance bin, find n_examples (pair, shift) tuples from good_pairs only.

    Only keeps examples where PatchICL still achieves min_picl_dice after the shift,
    so wrap-around artifacts that break both models are excluded.
    """
    def t(a):
        return torch.from_numpy(a).float().unsqueeze(0).unsqueeze(0).to(DEVICE)

    rng = random.Random(SEED)
    candidates = {b: [] for b in range(len(DIST_BINS))}

    pairs = list(good_pairs)
    rng.shuffle(pairs)

    for ti, ci in pairs:
        if all(len(v) >= n_examples for v in candidates.values()):
            break

        tgt_img  = normalize(to_gray(imgs[ti]))
        tgt_mask = masks[ti].astype(np.float32)
        ctx_img0 = normalize(to_gray(imgs[ci]))
        ctx_mask0 = masks[ci].astype(np.float32)

        tgt_c = centroid(tgt_mask)
        if tgt_c is None:
            continue

        for dy, dx in SHIFT_GRID:
            ctx_img  = np.roll(np.roll(ctx_img0,  dy, axis=0), dx, axis=1)
            ctx_mask = np.roll(np.roll(ctx_mask0, dy, axis=0), dx, axis=1)
            ctx_c    = centroid(ctx_mask)
            if ctx_c is None:
                continue

            dist = float(np.linalg.norm(tgt_c - ctx_c))
            for bi, (lo, hi) in enumerate(DIST_BINS):
                if lo <= dist < hi and len(candidates[bi]) < n_examples:
                    # Verify PatchICL still works after this shift
                    with torch.no_grad():
                        out = picl_model(
                            t(tgt_img),
                            context_in=t(ctx_img).unsqueeze(1),
                            context_out=t(ctx_mask).unsqueeze(1),
                            mode="val",
                        )
                    pred = (out["final_logit"].squeeze() > 0).cpu().numpy().astype(np.float32)
                    d = 2*(pred*tgt_mask).sum() / (pred.sum()+tgt_mask.sum()+1e-8)
                    if d < min_picl_dice:
                        continue
                    candidates[bi].append({
                        "tgt_img":  tgt_img,
                        "tgt_mask": tgt_mask,
                        "ctx_img":  ctx_img,
                        "ctx_mask": ctx_mask,
                        "dist":     dist,
                    })

    return candidates


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Load data
    npz   = np.load(os.path.join(DATA_ROOT, "uwaterlooskincancer_256.npz"))
    keys  = list(npz.keys())
    img_k = next(k for k in keys if "test" in k and ("image" in k or "img" in k))
    msk_k = next(k for k in keys if "test" in k and ("label" in k or "mask" in k))
    imgs  = npz[img_k]
    masks = npz[msk_k]
    fg    = [i for i in range(len(masks)) if masks[i].max() > 0]
    print(f"Loaded {len(imgs)} samples, {len(fg)} with foreground")

    # Load models early so prefilter can use PatchICL
    print("Loading models...")
    useg = load_universeg()
    picl = load_patchicl()

    # Pre-filter pairs to those where PatchICL works at zero shift
    print("Pre-filtering pairs...")
    good_pairs = prefilter_pairs(imgs, masks, fg, picl, min_dice=0.5)

    # Collect one example per distance bin
    print("Collecting examples...")
    candidates = collect_examples(imgs, masks, fg, good_pairs, picl, n_examples=1)
    for bi, examples in candidates.items():
        print(f"  bin {DIST_BINS[bi]}: {len(examples)} examples found "
              f"(dist={examples[0]['dist']:.1f}px)" if examples else f"  bin {DIST_BINS[bi]}: NONE")

    # (models already loaded above)

    # Build figure: rows = distance bins, cols = target | context | UniverSeg | PatchICL
    n_rows = len(DIST_BINS)
    n_cols = 4
    col_titles = ["Target + GT mask", "Context + mask (shifted)", "UniverSeg pred", "PatchICL v3 pred"]

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 2.8 * n_rows))
    fig.suptitle("uwaterlooskincancer — position sensitivity (S=1, 256×256)",
                 fontsize=11, fontweight="bold", y=1.01)

    # Column headers
    for ci, title in enumerate(col_titles):
        axes[0, ci].set_title(title, fontsize=9, fontweight="bold", pad=4)

    for bi, ex_list in candidates.items():
        row_axes = axes[bi]

        if not ex_list:
            for ax in row_axes:
                ax.axis("off")
                ax.text(0.5, 0.5, "no example found",
                        ha="center", va="center", transform=ax.transAxes)
            continue

        ex = ex_list[0]
        tgt_img  = ex["tgt_img"]
        tgt_mask = ex["tgt_mask"]
        ctx_img  = ex["ctx_img"]
        ctx_mask = ex["ctx_mask"]
        dist     = ex["dist"]

        # Row label
        row_axes[0].set_ylabel(f"{BIN_LABELS[bi]}\n({dist:.0f} px)",
                               fontsize=8, rotation=90, labelpad=4)

        # Predictions
        pred_useg = predict(useg, tgt_img, ctx_img, ctx_mask)
        pred_picl = predict(picl, tgt_img, ctx_img, ctx_mask)

        dice = lambda p, g: 2*(p*g).sum() / (p.sum()+g.sum()+1e-8)
        d_useg = dice(pred_useg, tgt_mask)
        d_picl = dice(pred_picl, tgt_mask)

        # col 0: target + GT
        overlay(row_axes[0], tgt_img, tgt_mask, (0.2, 0.8, 0.2))

        # col 1: context + shifted mask
        overlay(row_axes[1], ctx_img, ctx_mask, (0.2, 0.5, 1.0))

        # col 2: UniverSeg pred (red) + GT contour (green)
        overlay(row_axes[2], tgt_img, pred_useg, (1.0, 0.2, 0.2),
                label=f"Dice {d_useg:.3f}")
        # GT contour
        gt_rgba = np.zeros((*tgt_mask.shape, 4), dtype=np.float32)
        gt_rgba[..., 1] = 0.9
        gt_rgba[..., 3] = tgt_mask * 0.25
        row_axes[2].imshow(gt_rgba, interpolation="nearest")

        # col 3: PatchICL pred (orange) + GT contour (green)
        overlay(row_axes[3], tgt_img, pred_picl, (1.0, 0.6, 0.0),
                label=f"Dice {d_picl:.3f}")
        gt_rgba2 = np.zeros((*tgt_mask.shape, 4), dtype=np.float32)
        gt_rgba2[..., 1] = 0.9
        gt_rgba2[..., 3] = tgt_mask * 0.25
        row_axes[3].imshow(gt_rgba2, interpolation="nearest")

    # Legend
    legend_elems = [
        mpatches.Patch(color=(0.2, 0.8, 0.2), alpha=0.7, label="GT mask"),
        mpatches.Patch(color=(0.2, 0.5, 1.0), alpha=0.7, label="Context mask"),
        mpatches.Patch(color=(1.0, 0.2, 0.2), alpha=0.7, label="UniverSeg pred"),
        mpatches.Patch(color=(1.0, 0.6, 0.0), alpha=0.7, label="PatchICL v3 pred"),
    ]
    fig.legend(handles=legend_elems, loc="lower center", ncol=4,
               fontsize=8, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    save_dir = "results/presentation"
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "universeg_shift_vis.pdf")
    plt.savefig(path, bbox_inches="tight")
    print(f"\nFigure saved → {path}")
