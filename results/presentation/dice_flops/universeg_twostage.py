"""
Multi-resolution UniverSeg ablation on medsegbench datasets.

Conditions (all fed images at --size resolution):
  baseline   — UniverSeg: resize to 128 internally (model training size)
  native     — UniverSeg: fed at full --size resolution (no internal resize)
  twostage   — UniverSeg: stage1 (baseline) → stage2: 128-px crop centered on
               stage1 pred centroid (target) and GT mask centroid (context)
  oracle     — stage2 with target crop centered on GT mask centroid (upper bound)
  patchicl   — PatchICL v3 at full --size resolution

Usage:
  python universeg_twostage.py --size 256
  python universeg_twostage.py --size 256 --load   # reload saved results and replot
"""

import argparse
import os
import pickle
import random
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from omegaconf import OmegaConf

sys.path.insert(0, "/home/dpxuser/repos/UniverSeg")
sys.path.insert(0, "/home/dpxuser/ic_segmentation")

DATA_ROOT = "/nfs/data/nii/data1/Analysis/camaret___in_context_segmentation/ANALYSIS_20251122/data/medsegbench"
CKPT_PATH = "/nfs/data/nii/data1/Analysis/camaret___in_context_segmentation/ANALYSIS_20251122/results/checkpoints/2026-03-20_driven-wind-609/best_model.pt"
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CROP_SIZE = 128
SEED      = 42
METHODS   = ["baseline", "native", "twostage", "oracle", "patchicl"]


# ── model loading ─────────────────────────────────────────────────────────────

def load_universeg(input_size: int):
    from src.models.universeg_baseline import UniverSegBaseline
    model = UniverSegBaseline(pretrained=True, input_size=input_size).to(DEVICE).eval()
    n = sum(p.numel() for p in model.parameters())
    print(f"  UniverSeg (input_size={input_size}): {n:,} params")
    return model


def load_patchicl():
    from src.model_builder import build_patch_icl_model
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    cfg  = OmegaConf.create(ckpt["config"])
    cfg.checkpoint   = CKPT_PATH
    cfg.feature_mode = "on_the_fly"
    model = build_patch_icl_model(cfg, DEVICE, context_size=1, verbose=False).to(DEVICE).eval()
    n = sum(p.numel() for p in model.parameters())
    print(f"  PatchICL v3: {n:,} params")
    return model


# ── data ──────────────────────────────────────────────────────────────────────

def discover_datasets(size: int) -> list[str]:
    return sorted(
        f.replace(f"_{size}.npz", "")
        for f in os.listdir(DATA_ROOT)
        if f.endswith(f"_{size}.npz") and not f.startswith("medsegbench")
    )


def to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    if img.ndim == 3 and img.shape[-1] in (1, 3):
        return img.mean(axis=-1) if img.shape[-1] == 3 else img[..., 0]
    return img.mean(axis=0)


def normalize(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    lo, hi = img.min(), img.max()
    return (img - lo) / (hi - lo + 1e-8)


def load_dataset(name: str, size: int):
    npz  = np.load(os.path.join(DATA_ROOT, f"{name}_{size}.npz"))
    keys = list(npz.keys())
    img_key  = next(k for k in keys if "test" in k and ("image" in k or "img" in k))
    mask_key = next(k for k in keys if "test" in k and ("label" in k or "mask" in k))
    imgs  = np.stack([to_gray(npz[img_key][i]) for i in range(len(npz[img_key]))])
    masks = npz[mask_key]
    return imgs, masks


# ── helpers ───────────────────────────────────────────────────────────────────

def centroid(mask: np.ndarray):
    coords = np.argwhere(mask > 0)
    return coords.mean(axis=0) if len(coords) > 0 else None


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    p, g = pred.flatten(), gt.flatten()
    return float(2 * (p * g).sum() / (p.sum() + g.sum() + 1e-8))


def to_tensor(a: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(a).float().unsqueeze(0).unsqueeze(0).to(DEVICE)


def _run_model(model, tgt_img, ctx_img, ctx_mask) -> np.ndarray:
    with torch.no_grad():
        out = model(
            to_tensor(tgt_img),
            context_in=to_tensor(ctx_img).unsqueeze(1),
            context_out=to_tensor(ctx_mask).unsqueeze(1),
            mode="val",
        )
    return (out["final_logit"].squeeze() > 0).cpu().numpy().astype(np.float32)


def crop_centered(img: np.ndarray, cy: float, cx: float, size: int = CROP_SIZE):
    H, W = img.shape
    h2 = size // 2
    y0 = int(np.clip(round(cy) - h2, 0, H - size))
    x0 = int(np.clip(round(cx) - h2, 0, W - size))
    return img[y0:y0 + size, x0:x0 + size], y0, x0


# ── prediction variants ───────────────────────────────────────────────────────

def predict_all(useg_128, useg_native, picl, tgt_img, ctx_img, ctx_mask, tgt_mask_gt):
    """Run all five conditions for a single pair. Returns dict method→dice."""
    # ── baseline (internal 128 resize) ──
    coarse = _run_model(useg_128, tgt_img, ctx_img, ctx_mask)

    # ── native (no internal resize) ──
    native_pred = _run_model(useg_native, tgt_img, ctx_img, ctx_mask)

    # ── context crop (shared by twostage and oracle) ──
    ctx_c = centroid(ctx_mask)
    if ctx_c is not None:
        ctx_img_crop,  cy0, cx0 = crop_centered(ctx_img,  ctx_c[0], ctx_c[1])
        ctx_mask_crop            = ctx_mask[cy0:cy0 + CROP_SIZE, cx0:cx0 + CROP_SIZE]
    else:
        ctx_img_crop  = ctx_img[:CROP_SIZE, :CROP_SIZE]
        ctx_mask_crop = ctx_mask[:CROP_SIZE, :CROP_SIZE]
        cy0 = cx0 = 0

    # ── twostage (target crop centered on stage1 pred centroid) ──
    tgt_c_pred = centroid(coarse)
    if tgt_c_pred is not None:
        tgt_crop, ty0, tx0 = crop_centered(tgt_img, tgt_c_pred[0], tgt_c_pred[1])
        refined = _run_model(useg_128, tgt_crop, ctx_img_crop, ctx_mask_crop)
        twostage_pred = np.zeros_like(coarse)
        twostage_pred[ty0:ty0 + CROP_SIZE, tx0:tx0 + CROP_SIZE] = refined
    else:
        twostage_pred = coarse

    # ── oracle (target crop centered on GT target mask centroid) ──
    tgt_c_gt = centroid(tgt_mask_gt)
    if tgt_c_gt is not None:
        tgt_crop_or, oy0, ox0 = crop_centered(tgt_img, tgt_c_gt[0], tgt_c_gt[1])
        refined_or = _run_model(useg_128, tgt_crop_or, ctx_img_crop, ctx_mask_crop)
        oracle_pred = np.zeros_like(coarse)
        oracle_pred[oy0:oy0 + CROP_SIZE, ox0:ox0 + CROP_SIZE] = refined_or
    else:
        oracle_pred = coarse

    # ── patchicl ──
    picl_pred = _run_model(picl, tgt_img, ctx_img, ctx_mask)

    return {
        "baseline": dice(coarse,       tgt_mask_gt),
        "native":   dice(native_pred,  tgt_mask_gt),
        "twostage": dice(twostage_pred, tgt_mask_gt),
        "oracle":   dice(oracle_pred,  tgt_mask_gt),
        "patchicl": dice(picl_pred,    tgt_mask_gt),
    }


# ── experiment ────────────────────────────────────────────────────────────────

def run_dataset(useg_128, useg_native, picl, imgs, masks, pairs):
    scores = {m: [] for m in METHODS}
    for ti, ci in pairs:
        tgt_img  = normalize(imgs[ti])
        tgt_mask = (masks[ti] > 0).astype(np.float32)
        ctx_img  = normalize(imgs[ci])
        ctx_mask = (masks[ci] > 0).astype(np.float32)
        if centroid(tgt_mask) is None or centroid(ctx_mask) is None:
            continue
        pair_scores = predict_all(useg_128, useg_native, picl,
                                  tgt_img, ctx_img, ctx_mask, tgt_mask)
        for m, v in pair_scores.items():
            scores[m].append(v)
    return {m: np.array(v) for m, v in scores.items()}


# ── analysis & plotting ───────────────────────────────────────────────────────

def summarise(all_results):
    """Print a ranked summary table."""
    rows = []
    for ds, res in all_results.items():
        row = {"dataset": ds, "n": len(res["baseline"])}
        for m in METHODS:
            row[m] = float(np.nanmean(res[m])) if len(res[m]) > 0 else float("nan")
        rows.append(row)

    rows.sort(key=lambda r: r["baseline"], reverse=True)

    header = f"{'Dataset':>22}  {'n':>4}  " + "  ".join(f"{m:>10}" for m in METHODS)
    print("\n" + header)
    print("-" * len(header))
    for r in rows:
        vals = "  ".join(f"{r[m]:>10.4f}" for m in METHODS)
        print(f"{r['dataset']:>22}  {r['n']:>4}  {vals}")

    # Aggregate
    print("\nMean across all datasets:")
    for m in METHODS:
        vals = [r[m] for r in rows if not np.isnan(r[m])]
        print(f"  {m:>10}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    return rows


def plot_heatmap(all_results, size, save_dir):
    rows = sorted(all_results.keys())
    data = np.full((len(rows), len(METHODS)), np.nan)
    for i, ds in enumerate(rows):
        for j, m in enumerate(METHODS):
            v = all_results[ds][m]
            if len(v) > 0:
                data[i, j] = np.nanmean(v)

    # Sort datasets by baseline dice
    order = np.argsort(data[:, 0])[::-1]
    data  = data[order]
    rows  = [rows[i] for i in order]

    col_labels = ["baseline\n(→128)", f"native\n(→{size})", "two-stage\n(crop)", "oracle\n(crop GT)", "PatchICL v3"]

    fig, axes = plt.subplots(1, 2, figsize=(16, max(6, len(rows) * 0.38 + 2)),
                             gridspec_kw={"width_ratios": [3, 1]})

    # ── heatmap ──────────────────────────────────────────────────────────────
    ax = axes[0]
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1,
                   interpolation="nearest")
    ax.set_xticks(range(len(METHODS)))
    ax.set_xticklabels(col_labels, fontsize=9)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, fontsize=8)
    ax.set_title(f"Mean Dice  ({size}×{size} inputs, S=1)", fontsize=11, fontweight="bold")
    for i in range(len(rows)):
        for j in range(len(METHODS)):
            v = data[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=6.5, color="black" if 0.3 < v < 0.7 else "white")
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)

    # ── improvement over baseline ─────────────────────────────────────────────
    ax2 = axes[1]
    delta_methods = ["native", "twostage", "oracle", "patchicl"]
    delta_labels  = [f"native\n−base", "two-stage\n−base", "oracle\n−base", "PatchICL\n−base"]
    delta = data[:, [METHODS.index(m) for m in delta_methods]] - data[:, [0]]
    vlim = np.nanmax(np.abs(delta))
    im2 = ax2.imshow(delta, aspect="auto", cmap="RdBu", vmin=-vlim, vmax=vlim,
                     interpolation="nearest")
    ax2.set_xticks(range(len(delta_methods)))
    ax2.set_xticklabels(delta_labels, fontsize=9)
    ax2.set_yticks(range(len(rows)))
    ax2.set_yticklabels([], fontsize=8)
    ax2.set_title("Δ vs baseline", fontsize=11, fontweight="bold")
    for i in range(len(rows)):
        for j, dm in enumerate(delta_methods):
            v = delta[i, j]
            if not np.isnan(v):
                ax2.text(j, i, f"{v:+.2f}", ha="center", va="center",
                         fontsize=6.5, color="black" if abs(v) < vlim * 0.6 else "white")
    plt.colorbar(im2, ax=ax2, fraction=0.06, pad=0.02)

    plt.tight_layout()
    path = os.path.join(save_dir, f"universeg_twostage_{size}.pdf")
    plt.savefig(path, bbox_inches="tight")
    print(f"Figure saved → {path}")


def plot_aggregate(all_results, size, save_dir):
    """Box plot aggregating all datasets."""
    fig, ax = plt.subplots(figsize=(9, 4))
    data_per_method = []
    for m in METHODS:
        vals = []
        for res in all_results.values():
            vals.extend(res[m].tolist())
        data_per_method.append(vals)

    colors = ["steelblue", "royalblue", "coral", "mediumseagreen", "goldenrod"]
    bp = ax.boxplot(data_per_method, patch_artist=True, notch=False,
                    medianprops={"color": "black", "linewidth": 2})
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    col_labels = [f"baseline\n(→128)", f"native\n(→{size})", "two-stage\n(crop)", "oracle\n(crop GT)", "PatchICL v3"]
    ax.set_xticklabels(col_labels, fontsize=9)
    ax.set_ylabel("Dice", fontsize=10)
    ax.set_title(
        f"Dice distribution across all datasets  ({size}×{size} inputs, S=1)\n"
        f"{len(all_results)} datasets",
        fontsize=11, fontweight="bold",
    )
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    plt.tight_layout()
    path = os.path.join(save_dir, f"universeg_twostage_{size}_aggregate.pdf")
    plt.savefig(path, bbox_inches="tight")
    print(f"Figure saved → {path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size",    type=int, default=256, choices=[128, 256, 512],
                        help="Input image size (loads {dataset}_{size}.npz)")
    parser.add_argument("--n-pairs", type=int, default=20,
                        help="Pairs evaluated per dataset")
    parser.add_argument("--save-dir", default="results/presentation",
                        help="Directory for figures and pickled results")
    parser.add_argument("--load", action="store_true",
                        help="Load previously saved results and replot/analyse")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    results_path = os.path.join(args.save_dir, f"twostage_results_{args.size}.pkl")

    if args.load:
        print(f"Loading results from {results_path}")
        with open(results_path, "rb") as f:
            all_results = pickle.load(f)
    else:
        datasets = discover_datasets(args.size)
        print(f"Device: {DEVICE}")
        print(f"Input size: {args.size}×{args.size}")
        print(f"Datasets found: {len(datasets)}")
        print(f"Pairs per dataset: {args.n_pairs}\n")

        print("Loading models...")
        useg_128    = load_universeg(128)
        useg_native = load_universeg(args.size)
        picl        = load_patchicl()

        all_results = {}
        for di, ds_name in enumerate(datasets):
            print(f"\n[{di+1}/{len(datasets)}] {ds_name}")
            try:
                imgs, masks = load_dataset(ds_name, args.size)
            except Exception as e:
                print(f"  SKIP: {e}")
                continue

            fg = [i for i in range(len(masks)) if masks[i].max() > 0]
            if len(fg) < 2:
                print(f"  SKIP: only {len(fg)} fg samples")
                continue

            n_pairs = min(args.n_pairs, len(fg) - 1)
            print(f"  fg={len(fg)}, pairs={n_pairs}")

            random.seed(SEED)
            pairs = [tuple(random.sample(fg, 2)) for _ in range(n_pairs)]

            res = run_dataset(useg_128, useg_native, picl, imgs, masks, pairs)
            all_results[ds_name] = res

            means = {m: np.nanmean(res[m]) for m in METHODS}
            print("  " + "  ".join(f"{m}={means[m]:.3f}" for m in METHODS))

        with open(results_path, "wb") as f:
            pickle.dump(all_results, f)
        print(f"\nResults saved → {results_path}")

    summarise(all_results)
    plot_heatmap(all_results, args.size, args.save_dir)
    plot_aggregate(all_results, args.size, args.save_dir)


if __name__ == "__main__":
    main()
