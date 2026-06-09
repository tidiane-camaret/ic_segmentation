"""
UniverSeg ablation on TotalSeg val labels (same conditions as universeg_twostage.py).

Conditions (all at --size resolution):
  baseline   — UniverSeg resize-to-128 internally
  native     — UniverSeg at full --size resolution
  twostage   — stage1 baseline → stage2 128-px crop on pred centroid
  oracle     — stage2 with target crop on GT centroid (upper bound)
  patchicl   — PatchICL v3 at full --size resolution

Data: TotalSeg zopt format, val label split, val+test case split, S=1 context.

Usage:
  python results/presentation/universeg_totalseg.py --size 256
  python results/presentation/universeg_totalseg.py --size 256 --load
"""

import argparse
import os
import pickle
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

sys.path.insert(0, "/home/dpxuser/repos/UniverSeg")
sys.path.insert(0, "/home/dpxuser/ic_segmentation")

DATA_DIR  = "/nfs/data/nii/data1/Analysis/camaret___in_context_segmentation/ANALYSIS_20251122/data"
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


def load_patchicl(context_size=3):
    from src.model_builder import build_patch_icl_model
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    cfg  = OmegaConf.create(ckpt["config"])
    cfg.checkpoint   = CKPT_PATH
    cfg.feature_mode = "on_the_fly"
    model = build_patch_icl_model(cfg, DEVICE, context_size=context_size, verbose=False).to(DEVICE).eval()
    n = sum(p.numel() for p in model.parameters())
    print(f"  PatchICL v3: {n:,} params")
    return model


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

def _run_model_multi(model, tgt_img, ctx_imgs_all, ctx_masks_all) -> np.ndarray:
    """Run model with S context images at once (UniverSeg multi-support)."""
    S, H, W = ctx_imgs_all.shape
    tgt_t   = to_tensor(tgt_img)                                             # (1,1,H,W)
    ctx_in  = torch.from_numpy(ctx_imgs_all).float().unsqueeze(1).to(DEVICE) # (S,1,H,W)
    ctx_out = torch.from_numpy(ctx_masks_all).float().unsqueeze(1).to(DEVICE)# (S,1,H,W)
    with torch.no_grad():
        out = model(tgt_t, context_in=ctx_in.unsqueeze(0),
                    context_out=ctx_out.unsqueeze(0), mode="val")
    return (out["final_logit"].squeeze() > 0).cpu().numpy().astype(np.float32)


def predict_all(useg_128, useg_native, picl, tgt_img, ctx_img, ctx_mask, tgt_mask_gt,
                ctx_imgs_all=None, ctx_masks_all=None):
    """Run all five conditions for one (target, context) pair.

    All UniverSeg and PatchICL conditions receive all S context images.
    Two-stage/oracle use the first context slot for the crop centroid anchor.
    """
    # Determine how to call UniverSeg: multi-support when S>1, single otherwise
    if ctx_imgs_all is not None and len(ctx_imgs_all) > 1:
        coarse       = _run_model_multi(useg_128,    tgt_img, ctx_imgs_all, ctx_masks_all)
        native_pred  = _run_model_multi(useg_native, tgt_img, ctx_imgs_all, ctx_masks_all)
    else:
        coarse       = _run_model(useg_128,    tgt_img, ctx_img, ctx_mask)
        native_pred  = _run_model(useg_native, tgt_img, ctx_img, ctx_mask)

    # Two-stage/oracle use first context slot centroid as anchor
    ctx_c = centroid(ctx_mask)
    if ctx_c is not None:
        ctx_img_crop, cy0, cx0 = crop_centered(ctx_img, ctx_c[0], ctx_c[1])
        ctx_mask_crop = ctx_mask[cy0:cy0 + CROP_SIZE, cx0:cx0 + CROP_SIZE]
    else:
        ctx_img_crop  = ctx_img[:CROP_SIZE, :CROP_SIZE]
        ctx_mask_crop = ctx_mask[:CROP_SIZE, :CROP_SIZE]

    # twostage
    tgt_c_pred = centroid(coarse)
    if tgt_c_pred is not None:
        tgt_crop, ty0, tx0 = crop_centered(tgt_img, tgt_c_pred[0], tgt_c_pred[1])
        refined = _run_model(useg_128, tgt_crop, ctx_img_crop, ctx_mask_crop)
        twostage_pred = np.zeros_like(coarse)
        twostage_pred[ty0:ty0 + CROP_SIZE, tx0:tx0 + CROP_SIZE] = refined
    else:
        twostage_pred = coarse

    # oracle
    tgt_c_gt = centroid(tgt_mask_gt)
    if tgt_c_gt is not None:
        tgt_crop_or, oy0, ox0 = crop_centered(tgt_img, tgt_c_gt[0], tgt_c_gt[1])
        refined_or = _run_model(useg_128, tgt_crop_or, ctx_img_crop, ctx_mask_crop)
        oracle_pred = np.zeros_like(coarse)
        oracle_pred[oy0:oy0 + CROP_SIZE, ox0:ox0 + CROP_SIZE] = refined_or
    else:
        oracle_pred = coarse

    # patchicl — uses all S context pairs
    if ctx_imgs_all is not None and len(ctx_imgs_all) > 1:
        tgt_t = to_tensor(tgt_img)
        ctx_in  = torch.from_numpy(ctx_imgs_all).float().unsqueeze(1).to(DEVICE)   # (S,1,H,W)
        ctx_out = torch.from_numpy(ctx_masks_all).float().unsqueeze(1).to(DEVICE)
        with torch.no_grad():
            out = picl(tgt_t, context_in=ctx_in.unsqueeze(0), context_out=ctx_out.unsqueeze(0), mode="val")
        picl_pred = (out["final_logit"].squeeze() > 0).cpu().numpy().astype(np.float32)
    else:
        picl_pred = _run_model(picl, tgt_img, ctx_img, ctx_mask)

    return {
        "baseline": dice(coarse,        tgt_mask_gt),
        "native":   dice(native_pred,   tgt_mask_gt),
        "twostage": dice(twostage_pred, tgt_mask_gt),
        "oracle":   dice(oracle_pred,   tgt_mask_gt),
        "patchicl": dice(picl_pred,     tgt_mask_gt),
    }


# ── analysis & plotting ───────────────────────────────────────────────────────

def summarise(all_results):
    """Print ranked summary and means. all_results: {label_id: {method: [scores]}}"""
    rows = []
    for label_id, res in all_results.items():
        row = {"label": label_id, "n": len(res["baseline"])}
        for m in METHODS:
            row[m] = float(np.nanmean(res[m])) if len(res[m]) > 0 else float("nan")
        rows.append(row)

    rows.sort(key=lambda r: r["baseline"], reverse=True)

    header = f"{'Label':>35}  {'n':>4}  " + "  ".join(f"{m:>10}" for m in METHODS)
    print("\n" + header)
    print("-" * len(header))
    for r in rows:
        vals = "  ".join(f"{r[m]:>10.4f}" for m in METHODS)
        print(f"{r['label']:>35}  {r['n']:>4}  {vals}")

    print("\nPer-label mean (each label weighted equally):")
    for m in METHODS:
        vals = [r[m] for r in rows if not np.isnan(r[m])]
        print(f"  {m:>10}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    print("\nPer-slice mean (each sample weighted equally, as in eval.py):")
    for m in METHODS:
        all_scores = []
        for ds, res in all_results.items():
            all_scores.extend(res[m])
        print(f"  {m:>10}: {np.mean(all_scores):.4f} ± {np.std(all_scores):.4f}  (n={len(all_scores)})")

    return rows


def plot_heatmap(all_results, size, save_dir, context_size=1, label_split="val"):
    rows = sorted(all_results.keys())
    data = np.full((len(rows), len(METHODS)), np.nan)
    for i, label in enumerate(rows):
        for j, m in enumerate(METHODS):
            v = all_results[label][m]
            if len(v) > 0:
                data[i, j] = np.nanmean(v)

    order = np.argsort(data[:, 0])[::-1]
    data  = data[order]
    rows  = [rows[i] for i in order]

    col_labels = ["baseline\n(→128)", f"native\n(→{size})", "two-stage\n(crop)", "oracle\n(crop GT)", "PatchICL v3"]

    fig, axes = plt.subplots(1, 2, figsize=(16, max(6, len(rows) * 0.38 + 2)),
                             gridspec_kw={"width_ratios": [3, 1]})

    ax = axes[0]
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks(range(len(METHODS)))
    ax.set_xticklabels(col_labels, fontsize=9)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, fontsize=7)
    ax.set_title(f"TotalSeg {label_split} labels — Mean Dice  ({size}×{size}, S={context_size})", fontsize=11, fontweight="bold")
    for i in range(len(rows)):
        for j in range(len(METHODS)):
            v = data[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=6, color="black" if 0.3 < v < 0.7 else "white")
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)

    ax2 = axes[1]
    delta_methods = ["native", "twostage", "oracle", "patchicl"]
    delta_labels  = ["native\n−base", "two-stage\n−base", "oracle\n−base", "PatchICL\n−base"]
    delta = data[:, [METHODS.index(m) for m in delta_methods]] - data[:, [0]]
    vlim = max(np.nanmax(np.abs(delta)), 0.01)
    im2 = ax2.imshow(delta, aspect="auto", cmap="RdBu", vmin=-vlim, vmax=vlim, interpolation="nearest")
    ax2.set_xticks(range(len(delta_methods)))
    ax2.set_xticklabels(delta_labels, fontsize=9)
    ax2.set_yticks(range(len(rows)))
    ax2.set_yticklabels([], fontsize=7)
    ax2.set_title("Δ vs baseline", fontsize=11, fontweight="bold")
    for i in range(len(rows)):
        for j in range(len(delta_methods)):
            v = delta[i, j]
            if not np.isnan(v):
                ax2.text(j, i, f"{v:+.2f}", ha="center", va="center",
                         fontsize=6, color="black" if abs(v) < vlim * 0.6 else "white")
    plt.colorbar(im2, ax=ax2, fraction=0.06, pad=0.02)

    plt.tight_layout()
    path = os.path.join(save_dir, f"universeg_totalseg_{size}_s{context_size}_{label_split}.pdf")
    plt.savefig(path, bbox_inches="tight")
    print(f"Figure saved → {path}")


def plot_aggregate(all_results, size, save_dir, context_size=1, label_split="val"):
    fig, ax = plt.subplots(figsize=(9, 4))
    data_per_method = []
    for m in METHODS:
        vals = []
        for res in all_results.values():
            vals.extend(res[m])
        data_per_method.append(vals)

    colors = ["steelblue", "royalblue", "coral", "mediumseagreen", "goldenrod"]
    bp = ax.boxplot(data_per_method, patch_artist=True,
                    medianprops={"color": "black", "linewidth": 2})
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    col_labels = ["baseline\n(→128)", f"native\n(→{size})", "two-stage\n(crop)", "oracle\n(crop GT)", "PatchICL v3"]
    ax.set_xticklabels(col_labels, fontsize=9)
    ax.set_ylabel("Dice", fontsize=10)
    ax.set_title(
        f"TotalSeg {label_split} labels — Dice distribution  ({size}×{size}, S={context_size})\n"
        f"{len(all_results)} labels",
        fontsize=11, fontweight="bold",
    )
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    plt.tight_layout()
    path = os.path.join(save_dir, f"universeg_totalseg_{size}_s{context_size}_{label_split}_aggregate.pdf")
    plt.savefig(path, bbox_inches="tight")
    print(f"Figure saved → {path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size",         type=int, default=256,
                        help="Input image size fed to models")
    parser.add_argument("--label-split",        default="val", choices=["val", "train"],
                        help="Label split to evaluate: val (held-out) or train (seen during training)")
    parser.add_argument("--context-size",       type=int, default=3,
                        help="Number of context images (S)")
    parser.add_argument("--n-pairs",            type=int, default=20,
                        help="Max pairs evaluated per label")
    parser.add_argument("--max-slices-per-group", type=int, default=10,
                        help="Max slices per (case,label) group — 10 matches March-20 eval conditions")
    parser.add_argument("--slice-selection",    default="all",
                        choices=["all", "random", "stride", "stride_peak"],
                        help="Slice subsampling strategy — stride_peak matches eval.py")
    parser.add_argument("--save-dir",     default="results/presentation",
                        help="Directory for figures and pickled results")
    parser.add_argument("--load", action="store_true",
                        help="Reload saved results and replot without re-running")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    results_path = os.path.join(args.save_dir, f"totalseg_results_{args.size}_s{args.context_size}_{args.label_split}.pkl")

    if args.load:
        print(f"Loading results from {results_path}")
        with open(results_path, "rb") as f:
            all_results = pickle.load(f)
    else:
        from src.dataloaders.totalseg2d_zopt_dataloader import TotalSeg2DZOptDataset

        root_dir   = str(Path(DATA_DIR) / "totalseg_3d_zopt")
        stats_path = str(Path(DATA_DIR) / "totalseg_3d_zopt" / "stats.pkl")

        print(f"Device: {DEVICE}")
        print(f"Input size: {args.size}×{args.size}")
        print("Loading TotalSeg val dataset...")

        dataset = TotalSeg2DZOptDataset(
            root_dir=root_dir,
            stats_path=stats_path,
            label_id_list=args.label_split,
            image_size=(args.size, args.size),
            context_size=args.context_size,
            split=["val", "test"],
            random_context=False,
            augment=False,
            min_coverage=30,
            min_coverage_ratio=0.0,
            same_case_context_ratio=1.0,
            max_slices_per_group=args.max_slices_per_group,
            slice_selection=args.slice_selection,
        )
        print(f"Val dataset size: {len(dataset)} samples")

        # Group sample indices by label for efficient per-label sampling
        import random as _random
        _random.seed(SEED)
        label_to_indices = defaultdict(list)
        for idx, (case_id, label_id, z_idx) in enumerate(dataset.samples):
            label_to_indices[label_id].append(idx)
        for label_id in label_to_indices:
            _random.shuffle(label_to_indices[label_id])
        print(f"Val labels: {sorted(label_to_indices.keys())}")

        print("Loading models...")
        useg_128    = load_universeg(128)
        useg_native = load_universeg(args.size)
        picl        = load_patchicl(args.context_size)

        all_results = defaultdict(lambda: {m: [] for m in METHODS})

        for li, label_id in enumerate(sorted(label_to_indices.keys())):
            indices = label_to_indices[label_id]
            n_done  = 0
            print(f"\n[{li+1}/{len(label_to_indices)}] {label_id} ({len(indices)} slices)")
            for idx in indices:
                if n_done >= args.n_pairs:
                    break
                sample = dataset[idx]
                tgt_img  = sample["image"][0].numpy()
                tgt_mask = (sample["label"][0].numpy() > 0).astype(np.float32)
                # UniverSeg: use first context slot (S=1 style)
                ctx_img  = sample["context_in"][0, 0].numpy()
                ctx_mask = (sample["context_out"][0, 0].numpy() > 0).astype(np.float32)

                if centroid(tgt_mask) is None or centroid(ctx_mask) is None:
                    continue

                # PatchICL: use all S context slots
                ctx_imgs_all  = sample["context_in"][:, 0].numpy()
                ctx_masks_all = (sample["context_out"][:, 0].numpy() > 0).astype(np.float32)

                pair_scores = predict_all(useg_128, useg_native, picl,
                                          tgt_img, ctx_img, ctx_mask, tgt_mask,
                                          ctx_imgs_all, ctx_masks_all)
                for m, v in pair_scores.items():
                    all_results[label_id][m].append(v)
                n_done += 1

            means = {m: np.nanmean(all_results[label_id][m]) for m in METHODS}
            print("  " + "  ".join(f"{m}={means[m]:.3f}" for m in METHODS))

        # Convert to plain dict for pickling
        all_results = {k: {m: list(v) for m, v in res.items()}
                       for k, res in all_results.items()}

        with open(results_path, "wb") as f:
            pickle.dump(all_results, f)
        print(f"\nResults saved → {results_path}")

    summarise(all_results)
    plot_heatmap(all_results, args.size, args.save_dir, args.context_size, args.label_split)
    plot_aggregate(all_results, args.size, args.save_dir, args.context_size, args.label_split)


if __name__ == "__main__":
    main()
