"""
Sample and plot a few (target image, target mask, context image, context mask)
for a given label/dataset.

Usage — TotalSeg:
  python results/presentation/slices/plot_slices.py --source totalseg --label liver --n 6
  python results/presentation/slices/plot_slices.py --source totalseg --label liver --n 4 --size 256

Usage — MedSegBench:
  python results/presentation/slices/plot_slices.py --source medsegbench --ds-name skin_lesion --n 6
  python results/presentation/slices/plot_slices.py --source medsegbench --ds-name skin_lesion --n 6 --size 256
"""

import argparse
import os
import sys
from pathlib import Path
import random

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, "/home/dpxuser/ic_segmentation")

DATA_DIR      = "/nfs/data/nii/data1/Analysis/camaret___in_context_segmentation/ANALYSIS_20251122/data"
MEDSEG_ROOT   = str(Path(DATA_DIR) / "medsegbench")
SEED          = 42


# ── medsegbench helpers (mirrored from universeg_twostage.py) ─────────────────

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


def load_medseg_dataset(ds_name: str, size: int):
    npz      = np.load(os.path.join(MEDSEG_ROOT, f"{ds_name}_{size}.npz"))
    keys     = list(npz.keys())
    img_key  = next(k for k in keys if "test" in k and ("image" in k or "img" in k))
    mask_key = next(k for k in keys if "test" in k and ("label" in k or "mask" in k))
    imgs  = np.stack([to_gray(npz[img_key][i]) for i in range(len(npz[img_key]))])
    masks = npz[mask_key]
    return imgs, masks


def discover_medseg_datasets(size: int):
    return sorted(
        f.replace(f"_{size}.npz", "")
        for f in os.listdir(MEDSEG_ROOT)
        if f.endswith(f"_{size}.npz") and not f.startswith("medsegbench")
    )


# ── shared plotting ───────────────────────────────────────────────────────────

def make_figure(rows_data, title, save_dir, filename):
    """rows_data: list of (tgt_img, tgt_mask, ctx_img, ctx_mask) arrays."""
    ncols  = 3
    nrows  = len(rows_data)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 3.5))
    if nrows == 1:
        axes = [axes]

    col_titles = ["Target image", "Target mask", "Context image + mask"]
    for col, t in enumerate(col_titles):
        axes[0][col].set_title(t, fontsize=11, fontweight="bold")

    for row, (tgt_img, tgt_mask, ctx_img, ctx_mask) in enumerate(rows_data):
        ax_tgt, ax_tgt_msk, ax_ctx = axes[row]
        ax_tgt.imshow(tgt_img,      cmap="gray", vmin=0, vmax=1)
        ax_tgt_msk.imshow(tgt_mask, cmap="gray", vmin=0, vmax=1)
        ax_ctx.imshow(ctx_img,      cmap="gray", vmin=0, vmax=1)
        ax_ctx.imshow(np.ma.masked_where(ctx_mask == 0, ctx_mask), cmap="autumn", vmin=0, vmax=1, alpha=0.7)
        for ax in (ax_tgt, ax_tgt_msk, ax_ctx):
            ax.axis("off")

    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    for ext, kw in [(".pdf", {}), (".png", {"dpi": 150})]:
        path = os.path.join(save_dir, filename + ext)
        plt.savefig(path, bbox_inches="tight", **kw)
        print(f"Figure saved → {path}")

    plt.close(fig)


# ── loaders ───────────────────────────────────────────────────────────────────

def load_totalseg(args):
    from src.dataloaders.totalseg2d_zopt_dataloader import TotalSeg2DZOptDataset

    root_dir   = str(Path(DATA_DIR) / "totalseg_3d_zopt")
    stats_path = str(Path(DATA_DIR) / "totalseg_3d_zopt" / "stats.pkl")

    dataset = TotalSeg2DZOptDataset(
        root_dir=root_dir,
        stats_path=stats_path,
        label_id_list=[args.label],
        image_size=(args.size, args.size),
        context_size=1,
        split=["val", "test"],
        random_context=False,
        augment=False,
        min_coverage=30,
        min_coverage_ratio=0.0,
        same_case_context_ratio=1.0,
        max_slices_per_group=2,
        slice_selection="stride_peak",
    )
    print(f"TotalSeg dataset size: {len(dataset)} samples for label '{args.label}'")

    indices = [i for i, (_, lid, _) in enumerate(dataset.samples) if lid == args.label]
    random.shuffle(indices)
    indices = indices[: args.n]

    rows = []
    for idx in indices:
        s = dataset[idx]
        rows.append((
            s["image"][0].numpy(),
            s["label"][0].numpy(),
            s["context_in"][0, 0].numpy(),
            s["context_out"][0, 0].numpy(),
        ))
    return rows, f"TotalSeg — {args.label}  |  {args.size}×{args.size}", f"slices_totalseg_{args.label}_{args.size}"


def load_medseg(args):
    if args.ds_name == "list":
        print("Available medsegbench datasets:")
        for ds in discover_medseg_datasets(args.size):
            print(f"  {ds}")
        sys.exit(0)

    imgs, masks = load_medseg_dataset(args.ds_name, args.size)
    print(f"MedSegBench '{args.ds_name}': {len(imgs)} images")

    fg = [i for i in range(len(masks)) if masks[i].max() > 0]
    pairs = []
    used  = set()
    attempts = 0
    while len(pairs) < args.n and attempts < len(fg) * 10:
        ti, ci = random.sample(fg, 2)
        if ti not in used:
            pairs.append((ti, ci))
            used.add(ti)
        attempts += 1

    rows = []
    for ti, ci in pairs:
        tgt_img  = normalize(imgs[ti])
        tgt_mask = (masks[ti] > 0).astype(np.float32)
        ctx_img  = normalize(imgs[ci])
        ctx_mask = (masks[ci] > 0).astype(np.float32)
        rows.append((tgt_img, tgt_mask, ctx_img, ctx_mask))

    return rows, f"MedSegBench — {args.ds_name}  |  {args.size}×{args.size}", f"slices_medseg_{args.ds_name}_{args.size}"


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source",   default="totalseg", choices=["totalseg", "medsegbench"],
                        help="Data source")
    parser.add_argument("--label",    default="liver",
                        help="[totalseg] label class")
    parser.add_argument("--ds-name",  default="skin_lesion",
                        help="[medsegbench] dataset name, or 'list' to print available datasets")
    parser.add_argument("--n",        type=int, default=6,   help="Number of samples to plot")
    parser.add_argument("--size",     type=int, default=256, help="Image size")
    parser.add_argument("--save-dir", default="results/presentation/slices")
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)

    if args.source == "totalseg":
        rows, title, filename = load_totalseg(args)
    else:
        rows, title, filename = load_medseg(args)

    make_figure(rows, title, args.save_dir, filename)


if __name__ == "__main__":
    main()
