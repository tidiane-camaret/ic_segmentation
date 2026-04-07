"""CVPR-grade qualitative comparison between PatchICL and UniverSeg.

Layout: N rows x 4 columns [Context, Target + GT, PatchICL, UniverSeg]
Rows grouped into PatchICL wins (green left accent) and UniverSeg wins (red left accent).

Usage:
    python visualize_comparison_cvpr.py \
        --patchicl-path /path/to/patchicl_npy \
        --universeg-path /path/to/universeg_npy \
        --cases s0013_kidney_left_None s0042_rib_left_9_None \
        --win-indices 0 1 \
        -o imgs/figures/qualitative_comparison.pdf
"""

import argparse
import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.patches import Rectangle

# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------
PRED_COLOR_PATCHICL = "#2980b9"   # blue — PatchICL throughout paper
PRED_COLOR_UNIVERSEG = "#e67e22"  # orange — UniverSeg throughout paper
GT_COLOR = "#f1c40f"              # yellow GT contour
CONTEXT_MASK_COLOR = "#f1c40f"    # yellow context mask (same as GT for consistency)
MASK_ALPHA = 0.30                 # fill opacity
CONTOUR_LW = 1.2
WIN_ACCENT = "#2980b9"            # blue accent bar for PatchICL-win rows
LOSS_ACCENT = "#e67e22"           # orange accent bar for UniverSeg-win rows
ACCENT_WIDTH = 3                  # points

FONT_FAMILY = "DejaVu Sans"
LABEL_FONTSIZE = 7
TITLE_FONTSIZE = 8
DPI = 300
CELL_SIZE = 1.0  # inches per cell


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_nifti_2d(path):
    """Load a NIfTI file and collapse to 2D if needed (take middle slice)."""
    data = nib.load(path).get_fdata()
    if data.ndim == 3:
        data = data[:, :, data.shape[2] // 2]
    return data


def create_rgba(mask, color_hex, alpha):
    """Create an RGBA overlay from a binary mask."""
    rgb = mpl.colors.to_rgb(color_hex)
    rgba = np.zeros((*mask.shape, 4))
    rgba[..., :3] = rgb
    rgba[..., 3] = (mask > 0.5).astype(float) * alpha
    return rgba


def overlay_with_contour(ax, img, mask, color_hex, alpha=MASK_ALPHA, contour_lw=CONTOUR_LW):
    """Show grayscale image with colored mask overlay and contour."""
    ax.imshow(img, cmap="gray", interpolation="none")
    if mask is not None and mask.max() > 0:
        ax.imshow(create_rgba(mask, color_hex, alpha), interpolation="none")
        ax.contour(mask, levels=[0.5], colors=[color_hex], linewidths=contour_lw)


def clean_axis(ax):
    """Remove ticks and spines."""
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def add_accent_bar(ax, color, side="left", width=ACCENT_WIDTH):
    """Add a thin colored bar on the left edge of an axis to indicate win/loss."""
    spine = ax.spines[side]
    spine.set_visible(True)
    spine.set_edgecolor(color)
    spine.set_linewidth(width)


def format_label(case_name):
    """Convert case_name to a readable label, e.g. 's0013_kidney_left_None' -> 'kidney left'."""
    parts = case_name.split("_")
    # Drop subject ID prefix (s0013) and trailing None
    parts = [p for p in parts if not p.startswith("s0") and p != "None"]
    return " ".join(parts)


def dice_score(pred, gt):
    """Compute Dice coefficient between two binary masks."""
    pred = (pred > 0.5).astype(bool)
    gt = (gt > 0.5).astype(bool)
    intersection = (pred & gt).sum()
    total = pred.sum() + gt.sum()
    if total == 0:
        return 1.0 if intersection == 0 else 0.0
    return 2.0 * intersection / total


# ---------------------------------------------------------------------------
# Core figure
# ---------------------------------------------------------------------------
def plot_comparison(
    patchicl_path: str,
    universeg_path: str,
    case_names: list[str],
    win_indices: list[int],
    save_path: str = None,
    threshold: float = 0.5,
):
    """Create CVPR-grade comparison figure.

    Parameters
    ----------
    patchicl_path : path to PatchICL prediction folders
    universeg_path : path to UniverSeg prediction folders
    case_names : list of case folder names
    win_indices : indices into case_names where PatchICL wins (rest are losses)
    save_path : output path (PDF recommended)
    threshold : binarization threshold for predictions
    """
    n_rows = len(case_names)
    n_cols = 4  # Context | Target+GT | PatchICL | UniverSeg

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(CELL_SIZE * n_cols, CELL_SIZE * n_rows),
    )
    if n_rows == 1:
        axes = [axes]

    for ri, case_name in enumerate(case_names):
        is_win = ri in win_indices

        # --- Load data ---
        pi_dir = os.path.join(patchicl_path, case_name)
        us_dir = os.path.join(universeg_path, case_name)

        img = load_nifti_2d(os.path.join(pi_dir, "img.nii.gz"))
        gt_mask = load_nifti_2d(os.path.join(pi_dir, "gt_mask.nii.gz"))
        patchicl_pred = load_nifti_2d(os.path.join(pi_dir, "final_pred_mask.nii.gz"))

        context_img = load_nifti_2d(os.path.join(pi_dir, "context0_img.nii.gz"))
        context_mask = load_nifti_2d(os.path.join(pi_dir, "context0_gt_mask.nii.gz"))

        if os.path.exists(us_dir):
            universeg_pred = load_nifti_2d(os.path.join(us_dir, "final_pred_mask.nii.gz"))
        else:
            universeg_pred = np.zeros_like(gt_mask)
            print(f"WARNING: UniverSeg prediction not found for {case_name}")

        # Binarize predictions
        patchicl_binary = (patchicl_pred > threshold).astype(float)
        universeg_binary = (universeg_pred > threshold).astype(float)

        # Compute Dice scores
        gt_binary = (gt_mask > 0.5).astype(float)
        dice_pi = dice_score(patchicl_binary, gt_binary)
        dice_us = dice_score(universeg_binary, gt_binary)
        print(f"{case_name:40s}  PatchICL: {dice_pi:.3f}  UniverSeg: {dice_us:.3f}  "
              f"{'WIN' if is_win else 'LOSS'}")

        row = axes[ri]

        # --- Col 0: Context image + mask ---
        overlay_with_contour(row[0], context_img, context_mask, CONTEXT_MASK_COLOR)

        # --- Col 1: Target + GT ---
        overlay_with_contour(row[1], img, gt_mask, GT_COLOR)

        # --- Col 2: PatchICL prediction (+ GT contour for reference) ---
        overlay_with_contour(row[2], img, patchicl_binary, PRED_COLOR_PATCHICL)
        if gt_mask is not None and gt_mask.max() > 0:
            row[2].contour(gt_mask, levels=[0.5], colors=[GT_COLOR],
                           linewidths=CONTOUR_LW * 0.8, linestyles="dashed")
        # Dice annotation
        row[2].text(
            0.5, 0.02, f"{dice_pi:.2f}",
            transform=row[2].transAxes, fontsize=LABEL_FONTSIZE,
            fontfamily=FONT_FAMILY, va="bottom", ha="center",
            color="white", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="black", alpha=0.6, edgecolor="none"),
        )

        # --- Col 3: UniverSeg prediction (+ GT contour for reference) ---
        overlay_with_contour(row[3], img, universeg_binary, PRED_COLOR_UNIVERSEG)
        if gt_mask is not None and gt_mask.max() > 0:
            row[3].contour(gt_mask, levels=[0.5], colors=[GT_COLOR],
                           linewidths=CONTOUR_LW * 0.8, linestyles="dashed")
        # Dice annotation
        row[3].text(
            0.5, 0.02, f"{dice_us:.2f}",
            transform=row[3].transAxes, fontsize=LABEL_FONTSIZE,
            fontfamily=FONT_FAMILY, va="bottom", ha="center",
            color="white", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="black", alpha=0.6, edgecolor="none"),
        )

        # --- Accent bar ---
        accent_color = WIN_ACCENT if is_win else LOSS_ACCENT
        add_accent_bar(row[0], accent_color)

        # Clean all axes
        for ax in row:
            clean_axis(ax)

    # --- Column titles ---
    col_titles = ["Context", "Target + GT", "PatchICL", "UniverSeg"]
    for ci, title in enumerate(col_titles):
        axes[0][ci].set_title(title, fontsize=LABEL_FONTSIZE, fontfamily=FONT_FAMILY, pad=4)

    fig.subplots_adjust(wspace=0.03, hspace=0.06)

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=DPI, bbox_inches="tight", pad_inches=0.02,
                    transparent=False, facecolor="white")
        print(f"Saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CVPR-grade PatchICL vs UniverSeg comparison figure",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python visualize_comparison_cvpr.py \\
      --patchicl-path results/patchicl_nii \\
      --universeg-path results/universeg_nii \\
      --cases s0013_rib_left_9_None s0042_skull_None \\
             s0007_vertebrae_T8_None s0021_iliac_artery_left_None \\
      --win-indices 0 1 \\
      -o imgs/figures/qualitative_comparison.pdf
""",
    )
    #parser.add_argument("--patchicl-path", required=True, help="Path to PatchICL prediction folders")
    #parser.add_argument("--universeg-path", required=True, help="Path to UniverSeg prediction folders")
    parser.add_argument("--cases", nargs="+", required=True, help="Case folder names to plot (in row order)")
    parser.add_argument("--win-indices", nargs="+", type=int, default=[],
                        help="Row indices (0-based) where PatchICL wins (rest are losses)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Binarization threshold")
    parser.add_argument("-o", "--output", type=str, default=None, help="Output path (PDF recommended)")
    args = parser.parse_args()
    args.patchicl_path =  "/nfs/data/nii/data1/Analysis/camaret___in_context_segmentation/ANALYSIS_20251122/results/totalsegmri2d_patch_icl/" 
    args.universeg_path = "/nfs/data/nii/data1/Analysis/camaret___in_context_segmentation/ANALYSIS_20251122/results/totalsegmri2d_universeg/" 
    plot_comparison(
        patchicl_path=args.patchicl_path,
        universeg_path=args.universeg_path,
        case_names=args.cases,
        win_indices=args.win_indices,
        save_path=args.output,
        threshold=args.threshold,
    )