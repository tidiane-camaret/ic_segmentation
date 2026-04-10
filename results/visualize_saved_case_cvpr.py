"""CVPR-grade patch selection visualization.

Layout per example: 3 rows (levels) x 3 columns [Target + patches, Prediction, GT]
Two examples side-by-side (success + failure) in a single figure.

Usage:
    python visualize_case_cvpr.py case_dir_win case_dir_loss -o imgs/figures/patch_selection_qualitative.pdf
    python visualize_case_cvpr.py case_dir_single -o imgs/figures/patch_selection_single.pdf
"""

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from scipy.ndimage import zoom

# ---------------------------------------------------------------------------
# Style constants – tweak these to taste
# ---------------------------------------------------------------------------
PRED_COLOR = "#8fa3fd"  # green prediction contour + fill
PRED_ALPHA = 0.25  # fill opacity for prediction
GT_COLOR = "#6f56fd"  # red/crimson GT contour + fill
PATCH_COLOR = "#ffffb2"  # yellow patch boxes
PATCH_BG_COLOR = "black"  # shadow outline for patch boxes
CONTOUR_LW = 1.2  # contour line width
PATCH_LW = 0.5  # patch box line width
PATCH_SHADOW_LW = 0.5  # shadow behind patch box
FONT_FAMILY = "DejaVu Sans"
LABEL_FONTSIZE = 7  # row / column labels
TITLE_FONTSIZE = 8  # example titles (structure + dice)
DPI = 300
CELL_SIZE = 0.95  # inches per cell


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def draw_patch_boxes(ax, level_res, coords, patch_size, color=PATCH_COLOR,
                     lw=PATCH_LW, shadow_color=PATCH_BG_COLOR, shadow_lw=PATCH_SHADOW_LW):
    """Draw patch bounding boxes with a dark shadow for contrast."""
    if coords is None or len(coords) == 0:
        return
    for k in range(coords.shape[0]):
        r, c = int(coords[k, 0]), int(coords[k, 1])
        # Shadow rectangle (slightly thicker, dark)
        ax.add_patch(Rectangle(
            (c, r), patch_size, patch_size,
            linewidth=shadow_lw, edgecolor=shadow_color, facecolor="none",
        ))
        # Foreground rectangle
        ax.add_patch(Rectangle(
            (c, r), patch_size, patch_size,
            linewidth=lw, edgecolor=color, facecolor="none",
        ))


def downsample(arr, target_res):
    """Downsample 2D array to target resolution with bilinear interpolation."""
    if arr is None:
        return None
    H, W = arr.shape[:2]
    if H == target_res and W == target_res:
        return arr
    return zoom(arr, (target_res / H, target_res / W), order=1)


def load_case(case_dir: Path) -> dict:
    """Load all saved data for a case."""
    case_dir = Path(case_dir)
    data = {}
    if (case_dir / "metadata.npz").exists():
        meta = np.load(case_dir / "metadata.npz", allow_pickle=True)
        data["metadata"] = {
            k: meta[k].item() if meta[k].ndim == 0 else meta[k] for k in meta.files
        }
    for name in ["img", "gt_mask", "pred_mask"]:
        path = case_dir / f"{name}.npy"
        if path.exists():
            data[name] = np.load(path)
    data["levels"] = []
    li = 0
    while True:
        lp = case_dir / f"level{li}_data.npz"
        if not lp.exists():
            break
        level_data = np.load(lp, allow_pickle=True)
        data["levels"].append({k: level_data[k] for k in level_data.files})
        li += 1
    return data


def make_label(metadata: dict) -> str:
    """Build a compact label like 'rib_left_9 (Dice 0.72)'."""
    label_id = metadata.get("label_id", "unknown")
    # Clean up underscores for display
    name = label_id.replace("_", " ")
    dice = metadata.get("dice", None)
    if dice is not None:
        return f"{name}  (Dice {dice:.2f})"
    return name


# ---------------------------------------------------------------------------
# Core rendering: one example (3 levels x 3 columns)
# ---------------------------------------------------------------------------
def render_example(axes, data, max_levels=3, show_col_titles=False, example_title=None):
    """Render one example into a (n_levels x 3) block of axes.

    Columns: [Target + patch boxes | Prediction overlay | GT overlay]
    """
    img = data.get("img")
    gt_mask = data.get("gt_mask")
    levels = data.get("levels", [])[:max_levels]
    n_levels = len(levels)

    for li, level in enumerate(levels):
        level_res = int(level.get("level_res", 32))
        patch_size = int(level.get("patch_size", 16))
        target_coords = level.get("target_coords")
        combined_pred = level.get("combined_pred")

        img_ds = downsample(img, level_res)
        gt_ds = downsample(gt_mask, level_res)

        ax_target = axes[li][0]
        ax_pred = axes[li][1]
        ax_gt = axes[li][2]

        # --- Column 1: Target + patch boxes ---
        ax_target.imshow(img_ds, cmap="gray", interpolation="none")
        draw_patch_boxes(ax_target, level_res, target_coords, patch_size)

        # --- Column 2: Prediction overlay ---
        ax_pred.imshow(img_ds, cmap="gray", interpolation="none")
        if combined_pred is not None:
            comb = combined_pred.squeeze()
            # Semi-transparent fill
            pred_rgba = np.zeros((*comb.shape, 4))
            pred_rgba[..., :3] = mpl.colors.to_rgb(PRED_COLOR)
            pred_rgba[..., 3] = (comb > 0.5).astype(float) * PRED_ALPHA
            ax_pred.imshow(pred_rgba, interpolation="none")
            # Contour
            ax_pred.contour(comb, levels=[0.5], colors=[PRED_COLOR], linewidths=CONTOUR_LW)
        # GT contour for reference
        if gt_ds is not None:
            ax_gt.contour(gt_ds, levels=[0.5], colors=[GT_COLOR],
                          linewidths=CONTOUR_LW, linestyles="dashed")

        # --- Column 3: GT overlay ---
        ax_gt.imshow(img_ds, cmap="gray", interpolation="none")
        if gt_ds is not None:
            gt_rgba = np.zeros((*gt_ds.shape, 4))
            gt_rgba[..., :3] = mpl.colors.to_rgb(GT_COLOR)
            gt_rgba[..., 3] = (gt_ds > 0.5).astype(float) * PRED_ALPHA
            ax_gt.imshow(gt_rgba, interpolation="none")
            ax_gt.contour(gt_ds, levels=[0.5], colors=[GT_COLOR], linewidths=CONTOUR_LW)

        # Row label on the left
        ax_target.text(
            -0.05, 0.5, f"L{li+1} ({level_res})",
            transform=ax_target.transAxes, fontsize=LABEL_FONTSIZE,
            fontfamily=FONT_FAMILY, va="center", ha="right", rotation=90,
        )

        # Clean up all axes
        for ax in [ax_target, ax_pred, ax_gt]:
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

    # Column titles (only on first row)
    if show_col_titles:
        col_titles = ["Target", "Prediction", "Ground truth"]
        for ci, title in enumerate(col_titles):
            axes[0][ci].set_title(title, fontsize=LABEL_FONTSIZE, fontfamily=FONT_FAMILY,
                                  pad=3)

    # Example title above the block
    if example_title:
        axes[0][1].text(
            0.5, 1.25, example_title,
            transform=axes[0][1].transAxes, fontsize=TITLE_FONTSIZE,
            fontfamily=FONT_FAMILY, fontweight="bold",
            va="bottom", ha="center",
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def visualize_single(case_dir: Path, save_path: Path = None, max_levels: int = 3):
    """Single-example figure (one column-width panel)."""
    data = load_case(case_dir)
    n_levels = min(len(data.get("levels", [])), max_levels)

    if n_levels == 0:
        case_dir = Path(case_dir)
        print(f"ERROR: No level data found in {case_dir.resolve()}")
        print(f"  Contents: {sorted(p.name for p in case_dir.iterdir()) if case_dir.is_dir() else 'NOT A DIRECTORY'}")
        print(f"  Expected files: level0_data.npz, level1_data.npz, ...")
        return

    n_cols = 3

    fig, axes = plt.subplots(
        n_levels, n_cols,
        figsize=(CELL_SIZE * n_cols, CELL_SIZE * n_levels),
    )
    if n_levels == 1:
        axes = [axes]

    title = make_label(data.get("metadata", {}))
    render_example(axes, data, max_levels=max_levels, show_col_titles=True,
                   example_title=title)

    fig.subplots_adjust(wspace=0.03, hspace=0.06)
    _save_or_show(fig, save_path)


def visualize_pair(case_dir_win: Path, case_dir_loss: Path,
                   save_path: Path = None, max_levels: int = 3):
    """Two-example figure (win + loss) for the paper."""
    data_win = load_case(case_dir_win)
    data_loss = load_case(case_dir_loss)

    n_levels = min(
        len(data_win.get("levels", [])),
        len(data_loss.get("levels", [])),
        max_levels,
    )

    if n_levels == 0:
        for label, d in [("win", case_dir_win), ("loss", case_dir_loss)]:
            d = Path(d)
            n = len(load_case(d).get("levels", []))
            print(f"  {label}: {d.resolve()} -> {n} levels found")
        print("ERROR: Need at least 1 level in both cases.")
        return

    # 3 columns per example, thin gap column in between
    n_cols = 3 + 1 + 3  # win(3) + spacer(1) + loss(3)

    fig, axes_flat = plt.subplots(
        n_levels, n_cols,
        figsize=(CELL_SIZE * 3 * 2 + 0.3, CELL_SIZE * n_levels),
        gridspec_kw={"width_ratios": [1, 1, 1, 0.08, 1, 1, 1]},
    )
    if n_levels == 1:
        axes_flat = [axes_flat]

    # Hide spacer column
    for row in axes_flat:
        row[3].set_visible(False)

    # Split into left (win) and right (loss) axes
    axes_win = [row[:3] for row in axes_flat]
    axes_loss = [row[4:] for row in axes_flat]

    title_win = make_label(data_win.get("metadata", {}))
    title_loss = make_label(data_loss.get("metadata", {}))

    render_example(axes_win, data_win, max_levels=n_levels,
                   show_col_titles=True, example_title=title_win)
    render_example(axes_loss, data_loss, max_levels=n_levels,
                   show_col_titles=True, example_title=title_loss)

    fig.subplots_adjust(wspace=0.03, hspace=0.06)
    _save_or_show(fig, save_path)


def _save_or_show(fig, save_path):
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
        description="CVPR-grade patch selection visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single case
  python visualize_case_cvpr.py case_dir -o figure.pdf

  # Win/loss pair
  python visualize_case_cvpr.py case_win case_loss -o figure.pdf
""",
    )
    parser.add_argument("case_dirs", nargs="+", type=str,
                        help="1 or 2 case directories (single or win/loss pair)")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Output path (PDF recommended). Omit to show interactively.")
    parser.add_argument("--max-levels", type=int, default=3,
                        help="Max cascade levels to show (default: 3)")
    args = parser.parse_args()

    if len(args.case_dirs) == 1:
        visualize_single(args.case_dirs[0], args.output, args.max_levels)
    elif len(args.case_dirs) == 2:
        visualize_pair(args.case_dirs[0], args.case_dirs[1], args.output, args.max_levels)
    else:
        parser.error("Provide 1 (single) or 2 (win/loss pair) case directories.")