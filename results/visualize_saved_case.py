"""Visualize a saved case from save_imgs_masks_npy output.

Layout: One row per level with columns [Ctx1, Ctx2, Target, Combined, GT]
All images shown at their native resolution (no upsampling).
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from scipy.ndimage import zoom


def draw_patch_boxes(ax, img_shape, coords, patch_size, level_res, color="red", linewidth=1):
    """Draw patch bounding boxes on image."""
    if coords is None or len(coords) == 0:
        return
    H, W = img_shape[:2]
    scale_h, scale_w = H / level_res, W / level_res
    scaled_patch_h = int(patch_size * scale_h)
    scaled_patch_w = int(patch_size * scale_w)

    for k in range(coords.shape[0]):
        r, c = coords[k, 0], coords[k, 1]
        r_start = int(r * scale_h)
        c_start = int(c * scale_w)
        rect = Rectangle(
            (c_start, r_start),
            scaled_patch_w,
            scaled_patch_h,
            linewidth=linewidth,
            edgecolor=color,
            facecolor="none",
        )
        ax.add_patch(rect)


def downsample_to_level(arr, level_res):
    """Downsample array to level resolution."""
    if arr is None:
        return None
    H, W = arr.shape[:2]
    if H == level_res and W == level_res:
        return arr
    zoom_factors = (level_res / H, level_res / W)
    return zoom(arr, zoom_factors, order=1)


def load_case(case_dir: Path) -> dict:
    """Load all saved data for a case."""
    case_dir = Path(case_dir)
    data = {}

    # Load metadata
    if (case_dir / "metadata.npz").exists():
        meta = np.load(case_dir / "metadata.npz", allow_pickle=True)
        data["metadata"] = {k: meta[k].item() if meta[k].ndim == 0 else meta[k] for k in meta.files}

    # Load images
    for name in ["img", "gt_mask", "pred_mask", "context_imgs", "context_masks"]:
        path = case_dir / f"{name}.npy"
        if path.exists():
            data[name] = np.load(path)

    # Load level data
    data["levels"] = []
    level_idx = 0
    while True:
        level_path = case_dir / f"level{level_idx}_data.npz"
        if not level_path.exists():
            break
        level_data = np.load(level_path, allow_pickle=True)
        data["levels"].append({k: level_data[k] for k in level_data.files})
        level_idx += 1

    return data


def visualize_case(case_dir: Path, save_path: Path = None, max_context: int = 2, max_levels: int = 3):
    """Create visualization from saved case data.

    Layout: One row per level, columns [Ctx1, Ctx2, Target, Combined, GT]
    All shown at native level resolution.
    """
    data = load_case(case_dir)

    img = data.get("img")
    gt_mask = data.get("gt_mask")
    context_imgs = data.get("context_imgs")  # [k, 1, H, W]
    context_masks = data.get("context_masks")  # [k, 1, H, W]
    levels = data.get("levels", [])[:max_levels]
    metadata = data.get("metadata", {})

    n_levels = len(levels)
    if n_levels == 0:
        print("No level data found")
        return

    n_ctx = min(context_imgs.shape[0] if context_imgs is not None else 0, max_context)

    # Layout: [Ctx1, Ctx2, Target, Combined, GT] per row (one row per level)
    n_cols = max_context + 3  # ctx1, ctx2, target, combined, gt
    n_rows = n_levels

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
    if n_rows == 1:
        axes = [axes]

    for li, level in enumerate(levels):
        row = axes[li]
        level_res = int(level.get("level_res", 32))
        patch_size = int(level.get("patch_size", 16))
        target_coords = level.get("target_coords")
        context_coords = level.get("context_coords")

        # Downsample images to level resolution
        img_ds = downsample_to_level(img, level_res)
        gt_ds = downsample_to_level(gt_mask, level_res)

        col_idx = 0

        # Context images (downsampled to level res)
        for ci in range(max_context):
            if ci < n_ctx and context_imgs is not None:
                ctx_img = downsample_to_level(context_imgs[ci, 0], level_res)
                ctx_mask = downsample_to_level(context_masks[ci, 0], level_res) if context_masks is not None else None
                row[col_idx].imshow(ctx_img, cmap="gray")
                if ctx_mask is not None:
                    row[col_idx].imshow(ctx_mask, cmap="Reds", alpha=0.4)
                    row[col_idx].contour(ctx_mask, colors="cyan", linewidths=1, levels=[0.5])
                # Draw context patch boxes for this context
                if context_coords is not None:
                    # Context coords are flattened [K_ctx, 2], need to split by context
                    K_per_ctx = context_coords.shape[0] // n_ctx if n_ctx > 0 else 0
                    if K_per_ctx > 0:
                        ci_coords = context_coords[ci * K_per_ctx:(ci + 1) * K_per_ctx]
                        draw_patch_boxes(row[col_idx], (level_res, level_res), ci_coords, patch_size, level_res, "cyan", 1)
                if li == 0:
                    row[col_idx].set_title(f"Ctx {ci + 1}")
            else:
                row[col_idx].axis("off")
            row[col_idx].axis("off")
            col_idx += 1

        # Target with patch boxes
        row[col_idx].imshow(img_ds, cmap="gray")
        if target_coords is not None:
            draw_patch_boxes(row[col_idx], (level_res, level_res), target_coords, patch_size, level_res, "lime", 1)
        if li == 0:
            row[col_idx].set_title("Target")
        row[col_idx].set_ylabel(f"L{li} ({level_res}x{level_res})", fontsize=10)
        row[col_idx].axis("off")
        col_idx += 1

        # Combined prediction (at native level resolution)
        combined_pred = level.get("combined_pred")
        row[col_idx].imshow(img_ds, cmap="gray")
        if combined_pred is not None:
            comb = combined_pred.squeeze()
            row[col_idx].imshow(comb, cmap="Reds", alpha=0.5)
            row[col_idx].contour(comb, colors="lime", linewidths=1, levels=[0.5])
        if gt_ds is not None:
            row[col_idx].contour(gt_ds, colors="yellow", linewidths=1, levels=[0.5])
        if li == 0:
            row[col_idx].set_title("Combined")
        row[col_idx].axis("off")
        col_idx += 1

        # GT at level resolution
        row[col_idx].imshow(img_ds, cmap="gray")
        if gt_ds is not None:
            row[col_idx].imshow(gt_ds, cmap="Reds", alpha=0.5)
            row[col_idx].contour(gt_ds, colors="yellow", linewidths=1, levels=[0.5])
        if li == 0:
            row[col_idx].set_title("GT")
        row[col_idx].axis("off")

    # Title
    case_id = metadata.get("case_id", "unknown")
    label_id = metadata.get("label_id", "unknown")
    axis = metadata.get("axis", "unknown")
    dice = metadata.get("dice", 0)
    fig.suptitle(f"{case_id} / {label_id} / {axis} (Dice={dice:.3f})", fontsize=12)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    else:
        plt.show()

    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize saved case data")
    parser.add_argument("case_dir", type=str, help="Path to case directory containing npy/npz files")
    parser.add_argument("-o", "--output", type=str, default="results/viz_by_level.png", help="Output path for PNG (default: show)")
    parser.add_argument("--max-context", type=int, default=1, help="Max context images to show")
    parser.add_argument("--max-levels", type=int, default=3, help="Max levels to show")
    args = parser.parse_args()

    visualize_case(args.case_dir, args.output, args.max_context, args.max_levels)
