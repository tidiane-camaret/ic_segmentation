"""
Iterative 3D Segmentation Evaluation.

Segments a volume slice-by-slice, using:
- Initial context from a reference case
- Previously segmented slices as additional context (propagation)

This evaluates the model's ability to propagate segmentation through a volume.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import hydra
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.train_utils import seed_everything


def load_case_slices(
    h5_path: Path,
    label_id: str,
    z_indices: Optional[List[int]] = None,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """Load all z-slices for a case/label from z-optimized HDF5.

    Returns:
        images: (N, H, W) float32
        masks: (N, H, W) float32
        z_indices: list of z indices loaded
    """
    with h5py.File(h5_path, 'r') as h5f:
        if z_indices is None:
            # Load all available z-indices from metadata
            z_indices = list(h5f['meta/z_indices'][:])

        D, H, W = h5f['meta'].attrs['shape']

        images = []
        masks = []
        valid_indices = []

        for z_idx in z_indices:
            if z_idx >= D:
                continue
            img = h5f['ct'][z_idx, :, :]

            # Check if label exists
            if label_id not in h5f['masks']:
                continue
            mask = h5f[f'masks/{label_id}'][z_idx, :, :]

            images.append(img.astype(np.float32))
            masks.append(mask.astype(np.float32))
            valid_indices.append(z_idx)

        if not images:
            return None, None, []

        return np.stack(images), np.stack(masks), valid_indices


def normalize_image(img: np.ndarray, modality: str = "ct") -> np.ndarray:
    """Normalize image based on modality."""
    if modality == "mri":
        nonzero_mask = img > 0
        if nonzero_mask.any():
            nonzero_vals = img[nonzero_mask]
            a_min = np.percentile(nonzero_vals, 0.5)
            a_max = np.percentile(nonzero_vals, 99.5)
        else:
            a_min, a_max = img.min(), img.max()
        if a_max - a_min < 1e-6:
            a_max = a_min + 1.0
    else:
        a_min, a_max = -500.0, 1000.0

    img = np.clip(img, a_min, a_max)
    img = (img - a_min) / (a_max - a_min)
    return img


def resize_slice(arr: np.ndarray, size: Tuple[int, int], mode: str = "bilinear") -> np.ndarray:
    """Resize 2D array."""
    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).float()
    resized = F.interpolate(tensor, size=size, mode=mode, align_corners=False if mode == "bilinear" else None)
    return resized.squeeze().numpy()


def compute_dice(pred: np.ndarray, gt: np.ndarray, threshold: float = 0.5) -> float:
    """Compute Dice score."""
    pred_bin = (pred > threshold).astype(np.float32)
    gt_bin = (gt > threshold).astype(np.float32)

    intersection = (pred_bin * gt_bin).sum()
    union = pred_bin.sum() + gt_bin.sum()

    if union < 1e-6:
        return 1.0 if intersection < 1e-6 else 0.0

    return (2 * intersection) / union


def get_slice_order(n_slices: int, direction: str = "forward") -> List[int]:
    """Get slice processing order.

    Args:
        direction: "forward" (0→N), "backward" (N→0), "center_out" (center→edges)
    """
    if direction == "forward":
        return list(range(n_slices))
    elif direction == "backward":
        return list(range(n_slices - 1, -1, -1))
    elif direction == "center_out":
        center = n_slices // 2
        order = [center]
        for offset in range(1, n_slices):
            if center + offset < n_slices:
                order.append(center + offset)
            if center - offset >= 0:
                order.append(center - offset)
        return order
    else:
        raise ValueError(f"Unknown direction: {direction}")


@torch.no_grad()
def segment_volume_iterative(
    model,
    target_images: np.ndarray,  # (N, H, W) normalized
    target_masks_gt: np.ndarray,  # (N, H, W) for evaluation
    context_images: np.ndarray,  # (C, H, W) normalized
    context_masks: np.ndarray,  # (C, H, W)
    device: torch.device,
    image_size: Tuple[int, int],
    num_initial_context: int = 3,
    num_propagated_context: int = 2,
    propagation_direction: str = "forward",
    use_gt_propagation: bool = False,  # For ablation: use GT instead of predictions
) -> Dict:
    """Segment a volume slice-by-slice with context propagation.

    Args:
        target_images: Target slices to segment (N, H, W)
        target_masks_gt: Ground truth masks for evaluation
        context_images: Initial context images from reference case
        context_masks: Initial context masks from reference case
        num_initial_context: Number of context slices to use from reference
        num_propagated_context: Number of previous predictions to include as context
        propagation_direction: Order to process slices
        use_gt_propagation: If True, use GT masks for propagation (ablation)

    Returns:
        Dict with predictions, per-slice dice, and overall metrics
    """
    n_target = len(target_images)
    n_context = len(context_images)

    # Resize GT to image_size so it matches predictions
    target_masks_gt = np.stack([
        resize_slice(m, image_size, mode="nearest") for m in target_masks_gt
    ])

    # Limit initial context
    if n_context > num_initial_context:
        # Sample evenly
        indices = np.linspace(0, n_context - 1, num_initial_context, dtype=int)
        context_images = context_images[indices]
        context_masks = context_masks[indices]

    # Resize and prepare initial context
    ctx_imgs_resized = []
    ctx_masks_resized = []
    for i in range(len(context_images)):
        ctx_imgs_resized.append(resize_slice(context_images[i], image_size))
        ctx_masks_resized.append(resize_slice(context_masks[i], image_size))

    # Get processing order
    slice_order = get_slice_order(n_target, propagation_direction)

    # Storage for predictions (in original slice order)
    predictions = [None] * n_target
    per_slice_dice = [None] * n_target

    # Context pool: (image, mask, original_idx)
    propagated_context: List[Tuple[np.ndarray, np.ndarray, int]] = []

    model.eval()

    for step, slice_idx in enumerate(slice_order):
        # Prepare target
        target_img = resize_slice(target_images[slice_idx], image_size)
        target_gt = target_masks_gt[slice_idx]  # Already resized to image_size

        # Build context: initial + propagated
        all_ctx_imgs = list(ctx_imgs_resized)
        all_ctx_masks = list(ctx_masks_resized)

        # Add propagated context (most recent N)
        if propagated_context:
            recent = propagated_context[-num_propagated_context:]
            for img, mask, _ in recent:
                all_ctx_imgs.append(img)
                all_ctx_masks.append(mask)

        # Ensure we have at least 1 context
        if not all_ctx_imgs:
            # Fallback: use target GT (shouldn't happen normally)
            all_ctx_imgs.append(target_gt)
            all_ctx_masks.append(target_gt)

        # Build batch tensors
        target_tensor = torch.from_numpy(target_img).unsqueeze(0).unsqueeze(0).float().to(device)
        target_gt_tensor = torch.from_numpy(target_gt).unsqueeze(0).unsqueeze(0).float().to(device)

        context_in = torch.stack([
            torch.from_numpy(img).unsqueeze(0).float() for img in all_ctx_imgs
        ]).unsqueeze(0).to(device)  # (1, C, 1, H, W)

        context_out = torch.stack([
            torch.from_numpy(mask).unsqueeze(0).float() for mask in all_ctx_masks
        ]).unsqueeze(0).to(device)  # (1, C, 1, H, W)

        # Run model
        outputs = model(
            target_tensor,
            labels=target_gt_tensor,
            context_in=context_in,
            context_out=context_out,
            mode="val",
        )

        # Get prediction
        final_logit = outputs.get("final_logit")
        if final_logit is None:
            final_logit = outputs.get("logit", outputs.get("pred"))

        pred_prob = torch.sigmoid(final_logit).squeeze().cpu().numpy()

        # Store prediction
        predictions[slice_idx] = pred_prob

        # Compute per-slice dice
        dice = compute_dice(pred_prob, target_gt)
        per_slice_dice[slice_idx] = dice

        # Add to propagated context
        if use_gt_propagation:
            prop_mask = target_gt
        else:
            prop_mask = pred_prob

        propagated_context.append((target_img, prop_mask, slice_idx))

    # Stack predictions
    predictions_stack = np.stack(predictions)

    # Compute overall 3D dice
    overall_dice = compute_dice(predictions_stack, target_masks_gt)

    # Compute dice at different propagation depths
    depth_dices = {}
    for depth in [5, 10, 20, 50]:
        if depth <= n_target:
            order_subset = slice_order[:depth]
            preds_subset = [predictions[i] for i in order_subset]
            gts_subset = [target_masks_gt[i] for i in order_subset]
            if preds_subset:
                depth_dices[f"dice_first_{depth}"] = compute_dice(
                    np.stack(preds_subset), np.stack(gts_subset)
                )

    return {
        "predictions": predictions_stack,
        "per_slice_dice": per_slice_dice,
        "overall_dice": overall_dice,
        "mean_slice_dice": np.mean([d for d in per_slice_dice if d is not None]),
        "slice_order": slice_order,
        **depth_dices,
    }


def save_iterative_slices(
    target_imgs: np.ndarray,       # (N, H, W) normalized
    target_masks_gt: np.ndarray,   # (N, H, W) at image_size
    predictions: np.ndarray,       # (N, H, W) probabilities
    ctx_imgs: np.ndarray,          # (C, H, W) initial context images
    ctx_masks: np.ndarray,         # (C, H, W) initial context masks
    per_slice_dice: List[float],
    slice_order: List[int],
    case_id: str,
    label_id: str,
    save_dir: Path,
    use_wandb: bool = False,
    max_context: int = 2,
    n_sample_slices: int = 4,
) -> None:
    """Save representative predicted slices as images and log to wandb.

    Layout per slice row: [Ctx1, Ctx2, Target+GT, Pred probs, Binary+GT contour]
    Samples first, middle, last, and worst-dice slices from slice_order.
    """
    import matplotlib.pyplot as plt

    try:
        import wandb
        wandb_available = use_wandb and wandb.run is not None
    except ImportError:
        wandb_available = False

    save_dir.mkdir(parents=True, exist_ok=True)

    n_slices = len(slice_order)
    # Pick representative slice positions in processing order
    sample_positions = sorted(set([
        0,
        n_slices // 3,
        2 * n_slices // 3,
        n_slices - 1,
        int(np.argmin([per_slice_dice[i] for i in slice_order])),  # worst dice
    ]))[:n_sample_slices]
    sample_indices = [slice_order[p] for p in sample_positions]

    n_ctx_show = min(max_context, len(ctx_imgs))
    n_cols = n_ctx_show + 3  # ctx images + target+GT + pred probs + binary+GT
    n_rows = len(sample_indices)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 3.5 * n_rows), squeeze=False)

    for row_idx, slice_idx in enumerate(sample_indices):
        ax = axes[row_idx]
        col = 0
        step = slice_order.index(slice_idx)
        dice = per_slice_dice[slice_idx]

        # Context images
        for ci in range(n_ctx_show):
            ax[col].imshow(ctx_imgs[ci], cmap="gray")
            ax[col].imshow(ctx_masks[ci], cmap="Reds", alpha=0.4)
            ax[col].contour(ctx_masks[ci], colors="cyan", linewidths=0.8)
            ax[col].set_title(f"Ctx {ci + 1}", fontsize=8)
            ax[col].axis("off")
            col += 1

        # Target + GT overlay
        ax[col].imshow(target_imgs[slice_idx], cmap="gray")
        ax[col].imshow(target_masks_gt[slice_idx], cmap="Reds", alpha=0.4)
        ax[col].contour(target_masks_gt[slice_idx], colors="yellow", linewidths=0.8)
        ax[col].set_title(f"Target+GT (z={slice_idx})", fontsize=8)
        ax[col].axis("off")
        col += 1

        # Prediction probability map
        im = ax[col].imshow(predictions[slice_idx], cmap="hot", vmin=0, vmax=1)
        ax[col].set_title(f"Pred probs (step={step})", fontsize=8)
        fig.colorbar(im, ax=ax[col], fraction=0.046, pad=0.04)
        ax[col].axis("off")
        col += 1

        # Binary prediction + GT contour
        pred_bin = (predictions[slice_idx] > 0.5).astype(float)
        ax[col].imshow(target_imgs[slice_idx], cmap="gray")
        ax[col].imshow(pred_bin, cmap="Greens", alpha=0.4)
        ax[col].contour(target_masks_gt[slice_idx], colors="yellow", linewidths=0.8, linestyles="--")
        ax[col].contour(pred_bin, colors="lime", linewidths=0.8)
        ax[col].set_title(f"Binary (dice={dice:.3f})", fontsize=8)
        ax[col].axis("off")

    safe_label = label_id.replace("/", "_")
    fig.suptitle(f"{case_id} / {label_id}", fontsize=10, y=1.01)
    fig.tight_layout()

    save_path = save_dir / f"{case_id}_{safe_label}.png"
    fig.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)

    if wandb_available:
        overall_dice = np.mean([d for d in per_slice_dice if d is not None])
        wandb.log({
            f"iterative/{safe_label}": wandb.Image(
                str(save_path),
                caption=f"{case_id}/{label_id} (dice={overall_dice:.3f})"
            )
        })


def find_context_case(
    h5_files: List[Path],
    target_case: Path,
    label_id: str,
    stats: Dict,
) -> Optional[Path]:
    """Find a suitable context case (different from target, has the label)."""
    target_stem = target_case.stem

    for h5_file in h5_files:
        if h5_file.stem == target_stem:
            continue

        case_stats = stats.get(h5_file.stem, {})
        if label_id in case_stats.get("labels", {}):
            return h5_file

    return None


@hydra.main(version_base=None, config_path="../configs", config_name="eval")
def main(cfg: DictConfig) -> None:
    """Main iterative evaluation function."""
    import pickle

    # Initialize
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(cfg.training.seed)

    print(f"Using device: {device}")

    # Wandb logging
    if cfg.logging.use_wandb:
        import wandb
        wandb.init(
            project=cfg.logging.wandb_project_eval,
            config=OmegaConf.to_container(cfg, resolve=True),
            tags=["iterative"],
        )
        run_name = wandb.run.name
    else:
        run_name = "iterative_eval"

    # Get image size
    img_size = cfg.preprocessing.image_size
    if isinstance(img_size, (list, tuple)):
        image_size = tuple(img_size[:2])
    else:
        image_size = (img_size, img_size)

    # Load data paths (zopt format required)
    dataloader_type = cfg.get("dataloader_type", "zopt")
    if dataloader_type != "zopt":
        print("Warning: Iterative eval works best with zopt dataloader. Switching to zopt.")
        dataloader_type = "zopt"

    base_dataset = cfg.get("base_dataset", "totalseg")
    data_dir = Path(cfg.paths.DATA_DIR)
    data_subdir = data_dir / f"{base_dataset}_3d_zopt"
    stats_path = data_subdir / "stats.pkl"

    print(f"Data directory: {data_subdir}")

    if not data_subdir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_subdir}")

    # Load stats
    with open(stats_path, "rb") as f:
        stats = pickle.load(f)
    print(f"Loaded stats for {len(stats)} cases")

    # Get HDF5 files
    h5_files = sorted(list(data_subdir.glob("*.h5")))

    # Filter by split
    val_split = cfg.get("val_split", ["val", "test"])
    if val_split:
        import pandas as pd
        meta_path = data_subdir.parent / "totalseg" / "meta.csv"
        if meta_path.exists():
            df = pd.read_csv(meta_path, sep=";")
            if isinstance(val_split, str):
                split_case_ids = set(df["image_id"][df["split"] == val_split].tolist())
            else:
                split_case_ids = set(df["image_id"][df["split"].isin(val_split)].tolist())
            h5_files = [f for f in h5_files if f.stem in split_case_ids]
            print(f"Filtered to {len(h5_files)} cases for split {val_split}")

    # Limit cases
    max_cases = cfg.get("max_cases", None)
    if OmegaConf.is_dict(max_cases) or isinstance(max_cases, dict):
        max_cases = max_cases.get("val", None)
    if max_cases is not None and len(h5_files) > max_cases:
        h5_files = h5_files[:max_cases]
        print(f"Limited to {max_cases} cases")

    # Get labels to evaluate
    from data.label_ids_totalseg import get_label_ids
    val_labels = cfg.val_label_ids if isinstance(cfg.val_label_ids, str) else list(cfg.val_label_ids)
    if isinstance(val_labels, str):
        modality = "mri" if "mri" in base_dataset else "ct"
        val_labels = get_label_ids(val_labels, modality=modality)

    max_labels = cfg.get("max_labels", None)
    if max_labels:
        val_labels = val_labels[:max_labels]
    print(f"Evaluating {len(val_labels)} labels")

    # Iterative eval config
    iter_cfg = cfg.get("iterative", {})
    num_initial_context = iter_cfg.get("num_initial_context", 3)
    num_propagated_context = iter_cfg.get("num_propagated_context", 2)
    propagation_direction = iter_cfg.get("propagation_direction", "forward")
    use_gt_propagation = iter_cfg.get("use_gt_propagation", False)

    print(f"Iterative config: initial_ctx={num_initial_context}, prop_ctx={num_propagated_context}, "
          f"direction={propagation_direction}, use_gt={use_gt_propagation}")

    # Load model
    from src.model_builder import build_model
    model = build_model(
        cfg, device,
        context_size=num_initial_context + num_propagated_context,
    )

    model = model.to(device)
    model.eval()

    # Setup image save directory
    results_dir = Path(cfg.paths.RESULTS_DIR)
    img_save_dir = results_dir / "iterative_eval" / (run_name if cfg.logging.use_wandb else "local")
    img_save_dir.mkdir(parents=True, exist_ok=True)
    save_images = cfg.logging.get("save_imgs_masks", True)
    print(f"Saving images to: {img_save_dir}")

    # Evaluation loop
    results = []
    modality = "mri" if "mri" in base_dataset else "ct"

    pbar = tqdm(h5_files, desc="Evaluating cases")
    for target_h5 in pbar:
        target_case_id = target_h5.stem
        case_stats = stats.get(target_case_id, {})
        available_labels = set(case_stats.get("labels", {}).keys())

        for label_id in val_labels:
            if label_id not in available_labels:
                continue

            # Load target slices
            target_imgs, target_masks, z_indices = load_case_slices(target_h5, label_id)
            if target_imgs is None or len(target_imgs) < 3:
                continue

            # Skip if mask is mostly empty
            if target_masks.sum() < 100:
                continue

            # Find context case
            context_h5 = find_context_case(h5_files, target_h5, label_id, stats)
            if context_h5 is None:
                continue

            # Load context slices
            ctx_imgs, ctx_masks, _ = load_case_slices(context_h5, label_id)
            if ctx_imgs is None or len(ctx_imgs) < 1:
                continue

            # Normalize images
            target_imgs_norm = np.stack([normalize_image(img, modality) for img in target_imgs])
            ctx_imgs_norm = np.stack([normalize_image(img, modality) for img in ctx_imgs])

            # Run iterative segmentation
            try:
                result = segment_volume_iterative(
                    model=model,
                    target_images=target_imgs_norm,
                    target_masks_gt=target_masks,
                    context_images=ctx_imgs_norm,
                    context_masks=ctx_masks,
                    device=device,
                    image_size=image_size,
                    num_initial_context=num_initial_context,
                    num_propagated_context=num_propagated_context,
                    propagation_direction=propagation_direction,
                    use_gt_propagation=use_gt_propagation,
                )
            except Exception as e:
                print(f"Error processing {target_case_id}/{label_id}: {e}")
                continue

            results.append({
                "case_id": target_case_id,
                "context_case_id": context_h5.stem,
                "label_id": label_id,
                "n_slices": len(target_imgs),
                "overall_dice": result["overall_dice"],
                "mean_slice_dice": result["mean_slice_dice"],
                **{k: v for k, v in result.items() if k.startswith("dice_first_")},
            })

            # Save representative slices as images
            if save_images:
                try:
                    gt_resized = np.stack([
                        resize_slice(m, image_size, mode="nearest") for m in target_masks
                    ])
                    ctx_masks_resized = np.stack([
                        resize_slice(m, image_size, mode="nearest")
                        for m in ctx_masks[:num_initial_context]
                    ])
                    ctx_imgs_resized_vis = np.stack([
                        resize_slice(img, image_size) for img in ctx_imgs_norm[:num_initial_context]
                    ])
                    save_iterative_slices(
                        target_imgs=target_imgs_norm,
                        target_masks_gt=gt_resized,
                        predictions=result["predictions"],
                        ctx_imgs=ctx_imgs_resized_vis,
                        ctx_masks=ctx_masks_resized,
                        per_slice_dice=result["per_slice_dice"],
                        slice_order=result["slice_order"],
                        case_id=target_case_id,
                        label_id=label_id,
                        save_dir=img_save_dir,
                        use_wandb=cfg.logging.use_wandb,
                    )
                except Exception as e:
                    print(f"Warning: Failed to save images for {target_case_id}/{label_id}: {e}")

            pbar.set_postfix({
                "dice": f"{result['overall_dice']:.3f}",
                "label": label_id[:10],
            })

    # Summary
    if results:
        overall_dices = [r["overall_dice"] for r in results]
        mean_slice_dices = [r["mean_slice_dice"] for r in results]

        print(f"\n=== Iterative Evaluation Results ===")
        print(f"Cases evaluated: {len(set(r['case_id'] for r in results))}")
        print(f"Case-label pairs: {len(results)}")
        print(f"Mean 3D Dice: {np.mean(overall_dices):.4f} ± {np.std(overall_dices):.4f}")
        print(f"Mean per-slice Dice: {np.mean(mean_slice_dices):.4f} ± {np.std(mean_slice_dices):.4f}")

        # Per-label results
        print("\nPer-label results:")
        label_dices = {}
        for r in results:
            label_id = r["label_id"]
            if label_id not in label_dices:
                label_dices[label_id] = []
            label_dices[label_id].append(r["overall_dice"])

        for label_id, dices in sorted(label_dices.items(), key=lambda x: -np.mean(x[1])):
            print(f"  {label_id}: {np.mean(dices):.4f} (n={len(dices)})")

        # Depth analysis
        print("\nDice by propagation depth:")
        for depth in [5, 10, 20, 50]:
            key = f"dice_first_{depth}"
            vals = [r[key] for r in results if key in r]
            if vals:
                print(f"  First {depth} slices: {np.mean(vals):.4f}")

        # Log to wandb
        if cfg.logging.use_wandb:
            wandb.log({
                "mean_3d_dice": np.mean(overall_dices),
                "std_3d_dice": np.std(overall_dices),
                "mean_slice_dice": np.mean(mean_slice_dices),
                "n_cases": len(set(r["case_id"] for r in results)),
                "n_pairs": len(results),
            })

            # Log per-label
            for label_id, dices in label_dices.items():
                wandb.log({f"dice_label/{label_id}": np.mean(dices)})

            wandb.finish()
    else:
        print("No results collected. Check data paths and labels.")


if __name__ == "__main__":
    main()
