"""Training and validation utilities for PatchICL."""

import os
import random
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Global thread pool for async image saving (avoid creating threads per call)
_IMAGE_SAVE_EXECUTOR: Optional[ThreadPoolExecutor] = None


def _get_image_save_executor() -> ThreadPoolExecutor:
    """Get or create the global thread pool for async image saving."""
    global _IMAGE_SAVE_EXECUTOR
    if _IMAGE_SAVE_EXECUTOR is None:
        _IMAGE_SAVE_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="img_save")
    return _IMAGE_SAVE_EXECUTOR

from src.models.patch_icl_v2.metrics import (
    PRED_THRESHOLD,
    GT_AREA_THRESHOLD,
    compute_all_metrics,
    compute_per_sample_dice,
    compute_pixel_mae,
)


def seed_everything(seed: int) -> None:
    """Set random seeds for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _save_sample_images(
    label_samples: dict,
    save_dir: Path,
    epoch: int,
    prefix: str = "train",
    max_samples: int = 20,
) -> None:
    """Save sample images with patches to disk."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    epoch_dir = save_dir / f"{prefix}_epoch_{epoch:04d}"
    epoch_dir.mkdir(parents=True, exist_ok=True)

    def draw_patch_boxes(ax, img, coords, patch_size, level_res, color="red"):
        """Draw patch bounding boxes on image."""
        if coords is None or level_res is None:
            return
        # Handle edge cases with coords shape
        if coords.ndim == 0 or coords.numel() == 0:
            return
        if coords.ndim == 1:
            coords = coords.unsqueeze(0)  # [2] -> [1, 2]
        if coords.shape[-1] != 2:
            return

        H, W = img.shape[:2]
        scale_h, scale_w = H / level_res, W / level_res
        scaled_patch_h = int(patch_size * scale_h)
        scaled_patch_w = int(patch_size * scale_w)

        for k in range(coords.shape[0]):
            coord = coords[k]
            if coord.ndim == 0:
                continue
            r, c = coord[0].item(), coord[1].item()
            r_start = int(r * scale_h)
            c_start = int(c * scale_w)
            rect = Rectangle(
                (c_start, r_start),
                scaled_patch_w,
                scaled_patch_h,
                linewidth=1,
                edgecolor=color,
                facecolor="none",
            )
            ax.add_patch(rect)

    wandb_images = []
    try:
        import wandb

        wandb_available = True
    except ImportError:
        wandb_available = False

    for label_id, sample in list(label_samples.items())[:max_samples]:
        img = sample["image"].squeeze().numpy()
        gt = sample["label"].squeeze().numpy()
        dice = sample["dice"]
        ctx_in = sample.get("context_in")
        ctx_out = sample.get("context_out")

        # Build per-level info (backward compat with single-level samples)
        levels_info = sample.get("levels", [])
        if not levels_info:
            levels_info = [
                {
                    "target_coords": sample.get("target_coords"),
                    "context_coords": sample.get("context_coords"),
                    "patch_size": sample.get("patch_size", 16),
                    "level_res": sample.get("level_res", 32),
                    "pred_probs": sample.get("pred_probs"),
                }
            ]

        n_levels = len(levels_info)
        n_ctx = ctx_in.shape[0] if ctx_in is not None else 0
        n_cols = max(1 + n_ctx, 4)
        n_rows = 2 * n_levels

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
        if n_rows == 2:
            axes = [axes[0], axes[1]]  # keep 2D indexing consistent

        for li, level_info in enumerate(levels_info):
            top_row = 2 * li
            bot_row = 2 * li + 1
            l_target_coords = level_info.get("target_coords")
            l_context_coords = level_info.get("context_coords")
            l_patch_size = level_info.get("patch_size", 16)
            l_level_res = level_info.get("level_res", 32)
            l_pred_probs = level_info.get("pred_probs")
            l_refined_probs = level_info.get("refined_probs")

            # --- Top row: target + context images with patch boxes ---
            axes[top_row][0].imshow(img, cmap="gray")
            axes[top_row][0].imshow(gt, cmap="Reds", alpha=0.4)
            axes[top_row][0].contour(gt, colors="yellow", linewidths=1)
            if l_target_coords is not None and l_level_res is not None:
                draw_patch_boxes(
                    axes[top_row][0],
                    img,
                    l_target_coords,
                    l_patch_size,
                    l_level_res,
                    color="lime",
                )
            axes[top_row][0].set_title(f"Level {li} (res {l_level_res}) - Target")
            axes[top_row][0].axis("off")

            for ci in range(n_ctx):
                ctx_img = ctx_in[ci].squeeze().numpy()
                ctx_mask = ctx_out[ci].squeeze().numpy()
                axes[top_row][1 + ci].imshow(ctx_img, cmap="gray")
                axes[top_row][1 + ci].imshow(ctx_mask, cmap="Reds", alpha=0.4)
                axes[top_row][1 + ci].contour(ctx_mask, colors="cyan", linewidths=1)
                if (
                    l_context_coords is not None
                    and l_level_res is not None
                    and l_context_coords.ndim == 2
                ):
                    try:
                        total_patches = l_context_coords.shape[0]
                        K_per_ctx = total_patches // n_ctx
                        ci_coords = l_context_coords[
                            ci * K_per_ctx : (ci + 1) * K_per_ctx
                        ]
                        draw_patch_boxes(
                            axes[top_row][1 + ci],
                            ctx_img,
                            ci_coords,
                            l_patch_size,
                            l_level_res,
                            color="cyan",
                        )
                    except (IndexError, TypeError, ZeroDivisionError):
                        pass
                axes[top_row][1 + ci].set_title(f"Context {ci + 1}")
                axes[top_row][1 + ci].axis("off")

            for j in range(1 + n_ctx, n_cols):
                axes[top_row][j].axis("off")

            # --- Bottom row: Downsized GT | Pred probs | Pred mask ---
            # Downsized GT
            if l_level_res is not None and l_level_res < gt.shape[0]:
                gt_t = torch.from_numpy(gt).unsqueeze(0).unsqueeze(0).float()
                scale = max(1, gt.shape[0] // l_level_res)
                gt_ds = (
                    F.avg_pool2d(gt_t, kernel_size=scale, stride=scale)
                    .squeeze()
                    .numpy()
                )
            else:
                gt_ds = gt
            axes[bot_row][0].imshow(gt_ds, cmap="gray")
            axes[bot_row][0].set_title(f"GT ({gt_ds.shape[0]}x{gt_ds.shape[1]})")
            axes[bot_row][0].axis("off")

            # Pred probs
            if l_pred_probs is not None:
                pp = l_pred_probs.squeeze().numpy()
                pred_probs = (pp - pp.min()) / (pp.max() - pp.min() + 1e-8)
                im = axes[bot_row][1].imshow(pred_probs, cmap="hot", vmin=0, vmax=1)
                axes[bot_row][1].set_title(f"Pred. Probs ({pp.shape[0]}x{pp.shape[1]})")
                plt.colorbar(im, ax=axes[bot_row][1], fraction=0.046, pad=0.04)
            else:
                axes[bot_row][1].set_title("Pred. Probs (N/A)")
            axes[bot_row][1].axis("off")

            # Pred mask with per-level dice
            if l_pred_probs is not None:
                pp = l_pred_probs.squeeze()
                # Resize pred to match gt_ds if needed (backward compat)
                if pp.shape[0] != gt_ds.shape[0]:
                    pp = F.interpolate(
                        pp.unsqueeze(0).unsqueeze(0).float(),
                        size=(gt_ds.shape[0], gt_ds.shape[1]),
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze()
                pp = pp.numpy() if hasattr(pp, "numpy") else pp
                pred_mask = (pp > PRED_THRESHOLD).astype(float)
                gt_binary = (gt_ds > GT_AREA_THRESHOLD).astype(float)
                inter = (pred_mask * gt_binary).sum()
                union_val = pred_mask.sum() + gt_binary.sum()
                level_dice = (2 * inter + 1e-6) / (union_val + 1e-6)
                axes[bot_row][2].imshow(pred_mask, cmap="gray")
                axes[bot_row][2].set_title(f"Pred (dice={level_dice:.3f})")
            else:
                axes[bot_row][2].set_title("Pred (N/A)")
            axes[bot_row][2].axis("off")

            # Refined probs (sampling weights for this level from progressive refinement)
            if l_refined_probs is not None and n_cols > 3:
                rp = l_refined_probs.squeeze().numpy()
                im = axes[bot_row][3].imshow(rp, cmap="hot", vmin=0, vmax=1)
                axes[bot_row][3].set_title(
                    f"Refined Pred. Probs ({rp.shape[0]}x{rp.shape[1]})"
                )
                plt.colorbar(im, ax=axes[bot_row][3], fraction=0.046, pad=0.04)
            elif n_cols > 3:
                axes[bot_row][3].set_title("Refined Pred. Probs (N/A)")
            if n_cols > 3:
                axes[bot_row][3].axis("off")

            for j in range(4, n_cols):
                axes[bot_row][j].axis("off")

        fig.suptitle(f"{label_id} (final dice={dice:.3f})", fontsize=12, y=1.01)
        fig.tight_layout()
        safe_label = label_id.replace("/", "_")
        save_path = epoch_dir / f"{safe_label}.png"
        fig.savefig(save_path, dpi=100, bbox_inches="tight")
        plt.close(fig)

        # Log the saved image to wandb
        if wandb_available:
            wandb_img = wandb.Image(
                str(save_path), caption=f"{label_id} (dice={dice:.3f})"
            )
            wandb_images.append(wandb_img)

    # Log all images at once to wandb
    if wandb_images and wandb_available and wandb.run is not None:
        wandb.log({f"{prefix}/saved_samples": wandb_images, "epoch": epoch})


def _save_sample_images_async(
    label_samples: dict,
    save_dir: Path,
    epoch: int,
    prefix: str = "train",
    max_samples: int = 20,
) -> None:
    """Save sample images asynchronously in background thread.

    Copies data and submits to thread pool to avoid blocking training.
    """
    # Deep copy the samples dict to avoid race conditions
    # (tensors are already .cpu() in _collect_sample_for_viz)
    import copy
    samples_copy = copy.deepcopy(label_samples)

    executor = _get_image_save_executor()
    executor.submit(_save_sample_images, samples_copy, save_dir, epoch, prefix, max_samples)


def _collect_sample_for_viz(
    i: int,
    images: torch.Tensor,
    labels: torch.Tensor,
    outputs: dict,
    context_in: torch.Tensor | None,
    context_out: torch.Tensor | None,
    dice_val: float,
) -> dict:
    """Collect sample data for visualization."""
    level_outputs = outputs.get("level_outputs", [])

    # Get pred_probs and pred_binary from last level or final_logit
    if level_outputs:
        pred_probs = torch.sigmoid(level_outputs[-1]["pred"][i]).detach().cpu()
    else:
        pred_probs = torch.sigmoid(outputs["final_logit"][i]).detach().cpu()
    pred_binary = (pred_probs > PRED_THRESHOLD).float()

    # Collect per-level info
    levels = []
    for level_out in level_outputs:
        l_coords = level_out.get("coords")
        if l_coords is not None:
            l_coords = l_coords[i].detach().cpu()
        l_ctx_coords = level_out.get("context_coords")
        if l_ctx_coords is not None:
            l_ctx_coords = l_ctx_coords[i].detach().cpu()
        l_pred = level_out.get("pred")
        l_pred_probs = None
        if l_pred is not None:
            l_pred_probs = torch.sigmoid(l_pred[i]).detach().cpu()
        l_refined = level_out.get("refined_probs")
        if l_refined is not None:
            l_refined = l_refined[i].detach().cpu()
        levels.append(
            {
                "target_coords": l_coords,
                "context_coords": l_ctx_coords,
                "patch_size": level_out.get("patch_size", 16),
                "level_res": level_out.get("level_res", 32),
                "pred_probs": l_pred_probs,
                "refined_probs": l_refined,
            }
        )

    return {
        "image": images[i].detach().cpu(),
        "label": labels[i].detach().cpu(),
        "pred": pred_binary,
        "pred_probs": pred_probs,
        "dice": dice_val,
        "context_in": context_in[i].detach().cpu() if context_in is not None else None,
        "context_out": (
            context_out[i].detach().cpu() if context_out is not None else None
        ),
        "levels": levels,
    }


def train_epoch(
    model,
    train_loader,
    optimizer,
    device,
    epoch,
    print_every,
    grad_accumulate_steps=1,
    accelerator=None,
    use_wandb=False,
    log_every=10,
    save_dir: Optional[Path] = None,
    save_every_n_epochs: int = 5,
    compute_metrics_every: int = 10,
):
    """Run one training epoch."""
    model.train()
    is_main = accelerator is None or accelerator.is_main_process
    unwrapped_model = (
        accelerator.unwrap_model(model) if accelerator is not None else model
    )

    if use_wandb and is_main:
        try:
            import wandb
        except ImportError:
            use_wandb = False

    # Metrics - dynamically accumulate all loss keys from compute_loss
    from collections import defaultdict

    loss_accum = defaultdict(float)  # sums for all compute_loss keys
    loss_count = defaultdict(int)  # counts (some keys appear conditionally)
    dice_accum = defaultdict(float)  # per-level dice accumulators
    dice_count = defaultdict(int)
    total_local_dice = 0.0
    total_final_dice = 0.0
    total_context_dice = 0.0
    total_local_softdice = 0.0
    total_final_softdice = 0.0
    total_context_softdice = 0.0
    total_final_pixel_mae = 0.0
    context_dice_count = 0
    label_dice_scores = {}  # Per-label tracking
    label_samples = {}  # Store one sample per label for wandb image logging

    pbar = tqdm(
        train_loader,
        desc=f"Epoch {epoch}",
        disable=not is_main,
        unit="batch",
        dynamic_ncols=True,
    )

    for idx, batch in enumerate(pbar):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        context_in = batch.get("context_in")
        context_out = batch.get("context_out")
        if context_in is not None:
            context_in = context_in.to(device)
        if context_out is not None:
            context_out = context_out.to(device)

        target_features = batch.get("target_features")
        context_features = batch.get("context_features")
        if target_features is not None:
            target_features = target_features.to(device)
        if context_features is not None:
            context_features = context_features.to(device)

        if labels.dim() == 3:
            labels = labels.unsqueeze(1)

        if idx % grad_accumulate_steps == 0:
            optimizer.zero_grad()

        outputs = model(
            images,
            labels=labels,
            context_in=context_in,
            context_out=context_out,
            target_features=target_features,
            context_features=context_features,
            mode="train",
        )
        losses = unwrapped_model.compute_loss(outputs, labels)
        loss = losses["total_loss"]

        # Compute Dice scores periodically (not every batch to save compute)
        should_compute_metrics = (idx % compute_metrics_every == 0) or (
            idx == len(train_loader) - 1
        )
        if should_compute_metrics:
            with torch.no_grad():
                # Use return_per_sample=True to avoid redundant interpolation
                metrics = compute_all_metrics(outputs, labels, return_per_sample=True)

                # Accumulate metrics
                total_local_dice += metrics.get("local_dice", torch.tensor(0.0)).item()
                total_local_softdice += metrics.get(
                    "local_soft_dice", torch.tensor(0.0)
                ).item()
                total_final_dice += metrics.get("final_dice", torch.tensor(0.0)).item()
                total_final_softdice += metrics.get(
                    "final_soft_dice", torch.tensor(0.0)
                ).item()
                total_final_pixel_mae += metrics.get(
                    "final_pixel_mae", torch.tensor(0.0)
                ).item()

                if "context_dice" in metrics:
                    total_context_dice += metrics["context_dice"].item()
                    total_context_softdice += metrics["context_soft_dice"].item()
                    context_dice_count += 1

                # Per-level dice accumulation
                level_outputs = outputs.get("level_outputs", [])
                for li in range(len(level_outputs)):
                    for suffix in ["dice", "soft_dice"]:
                        key = f"level_{li}_{suffix}"
                        if key in metrics:
                            dice_accum[key] += metrics[key].item()
                            dice_count[key] += 1

                # Per-sample dice for per-label tracking (reuse from compute_all_metrics)
                per_sample_dice = metrics.get("per_sample_dice")
                if per_sample_dice is None:
                    per_sample_dice = compute_per_sample_dice(outputs, labels)
                batch_label_ids = batch.get("label_ids") or batch.get(
                    "label_id", [None] * images.shape[0]
                )
                for i in range(images.shape[0]):
                    label_id = batch_label_ids[i] if batch_label_ids else "unknown"
                    dice_val = per_sample_dice[i].item()
                    if label_id not in label_dice_scores:
                        label_dice_scores[label_id] = []
                    label_dice_scores[label_id].append(dice_val)

                    # Store one sample per label for image logging (wandb or disk)
                    should_save = (use_wandb or save_dir is not None) and is_main
                    if should_save and label_id not in label_samples:
                        label_samples[label_id] = _collect_sample_for_viz(
                            i,
                            images,
                            labels,
                            outputs,
                            context_in,
                            context_out,
                            dice_val,
                        )

        # Backward
        scaled_loss = loss / grad_accumulate_steps
        if accelerator is not None:
            accelerator.backward(scaled_loss)
        else:
            scaled_loss.backward()

        if (idx + 1) % grad_accumulate_steps == 0:
            # Clip gradients to prevent NaN from extreme BCE gradients
            if accelerator is not None:
                accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # Track metrics
        # Accumulate all loss keys dynamically
        for key, val in losses.items():
            v = val.item() if hasattr(val, "item") else float(val)
            loss_accum[key] += v
            loss_count[key] += 1

        # Update progress bar
        n_batches = idx + 1
        avg_loss = (
            loss_accum["total_loss"] / loss_count["total_loss"]
            if loss_count["total_loss"]
            else 0
        )
        # Use metric count for averaging since we don't compute every batch
        metric_batches = (idx // compute_metrics_every) + 1
        pbar.set_postfix(
            {
                "loss": f"{avg_loss:.4f}",
                "mae": (
                    f"{total_final_pixel_mae / metric_batches:.4f}"
                    if metric_batches > 0
                    else "N/A"
                ),
                "sdice": (
                    f"{total_final_softdice / metric_batches:.4f}"
                    if metric_batches > 0
                    else "N/A"
                ),
            }
        )

        # Log to wandb
        if use_wandb and is_main and idx % log_every == 0:
            global_step = epoch * len(train_loader) + idx
            metric_batches = (idx // compute_metrics_every) + 1
            ctx_dice_avg = (
                total_context_dice / context_dice_count
                if context_dice_count > 0
                else 0.0
            )
            ctx_softdice_avg = (
                total_context_softdice / context_dice_count
                if context_dice_count > 0
                else 0.0
            )
            log_dict = {
                "train_batch/local_dice": (
                    total_local_dice / metric_batches if metric_batches > 0 else 0
                ),
                "train_batch/final_dice": (
                    total_final_dice / metric_batches if metric_batches > 0 else 0
                ),
                "train_batch/context_dice": ctx_dice_avg,
                "train_batch/local_soft_dice": (
                    total_local_softdice / metric_batches if metric_batches > 0 else 0
                ),
                "train_batch/final_soft_dice": (
                    total_final_softdice / metric_batches if metric_batches > 0 else 0
                ),
                "train_batch/final_pixel_mae": (
                    total_final_pixel_mae / metric_batches if metric_batches > 0 else 0
                ),
                "train_batch/context_soft_dice": ctx_softdice_avg,
                "global_step": global_step,
            }
            # Log all compute_loss keys (includes refined_probs dices)
            for key in loss_accum:
                log_dict[f"train_batch/{key}"] = loss_accum[key] / loss_count[key]
            # Log per-level dice metrics
            for key in dice_accum:
                if dice_count[key] > 0:
                    log_dict[f"train_batch/{key}"] = dice_accum[key] / dice_count[key]
            wandb.log(log_dict, step=global_step)

        # Print progress
        if print_every and idx % print_every == 0 and is_main:
            avg_loss = (
                loss_accum["total_loss"] / loss_count["total_loss"]
                if loss_count["total_loss"]
                else 0
            )
            metric_batches = (idx // compute_metrics_every) + 1
            print(
                f"Epoch {epoch:04d} | Batch {idx:04d} | "
                f"Loss: {avg_loss:.5f} | "
                f"PatchDice: {total_local_dice / metric_batches:.5f} | "
                f"FinalDice: {total_final_dice / metric_batches:.5f} | "
                f"SoftDice: {total_final_softdice / metric_batches:.5f}"
            )

        del outputs, losses

    # Use actual number of metric computations for final averaging
    n_metric_computations = len(
        [
            i
            for i in range(len(train_loader))
            if i % compute_metrics_every == 0 or i == len(train_loader) - 1
        ]
    )
    n_metric_computations = max(n_metric_computations, 1)  # Avoid division by zero

    ctx_dice_final = (
        total_context_dice / context_dice_count if context_dice_count > 0 else 0.0
    )
    ctx_softdice_final = (
        total_context_softdice / context_dice_count if context_dice_count > 0 else 0.0
    )
    label_avg_dice = {
        label_id: sum(scores) / len(scores)
        for label_id, scores in label_dice_scores.items()
    }

    # Save train images to disk periodically (async to avoid blocking)
    if (
        save_dir is not None
        and is_main
        and label_samples
        and (epoch % save_every_n_epochs == 0)
    ):
        _save_sample_images_async(
            label_samples,
            save_dir,
            epoch,
            prefix="train",
            max_samples=len(label_samples),
        )

    # Build result dict from all accumulated loss keys
    result = {key: loss_accum[key] / loss_count[key] for key in loss_accum}
    # Add dice metrics (averaged over actual metric computation batches)
    result.update(
        {
            "local_dice": total_local_dice / n_metric_computations,
            "final_dice": total_final_dice / n_metric_computations,
            "context_dice": ctx_dice_final,
            "local_soft_dice": total_local_softdice / n_metric_computations,
            "final_soft_dice": total_final_softdice / n_metric_computations,
            "final_pixel_mae": total_final_pixel_mae / n_metric_computations,
            "context_soft_dice": ctx_softdice_final,
            "per_label": label_avg_dice,
        }
    )
    # Add per-level dice metrics
    for key in dice_accum:
        if dice_count[key] > 0:
            result[key] = dice_accum[key] / dice_count[key]
    return result


@torch.no_grad()
def validate(
    model,
    val_loader,
    device,
    save_dir: Optional[Path] = None,
    max_save_batches: int = 2,
    accelerator=None,
    use_wandb: bool = False,
    epoch: int = 0,
):
    """Run validation."""
    model.eval()  # Keep train mode for BatchNorm consistency
    is_main = accelerator is None or accelerator.is_main_process
    unwrapped_model = (
        accelerator.unwrap_model(model) if accelerator is not None else model
    )

    from collections import defaultdict

    dice_accum = defaultdict(float)
    dice_count = defaultdict(int)
    total_loss = 0.0
    total_local_dice = 0.0
    total_final_dice = 0.0
    total_context_dice = 0.0
    total_local_softdice = 0.0
    total_final_softdice = 0.0
    total_context_softdice = 0.0
    total_final_pixel_mae = 0.0
    context_dice_count = 0

    # Per-case tracking
    case_results = []
    label_dice_scores = {}
    label_samples = {}  # Store one sample per label for wandb image logging

    pbar = tqdm(
        val_loader,
        desc="Validating",
        disable=not is_main,
        unit="batch",
        dynamic_ncols=True,
    )

    for batch_idx, batch in enumerate(pbar):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        context_in = batch.get("context_in")
        context_out = batch.get("context_out")
        if context_in is not None:
            context_in = context_in.to(device)
        if context_out is not None:
            context_out = context_out.to(device)

        target_features = batch.get("target_features")
        context_features = batch.get("context_features")
        if target_features is not None:
            target_features = target_features.to(device)
        if context_features is not None:
            context_features = context_features.to(device)

        if labels.dim() == 3:
            labels = labels.unsqueeze(1)

        outputs = model(
            images,
            labels=labels,
            context_in=context_in,
            context_out=context_out,
            target_features=target_features,
            context_features=context_features,
            mode="val",
        )
        # Compute val loss at level resolution (matches training loss)
        level_outputs = outputs.get("level_outputs", [])
        if level_outputs:
            last_pred = level_outputs[-1]["pred"]
            sf = labels.shape[-1] // last_pred.shape[-1]
            labels_ds = F.avg_pool2d(labels.float(), kernel_size=sf, stride=sf)
            loss = unwrapped_model.aggreg_criterion(last_pred, labels_ds)
        else:
            loss = unwrapped_model.aggreg_criterion(
                outputs["final_logit"], labels.float()
            )
        total_loss += loss.item()

        # Compute all dice metrics using centralized function (with per-sample dice)
        metrics = compute_all_metrics(outputs, labels, return_per_sample=True)

        total_local_dice += metrics.get("local_dice", torch.tensor(0.0)).item()
        total_local_softdice += metrics.get("local_soft_dice", torch.tensor(0.0)).item()
        total_final_dice += metrics.get("final_dice", torch.tensor(0.0)).item()
        total_final_softdice += metrics.get("final_soft_dice", torch.tensor(0.0)).item()
        total_final_pixel_mae += metrics.get("final_pixel_mae", torch.tensor(0.0)).item()

        if "context_dice" in metrics:
            total_context_dice += metrics["context_dice"].item()
            total_context_softdice += metrics["context_soft_dice"].item()
            context_dice_count += 1

        # Per-level dice accumulation
        level_outputs = outputs.get("level_outputs", [])
        for li in range(len(level_outputs)):
            for suffix in [
                "dice",
                "soft_dice",
                "pixel_mae",
                "refined_probs_dice",
                "refined_probs_soft_dice",
            ]:
                key = f"level_{li}_{suffix}"
                if key in metrics:
                    dice_accum[key] += metrics[key].item()
                    dice_count[key] += 1

        # Per-case tracking (reuse per_sample_dice from compute_all_metrics)
        per_sample_dice = metrics.get("per_sample_dice")
        if per_sample_dice is None:
            per_sample_dice = compute_per_sample_dice(outputs, labels)
        batch_case_ids = batch.get("target_case_ids") or batch.get(
            "case_id", [None] * images.shape[0]
        )
        batch_label_ids = batch.get("label_ids") or batch.get(
            "label_id", [None] * images.shape[0]
        )
        batch_axes = batch.get("axes", [None] * images.shape[0])
        for i in range(images.shape[0]):
            case_id = (
                batch_case_ids[i] if batch_case_ids else f"batch{batch_idx}_sample{i}"
            )
            label_id = batch_label_ids[i] if batch_label_ids else "unknown"
            axis = batch_axes[i] if batch_axes else None
            dice_val = per_sample_dice[i].item()
            case_results.append(
                {
                    "case_id": case_id,
                    "label_id": label_id,
                    "axis": axis,
                    "dice": dice_val,
                }
            )
            if label_id not in label_dice_scores:
                label_dice_scores[label_id] = []
            label_dice_scores[label_id].append(dice_val)

            # Store one sample per label for visualization
            if is_main and label_id not in label_samples:
                label_samples[label_id] = _collect_sample_for_viz(
                    i, images, labels, outputs, context_in, context_out, dice_val
                )

        # Update progress bar
        n_batches = batch_idx + 1
        pbar.set_postfix(
            {
                "loss": f"{total_loss / n_batches:.4f}",
                "mae": f"{total_final_pixel_mae / n_batches:.4f}",
                "sdice": f"{total_final_softdice / n_batches:.4f}",
            }
        )

    n = len(val_loader)
    ctx_dice_final = (
        total_context_dice / context_dice_count if context_dice_count > 0 else 0.0
    )
    ctx_softdice_final = (
        total_context_softdice / context_dice_count if context_dice_count > 0 else 0.0
    )
    label_avg_dice = {
        label_id: sum(scores) / len(scores)
        for label_id, scores in label_dice_scores.items()
    }

    # Save images to disk if save_dir is provided (async to avoid blocking)
    if save_dir is not None and is_main and label_samples:
        _save_sample_images_async(
            label_samples,
            save_dir,
            epoch,
            prefix="val",
            max_samples=len(label_samples),
        )

    detailed_results = {
        "per_case": case_results,
        "per_label": label_avg_dice,
        "local_soft_dice": total_local_softdice / n,
        "final_soft_dice": total_final_softdice / n,
        "final_pixel_mae": total_final_pixel_mae / n,
        "context_soft_dice": ctx_softdice_final,
    }
    # Add per-level dice metrics
    for key in dice_accum:
        if dice_count[key] > 0:
            detailed_results[key] = dice_accum[key] / dice_count[key]

    return (
        total_loss / n,
        total_local_dice / n,
        total_final_dice / n,
        ctx_dice_final,
        detailed_results,
    )
