"""Training and validation utilities for PatchICL."""

import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


# Thresholds for hard-dice metric binarization
PRED_THRESHOLD = 0.5    # sigmoid probability → binary prediction
GT_AREA_THRESHOLD = 0.25  # soft avg-pooled GT → binary (≥25% coverage = foreground)


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
            levels_info = [{
                "target_coords": sample.get("target_coords"),
                "context_coords": sample.get("context_coords"),
                "patch_size": sample.get("patch_size", 16),
                "level_res": sample.get("level_res", 32),
                "pred_probs": sample.get("pred_probs"),
            }]

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

            # --- Top row: target + context images with patch boxes ---
            axes[top_row][0].imshow(img, cmap="gray")
            axes[top_row][0].imshow(gt, cmap="Reds", alpha=0.3)
            if l_target_coords is not None and l_level_res is not None:
                draw_patch_boxes(
                    axes[top_row][0], img, l_target_coords,
                    l_patch_size, l_level_res, color="lime",
                )
            axes[top_row][0].set_title(f"Level {li} (res {l_level_res}) - Target")
            axes[top_row][0].axis("off")

            for ci in range(n_ctx):
                ctx_img = ctx_in[ci].squeeze().numpy()
                ctx_mask = ctx_out[ci].squeeze().numpy()
                axes[top_row][1 + ci].imshow(ctx_img, cmap="gray")
                axes[top_row][1 + ci].imshow(ctx_mask, cmap="Reds", alpha=0.3)
                if (
                    l_context_coords is not None
                    and l_level_res is not None
                    and l_context_coords.ndim == 2
                ):
                    try:
                        total_patches = l_context_coords.shape[0]
                        K_per_ctx = total_patches // n_ctx
                        ci_coords = l_context_coords[ci * K_per_ctx : (ci + 1) * K_per_ctx]
                        draw_patch_boxes(
                            axes[top_row][1 + ci], ctx_img, ci_coords,
                            l_patch_size, l_level_res, color="cyan",
                        )
                    except (IndexError, TypeError, ZeroDivisionError):
                        pass
                axes[top_row][1 + ci].set_title(f"Context {ci + 1}")
                axes[top_row][1 + ci].axis("off")

            for j in range(1 + n_ctx, n_cols):
                axes[top_row][j].axis("off")

            # --- Bottom row: Downsized GT | Pred heatmap | Pred mask ---
            # Downsized GT
            if l_level_res is not None and l_level_res < gt.shape[0]:
                gt_t = torch.from_numpy(gt).unsqueeze(0).unsqueeze(0).float()
                scale = max(1, gt.shape[0] // l_level_res)
                gt_ds = F.avg_pool2d(gt_t, kernel_size=scale, stride=scale).squeeze().numpy()
            else:
                gt_ds = gt
            axes[bot_row][0].imshow(gt_ds, cmap="gray")
            axes[bot_row][0].set_title(f"GT ({gt_ds.shape[0]}x{gt_ds.shape[1]})")
            axes[bot_row][0].axis("off")

            # Pred heatmap
            if l_pred_probs is not None:
                pp = l_pred_probs.squeeze().numpy()
                pred_heatmap = (pp - pp.min()) / (pp.max() - pp.min() + 1e-8)
                im = axes[bot_row][1].imshow(pred_heatmap, cmap="hot", vmin=0, vmax=1)
                axes[bot_row][1].set_title(f"Heatmap ({pp.shape[0]}x{pp.shape[1]})")
                plt.colorbar(im, ax=axes[bot_row][1], fraction=0.046, pad=0.04)
            else:
                axes[bot_row][1].set_title("Heatmap (N/A)")
            axes[bot_row][1].axis("off")

            # Pred mask with per-level dice
            if l_pred_probs is not None:
                pp = l_pred_probs.squeeze()
                # Resize pred to match gt_ds if needed (backward compat)
                if pp.shape[0] != gt_ds.shape[0]:
                    pp = F.interpolate(
                        pp.unsqueeze(0).unsqueeze(0).float(),
                        size=(gt_ds.shape[0], gt_ds.shape[1]),
                        mode='bilinear', align_corners=False,
                    ).squeeze()
                pp = pp.numpy() if hasattr(pp, 'numpy') else pp
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

            for j in range(3, n_cols):
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

    # Metrics
    total_loss = 0.0
    total_local_dice = 0.0
    total_final_dice = 0.0
    total_context_dice = 0.0
    total_local_softdice = 0.0
    total_final_softdice = 0.0
    total_context_softdice = 0.0
    context_dice_count = 0
    total_target_patch = 0.0
    total_target_aggreg = 0.0
    total_context_patch = 0.0
    total_context_aggreg = 0.0
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

        # Compute Dice scores
        with torch.no_grad():
            # Local dice (patch level)
            patch_logits = outputs["patch_logits"]
            patch_labels = outputs["patch_labels"]
            patch_probs = torch.sigmoid(patch_logits)
            patch_pred_binary = (patch_probs > PRED_THRESHOLD).float()
            patch_labels_float = patch_labels.float()
            patch_labels_binary = (patch_labels > GT_AREA_THRESHOLD).float()
            # Hard dice
            patch_intersection = (patch_pred_binary * patch_labels_binary).sum(
                dim=(1, 2, 3, 4)
            )
            patch_union = patch_pred_binary.sum(
                dim=(1, 2, 3, 4)
            ) + patch_labels_binary.sum(dim=(1, 2, 3, 4))
            local_dice = (2 * patch_intersection + 1e-6) / (patch_union + 1e-6)
            total_local_dice += local_dice.mean().item()
            # Soft dice
            soft_intersection = (patch_probs * patch_labels_float).sum(dim=(1, 2, 3, 4))
            soft_denom = patch_probs.sum(dim=(1, 2, 3, 4)) + patch_labels_float.sum(
                dim=(1, 2, 3, 4)
            )
            local_softdice = (2 * soft_intersection + 1e-6) / (soft_denom + 1e-6)
            total_local_softdice += local_softdice.mean().item()

            # Final dice (at level resolution)
            level_outputs = outputs.get("level_outputs")
            if level_outputs:
                level_pred = level_outputs[-1]["pred"]
                level_res = level_pred.shape[-1]
                scale_factor = labels.shape[-1] // level_res
                # avg_pool2d gives fractional coverage per patch (avoids
                # inflating sparse masks the way max_pool2d does)
                labels_ds = F.avg_pool2d(
                    labels.float(), kernel_size=scale_factor, stride=scale_factor
                )
                pred_probs = torch.sigmoid(level_pred)
                pred_binary = (pred_probs > PRED_THRESHOLD).float()
                labels_float = labels_ds  # soft target for soft dice
                labels_binary = (labels_ds > GT_AREA_THRESHOLD).float()
            else:
                pred_probs = torch.sigmoid(outputs["final_logit"])
                pred_binary = (pred_probs > PRED_THRESHOLD).float()
                labels_float = labels.float()
                labels_binary = (labels > GT_AREA_THRESHOLD).float()
            spatial_dims = tuple(range(2, pred_binary.dim()))
            # Hard dice
            intersection = (pred_binary * labels_binary).sum(dim=spatial_dims)
            union = pred_binary.sum(dim=spatial_dims) + labels_binary.sum(
                dim=spatial_dims
            )
            final_dice = (2 * intersection + 1e-6) / (union + 1e-6)
            total_final_dice += final_dice.mean().item()
            # Soft dice
            soft_inter = (pred_probs * labels_float).sum(dim=spatial_dims)
            soft_denom = pred_probs.sum(dim=spatial_dims) + labels_float.sum(
                dim=spatial_dims
            )
            final_softdice = (2 * soft_inter + 1e-6) / (soft_denom + 1e-6)
            total_final_softdice += final_softdice.mean().item()

            # Per-label tracking
            batch_label_ids = batch.get("label_ids") or batch.get(
                "label_id", [None] * images.shape[0]
            )
            for i in range(images.shape[0]):
                label_id = batch_label_ids[i] if batch_label_ids else "unknown"
                dice_val = (
                    final_dice[i].item() if final_dice.dim() > 0 else final_dice.item()
                )
                if label_id not in label_dice_scores:
                    label_dice_scores[label_id] = []
                label_dice_scores[label_id].append(dice_val)
                # Store one sample per label for image logging (wandb or disk)
                should_save = (use_wandb or save_dir is not None) and is_main
                if should_save and label_id not in label_samples:
                    # Collect per-level info for visualization
                    levels = []
                    if level_outputs:
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
                            levels.append({
                                "target_coords": l_coords,
                                "context_coords": l_ctx_coords,
                                "patch_size": level_out.get("patch_size", 16),
                                "level_res": level_out.get("level_res", 32),
                                "pred_probs": l_pred_probs,
                            })

                    label_samples[label_id] = {
                        "image": images[i].detach().cpu(),
                        "label": labels[i].detach().cpu(),
                        "pred": pred_binary[i].detach().cpu(),
                        "pred_probs": pred_probs[i].detach().cpu(),
                        "dice": dice_val,
                        "context_in": (
                            context_in[i].detach().cpu()
                            if context_in is not None
                            else None
                        ),
                        "context_out": (
                            context_out[i].detach().cpu()
                            if context_out is not None
                            else None
                        ),
                        "levels": levels,
                    }

            # Context dice
            if level_outputs:
                context_pred = level_outputs[-1].get("context_pred")
                context_labels = level_outputs[-1].get("context_labels")
                if context_pred is not None and context_labels is not None:
                    ctx_probs = torch.sigmoid(context_pred)
                    ctx_pred_binary = (ctx_probs > PRED_THRESHOLD).float()
                    ctx_labels_float = context_labels.float()
                    ctx_labels_binary = (context_labels > GT_AREA_THRESHOLD).float()
                    # Hard dice
                    ctx_intersection = (ctx_pred_binary * ctx_labels_binary).sum(
                        dim=(2, 3, 4)
                    )
                    ctx_union = ctx_pred_binary.sum(
                        dim=(2, 3, 4)
                    ) + ctx_labels_binary.sum(dim=(2, 3, 4))
                    ctx_dice = (2 * ctx_intersection + 1e-6) / (ctx_union + 1e-6)
                    total_context_dice += ctx_dice.mean().item()
                    # Soft dice
                    ctx_soft_inter = (ctx_probs * ctx_labels_float).sum(dim=(2, 3, 4))
                    ctx_soft_denom = ctx_probs.sum(
                        dim=(2, 3, 4)
                    ) + ctx_labels_float.sum(dim=(2, 3, 4))
                    ctx_softdice = (2 * ctx_soft_inter + 1e-6) / (ctx_soft_denom + 1e-6)
                    total_context_softdice += ctx_softdice.mean().item()
                    context_dice_count += 1

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
        total_loss += loss.item()
        total_target_patch += losses.get("target_patch_loss", torch.tensor(0.0)).item()
        total_target_aggreg += losses.get(
            "target_aggreg_loss", torch.tensor(0.0)
        ).item()
        total_context_patch += losses.get(
            "context_patch_loss", torch.tensor(0.0)
        ).item()
        total_context_aggreg += losses.get(
            "context_aggreg_loss", torch.tensor(0.0)
        ).item()

        # Update progress bar
        n_batches = idx + 1
        pbar.set_postfix(
            {
                "loss": f"{total_loss / n_batches:.4f}",
                "dice": f"{total_final_dice / n_batches:.4f}",
                "sdice": f"{total_final_softdice / n_batches:.4f}",
            }
        )

        # Log to wandb
        if use_wandb and is_main and idx % log_every == 0:
            global_step = epoch * len(train_loader) + idx
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
            wandb.log(
                {
                    "train_batch/loss": total_loss / n_batches,
                    "train_batch/local_dice": total_local_dice / n_batches,
                    "train_batch/final_dice": total_final_dice / n_batches,
                    "train_batch/context_dice": ctx_dice_avg,
                    "train_batch/local_softdice": total_local_softdice / n_batches,
                    "train_batch/final_softdice": total_final_softdice / n_batches,
                    "train_batch/context_softdice": ctx_softdice_avg,
                    "train_batch/target_patch_loss": total_target_patch / n_batches,
                    "train_batch/target_aggreg_loss": total_target_aggreg / n_batches,
                    "train_batch/context_patch_loss": total_context_patch / n_batches,
                    "train_batch/context_aggreg_loss": total_context_aggreg / n_batches,
                    "global_step": global_step,
                },
                step=global_step,
            )

        # Print progress
        if print_every and idx % print_every == 0 and is_main:
            ctx_dice_avg = (
                total_context_dice / context_dice_count
                if context_dice_count > 0
                else 0.0
            )
            print(
                f"Epoch {epoch:04d} | Batch {idx:04d} | "
                f"Loss: {total_loss / n_batches:.5f} | "
                f"LocalDice: {total_local_dice / n_batches:.5f} | "
                f"FinalDice: {total_final_dice / n_batches:.5f} | "
                f"SoftDice: {total_final_softdice / n_batches:.5f} | "
                f"CtxDice: {ctx_dice_avg:.5f}"
            )

        del outputs, losses
        if idx % 10 == 0:
            torch.cuda.empty_cache()

    n = len(train_loader)
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

    # Save train images to disk periodically
    if (
        save_dir is not None
        and is_main
        and label_samples
        and (epoch % save_every_n_epochs == 0)
    ):
        _save_sample_images(label_samples, save_dir, epoch, prefix="train", max_samples=len(label_samples))

    return {
        "loss": total_loss / n,
        "local_dice": total_local_dice / n,
        "final_dice": total_final_dice / n,
        "context_dice": ctx_dice_final,
        "local_softdice": total_local_softdice / n,
        "final_softdice": total_final_softdice / n,
        "context_softdice": ctx_softdice_final,
        "target_patch_loss": total_target_patch / n,
        "target_aggreg_loss": total_target_aggreg / n,
        "context_patch_loss": total_context_patch / n,
        "context_aggreg_loss": total_context_aggreg / n,
        "target_loss": (total_target_patch + total_target_aggreg) / n,
        "context_loss": (total_context_patch + total_context_aggreg) / n,
        "patch_loss_total": (total_target_patch + total_context_patch) / n,
        "aggreg_loss_total": (total_target_aggreg + total_context_aggreg) / n,
        "aggreg_loss": total_target_aggreg / n,
        "local_loss": total_target_patch / n,
        "agg_loss": total_loss / n,
        "per_label": label_avg_dice,
    }


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

    total_loss = 0.0
    total_local_dice = 0.0
    total_final_dice = 0.0
    total_context_dice = 0.0
    total_local_softdice = 0.0
    total_final_softdice = 0.0
    total_context_softdice = 0.0
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
        predictions = outputs["final_logit"]
        loss = unwrapped_model.aggreg_criterion(predictions, labels.float())
        total_loss += loss.item()

        # Local dice
        patch_logits = outputs["patch_logits"]
        patch_labels = outputs["patch_labels"]
        patch_probs = torch.sigmoid(patch_logits)
        patch_pred_binary = (patch_probs > PRED_THRESHOLD).float()
        patch_labels_float = patch_labels.float()
        patch_labels_binary = (patch_labels > GT_AREA_THRESHOLD).float()
        # Hard dice
        patch_intersection = (patch_pred_binary * patch_labels_binary).sum(
            dim=(1, 2, 3, 4)
        )
        patch_union = patch_pred_binary.sum(dim=(1, 2, 3, 4)) + patch_labels_binary.sum(
            dim=(1, 2, 3, 4)
        )
        local_dice = (2 * patch_intersection + 1e-6) / (patch_union + 1e-6)
        total_local_dice += local_dice.mean().item()
        # Soft dice
        soft_intersection = (patch_probs * patch_labels_float).sum(dim=(1, 2, 3, 4))
        soft_denom = patch_probs.sum(dim=(1, 2, 3, 4)) + patch_labels_float.sum(
            dim=(1, 2, 3, 4)
        )
        local_softdice = (2 * soft_intersection + 1e-6) / (soft_denom + 1e-6)
        total_local_softdice += local_softdice.mean().item()

        # Final dice
        level_outputs = outputs.get("level_outputs", [])
        if level_outputs:
            level_pred = level_outputs[-1]["pred"]
            level_res = level_pred.shape[-1]
            scale_factor = labels.shape[-1] // level_res
            # avg_pool2d gives fractional coverage per patch (avoids
            # inflating sparse masks the way max_pool2d does)
            labels_ds = F.avg_pool2d(
                labels.float(), kernel_size=scale_factor, stride=scale_factor
            )
            pred_probs = torch.sigmoid(level_pred)
            pred_binary = (pred_probs > PRED_THRESHOLD).float()
            labels_float = labels_ds  # soft target for soft dice
            labels_binary = (labels_ds > GT_AREA_THRESHOLD).float()
        else:
            pred_probs = torch.sigmoid(predictions)
            pred_binary = (pred_probs > PRED_THRESHOLD).float()
            labels_float = labels.float()
            labels_binary = (labels > GT_AREA_THRESHOLD).float()
        spatial_dims = tuple(range(2, pred_binary.dim()))
        # Hard dice
        intersection = (pred_binary * labels_binary).sum(dim=spatial_dims)
        union = pred_binary.sum(dim=spatial_dims) + labels_binary.sum(dim=spatial_dims)
        final_dice = (2 * intersection + 1e-6) / (union + 1e-6)
        total_final_dice += final_dice.mean().item()
        # Soft dice
        soft_inter = (pred_probs * labels_float).sum(dim=spatial_dims)
        soft_denom = pred_probs.sum(dim=spatial_dims) + labels_float.sum(
            dim=spatial_dims
        )
        final_softdice = (2 * soft_inter + 1e-6) / (soft_denom + 1e-6)
        total_final_softdice += final_softdice.mean().item()

        # Per-case tracking
        batch_case_ids = batch.get("case_id", [None] * images.shape[0])
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
            dice_val = (
                final_dice[i].item() if final_dice.dim() > 0 else final_dice.item()
            )
            case_results.append(
                {"case_id": case_id, "label_id": label_id, "axis": axis, "dice": dice_val}
            )
            if label_id not in label_dice_scores:
                label_dice_scores[label_id] = []
            label_dice_scores[label_id].append(dice_val)
            # Store one sample per label for wandb image logging
            if is_main and label_id not in label_samples:
                # Collect per-level info for visualization
                levels = []
                if level_outputs:
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
                        levels.append({
                            "target_coords": l_coords,
                            "context_coords": l_ctx_coords,
                            "patch_size": level_out.get("patch_size", 16),
                            "level_res": level_out.get("level_res", 32),
                            "pred_probs": l_pred_probs,
                        })

                label_samples[label_id] = {
                    "image": images[i].detach().cpu(),
                    "label": labels[i].detach().cpu(),
                    "pred": pred_binary[i].detach().cpu(),
                    "pred_probs": pred_probs[i].detach().cpu(),
                    "dice": dice_val,
                    "context_in": (
                        context_in[i].detach().cpu() if context_in is not None else None
                    ),
                    "context_out": (
                        context_out[i].detach().cpu()
                        if context_out is not None
                        else None
                    ),
                    "levels": levels,
                }

        # Context dice
        if level_outputs:
            context_pred = level_outputs[-1].get("context_pred")
            context_labels = level_outputs[-1].get("context_labels")
            if context_pred is not None and context_labels is not None:
                ctx_probs = torch.sigmoid(context_pred)
                ctx_pred_binary = (ctx_probs > PRED_THRESHOLD).float()
                ctx_labels_float = context_labels.float()
                ctx_labels_binary = (context_labels > GT_AREA_THRESHOLD).float()
                # Hard dice
                ctx_intersection = (ctx_pred_binary * ctx_labels_binary).sum(
                    dim=(2, 3, 4)
                )
                ctx_union = ctx_pred_binary.sum(dim=(2, 3, 4)) + ctx_labels_binary.sum(
                    dim=(2, 3, 4)
                )
                ctx_dice = (2 * ctx_intersection + 1e-6) / (ctx_union + 1e-6)
                total_context_dice += ctx_dice.mean().item()
                # Soft dice
                ctx_soft_inter = (ctx_probs * ctx_labels_float).sum(dim=(2, 3, 4))
                ctx_soft_denom = ctx_probs.sum(dim=(2, 3, 4)) + ctx_labels_float.sum(
                    dim=(2, 3, 4)
                )
                ctx_softdice = (2 * ctx_soft_inter + 1e-6) / (ctx_soft_denom + 1e-6)
                total_context_softdice += ctx_softdice.mean().item()
                context_dice_count += 1

        # Update progress bar
        n_batches = batch_idx + 1
        pbar.set_postfix(
            {
                "loss": f"{total_loss / n_batches:.4f}",
                "dice": f"{total_final_dice / n_batches:.4f}",
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


    # Save images to disk if save_dir is provided
    if save_dir is not None and is_main and label_samples:
        _save_sample_images(
            label_samples,
            save_dir,
            epoch,
            prefix="val",
            max_samples=len(label_samples),
        )

    detailed_results = {
        "per_case": case_results,
        "per_label": label_avg_dice,
        "local_softdice": total_local_softdice / n,
        "final_softdice": total_final_softdice / n,
        "context_softdice": ctx_softdice_final,
    }

    return (
        total_loss / n,
        total_local_dice / n,
        total_final_dice / n,
        ctx_dice_final,
        detailed_results,
    )
