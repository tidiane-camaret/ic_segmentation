"""Training and validation utilities for PatchICL."""
import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


def seed_everything(seed: int) -> None:
    """Set random seeds for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _log_sample_images(label_samples: dict, epoch: int, prefix: str = "train") -> None:
    """Log one image + prediction per label to wandb."""
    try:
        import wandb
    except ImportError:
        return

    wandb_images = []
    for label_id, sample in label_samples.items():
        img = sample["image"]  # [C, H, W]
        label = sample["label"]  # [1, H, W] or [C, H, W]
        pred = sample["pred"]  # [1, H, W] or [C, H, W]
        dice = sample["dice"]

        # Convert image to [H, W, C] for wandb (handle grayscale and RGB)
        if img.shape[0] == 1:
            # Grayscale: repeat to RGB
            img_np = img.squeeze(0).numpy()
            img_np = np.stack([img_np, img_np, img_np], axis=-1)
        elif img.shape[0] == 3:
            img_np = img.permute(1, 2, 0).numpy()
        else:
            # Take first 3 channels or first channel
            img_np = img[:3].permute(1, 2, 0).numpy() if img.shape[0] >= 3 else img[0].numpy()
            if img_np.ndim == 2:
                img_np = np.stack([img_np, img_np, img_np], axis=-1)

        # Normalize to [0, 255]
        img_np = ((img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8) * 255).astype(np.uint8)

        # Get mask data (take first channel if multi-channel)
        gt_mask = (label[0].numpy() > 0).astype(np.uint8)
        pred_mask = (pred[0].numpy() > 0).astype(np.uint8) * 2  # Use 2 for pred to distinguish

        wandb_img = wandb.Image(
            img_np,
            masks={
                "ground_truth": {
                    "mask_data": gt_mask,
                    "class_labels": {0: "background", 1: "foreground"},
                },
                "prediction": {
                    "mask_data": pred_mask,
                    "class_labels": {0: "background", 2: "prediction"},
                },
            },
            caption=f"{label_id} (dice={dice:.3f})",
        )
        wandb_images.append(wandb_img)

    if wandb_images:
        wandb.log({f"{prefix}/samples": wandb_images, "epoch": epoch})


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
):
    """Run one training epoch."""
    model.train()
    is_main = accelerator is None or accelerator.is_main_process
    unwrapped_model = accelerator.unwrap_model(model) if accelerator is not None else model

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

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}", disable=not is_main, unit="batch", dynamic_ncols=True)

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
            patch_pred_binary = (patch_probs > 0.5).float()
            patch_labels_float = patch_labels.float()
            patch_labels_binary = (patch_labels > 0).float()
            # Hard dice
            patch_intersection = (patch_pred_binary * patch_labels_binary).sum(dim=(1, 2, 3, 4))
            patch_union = patch_pred_binary.sum(dim=(1, 2, 3, 4)) + patch_labels_binary.sum(dim=(1, 2, 3, 4))
            local_dice = (2 * patch_intersection + 1e-6) / (patch_union + 1e-6)
            total_local_dice += local_dice.mean().item()
            # Soft dice
            soft_intersection = (patch_probs * patch_labels_float).sum(dim=(1, 2, 3, 4))
            soft_denom = patch_probs.sum(dim=(1, 2, 3, 4)) + patch_labels_float.sum(dim=(1, 2, 3, 4))
            local_softdice = (2 * soft_intersection + 1e-6) / (soft_denom + 1e-6)
            total_local_softdice += local_softdice.mean().item()

            # Final dice (at level resolution)
            level_outputs = outputs.get("level_outputs")
            if level_outputs:
                level_pred = level_outputs[-1]["pred"]
                level_res = level_pred.shape[-1]
                scale_factor = labels.shape[-1] // level_res
                labels_ds = F.max_pool2d(labels.float(), kernel_size=scale_factor, stride=scale_factor)
                pred_probs = torch.sigmoid(level_pred)
                pred_binary = (pred_probs > 0.5).float()
                labels_float = labels_ds
                labels_binary = (labels_ds > 0).float()
            else:
                pred_probs = torch.sigmoid(outputs["final_logit"])
                pred_binary = (pred_probs > 0.5).float()
                labels_float = labels.float()
                labels_binary = (labels > 0).float()
            spatial_dims = tuple(range(2, pred_binary.dim()))
            # Hard dice
            intersection = (pred_binary * labels_binary).sum(dim=spatial_dims)
            union = pred_binary.sum(dim=spatial_dims) + labels_binary.sum(dim=spatial_dims)
            final_dice = (2 * intersection + 1e-6) / (union + 1e-6)
            total_final_dice += final_dice.mean().item()
            # Soft dice
            soft_inter = (pred_probs * labels_float).sum(dim=spatial_dims)
            soft_denom = pred_probs.sum(dim=spatial_dims) + labels_float.sum(dim=spatial_dims)
            final_softdice = (2 * soft_inter + 1e-6) / (soft_denom + 1e-6)
            total_final_softdice += final_softdice.mean().item()

            # Per-label tracking
            batch_label_ids = batch.get("label_ids") or batch.get("label_id", [None] * images.shape[0])
            for i in range(images.shape[0]):
                label_id = batch_label_ids[i] if batch_label_ids else "unknown"
                dice_val = final_dice[i].item() if final_dice.dim() > 0 else final_dice.item()
                if label_id not in label_dice_scores:
                    label_dice_scores[label_id] = []
                label_dice_scores[label_id].append(dice_val)
                # Store one sample per label for wandb image logging
                if use_wandb and is_main and label_id not in label_samples:
                    label_samples[label_id] = {
                        "image": images[i].detach().cpu(),
                        "label": labels[i].detach().cpu(),
                        "pred": pred_binary[i].detach().cpu(),
                        "pred_probs": pred_probs[i].detach().cpu(),
                        "dice": dice_val,
                    }

            # Context dice
            if level_outputs:
                context_pred = level_outputs[-1].get("context_pred")
                context_labels = level_outputs[-1].get("context_labels")
                if context_pred is not None and context_labels is not None:
                    ctx_probs = torch.sigmoid(context_pred)
                    ctx_pred_binary = (ctx_probs > 0.5).float()
                    ctx_labels_float = context_labels.float()
                    ctx_labels_binary = (context_labels > 0).float()
                    # Hard dice
                    ctx_intersection = (ctx_pred_binary * ctx_labels_binary).sum(dim=(2, 3, 4))
                    ctx_union = ctx_pred_binary.sum(dim=(2, 3, 4)) + ctx_labels_binary.sum(dim=(2, 3, 4))
                    ctx_dice = (2 * ctx_intersection + 1e-6) / (ctx_union + 1e-6)
                    total_context_dice += ctx_dice.mean().item()
                    # Soft dice
                    ctx_soft_inter = (ctx_probs * ctx_labels_float).sum(dim=(2, 3, 4))
                    ctx_soft_denom = ctx_probs.sum(dim=(2, 3, 4)) + ctx_labels_float.sum(dim=(2, 3, 4))
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
            optimizer.step()

        # Track metrics
        total_loss += loss.item()
        total_target_patch += losses.get("target_patch_loss", torch.tensor(0.0)).item()
        total_target_aggreg += losses.get("target_aggreg_loss", torch.tensor(0.0)).item()
        total_context_patch += losses.get("context_patch_loss", torch.tensor(0.0)).item()
        total_context_aggreg += losses.get("context_aggreg_loss", torch.tensor(0.0)).item()

        # Update progress bar
        n_batches = idx + 1
        pbar.set_postfix({
            "loss": f"{total_loss / n_batches:.4f}",
            "dice": f"{total_final_dice / n_batches:.4f}",
            "sdice": f"{total_final_softdice / n_batches:.4f}",
        })

        # Log to wandb
        if use_wandb and is_main and idx % log_every == 0:
            global_step = epoch * len(train_loader) + idx
            ctx_dice_avg = total_context_dice / context_dice_count if context_dice_count > 0 else 0.0
            ctx_softdice_avg = total_context_softdice / context_dice_count if context_dice_count > 0 else 0.0
            wandb.log({
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
            }, step=global_step)

        # Print progress
        if print_every and idx % print_every == 0 and is_main:
            ctx_dice_avg = total_context_dice / context_dice_count if context_dice_count > 0 else 0.0
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
    ctx_dice_final = total_context_dice / context_dice_count if context_dice_count > 0 else 0.0
    ctx_softdice_final = total_context_softdice / context_dice_count if context_dice_count > 0 else 0.0
    label_avg_dice = {label_id: sum(scores) / len(scores) for label_id, scores in label_dice_scores.items()}

    # Log one image per label to wandb
    if use_wandb and is_main and label_samples:
        _log_sample_images(label_samples, epoch, prefix="train")

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
    unwrapped_model = accelerator.unwrap_model(model) if accelerator is not None else model

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

    pbar = tqdm(val_loader, desc="Validating", disable=not is_main, unit="batch", dynamic_ncols=True)

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
        patch_pred_binary = (patch_probs > 0.5).float()
        patch_labels_float = patch_labels.float()
        patch_labels_binary = (patch_labels > 0).float()
        # Hard dice
        patch_intersection = (patch_pred_binary * patch_labels_binary).sum(dim=(1, 2, 3, 4))
        patch_union = patch_pred_binary.sum(dim=(1, 2, 3, 4)) + patch_labels_binary.sum(dim=(1, 2, 3, 4))
        local_dice = (2 * patch_intersection + 1e-6) / (patch_union + 1e-6)
        total_local_dice += local_dice.mean().item()
        # Soft dice
        soft_intersection = (patch_probs * patch_labels_float).sum(dim=(1, 2, 3, 4))
        soft_denom = patch_probs.sum(dim=(1, 2, 3, 4)) + patch_labels_float.sum(dim=(1, 2, 3, 4))
        local_softdice = (2 * soft_intersection + 1e-6) / (soft_denom + 1e-6)
        total_local_softdice += local_softdice.mean().item()

        # Final dice
        level_outputs = outputs.get("level_outputs", [])
        if level_outputs:
            level_pred = level_outputs[-1]["pred"]
            level_res = level_pred.shape[-1]
            scale_factor = labels.shape[-1] // level_res
            labels_ds = F.max_pool2d(labels.float(), kernel_size=scale_factor, stride=scale_factor)
            pred_probs = torch.sigmoid(level_pred)
            pred_binary = (pred_probs > 0.5).float()
            labels_float = labels_ds
            labels_binary = (labels_ds > 0).float()
        else:
            pred_probs = torch.sigmoid(predictions)
            pred_binary = (pred_probs > 0.5).float()
            labels_float = labels.float()
            labels_binary = (labels > 0).float()
        spatial_dims = tuple(range(2, pred_binary.dim()))
        # Hard dice
        intersection = (pred_binary * labels_binary).sum(dim=spatial_dims)
        union = pred_binary.sum(dim=spatial_dims) + labels_binary.sum(dim=spatial_dims)
        final_dice = (2 * intersection + 1e-6) / (union + 1e-6)
        total_final_dice += final_dice.mean().item()
        # Soft dice
        soft_inter = (pred_probs * labels_float).sum(dim=spatial_dims)
        soft_denom = pred_probs.sum(dim=spatial_dims) + labels_float.sum(dim=spatial_dims)
        final_softdice = (2 * soft_inter + 1e-6) / (soft_denom + 1e-6)
        total_final_softdice += final_softdice.mean().item()

        # Per-case tracking
        batch_case_ids = batch.get("case_id", [None] * images.shape[0])
        batch_label_ids = batch.get("label_ids") or batch.get("label_id", [None] * images.shape[0])
        for i in range(images.shape[0]):
            case_id = batch_case_ids[i] if batch_case_ids else f"batch{batch_idx}_sample{i}"
            label_id = batch_label_ids[i] if batch_label_ids else "unknown"
            dice_val = final_dice[i].item() if final_dice.dim() > 0 else final_dice.item()
            case_results.append({"case_id": case_id, "label_id": label_id, "dice": dice_val})
            if label_id not in label_dice_scores:
                label_dice_scores[label_id] = []
            label_dice_scores[label_id].append(dice_val)
            # Store one sample per label for wandb image logging
            if is_main and label_id not in label_samples:
                label_samples[label_id] = {
                    "image": images[i].detach().cpu(),
                    "label": labels[i].detach().cpu(),
                    "pred": pred_binary[i].detach().cpu(),
                    "pred_probs": pred_probs[i].detach().cpu(),
                    "dice": dice_val,
                }

        # Context dice
        if level_outputs:
            context_pred = level_outputs[-1].get("context_pred")
            context_labels = level_outputs[-1].get("context_labels")
            if context_pred is not None and context_labels is not None:
                ctx_probs = torch.sigmoid(context_pred)
                ctx_pred_binary = (ctx_probs > 0.5).float()
                ctx_labels_float = context_labels.float()
                ctx_labels_binary = (context_labels > 0).float()
                # Hard dice
                ctx_intersection = (ctx_pred_binary * ctx_labels_binary).sum(dim=(2, 3, 4))
                ctx_union = ctx_pred_binary.sum(dim=(2, 3, 4)) + ctx_labels_binary.sum(dim=(2, 3, 4))
                ctx_dice = (2 * ctx_intersection + 1e-6) / (ctx_union + 1e-6)
                total_context_dice += ctx_dice.mean().item()
                # Soft dice
                ctx_soft_inter = (ctx_probs * ctx_labels_float).sum(dim=(2, 3, 4))
                ctx_soft_denom = ctx_probs.sum(dim=(2, 3, 4)) + ctx_labels_float.sum(dim=(2, 3, 4))
                ctx_softdice = (2 * ctx_soft_inter + 1e-6) / (ctx_soft_denom + 1e-6)
                total_context_softdice += ctx_softdice.mean().item()
                context_dice_count += 1

        # Update progress bar
        n_batches = batch_idx + 1
        pbar.set_postfix({
            "loss": f"{total_loss / n_batches:.4f}",
            "dice": f"{total_final_dice / n_batches:.4f}",
            "sdice": f"{total_final_softdice / n_batches:.4f}",
        })

    n = len(val_loader)
    ctx_dice_final = total_context_dice / context_dice_count if context_dice_count > 0 else 0.0
    ctx_softdice_final = total_context_softdice / context_dice_count if context_dice_count > 0 else 0.0
    label_avg_dice = {label_id: sum(scores) / len(scores) for label_id, scores in label_dice_scores.items()}

    # Log one image per label to wandb
    if use_wandb and is_main and label_samples:
        _log_sample_images(label_samples, epoch, prefix="val")

    detailed_results = {
        "per_case": case_results,
        "per_label": label_avg_dice,
        "local_softdice": total_local_softdice / n,
        "final_softdice": total_final_softdice / n,
        "context_softdice": ctx_softdice_final,
    }

    return total_loss / n, total_local_dice / n, total_final_dice / n, ctx_dice_final, detailed_results
