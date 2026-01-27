import os
import random
from pathlib import Path
from typing import Dict, Optional

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import DataLoader


def seed_everything(seed: int) -> None:
    """Set random seeds for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_dataloaders(config: Dict, DatasetClass, collate_fn) -> tuple:
    """Build train and validation dataloaders."""
    ds_cfg = config["dataset_parameters"]

    train_dataset = DatasetClass(
        root_dir=ds_cfg["dataset_path"],
        label_id_list=ds_cfg["train_label_ids"],
        context_size=0,
        image_size=tuple(ds_cfg["image_size"]),
        spacing=tuple(ds_cfg["spacing"]),
        split="train",
        random_context=False,
        max_ds_len=ds_cfg.get("max_ds_len"),
    )

    val_dataset = DatasetClass(
        root_dir=ds_cfg["dataset_path"],
        label_id_list=ds_cfg["val_label_ids"],
        context_size=0,
        image_size=tuple(ds_cfg["image_size"]),
        spacing=tuple(ds_cfg["spacing"]),
        split="val",
        random_context=False,
        max_ds_len=ds_cfg.get("max_ds_len"),
    )

    print(f"Train dataset: {len(train_dataset)} samples")
    print(f"Val dataset: {len(val_dataset)} samples")

    train_loader = DataLoader(
        train_dataset,
        batch_size=ds_cfg["train_batch_size"],
        shuffle=True,
        num_workers=ds_cfg["num_workers"],
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=ds_cfg["val_batch_size"],
        shuffle=False,
        num_workers=ds_cfg["num_workers"],
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader


def train_epoch(model, train_loader, optimizer, device, epoch, print_every, grad_accumulate_steps=1, accelerator=None, use_wandb=False, log_every=10):
    """Run one training epoch. Model must have loss functions set via set_loss_functions().
    
    Args:
        use_wandb: If True, log batch metrics to wandb
        log_every: Log to wandb every N batches (default: 10)
    """
    model.train()
    is_main = accelerator is None or accelerator.is_main_process
    
    # Import wandb if needed
    if use_wandb and is_main:
        try:
            import wandb
        except ImportError:
            use_wandb = False
    # Get unwrapped model for accessing custom methods (compute_loss, etc.)
    unwrapped_model = accelerator.unwrap_model(model) if accelerator is not None else model
    total_loss = 0.0
    total_aggreg = 0.0
    total_local = 0.0
    total_agg = 0.0
    total_local_dice = 0.0
    total_final_dice = 0.0

    # Track detailed losses
    total_target_patch = 0.0
    total_target_aggreg = 0.0
    total_context_patch = 0.0
    total_context_aggreg = 0.0
    total_context_dice = 0.0
    context_dice_count = 0

    # Track feature losses
    total_feature_patch = 0.0
    total_feature_aggreg = 0.0
    total_context_feature_patch = 0.0
    total_context_feature_aggreg = 0.0

    for idx, batch in enumerate(train_loader):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        # Get context data if available
        context_in = batch.get("context_in", None)
        context_out = batch.get("context_out", None)
        if context_in is not None:
            context_in = context_in.to(device)
        if context_out is not None:
            context_out = context_out.to(device)

        # Get pre-computed features if available
        target_features = batch.get("target_features", None)
        context_features = batch.get("context_features", None)
        if target_features is not None:
            target_features = target_features.to(device)
        if context_features is not None:
            context_features = context_features.to(device)

        # Ensure labels have channel dim
        if labels.dim() == 3:
            labels = labels.unsqueeze(1)

        # Zero gradients at start of accumulation window
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
            # Local dice: on sampled patches
            patch_logits = outputs["patch_logits"]  # [B, K, 1, ps, ps]
            patch_labels = outputs["patch_labels"]  # [B, K, 1, ps, ps]
            patch_pred_binary = (torch.sigmoid(patch_logits) > 0.5).float()
            patch_labels_binary = (patch_labels > 0).float()
            # Compute dice per patch then average
            B, K = patch_logits.shape[:2]
            patch_intersection = (patch_pred_binary * patch_labels_binary).sum(dim=(2, 3, 4))
            patch_union = patch_pred_binary.sum(dim=(2, 3, 4)) + patch_labels_binary.sum(dim=(2, 3, 4))
            local_dice = (2 * patch_intersection + 1e-6) / (patch_union + 1e-6)
            total_local_dice += local_dice.mean().item()

            # Final dice: on full prediction
            pred_binary = (torch.sigmoid(outputs["final_logit"]) > 0.5).float()
            labels_binary = (labels > 0).float()
            spatial_dims = tuple(range(2, pred_binary.dim()))
            intersection = (pred_binary * labels_binary).sum(dim=spatial_dims)
            union = pred_binary.sum(dim=spatial_dims) + labels_binary.sum(dim=spatial_dims)
            final_dice = (2 * intersection + 1e-6) / (union + 1e-6)
            total_final_dice += final_dice.mean().item()

            # Context dice: on aggregated context predictions vs context GT
            # Get from finest level outputs
            level_outputs = outputs.get("level_outputs")
            if level_outputs:
                finest_level = level_outputs[-1]
                context_pred = finest_level.get("context_pred")  # [B, k, 1, res, res]
                context_labels = finest_level.get("context_labels")  # [B, k, 1, res, res]
                if context_pred is not None and context_labels is not None:
                    ctx_pred_binary = (torch.sigmoid(context_pred) > 0.5).float()
                    ctx_labels_binary = (context_labels > 0).float()
                    # Compute dice per context image then average
                    ctx_intersection = (ctx_pred_binary * ctx_labels_binary).sum(dim=(2, 3, 4))
                    ctx_union = ctx_pred_binary.sum(dim=(2, 3, 4)) + ctx_labels_binary.sum(dim=(2, 3, 4))
                    ctx_dice = (2 * ctx_intersection + 1e-6) / (ctx_union + 1e-6)
                    total_context_dice += ctx_dice.mean().item()
                    context_dice_count += 1

        # Scale loss for gradient accumulation
        scaled_loss = loss / grad_accumulate_steps
        if accelerator is not None:
            accelerator.backward(scaled_loss)
        else:
            scaled_loss.backward()

        # Step optimizer at end of accumulation window
        if (idx + 1) % grad_accumulate_steps == 0:
            optimizer.step()

        total_loss += loss.item()
        total_aggreg += losses.get("aggreg_loss", torch.tensor(0.0)).item()
        total_local += losses["local_loss"].item()
        total_agg += losses["agg_loss"].item()

        # Track detailed losses (use .item() to get scalar, handle tensor(0.0) case)
        total_target_patch += losses.get("target_patch_loss", torch.tensor(0.0)).item()
        total_target_aggreg += losses.get("target_aggreg_loss", torch.tensor(0.0)).item()
        total_context_patch += losses.get("context_patch_loss", torch.tensor(0.0)).item()
        total_context_aggreg += losses.get("context_aggreg_loss", torch.tensor(0.0)).item()

        # Track feature losses
        total_feature_patch += losses.get("target_feature_patch_loss", torch.tensor(0.0)).item()
        total_feature_aggreg += losses.get("target_feature_aggreg_loss", torch.tensor(0.0)).item()
        total_context_feature_patch += losses.get("context_feature_patch_loss", torch.tensor(0.0)).item()
        total_context_feature_aggreg += losses.get("context_feature_aggreg_loss", torch.tensor(0.0)).item()

        # Free memory
        del outputs, losses
        if idx % 10 == 0:
            torch.cuda.empty_cache()

        if print_every and idx % print_every == 0 and is_main:
            ctx_dice_avg = total_context_dice / context_dice_count if context_dice_count > 0 else 0.0
            print(
                f"Epoch {epoch:04d} | Batch {idx:04d} | "
                f"Loss: {total_loss / (idx + 1):.5f} | "
                f"Last level target patch Dice: {total_local_dice / (idx + 1):.5f} | "
                f"Last level target aggreg Dice: {total_final_dice / (idx + 1):.5f} | "
                f"Last level context aggreg Dice: {ctx_dice_avg:.5f}"
            )
            print(
                f"  Losses -> "
                f"TargetPatch: {total_target_patch / (idx + 1):.4f} | "
                f"TargetAggreg: {total_target_aggreg / (idx + 1):.4f} | "
                f"ContextPatch: {total_context_patch / (idx + 1):.4f} | "
                f"ContextAggreg: {total_context_aggreg / (idx + 1):.4f}"
            )

        # Log to wandb during training
        if use_wandb and is_main and idx % log_every == 0:
            global_step = epoch * len(train_loader) + idx
            ctx_dice_avg = total_context_dice / context_dice_count if context_dice_count > 0 else 0.0
            wandb.log({
                "train_batch/loss": total_loss / (idx + 1),
                "train_batch/local_dice": total_local_dice / (idx + 1),
                "train_batch/final_dice": total_final_dice / (idx + 1),
                "train_batch/context_dice": ctx_dice_avg,
                "train_batch/target_patch_loss": total_target_patch / (idx + 1),
                "train_batch/target_aggreg_loss": total_target_aggreg / (idx + 1),
                "train_batch/context_patch_loss": total_context_patch / (idx + 1),
                "train_batch/context_aggreg_loss": total_context_aggreg / (idx + 1),
                "train_batch/feature_patch_loss": total_feature_patch / (idx + 1),
                "train_batch/feature_aggreg_loss": total_feature_aggreg / (idx + 1),
                "train_batch/epoch": epoch,
                "train_batch/batch": idx,
                "global_step": global_step,
            }, step=global_step)

    n = len(train_loader)
    ctx_dice_final = total_context_dice / context_dice_count if context_dice_count > 0 else 0.0
    return {
        "loss": total_loss / n,
        "local_dice": total_local_dice / n,
        "final_dice": total_final_dice / n,
        "context_dice": ctx_dice_final,
        "aggreg_loss": total_aggreg / n,
        "local_loss": total_local / n,
        "agg_loss": total_agg / n,
        # Detailed losses
        "target_patch_loss": total_target_patch / n,
        "target_aggreg_loss": total_target_aggreg / n,
        "context_patch_loss": total_context_patch / n,
        "context_aggreg_loss": total_context_aggreg / n,
        "target_loss": (total_target_patch + total_target_aggreg) / n,
        "context_loss": (total_context_patch + total_context_aggreg) / n,
        "patch_loss_total": (total_target_patch + total_context_patch) / n,
        "aggreg_loss_total": (total_target_aggreg + total_context_aggreg) / n,
        # Feature losses
        "target_feature_patch_loss": total_feature_patch / n,
        "target_feature_aggreg_loss": total_feature_aggreg / n,
        "context_feature_patch_loss": total_context_feature_patch / n,
        "context_feature_aggreg_loss": total_context_feature_aggreg / n,
    }

def save_predictions(save_dir: Path, case_ids: list, images, labels, outputs, max_samples=10,
                     context_in=None, context_out=None, batch_idx=0):
    """Save images, masks, and predictions to NIfTI files organized by case ID."""
    save_dir = Path(save_dir)
    B = min(images.shape[0], max_samples)

    for i in range(B):
        case_id = case_ids[i] if case_ids else f"case{i:04d}"
        # Include batch_idx and sample index to avoid overwrites when same case_id appears multiple times
        case_dir = save_dir / f"{case_id}_b{batch_idx:02d}_s{i:02d}"
        case_dir.mkdir(parents=True, exist_ok=True)

        # Input image
        img_nib = nib.Nifti1Image(images[i, 0].cpu().numpy(), affine=np.eye(4))
        nib.save(img_nib, case_dir / "img.nii.gz")

        # Ground truth mask
        gt_nib = nib.Nifti1Image(labels[i, 0].cpu().numpy().astype(np.float32), affine=np.eye(4))
        nib.save(gt_nib, case_dir / "gt_mask.nii.gz")

        # Coarse/global prediction (resize to original image size)
        coarse = outputs["coarse_pred"][i:i+1]  # Keep batch dim for interpolate
        target_size = images.shape[2:]  # (H, W) or (D, H, W)
        coarse_resized = torch.nn.functional.interpolate(
            coarse, size=target_size, mode="nearest"
        )
        coarse_nib = nib.Nifti1Image(coarse_resized[0, 0].cpu().numpy().astype(np.float32), affine=np.eye(4))
        nib.save(coarse_nib, case_dir / "coarse_pred_mask.nii.gz")

        # Final aggregated prediction
        final = torch.sigmoid(outputs["final_logit"][i, 0]).cpu().numpy()
        final_nib = nib.Nifti1Image(final.astype(np.float32), affine=np.eye(4))
        nib.save(final_nib, case_dir / "final_pred_mask.nii.gz")

        # Save first few patches and their predictions
        K = min(4, outputs["patches"].shape[1])
        for k in range(K):
            patch_img = outputs["patches"][i, k, 0].cpu().numpy()
            patch_img_nib = nib.Nifti1Image(patch_img.astype(np.float32), affine=np.eye(4))
            nib.save(patch_img_nib, case_dir / f"patch{k}_img.nii.gz")

            patch_label = outputs["patch_labels"][i, k, 0].cpu().numpy()
            patch_label_nib = nib.Nifti1Image(patch_label.astype(np.float32), affine=np.eye(4))
            nib.save(patch_label_nib, case_dir / f"patch{k}_gt_mask.nii.gz")

            patch_pred = torch.sigmoid(outputs["patch_logits"][i, k, 0]).cpu().numpy()
            patch_pred_nib = nib.Nifti1Image(patch_pred.astype(np.float32), affine=np.eye(4))
            nib.save(patch_pred_nib, case_dir / f"patch{k}_pred_mask.nii.gz")

        # Save context images and masks if provided
        if context_in is not None and context_out is not None:
            n_ctx = context_in.shape[1]  # [B, k, C, H, W]
            for c in range(n_ctx):
                ctx_img = context_in[i, c, 0].cpu().numpy()
                ctx_img_nib = nib.Nifti1Image(ctx_img.astype(np.float32), affine=np.eye(4))
                nib.save(ctx_img_nib, case_dir / f"context{c}_img.nii.gz")

                ctx_gt = context_out[i, c, 0].cpu().numpy()
                ctx_gt_nib = nib.Nifti1Image(ctx_gt.astype(np.float32), affine=np.eye(4))
                nib.save(ctx_gt_nib, case_dir / f"context{c}_gt_mask.nii.gz")

        # Save context patches if available in outputs
        if outputs.get("context_patches") is not None:
            ctx_patches = outputs["context_patches"]  # [B, K*k, C, ps, ps]
            ctx_labels = outputs["context_patch_labels"]  # [B, K*k, 1, ps, ps]
            n_ctx_patches = min(4, ctx_patches.shape[1])
            for c in range(n_ctx_patches):
                ctx_p_img = ctx_patches[i, c, 0].cpu().numpy()
                ctx_p_img_nib = nib.Nifti1Image(ctx_p_img.astype(np.float32), affine=np.eye(4))
                nib.save(ctx_p_img_nib, case_dir / f"context_patch{c}_img.nii.gz")

                ctx_p_gt = ctx_labels[i, c, 0].cpu().numpy()
                ctx_p_gt_nib = nib.Nifti1Image(ctx_p_gt.astype(np.float32), affine=np.eye(4))
                nib.save(ctx_p_gt_nib, case_dir / f"context_patch{c}_gt_mask.nii.gz")

        # Save patch position visualization for each level
        level_outputs = outputs.get("level_outputs", [])
        for level_idx, level_out in enumerate(level_outputs):
            coords = level_out.get("coords")  # [B, K, 2]
            pred = level_out.get("pred")  # [B, 1, res, res]
            if coords is None or pred is None:
                continue

            level_res = pred.shape[-1]
            patch_size = level_out.get("patches", outputs["patches"]).shape[-1]

            # Create visualization: copy of image with bounding boxes
            # Scale coordinates from level resolution to full image resolution
            H, W = images.shape[2], images.shape[3]
            scale_h, scale_w = H / level_res, W / level_res

            # Start with the input image
            vis_img = images[i, 0].cpu().numpy().copy()

            # Draw bounding boxes by setting border pixels to max value
            box_value = vis_img.max() + 0.2 * (vis_img.max() - vis_img.min())
            coords_i = coords[i].cpu().numpy()  # [K, 2]

            for k in range(coords_i.shape[0]):
                # Coordinates are (row, col) at level resolution
                r, c = coords_i[k]
                # Scale to full resolution
                r_start = int(r * scale_h)
                c_start = int(c * scale_w)
                r_end = int(min(r_start + patch_size * scale_h, H))
                c_end = int(min(c_start + patch_size * scale_w, W))

                # Draw rectangle border (2 pixels thick)
                thickness = 2
                # Top and bottom edges
                vis_img[r_start:r_start+thickness, c_start:c_end] = box_value
                vis_img[max(0, r_end-thickness):r_end, c_start:c_end] = box_value
                # Left and right edges
                vis_img[r_start:r_end, c_start:c_start+thickness] = box_value
                vis_img[r_start:r_end, max(0, c_end-thickness):c_end] = box_value

            vis_nib = nib.Nifti1Image(vis_img.astype(np.float32), affine=np.eye(4))
            nib.save(vis_nib, case_dir / f"level{level_idx}_patch_positions_mask.nii.gz")


def _extract_patch_labels(labels: torch.Tensor, coords: torch.Tensor, patch_size: int, level_res: int) -> torch.Tensor:
    """Extract GT patch labels using coordinates from model output.

    Args:
        labels: [B, 1, H, W] - full resolution GT labels
        coords: [B, K, 2] - patch coordinates at level resolution
        patch_size: size of patches at level resolution
        level_res: resolution of the level

    Returns:
        patch_labels: [B, K, 1, ps, ps] - extracted GT patches
    """
    B, _, H, W = labels.shape
    K = coords.shape[1]
    scale = H / level_res

    # Scale coordinates to full resolution
    coords_scaled = (coords.float() * scale).long()
    ps_scaled = int(patch_size * scale)

    patch_labels = []
    for b in range(B):
        batch_patches = []
        for k in range(K):
            h, w = coords_scaled[b, k].tolist()
            h = min(h, H - ps_scaled)
            w = min(w, W - ps_scaled)
            patch = labels[b, :, h:h+ps_scaled, w:w+ps_scaled]
            # Resize to match model's patch size
            patch = torch.nn.functional.interpolate(
                patch.unsqueeze(0).float(), size=(patch_size, patch_size), mode='nearest'
            ).squeeze(0)
            batch_patches.append(patch)
        patch_labels.append(torch.stack(batch_patches))

    return torch.stack(patch_labels)  # [B, K, 1, ps, ps]


@torch.no_grad()
def validate(
    model,
    val_loader,
    device,
    save_dir: Optional[Path] = None,
    max_save_batches: int = 2,
    accelerator=None,
):
    """Run validation without oracle guidance (realistic inference).
    Uses model.aggreg_criterion for loss computation."""
    model.eval()
    is_main = accelerator is None or accelerator.is_main_process
    # Get unwrapped model for accessing custom attributes (aggreg_criterion, etc.)
    unwrapped_model = accelerator.unwrap_model(model) if accelerator is not None else model
    total_loss = 0.0
    total_local_dice = 0.0
    total_final_dice = 0.0
    total_context_dice = 0.0
    context_dice_count = 0

    for batch_idx, batch in enumerate(val_loader):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        # Get context data if available
        context_in = batch.get("context_in", None)
        context_out = batch.get("context_out", None)
        if context_in is not None:
            context_in = context_in.to(device)
        if context_out is not None:
            context_out = context_out.to(device)

        # Get pre-computed features if available
        target_features = batch.get("target_features", None)
        context_features = batch.get("context_features", None)
        if target_features is not None:
            target_features = target_features.to(device)
        if context_features is not None:
            context_features = context_features.to(device)

        # Ensure labels have channel dim
        if labels.dim() == 3:
            labels = labels.unsqueeze(1)

        # Pass labels to model - oracle behavior is controlled by model config
        # (oracle_levels_valid). If oracle=False, model uses prev_pred for sampling.
        # If oracle=True, model uses GT. This allows config to control behavior.
        outputs = model(
            images,
            labels=labels,
            context_in=context_in,
            context_out=context_out,
            target_features=target_features,
            context_features=context_features,
            mode="test",
        )
        predictions = outputs["final_logit"]

        loss = unwrapped_model.aggreg_criterion(predictions, labels.float())
        total_loss += loss.item()

        # Local dice: extract GT patches using coordinates from model
        patch_logits = outputs["patch_logits"]  # [B, K, 1, ps, ps]
        patch_coords = outputs["patch_coords"]  # [B, K, 2]
        patch_size = patch_logits.shape[-1]

        # Get level resolution from level_outputs if available
        level_outputs = outputs.get("level_outputs", [])
        if level_outputs:
            level_res = level_outputs[-1]["pred"].shape[-1]
        else:
            # Fallback: assume single level at coarse_pred resolution
            level_res = outputs["coarse_pred"].shape[-1]

        # Extract GT patch labels post-hoc
        patch_labels = _extract_patch_labels(labels, patch_coords, patch_size, level_res)
        patch_labels = patch_labels.to(device)

        patch_pred_binary = (torch.sigmoid(patch_logits) > 0.5).float()
        patch_labels_binary = (patch_labels > 0).float()
        # Compute dice per patch then average
        patch_intersection = (patch_pred_binary * patch_labels_binary).sum(dim=(2, 3, 4))
        patch_union = patch_pred_binary.sum(dim=(2, 3, 4)) + patch_labels_binary.sum(dim=(2, 3, 4))
        local_dice = (2 * patch_intersection + 1e-6) / (patch_union + 1e-6)
        total_local_dice += local_dice.mean().item()

        # Final dice: on full prediction
        pred_binary = (torch.sigmoid(predictions) > 0.5).float()
        labels_binary = (labels > 0).float()
        spatial_dims = tuple(range(2, pred_binary.dim()))
        intersection = (pred_binary * labels_binary).sum(dim=spatial_dims)
        union = pred_binary.sum(dim=spatial_dims) + labels_binary.sum(dim=spatial_dims)
        final_dice = (2 * intersection + 1e-6) / (union + 1e-6)
        total_final_dice += final_dice.mean().item()

        # Context dice: on aggregated context predictions vs context GT
        if level_outputs:
            finest_level = level_outputs[-1]
            context_pred = finest_level.get("context_pred")  # [B, k, 1, res, res]
            context_labels = finest_level.get("context_labels")  # [B, k, 1, res, res]
            if context_pred is not None and context_labels is not None:
                ctx_pred_binary = (torch.sigmoid(context_pred) > 0.5).float()
                ctx_labels_binary = (context_labels > 0).float()
                # Compute dice per context image then average
                ctx_intersection = (ctx_pred_binary * ctx_labels_binary).sum(dim=(2, 3, 4))
                ctx_union = ctx_pred_binary.sum(dim=(2, 3, 4)) + ctx_labels_binary.sum(dim=(2, 3, 4))
                ctx_dice = (2 * ctx_intersection + 1e-6) / (ctx_union + 1e-6)
                total_context_dice += ctx_dice.mean().item()
                context_dice_count += 1

        # Save outputs if requested (only on main process)
        if save_dir is not None and batch_idx < max_save_batches and is_main:
            case_ids = batch.get("case_id", None)
            save_predictions(save_dir, case_ids, images, labels, outputs,
                           context_in=context_in, context_out=context_out, batch_idx=batch_idx)

    if save_dir is not None and is_main:
        print(f"  Saved validation outputs to {save_dir}")

    n = len(val_loader)
    ctx_dice_final = total_context_dice / context_dice_count if context_dice_count > 0 else 0.0
    return total_loss / n, total_local_dice / n, total_final_dice / n, ctx_dice_final

