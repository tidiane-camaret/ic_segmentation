import os
import random
from pathlib import Path
from typing import Dict, Optional
import numpy as np
import nibabel as nib
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


def train_epoch(model, train_loader, criterion, optimizer, device, epoch, print_every):
    """Run one training epoch."""
    model.train()
    total_loss = 0.0
    total_global = 0.0
    total_local = 0.0
    total_agg = 0.0
    total_dice = 0.0

    for idx, batch in enumerate(train_loader):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        # Ensure labels have channel dim
        if labels.dim() == 3:
            labels = labels.unsqueeze(1)

        optimizer.zero_grad()

        outputs = model(images, labels=labels, mode="train")
        losses = model.compute_loss(outputs, labels, criterion)
        loss = losses["total_loss"]

        # Compute Dice score
        with torch.no_grad():
            pred_binary = (torch.sigmoid(outputs["final_logit"]) > 0.5).float()
            labels_binary = (labels > 0).float()
            spatial_dims = tuple(range(2, pred_binary.dim()))
            intersection = (pred_binary * labels_binary).sum(dim=spatial_dims)
            union = pred_binary.sum(dim=spatial_dims) + labels_binary.sum(dim=spatial_dims)
            dice = (2 * intersection + 1e-6) / (union + 1e-6)
            total_dice += dice.mean().item()

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_global += losses["global_loss"].item()
        total_local += losses["local_loss"].item()
        total_agg += losses["agg_loss"].item()

        if print_every and idx % print_every == 0:
            print(
                f"Epoch {epoch:04d} | Batch {idx:04d} | "
                f"Loss: {total_loss / (idx + 1):.5f} | "
                f"Dice: {total_dice / (idx + 1):.5f} | "
                f"Global: {total_global / (idx + 1):.5f} | "
                f"Local: {total_local / (idx + 1):.5f} | "
                f"Agg: {total_agg / (idx + 1):.5f}"
            )

    n = len(train_loader)
    return {
        "loss": total_loss / n,
        "dice": total_dice / n,
        "global_loss": total_global / n,
        "local_loss": total_local / n,
        "agg_loss": total_agg / n,
    }

def save_predictions(save_dir: Path, case_ids: list, images, labels, outputs, max_samples=4,
                     context_in=None, context_out=None):
    """Save images, masks, and predictions to NIfTI files organized by case ID."""
    save_dir = Path(save_dir)
    B = min(images.shape[0], max_samples)

    for i in range(B):
        case_id = case_ids[i] if case_ids else f"s{i:04d}"
        case_dir = save_dir / case_id
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


@torch.no_grad()
def validate(
    model,
    val_loader,
    criterion,
    device,
    save_dir: Optional[Path] = None,
    max_save_batches: int = 2,
):
    """Run validation with optional saving of predictions."""
    model.eval()
    total_loss = 0.0
    total_dice = 0.0

    for batch_idx, batch in enumerate(val_loader):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        # Ensure labels have channel dim
        if labels.dim() == 3:
            labels = labels.unsqueeze(1)

        # Pass labels for oracle global branch (for now - later replace with learned global)
        outputs = model(images, labels=labels, mode="test")
        predictions = outputs["final_logit"]

        loss = criterion(predictions, labels.float())
        total_loss += loss.item()

        # Dice score (works for both 2D and 3D)
        # Binarize both predictions and labels (labels may be multi-class)
        pred_binary = (torch.sigmoid(predictions) > 0.5).float()
        labels_binary = (labels > 0).float()
        spatial_dims = tuple(range(2, pred_binary.dim()))
        intersection = (pred_binary * labels_binary).sum(dim=spatial_dims)
        union = pred_binary.sum(dim=spatial_dims) + labels_binary.sum(dim=spatial_dims)
        dice = (2 * intersection + 1e-6) / (union + 1e-6)
        total_dice += dice.mean().item()

        # Save outputs if requested
        if save_dir is not None and batch_idx < max_save_batches:
            case_ids = batch.get("case_id", None)
            context_in = batch.get("context_in", None)
            context_out = batch.get("context_out", None)
            if context_in is not None:
                context_in = context_in.to(device)
            if context_out is not None:
                context_out = context_out.to(device)
            save_predictions(save_dir, case_ids, images, labels, outputs,
                           context_in=context_in, context_out=context_out)

    if save_dir is not None:
        print(f"  Saved validation outputs to {save_dir}")

    return total_loss / len(val_loader), total_dice / len(val_loader)

