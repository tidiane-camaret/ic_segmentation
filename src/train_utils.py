import os
import random
from typing import Dict
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


def train_epoch(model, train_loader, criterion, optimizer, device, epoch, print_every):
    """Run one training epoch."""
    model.train()
    total_loss = 0.0
    total_global = 0.0
    total_local = 0.0
    total_agg = 0.0

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
                f"Global: {total_global / (idx + 1):.5f} | "
                f"Local: {total_local / (idx + 1):.5f} | "
                f"Agg: {total_agg / (idx + 1):.5f}"
            )

    n = len(train_loader)
    return {
        "loss": total_loss / n,
        "global_loss": total_global / n,
        "local_loss": total_local / n,
        "agg_loss": total_agg / n,
    }

@torch.no_grad()
def validate(model, val_loader, criterion, device):
    """Run validation."""
    model.eval()
    total_loss = 0.0
    total_dice = 0.0

    for batch in val_loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        # Ensure labels have channel dim
        if labels.dim() == 3:
            labels = labels.unsqueeze(1)

        outputs = model(images, labels=None, mode="test")
        predictions = outputs["final_logit"]

        loss = criterion(predictions, labels.float())
        total_loss += loss.item()

        # Dice score (works for both 2D and 3D)
        pred_binary = (torch.sigmoid(predictions) > 0.5).float()
        spatial_dims = tuple(range(2, pred_binary.dim()))  # (2, 3) for 2D, (2, 3, 4) for 3D
        intersection = (pred_binary * labels).sum(dim=spatial_dims)
        union = pred_binary.sum(dim=spatial_dims) + labels.sum(dim=spatial_dims)
        dice = (2 * intersection + 1e-6) / (union + 1e-6)
        total_dice += dice.mean().item()

    return total_loss / len(val_loader), total_dice / len(val_loader)

