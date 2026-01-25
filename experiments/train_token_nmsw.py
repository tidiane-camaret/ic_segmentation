"""
Train Token-based NMSW on TotalSegmentator dataset.

Token-based NMSW treats selected patches as tokens that can attend
to each other via a transformer, enabling cross-patch communication.

Usage:
    python scripts/train_token_nmsw.py
    python scripts/train_token_nmsw.py --no-wandb
    python scripts/train_token_nmsw.py --pos-encoding relative
"""

import os
import sys
import random
import argparse
from pathlib import Path
from typing import Dict, List

import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, "/software/notebooks/camaret/repos/SegFormer3D")

from src.totalseg_dataloader import TotalSegmentatorDataset
from ic_segmentation.old.config import load_config
from src.token_nmsw import TokenNMSW

# SegFormer3D imports for loss
from losses.losses import build_loss_fn


class TotalSegDatasetWrapper(TotalSegmentatorDataset):
    """Wrapper to adapt output for TokenNMSW trainer."""

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.context_size == 0:
            target_case_id, sampled_label_id = self.case_label_id_pairs[idx]
            target_case_folder = self.root_dir / target_case_id
            target_data = self._load_single_case(target_case_folder, sampled_label_id)
            return {
                "image": target_data["image"],
                "label": target_data["label"],
                "case_id": target_data["case_id"],
                "label_id": sampled_label_id,
            }
        else:
            data = super().__getitem__(idx)
            return {
                "image": data["target_in"],
                "label": data["target_out"],
                "case_id": data["target_case_id"],
                "label_id": data["label_id"],
            }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate function for the dataloader."""
    images = torch.stack([item["image"] for item in batch], dim=0)
    labels = torch.stack([item["label"] for item in batch], dim=0)
    return {
        "image": images,
        "label": labels,
        "case_ids": [item["case_id"] for item in batch],
        "label_ids": [item["label_id"] for item in batch],
    }


def seed_everything(seed: int) -> None:
    """Set random seeds for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_dataloaders(config: Dict) -> tuple:
    """Build train and validation dataloaders."""
    ds_cfg = config["dataset_parameters"]

    train_dataset = TotalSegDatasetWrapper(
        root_dir=ds_cfg["dataset_path"],
        label_id_list=ds_cfg["train_label_ids"],
        context_size=0,
        image_size=tuple(ds_cfg["image_size"]),
        spacing=tuple(ds_cfg["spacing"]),
        split="train",
        random_context=False,
        max_ds_len=ds_cfg.get("max_ds_len"),
    )

    val_dataset = TotalSegDatasetWrapper(
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

        outputs = model(images, labels=None, mode="test")
        predictions = outputs["final_logit"]

        loss = criterion(predictions, labels)
        total_loss += loss.item()

        # Dice score
        pred_binary = (torch.sigmoid(predictions) > 0.5).float()
        intersection = (pred_binary * labels).sum(dim=(2, 3, 4))
        union = pred_binary.sum(dim=(2, 3, 4)) + labels.sum(dim=(2, 3, 4))
        dice = (2 * intersection + 1e-6) / (union + 1e-6)
        total_dice += dice.mean().item()

    return total_loss / len(val_loader), total_dice / len(val_loader)


def main():
    parser = argparse.ArgumentParser(description="Train Token-based NMSW")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb")
    parser.add_argument("--pos-encoding", type=str, default=None,
                        choices=["sinusoidal", "learnable", "relative"],
                        help="Override positional encoding type")
    parser.add_argument("--num-layers", type=int, default=None,
                        help="Override number of transformer layers")
    parser.add_argument("--num-patches", type=int, default=None,
                        help="Override number of patches to select")
    args = parser.parse_args()

    # Load config
    config = load_config()
    train_cfg = config.get("train_totalseg", {})
    token_cfg = train_cfg.get("token_nmsw", {})

    # Apply CLI overrides
    if args.pos_encoding:
        token_cfg["pos_encoding"] = args.pos_encoding
    if args.num_layers:
        token_cfg["num_layers"] = args.num_layers
    if args.num_patches:
        token_cfg["num_patches"] = args.num_patches

    # Build full config
    full_config = {
        "project": train_cfg.get("project", "patch_icl_train"),
        "model_parameters": {
            "in_channels": 1,
            "num_classes": 1,
            "patch_size": token_cfg.get("patch_size", 8),
            "num_patches": token_cfg.get("num_patches", 100),
            "num_random_patches": token_cfg.get("num_random_patches", 0),
            "global_base_channels": token_cfg.get("global_base_channels", 16),
            "down_size_rate": tuple(token_cfg.get("down_size_rate", [2, 2, 2])),
            "embed_dim": token_cfg.get("embed_dim", 256),
            "num_heads": token_cfg.get("num_heads", 8),
            "num_layers": token_cfg.get("num_layers", 8),
            "mlp_ratio": token_cfg.get("mlp_ratio", 4.0),
            "dropout": token_cfg.get("dropout", 0.1),
            "pos_encoding": token_cfg.get("pos_encoding", "learnable"),
            "tau": token_cfg.get("starting_tau", 2/3),
            "global_loss_weight": token_cfg.get("global_loss_weight", 1.0),
            "local_loss_weight": token_cfg.get("local_loss_weight", 1.0),
            "agg_loss_weight": token_cfg.get("agg_loss_weight", 1.0),
            "entropy_multiplier": token_cfg.get("entropy_multiplier", 1e-5),
        },
        "loss_fn": train_cfg.get("loss_fn", {"loss_type": "dice", "loss_args": None}),
        "optimizer": train_cfg.get("optimizer", {
            "optimizer_type": "adamw",
            "optimizer_args": {"lr": 1e-4, "weight_decay": 0.01},
        }),
        "training_parameters": {
            "seed": train_cfg.get("training_parameters", {}).get("seed", 42),
            "num_epochs": train_cfg.get("training_parameters", {}).get("num_epochs", 200),
            "print_every": train_cfg.get("training_parameters", {}).get("print_every", 50),
            "checkpoint_save_dir": train_cfg.get("training_parameters", {}).get("checkpoint_save_dir")
                or str(Path(config.get("RESULTS_DIR", "results")) / "token_nmsw"),
        },
        "dataset_parameters": {
            "dataset_path": train_cfg.get("dataset_path"),
            "train_label_ids": train_cfg.get("train_label_ids", ["heart"]),
            "val_label_ids": train_cfg.get("val_label_ids", ["heart"]),
            "image_size": train_cfg.get("image_size", [128, 128, 128]),
            "spacing": train_cfg.get("spacing", [1.5, 1.5, 1.5]),
            "train_batch_size": train_cfg.get("train_batch_size", 2),
            "val_batch_size": train_cfg.get("val_batch_size", 1),
            "num_workers": train_cfg.get("num_workers", 4),
            "max_ds_len": train_cfg.get("max_ds_len"),
        },
    }

    # Set seed
    seed_everything(full_config["training_parameters"]["seed"])

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create checkpoint dir
    ckpt_dir = Path(full_config["training_parameters"]["checkpoint_save_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Wandb
    if not args.no_wandb:
        import wandb
        wandb.init(
            project=full_config["project"],
            config=full_config,
            name=f"token_nmsw_{token_cfg.get('pos_encoding', 'learnable')}",
        )

    print("=" * 60)
    print("Training Token-based NMSW")
    print("=" * 60)
    print(f"Patch size: {full_config['model_parameters']['patch_size']}³")
    print(f"Num patches: {full_config['model_parameters']['num_patches']}")
    print(f"Transformer layers: {full_config['model_parameters']['num_layers']}")
    print(f"Positional encoding: {full_config['model_parameters']['pos_encoding']}")
    print("=" * 60)

    # Build dataloaders
    train_loader, val_loader = build_dataloaders(full_config)

    # Build model
    model = TokenNMSW(**full_config["model_parameters"]).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")

    # Loss
    criterion = build_loss_fn(
        full_config["loss_fn"]["loss_type"],
        full_config["loss_fn"].get("loss_args"),
    )

    # Optimizer
    opt_cfg = full_config["optimizer"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=opt_cfg["optimizer_args"]["lr"],
        weight_decay=opt_cfg["optimizer_args"]["weight_decay"],
    )

    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=full_config["training_parameters"]["num_epochs"],
        eta_min=1e-6,
    )

    # Training loop
    best_dice = 0.0
    num_epochs = full_config["training_parameters"]["num_epochs"]
    print_every = full_config["training_parameters"]["print_every"]

    for epoch in tqdm(range(num_epochs), desc="Training"):
        train_losses = train_epoch(
            model, train_loader, criterion, optimizer,
            device, epoch, print_every
        )

        val_loss, val_dice = validate(model, val_loader, criterion, device)

        scheduler.step()

        print(
            f"Epoch {epoch:04d} | "
            f"Train: {train_losses['loss']:.5f} | "
            f"Val Loss: {val_loss:.5f} | "
            f"Val Dice: {val_dice:.5f} | "
            f"LR: {scheduler.get_last_lr()[0]:.2e}"
        )

        if not args.no_wandb:
            wandb.log({
                "epoch": epoch,
                "train_loss": train_losses["loss"],
                "train_global_loss": train_losses["global_loss"],
                "train_local_loss": train_losses["local_loss"],
                "train_agg_loss": train_losses["agg_loss"],
                "val_loss": val_loss,
                "val_dice": val_dice,
                "lr": scheduler.get_last_lr()[0],
            })

        # Save best
        if val_dice > best_dice:
            best_dice = val_dice
            print(f"  -> New best dice: {best_dice:.5f}")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_dice": best_dice,
                "config": full_config,
            }, ckpt_dir / "best_model.pt")

    print(f"\nTraining complete! Best Dice: {best_dice:.5f}")

    if not args.no_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
