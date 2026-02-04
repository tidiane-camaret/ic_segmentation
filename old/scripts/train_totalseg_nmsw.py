"""
Train SegFormer3D with NMSW patch sampling on TotalSegmentator dataset.

NMSW (No More Sliding Window) replaces costly sliding window with
differentiable top-k patch selection based on learned objectness scores.

Usage:
    python scripts/train_totalseg_nmsw.py
    python scripts/train_totalseg_nmsw.py --no-wandb
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
from accelerate import Accelerator
from tqdm import tqdm

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, "/software/notebooks/camaret/repos/SegFormer3D")

from src.totalseg_dataloader import TotalSegmentatorDataset
from ic_segmentation.old.config import load_config
from src.nmsw_segformer import NMSWSegFormer3D, TauScheduler

# SegFormer3D imports
from losses.losses import build_loss_fn
from optimizers.optimizers import build_optimizer
from optimizers.schedulers import build_scheduler


class TotalSegDatasetWrapper(TotalSegmentatorDataset):
    """Wrapper to adapt TotalSegmentatorDataset output for NMSW trainer.

    When context_size=0, bypasses context loading and only loads target image/label.
    """

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.context_size == 0:
            # Bypass context loading - directly load target only
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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_dataloaders(config: Dict, accelerator) -> tuple:
    """Build train and validation dataloaders."""
    ds_cfg = config["dataset_parameters"]

    # Build train dataset
    train_dataset = TotalSegDatasetWrapper(
        root_dir=ds_cfg["dataset_path"],
        label_id_list=ds_cfg["train_label_ids"],
        context_size=0,  # No context for SegFormer3D
        image_size=tuple(ds_cfg["image_size"]),
        spacing=tuple(ds_cfg["spacing"]),
        split="train",
        random_context=False,
        max_ds_len=ds_cfg.get("max_ds_len"),
    )

    # Build val dataset
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

    accelerator.print(f"Train dataset: {len(train_dataset)} samples")
    accelerator.print(f"Val dataset: {len(val_dataset)} samples")

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


def train_epoch(
    model,
    train_loader,
    criterion,
    optimizer,
    accelerator,
    epoch: int,
    print_every: int,
):
    """Run one training epoch with NMSW."""
    model.train()
    epoch_loss = 0.0
    epoch_global_loss = 0.0
    epoch_local_loss = 0.0
    epoch_agg_loss = 0.0

    for idx, batch in enumerate(train_loader):
        with accelerator.accumulate(model):
            images = batch["image"]
            labels = batch["label"]

            optimizer.zero_grad(set_to_none=True)

            # Forward pass with labels for patch extraction
            outputs = model(images, labels=labels, mode="train")

            # Compute NMSW loss
            losses = model.compute_loss(outputs, labels, criterion)
            loss = losses["total_loss"]

            accelerator.backward(loss)
            optimizer.step()

            epoch_loss += loss.detach().item()
            epoch_global_loss += losses["global_loss"].detach().item()
            epoch_local_loss += losses["local_loss"].detach().item()
            epoch_agg_loss += losses["agg_loss"].detach().item()

            if print_every and idx % print_every == 0:
                accelerator.print(
                    f"Epoch {epoch:04d} | Batch {idx:04d} | "
                    f"Loss: {epoch_loss / (idx + 1):.5f} | "
                    f"Global: {epoch_global_loss / (idx + 1):.5f} | "
                    f"Local: {epoch_local_loss / (idx + 1):.5f} | "
                    f"Agg: {epoch_agg_loss / (idx + 1):.5f} | "
                    f"Tau: {model.current_tau:.4f}"
                )

    n = len(train_loader)
    return {
        "loss": epoch_loss / n,
        "global_loss": epoch_global_loss / n,
        "local_loss": epoch_local_loss / n,
        "agg_loss": epoch_agg_loss / n,
    }


def validate(model, val_loader, criterion, accelerator):
    """Run validation with NMSW."""
    model.eval()
    total_loss = 0.0
    total_dice = 0.0

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"]
            labels = batch["label"]

            # Forward pass (inference mode)
            outputs = model(images, labels=None, mode="test")
            predictions = outputs["final_logit"]

            # Compute aggregation loss only for validation
            loss = criterion(predictions, labels)
            total_loss += loss.item()

            # Compute dice score
            pred_binary = (torch.sigmoid(predictions) > 0.5).float()
            intersection = (pred_binary * labels).sum(dim=(2, 3, 4))
            union = pred_binary.sum(dim=(2, 3, 4)) + labels.sum(dim=(2, 3, 4))
            dice = (2 * intersection + 1e-6) / (union + 1e-6)
            total_dice += dice.mean().item()

    avg_loss = total_loss / len(val_loader)
    avg_dice = total_dice / len(val_loader)
    return avg_loss, avg_dice


def main():
    parser = argparse.ArgumentParser(description="Train SegFormer3D with NMSW on TotalSegmentator")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    args = parser.parse_args()

    # Load config
    config = load_config()
    train_cfg = config.get("train_totalseg", {})
    nmsw_cfg = train_cfg.get("nmsw", {})

    # Merge with defaults
    full_config = {
        "project": train_cfg.get("project", "patch_icl_train"),
        "wandb_parameters": {
            "entity": train_cfg.get("wandb_entity"),
            "group": train_cfg.get("wandb_group", "nmsw"),
            "name": train_cfg.get("wandb_name", "segformer3d_nmsw_train"),
            "mode": "disabled" if args.no_wandb else "online",
        },
        "model_name": "nmsw_segformer3d",
        "model_parameters": train_cfg.get("model_parameters", {
            "in_channels": 1,
            "sr_ratios": [4, 2, 1, 1],
            "embed_dims": [32, 64, 160, 256],
            "patch_kernel_size": [7, 3, 3, 3],
            "patch_stride": [4, 2, 2, 2],
            "patch_padding": [3, 1, 1, 1],
            "mlp_ratios": [4, 4, 4, 4],
            "num_heads": [1, 2, 5, 8],
            "depths": [2, 2, 2, 2],
            "num_classes": 1,
            "decoder_dropout": 0.0,
            "decoder_head_embedding_dim": 256,
        }),
        "nmsw_parameters": {
            "down_size_rate": nmsw_cfg.get("down_size_rate", [2, 2, 2]),
            "patch_size": nmsw_cfg.get("patch_size", [128, 128, 128]),
            "overlap": nmsw_cfg.get("overlap", 0.5),
            "num_train_patches": nmsw_cfg.get("num_train_patches", 4),
            "num_train_random_patches": nmsw_cfg.get("num_train_random_patches", 1),
            "num_inference_patches": nmsw_cfg.get("num_inference_patches", 8),
            "starting_tau": nmsw_cfg.get("starting_tau", 2/3),
            "final_tau": nmsw_cfg.get("final_tau", 2/3),
            "tau_decay_epochs": nmsw_cfg.get("tau_decay_epochs", 100),
            "global_model_scale": nmsw_cfg.get("global_model_scale", 0.5),
            "add_aggregation_module": nmsw_cfg.get("add_aggregation_module", False),
            "global_loss_weight": nmsw_cfg.get("global_loss_weight", 1.0),
            "local_loss_weight": nmsw_cfg.get("local_loss_weight", 1.0),
            "agg_loss_weight": nmsw_cfg.get("agg_loss_weight", 1.0),
            "entropy_multiplier": nmsw_cfg.get("entropy_multiplier", 1e-5),
        },
        "loss_fn": train_cfg.get("loss_fn", {"loss_type": "dice", "loss_args": None}),
        "optimizer": train_cfg.get("optimizer", {
            "optimizer_type": "adamw",
            "optimizer_args": {"lr": 1e-4, "weight_decay": 0.01},
        }),
        "warmup_scheduler": train_cfg.get("warmup_scheduler", {
            "enabled": True,
            "warmup_epochs": 10,
        }),
        "train_scheduler": train_cfg.get("train_scheduler", {
            "scheduler_type": "cosine_annealing_wr",
            "scheduler_args": {"t_0_epochs": 100, "t_mult": 1, "min_lr": 1e-6},
        }),
        "training_parameters": {
            "seed": train_cfg.get("training_parameters", {}).get("seed", 42),
            "num_epochs": train_cfg.get("training_parameters", {}).get("num_epochs", 200),
            "print_every": train_cfg.get("training_parameters", {}).get("print_every", 50),
            "grad_accumulate_steps": train_cfg.get("training_parameters", {}).get("grad_accumulate_steps", 1),
            "checkpoint_save_dir": train_cfg.get("training_parameters", {}).get("checkpoint_save_dir")
                or str(Path(config.get("RESULTS_DIR", "results")) / "segformer3d_nmsw"),
        },
        "dataset_parameters": {
            "dataset_path": train_cfg.get("dataset_path", config.get("eval_totalseg", {}).get("dataset_path")),
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

    # Create checkpoint dir
    ckpt_dir = Path(full_config["training_parameters"]["checkpoint_save_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Initialize accelerator
    accelerator = Accelerator(
        log_with="wandb" if not args.no_wandb else None,
        gradient_accumulation_steps=full_config["training_parameters"]["grad_accumulate_steps"],
    )

    if not args.no_wandb:
        accelerator.init_trackers(
            project_name=full_config["project"],
            config=full_config,
            init_kwargs={"wandb": full_config["wandb_parameters"]},
        )

    accelerator.print("=" * 60)
    accelerator.print("Training SegFormer3D with NMSW on TotalSegmentator")
    accelerator.print("=" * 60)

    # Build dataloaders
    train_loader, val_loader = build_dataloaders(full_config, accelerator)

    # Build model
    model = NMSWSegFormer3D(
        config=full_config,
        nmsw_config=full_config["nmsw_parameters"],
    )
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    accelerator.print(f"Model parameters: {num_params:,}")

    # Build loss, optimizer, schedulers
    criterion = build_loss_fn(
        full_config["loss_fn"]["loss_type"],
        full_config["loss_fn"].get("loss_args"),
    )

    optimizer = build_optimizer(
        model,
        full_config["optimizer"]["optimizer_type"],
        full_config["optimizer"]["optimizer_args"],
    )

    warmup_scheduler = build_scheduler(optimizer, "warmup_scheduler", full_config)
    training_scheduler = build_scheduler(optimizer, "training_scheduler", full_config)

    # Tau scheduler for Gumbel-softmax temperature
    tau_scheduler = TauScheduler(
        model=model,
        starting_tau=full_config["nmsw_parameters"]["starting_tau"],
        final_tau=full_config["nmsw_parameters"]["final_tau"],
        decay_epochs=full_config["nmsw_parameters"]["tau_decay_epochs"],
        total_epochs=full_config["training_parameters"]["num_epochs"],
    )

    # Prepare with accelerator
    model = accelerator.prepare_model(model)
    optimizer = accelerator.prepare_optimizer(optimizer)
    train_loader = accelerator.prepare_data_loader(train_loader)
    val_loader = accelerator.prepare_data_loader(val_loader)
    warmup_scheduler = accelerator.prepare_scheduler(warmup_scheduler)
    training_scheduler = accelerator.prepare_scheduler(training_scheduler)

    # Training loop
    best_dice = 0.0
    warmup_epochs = full_config["warmup_scheduler"]["warmup_epochs"]
    num_epochs = full_config["training_parameters"]["num_epochs"]
    print_every = full_config["training_parameters"]["print_every"]

    for epoch in tqdm(range(num_epochs), desc="Training"):
        # Select LR scheduler
        if full_config["warmup_scheduler"]["enabled"] and epoch < warmup_epochs:
            scheduler = warmup_scheduler
        else:
            scheduler = training_scheduler

        # Update tau
        tau_scheduler.step(epoch)

        # Train
        train_losses = train_epoch(
            model, train_loader, criterion, optimizer,
            accelerator, epoch, print_every
        )

        # Validate
        val_loss, val_dice = validate(model, val_loader, criterion, accelerator)

        # Step scheduler
        scheduler.step()

        # Log
        accelerator.print(
            f"Epoch {epoch:04d} | "
            f"Train Loss: {train_losses['loss']:.5f} | "
            f"Val Loss: {val_loss:.5f} | "
            f"Val Dice: {val_dice:.5f} | "
            f"LR: {scheduler.get_last_lr()[0]:.2e} | "
            f"Tau: {tau_scheduler.get_tau():.4f}"
        )

        if not args.no_wandb:
            accelerator.log({
                "epoch": epoch,
                "train_loss": train_losses["loss"],
                "train_global_loss": train_losses["global_loss"],
                "train_local_loss": train_losses["local_loss"],
                "train_agg_loss": train_losses["agg_loss"],
                "val_loss": val_loss,
                "val_dice": val_dice,
                "lr": scheduler.get_last_lr()[0],
                "tau": tau_scheduler.get_tau(),
            })

        # Save best model
        if val_dice > best_dice:
            best_dice = val_dice
            accelerator.print(f"  -> New best dice: {best_dice:.5f}, saving checkpoint")
            accelerator.save_state(str(ckpt_dir / "best_checkpoint"), safe_serialization=False)

    accelerator.print(f"\nTraining complete! Best Dice: {best_dice:.5f}")

    if not args.no_wandb:
        accelerator.end_training()


if __name__ == "__main__":
    main()
