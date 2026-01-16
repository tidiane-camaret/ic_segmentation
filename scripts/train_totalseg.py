"""
Train SegFormer3D or NMSW models on TotalSegmentator dataset.

Usage:
    python scripts/train_totalseg.py
    python scripts/train_totalseg.py --model-type nmsw_token
    python scripts/train_totalseg.py --no-wandb
"""

import argparse
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import yaml
from accelerate import Accelerator
from termcolor import colored
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, "/software/notebooks/camaret/repos/SegFormer3D")

# SegFormer3D imports
from architectures.segformer3d import build_segformer3d_model
from losses.losses import build_loss_fn
from optimizers.optimizers import build_optimizer
from optimizers.schedulers import build_scheduler
from src.config import load_config
from src.totalseg_dataloader import TotalSegmentatorDataset


class TotalSegDatasetWrapper(TotalSegmentatorDataset):
    """Wrapper to adapt TotalSegmentatorDataset output for SegFormer3D trainer.

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


def train_epoch(model, train_loader, criterion, optimizer, accelerator, epoch: int, print_every: int, model_type: str = "segformer3d"):
    """Run one training epoch."""
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

            if model_type in ["nmsw", "nmsw_token"]:
                # NMSW models have different forward/loss interface
                outputs = model(x=images, labels=labels, mode="train")
                losses = model.compute_loss(outputs, labels, criterion)
                loss = losses["total_loss"]
                
                epoch_global_loss += losses["global_loss"].detach().item()
                epoch_local_loss += losses["local_loss"].detach().item()
                epoch_agg_loss += losses["agg_loss"].detach().item()
            else:
                # Standard SegFormer3D
                predictions = model(images)
                loss = criterion(predictions, labels)

            accelerator.backward(loss)
            optimizer.step()

            epoch_loss += loss.detach().item()

            if print_every and idx % print_every == 0:
                if model_type in ["nmsw", "nmsw_token"]:
                    accelerator.print(
                        f"Epoch {epoch:04d} | Batch {idx:04d} | "
                        f"Loss: {epoch_loss / (idx + 1):.5f} | "
                        f"Global: {epoch_global_loss / (idx + 1):.5f} | "
                        f"Local: {epoch_local_loss / (idx + 1):.5f} | "
                        f"Agg: {epoch_agg_loss / (idx + 1):.5f}"
                    )
                else:
                    accelerator.print(
                        f"Epoch {epoch:04d} | Batch {idx:04d} | "
                        f"Loss: {epoch_loss / (idx + 1):.5f}"
                    )

    return epoch_loss / len(train_loader)


def validate(model, val_loader, criterion, accelerator, model_type: str = "segformer3d"):
    """Run validation."""
    model.eval()
    total_loss = 0.0
    total_dice = 0.0

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"]
            labels = batch["label"]

            if model_type in ["nmsw", "nmsw_token"]:
                # NMSW models return dict with final_logit
                outputs = model(x=images, labels=None, mode="test")
                predictions = outputs["final_logit"]
                loss = criterion(predictions, labels)
            else:
                # Standard SegFormer3D
                predictions = model(images)
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
    parser = argparse.ArgumentParser(description="Train SegFormer3D or NMSW models on TotalSegmentator")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument(
        "--model-type",
        type=str,
        default=None,
        choices=["segformer3d", "nmsw", "nmsw_token"],
        help="Model type: 'segformer3d', 'nmsw', or 'nmsw_token' (default: from config or segformer3d)"
    )
    args = parser.parse_args()

    # Load config
    config = load_config()
    train_cfg = config.get("train_totalseg", {})
    nmsw_cfg = train_cfg.get("nmsw", {})
    paths_cfg = config.get("paths", {})

    # Determine model type
    if args.model_type is not None:
        model_type = args.model_type
    elif nmsw_cfg.get("enabled", False):
        model_type = "nmsw_token" if nmsw_cfg.get("use_token_branch", False) else "nmsw"
    else:
        model_type = "segformer3d"

    # Merge with defaults
    full_config = {
        "project": train_cfg.get("wandb_project", "segformer3d_totalseg"),
        "model_type": model_type,
        "wandb_parameters": {
            "entity": train_cfg.get("wandb_entity"),
            "group": train_cfg.get("wandb_group", "totalseg"),
            "name": train_cfg.get("wandb_name", f"{model_type}_train"),
            "mode": "disabled" if args.no_wandb else "online",
        },
        "model_name": model_type,
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
                or str(Path(config.get("RESULTS_DIR", "results")) / "segformer3d_totalseg"),
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
        "sliding_window_inference": {"sw_batch_size": 4, "roi": [128, 128, 128]},
        "ema": {"enabled": False, "ema_decay": 0.999, "val_ema_every": 1},
        "nmsw_config": nmsw_cfg,
    }

    # Update dataset path from paths config if not set
    if full_config["dataset_parameters"]["dataset_path"] is None:
        full_config["dataset_parameters"]["dataset_path"] = paths_cfg.get("totalseg_dataset")

    # Update checkpoint dir to include model type
    if train_cfg.get("training_parameters", {}).get("checkpoint_save_dir") is None:
        full_config["training_parameters"]["checkpoint_save_dir"] = str(
            Path(paths_cfg.get("RESULTS_DIR", config.get("RESULTS_DIR", "results"))) / f"{model_type}_totalseg"
        )

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
    accelerator.print(f"Training {model_type} on TotalSegmentator")
    accelerator.print("=" * 60)

    # Build dataloaders
    train_loader, val_loader = build_dataloaders(full_config, accelerator)

    # Build model based on model_type
    if model_type == "nmsw_token":
        from src.nmsw_token_based import NMSWTokenSegFormer3D, TokenTauScheduler
        model = NMSWTokenSegFormer3D(
            config=full_config,
            nmsw_config=full_config["nmsw_config"],
        )
        tau_scheduler = TokenTauScheduler(
            model=model,
            starting_tau=full_config["nmsw_config"].get("starting_tau", 2/3),
            final_tau=full_config["nmsw_config"].get("final_tau", 2/3),
            decay_epochs=full_config["nmsw_config"].get("tau_decay_epochs", 100),
            total_epochs=full_config["training_parameters"]["num_epochs"],
        )
        accelerator.print(f"Using NMSW Token-based model with tau={tau_scheduler.get_tau():.4f}")
    elif model_type == "nmsw":
        from src.nmsw_segformer import NMSWSegFormer3D, TauScheduler
        model = NMSWSegFormer3D(
            config=full_config,
            nmsw_config=full_config["nmsw_config"],
        )
        tau_scheduler = TauScheduler(
            model=model,
            starting_tau=full_config["nmsw_config"].get("starting_tau", 2/3),
            final_tau=full_config["nmsw_config"].get("final_tau", 2/3),
            decay_epochs=full_config["nmsw_config"].get("tau_decay_epochs", 100),
            total_epochs=full_config["training_parameters"]["num_epochs"],
        )
        accelerator.print(f"Using NMSW SegFormer3D model with tau={tau_scheduler.get_tau():.4f}")
    else:
        model = build_segformer3d_model(full_config)
        tau_scheduler = None
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
        # Update tau for NMSW models
        if tau_scheduler is not None:
            tau_scheduler.step(epoch)
            current_tau = tau_scheduler.get_tau()
        else:
            current_tau = None

        # Select scheduler
        if full_config["warmup_scheduler"]["enabled"] and epoch < warmup_epochs:
            scheduler = warmup_scheduler
        else:
            scheduler = training_scheduler

        # Train
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer,
            accelerator, epoch, print_every, model_type=model_type
        )

        # Validate
        val_loss, val_dice = validate(model, val_loader, criterion, accelerator, model_type=model_type)

        # Step scheduler
        scheduler.step()

        # Log
        log_msg = (
            f"Epoch {epoch:04d} | "
            f"Train Loss: {train_loss:.5f} | "
            f"Val Loss: {val_loss:.5f} | "
            f"Val Dice: {val_dice:.5f} | "
            f"LR: {scheduler.get_last_lr()[0]:.2e}"
        )
        if current_tau is not None:
            log_msg += f" | Tau: {current_tau:.4f}"
        accelerator.print(log_msg)

        if not args.no_wandb:
            log_dict = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_dice": val_dice,
                "lr": scheduler.get_last_lr()[0],
            }
            if current_tau is not None:
                log_dict["tau"] = current_tau
            accelerator.log(log_dict)

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
