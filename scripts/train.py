""" training script for image segmentation models """
import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, "/software/notebooks/camaret/repos")

from SegFormer3D.architectures.segformer3d import build_segformer3d_model
from SegFormer3D.losses.losses import build_loss_fn
from src.config import load_config
from src.train_utils import build_dataloaders, seed_everything, train_epoch, validate

# get config from config.yaml 
config = load_config()
paths = config["paths"]
train_config = config["train"]

# set seed
seed_everything(train_config["training_parameters"]["seed"])

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create checkpoint dir
ckpt_dir = Path(paths["ckpts"]["save_dir"])
ckpt_dir.mkdir(parents=True, exist_ok=True)

### get dataset class
if train_config["dataset"] == "totalseg_no_context":
    from src.dataloaders.totalseg_dataloader_no_context import (
        TotalSegDatasetNoContext,
        collate_fn,
    )
    DatasetClass = TotalSegDatasetNoContext
elif train_config["dataset"] == "totalseg":
    from src.dataloaders.totalseg_dataloader import (
        TotalSegmentatorDataset as DatasetClass,
    )
    collate_fn = None
elif train_config["dataset"] == "totalseg2d":
    from src.dataloaders.totalseg2d_dataloader import (
        get_dataloader as get_totalseg2d_dataloader,
    )
elif train_config["dataset"].split("_")[0] == "medsegbench":
    from src.dataloaders.medsegbench_dataloader import get_dataloader
else:
    raise ValueError(f"Unknown dataset: {train_config['dataset']}")


# logging
if train_config["logging"]["use_wandb"]:
    import wandb
    wandb.init(
        project=train_config["logging"].get("wandb_project"),
        config=train_config,
        name=train_config["method"]+"+"+train_config["dataset"],
    )

# Build dataloaders
"""
train_loader, val_loader = build_dataloaders(
    train_config, DatasetClass, collate_fn
)
"""
if train_config["dataset"] == "totalseg2d":
    train_loader = get_totalseg2d_dataloader(
        root_dir=config["paths"]["totalseg2d"],
        stats_path=config["paths"]["totalseg_stats"],
        label_id_list=train_config["train_label_ids"],
        context_size=train_config["context_size"],
        batch_size=train_config["train_batch_size"],
        image_size=tuple(train_config["preprocessing"]["image_size"][:2]),
        crop_to_bbox=train_config["preprocessing"]["crop_to_bbox"],
        bbox_padding=train_config["preprocessing"]["bbox_padding"],
        num_workers=train_config["training_parameters"].get("num_workers", 4),
        split="train",
        shuffle=True,
        load_dinov3_features=train_config.get("load_dinov3_features", True),
    )
    val_loader = get_totalseg2d_dataloader(
        root_dir=config["paths"]["totalseg2d"],
        stats_path=config["paths"]["totalseg_stats"],
        label_id_list=train_config["val_label_ids"],
        context_size=train_config["context_size"],
        batch_size=train_config["val_batch_size"],
        image_size=tuple(train_config["preprocessing"]["image_size"][:2]),
        crop_to_bbox=train_config["preprocessing"]["crop_to_bbox"],
        bbox_padding=train_config["preprocessing"]["bbox_padding"],
        num_workers=train_config["training_parameters"].get("num_workers", 4),
        split="val",
        shuffle=False,
        load_dinov3_features=train_config.get("load_dinov3_features", True),
    )
else:
    train_loader = get_dataloader(
        dataset_name=train_config["dataset"],
        split="train",
        root=config["paths"]["medsegbench"],
        image_size=train_config["preprocessing"]["image_size"][0],
        download=train_config["download"],
        label_ids=train_config["train_label_ids"],
        context_size=train_config["context_size"],
        load_dinov3_features=train_config.get("load_dinov3_features", True),
    )
    val_loader = get_dataloader(
        dataset_name=train_config["dataset"],
        split="val",
        root=config["paths"]["medsegbench"],
        image_size=train_config["preprocessing"]["image_size"][0],
        download=train_config["download"],
        label_ids=train_config["val_label_ids"],
        context_size=train_config["context_size"],
        load_dinov3_features=train_config.get("load_dinov3_features", True),
    )

# get model 
if train_config["method"] == "segformer3d":
    sys.path.insert(0, "/software/notebooks/camaret/repos/SegFormer3D")
    model = build_segformer3d_model(config["model_params"]["segformer3d"]).to(device)
elif train_config["method"] == "global_local":
    from src.models.global_local import GlobalLocalModel
    model = GlobalLocalModel(
        config["model_params"]["global_local"],
        context_size=train_config.get("context_size", 0),
    ).to(device)
elif train_config["method"] == "patch_icl":
    from src.models.patch_icl import PatchICL
    model = PatchICL(
        config["model_params"]["patch_icl"],
        context_size=train_config.get("context_size", 0),
    ).to(device)
else:
    raise ValueError(f"Unknown method: {train_config['method']}")

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model parameters: {num_params:,}")

# Loss
criterion = build_loss_fn(
    train_config["loss_fn"]["loss_type"],
    train_config["loss_fn"].get("loss_args"),
)

# Optimizer
opt_cfg = train_config["optimizer"]
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=opt_cfg["optimizer_args"]["lr"],
    weight_decay=opt_cfg["optimizer_args"]["weight_decay"],
)

# Schedulers
warmup_cfg = train_config.get("warmup_scheduler", {})
sched_cfg = train_config.get("train_scheduler", {})

# Main scheduler (after warmup)
sched_type = sched_cfg.get("scheduler_type", "cosine_annealing")
sched_args = sched_cfg.get("scheduler_args", {})

if sched_type == "cosine_annealing_wr":
    main_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=sched_args.get("t_0_epochs", 100),
        T_mult=sched_args.get("t_mult", 1),
        eta_min=sched_args.get("min_lr", 1e-6),
    )
else:
    main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=train_config["training_parameters"]["num_epochs"],
        eta_min=sched_args.get("min_lr", 1e-6),
    )

# Warmup scheduler (linear warmup)
warmup_enabled = warmup_cfg.get("enabled", False)
warmup_epochs = warmup_cfg.get("warmup_epochs", 10) if warmup_enabled else 0

if warmup_enabled:
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.01,
        end_factor=1.0,
        total_iters=warmup_epochs,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[warmup_epochs],
    )
else:
    scheduler = main_scheduler

# Training loop
best_dice = 0.0
num_epochs = train_config["training_parameters"]["num_epochs"]
print_every = train_config["training_parameters"]["print_every"]

for epoch in tqdm(range(num_epochs), desc="Training"):
    train_losses = train_epoch(
        model, train_loader, criterion, optimizer,
        device, epoch, print_every
    )

    # Validation with optional saving (overwrites each time)
    save_imgs = train_config["logging"].get("save_imgs_masks", False)
    save_dir = None
    if save_imgs and epoch % 10 == 0:  # Save every 10 epochs
        save_dir = Path(paths["RESULTS_DIR"]) / f"{train_config['dataset']}"

    val_loss, val_local_dice, val_final_dice, val_context_dice = validate(
        model, val_loader, criterion, device,
        save_dir=save_dir, max_save_batches=2
    )

    scheduler.step()

    print(
        f"Epoch {epoch:04d} | "
        f"Train Loss: {train_losses['loss']:.5f} | "
        f"Train LocalDice: {train_losses['local_dice']:.5f} | "
        f"Train FinalDice: {train_losses['final_dice']:.5f} | "
        f"Val Loss: {val_loss:.5f} | "
        f"Val LocalDice: {val_local_dice:.5f} | "
        f"Val FinalDice: {val_final_dice:.5f} | "
        f"Val CtxDice: {val_context_dice:.5f} | "
        f"LR: {scheduler.get_last_lr()[0]:.2e}"
    )
    print(
        f"  Losses -> "
        f"TargetPatch: {train_losses.get('target_patch_loss', 0):.4f} | "
        f"TargetGlobal: {train_losses.get('target_global_loss', 0):.4f} | "
        f"ContextPatch: {train_losses.get('context_patch_loss', 0):.4f} | "
        f"ContextGlobal: {train_losses.get('context_global_loss', 0):.4f}"
    )

    if train_config["logging"]["use_wandb"]:
        log_dict = {
            "epoch": epoch,
            "train_loss": train_losses["loss"],
            "train_local_dice": train_losses["local_dice"],
            "train_final_dice": train_losses["final_dice"],
            "train_context_dice": train_losses.get("context_dice", 0),
            # Legacy losses
            "train_global_loss": train_losses["global_loss"],
            "train_local_loss": train_losses["local_loss"],
            "train_agg_loss": train_losses["agg_loss"],
            # Patch vs Global losses
            "train_target_patch_loss": train_losses.get("target_patch_loss", 0),
            "train_target_global_loss": train_losses.get("target_global_loss", 0),
            "train_context_patch_loss": train_losses.get("context_patch_loss", 0),
            "train_context_global_loss": train_losses.get("context_global_loss", 0),
            # Target vs Context losses
            "train_target_loss": train_losses.get("target_loss", 0),
            "train_context_loss": train_losses.get("context_loss", 0),
            # Combined totals
            "train_patch_loss_total": train_losses.get("patch_loss_total", 0),
            "train_global_loss_total": train_losses.get("global_loss_total", 0),
            # Validation
            "val_loss": val_loss,
            "val_local_dice": val_local_dice,
            "val_final_dice": val_final_dice,
            "val_context_dice": val_context_dice,
            "lr": scheduler.get_last_lr()[0],
        }
        # Add per-level losses if available
        for i in range(10):  # Support up to 10 levels
            if f"level_{i}_target_patch_loss" in train_losses:
                log_dict[f"train_level_{i}_target_patch_loss"] = train_losses[f"level_{i}_target_patch_loss"]
                log_dict[f"train_level_{i}_target_global_loss"] = train_losses.get(f"level_{i}_target_global_loss", 0)
                log_dict[f"train_level_{i}_context_patch_loss"] = train_losses.get(f"level_{i}_context_patch_loss", 0)
                log_dict[f"train_level_{i}_context_global_loss"] = train_losses.get(f"level_{i}_context_global_loss", 0)
        wandb.log(log_dict)

    # Save best (based on final dice)
    if val_final_dice > best_dice:
        best_dice = val_final_dice
        print(f"  -> New best dice: {best_dice:.5f}")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_dice": best_dice,
            "config": config,
        }, ckpt_dir / "best_model.pt")

print(f"\nTraining complete! Best Dice: {best_dice:.5f}")

if train_config["logging"]["use_wandb"]:
    wandb.finish()
