""" training script for image segmentation models """
import sys
from pathlib import Path
from turtle import down

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
    )
    val_loader = get_totalseg2d_dataloader(
        root_dir=config["paths"]["totalseg2d"],
        stats_path=config["paths"]["totalseg_stats"],
        label_id_list=train_config["val_label_ids"],
        context_size=train_config["context_size"],
        batch_size=train_config["train_batch_size"],
        image_size=tuple(train_config["preprocessing"]["image_size"][:2]),
        crop_to_bbox=train_config["preprocessing"]["crop_to_bbox"],
        bbox_padding=train_config["preprocessing"]["bbox_padding"],
        num_workers=train_config["training_parameters"].get("num_workers", 4),
        split="val",
        shuffle=False,
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
    )
    val_loader = get_dataloader(
        dataset_name=train_config["dataset"],
        split="val",
        root=config["paths"]["medsegbench"],
        image_size=train_config["preprocessing"]["image_size"][0],
        download=train_config["download"],
        label_ids=train_config["val_label_ids"],
        context_size=train_config["context_size"],
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

# Scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=train_config["training_parameters"]["num_epochs"],
    eta_min=1e-6,
)

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

    val_loss, val_local_dice, val_final_dice = validate(
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
        f"LR: {scheduler.get_last_lr()[0]:.2e}"
    )

    if train_config["logging"]["use_wandb"]:
        wandb.log({
            "epoch": epoch,
            "train_loss": train_losses["loss"],
            "train_local_dice": train_losses["local_dice"],
            "train_final_dice": train_losses["final_dice"],
            "train_global_loss": train_losses["global_loss"],
            "train_local_loss": train_losses["local_loss"],
            "train_agg_loss": train_losses["agg_loss"],
            "val_loss": val_loss,
            "val_local_dice": val_local_dice,
            "val_final_dice": val_final_dice,
            "lr": scheduler.get_last_lr()[0],
        })

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
