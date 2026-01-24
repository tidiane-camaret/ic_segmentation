"""Training script for image segmentation models with Hydra config management."""
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from tqdm import tqdm

sys.path.insert(0, "/software/notebooks/camaret/repos")
sys.path.insert(0, "/work/dlclarge2/ndirt-SegFM3D/repos")
#from SegFormer3D.architectures.segformer3d import build_segformer3d_model
sys.path.insert(0, "/work/dlclarge2/ndirt-SegFM3D/ic_segmentation")
from SegFormer3D.losses.losses import build_loss_fn
from src.train_utils import seed_everything, train_epoch, validate


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    """Main training function."""
    # Print config
    print(OmegaConf.to_yaml(cfg))

    # Set seed
    seed_everything(cfg.training.seed)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create checkpoint dir
    ckpt_dir = Path(cfg.paths.ckpts.save_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Get dataset class and dataloader
    if cfg.dataset == "totalseg_no_context":
        from src.dataloaders.totalseg_dataloader_no_context import (
            TotalSegDatasetNoContext,
            collate_fn,
        )
        DatasetClass = TotalSegDatasetNoContext
    elif cfg.dataset == "totalseg":
        from src.dataloaders.totalseg_dataloader import (
            TotalSegmentatorDataset as DatasetClass,
        )
        collate_fn = None
    elif cfg.dataset == "totalseg2d":
        from src.dataloaders.totalseg2d_dataloader import (
            get_dataloader as get_totalseg2d_dataloader,
        )
    elif cfg.dataset.split("_")[0] == "medsegbench":
        from src.dataloaders.medsegbench_dataloader import get_dataloader
    else:
        raise ValueError(f"Unknown dataset: {cfg.dataset}")

    # Logging
    if cfg.logging.use_wandb:
        import wandb
        wandb.init(
            project=cfg.logging.wandb_project,
            config=OmegaConf.to_container(cfg, resolve=True),  # wandb needs plain dict
            name=f"{cfg.method}+{cfg.dataset}",
        )

    # Build dataloaders
    if cfg.dataset == "totalseg2d":
        train_loader = get_totalseg2d_dataloader(
            root_dir=cfg.paths.totalseg2d,
            stats_path=cfg.paths.totalseg_stats,
            label_id_list=list(cfg.train_label_ids),
            context_size=cfg.context_size,
            batch_size=cfg.train_batch_size,
            image_size=tuple(cfg.preprocessing.image_size[:2]),
            crop_to_bbox=cfg.preprocessing.crop_to_bbox,
            bbox_padding=cfg.preprocessing.bbox_padding,
            num_workers=cfg.training.get("num_workers", 4),
            split="train",
            shuffle=True,
            load_dinov3_features=cfg.get("load_dinov3_features", True),
            max_ds_len=cfg.get("max_ds_len"),
        )
        val_loader = get_totalseg2d_dataloader(
            root_dir=cfg.paths.totalseg2d,
            stats_path=cfg.paths.totalseg_stats,
            label_id_list=list(cfg.val_label_ids),
            context_size=cfg.context_size,
            batch_size=cfg.val_batch_size,
            image_size=tuple(cfg.preprocessing.image_size[:2]),
            crop_to_bbox=cfg.preprocessing.crop_to_bbox,
            bbox_padding=cfg.preprocessing.bbox_padding,
            num_workers=cfg.training.get("num_workers", 4),
            split="val",
            shuffle=False,
            load_dinov3_features=cfg.get("load_dinov3_features", True),
            max_ds_len=cfg.get("max_ds_len"),
        )
    else:
        train_loader = get_dataloader(
            dataset_name=cfg.dataset,
            split="train",
            root=cfg.paths.medsegbench,
            image_size=cfg.preprocessing.image_size[0],
            download=cfg.download,
            label_ids=list(cfg.train_label_ids),
            context_size=cfg.context_size,
            load_dinov3_features=cfg.get("load_dinov3_features", True),
        )
        val_loader = get_dataloader(
            dataset_name=cfg.dataset,
            split="val",
            root=cfg.paths.medsegbench,
            image_size=cfg.preprocessing.image_size[0],
            download=cfg.download,
            label_ids=list(cfg.val_label_ids),
            context_size=cfg.context_size,
            load_dinov3_features=cfg.get("load_dinov3_features", True),
        )

    # Get model
    if cfg.method == "segformer3d":
        sys.path.insert(0, "/software/notebooks/camaret/repos/SegFormer3D")
        model = build_segformer3d_model(cfg.model.segformer3d).to(device)
    elif cfg.method == "global_local":
        from src.models.global_local import GlobalLocalModel
        model = GlobalLocalModel(
            cfg.model.global_local,
            context_size=cfg.get("context_size", 0),
        ).to(device)
    elif cfg.method == "patch_icl":
        from src.models.patch_icl import PatchICL
        model = PatchICL(
            cfg.model.patch_icl,
            context_size=cfg.get("context_size", 0),
        ).to(device)

        # Build and set loss functions from patch_icl config
        loss_cfg = cfg.model.patch_icl.get("loss", {})
        patch_loss_cfg = loss_cfg.get("patch_loss", {"type": "dice", "args": None})
        aggreg_loss_cfg = loss_cfg.get("aggreg_loss", {"type": "dice", "args": None})

        patch_criterion = build_loss_fn(patch_loss_cfg["type"], patch_loss_cfg.get("args"))
        aggreg_criterion = build_loss_fn(aggreg_loss_cfg["type"], aggreg_loss_cfg.get("args"))
        model.set_loss_functions(patch_criterion, aggreg_criterion)
        print(f"Loss functions: patch={patch_loss_cfg['type']}, aggreg={aggreg_loss_cfg['type']}")
    else:
        raise ValueError(f"Unknown method: {cfg.method}")

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")

    # Optimizer
    opt_type = cfg.optimizer.get("optimizer_type", "adamw").lower()
    opt_args = cfg.optimizer.optimizer_args
    if opt_type == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=opt_args.lr,
            weight_decay=opt_args.weight_decay,
        )
    elif opt_type == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=opt_args.lr,
            weight_decay=opt_args.weight_decay,
        )
    elif opt_type == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=opt_args.lr,
            weight_decay=opt_args.weight_decay,
            momentum=opt_args.get("momentum", 0.9),
        )
    else:
        raise ValueError(f"Unknown optimizer type: {opt_type}")

    # Schedulers
    warmup_cfg = cfg.get("warmup_scheduler", {})
    sched_cfg = cfg.get("train_scheduler", {})

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
            T_max=cfg.training.num_epochs,
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
    num_epochs = cfg.training.num_epochs
    print_every = cfg.training.print_every
    grad_accumulate_steps = cfg.training.get("grad_accumulate_steps", 1)

    for epoch in tqdm(range(num_epochs), desc="Training"):
        train_losses = train_epoch(
            model, train_loader, optimizer,
            device, epoch, print_every, grad_accumulate_steps
        )

        # Validation with optional saving
        save_imgs = cfg.logging.get("save_imgs_masks", False)
        val_save_dir = None
        if save_imgs and epoch % 10 == 0:
            val_save_dir = Path(cfg.paths.RESULTS_DIR) / f"{cfg.dataset}"

        val_loss, val_local_dice, val_final_dice, val_context_dice = validate(
            model, val_loader, device,
            save_dir=val_save_dir, max_save_batches=2
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
            f"TargetAggreg: {train_losses.get('target_aggreg_loss', 0):.4f} | "
            f"ContextPatch: {train_losses.get('context_patch_loss', 0):.4f} | "
            f"ContextAggreg: {train_losses.get('context_aggreg_loss', 0):.4f} | "
            f"FeatPatch: {train_losses.get('target_feature_patch_loss', 0):.4f} | "
            f"FeatAggreg: {train_losses.get('target_feature_aggreg_loss', 0):.4f} | "
            f"CtxFeatPatch: {train_losses.get('context_feature_patch_loss', 0):.4f} | "
            f"CtxFeatAggreg: {train_losses.get('context_feature_aggreg_loss', 0):.4f}"
        )

        if cfg.logging.use_wandb:
            log_dict = {
                "epoch": epoch,
                "train_loss": train_losses["loss"],
                "train_local_dice": train_losses["local_dice"],
                "train_final_dice": train_losses["final_dice"],
                "train_context_dice": train_losses.get("context_dice", 0),
                "train_aggreg_loss": train_losses.get("aggreg_loss", 0),
                "train_local_loss": train_losses["local_loss"],
                "train_agg_loss": train_losses["agg_loss"],
                "train_target_patch_loss": train_losses.get("target_patch_loss", 0),
                "train_target_aggreg_loss": train_losses.get("target_aggreg_loss", 0),
                "train_context_patch_loss": train_losses.get("context_patch_loss", 0),
                "train_context_aggreg_loss": train_losses.get("context_aggreg_loss", 0),
                "train_target_loss": train_losses.get("target_loss", 0),
                "train_context_loss": train_losses.get("context_loss", 0),
                "train_patch_loss_total": train_losses.get("patch_loss_total", 0),
                "train_aggreg_loss_total": train_losses.get("aggreg_loss_total", 0),
                "train_target_feature_patch_loss": train_losses.get("target_feature_patch_loss", 0),
                "train_target_feature_aggreg_loss": train_losses.get("target_feature_aggreg_loss", 0),
                "train_context_feature_patch_loss": train_losses.get("context_feature_patch_loss", 0),
                "train_context_feature_aggreg_loss": train_losses.get("context_feature_aggreg_loss", 0),
                "val_loss": val_loss,
                "val_local_dice": val_local_dice,
                "val_final_dice": val_final_dice,
                "val_context_dice": val_context_dice,
                "lr": scheduler.get_last_lr()[0],
            }
            # Add per-level losses if available
            for i in range(10):
                if f"level_{i}_target_patch_loss" in train_losses:
                    log_dict[f"train_level_{i}_target_patch_loss"] = train_losses[f"level_{i}_target_patch_loss"]
                    log_dict[f"train_level_{i}_target_aggreg_loss"] = train_losses.get(f"level_{i}_target_aggreg_loss", 0)
                    log_dict[f"train_level_{i}_context_patch_loss"] = train_losses.get(f"level_{i}_context_patch_loss", 0)
                    log_dict[f"train_level_{i}_context_aggreg_loss"] = train_losses.get(f"level_{i}_context_aggreg_loss", 0)
                    log_dict[f"train_level_{i}_feature_patch_loss"] = train_losses.get(f"level_{i}_target_feature_patch_loss", 0)
                    log_dict[f"train_level_{i}_feature_aggreg_loss"] = train_losses.get(f"level_{i}_target_feature_aggreg_loss", 0)
                    log_dict[f"train_level_{i}_context_feature_patch_loss"] = train_losses.get(f"level_{i}_context_feature_patch_loss", 0)
                    log_dict[f"train_level_{i}_context_feature_aggreg_loss"] = train_losses.get(f"level_{i}_context_feature_aggreg_loss", 0)
            wandb.log(log_dict)

        # Save best
        if val_final_dice > best_dice:
            best_dice = val_final_dice
            print(f"  -> New best dice: {best_dice:.5f}")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_dice": best_dice,
                "config": OmegaConf.to_container(cfg, resolve=True),  # checkpoint needs plain dict
            }, ckpt_dir / "best_model.pt")

    print(f"\nTraining complete! Best Dice: {best_dice:.5f}")

    if cfg.logging.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
