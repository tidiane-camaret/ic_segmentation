"""Simplified training script for PatchICL with Hydra config."""
import sys
from pathlib import Path

import hydra
import torch
from accelerate import Accelerator, DistributedDataParallelKwargs
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.losses import build_loss_fn
from src.train_utils import seed_everything, train_epoch, validate


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    """Main training function."""
    # Initialize accelerator
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    device = accelerator.device

    seed_everything(cfg.training.seed)

    if accelerator.is_main_process:
        print(f"Using device: {device}, num_processes: {accelerator.num_processes}")

    # Create checkpoint dir
    ckpt_dir = Path(cfg.paths.ckpts.save_dir)
    if accelerator.is_main_process:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()

    # Wandb logging
    if cfg.logging.use_wandb and accelerator.is_main_process:
        import wandb
        wandb.init(project=cfg.logging.wandb_project, config=OmegaConf.to_container(cfg, resolve=True))

    # Dataloader (TotalSeg2D only)
    from src.dataloaders.totalseg2d_dataloader import get_dataloader as get_totalseg2d_dataloader

    feature_mode = cfg.get("feature_mode", "precomputed")
    load_precomputed_features = (feature_mode == "precomputed") and cfg.get("load_dinov3_features", True)

    train_labels = cfg.train_label_ids if isinstance(cfg.train_label_ids, str) else list(cfg.train_label_ids)
    val_labels = cfg.val_label_ids if isinstance(cfg.val_label_ids, str) else list(cfg.val_label_ids)

    train_loader = get_totalseg2d_dataloader(
        root_dir=cfg.paths.totalseg2d,
        stats_path=cfg.paths.totalseg_stats,
        label_id_list=train_labels,
        context_size=cfg.context_size,
        batch_size=cfg.train_batch_size,
        image_size=tuple(cfg.preprocessing.image_size[:2]),
        crop_to_bbox=cfg.preprocessing.crop_to_bbox,
        bbox_padding=cfg.preprocessing.bbox_padding,
        num_workers=cfg.training.get("num_workers", 4),
        split="train",
        shuffle=True,
        load_dinov3_features=load_precomputed_features,
        max_ds_len=cfg.get("max_ds_len"),
        random_coloring_nb=cfg.get("random_coloring_nb", 0),
    )
    val_loader = get_totalseg2d_dataloader(
        root_dir=cfg.paths.totalseg2d,
        stats_path=cfg.paths.totalseg_stats,
        label_id_list=val_labels,
        context_size=cfg.context_size,
        batch_size=cfg.val_batch_size,
        image_size=tuple(cfg.preprocessing.image_size[:2]),
        crop_to_bbox=cfg.preprocessing.crop_to_bbox,
        bbox_padding=cfg.preprocessing.bbox_padding,
        num_workers=cfg.training.get("num_workers", 4),
        split="val",
        shuffle=False,
        load_dinov3_features=load_precomputed_features,
        max_ds_len=cfg.get("max_ds_len"),
        random_coloring_nb=cfg.get("random_coloring_nb", 0),
    )

    # Model (PatchICL v2)
    from src.models.patch_icl_v2 import PatchICL

    patch_icl_cfg = OmegaConf.to_container(cfg.model.patch_icl, resolve=True)
    random_coloring_nb = cfg.get("random_coloring_nb", 0)
    patch_icl_cfg["num_mask_channels"] = 3 if random_coloring_nb > 0 else 1

    if accelerator.is_main_process:
        print(f"Mask channels: {patch_icl_cfg['num_mask_channels']} (random_coloring_nb={random_coloring_nb})")

    # Feature extractor for on-the-fly mode
    feature_extractor = None
    if feature_mode == "on_the_fly":
        fe_cfg = patch_icl_cfg.get("feature_extractor", None)
        extractor_type = fe_cfg.get("type", "meddino").lower() if fe_cfg else cfg.get("feature_extractor_type", "meddino").lower()

        if extractor_type in ["meddino", "meddinov3", "meddino_v3"]:
            from src.models.meddino_extractor import create_meddino_extractor
            if accelerator.is_main_process:
                print("Initializing MedDINOv3 for on-the-fly feature extraction...")
            if fe_cfg and fe_cfg.get("type") in ["meddino", "meddinov3", "meddino_v3"]:
                feature_extractor = create_meddino_extractor(
                    model_path=fe_cfg.get("model_path", cfg.paths.ckpts.meddino_vit),
                    target_size=fe_cfg.get("target_size", 256),
                    device=device,
                    layer_idx=fe_cfg.get("layer_idx", 11),
                    freeze=fe_cfg.get("freeze", True),
                )
            else:
                feature_extractor = create_meddino_extractor(
                    model_path=cfg.paths.ckpts.meddino_vit,
                    target_size=cfg.get("feature_extraction_resolution", 256),
                    device=device,
                    layer_idx=cfg.get("meddino_layer_idx", 11),
                    freeze=True,
                )
            if accelerator.is_main_process:
                print(f"Feature mode: on_the_fly (MedDINO)")

        elif extractor_type in ["medsam_v1", "medsam_v1_layer"]:
            from src.models.medsam_extractor import MedSAMv1LayerExtractor
            if accelerator.is_main_process:
                print("Initializing MedSAM v1 for on-the-fly feature extraction...")
            target_size = fe_cfg.get("target_size", 1024) if fe_cfg else 1024
            output_grid = fe_cfg.get("output_grid_size") if fe_cfg else None
            feature_extractor = MedSAMv1LayerExtractor(
                checkpoint_path=fe_cfg.get("checkpoint_path") if fe_cfg else None,
                target_size=target_size,
                device=device,
                layer_idx=fe_cfg.get("layer_idx", 11) if fe_cfg else 11,
                freeze=fe_cfg.get("freeze", True) if fe_cfg else True,
                output_grid_size=output_grid,
            )
            if accelerator.is_main_process:
                info = feature_extractor.get_feature_info()
                print(f"Feature mode: on_the_fly (MedSAM v1 layer {info['layer_idx']}, grid={info['output_grid_size']})")
        else:
            raise ValueError(f"Unknown feature_extractor_type: {extractor_type}")
    else:
        if accelerator.is_main_process:
            print("Feature mode: precomputed")

    model = PatchICL(patch_icl_cfg, context_size=cfg.get("context_size", 0), feature_extractor=feature_extractor)

    # Loss functions
    loss_cfg = patch_icl_cfg.get("loss", {})
    patch_loss_cfg = loss_cfg.get("patch_loss", {"type": "dice", "args": None})
    aggreg_loss_cfg = loss_cfg.get("aggreg_loss", {"type": "dice", "args": None})
    patch_criterion = build_loss_fn(patch_loss_cfg["type"], patch_loss_cfg.get("args"))
    aggreg_criterion = build_loss_fn(aggreg_loss_cfg["type"], aggreg_loss_cfg.get("args"))
    model.set_loss_functions(patch_criterion, aggreg_criterion)

    if accelerator.is_main_process:
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model parameters: {num_params:,}")
        print(f"Loss functions: patch={patch_loss_cfg['type']}, aggreg={aggreg_loss_cfg['type']}")

    # Optimizer
    opt_args = cfg.optimizer.optimizer_args
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt_args.lr, weight_decay=opt_args.weight_decay)

    # Scheduler
    warmup_cfg = cfg.get("warmup_scheduler", {})
    warmup_enabled = warmup_cfg.get("enabled", False)
    warmup_epochs = warmup_cfg.get("warmup_epochs", 10) if warmup_enabled else 0

    sched_cfg = cfg.get("train_scheduler", {})
    sched_args = sched_cfg.get("scheduler_args", {})
    main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.training.num_epochs, eta_min=sched_args.get("min_lr", 1e-6)
    )

    if warmup_enabled:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs)
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[warmup_epochs])
    else:
        scheduler = main_scheduler

    # Prepare for distributed training
    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(model, optimizer, train_loader, val_loader, scheduler)

    # Training loop
    best_dice = 0.0
    for epoch in tqdm(range(cfg.training.num_epochs), desc="Training", disable=not accelerator.is_main_process):
        train_losses = train_epoch(
            model, train_loader, optimizer, device, epoch, cfg.training.print_every,
            cfg.training.get("grad_accumulate_steps", 1),
            accelerator=accelerator, use_wandb=cfg.logging.use_wandb, log_every=cfg.training.get("log_every", 10)
        )

        val_loss, val_local_dice, val_final_dice, val_context_dice, _ = validate(
            model, val_loader, device, accelerator=accelerator
        )

        scheduler.step()

        if accelerator.is_main_process:
            print(
                f"Epoch {epoch:04d} | "
                f"Train Loss: {train_losses['loss']:.5f} | Train Dice: {train_losses['final_dice']:.5f} | "
                f"Val Loss: {val_loss:.5f} | Val Dice: {val_final_dice:.5f} | "
                f"LR: {scheduler.get_last_lr()[0]:.2e}"
            )

        if cfg.logging.use_wandb and accelerator.is_main_process:
            wandb.log({
                "epoch": epoch,
                "train_loss": train_losses["loss"],
                "train_local_dice": train_losses["local_dice"],
                "train_final_dice": train_losses["final_dice"],
                "train_context_dice": train_losses.get("context_dice", 0),
                "train_target_patch_loss": train_losses.get("target_patch_loss", 0),
                "train_target_aggreg_loss": train_losses.get("target_aggreg_loss", 0),
                "train_context_patch_loss": train_losses.get("context_patch_loss", 0),
                "train_context_aggreg_loss": train_losses.get("context_aggreg_loss", 0),
                "val_loss": val_loss,
                "val_local_dice": val_local_dice,
                "val_final_dice": val_final_dice,
                "val_context_dice": val_context_dice,
                "lr": scheduler.get_last_lr()[0],
            })

        # Save best
        if val_final_dice > best_dice:
            best_dice = val_final_dice
            if accelerator.is_main_process:
                print(f"  -> New best dice: {best_dice:.5f}")
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                unwrapped_model = accelerator.unwrap_model(model)
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": unwrapped_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_dice": best_dice,
                    "config": OmegaConf.to_container(cfg, resolve=True),
                }, ckpt_dir / "best_model.pt")

    if accelerator.is_main_process:
        print(f"\nTraining complete! Best Dice: {best_dice:.5f}")

    if cfg.logging.use_wandb and accelerator.is_main_process:
        wandb.finish()


if __name__ == "__main__":
    main()
