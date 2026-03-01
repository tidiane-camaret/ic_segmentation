"""Simplified training script for PatchICL with Hydra config."""

import sys
from pathlib import Path
from datetime import datetime
import hydra
import torch
from accelerate import Accelerator, DistributedDataParallelKwargs
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.losses import build_loss_fn
from src.train_utils import seed_everything, train_epoch, validate, wait_for_image_saves


def _get_image_size(cfg) -> tuple[int, int]:
    """Get image size as tuple, handling both scalar and list formats."""
    img_size = cfg.preprocessing.image_size
    if isinstance(img_size, (list, tuple)):
        return tuple(img_size[:2])
    return (img_size, img_size)


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    """Main training function."""
    # Initialize accelerator with optional mixed precision
    method = cfg.get("method", "patch_icl")
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=(method == "universeg"))
    mixed_precision = cfg.training.get(
        "mixed_precision", None
    )  # "fp16", "bf16", or None
    accelerator = Accelerator(
        kwargs_handlers=[ddp_kwargs],
        mixed_precision=mixed_precision,
    )
    device = accelerator.device

    seed_everything(cfg.training.seed)

    if accelerator.is_main_process:
        print(f"Using device: {device}, num_processes: {accelerator.num_processes}")

    # Wandb logging
    if cfg.logging.use_wandb and accelerator.is_main_process:
        import wandb

        wandb.init(
            project=cfg.logging.wandb_project,
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        run_name = wandb.run.name
    else:
        run_name = f"run_{accelerator.process_index}_{accelerator.num_processes}"
    # Create checkpoint dir
    date_str = datetime.today().strftime("%Y-%m-%d")
    save_dir = Path(cfg.paths.ckpts.save_dir) / f"{date_str}_{run_name}"
    if accelerator.is_main_process:
        save_dir.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()

    # Unified augmentation config (takes precedence if present)
    unified_aug_cfg = cfg.get("augmentation", {})
    use_unified_augmentation = unified_aug_cfg.get("enabled", False)
    augmentation_config = (
        OmegaConf.to_container(unified_aug_cfg, resolve=True)
        if use_unified_augmentation
        else None
    )

    # Legacy image augmentation config (only used if unified is not enabled)
    img_aug_cfg = cfg.get("image_augmentation", {})
    use_image_augmentation = img_aug_cfg.get("enabled", False) and not use_unified_augmentation
    augment_config = (
        OmegaConf.to_container(img_aug_cfg, resolve=True)
        if use_image_augmentation
        else None
    )

    # Dataloader
    dataset_type = cfg.get("dataset", "totalseg")
    feature_mode = cfg.get("feature_mode", "precomputed")

    if dataset_type == "medsegbench":
        from src.dataloaders.medsegbench_dataloader import (
            get_dataloader as get_medsegbench_dataloader,
        )

        msb_cfg = cfg.get("medsegbench", {})
        msb_datasets = msb_cfg.get("datasets", None)
        if msb_datasets is not None:
            msb_datasets = list(msb_datasets)

        # Support separate train/val datasets, falling back to shared 'datasets'
        msb_train_datasets = msb_cfg.get("train_datasets", None)
        msb_val_datasets = msb_cfg.get("val_datasets", None)
        if msb_train_datasets is not None:
            msb_train_datasets = list(msb_train_datasets)
        else:
            msb_train_datasets = msb_datasets
        if msb_val_datasets is not None:
            msb_val_datasets = list(msb_val_datasets)
        else:
            msb_val_datasets = msb_datasets

        max_samples = msb_cfg.get("max_samples_per_dataset", None)

        if accelerator.is_main_process:
            print(
                f"MedSegBench train datasets: {msb_train_datasets}, val datasets: {msb_val_datasets}"
            )

        train_loader = get_medsegbench_dataloader(
            data_root=cfg.paths.medsegbench,
            datasets=msb_train_datasets,
            split="train",
            context_size=cfg.context_size,
            batch_size=cfg.train_batch_size,
            image_size=_get_image_size(cfg),
            num_workers=cfg.training.get("num_workers", 4),
            shuffle=True,
            augment=use_image_augmentation,
            augment_config=augment_config,
            max_samples_per_dataset=max_samples,
        )
        val_loader = get_medsegbench_dataloader(
            data_root=cfg.paths.medsegbench,
            datasets=msb_val_datasets,
            split="val",
            context_size=cfg.context_size,
            batch_size=cfg.val_batch_size,
            image_size=_get_image_size(cfg),
            num_workers=cfg.training.get("num_workers", 4),
            shuffle=False,
            augment=False,
            max_samples_per_dataset=max_samples,
        )

    else:
        # TotalSeg2D dataloader
        from src.dataloaders.totalseg2d_dataloader_fast import (
            get_dataloader as get_totalseg2d_dataloader,
        )

        train_labels = (
            cfg.train_label_ids
            if isinstance(cfg.train_label_ids, str)
            else list(cfg.train_label_ids)
        )
        val_labels = (
            cfg.val_label_ids
            if isinstance(cfg.val_label_ids, str)
            else list(cfg.val_label_ids)
        )

        # Support list of splits for train/val
        train_split_cfg = cfg.get("train_split", "train")
        train_split = (
            list(train_split_cfg)
            if OmegaConf.is_list(train_split_cfg)
            else train_split_cfg
        )
        val_split_cfg = cfg.get("val_split", ["val", "test"])
        val_split = (
            list(val_split_cfg) if OmegaConf.is_list(val_split_cfg) else val_split_cfg
        )

        # Support separate max_ds_len for train/val, with fallback to single value
        max_ds_len_cfg = cfg.get("max_ds_len")
        if isinstance(max_ds_len_cfg, dict) or OmegaConf.is_dict(max_ds_len_cfg):
            max_ds_len_train = max_ds_len_cfg.get("train")
            max_ds_len_val = max_ds_len_cfg.get("val")
        else:
            max_ds_len_train = max_ds_len_cfg
            max_ds_len_val = max_ds_len_cfg

        # Support separate max_cases for train/val (limits unique cases, not samples)
        max_cases_cfg = cfg.get("max_cases")
        if isinstance(max_cases_cfg, dict) or OmegaConf.is_dict(max_cases_cfg):
            max_cases_train = max_cases_cfg.get("train")
            max_cases_val = max_cases_cfg.get("val")
        else:
            max_cases_train = max_cases_cfg
            max_cases_val = max_cases_cfg

        # CarveMix config (only for training)
        carve_mix_cfg = cfg.get("carve_mix", {})
        use_carve_mix = carve_mix_cfg.get("enabled", False)
        carve_mix_config = (
            OmegaConf.to_container(carve_mix_cfg, resolve=True)
            if use_carve_mix
            else None
        )

        # Legacy advanced augmentation config (only for training, if unified not used)
        adv_aug_cfg = cfg.get("advanced_augmentation", {})
        use_adv_aug = adv_aug_cfg.get("enabled", False) and not use_unified_augmentation
        adv_aug_config = (
            OmegaConf.to_container(adv_aug_cfg, resolve=True) if use_adv_aug else None
        )

        # Log augmentation mode
        if accelerator.is_main_process:
            if use_unified_augmentation:
                print(f"Using unified augmentation config")
            elif use_image_augmentation or use_carve_mix or use_adv_aug:
                print(f"Using legacy augmentation config (image_aug={use_image_augmentation}, "
                      f"carve_mix={use_carve_mix}, advanced={use_adv_aug})")
            else:
                print("Augmentation disabled")

        train_loader = get_totalseg2d_dataloader(
            root_dir=cfg.paths.dataset,
            stats_path=cfg.paths.dataset_stats,
            label_id_list=train_labels,
            context_size=cfg.context_size,
            batch_size=cfg.train_batch_size,
            image_size=_get_image_size(cfg),
            crop_to_bbox=cfg.preprocessing.crop_to_bbox,
            bbox_padding=cfg.preprocessing.bbox_padding,
            num_workers=cfg.training.get("num_workers", 4),
            split=train_split,
            shuffle=True,
            max_ds_len=max_ds_len_train,
            random_coloring_nb=cfg.get("random_coloring_nb", 0),
            # Legacy augmentation params
            augment=use_image_augmentation,
            augment_config=augment_config,
            carve_mix=use_carve_mix,
            carve_mix_config=carve_mix_config,
            advanced_augmentation=use_adv_aug,
            advanced_augmentation_config=adv_aug_config,
            # Unified augmentation config (takes precedence)
            augmentation_config=augmentation_config,
            max_labels=cfg.get("max_labels", None),
            max_cases=max_cases_train,
            class_balanced=cfg.get("class_balanced", False),
            slice_coverage_ratio=cfg.get("slice_coverage_ratio", 0.5),
        )
        val_loader = get_totalseg2d_dataloader(
            root_dir=cfg.paths.dataset,
            stats_path=cfg.paths.dataset_stats,
            label_id_list=val_labels,
            context_size=cfg.context_size,
            batch_size=cfg.val_batch_size,
            image_size=_get_image_size(cfg),
            crop_to_bbox=cfg.preprocessing.crop_to_bbox,
            bbox_padding=cfg.preprocessing.bbox_padding,
            num_workers=cfg.training.get("num_workers", 4),
            split=val_split,
            shuffle=False,
            max_ds_len=max_ds_len_val,
            random_coloring_nb=cfg.get("random_coloring_nb", 0),
            augment=False,  # No augmentation for validation
            max_labels=cfg.get("max_labels", None),
            max_cases=max_cases_val,
            slice_coverage_ratio=cfg.get("slice_coverage_ratio", 0.5),
        )

    # Model selection based on method config

    if method == "universeg":
        # UniverSeg baseline model
        from src.models.universeg_baseline import UniverSegBaseline

        universeg_cfg = cfg.model.get("universeg", {})
        model = UniverSegBaseline(
            pretrained=universeg_cfg.get("pretrained", True),
            input_size=universeg_cfg.get("input_size", 128),
            freeze=universeg_cfg.get("freeze", False),
        )

        if accelerator.is_main_process:
            print(f"Using UniverSeg model (input_size={universeg_cfg.get('input_size', 128)}, "
                  f"pretrained={universeg_cfg.get('pretrained', True)}, "
                  f"freeze={universeg_cfg.get('freeze', False)})")

        # UniverSeg doesn't use feature extractors
        feature_extractor = None
        patch_icl_cfg = {}  # Empty config for loss setup

    else:
        # Default: PatchICL v2
        from src.models.patch_icl_v2 import PatchICL

        patch_icl_cfg = OmegaConf.to_container(cfg.model.patch_icl, resolve=True)
        random_coloring_nb = cfg.get("random_coloring_nb", 0)
        patch_icl_cfg["num_mask_channels"] = 3 if random_coloring_nb > 0 else 1

        if accelerator.is_main_process:
            print(
                f"Mask channels: {patch_icl_cfg['num_mask_channels']} (random_coloring_nb={random_coloring_nb})"
            )

    # Feature extractor and PatchICL model creation (only for patch_icl method)
    if method != "universeg":
        feature_extractor = None
        if feature_mode == "on_the_fly":
            fe_cfg = patch_icl_cfg.get("feature_extractor", None)
            extractor_type = (
                fe_cfg.get("type", "meddino").lower()
                if fe_cfg
                else cfg.get("feature_extractor_type", "meddino").lower()
            )

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
                    print("Feature mode: on_the_fly (MedDINO)")

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
                    print(
                        f"Feature mode: on_the_fly (MedSAM v1 layer {info['layer_idx']}, grid={info['output_grid_size']})"
                    )
            elif extractor_type == "universeg":
                from src.models.universeg_extractor import UniverSegExtractor

                if accelerator.is_main_process:
                    print("Initializing UniverSeg for on-the-fly feature extraction...")
                feature_extractor = UniverSegExtractor(
                    layer_idx=fe_cfg.get("layer_idx", 3) if fe_cfg else 3,
                    device=device,
                    pretrained=fe_cfg.get("pretrained", True) if fe_cfg else True,
                    freeze=fe_cfg.get("freeze", True) if fe_cfg else True,
                    output_grid_size=fe_cfg.get("output_grid_size") if fe_cfg else None,
                    input_size=fe_cfg.get("input_size", 128) if fe_cfg else 128,
                    skip_preprocess=fe_cfg.get("skip_preprocess", True) if fe_cfg else True,
                )
                if accelerator.is_main_process:
                    info = feature_extractor.get_feature_info()
                    print(
                        f"Feature mode: on_the_fly (UniverSeg layers={info['layer_indices']}, "
                        f"dim={info['feature_dim']}, input={info['input_size']}x{info['input_size']}, "
                        f"grid={info['output_grid_size']}, skip_preprocess={info['skip_preprocess']})"
                    )
            elif extractor_type == "icl_encoder":
                from src.models.icl_encoder import ICLEncoder

                if accelerator.is_main_process:
                    print("Initializing ICLEncoder for on-the-fly feature extraction...")
                feature_extractor = ICLEncoder(
                    layer_idx=fe_cfg.get("layer_idx", "all") if fe_cfg else "all",
                    output_grid_size=fe_cfg.get("output_grid_size") if fe_cfg else None,
                    freeze=fe_cfg.get("freeze", False) if fe_cfg else False,
                )
                if accelerator.is_main_process:
                    info = feature_extractor.get_feature_info()
                    print(
                        f"Feature mode: on_the_fly (ICLEncoder layers={info['layer_indices']}, "
                        f"dim={info['feature_dim']}, grid={info['output_grid_size']})"
                    )
            elif extractor_type == "rad_dino":
                from src.models.rad_dino_extractor import RADDINOExtractor

                if accelerator.is_main_process:
                    print("Initializing RAD-DINO for on-the-fly feature extraction...")
                feature_extractor = RADDINOExtractor(
                    model_name=fe_cfg.get("model_name", "microsoft/rad-dino") if fe_cfg else "microsoft/rad-dino",
                    target_size=fe_cfg.get("target_size", 224) if fe_cfg else 224,
                    output_grid_size=fe_cfg.get("output_grid_size") if fe_cfg else None,
                    device=device,
                    freeze=fe_cfg.get("freeze", True) if fe_cfg else True,
                )
                if accelerator.is_main_process:
                    info = feature_extractor.get_feature_info()
                    print(
                        f"Feature mode: on_the_fly (RAD-DINO model={info['model_name']}, "
                        f"dim={info['feature_dim']}, grid={info['output_grid_size']}, frozen={info['frozen']})"
                    )
            else:
                raise ValueError(f"Unknown feature_extractor_type: {extractor_type}")
        else:
            if accelerator.is_main_process:
                print("Feature mode: precomputed")

        # Create PatchICL model
        model = PatchICL(
            patch_icl_cfg,
            context_size=cfg.get("context_size", 0),
            feature_extractor=feature_extractor,
        )

    # Loss functions
    if method == "universeg":
        # Simple loss config for UniverSeg
        loss_cfg = cfg.get("loss", {})
        patch_loss_cfg = loss_cfg.get("patch_loss", {"type": "dice", "args": None})
        aggreg_loss_cfg = loss_cfg.get("aggreg_loss", {"type": "dice", "args": None})
    else:
        loss_cfg = patch_icl_cfg.get("loss", {})
        patch_loss_cfg = loss_cfg.get("patch_loss", {"type": "dice", "args": None})
        aggreg_loss_cfg = loss_cfg.get("aggreg_loss", {"type": "dice", "args": None})

    patch_criterion = build_loss_fn(patch_loss_cfg["type"], patch_loss_cfg.get("args"))
    aggreg_criterion = build_loss_fn(
        aggreg_loss_cfg["type"], aggreg_loss_cfg.get("args")
    )
    model.set_loss_functions(patch_criterion, aggreg_criterion)

    # Optionally load model weights from checkpoint
    ckpt_path = cfg.get("checkpoint", None)
    reset_optimizer = cfg.get("reset_optimizer", False)  # Skip optimizer/scheduler state
    start_epoch = 0
    if ckpt_path:
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        ckpt_state_dict = checkpoint["model_state_dict"]
        model_state_dict = model.state_dict()

        # Check for feature extractor weights in checkpoint
        ckpt_fe_keys = [k for k in ckpt_state_dict.keys() if k.startswith("feature_extractor.")]
        model_fe_keys = [k for k in model_state_dict.keys() if k.startswith("feature_extractor.")]

        if accelerator.is_main_process:
            if ckpt_fe_keys:
                print(f"Checkpoint contains {len(ckpt_fe_keys)} feature_extractor keys")
            if model_fe_keys and not ckpt_fe_keys:
                print("WARNING: Model has feature_extractor but checkpoint does not - using fresh weights")

        # Load with strict=False but track what was loaded
        missing, unexpected = model.load_state_dict(ckpt_state_dict, strict=False)

        if not reset_optimizer:
            start_epoch = checkpoint.get("epoch", 0) + 1
        if accelerator.is_main_process:
            print(
                f"Loaded checkpoint from {ckpt_path} (epoch {checkpoint.get('epoch', '?')}, dice {checkpoint.get('best_dice', '?'):.4f})"
            )
            if missing:
                fe_missing = [k for k in missing if k.startswith("feature_extractor.")]
                other_missing = [k for k in missing if not k.startswith("feature_extractor.")]
                if fe_missing:
                    print(f"WARNING: {len(fe_missing)} feature_extractor keys missing from checkpoint")
                if other_missing:
                    print(f"Note: {len(other_missing)} other keys missing from checkpoint")
            if reset_optimizer:
                print("reset_optimizer=true: Starting fresh optimizer/scheduler (epoch 0)")
    else:
        if accelerator.is_main_process:
            print("No checkpoint loaded, training from scratch")
    if accelerator.is_main_process:
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model parameters: {num_params:,}")
        print(
            f"Loss functions: patch={patch_loss_cfg['type']}, aggreg={aggreg_loss_cfg['type']}"
        )

    # Optimizer
    opt_args = cfg.optimizer.optimizer_args
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=opt_args.lr, weight_decay=opt_args.weight_decay
    )

    # Load optimizer state from checkpoint (after optimizer is created)
    if ckpt_path and not reset_optimizer:
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        if "optimizer_state_dict" in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                if accelerator.is_main_process:
                    print("Loaded optimizer state from checkpoint")
            except ValueError as e:
                if accelerator.is_main_process:
                    print(f"Warning: Could not load optimizer state (architecture mismatch?): {e}")
                    print("Starting with fresh optimizer state")

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
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_epochs],
        )
    else:
        scheduler = main_scheduler

    # Prepare for distributed training
    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )

    # torch.compile encoder/attention/decoder after DDP wrapping (PatchICL only)
    if method != "universeg":
        backbone_cfg = patch_icl_cfg.get("backbone", {})
        if backbone_cfg.get("compile", False):
            if accelerator.is_main_process:
                print("Compiling backbone submodules with torch.compile...")
            bb = accelerator.unwrap_model(model).backbone
            bb.encoder = torch.compile(bb.encoder)
            bb.attention = torch.compile(bb.attention)
            bb.decoder = torch.compile(bb.decoder)


    # Load scheduler state after preparing for distributed training
    if ckpt_path and not reset_optimizer:
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        if "scheduler_state_dict" in checkpoint:
            try:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                if accelerator.is_main_process:
                    print(f"Loaded scheduler state from checkpoint")
            except Exception as e:
                if accelerator.is_main_process:
                    print(f"Warning: Could not load scheduler state: {e}")
                    print("Starting with fresh scheduler state")

    # Training loop
    best_dice = 0.0
    save_imgs = cfg.logging.get("save_imgs_masks", False)
    save_every_n_epochs = cfg.logging.get("save_every_n_epochs", 5)
    train_save_dir = save_dir / "train_images" if save_imgs else None
    val_save_dir = save_dir / "val_images" if save_imgs else None

    if save_imgs and accelerator.is_main_process:
        print(
            f"Saving images: train every {save_every_n_epochs} epochs to {train_save_dir}"
        )
        print(f"Saving images: val every epoch to {val_save_dir}")

    for epoch in tqdm(
        range(start_epoch, cfg.training.num_epochs),
        desc="Training",
        disable=not accelerator.is_main_process,
    ):
        train_losses = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            epoch,
            cfg.training.print_every,
            cfg.training.get("grad_accumulate_steps", 1),
            accelerator=accelerator,
            use_wandb=cfg.logging.use_wandb,
            log_every=cfg.training.get("log_every", 10),
            save_dir=train_save_dir,
            save_every_n_epochs=save_every_n_epochs,
            compute_metrics_every=cfg.training.get("compute_metrics_every", 10),
        )

        val_loss, val_local_dice, val_final_dice, val_context_dice, val_detailed = (
            validate(
                model,
                val_loader,
                device,
                accelerator=accelerator,
                use_wandb=cfg.logging.use_wandb,
                epoch=epoch,
                save_dir=val_save_dir,
            )
        )

        scheduler.step()

        if accelerator.is_main_process:
            print(
                f"Epoch {epoch:04d} | "
                f"Train Loss: {train_losses['total_loss']:.5f} | Train Dice: {train_losses['final_dice']:.5f} | "
                f"Val Loss: {val_loss:.5f} | Val Dice: {val_final_dice:.5f} | "
                f"LR: {scheduler.get_last_lr()[0]:.2e}"
            )

        if cfg.logging.use_wandb and accelerator.is_main_process:
            log_dict = {
                "epoch": epoch,
                "val_loss": val_loss,
                "val_local_dice": val_local_dice,
                "val_final_dice": val_final_dice,
                "val_context_dice": val_context_dice,
                "val_local_softdice": val_detailed.get("local_softdice", 0),
                "val_final_softdice": val_detailed.get("final_softdice", 0),
                "val_context_softdice": val_detailed.get("context_softdice", 0),
                "lr": scheduler.get_last_lr()[0],
            }
            # Log all train metrics dynamically
            for key, val in train_losses.items():
                if key == "per_label":
                    continue
                log_dict[f"train/{key}"] = val
            # Log per-label dice scores
            if "per_label" in train_losses:
                for label_id, dice_score in train_losses["per_label"].items():
                    log_dict[f"train_dice/{label_id}"] = dice_score
            if val_detailed and "per_label" in val_detailed:
                for label_id, dice_score in val_detailed["per_label"].items():
                    log_dict[f"val_dice/{label_id}"] = dice_score
            # Log per-level val metrics (level_N_dice, level_N_softdice, level_N_avg_probs_*)
            if val_detailed:
                skip_keys = {
                    "per_case",
                    "per_label",
                    "local_softdice",
                    "final_softdice",
                    "context_softdice",
                }
                for key, value in val_detailed.items():
                    if key not in skip_keys:
                        log_dict[f"val/{key}"] = value
                # Log mask size statistics from per_case results
                per_case = val_detailed.get("per_case", [])
                if per_case:
                    target_sizes = [c["target_mask_size"] for c in per_case if c.get("target_mask_size") is not None]
                    if target_sizes:
                        log_dict["val/target_mask_size_mean"] = sum(target_sizes) / len(target_sizes)
                        log_dict["val/target_mask_size_min"] = min(target_sizes)
                        log_dict["val/target_mask_size_max"] = max(target_sizes)
                    # Context mask sizes (average across all context images per sample)
                    ctx_sizes = []
                    for c in per_case:
                        if c.get("context_mask_sizes"):
                            ctx_sizes.extend(c["context_mask_sizes"])
                    if ctx_sizes:
                        log_dict["val/context_mask_size_mean"] = sum(ctx_sizes) / len(ctx_sizes)
                        log_dict["val/context_mask_size_min"] = min(ctx_sizes)
                        log_dict["val/context_mask_size_max"] = max(ctx_sizes)
                    # Log per-case table with mask sizes
                    case_table = wandb.Table(columns=[
                        "case_id", "label_id", "axis", "dice",
                        "target_mask_size", "context_mask_sizes"
                    ])
                    for c in per_case:
                        ctx_sizes_str = ",".join(map(str, c.get("context_mask_sizes") or []))
                        case_table.add_data(
                            c.get("case_id", ""),
                            c.get("label_id", ""),
                            c.get("axis", ""),
                            c.get("dice", 0),
                            c.get("target_mask_size", 0),
                            ctx_sizes_str,
                        )
                    log_dict["val/per_case"] = case_table
            wandb.log(log_dict)

        # Save best — sync all ranks before checkpoint to avoid collective mismatch
        accelerator.wait_for_everyone()
        if val_final_dice > best_dice:
            best_dice = val_final_dice
            if accelerator.is_main_process:
                print(f"  -> New best dice: {best_dice:.5f}")
                unwrapped_model = accelerator.unwrap_model(model)
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": unwrapped_model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "best_dice": best_dice,
                        "config": OmegaConf.to_container(cfg, resolve=True),
                    },
                    save_dir / "best_model.pt",
                )

    if accelerator.is_main_process:
        print(f"\nTraining complete! Best Dice: {best_dice:.5f}")

    if cfg.logging.use_wandb and accelerator.is_main_process:
        # Wait for background image saving threads to complete before closing wandb
        wait_for_image_saves()
        wandb.finish()


if __name__ == "__main__":
    main()
