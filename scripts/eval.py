"""
eval script
for each case in val set:
- save img/gt/pred logits/probits as npz
- save attention maps and register tokens 
"""
import datetime
import sys
from pathlib import Path

import hydra
import torch
from accelerate import Accelerator
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src import in meta cluster
from src.losses import build_loss_fn
from src.train_utils import seed_everything, train_epoch, validate


def measure_flops(model, val_loader, device, accelerator=None):
    """Measure forward-pass FLOPs on one batch using PyTorch's built-in counter."""
    try:
        from torch.utils.flop_counter import FlopCounterMode
    except ImportError:
        print("FlopCounterMode not available (requires PyTorch >= 2.1)")
        return None

    unwrapped = accelerator.unwrap_model(model) if accelerator else model
    unwrapped.eval()

    # Grab one batch
    batch = next(iter(val_loader))
    images = batch["image"].to(device)
    labels = batch["label"].to(device)
    if labels.dim() == 3:
        labels = labels.unsqueeze(1)

    context_in = batch.get("context_in")
    context_out = batch.get("context_out")
    if context_in is not None:
        context_in = context_in.to(device)
    if context_out is not None:
        context_out = context_out.to(device)

    target_features = batch.get("target_features")
    context_features = batch.get("context_features")
    if target_features is not None:
        target_features = target_features.to(device)
    if context_features is not None:
        context_features = context_features.to(device)

    batch_size = images.shape[0]

    with torch.no_grad():
        flop_counter = FlopCounterMode(display=False)
        with flop_counter:
            unwrapped(
                images, labels=labels,
                context_in=context_in, context_out=context_out,
                target_features=target_features, context_features=context_features,
                mode="val",
            )
        total_flops = flop_counter.get_total_flops()

    per_sample = total_flops / batch_size
    return {
        "total_flops_batch": total_flops,
        "flops_per_sample": per_sample,
        "gflops_per_sample": per_sample / 1e9,
        "batch_size": batch_size,
    }


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    """Main training function."""
    # Print config
    # print(OmegaConf.to_yaml(cfg))

    # Initialize accelerator for multi-GPU support
    from accelerate import DistributedDataParallelKwargs
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    device = accelerator.device

    # Set seed
    seed_everything(cfg.training.seed)

    if accelerator.is_main_process:
        print(f"Using device: {device}, num_processes: {accelerator.num_processes}")

    # Create checkpoint dir (only on main process)
    ckpt_dir = Path(cfg.paths.ckpts.save_dir)
    if accelerator.is_main_process:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()

    # Get dataset class and dataloader
    if cfg.dataset == "totalseg2d":
        from src.dataloaders.totalseg2d_dataloader import (
            get_dataloader as get_totalseg2d_dataloader,
        )
    elif cfg.dataset == "totalsegmri2d":
        from src.dataloaders.totalseg2d_dataloader import (
            get_dataloader as get_totalseg2d_dataloader,
        )
    elif cfg.dataset == "medsegbench":
        from src.dataloaders.medsegbench_dataloader import (
            get_dataloader as get_medsegbench_dataloader,
        )
    else:
        raise ValueError(f"Unknown dataset: {cfg.dataset}")

    # Logging (only on main process)
    if cfg.logging.use_wandb and accelerator.is_main_process:
        import wandb
        wandb.init(
            project=cfg.logging.wandb_project_eval,
            config=OmegaConf.to_container(cfg, resolve=True),  # wandb needs plain dict
        )
    # Build dataloaders
    # Support separate max_ds_len for train/val, with fallback to single value
    max_ds_len_cfg = cfg.get("max_ds_len")
    if isinstance(max_ds_len_cfg, dict) or OmegaConf.is_dict(max_ds_len_cfg):
        max_ds_len_val = max_ds_len_cfg.get("val")
    else:
        max_ds_len_val = max_ds_len_cfg

    if cfg.dataset == "totalseg2d":
        # Handle label_ids: keep string for split names, convert to list for explicit IDs
        val_labels = cfg.val_label_ids if isinstance(cfg.val_label_ids, str) else list(cfg.val_label_ids)
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
            split="train",
            shuffle=False,
            random_context=False,
            max_ds_len=max_ds_len_val,
            random_coloring_nb=cfg.get("random_coloring_nb", 0),
        )
    elif cfg.dataset == "totalsegmri2d":
        # Handle label_ids: keep string for split names, convert to list for explicit IDs
        val_labels = cfg.val_label_ids if isinstance(cfg.val_label_ids, str) else list(cfg.val_label_ids)
        val_loader = get_totalseg2d_dataloader(
            root_dir=cfg.paths.totalsegmri2d,
            stats_path=cfg.paths.totalsegmri_stats,
            label_id_list=val_labels,
            context_size=cfg.context_size,
            batch_size=cfg.val_batch_size,
            image_size=tuple(cfg.preprocessing.image_size[:2]),
            crop_to_bbox=cfg.preprocessing.crop_to_bbox,
            bbox_padding=cfg.preprocessing.bbox_padding,
            num_workers=cfg.training.get("num_workers", 4),
            split="val",
            shuffle=False,
            random_context=False,
            max_ds_len=max_ds_len_val,
            random_coloring_nb=cfg.get("random_coloring_nb", 0),
        )
    elif cfg.dataset == "medsegbench":
        msb_cfg = cfg.get("medsegbench", {})
        msb_val_datasets = msb_cfg.get("val_datasets", None)
        if msb_val_datasets is not None:
            msb_val_datasets = list(msb_val_datasets)
        else:
            msb_datasets = msb_cfg.get("datasets", None)
            msb_val_datasets = list(msb_datasets) if msb_datasets is not None else None
        max_samples = msb_cfg.get("max_samples_per_dataset", None)

        if accelerator.is_main_process:
            print(f"MedSegBench val datasets: {msb_val_datasets}")

        val_loader = get_medsegbench_dataloader(
            data_root=cfg.paths.medsegbench,
            datasets=msb_val_datasets,
            split="val",
            context_size=cfg.context_size,
            batch_size=cfg.val_batch_size,
            image_size=tuple(cfg.preprocessing.image_size[:2]),
            num_workers=cfg.training.get("num_workers", 4),
            shuffle=False,
            augment=False,
            max_samples_per_dataset=max_samples,
        )
    # Get model (don't move to device yet - accelerator.prepare handles that)
    if cfg.method == "patch_icl":
        from src.models.patch_icl_v2 import PatchICL

        # Set num_mask_channels based on random_coloring_nb
        patch_icl_cfg = OmegaConf.to_container(cfg.model.patch_icl, resolve=True)
        random_coloring_nb = cfg.get("random_coloring_nb", 0)
        patch_icl_cfg["num_mask_channels"] = 3 if random_coloring_nb > 0 else 1
        if accelerator.is_main_process:
            print(f"Mask channels: {patch_icl_cfg['num_mask_channels']} (random_coloring_nb={random_coloring_nb})")

        # Create feature extractor for on-the-fly mode if needed
        feature_extractor = None
        feature_mode = cfg.get("feature_mode", "precomputed")
        if feature_mode == "on_the_fly":
            fe_cfg = patch_icl_cfg.get("feature_extractor", None)
            if fe_cfg is not None:
                extractor_type = fe_cfg.get("type", "meddino").lower()
            else:
                extractor_type = cfg.get("feature_extractor_type", "meddino").lower()

            if extractor_type in ["meddino", "meddinov3", "meddino_v3"]:
                from src.models.meddino_extractor import create_meddino_extractor
                if accelerator.is_main_process:
                    print("Initializing MedDINOv3 for on-the-fly feature extraction...")
                if fe_cfg is not None:
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
                    print(f"Feature mode: on_the_fly (MedDINO, resolution={fe_cfg.get('target_size', 256) if fe_cfg else cfg.get('feature_extraction_resolution', 256)})")

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
                    print(f"Feature mode: on_the_fly (MedSAM v1 layer {info['layer_idx']}, "
                          f"input={info['target_size']}×{info['target_size']}, "
                          f"grid={info['output_grid_size']}×{info['output_grid_size']})")
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
                )
                if accelerator.is_main_process:
                    info = feature_extractor.get_feature_info()
                    print(f"Feature mode: on_the_fly (UniverSeg layers={info['layer_indices']}, "
                          f"dim={info['feature_dim']}, grid={info['output_grid_size']})")
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
                    print(f"Feature mode: on_the_fly (ICLEncoder layers={info['layer_indices']}, "
                          f"dim={info['feature_dim']}, grid={info['output_grid_size']})")
            else:
                raise ValueError(f"Unknown feature_extractor_type: {extractor_type}")
        else:
            if accelerator.is_main_process:
                print(f"Feature mode: precomputed")

        model = PatchICL(
            patch_icl_cfg,
            context_size=cfg.get("context_size", 0),
            feature_extractor=feature_extractor,
        )

        # Build and set loss functions from patch_icl config
        loss_cfg = patch_icl_cfg.get("loss", {})
        patch_loss_cfg = loss_cfg.get("patch_loss", {"type": "dice", "args": None})
        aggreg_loss_cfg = loss_cfg.get("aggreg_loss", {"type": "dice", "args": None})

        patch_criterion = build_loss_fn(patch_loss_cfg["type"], patch_loss_cfg.get("args"))
        aggreg_criterion = build_loss_fn(aggreg_loss_cfg["type"], aggreg_loss_cfg.get("args"))
        model.set_loss_functions(patch_criterion, aggreg_criterion)
        if accelerator.is_main_process:
            print(f"Loss functions: patch={patch_loss_cfg['type']}, aggreg={aggreg_loss_cfg['type']}")
    elif cfg.method == "universeg":
        from src.models.universeg_baseline import UniverSegBaseline

        model = UniverSegBaseline(pretrained=True)
        
        # Set loss functions for evaluation
        aggreg_criterion = build_loss_fn("dice", None)
        patch_criterion = build_loss_fn("dice", None)
        model.set_loss_functions(patch_criterion, aggreg_criterion)
        if accelerator.is_main_process:
            print("Using UniverSeg baseline model")
    else:
        raise ValueError(f"Unknown method: {cfg.method}")
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if accelerator.is_main_process:
        print(f"Model parameters: {num_params:,}")

    # Load checkpoint weights
    ckpt_path = cfg.get("checkpoint", None)
    if ckpt_path:
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        if accelerator.is_main_process:
            print(f"Loaded checkpoint from {ckpt_path} (epoch {checkpoint.get('epoch', '?')}, dice {checkpoint.get('best_dice', '?'):.4f})")
    else:
        if accelerator.is_main_process:
            print("No checkpoint loaded, using default weights")

    # Optimizer
    opt_type = cfg.optimizer.get("optimizer_type", "adamw").lower()
    opt_args = cfg.optimizer.optimizer_args
    if opt_type == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=opt_args.lr,
            weight_decay=opt_args.weight_decay,
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

    # Prepare for distributed training
    model, optimizer, val_loader, scheduler = accelerator.prepare(
        model, optimizer, val_loader, scheduler
    )

    # Training loop
    best_dice = 0.0
    num_epochs = cfg.training.num_epochs
    print_every = cfg.training.print_every
    grad_accumulate_steps = cfg.training.get("grad_accumulate_steps", 1)

    # Measure FLOPs (one-time, before validation)
    flop_info = None
    if accelerator.is_main_process:
        flop_info = measure_flops(model, val_loader, device, accelerator)
        if flop_info:
            print(f"FLOPs per sample: {flop_info['gflops_per_sample']:.3f} GFLOPs "
                  f"(batch_size={flop_info['batch_size']})")

    # Validation with optional saving
    save_imgs = cfg.logging.get("save_imgs_masks", False)
    if save_imgs:
        results_dir = Path(cfg.paths.get("RESULTS_DIR", ckpt_dir / "eval_results"))
        val_save_dir = results_dir / f"{cfg.dataset}_{cfg.method}_eval"
        if accelerator.is_main_process:
            print(f"Saving evaluation images to: {val_save_dir}")
    else:
        val_save_dir = None

    val_loss, val_local_dice, val_final_dice, val_context_dice, detailed_results = validate(
        model, val_loader, device,
        save_dir=val_save_dir, max_save_batches=len(val_loader),
        accelerator=accelerator, use_wandb=cfg.logging.use_wandb, epoch=0
    )
    if accelerator.is_main_process:
        print(
            f"Val Loss: {val_loss:.5f} | "
            f"Val LocalDice: {val_local_dice:.5f} | "
            f"Val FinalDice: {val_final_dice:.5f} | "
            f"Val CtxDice: {val_context_dice:.5f}"
        )
        # Print per-label dice
        print("\nPer-label Dice:")
        for label_id, dice in sorted(detailed_results["per_label"].items(), key=lambda x: x[1], reverse=True):
            print(f"  {label_id}: {dice:.4f}")

    if cfg.logging.use_wandb and accelerator.is_main_process:
        log_dict = {
            "val_loss": val_loss,
            "val_local_dice": val_local_dice,
            "val_final_dice": val_final_dice,
            "val_context_dice": val_context_dice,
        }
        if flop_info:
            log_dict["gflops_per_sample"] = flop_info["gflops_per_sample"]
            log_dict["flops_per_sample"] = flop_info["flops_per_sample"]
        # Log per-label dice
        for label_id, dice in detailed_results["per_label"].items():
            log_dict[f"dice_label/{label_id}"] = dice
        wandb.log(log_dict)

        # Log per-case results as a wandb Table
        case_table = wandb.Table(columns=["case_id", "label_id", "axis", "dice"])
        for result in detailed_results["per_case"]:
            case_table.add_data(result["case_id"], result["label_id"], result.get("axis") or "N/A", result["dice"])
        wandb.log({"per_case_dice": case_table})

    """
    # Save best (only on main process)
    if val_final_dice > best_dice:
        best_dice = val_final_dice
        if accelerator.is_main_process:
            print(f"  -> New best dice: {best_dice:.5f}")
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            model_name = f"_ds_{cfg.dataset}_method_{cfg.method}"
            model_name = model_name + "_" + wandb.run.name if cfg.logging.use_wandb else str(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
            unwrapped_model = accelerator.unwrap_model(model)
            torch.save({
                "model_state_dict": unwrapped_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_dice": best_dice,
                "config": OmegaConf.to_container(cfg, resolve=True),  # checkpoint needs plain dict
            }, ckpt_dir / f"{model_name}_best_model.pt")
    """
    if accelerator.is_main_process:
        print(f"\nVal complete! Avg Dice: {val_final_dice:.5f}")

    if cfg.logging.use_wandb and accelerator.is_main_process:
        wandb.finish()


if __name__ == "__main__":
    main()
