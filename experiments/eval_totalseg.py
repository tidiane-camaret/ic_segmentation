import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Optional, Union

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import nibabel as nib
import numpy as np
from ic_segmentation.old.config import config
from src.totalseg_dataloader import get_dataloader
from data.label_ids_totalseg import get_label_ids


def evaluate_totalseg(
    dataset_path: Optional[str] = None,
    model_path: Optional[str] = None,
    label_id_list: Optional[List[str]] = None,
    label_ids_split: Optional[Union[str, List[str]]] = None,
    context_size: Optional[int] = None,
    spacing: Optional[tuple] = None,
    sw_roi_size: Optional[tuple] = None,
    sw_overlap: Optional[float] = None,
    sampling_method: Optional[str] = None,
    # Dataloader parameters
    batch_size: Optional[int] = None,
    image_size: Optional[tuple] = None,
    num_workers: Optional[int] = None,
    split: Optional[str] = None,
    shuffle: Optional[bool] = None,
    random_context: Optional[bool] = None,
    max_ds_len: Optional[int] = None,
    # Logging and inspection
    use_wandb: Optional[bool] = None,
    save_imgs_masks: Optional[bool] = None,
    enable_inspection: Optional[bool] = None,
    max_inspect_cases: Optional[int] = None,
) -> float:
    """Evaluate Medverse model on TotalSeg validation set.
    Returns mean Dice score across validation batches.
    """
    # Load defaults from config if not provided
    eval_config = config.get("eval_totalseg", {})

    if dataset_path is None:
        dataset_path = eval_config.get("dataset_path", "/nfs/data/nii/data1/Analysis/camaret___in_context_segmentation/ANALYSIS_20251122/data/TotalSeg")
    if model_path is None:
        model_path = eval_config.get("model_path", "/nfs/norasys/notebooks/camaret/repos/Medverse/Medverse.ckpt")
    if label_ids_split is None:
        label_ids_split = eval_config.get("label_ids_split", "val")
    if context_size is None:
        context_size = eval_config.get("context_size", 1)
    if spacing is None:
        spacing = tuple(eval_config.get("spacing", [1.5, 1.5, 1.5]))
    if sw_roi_size is None:
        sw_roi_size = tuple(eval_config.get("sw_roi_size", [128, 128, 128]))
    if sw_overlap is None:
        sw_overlap = eval_config.get("sw_overlap", 0.25)
    if sampling_method is None:
        sampling_method = eval_config.get("sampling_method", "foreground")

    # Dataloader parameters
    if batch_size is None:
        batch_size = eval_config.get("batch_size", 1)
    if image_size is None:
        image_size_cfg = eval_config.get("image_size", None)
        image_size = tuple(image_size_cfg) if image_size_cfg is not None else None
    if num_workers is None:
        num_workers = eval_config.get("num_workers", 4)
    if split is None:
        split = eval_config.get("split", "val")
    if shuffle is None:
        shuffle = eval_config.get("shuffle", False)
    if random_context is None:
        random_context = eval_config.get("random_context", False)
    if max_ds_len is None:
        max_ds_len = eval_config.get("max_ds_len", None)

    # Logging and inspection
    if use_wandb is None:
        use_wandb = eval_config.get("use_wandb", True)
    if save_imgs_masks is None:
        save_imgs_masks = eval_config.get("save_imgs_masks", False)
    if enable_inspection is None:
        enable_inspection = eval_config.get("enable_inspection", False)
    if max_inspect_cases is None:
        max_inspect_cases = eval_config.get("max_inspect_cases", 1)

    # Validate sampling_method
    if sampling_method not in ["original", "foreground"]:
        raise ValueError(f"sampling_method must be 'original' or 'foreground', got: {sampling_method}")

    # Load label_id_list based on split (unless explicitly provided)
    if label_id_list is None:
        if isinstance(label_ids_split, list):
            # Custom list of labels provided
            label_id_list = label_ids_split
            print(f"Using custom list of {len(label_id_list)} labels")
        elif isinstance(label_ids_split, str):
            # Predefined split ("train", "val", "all")
            if label_ids_split not in ["train", "val", "all"]:
                raise ValueError(f"label_ids_split string must be 'train', 'val', or 'all', got: {label_ids_split}")
            label_id_list = get_label_ids(label_ids_split)
            print(f"Loaded {len(label_id_list)} labels from '{label_ids_split}' split")
        else:
            raise ValueError(f"label_ids_split must be a string ('train'/'val'/'all') or a list of label IDs, got type: {type(label_ids_split)}")

    # Print configuration
    print("=" * 60)
    print("Evaluation Configuration:")
    print(f"  dataset_path: {dataset_path}")
    print(f"  model_path: {model_path}")
    if isinstance(label_ids_split, list):
        print(f"  label_ids_split: custom list")
        print(f"  labels: {label_ids_split}")
    else:
        print(f"  label_ids_split: {label_ids_split}")
    print(f"  num_labels: {len(label_id_list)}")
    print()
    print("Model parameters:")
    print(f"  context_size: {context_size}")
    print(f"  spacing: {spacing}")
    print(f"  sw_roi_size: {sw_roi_size}")
    print(f"  sw_overlap: {sw_overlap}")
    print(f"  sampling_method: {sampling_method}")
    print()
    print("Dataloader parameters:")
    print(f"  batch_size: {batch_size}")
    print(f"  image_size: {image_size}")
    print(f"  num_workers: {num_workers}")
    print(f"  split: {split}")
    print(f"  shuffle: {shuffle}")
    print(f"  random_context: {random_context}")
    print(f"  max_ds_len: {max_ds_len}")
    print()
    print("Logging and inspection:")
    print(f"  use_wandb: {use_wandb}")
    print(f"  save_imgs_masks: {save_imgs_masks}")
    print(f"  enable_inspection: {enable_inspection}")
    print(f"  max_inspect_cases: {max_inspect_cases}")
    print("=" * 60)

    # Load dataset stats
    stats_dict_path = Path(config["DATA_DIR"]) / "TotalSeg" / "dataset_stats.json"
    with open(stats_dict_path, "r") as f:
        stats_dict = json.load(f)
    empty_segmentations = stats_dict["empty_segmentations"]

    # Lazy import wandb only if requested
    wandb_run = None
    if use_wandb:
        try:
            import wandb  # type: ignore

            wandb_config = {
                "dataset": "TotalSeg",
                "label_ids_split": "custom" if isinstance(label_ids_split, list) else label_ids_split,
                "num_labels": len(label_id_list),
                # Model parameters
                "context_size": context_size,
                "spacing": spacing,
                "sw_roi_size": sw_roi_size,
                "sw_overlap": sw_overlap,
                "sampling_method": sampling_method,
                # Dataloader parameters
                "batch_size": batch_size,
                "image_size": image_size,
                "num_workers": num_workers,
                "split": split,
                "shuffle": shuffle,
                "random_context": random_context,
                "max_ds_len": max_ds_len,
            }
            # Add label list if custom
            if isinstance(label_ids_split, list):
                wandb_config["label_ids_custom"] = label_ids_split

            wandb_run = wandb.init(
                project="ic_segmentation",
                config=wandb_config,
            )
            wandb.define_metric("dice")
        except Exception as e:
            print(f"WANDB disabled due to init error: {e}")
            wandb_run = None

    # Create validation dataloader
    val_loader = get_dataloader(
        root_dir=dataset_path,
        label_id_list=label_id_list,
        empty_segmentations=empty_segmentations,
        context_size=context_size,
        batch_size=batch_size,
        image_size=image_size,
        spacing=spacing,
        num_workers=num_workers,
        split=split,
        shuffle=shuffle,
        random_context=random_context,
        max_ds_len=max_ds_len,
    )

    # Import model/torch lazily to avoid import at module load
    import torch

    # Load model - choose based on sampling method
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if sampling_method == "foreground":
        from src.medverse_foreground_sampling import LightningModelForegroundSampling
        print("Loading model with foreground sampling...")
        model = (
            LightningModelForegroundSampling.load_from_checkpoint(model_path, map_location=device)
            .to(device)
            .eval()
        )
    elif sampling_method == "original":
        from medverse.lightning_model import LightningModel
        print("Loading model with original Medverse sampling...")
        model = (
            LightningModel.load_from_checkpoint(model_path, map_location=device)
            .to(device)
            .eval()
        )

    # Loop over validation set
    dice_scores: List[float] = []
    inspected_cases = 0

    for i, batch in enumerate(val_loader):

        print(f"Processing batch {i + 1}/{len(val_loader)}")
        print(f"  Case IDs: {batch['target_case_ids']}, Label IDs: {batch['label_ids']}")
        print(f"  Target img shape: {batch['target_in'].shape}, Min/Max: {batch['target_in'].min().item():.4f}/{batch['target_in'].max().item():.4f}")
        print(f"  Target mask shape: {batch['target_out'].shape}, Min/Max: {batch['target_out'].min().item():.4f}/{batch['target_out'].max().item():.4f}")
        print(f"  Context img shape: {batch['context_in'].shape}, Min/Max: {batch['context_in'].min().item():.4f}/{batch['context_in'].max().item():.4f}")
        print(f"  Context mask shape: {batch['context_out'].shape}, Min/Max: {batch['context_out'].min().item():.4f}/{batch['context_out'].max().item():.4f}")
        
        target_in = batch["target_in"].to(device)
        context_in = batch["context_in"].to(device)
        context_out = batch["context_out"].to(device)

        # Enable inspection for first few cases if requested (only for foreground sampling)
        if enable_inspection and inspected_cases < max_inspect_cases:
            if hasattr(model, 'enable_patch_inspection'):
                print(f"  Enabling patch inspection for this case")
                model.enable_patch_inspection()
            else:
                print(f"  Warning: Inspection not supported for '{sampling_method}' sampling method")
        else:
            if hasattr(model, 'disable_patch_inspection'):
                model.disable_patch_inspection()

        start_time = time.time()
        with torch.no_grad():
            output = model.autoregressive_inference(
                target_in=target_in,
                context_in=context_in,
                context_out=context_out,
                level=None,
                forward_l_arg=3,
                sw_roi_size=sw_roi_size,
                sw_overlap=sw_overlap,
                sw_batch_size_val=1,
            )
        end_time = time.time()
        print(f"Inference time: {end_time - start_time:.2f} seconds")

        # Save inspection data if enabled (only for foreground sampling)
        if enable_inspection and inspected_cases < max_inspect_cases:
            if hasattr(model, 'save_inspection_data_to_nifti'):
                case_id = batch["target_case_ids"][0]
                label_id = batch["label_ids"][0]
                inspect_dir = Path(config["RESULTS_DIR"]) / "totalseg_inspection"
                model.save_inspection_data_to_nifti(inspect_dir, case_id, label_id)
                inspected_cases += 1
                print(f"  Saved inspection data for {case_id}/{label_id}")

        print(f"Output Min/Max: {output.min().item():.4f}/{output.max().item():.4f}")
        print(f"Output Mean/Std: {output.mean().item():.4f}/{output.std().item():.4f}")
        
        prediction = output # torch.sigmoid(output) 
        # TODO Seems like model was trained to output probs, 
        # even though no activation at end of block. To check.

        print(f"  Prediction shape: {prediction.shape}")
        print(f"  Pred min/max: {prediction.min().item():.4f}/{prediction.max().item():.4f}")
        print(f"  Pred mean/std: {prediction.mean().item():.4f}/{prediction.std().item():.4f}")
        print(f"  Target min/max: {batch['target_out'].min().item():.4f}/{batch['target_out'].max().item():.4f}")
        print(f"  Target mean/std: {batch['target_out'].mean().item():.4f}/{batch['target_out'].std().item():.4f}")

        prediction_binary = (prediction > 0.5).float()
        target_out = batch["target_out"].to(device)
        print(
            f"Prediction shape: {prediction.shape}, Target shape: {target_out.shape}"
        )
        print(
            f"  Predicted voxels: {prediction_binary.sum().item()}, Target voxels: {target_out.sum().item()}"
        )
        intersection = (prediction_binary * target_out).sum()
        dice = (2.0 * intersection) / (
            prediction_binary.sum() + target_out.sum() + 1e-8
        )
        dice_scores.append(float(dice.item()))
        print(f"  Dice score: {dice.item():.4f}")

        # Compute Dice between target GT and each context GT
        context_dice_scores = []
        num_contexts = batch["context_out"].shape[1]  # Get number of context examples
        print(f"  Context-to-target GT Dice scores:")
        for ctx_idx in range(num_contexts):
            context_gt = batch["context_out"][:, ctx_idx, :, :, :, :].to(device)
            ctx_intersection = (context_gt * target_out).sum()
            ctx_dice = (2.0 * ctx_intersection) / (
                context_gt.sum() + target_out.sum() + 1e-8
            )
            context_dice_scores.append(float(ctx_dice.item()))

        if save_imgs_masks or enable_inspection:
            if enable_inspection:
                save_dir_root = Path(config["RESULTS_DIR"]) / "totalseg_inspection"
            else:
                save_dir_root = Path(config["RESULTS_DIR"]) / "totalseg_eval"
            for i, case_id in enumerate(batch["target_case_ids"]):
                label_id = batch["label_ids"][i]
                save_dir = save_dir_root / case_id
                save_dir.mkdir(exist_ok=True, parents=True)
                img_nib = nib.Nifti1Image(
                    batch["target_in"][i, 0].cpu().numpy(),
                affine=np.eye(4),
                )
                pred_mask_nib = nib.Nifti1Image(
                    prediction_binary[i, 0].cpu().numpy(),
                    affine=np.eye(4),
                )
                gt_mask_nib = nib.Nifti1Image(
                    batch["target_out"][i, 0].cpu().numpy(),
                    affine=np.eye(4),
                )
                # 1rst context image
                context_img_nib = nib.Nifti1Image(
                    batch["context_in"][i, 0, 0].cpu().numpy(),
                    affine=np.eye(4),
                )
                context_gt_mask_nib = nib.Nifti1Image(
                    batch["context_out"][i, 0, 0].cpu().numpy(),
                    affine=np.eye(4),
                )

                nib.save(img_nib, save_dir / f"{label_id}_img.nii.gz")
                nib.save(pred_mask_nib, save_dir / f"{label_id}_pred_mask.nii.gz")
                nib.save(gt_mask_nib, save_dir / f"{label_id}_gt_mask.nii.gz")
                nib.save(context_img_nib, save_dir / f"{label_id}_context_img.nii.gz")
                nib.save(context_gt_mask_nib, save_dir / f"{label_id}_context_gt_mask.nii.gz")
                print(f"  Saved images and masks to {save_dir}")

        if wandb_run is not None:
            for i, case_id in enumerate(batch["target_case_ids"]):
                """
                # compute target slice with most foreground voxels
                mri_slice = batch["target_out"][i, 0 , :, :, :].sum(dim=(0, 1))
                slice_idx = int(torch.argmax(mri_slice).item())
                target_img_slice = (
                    batch["target_in"][i, 0, :, :, slice_idx].cpu().numpy()
                )
                target_mask_slice = (
                    batch["target_out"][i, 0, :, :, slice_idx].cpu().numpy()
                )
                pred_mask_slice = prediction_binary[i, 0, :, :, slice_idx].cpu().numpy()

                # compute context slice with most foreground voxels
                mri_slice_context = (
                    batch["context_out"][i, 0, 0, :, :, :].sum(dim=(0, 1))
                )
                slice_idx_context = int(torch.argmax(mri_slice_context).item())
                context_img_slice = (
                    batch["context_in"][i, 0, 0, :, :, slice_idx_context]
                    .cpu()
                    .numpy()
                )
                context_mask_slice = (
                    batch["context_out"][i, 0, 0, :, :, slice_idx_context]
                    .cpu()
                    .numpy()
                )
                """

                try:
                    # type: ignore

                    wandb.log(
                        {
                            "label_id": batch["label_ids"][i],
                            "target_case_id": batch["target_case_ids"][i],
                            "gt_voxel_count": float(target_out.sum().item()),
                            "dice": float(dice.item()),
                            "inference_time_sec": float(end_time - start_time),
                            "context_to_target_dice_scores": context_dice_scores,
                        }
                    )
                except Exception as e:
                    print(f"WANDB log error (continuing without logging): {e}")

    mean_dice = float(sum(dice_scores) / max(1, len(dice_scores)))
    print(f"Mean Dice over {len(dice_scores)} batches: {mean_dice:.4f}")
    return mean_dice


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate Medverse model on TotalSeg validation set. "
                    "Parameters not specified will use values from config.yaml.",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Path to TotalSeg root directory (default: from config.yaml)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to Medverse checkpoint (.ckpt) (default: from config.yaml)",
    )
    parser.add_argument(
        "--label-ids-split",
        type=str,
        nargs="+",
        default=None,
        help="Label split: 'train', 'val', 'all', or space-separated list of label IDs (default: from config.yaml)",
    )
    parser.add_argument(
        "--context-size",
        type=int,
        default=None,
        help="Number of context examples (default: from config.yaml)",
    )
    parser.add_argument(
        "--save-imgs-masks",
        action="store_true",
        default=None,
        help="Save predicted masks and input images as NIfTI files",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases logging",
    )
    parser.add_argument(
        "--spacing",
        type=float,
        nargs=3,
        default=None,
        help="Resample spacing (x y z) (default: from config.yaml)",
    )
    parser.add_argument(
        "--sw-roi-size",
        type=int,
        nargs=3,
        default=None,
        help="Sliding window ROI size (default: from config.yaml)",
    )
    parser.add_argument(
        "--sw-overlap",
        type=float,
        default=None,
        help="Sliding window overlap (default: from config.yaml)",
    )
    parser.add_argument(
        "--sampling-method",
        type=str,
        default=None,
        choices=["original", "foreground"],
        help="Context sampling method: 'original' (Medverse) or 'foreground' (custom) (default: from config.yaml)",
    )
    parser.add_argument(
        "--enable-inspection",
        action="store_true",
        default=None,
        help="Enable detailed patch inspection (saves all patches to NIfTI)",
    )
    parser.add_argument(
        "--max-inspect-cases",
        type=int,
        default=None,
        help="Maximum number of cases to inspect (default: from config.yaml)",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_argparser()
    args = parser.parse_args(argv)

    # Convert spacing and sw_roi_size to tuples if provided
    spacing = tuple(args.spacing) if args.spacing is not None else None
    sw_roi_size = tuple(args.sw_roi_size) if args.sw_roi_size is not None else None

    # Handle wandb flag - if --no-wandb is specified, override config
    use_wandb = not args.no_wandb if args.no_wandb else None

    # Process label_ids_split
    # If single value in list and it's a predefined split, convert to string
    # Otherwise keep as list for custom labels
    label_ids_split = args.label_ids_split
    if label_ids_split is not None:
        if len(label_ids_split) == 1 and label_ids_split[0] in ["train", "val", "all"]:
            label_ids_split = label_ids_split[0]
        # else: keep as list for custom labels

    mean_dice = evaluate_totalseg(
        dataset_path=args.dataset_path,
        model_path=args.model_path,
        label_id_list=None,  # Will be loaded from split
        label_ids_split=label_ids_split,
        context_size=args.context_size,
        spacing=spacing,
        sw_roi_size=sw_roi_size,
        sw_overlap=args.sw_overlap,
        sampling_method=args.sampling_method,
        # Dataloader parameters (use config defaults)
        batch_size=None,
        image_size=None,
        num_workers=None,
        split=None,
        shuffle=None,
        random_context=None,
        max_ds_len=None,
        # Logging and inspection
        use_wandb=use_wandb,
        save_imgs_masks=args.save_imgs_masks,
        enable_inspection=args.enable_inspection,
        max_inspect_cases=args.max_inspect_cases,
    )

    return 0



if __name__ == "__main__":
    sys.exit(main())

