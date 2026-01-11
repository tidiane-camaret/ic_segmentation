import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Optional
import nibabel as nib
import numpy as np
from src.config import config
from src.totalseg_dataloader import get_dataloader

# Define label_ids
DEFAULT_LABEL_ID_LIST = [
        "heart",
]

"""
    "spinal_cord",
    "autochthon_left",
    "autochthon_right",
    "aorta",
    "esophagus",
    "lung_upper_lobe_left",
    "lung_upper_lobe_right",
    "costal_cartilages",
    "liver",
    "colon",
    "stomach",
    "femur_left",
    "femur_right",
]
"""

def evaluate_totalseg(
    dataset_path: str = "/nfs/data/nii/data1/Analysis/camaret___in_context_segmentation/ANALYSIS_20251122/data/TotalSeg",
    model_path: str = "/nfs/norasys/notebooks/camaret/repos/Medverse/Medverse.ckpt",
    label_id_list: Optional[List[str]] = None,
    context_size: int = 1,
    spacing: tuple = (1.5, 1.5, 1.5),
    sw_roi_size: tuple = (128, 128, 128),
    sw_overlap: float = 0.25,
    use_wandb: bool = True,
    save_imgs_masks: bool = False,
    enable_inspection: bool = False,
    max_inspect_cases: int = 1,
) -> float:
    """Evaluate Medverse model on TotalSeg validation set.
    Returns mean Dice score across validation batches.
    """
    if label_id_list is None:
        label_id_list = list(DEFAULT_LABEL_ID_LIST)

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

            wandb_run = wandb.init(
                project="ic_segmentation",
                config={
                    "context_size": context_size,
                    "label_id_list": label_id_list,
                    "dataset": "TotalSeg",
                },
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
        batch_size=1,
        image_size=None,
        spacing=spacing,
        num_workers=4,
        split="val",
        shuffle=False,
        random_context=False,
        max_ds_len=1,
    )

    # Import model/torch lazily to avoid import at module load
    import torch
    from src.medverse_foreground_sampling import LightningModelForegroundSampling

    # Load model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = (
        LightningModelForegroundSampling.load_from_checkpoint(model_path, map_location=device)
        .to(device)
        .eval()
    )

    # Loop over validation set
    dice_scores: List[float] = []
    inspected_cases = 0

    for i, batch in enumerate(val_loader):
        print(f"Processing batch {i + 1}/{len(val_loader)}")
        target_in = batch["target_in"].to(device)
        context_in = batch["context_in"].to(device)
        context_out = batch["context_out"].to(device)

        # Enable inspection for first few cases if requested
        if enable_inspection and inspected_cases < max_inspect_cases:
            print(f"  Enabling patch inspection for this case")
            model.enable_patch_inspection()
        else:
            if hasattr(model, 'disable_patch_inspection'):
                model.disable_patch_inspection()

        start_time = time.time()
        with torch.no_grad():
            prediction = model.autoregressive_inference(
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

        # Save inspection data if enabled
        if enable_inspection and inspected_cases < max_inspect_cases:
            case_id = batch["target_case_ids"][0]
            label_id = batch["label_ids"][0]
            inspect_dir = Path(config["RESULTS_DIR"]) / "totalseg_inspection"
            model.save_inspection_data_to_nifti(inspect_dir, case_id, label_id)
            inspected_cases += 1
            print(f"  Saved inspection data for {case_id}/{label_id}")

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

        if save_imgs_masks:
            for i, case_id in enumerate(batch["target_case_ids"]):
                label_id = batch["label_ids"][i]
                save_dir = Path(config["RESULTS_DIR"]) / "totalseg_eval" / case_id
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
                        }
                    )
                except Exception as e:
                    print(f"WANDB log error (continuing without logging): {e}")

    mean_dice = float(sum(dice_scores) / max(1, len(dice_scores)))
    print(f"Mean Dice over {len(dice_scores)} batches: {mean_dice:.4f}")
    return mean_dice


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate Medverse model on TotalSeg validation set",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="/nfs/data/nii/data1/Analysis/camaret___in_context_segmentation/ANALYSIS_20251122/data/TotalSeg",
        help="Path to TotalSeg root directory",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="/nfs/norasys/notebooks/camaret/repos/Medverse/Medverse.ckpt",
        help="Path to Medverse checkpoint (.ckpt)",
    )
    parser.add_argument(
        "--context-size",
        type=int,
        default=1,
        help="Number of context examples",
    )
    parser.add_argument(
        "--save-imgs-masks",
        action="store_true",
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
        default=(1.5, 1.5, 1.5),
        help="Resample spacing (x y z)",
    )
    parser.add_argument(
        "--sw-roi-size",
        type=int,
        nargs=3,
        default=(128, 128, 128),
        help="Sliding window ROI size",
    )
    parser.add_argument(
        "--sw-overlap",
        type=float,
        default=0.25,
        help="Sliding window overlap",
    )
    parser.add_argument(
        "--enable-inspection",
        action="store_true",
        help="Enable detailed patch inspection (saves all patches to NIfTI)",
    )
    parser.add_argument(
        "--max-inspect-cases",
        type=int,
        default=1,
        help="Maximum number of cases to inspect (default: 1)",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_argparser()
    args = parser.parse_args(argv)

    mean_dice = evaluate_totalseg(
        dataset_path=args.dataset_path,
        model_path=args.model_path,
        label_id_list=DEFAULT_LABEL_ID_LIST,
        context_size=args.context_size,
        spacing=tuple(args.spacing),
        sw_roi_size=tuple(args.sw_roi_size),
        sw_overlap=args.sw_overlap,
        use_wandb=not args.no_wandb,
        save_imgs_masks=args.save_imgs_masks,
        enable_inspection=args.enable_inspection,
        max_inspect_cases=args.max_inspect_cases,
    )



if __name__ == "__main__":
    sys.exit(main())

