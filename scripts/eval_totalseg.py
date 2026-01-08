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
sys.path.append("/nfs/norasys/notebooks/camaret/repos/Medverse")

# Define organs
DEFAULT_ORGAN_LIST = [
    "spinal_cord",
    "autochthon_left",
    "autochthon_right",
    "aorta",
    "esophagus",
    "lung_upper_lobe_left",
    "lung_upper_lobe_right",
    "costal_cartilages",
    "liver",
    "heart",
    "colon",
    "stomach",
    "femur_left",
    "femur_right",
]


def evaluate_totalseg(
    dataset_path: str = "/nfs/data/nii/data1/Analysis/camaret___in_context_segmentation/ANALYSIS_20251122/data/TotalSeg",
    model_path: str = "/nfs/norasys/notebooks/camaret/repos/Medverse/Medverse.ckpt",
    organ_list: Optional[List[str]] = None,
    context_size: int = 3,
    spacing: tuple = (1.5, 1.5, 1.5),
    sw_roi_size: tuple = (128, 128, 128),
    sw_overlap: float = 0.25,
    use_wandb: bool = True,
    save_imgs_masks: bool = False,
) -> float:
    """Evaluate Medverse model on TotalSeg validation set.

    Returns mean Dice score across validation batches.
    """
    if organ_list is None:
        organ_list = list(DEFAULT_ORGAN_LIST)

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
                    "organ_list": organ_list,
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
        organ_list=organ_list,
        empty_segmentations=empty_segmentations,
        context_size=context_size,
        batch_size=1,
        image_size=None,
        spacing=spacing,
        num_workers=4,
        split="val",
        shuffle=False,
        random_context=True,
    )

    # Import model/torch lazily to avoid import at module load
    import torch
    from medverse.lightning_model import LightningModel

    # Load model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = (
        LightningModel.load_from_checkpoint(model_path, map_location=device)
        .to(device)
        .eval()
    )

    # Loop over validation set
    dice_scores: List[float] = []
    for i, batch in enumerate(val_loader):
        print(f"Processing batch {i + 1}/{len(val_loader)}")
        target_in = batch["target_in"].to(device)
        context_in = batch["context_in"].to(device)
        context_out = batch["context_out"].to(device)

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
                organ = batch["organs"][i]
                save_dir = Path(config["RESULTS_DIR"]) / "totalseg_eval" / case_id
                save_dir.mkdir(exist_ok=True, parents=True)
                img_nib = nib.Nifti1Image(
                    batch["target_in"][i, 0].cpu().numpy(),
                affine=np.eye(4),
                )
                pred_mask_nib = nib.Nifti1Image(
                    prediction_binary[0, 0].cpu().numpy(),
                    affine=np.eye(4),
                )
                gt_mask_nib = nib.Nifti1Image(
                    batch["target_out"][0, 0].cpu().numpy(),
                    affine=np.eye(4),
                )
                nib.save(img_nib, save_dir / f"{organ}_img.nii.gz")
                nib.save(pred_mask_nib, save_dir / f"{organ}_pred_mask.nii.gz")
                nib.save(gt_mask_nib, save_dir / f"{organ}_gt_mask.nii.gz")
                print(f"  Saved images and masks to {save_dir}")

        if wandb_run is not None:
            for i, case_id in enumerate(batch["target_case_ids"]):
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

                try:
                    # type: ignore
                    
                    wandb.log(
                        {
                            "organ": batch["organs"][i],
                            "target_case_id": batch["target_case_ids"][i],
                            "gt_voxel_count": float(target_out.sum().item()),
                            "dice": float(dice.item()),
                            "inference_time_sec": float(end_time - start_time),
                            "target_image": wandb.Image(
                                target_img_slice,
                                masks={
                                    "pred": {
                                        "mask_data": pred_mask_slice,
                                        "class_labels": {0: "background", 1: "organ"},
                                    },
                                    "gt": {
                                        "mask_data": target_mask_slice,
                                        "class_labels": {0: "background", 1: "organ"},
                                    },
                                },
                            ),
                            "context_image": wandb.Image(
                                context_img_slice,
                                masks={
                                    "gt": {
                                        "mask_data": context_mask_slice,
                                        "class_labels": {0: "background", 1: "organ"},
                                    }
                                },
                            ),
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
        default=3,
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
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_argparser()
    args = parser.parse_args(argv)

    mean_dice = evaluate_totalseg(
        dataset_path=args.dataset_path,
        model_path=args.model_path,
        organ_list=DEFAULT_ORGAN_LIST,
        context_size=args.context_size,
        spacing=tuple(args.spacing),
        sw_roi_size=tuple(args.sw_roi_size),
        sw_overlap=args.sw_overlap,
        use_wandb=not args.no_wandb,
        save_imgs_masks=args.save_imgs_masks,
    )
    # Import images and masks to nora project
    project_name = "camaret___in_context_segmentation"

    import os 
    pattern = r'(?P<patients_id>totalseg_eval)/(?P<studies_id>s\d{4})/.*_img\.nii\.gz$'
    os.system(f"nora -p {project_name} --importfiles {config['RESULTS_DIR']} '{pattern}'")

    pattern = r'(?P<patients_id>totalseg_eval)/(?P<studies_id>s\d{4})/.*_gt_mask\.nii\.gz$'
    os.system(f"nora -p {project_name} --importfiles {config['RESULTS_DIR']} '{pattern}'")

    pattern = r'(?P<patients_id>totalseg_eval)/(?P<studies_id>s\d{4})/.*_pred_mask\.nii\.gz$'
    os.system(f"nora -p {project_name} --importfiles {config['RESULTS_DIR']} '{pattern}'")

    os.system(f"nora -p {project_name} --addtag mask --select '*' '*_gt_mask.nii.gz'")
    os.system(f"nora -p {project_name} --addtag mask --select '*' '*_pred_mask.nii.gz'")


if __name__ == "__main__":
    sys.exit(main())

