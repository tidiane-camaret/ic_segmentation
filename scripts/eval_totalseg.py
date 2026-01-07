import sys
import json
import wandb
import time
from pathlib import Path
from src.config import config

sys.path.append("/nfs/norasys/notebooks/camaret/repos/Medverse")
from medverse.data.totalseg_dataloader import get_dataloader

# Define organs (from your scan results)
organ_list = [
    "spinal_cord",
    "autochthon_left",
    "autochthon_right",
    "aorta",
    "esophagus",
]


dataset_path = "/nfs/data/nii/data1/Analysis/camaret___in_context_segmentation/ANALYSIS_20251122/data/TotalSeg"
context_size = 3
model_path = "/nfs/norasys/notebooks/camaret/repos/Medverse/Medverse.ckpt"
stats_dict_path = Path(config["DATA_DIR"]) / "TotalSeg" /"dataset_stats.json"
with open(stats_dict_path, 'r') as f:
    stats_dict = json.load(f)
empty_segmentations = stats_dict['empty_segmentations']

use_wandb = True
if use_wandb:
    wandb.init(
        project="ic_segmentation",
        #name=run_name,
        config={
            "context_size": context_size,
            "organ_list": organ_list,
            "dataset" : "TotalSeg",
        },
    )
    wandb.define_metric("dice", summary="mean")
"""
# Create training dataloader
train_loader = get_dataloader(
    root_dir=dataset_path,
    organ_list=organ_list,
    context_size=context_size,              # Use 1 example pair
    batch_size=1,                # Usually 1 for 3D medical images
    image_size=(128, 128, 128),  # Resize to 128³
    spacing=(1.5, 1.5, 1.5),     # Resample to 1.5mm isotropic
    num_workers=4,
    mode='train',
    shuffle=True,
    random_context=True,         # Randomly sample contexts each epoch
)
"""
# Create validation dataloader
val_loader = get_dataloader(
    root_dir=dataset_path,
    organ_list=organ_list,
    empty_segmentations=empty_segmentations,
    context_size=context_size,
    batch_size=1,
    image_size=None,  # Dynamically resize to target size
    spacing=(1.5, 1.5, 1.5),
    num_workers=4,
    mode='val',
    shuffle=False,
    random_context=False,        # Use same contexts for reproducibility
)

from medverse.lightning_model import LightningModel
import torch

# Load model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = LightningModel.load_from_checkpoint(model_path, map_location=device).to(device).eval()

"""
# Get a batch
batch = next(iter(val_loader))

# Extract inputs
target_in = batch['target_in'].cuda()        # [1, 1, 128, 128, 128]
context_in = batch['context_in'].cuda()      # [1, 5, 1, 128, 128, 128]
context_out = batch['context_out'].cuda()    # [1, 5, 1, 128, 128, 128]

# Run inference
with torch.no_grad():
    prediction = model.autoregressive_inference(
        target_in=target_in,
        context_in=context_in,
        context_out=context_out,
        level=None,                    # Auto-calculate levels
        forward_l_arg=3,               # Process 3 contexts at a time
        sw_roi_size=(128, 128, 128),
        sw_overlap=0.25,
        sw_batch_size_val=1,
    )

print(f"Prediction shape: {prediction.shape}")  # [1, 1, 128, 128, 128]

# compute dice

prediction_binary = (prediction > 0.5).float()
target_out = batch['target_out'].cuda()
intersection = (prediction_binary * target_out).sum()
dice = (2.0 * intersection) / (prediction_binary.sum() + target_out.sum() + 1e-8)
print(f"Dice score: {dice.item():.4f}")
"""
## loop over validation set
dice_scores = []
for i, batch in enumerate(val_loader):
    print(f"Processing batch {i+1}/{len(val_loader)}")
    target_in = batch['target_in'].cuda()
    context_in = batch['context_in'].cuda()
    context_out = batch['context_out'].cuda()

    start_time = time.time()
    with torch.no_grad():
        prediction = model.autoregressive_inference(
            target_in=target_in,
            context_in=context_in,
            context_out=context_out,
            level=None,
            forward_l_arg=3,
            sw_roi_size=(128, 128, 128),
            sw_overlap=0.25,
            sw_batch_size_val=1,
        )
    end_time = time.time()
    print(f"Inference time: {end_time - start_time:.2f} seconds")

    prediction_binary = (prediction > 0.5).float()
    target_out = batch['target_out'].cuda()
    print(f"Prediction shape: {prediction.shape}, Target shape: {target_out.shape}")
    print(f"  Predicted voxels: {prediction_binary.sum().item()}, Target voxels: {target_out.sum().item()}")
    intersection = (prediction_binary * target_out).sum()
    dice = (2.0 * intersection) / (prediction_binary.sum() + target_out.sum() + 1e-8)
    dice_scores.append(dice.item())
    print(f"  Dice score: {dice.item():.4f}")
    if use_wandb:
        # compute target slice with most foreground voxels
        mri_slice = batch['target_in'][0, 0, :, :, :].sum(dim=(0, 1))
        slice_idx = torch.argmax(mri_slice).item()
        target_img_slice = batch['target_in'][0, 0, :, :, slice_idx].cpu().numpy()
        target_mask_slice = batch['target_out'][0, 0, :, :, slice_idx].cpu().numpy()
        pred_mask_slice = prediction_binary[0, 0, :, :, slice_idx].cpu().numpy()

        # compute context slice with most foreground voxels
        mri_slice_context = batch['context_in'][0, 0, 0, :, :, :].sum(dim=(0, 1))
        slice_idx_context = torch.argmax(mri_slice_context).item()
        context_img_slice = batch['context_in'][0, 0, 0, :, :, slice_idx_context].cpu().numpy()
        context_mask_slice = batch['context_out'][0, 0, 0, :, :, slice_idx_context].cpu().numpy()

        wandb.log({
            "organ": batch["organs"][0],
            "target_case_id": batch["target_case_id"][0],
            "gt_voxel_count": target_out.sum().item(),
            "dice": dice.item(),
            "inference_time_sec": end_time - start_time,
            "target_image": wandb.Image(
                target_img_slice,
                masks={
                    "pred":{"mask_data": pred_mask_slice, "class_labels": {0: "background", 1: "organ"}},
                    "gt": {"mask_data": target_mask_slice, "class_labels": {0: "background", 1: "organ"}},
                    }
                ),
            "context_image": wandb.Image(
                context_img_slice,
                masks={
                    "gt": {"mask_data": context_mask_slice, "class_labels": {0: "background", 1: "organ"}},
                    }
                )
        })

    if i >= 1:  # Limit to first 10 samples for demo
        break