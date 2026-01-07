# 1) Load the cases in config["RESULTS_DIR"] from tasks_dict
# 2) preprocess and run medverse pred
# 3) eval output (dice, nsd) and add mask pred to nora project

import os
import torch
import sys
sys.path.append("/nfs/norasys/notebooks/camaret/repos/Medverse")
sys.path.append("/software/notebooks/camaret/repos/Neuroverse3D")
from utils.dataloading import structure_data
from medverse.lightning_model import LightningModel
import numpy as np
import nibabel as nib
from pathlib import Path
from src.config import config
from src.utils import load_seg_data
from scripts.tasks_dict import tasks_dict

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_path = "/nfs/norasys/notebooks/camaret/repos/Medverse/Medverse.ckpt"
model = LightningModel.load_from_checkpoint(model_path, map_location=device).to(device).eval()
project_name = "camaret___in_context_segmentation"


seg_class = [1]
nb_cases = 5  # number of context cases + 1 (target case). None for all cases


for task_name, case_list in tasks_dict.items():
    # remove underscore from task_name
    task_name = task_name.replace("_", "")
    export_dir = Path(config["RESULTS_DIR"]) / task_name
    export_dir.mkdir(exist_ok=True)
    img_dir = export_dir / 'imgs'
    lab_dir = export_dir / 'labs'
    img_dir.mkdir(exist_ok=True)
    lab_dir.mkdir(exist_ok=True)

    ### Run inference using Medverse 

    images, labels = load_seg_data(img_dir, lab_dir, nb_max=nb_cases)
    print('Shape of images:',images.shape, '\nShape of labels:',labels.shape)

    target_in, context_in, target_out_raw, context_out_raw = structure_data(images, labels, index = 0, verbose = True)


    unique_masks = np.unique(target_out_raw) 
    print('Unique classes in the mask:', unique_masks)
    
    context_out = np.isin(context_out_raw, seg_class).astype(np.float32)
    target_out = np.isin(target_out_raw, seg_class).astype(np.float32)
    print('After binarization, unique classes in the target_out:', np.unique(target_out))
    
    target_in = torch.Tensor(target_in).to(device)  # (1, 1, H, W, D)
    target_out = torch.Tensor(target_out).to(device)  # (1, 1, H, W, D)
    context_in = torch.Tensor(context_in).to(device)  # (1, 1, H, W, D)
    context_out = torch.Tensor(context_out).to(device)  # (1, 1, H, W, D)
    # Normalize
    target_in = model.normalize_3d_volume(target_in)
    #target_out = model.normalize_3d_volume(target_out)
    context_in = model.normalize_3d_volume(context_in)
    context_out = model.normalize_3d_volume(context_out)


    # Inference
    with torch.no_grad():
        mask = model.autoregressive_inference(target_in,
                                            context_in,
                                            context_out,
                                            forward_l_arg=2, # min-context size. Lower if GPU memory is limited, min=1. No effect on the results.
                                            )
    # Convert mask to numpy array if it's a torch tensor
    mask_np = mask.cpu().detach().numpy() if hasattr(mask, 'cpu') else mask
    # Remove batch dimension 
    mask_np = mask_np[0,0]
    print(mask_np.shape)
    print(np.min(mask_np), np.max(mask_np))

    # Threshold: values > 0.5 become 1, else 0
    mask_np = (mask_np > 0.5).astype(np.int8)
    print(np.unique(mask_np, return_counts=True))

    target_out_np = target_out.cpu().detach().numpy()
    target_out_np = target_out_np[0,0]
    print(target_out_np.shape)
    print(np.unique(target_out_np, return_counts=True))
    target_out_np = (target_out_np > 0.5).astype(np.int8)
    print(np.unique(target_out_np, return_counts=True))

    # compute dsc and nsd
    sys.path.append("/nfs/norasys/notebooks/camaret/cvpr25/CVPR-MedSegFMCompetition")
    from SurfaceDice import (
        compute_surface_distances,
        compute_surface_dice_at_tolerance,
        compute_dice_coefficient,
    )
    dsc = compute_dice_coefficient(mask_np, target_out_np)

    # Load reference image to get affine
    # get first file of img_dir
    first_case_filename = sorted(list(img_dir.glob("*_img.nii.gz")))[0].name
    ref_nii = nib.load(img_dir / first_case_filename)
    affine = ref_nii.affine
    surface_distance = compute_surface_distances(mask_np, target_out_np, spacing_mm=np.diag(affine)[:3])
    nsd = compute_surface_dice_at_tolerance(surface_distance, tolerance_mm=2.0)
    print(f'Dice Similarity Coefficient (DSC): {dsc:.4f}')
    print(f'Normalized Surface Dice (NSD): {nsd:.4f}')

    # write mask to nifti 

    # Create NIfTI image with reference affine
    mask_nii = nib.Nifti1Image(mask_np, affine=affine)
    gt_nii = nib.Nifti1Image(target_out_np, affine=affine)

    # Save to file
    first_case_name = first_case_filename.split(".")[0].split("_img")[0]
    lab_pred_dir = export_dir / 'labs_pred'
    lab_pred_dir.mkdir(exist_ok=True)
    nib.save(mask_nii, lab_pred_dir / f'{first_case_name}_pred_medverse.nii.gz')
    nib.save(gt_nii, lab_pred_dir / f'{first_case_name}_gt_medverse.nii.gz')

    # save to nora project 
    # Match everything up to the last underscore before the file suffix
    # Match only the filename part (after last /)
    lab_regex = r"(?<patients_id>[^/_]+)_(?<studies_id>[^_]+)_pred_medverse\.nii\.gz"
    os.system(f"nora -p {project_name} --importfiles {lab_pred_dir} '{lab_regex}'")


    lab_regex = r"(?<patients_id>[^/_]+)_(?<studies_id>[^_]+)_gt_medverse\.nii\.gz"
    os.system(f"nora -p {project_name} --importfiles {lab_pred_dir} '{lab_regex}'")



    # save dsc and nsd to a df  
    import pandas as pd
    results_path = Path("results") / 'eval.csv'

    # Load existing results or create new DataFrame
    if results_path.exists():
        results_df = pd.read_csv(results_path)
    else:
        results_df = pd.DataFrame(columns=['TaskName', 'DSC_medverse', 'NSD_medverse'])

    # Create new row as DataFrame
    new_row = pd.DataFrame([{'TaskName': task_name, 'DSC_medverse': dsc, 'NSD_medverse': nsd}])

    # Concatenate
    results_df = pd.concat([results_df, new_row], ignore_index=True)

    # Save
    results_df.to_csv(results_path, index=False)

os.system(f"nora -p {project_name} --addtag mask --select '*' '*pred_medverse.nii.gz'")
os.system(f"nora -p {project_name} --addtag mask --select '*' '*gt_medverse.nii.gz'")