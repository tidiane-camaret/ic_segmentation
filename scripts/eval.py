### Evaluate neuroverse3D on a given list of cases
# 1) read the case list and scale the images/masks to (128, 128, 128) 
# 2) save cases as nii files in a temp folder
# 3) run inference using pretrained neuroverse3D model
# 4) compute DICE score between predicted and ground truth masks 


import os
import sys
import warnings

import numpy as np
import nibabel as nib
from nilearn.image import resample_img
from pathlib import Path
from src.config import config

project_name = "camaret___in_context_segmentation"
origin_image_dir = os.path.join(config["DATA_DIR"], "3D_val_npz")
origin_label_dir = os.path.join(config["DATA_DIR"], "3D_val_gt_interactive_seg")
tasks_dict = {
 'MR_Head_HNTSMRG24': ['MR_Head_HNTSMRG24_152_preRT_T2',
  'MR_Head_HNTSMRG24_154_midRT_T2',
  'MR_Head_HNTSMRG24_23_preRT_T2',
  'MR_Head_HNTSMRG24_77_midRT_T2',
  'MR_Head_HNTSMRG24_86_midRT_T2'],
 'MR_ISLES2022_ADC': ['MR_ISLES2022_ADC_sub-strokecase0035',
  'MR_ISLES2022_ADC_sub-strokecase0038',
  'MR_ISLES2022_ADC_sub-strokecase0077',
  'MR_ISLES2022_ADC_sub-strokecase0080',
  'MR_ISLES2022_ADC_sub-strokecase0118',
  'MR_ISLES2022_ADC_sub-strokecase0139',
  'MR_ISLES2022_ADC_sub-strokecase0200',
  'MR_ISLES2022_ADC_sub-strokecase0243'],
 'MR_ISLES2022_DWI': ['MR_ISLES2022_DWI_sub-strokecase0008',
  'MR_ISLES2022_DWI_sub-strokecase0015',
  'MR_ISLES2022_DWI_sub-strokecase0032',
  'MR_ISLES2022_DWI_sub-strokecase0086',
  'MR_ISLES2022_DWI_sub-strokecase0098',
  'MR_ISLES2022_DWI_sub-strokecase0124',
  'MR_ISLES2022_DWI_sub-strokecase0142',
  'MR_ISLES2022_DWI_sub-strokecase0190',
  'MR_ISLES2022_DWI_sub-strokecase0212',
  'MR_ISLES2022_DWI_sub-strokecase0245'],
 'MR_ProstateADC_ADC_ProstateX': ['MR_ProstateADC_ADC-ProstateX-0002',
  'MR_ProstateADC_ADC-ProstateX-0010',
  'MR_ProstateADC_ADC-ProstateX-0021',
  'MR_ProstateADC_ADC-ProstateX-0038',
  'MR_ProstateADC_ADC-ProstateX-0053',
  'MR_ProstateADC_ADC-ProstateX-0057',
  'MR_ProstateADC_ADC-ProstateX-0076',
  'MR_ProstateADC_ADC-ProstateX-0096',
  'MR_ProstateADC_ADC-ProstateX-0099',
  'MR_ProstateADC_ADC-ProstateX-0123',
  'MR_ProstateADC_ADC-ProstateX-0128',
  'MR_ProstateADC_ADC-ProstateX-0134',
  'MR_ProstateADC_ADC-ProstateX-0148',
  'MR_ProstateADC_ADC-ProstateX-0158',
  'MR_ProstateADC_ADC-ProstateX-0170',
  'MR_ProstateADC_ADC-ProstateX-0189',
  'MR_ProstateADC_ADC-ProstateX-0204',
  'MR_ProstateADC_ADC-ProstateX-0222',
  'MR_ProstateADC_ADC-ProstateX-0228',
  'MR_ProstateADC_ADC-ProstateX-0229',
  'MR_ProstateADC_ADC-ProstateX-0231',
  'MR_ProstateADC_ADC-ProstateX-0246',
  'MR_ProstateADC_ADC-ProstateX-0294',
  'MR_ProstateADC_ADC-ProstateX-0323',
  'MR_ProstateADC_ADC-ProstateX-0324',
  'MR_ProstateADC_ADC-ProstateX-0339',
  'MR_ProstateADC_ADC-ProstateX-0340'],
 'MR_ProstateT2': ['MR_ProstateT2_NCI-Prostate3T-01-0018',
  'MR_ProstateT2_NCI-Prostate3T-01-0023',
  'MR_ProstateT2_NCI-Prostate3T-01-0027',
  'MR_ProstateT2_T2-MSD-prostate_04',
  'MR_ProstateT2_T2-MSD-prostate_26',
  'MR_ProstateT2_T2-MSD-prostate_38',
  'MR_ProstateT2_T2-MSD-prostate_39',
  'MR_ProstateT2_T2-MSD-prostate_41',
  'MR_ProstateT2_T2-ProstateX-0016',
  'MR_ProstateT2_T2-ProstateX-0024',
  'MR_ProstateT2_T2-ProstateX-0040',
  'MR_ProstateT2_T2-ProstateX-0062',
  'MR_ProstateT2_T2-ProstateX-0074',
  'MR_ProstateT2_T2-ProstateX-0085',
  'MR_ProstateT2_T2-ProstateX-0086',
  'MR_ProstateT2_T2-ProstateX-0099',
  'MR_ProstateT2_T2-ProstateX-0105',
  'MR_ProstateT2_T2-ProstateX-0107',
  'MR_ProstateT2_T2-ProstateX-0113',
  'MR_ProstateT2_T2-ProstateX-0120',
  'MR_ProstateT2_T2-ProstateX-0140',
  'MR_ProstateT2_T2-ProstateX-0152',
  'MR_ProstateT2_T2-ProstateX-0158',
  'MR_ProstateT2_T2-ProstateX-0177',
  'MR_ProstateT2_T2-ProstateX-0189',
  'MR_ProstateT2_T2-ProstateX-0203',
  'MR_ProstateT2_T2-ProstateX-0206_5',
  'MR_ProstateT2_T2-ProstateX-0214_5',
  'MR_ProstateT2_T2-ProstateX-0221_4',
  'MR_ProstateT2_T2-ProstateX-0230_4',
  'MR_ProstateT2_T2-ProstateX-0243_4',
  'MR_ProstateT2_T2-ProstateX-0244_4',
  'MR_ProstateT2_T2-ProstateX-0249_4',
  'MR_ProstateT2_T2-ProstateX-0251_5'],
 'MR_QIN_PROSTATE_Lesion_PCAMPMRI': ['MR_QIN-PROSTATE-Lesion_PCAMPMRI-00003_0_1_MR',
  'MR_QIN-PROSTATE-Lesion_PCAMPMRI-00007_0_0_MR',
  'MR_QIN-PROSTATE-Lesion_PCAMPMRI-00010_0_0_MR',
  'MR_QIN-PROSTATE-Lesion_PCAMPMRI-00012_1_0_MR',
  'MR_QIN-PROSTATE-Lesion_PCAMPMRI-00012_1_2_MR',
  'MR_QIN-PROSTATE-Lesion_PCAMPMRI-00013_1_0_MR'],
 'MR_QIN_PROSTATE_Prostate_PCAMPMRI': ['MR_QIN-PROSTATE-Prostate_PCAMPMRI-00001_0_2_MR',
  'MR_QIN-PROSTATE-Prostate_PCAMPMRI-00005_1_2_MR',
  'MR_QIN-PROSTATE-Prostate_PCAMPMRI-00010_1_1_MR',
  'MR_QIN-PROSTATE-Prostate_PCAMPMRI-00011_1_1_MR',
  'MR_QIN-PROSTATE-Prostate_PCAMPMRI-00011_1_2_MR',
  'MR_QIN-PROSTATE-Prostate_PCAMPMRI-00012_1_1_MR',
  'MR_QIN-PROSTATE-Prostate_PCAMPMRI-00014_0_2_MR',
  'MR_QIN-PROSTATE-Prostate_PCAMPMRI-00015_0_0_MR'],
 'MR_Spider': ['MR_Spider_101_t2_spi',
  'MR_Spider_117_t1_spi',
  'MR_Spider_118_t2_spi',
  'MR_Spider_11_t2_spi',
  'MR_Spider_121_t1_spi',
  'MR_Spider_131_t1_spi',
  'MR_Spider_132_t1_spi',
  'MR_Spider_142_t1_spi',
  'MR_Spider_142_t2_spi',
  'MR_Spider_143_t1_spi',
  'MR_Spider_147_t2_spi',
  'MR_Spider_152_t2_SPACE_spi',
  'MR_Spider_166_t1_spi',
  'MR_Spider_170_t1_spi',
  'MR_Spider_172_t1_spi',
  'MR_Spider_190_t2_spi',
  'MR_Spider_191_t2_SPACE_spi',
  'MR_Spider_198_t1_spi',
  'MR_Spider_1_t1_spi',
  'MR_Spider_214_t2_spi',
  'MR_Spider_217_t2_spi',
  'MR_Spider_229_t2_spi',
  'MR_Spider_234_t1_spi',
  'MR_Spider_237_t2_spi',
  'MR_Spider_239_t2_SPACE_spi',
  'MR_Spider_23_t2_spi',
  'MR_Spider_252_t1_spi',
  'MR_Spider_29_t1_spi',
  'MR_Spider_31_t2_spi',
  'MR_Spider_36_t2_spi',
  'MR_Spider_44_t1_spi',
  'MR_Spider_45_t1_spi',
  'MR_Spider_4_t1_spi',
  'MR_Spider_50_t2_SPACE_spi',
  'MR_Spider_52_t1_spi',
  'MR_Spider_56_t1_spi',
  'MR_Spider_59_t2_spi',
  'MR_Spider_61_t1_spi',
  'MR_Spider_62_t1_spi',
  'MR_Spider_67_t2_spi',
  'MR_Spider_77_t1_spi',
  'MR_Spider_78_t2_spi',
  'MR_Spider_80_t2_spi',
  'MR_Spider_94_t1_spi'],
 'MR_heart': ['MR_heart-ACDC_ACDC_patient006_frame01_myo',
  'MR_heart-ACDC_ACDC_patient017_frame09_myo',
  'MR_heart-ACDC_ACDC_patient019_frame11_myo',
  'MR_heart-ACDC_ACDC_patient022_frame11_myo',
  'MR_heart-ACDC_ACDC_patient023_frame01_myo',
  'MR_heart-ACDC_ACDC_patient027_frame01_myo',
  'MR_heart-ACDC_ACDC_patient027_frame11_myo',
  'MR_heart-ACDC_ACDC_patient044_frame11_myo',
  'MR_heart-ACDC_ACDC_patient055_frame01_myo',
  'MR_heart-ACDC_ACDC_patient057_frame09_myo',
  'MR_heart-ACDC_ACDC_patient061_frame01_myo',
  'MR_heart-ACDC_ACDC_patient070_frame01_myo',
  'MR_heart-ACDC_ACDC_patient074_frame01_myo',
  'MR_heart-ACDC_ACDC_patient076_frame01_myo',
  'MR_heart-ACDC_ACDC_patient077_frame09_myo',
  'MR_heart-ACDC_ACDC_patient083_frame01_myo',
  'MR_heart-ACDC_ACDC_patient091_frame01_myo',
  'MR_heart-ACDC_ACDC_patient096_frame08_myo',
  'MR_heart-ACDC_ACDC_patient098_frame01_myo',
  'MR_heart-ACDC_ACDC_patient104_frame01_myo',
  'MR_heart-ACDC_ACDC_patient107_frame10_myo',
  'MR_heart-ACDC_ACDC_patient110_frame11_myo',
  'MR_heart-ACDC_ACDC_patient114_frame11_myo',
  'MR_heart-ACDC_ACDC_patient116_frame01_myo',
  'MR_heart-ACDC_ACDC_patient119_frame09_myo',
  'MR_heart-ACDC_ACDC_patient120_frame08_myo',
  'MR_heart-ACDC_ACDC_patient123_frame11_myo',
  'MR_heart-ACDC_ACDC_patient124_frame01_myo',
  'MR_heart-ACDC_ACDC_patient137_frame01_myo',
  'MR_heart-ACDC_ACDC_patient146_frame10_myo'],
 'PET_autoPET_fdg': ['PET_autoPET_fdg_43647ff727_03-02-2006-NA-PET-CT Ganzkoerper  primaer mit KM-05703',
  'PET_autoPET_fdg_68f73c4518_11-13-2004-NA-PET-CT Ganzkoerper  primaer mit KM-36034',
  'PET_autoPET_fdg_802f19931c_06-30-2002-NA-PET-CT Ganzkoerper  primaer mit KM-43581',
  'PET_autoPET_fdg_de118d7ab9_06-02-2001-NA-PET-CT Ganzkoerper  primaer mit KM-18077',
  'PET_autoPET_fdg_ebb0045704_09-27-2002-NA-Unspecified CT-08213']}
target_shape = (128, 128, 128)

# load model
import torch

sys.path.append("/software/notebooks/camaret/repos/Neuroverse3D")
from neuroverse3D.lightning_model import LightningModel
from utils.dataloading import *
from utils.task_synthesis import *
device = "cuda:0"
checkpoint_path = '/software/notebooks/camaret/repos/Neuroverse3D/checkpoint/neuroverse3D.ckpt'
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
hparams = checkpoint['hyper_parameters']
# load model
import warnings
warnings.filterwarnings('ignore')
model = LightningModel.load_from_checkpoint(checkpoint_path, map_location=torch.device(device))
print('Load checkpoint from:', checkpoint_path)
total_params = sum(p.numel() for p in model.parameters())
print("Total number of parameters: ", total_params)

    # Load data
from utils.dataloading import *
from utils.task_synthesis import *

for task_name, case_list in tasks_dict.items():
    # remove underscore from task_name
    task_name = task_name.replace("_", "")
    export_dir = Path(config["RESULTS_DIR"]) / task_name
    export_dir.mkdir(exist_ok=True)
    img_dir = export_dir / 'imgs'
    lab_dir = export_dir / 'labs'
    img_dir.mkdir(exist_ok=True)
    lab_dir.mkdir(exist_ok=True)

    # copy and resize images/masks to export dir
    case_list = case_list[:3]  # for testing, only use first 3 cases
    for case_name in case_list:
  
        print("Processing case: ", case_name)
        case_num = case_name[-8:]
        # remove underscores from case_num
        case_num = case_num.replace("_", "")
        print("Case number: ", case_num)

        # Use the original img_dir and gts_dir variables for loading
        img_path = os.path.join(origin_image_dir, case_name + ".npz")
        img_data = np.load(img_path)
        img = img_data["imgs"]
        img_spacing = img_data["spacing"]
        img_spacing = img_spacing[::-1]  # reverse spacing order
        
        gt_path = os.path.join(origin_label_dir, case_name + ".npz")
        gt_data = np.load(gt_path)
        gt = gt_data["gts"]
        gt_spacing = gt_data["spacing"]
        gt_spacing = gt_spacing[::-1]  # reverse spacing order
        
        print("Image shape: ", img.shape)
        print("GT shape: ", gt.shape)
        print(f"classes in GT : {np.unique(gt)}")

        img_nii = nib.Nifti1Image(img, affine=np.diag(np.append(img_spacing, 1)))
        gt_nii = nib.Nifti1Image(gt, affine=np.diag(np.append(gt_spacing, 1)))

        
        # Calculate zoom factors
        zoom_factors = np.array(img.shape) / np.array(target_shape)
        
        # New spacing = old spacing * zoom factors
        new_spacing = img_spacing * zoom_factors
        target_affine = np.diag(np.append(new_spacing, 1))
        
        resampled_img = resample_img(img_nii, 
                                    target_affine=target_affine, 
                                    target_shape=target_shape, 
                                    interpolation='continuous',
                                    force_resample=True)
        
        resampled_gt = resample_img(gt_nii, 
                                    target_affine=target_affine, 
                                    target_shape=target_shape, 
                                    interpolation='nearest',
                                    force_resample=True)

        # Save to output directories
        nib.save(resampled_img, os.path.join(img_dir, task_name + "_" + case_num + "_img.nii.gz"))
        nib.save(resampled_gt, os.path.join(lab_dir, task_name + "_" + case_num + "_gt_all.nii.gz"))

    # Match everything up to the last underscore before the file suffix
    # Match only the filename part (after last /)
    img_regex = r"(?<patients_id>[^/_]+)_(?<studies_id>[^_]+)_img\.nii\.gz"
    lab_regex = r"(?<patients_id>[^/_]+)_(?<studies_id>[^_]+)_gt_all\.nii\.gz"

    os.system(f"nora -p {project_name} --importfiles {img_dir} '{img_regex}'")
    os.system(f"nora -p {project_name} --importfiles {lab_dir} '{lab_regex}'")
    os.system(f"nora -p {project_name} --addtag mask --select '*' '*gt_all.nii.gz'")

    ### Run inference using neuroverse3D model

    images, labels = load_seg_data(img_dir, lab_dir) # load data
    size_check(images), size_check(labels) # check size

    print('Shape of images:',images.shape, '\nShape of labels:',labels.shape)

    target_in, context_in, target_out_raw, context_out_raw = structure_data(images, labels, index = 0, verbose = True)

    unique_masks = np.unique(target_out_raw) 
    print('Unique classes in the mask:', unique_masks)
    seg_class = [1]
    context_out = np.isin(context_out_raw, seg_class).astype(np.float32)
    target_out = np.isin(target_out_raw, seg_class).astype(np.float32)

    # Normalization
    target_out = normalize_3d_volume(torch.tensor(target_out).to(device))
    target_in = normalize_3d_volume(torch.tensor(target_in).to(device))
    context_in = normalize_3d_volume(torch.tensor(context_in))
    context_out = normalize_3d_volume(torch.tensor(context_out))

    # Run inference
    with torch.no_grad():
        mask = model.forward(target_in, context_in, context_out, gs = 2) # gs control the size of mini-context

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
    nib.save(mask_nii, lab_pred_dir / f'{first_case_name}_pred.nii.gz')
    nib.save(gt_nii, lab_pred_dir / f'{first_case_name}_gt.nii.gz')

    # save to nora project 
    # Match everything up to the last underscore before the file suffix
    # Match only the filename part (after last /)
    lab_regex = r"(?<patients_id>[^/_]+)_(?<studies_id>[^_]+)_pred\.nii\.gz"
    os.system(f"nora -p {project_name} --importfiles {lab_pred_dir} '{lab_regex}'")
    os.system(f"nora -p {project_name} --addtag mask --select '*' '*pred.nii.gz'")

    lab_regex = r"(?<patients_id>[^/_]+)_(?<studies_id>[^_]+)_gt\.nii\.gz"
    os.system(f"nora -p {project_name} --importfiles {lab_pred_dir} '{lab_regex}'")
    os.system(f"nora -p {project_name} --addtag mask --select '*' '*gt.nii.gz'")


    # save dsc and nsd to a df  
    import pandas as pd
    results_path = Path(config["RESULTS_DIR"]) / 'evaluation_results.csv'

    # Load existing results or create new DataFrame
    if results_path.exists():
        results_df = pd.read_csv(results_path)
    else:
        results_df = pd.DataFrame(columns=['TaskName', 'DSC', 'NSD'])

    # Create new row as DataFrame
    new_row = pd.DataFrame([{'TaskName': task_name, 'DSC': dsc, 'NSD': nsd}])

    # Concatenate
    results_df = pd.concat([results_df, new_row], ignore_index=True)

    # Save
    results_df.to_csv(results_path, index=False)