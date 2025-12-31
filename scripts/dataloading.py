# 1) Load the SegFM3D cases from tasks_dict
# 2) save img and labels as nii in config["RESULTS_DIR"]
# 3) add files to nora project

import os

import numpy as np
import nibabel as nib
from pathlib import Path
from src.config import config
from scripts.tasks_dict import tasks_dict

project_name = "camaret___in_context_segmentation"
origin_image_dir = os.path.join(config["DATA_DIR"], "3D_val_npz")
origin_label_dir = os.path.join(config["DATA_DIR"], "3D_val_gt_interactive_seg")


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

    for case_name in case_list:
  
        print("Processing case: ", case_name)
        case_num = case_name.replace("_", "")[-8:]
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


        # Save to output directories
        nib.save(img_nii, os.path.join(img_dir, case_num + "_img.nii.gz"))
        nib.save(gt_nii, os.path.join(lab_dir, case_num + "_gt_all.nii.gz"))

    # add image and label files to nora project
    img_regex = r"(?<patients_id>[^/_]+)_(?<studies_id>[^_]+)_img\.nii\.gz"
    lab_regex = r"(?<patients_id>[^/_]+)_(?<studies_id>[^_]+)_gt_all\.nii\.gz"

    os.system(f"nora -p {project_name} --importfiles {img_dir} '{img_regex}'")
    os.system(f"nora -p {project_name} --importfiles {lab_dir} '{lab_regex}'")

os.system(f"nora -p {project_name} --addtag mask --select '*' '*gt_all.nii.gz'")