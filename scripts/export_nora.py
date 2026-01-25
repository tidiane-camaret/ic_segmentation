    # Import images and masks to nora project
import os
from pathlib import Path

from ic_segmentation.old.config import load_config
config = load_config()
dir_to_export = Path(config["paths"]["RESULTS_DIR"]) 
patient_name = config["train"]["dataset"]

project_name = "camaret___in_context_segmentation"

pattern = rf'(?P<patients_id>{patient_name})/(?P<studies_id>s\d{{4}})/.*img\.nii\.gz$'
os.system(f"nora -p {project_name} --importfiles {dir_to_export} '{pattern}'")

pattern = rf'(?P<patients_id>{patient_name})/(?P<studies_id>s\d{{4}})/.*gt_mask\.nii\.gz$'
os.system(f"nora -p {project_name} --importfiles {dir_to_export} '{pattern}'")

pattern = rf'(?P<patients_id>{patient_name})/(?P<studies_id>s\d{{4}})/.*pred_mask\.nii\.gz$'
os.system(f"nora -p {project_name} --importfiles {dir_to_export} '{pattern}'")

os.system(f"nora -p {project_name} --addtag mask --select '{patient_name}/s*/*gt_mask.nii.gz'")
os.system(f"nora -p {project_name} --addtag mask --select '{patient_name}/s*/*pred_mask.nii.gz'")
os.system(f"nora -p {project_name} --addtag mask --select '{patient_name}/s*/*context_gt_mask.nii.gz'")