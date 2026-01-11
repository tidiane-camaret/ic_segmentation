    # Import images and masks to nora project
import os 
from src.config import config
from pathlib import Path
dir_to_export = Path(config["RESULTS_DIR"]) 
patient_name = "totalseg_inspection" # "totalseg_eval"

project_name = "camaret___in_context_segmentation"

pattern = rf'(?P<patients_id>{patient_name})/(?P<studies_id>s\d{{4}})/.*_img\.nii\.gz$'
os.system(f"nora -p {project_name} --importfiles {dir_to_export} '{pattern}'")

pattern = rf'(?P<patients_id>{patient_name})/(?P<studies_id>s\d{{4}})/.*_gt_mask\.nii\.gz$'
os.system(f"nora -p {project_name} --importfiles {dir_to_export} '{pattern}'")

pattern = rf'(?P<patients_id>{patient_name})/(?P<studies_id>s\d{{4}})/.*_pred_mask\.nii\.gz$'
os.system(f"nora -p {project_name} --importfiles {dir_to_export} '{pattern}'")

pattern = rf'(?P<patients_id>{patient_name})/(?P<studies_id>s\d{{4}})/.*_context_img\.nii\.gz$'
os.system(f"nora -p {project_name} --importfiles {dir_to_export} '{pattern}'")

pattern = rf'(?P<patients_id>{patient_name})/(?P<studies_id>s\d{{4}})/.*_context_gt_mask\.nii\.gz$'
os.system(f"nora -p {project_name} --importfiles {dir_to_export} '{pattern}'")

os.system(f"nora -p {project_name} --addtag mask --select '*' '*_gt_mask.nii.gz'")
os.system(f"nora -p {project_name} --addtag mask --select '*' '*_pred_mask.nii.gz'")
os.system(f"nora -p {project_name} --addtag mask --select '*' '*_context_gt_mask.nii.gz'")