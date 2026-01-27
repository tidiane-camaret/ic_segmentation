"""Export images and masks to nora project."""
import os
from pathlib import Path

from hydra import compose, initialize_config_dir

# Initialize Hydra with the config directory
config_path = Path(__file__).parent.parent / "configs"
initialize_config_dir(config_dir=str(config_path.absolute()), version_base=None)
cfg = compose(config_name="train")

dir_to_export = Path(cfg.paths.RESULTS_DIR)
patient_name = cfg.dataset

project_name = "camaret___in_context_segmentation"

pattern = rf'(?P<patients_id>{patient_name})/(?P<studies_id>[^/]+)/.*img\.nii\.gz$'
os.system(f"nora -p {project_name} --importfiles {dir_to_export} '{pattern}'")

pattern = rf'(?P<patients_id>{patient_name})/(?P<studies_id>[^/]+)/.*mask\.nii\.gz$'
os.system(f"nora -p {project_name} --importfiles {dir_to_export} '{pattern}'")


os.system(f"nora -p {project_name} --addtag mask --select '{patient_name}/s*/*gt_mask.nii.gz'")
os.system(f"nora -p {project_name} --addtag mask --select '{patient_name}/s*/*pred_mask.nii.gz'")
os.system(f"nora -p {project_name} --addtag mask --select '{patient_name}/s*/*context_gt_mask.nii.gz'")
