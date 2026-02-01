"""Export images and masks to nora project."""
import os
import shutil
import tempfile
from pathlib import Path

from hydra import compose, initialize_config_dir

# Initialize Hydra with the config directory
config_path = Path(__file__).parent.parent / "configs"
initialize_config_dir(config_dir=str(config_path.absolute()), version_base=None)
cfg = compose(config_name="train")

dir_to_export = Path(cfg.paths.RESULTS_DIR) 
cfg.dataset = "totalsegmri2d"
patient_name = cfg.dataset + "_" + cfg.method

project_name = "camaret___in_context_segmentation"

# WORKAROUND: NORA's Node.js code uses recursive directory traversal that overflows
# with large directories. Create a temp dir with symlink to only our patient folder.
patient_dir = dir_to_export / patient_name

with tempfile.TemporaryDirectory() as tmpdir:
    # Create symlink to just our patient folder
    symlink_path = Path(tmpdir) / patient_name
    symlink_path.symlink_to(patient_dir)
    
    pattern = rf'(?P<patients_id>{patient_name})/(?P<studies_id>[^/]+)/[^/]*img\.nii\.gz$'
    os.system(f"nora -p {project_name} --importfiles {tmpdir} '{pattern}'")

    pattern = rf'(?P<patients_id>{patient_name})/(?P<studies_id>[^/]+)/[^/]*mask\.nii\.gz$'
    os.system(f"nora -p {project_name} --importfiles {tmpdir} '{pattern}'")


os.system(f"nora -p {project_name} --addtag mask --select '{patient_name}/s*/*gt_mask.nii.gz'")
os.system(f"nora -p {project_name} --addtag mask --select '{patient_name}/s*/*pred_mask.nii.gz'")
os.system(f"nora -p {project_name} --addtag mask --select '{patient_name}/s*/*context_gt_mask.nii.gz'")
