"""
TotalSegmentator label IDs with train/val split.

Labels are sorted by total volume (largest first).
The split is created with a fixed random seed (42) for reproducibility.
"""

from pathlib import Path
from typing import cast

import pandas as pd
from omegaconf import OmegaConf


def _get_data_dir_from_config() -> Path:
    """Load DATA_DIR from cluster config based on current environment."""
    config_dir = Path(__file__).parent.parent / "configs" / "cluster"

    # Try each cluster config and match by root path
    for config_file in config_dir.glob("*.yaml"):
        try:
            cfg = OmegaConf.load(config_file)
            if "paths" in cfg and "root" in cfg.paths:
                root = Path(cfg.paths.root)
                # Check if we're in this cluster's root directory
                if root.exists() and str(Path(__file__).resolve()).startswith(
                    str(root)
                ):
                    # Resolve DATA_DIR (handles ${.root} interpolation)
                    full_cfg = cast(dict, OmegaConf.to_container(cfg, resolve=True))
                    return Path(full_cfg["paths"]["DATA_DIR"])
        except Exception:
            continue

    # Fallback to default
    return Path("/nfs/data/nii/data1/Analysis/camaret___in_context_segmentation/ANALYSIS_20251122/data")


# Labels by total volume (occurrences * avg_volume) from totalseg_2d.ipynb

_all_label_ids = [
    "liver",
    "lung_upper_lobe_left",
    "lung_lower_lobe_right",
    "lung_lower_lobe_left",
    "lung_upper_lobe_right",
    "small_bowel",
    "colon",
    "heart",
    "autochthon_left",
    "autochthon_right",
    "lung_middle_lobe_right",
    "gluteus_maximus_right",
    "gluteus_maximus_left",
    "stomach",
    "hip_left",
    "hip_right",
    "aorta",
    "spleen",
    "brain",
    "iliopsoas_left",
    "iliopsoas_right",
    "gluteus_medius_left",
    "gluteus_medius_right",
    "skull",
    "costal_cartilages",
    "urinary_bladder",
    "femur_left",
    "femur_right",
    "sacrum",
    "kidney_left",
    "kidney_right",
    "spinal_cord",
    "scapula_left",
    "scapula_right",
    "inferior_vena_cava",
    "pancreas",
    "sternum",
    "vertebrae_L4",
    "vertebrae_L3",
    "vertebrae_L1",
    "vertebrae_T12",
    "vertebrae_L5",
    "vertebrae_L2",
    "vertebrae_T11",
    "duodenum",
    "gluteus_minimus_right",
    "humerus_right",
    "gluteus_minimus_left",
    "humerus_left",
    "vertebrae_T10",
    "esophagus",
    "vertebrae_S1",
    "vertebrae_T9",
    "trachea",
    "vertebrae_T8",
    "vertebrae_T7",
    "iliac_vena_left",
    "vertebrae_T6",
    "vertebrae_T5",
    "clavicula_right",
    "portal_vein_and_splenic_vein",
    "vertebrae_T4",
    "clavicula_left",
    "vertebrae_T2",
    "pulmonary_vein",
    "rib_left_7",
    "rib_right_7",
    "vertebrae_T3",
    "vertebrae_T1",
    "rib_left_6",
    "rib_right_6",
    "gallbladder",
    "iliac_vena_right",
    "superior_vena_cava",
    "rib_right_8",
    "rib_left_8",
    "rib_right_9",
    "rib_left_9",
    "rib_right_5",
    "rib_left_5",
    "rib_left_4",
    "rib_right_4",
    "rib_left_10",
    "rib_right_10",
    "iliac_artery_left",
    "iliac_artery_right",
    "thyroid_gland",
    "rib_left_3",
    "rib_right_3",
    "prostate",
    "vertebrae_C7",
    "brachiocephalic_vein_left",
    "rib_left_2",
    "rib_right_2",
    "rib_left_11",
    "rib_right_1",
    "rib_left_1",
    "rib_right_11",
    "subclavian_artery_left",
    "subclavian_artery_right",
    "atrial_appendage_left",
    "vertebrae_C6",
    "brachiocephalic_vein_right",
    "brachiocephalic_trunk",
    "vertebrae_C2",
    "common_carotid_artery_left",
    "vertebrae_C5",
    "vertebrae_C1",
    "adrenal_gland_left",
    "adrenal_gland_right",
    "vertebrae_C4",
    "vertebrae_C3",
    "rib_left_12",
    "rib_right_12",
    "kidney_cyst_right",
    "kidney_cyst_left",
    "common_carotid_artery_right",
]


category_map = {
    # --- ORGANS (ABDOMINAL & PELVIC) ---
    "esophagus": "Organs (Abd/Pelvis)",
    "stomach": "Organs (Abd/Pelvis)",
    "duodenum": "Organs (Abd/Pelvis)",
    "small_bowel": "Organs (Abd/Pelvis)",
    "colon": "Organs (Abd/Pelvis)",
    "liver": "Organs (Abd/Pelvis)",
    "gallbladder": "Organs (Abd/Pelvis)",
    "pancreas": "Organs (Abd/Pelvis)",
    "spleen": "Organs (Abd/Pelvis)",
    "kidney_left": "Organs (Abd/Pelvis)",
    "kidney_right": "Organs (Abd/Pelvis)",
    "urinary_bladder": "Organs (Abd/Pelvis)",
    "prostate": "Organs (Abd/Pelvis)",
    "adrenal_gland_right": "Organs (Abd/Pelvis)",
    "adrenal_gland_left": "Organs (Abd/Pelvis)",
    "kidney_cyst_left": "Organs (Abd/Pelvis)",
    "kidney_cyst_right": "Organs (Abd/Pelvis)",
    # --- ORGANS (THORAX, HEAD & SPINE) ---
    "heart": "Organs (Thorax/Head/Spine)",
    "lung_upper_lobe_left": "Organs (Thorax/Head/Spine)",
    "lung_lower_lobe_left": "Organs (Thorax/Head/Spine)",
    "lung_upper_lobe_right": "Organs (Thorax/Head/Spine)",
    "lung_middle_lobe_right": "Organs (Thorax/Head/Spine)",
    "lung_lower_lobe_right": "Organs (Thorax/Head/Spine)",
    "trachea": "Organs (Thorax/Head/Spine)",
    "thyroid_gland": "Organs (Thorax/Head/Spine)",
    "brain": "Organs (Thorax/Head/Spine)",
    "spinal_cord": "Organs (Thorax/Head/Spine)",
    "atrial_appendage_left": "Organs (Thorax/Head/Spine)",
    # --- BONES (SPINE) ---
    "vertebrae_C7": "Bones (Spine)",
    "vertebrae_T1": "Bones (Spine)",
    "vertebrae_T2": "Bones (Spine)",
    "vertebrae_T3": "Bones (Spine)",
    "vertebrae_T4": "Bones (Spine)",
    "vertebrae_T5": "Bones (Spine)",
    "vertebrae_T6": "Bones (Spine)",
    "vertebrae_T7": "Bones (Spine)",
    "vertebrae_T8": "Bones (Spine)",
    "vertebrae_T9": "Bones (Spine)",
    "vertebrae_T10": "Bones (Spine)",
    "vertebrae_T11": "Bones (Spine)",
    "vertebrae_T12": "Bones (Spine)",
    "vertebrae_L1": "Bones (Spine)",
    "vertebrae_L2": "Bones (Spine)",
    "vertebrae_L3": "Bones (Spine)",
    "vertebrae_L4": "Bones (Spine)",
    "vertebrae_L5": "Bones (Spine)",
    "vertebrae_S1": "Bones (Spine)",
    "sacrum": "Bones (Spine)",
    "vertebrae_C1": "Bones (Spine)",
    "vertebrae_C2": "Bones (Spine)",
    "vertebrae_C3": "Bones (Spine)",
    "vertebrae_C4": "Bones (Spine)",
    "vertebrae_C5": "Bones (Spine)",
    "vertebrae_C6": "Bones (Spine)",
    # --- BONES (RIBS & STERNUM) ---
    "sternum": "Bones (Ribs/Sternum)",
    "costal_cartilages": "Bones (Ribs/Sternum)",
    "rib_left_1": "Bones (Ribs/Sternum)",
    "rib_left_2": "Bones (Ribs/Sternum)",
    "rib_left_3": "Bones (Ribs/Sternum)",
    "rib_left_4": "Bones (Ribs/Sternum)",
    "rib_left_5": "Bones (Ribs/Sternum)",
    "rib_left_6": "Bones (Ribs/Sternum)",
    "rib_left_7": "Bones (Ribs/Sternum)",
    "rib_left_8": "Bones (Ribs/Sternum)",
    "rib_left_9": "Bones (Ribs/Sternum)",
    "rib_left_10": "Bones (Ribs/Sternum)",
    "rib_left_11": "Bones (Ribs/Sternum)",
    "rib_right_1": "Bones (Ribs/Sternum)",
    "rib_right_2": "Bones (Ribs/Sternum)",
    "rib_right_3": "Bones (Ribs/Sternum)",
    "rib_right_4": "Bones (Ribs/Sternum)",
    "rib_right_5": "Bones (Ribs/Sternum)",
    "rib_right_6": "Bones (Ribs/Sternum)",
    "rib_right_7": "Bones (Ribs/Sternum)",
    "rib_right_8": "Bones (Ribs/Sternum)",
    "rib_right_9": "Bones (Ribs/Sternum)",
    "rib_right_10": "Bones (Ribs/Sternum)",
    "rib_right_11": "Bones (Ribs/Sternum)",
    "rib_left_12": "Bones (Ribs/Sternum)",
    "rib_right_12": "Bones (Ribs/Sternum)",
    # --- BONES (LIMBS, SHOULDER & PELVIS) ---
    "skull": "Bones (Limbs/Shoulder/Pelvis)",
    "clavicula_left": "Bones (Limbs/Shoulder/Pelvis)",
    "clavicula_right": "Bones (Limbs/Shoulder/Pelvis)",
    "scapula_left": "Bones (Limbs/Shoulder/Pelvis)",
    "scapula_right": "Bones (Limbs/Shoulder/Pelvis)",
    "humerus_left": "Bones (Limbs/Shoulder/Pelvis)",
    "humerus_right": "Bones (Limbs/Shoulder/Pelvis)",
    "hip_left": "Bones (Limbs/Shoulder/Pelvis)",
    "hip_right": "Bones (Limbs/Shoulder/Pelvis)",
    "femur_left": "Bones (Limbs/Shoulder/Pelvis)",
    "femur_right": "Bones (Limbs/Shoulder/Pelvis)",
    # --- MUSCLES ---
    "autochthon_left": "Muscles",
    "autochthon_right": "Muscles",
    "iliopsoas_left": "Muscles",
    "iliopsoas_right": "Muscles",
    "gluteus_maximus_left": "Muscles",
    "gluteus_maximus_right": "Muscles",
    "gluteus_medius_left": "Muscles",
    "gluteus_medius_right": "Muscles",
    "gluteus_minimus_left": "Muscles",
    "gluteus_minimus_right": "Muscles",
    # --- VESSELS ---
    "aorta": "Vessels",
    "iliac_artery_left": "Vessels",
    "iliac_artery_right": "Vessels",
    "subclavian_artery_left": "Vessels",
    "subclavian_artery_right": "Vessels",
    "superior_vena_cava": "Vessels",
    "inferior_vena_cava": "Vessels",
    "brachiocephalic_vein_left": "Vessels",
    "iliac_vena_left": "Vessels",
    "iliac_vena_right": "Vessels",
    "pulmonary_vein": "Vessels",
    "portal_vein_and_splenic_vein": "Vessels",
    "common_carotid_artery_left": "Vessels",
    "common_carotid_artery_right": "Vessels",
    "brachiocephalic_vein_right": "Vessels",
    "brachiocephalic_trunk": "Vessels",
}

category_map_mri = {
    # --- ORGANS (ABDOMINAL & PELVIC) ---
    "esophagus": "Organs (Abd/Pelvis)",
    "stomach": "Organs (Abd/Pelvis)",
    "duodenum": "Organs (Abd/Pelvis)",
    "small_bowel": "Organs (Abd/Pelvis)",
    "colon": "Organs (Abd/Pelvis)",
    "liver": "Organs (Abd/Pelvis)",
    "gallbladder": "Organs (Abd/Pelvis)",
    "pancreas": "Organs (Abd/Pelvis)",
    "spleen": "Organs (Abd/Pelvis)",
    "kidney_left": "Organs (Abd/Pelvis)",
    "kidney_right": "Organs (Abd/Pelvis)",
    "urinary_bladder": "Organs (Abd/Pelvis)",
    "prostate": "Organs (Abd/Pelvis)",
    "adrenal_gland_left": "Organs (Abd/Pelvis)",
    "adrenal_gland_right": "Organs (Abd/Pelvis)",
    # --- ORGANS (THORAX & HEAD/SPINE) ---
    "heart": "Organs (Thorax/Head/Spine)",
    "lung_left": "Organs (Thorax/Head/Spine)",
    "lung_right": "Organs (Thorax/Head/Spine)",
    "brain": "Organs (Thorax/Head/Spine)",
    "spinal_cord": "Organs (Thorax/Head/Spine)",
    # --- BONES (SPINE) ---
    "vertebrae": "Bones (Spine)",
    "intervertebral_discs": "Bones (Spine)",
    "sacrum": "Bones (Spine)",
    # --- BONES (LIMBS & PELVIS) ---
    "hip_left": "Bones (Limbs/Pelvis)",
    "hip_right": "Bones (Limbs/Pelvis)",
    "femur_left": "Bones (Limbs/Pelvis)",
    "femur_right": "Bones (Limbs/Pelvis)",
    "humerus_left": "Bones (Limbs/Pelvis)",
    "humerus_right": "Bones (Limbs/Pelvis)",
    "tibia": "Bones (Limbs/Pelvis)",
    "fibula": "Bones (Limbs/Pelvis)",
    # --- MUSCLES (TRUNK) ---
    "autochthon_left": "Muscles (Trunk)",
    "autochthon_right": "Muscles (Trunk)",
    "iliopsoas_left": "Muscles (Trunk)",
    "iliopsoas_right": "Muscles (Trunk)",
    "gluteus_maximus_left": "Muscles (Trunk)",
    "gluteus_maximus_right": "Muscles (Trunk)",
    "gluteus_medius_left": "Muscles (Trunk)",
    "gluteus_medius_right": "Muscles (Trunk)",
    "gluteus_minimus_left": "Muscles (Trunk)",
    "gluteus_minimus_right": "Muscles (Trunk)",
    # --- MUSCLES (THIGH) ---
    "quadriceps_femoris_left": "Muscles (Thigh)",
    "quadriceps_femoris_right": "Muscles (Thigh)",
    "sartorius_left": "Muscles (Thigh)",
    "sartorius_right": "Muscles (Thigh)",
    "thigh_medial_compartment_left": "Muscles (Thigh)",
    "thigh_medial_compartment_right": "Muscles (Thigh)",
    "thigh_posterior_compartment_left": "Muscles (Thigh)",
    "thigh_posterior_compartment_right": "Muscles (Thigh)",
    # --- VESSELS ---
    "aorta": "Vessels",
    "iliac_artery_left": "Vessels",
    "iliac_artery_right": "Vessels",
    "inferior_vena_cava": "Vessels",
    "iliac_vena_left": "Vessels",
    "iliac_vena_right": "Vessels",
    "portal_vein_and_splenic_vein": "Vessels",
}

# Cache for label stats per modality
_label_stats_cache = {}


def _get_label_splits(modality="ct"):
    """Load and cache label train/val splits for the given modality."""
    if modality not in _label_stats_cache:
        subdir = "totalsegmri" if modality == "mri" else "totalseg"
        data_dir = _get_data_dir_from_config() / subdir
        stats_df = pd.read_csv(data_dir / "label_stats.csv", index_col="label_id")
        _label_stats_cache[modality] = {
            "train": stats_df[stats_df["split"] == "train"].index.tolist(),
            "val": stats_df[stats_df["split"] == "val"].index.tolist(),
        }
    return _label_stats_cache[modality]


def get_label_ids(split="all", max_labels=None, modality="ct"):
    """Get label IDs for a given split and modality.

    Args:
        split: "train", "val", or "all"
        max_labels: Optional limit on number of labels
        modality: "ct" or "mri"
    """
    splits = _get_label_splits(modality)
    train_labels = splits["train"]
    val_labels = splits["val"]

    if split == "all":
        label_ids = train_labels + val_labels
    elif split == "train":
        label_ids = train_labels
    elif split == "val":
        label_ids = val_labels
    else:
        raise ValueError(f"split must be 'train', 'val', or 'all', got: {split}")

    if max_labels is not None:
        label_ids = label_ids[:max_labels]
    return label_ids
