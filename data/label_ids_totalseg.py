"""
TotalSegmentator label IDs with train/val split.

The split is created with a fixed random seed (42) for reproducibility.
Train: 80% of labels, Val: 20% of labels
"""
"""
import random

label_ids = [
    "vertebrae_C5",
    "common_carotid_artery_left",
    "clavicula_right",
    "brachiocephalic_vein_left",
    "heart",
    "trachea",
    "vertebrae_T2",
    "rib_right_9",
    "rib_right_4",
    "vertebrae_L2",
    "thyroid_gland",
    "urinary_bladder",
    "vertebrae_C7",
    "humerus_left",
    "pulmonary_vein",
    "costal_cartilages",
    "spinal_cord",
    "vertebrae_T11",
    "scapula_right",
    "inferior_vena_cava",
    "prostate",
    "rib_left_2",
    "rib_right_6",
    "vertebrae_S1",
    "sacrum",
    "autochthon_right",
    "vertebrae_C1",
    "lung_lower_lobe_right",
    "subclavian_artery_right",
    "aorta",
    "rib_left_9",
    "iliac_vena_left",
    "lung_middle_lobe_right",
    "kidney_left",
    "rib_right_10",
    "iliopsoas_right",
    "vertebrae_T6",
    "rib_left_4",
    "liver",
    "humerus_right",
    "rib_left_11",
    "hip_left",
    "brachiocephalic_trunk",
    "lung_upper_lobe_left",
    "portal_vein_and_splenic_vein",
    "subclavian_artery_left",
    "vertebrae_C3",
    "gluteus_medius_right",
    "rib_right_12",
    "kidney_cyst_right",
    "rib_right_2",
    "vertebrae_T9",
    "iliac_vena_right",
    "vertebrae_L4",
    "rib_left_6",
    "vertebrae_T4",
    "pancreas",
    "vertebrae_T7",
    "rib_left_5",
    "small_bowel",
    "gallbladder",
    "rib_right_1",
    "brachiocephalic_vein_right",
    "rib_right_11",
    "superior_vena_cava",
    "rib_left_8",
    "colon",
    "lung_lower_lobe_left",
    "adrenal_gland_left",
    "autochthon_left",
    "iliac_artery_right",
    "rib_left_12",
    "stomach",
    "atrial_appendage_left",
    "rib_left_7",
    "vertebrae_T5",
    "gluteus_minimus_right",
    "vertebrae_L5",
    "vertebrae_T8",
    "clavicula_left",
    "rib_right_3",
    "adrenal_gland_right",
    "scapula_left",
    "vertebrae_C2",
    "rib_left_10",
    "lung_upper_lobe_right",
    "spleen",
    "iliac_artery_left",
    "gluteus_maximus_right",
    "vertebrae_L3",
    "rib_right_5",
    "rib_right_8",
    "gluteus_maximus_left",
    "rib_left_1",
    "vertebrae_T3",
    "vertebrae_T12",
    "femur_right",
    "esophagus",
    "gluteus_medius_left",
    "vertebrae_C4",
    "rib_right_7",
    "hip_right",
    "vertebrae_L1",
    "kidney_cyst_left",
    "gluteus_minimus_left",
    "iliopsoas_left",
    "vertebrae_T1",
    "skull",
    "sternum",
    "rib_left_3",
    "vertebrae_T10",
    "brain",
    "vertebrae_C6",
    "duodenum",
    "common_carotid_artery_right",
    "femur_left",
    "kidney_right",
]

# Top 50 labels by total volume (occurrences * avg_volume) from totalseg_2d.ipynb
label_ids = ['liver',
 'lung_upper_lobe_left',
 'lung_lower_lobe_right',
 'lung_lower_lobe_left',
 'lung_upper_lobe_right',
 'small_bowel',
 'colon',
 'heart',
 'autochthon_left',
 'autochthon_right',
 'lung_middle_lobe_right',
 'gluteus_maximus_right',
 'gluteus_maximus_left',
 'stomach',
 'hip_left',
 'hip_right',
 'aorta',
 'spleen',
 'brain',
 'iliopsoas_left',
 'iliopsoas_right',
 'gluteus_medius_left',
 'gluteus_medius_right',
 'skull',
 'costal_cartilages',
 'urinary_bladder',
 'femur_left',
 'femur_right',
 'sacrum',
 'kidney_left',
 'kidney_right',
 'spinal_cord',
 'scapula_left',
 'scapula_right',
 'inferior_vena_cava',
 'pancreas',
 'sternum',
 'vertebrae_L4',
 'vertebrae_L3',
 'vertebrae_L1',
 'vertebrae_T12',
 'vertebrae_L5',
 'vertebrae_L2',
 'vertebrae_T11',
 'duodenum',
 'gluteus_minimus_right',
 'humerus_right',
 'gluteus_minimus_left',
 'humerus_left',
 'vertebrae_T10']


# Train/Val split (80/20 split with fixed random seed for reproducibility)
_rng = random.Random(42)
_shuffled = label_ids.copy()
_rng.shuffle(_shuffled)
_split_idx = int(len(_shuffled) * 0.8)

label_ids_train = _shuffled[:_split_idx]
label_ids_val = _shuffled[_split_idx:]

# Clean up temporary variables
del _rng, _shuffled, _split_idx
"""

label_ids_train = [
    'liver',
    #'lung_upper_lobe_left',
    #'lung_lower_lobe_left',
    'lung_upper_lobe_right',
    'lung_lower_lobe_right',
    'lung_middle_lobe_right',
    'heart',
    'aorta',
    'spleen',
    'brain',
    'skull',
    #'kidney_left',
    'kidney_right',
    'urinary_bladder',
    'sacrum',
    'costal_cartilages'
]

label_ids_val = [
    'small_bowel',
    'colon',
    'stomach',
    'autochthon_left',
    #'autochthon_right',
    'gluteus_maximus_left',
    #'gluteus_maximus_right',
    'gluteus_medius_left',
    #'gluteus_medius_right',
    'iliopsoas_left',
    #'iliopsoas_right',
    'hip_left',
    #'hip_right',
    'femur_left',
    #'femur_right'
]

def get_label_ids(split="all"):
    """
    Get label IDs for specified split.

    Args:
        split: One of "train", "val", or "all"

    Returns:
        List of label IDs for the specified split
    """
    if split == "train":
        return label_ids_train
    elif split == "val":
        return label_ids_val
    elif split == "all":
        return label_ids
    else:
        raise ValueError(f"split must be 'train', 'val', or 'all', got: {split}")
