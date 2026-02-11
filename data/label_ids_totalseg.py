"""
TotalSegmentator label IDs with train/val split.

Labels are sorted by total volume (largest first).
The split is created with a fixed random seed (42) for reproducibility.
Train: 80% of labels, Val: 20% of labels
"""

import random

# Labels by total volume (occurrences * avg_volume) from totalseg_2d.ipynb
_all_label_ids = ['liver',
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
 'vertebrae_T10',
 'esophagus',
 'vertebrae_S1',
 'vertebrae_T9',
 'trachea',
 'vertebrae_T8',
 'vertebrae_T7',
 'iliac_vena_left',
 'vertebrae_T6',
 'vertebrae_T5',
 'clavicula_right',
 'portal_vein_and_splenic_vein',
 'vertebrae_T4',
 'clavicula_left',
 'vertebrae_T2',
 'pulmonary_vein',
 'rib_left_7',
 'rib_right_7',
 'vertebrae_T3',
 'vertebrae_T1',
 'rib_left_6',
 'rib_right_6',
 'gallbladder',
 'iliac_vena_right',
 'superior_vena_cava',
 'rib_right_8',
 'rib_left_8',
 'rib_right_9',
 'rib_left_9',
 'rib_right_5',
 'rib_left_5',
 'rib_left_4',
 'rib_right_4',
'rib_left_10',
 'rib_right_10',
 'iliac_artery_left',
 'iliac_artery_right',
 'thyroid_gland',
 'rib_left_3',
 'rib_right_3',
 'prostate',
 'vertebrae_C7',
 'brachiocephalic_vein_left',
 'rib_left_2',
 'rib_right_2',
 'rib_left_11',
 'rib_right_1',
 'rib_left_1',
 'rib_right_11',
 'subclavian_artery_left',
 'subclavian_artery_right',
 'atrial_appendage_left',
 'vertebrae_C6',
 'brachiocephalic_vein_right',
 'brachiocephalic_trunk',
 'vertebrae_C2',
 'common_carotid_artery_left',
 'vertebrae_C5',
 'vertebrae_C1',
 'adrenal_gland_left',
 'adrenal_gland_right',
 'vertebrae_C4',
 'vertebrae_C3',
 'rib_left_12',
 'rib_right_12',
 'kidney_cyst_right',
 'kidney_cyst_left',
 'common_carotid_artery_right']

# For backwards compatibility
label_ids = _all_label_ids


def _split_labels(labels, seed=42):
    """Split labels into train/val (80/20) with fixed seed."""
    rng = random.Random(seed)
    shuffled = labels.copy()
    rng.shuffle(shuffled)
    split_idx = int(len(shuffled) * 0.8)
    return shuffled[:split_idx], shuffled[split_idx:]


# Default splits using all labels (for backwards compatibility)
label_ids_train, label_ids_val = _split_labels(_all_label_ids)


def get_label_ids(split="all", max_labels=None):
    """
    Get label IDs for specified split.

    Args:
        split: One of "train", "val", or "all"
        max_labels: If provided, first take the top n labels by volume,
            then apply the train/val split to those n labels.

    Returns:
        List of label IDs for the specified split
    """
    # First, select top n labels by volume (or all if max_labels is None)
    if max_labels is not None:
        base_labels = _all_label_ids[:max_labels]
    else:
        base_labels = _all_label_ids

    # Then apply split
    if split == "all":
        return base_labels
    elif split == "train":
        train, _ = _split_labels(base_labels)
        return train
    elif split == "val":
        _, val = _split_labels(base_labels)
        return val
    else:
        raise ValueError(f"split must be 'train', 'val', or 'all', got: {split}")
