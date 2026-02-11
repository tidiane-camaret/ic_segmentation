"""
TotalSegmentator 2D DataLoader

Simplified dataloader for 2D slices extracted from TotalSegmentator.
Supports in-context learning by sampling k context examples with the same label.

Label splits (train/val) are defined in data/label_ids_totalseg.py.
"""

import pickle
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import albumentations as A
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from data.label_ids_totalseg import get_label_ids
from src.dataloaders.augmentations import (
    create_augmentation_transforms,
    carve_mix_2d,
    foreground_random_crop,
    perturb_mask,
    degrade_resolution,
    random_intensity_shift,
)


def define_colors_by_mean_sep(num_colors=133, channelsep=7):
    num_sep_per_channel = channelsep
    separation_per_channel = 256 // num_sep_per_channel

    color_dict = {}
    for location in range(num_colors):
        num_seq_r = location // num_sep_per_channel ** 2
        num_seq_g = (location % num_sep_per_channel ** 2) // num_sep_per_channel
        num_seq_b = location % num_sep_per_channel
        
        R = 255 - num_seq_r * separation_per_channel
        G = 255 - num_seq_g * separation_per_channel
        B = 255 - num_seq_b * separation_per_channel
        
        color_dict[location] = (R, G, B)
    return color_dict

class TotalSeg2DDataset(Dataset):
    """
    Dataset for TotalSegmentator 2D slices.

    Loads pre-extracted 2D slices (z, y, x axes) from TotalSeg2D directory.
    Each sample includes k context examples with the same label.

    Directory structure expected:
        TotalSeg2D/
            s0001/
                liver/
                    z_slice.nii.gz, z_slice_img.nii.gz
                    y_slice.nii.gz, y_slice_img.nii.gz
                    x_slice.nii.gz, x_slice_img.nii.gz
                kidney_left/
                    ...
            s0002/
                ...

    Args:
        root_dir: Path to TotalSeg2D directory
        stats_path: Path to totalseg_stats.pkl file
        label_id_list: List of label IDs to include, OR a string specifying a predefined
            split: "train", "val", or "all" (uses splits from data/label_ids_totalseg.py)
        context_size: Number of context examples per sample
        axes: Which axes to include ('z', 'y', 'x' or subset)
        image_size: Optional target size for resizing (H, W)
        crop_to_bbox: If True, crop images around the label bounding box
        bbox_padding: Padding (in pixels) to add around bbox when cropping
        split: 'train', 'val', or 'test' - filters by case/patient (requires meta.csv)
        random_context: If True, randomly sample context cases
        load_dinov3_features: If True, load pre-computed DINOv3 features from
            {axis}_slice_img_dinov3.npz files. Features are [196, 1024] patch tokens.
    """

    def __init__(
        self,
        root_dir: str,
        stats_path: str,
        label_id_list: Union[List[str], str],
        context_size: int = 3,
        axes: Tuple[str, ...] = ("z", "y", "x"),
        image_size: Optional[Tuple[int, int]] = None,
        crop_to_bbox: bool = False,
        bbox_padding: int = 10,
        split: Optional[str] = None,
        random_context: bool = True,
        max_ds_len: Optional[int] = None,
        load_dinov3_features: bool = False,
        random_coloring_nb: int = 0,
        feature_layer_idx: int = 11,
        feature_layers: Optional[List[int]] = None,
        augment: bool = False,
        augment_config: Optional[Dict] = None,
        carve_mix: bool = False,
        carve_mix_config: Optional[Dict] = None,
        advanced_augmentation: bool = False,
        advanced_augmentation_config: Optional[Dict] = None,
        max_labels: Optional[int] = None,
    ):
        """
        Args:
            ...
            random_coloring_nb: Number of labels to sample for random coloring mode.
                If > 0, samples this many labels, finds cases with all labels,
                assigns random RGB colors, and returns 3-channel masks.
                If 0, uses standard single-label binary masks (default behavior).
            feature_layer_idx: Which MedDINO layer to load features from (default: 11).
                Only used when load_dinov3_features=True.
            feature_layers: If provided, load features from multiple layers as a list.
                Overrides feature_layer_idx. Returns dict with features from each layer.
            augment: If True, apply data augmentation (rotation, scale, elastic, intensity).
            augment_config: Optional dict with augmentation parameters:
                - rotation_limit: Max rotation angle in degrees (default: 15)
                - scale_limit: Max scale factor deviation (default: 0.1)
                - elastic_alpha: Elastic deformation intensity (default: 50)
                - elastic_sigma: Elastic deformation smoothness (default: 5)
                - brightness_limit: Max brightness shift (default: 0.1)
                - contrast_limit: Max contrast change (default: 0.1)
                - gamma_limit: Gamma range as tuple (default: (80, 120))
                - noise_std_range: Gaussian noise std dev range (default: (0.02, 0.05))
        """
        self.root_dir = Path(root_dir)

        # Resolve label_id_list if it's a string split name
        if isinstance(label_id_list, str):
            self.label_id_list = get_label_ids(label_id_list, max_labels=max_labels)
            max_labels_str = f" (top {max_labels} by volume)" if max_labels else ""
            print(f"Using {label_id_list} label split: {len(self.label_id_list)} labels{max_labels_str}")
        else:
            if max_labels is not None:
                self.label_id_list = label_id_list[:max_labels]
            else:
                self.label_id_list = label_id_list

        self.context_size = context_size
        self.axes = axes
        self.image_size = image_size
        self.crop_to_bbox = crop_to_bbox
        self.bbox_padding = bbox_padding
        self.random_context = random_context
        self.max_ds_len = max_ds_len
        self.load_dinov3_features = load_dinov3_features
        self.random_coloring_nb = random_coloring_nb
        self.feature_layer_idx = feature_layer_idx
        self.feature_layers = feature_layers

        # Setup augmentation
        self.augment = augment
        if augment:
            aug_cfg = augment_config or {}
            self.spatial_transform, self.intensity_transform = create_augmentation_transforms(
                rotation_limit=aug_cfg.get('rotation_limit', 15.0),
                scale_limit=aug_cfg.get('scale_limit', 0.1),
                elastic_alpha=aug_cfg.get('elastic_alpha', 50.0),
                elastic_sigma=aug_cfg.get('elastic_sigma', 5.0),
                brightness_limit=aug_cfg.get('brightness_limit', 0.1),
                contrast_limit=aug_cfg.get('contrast_limit', 0.1),
                gamma_limit=aug_cfg.get('gamma_limit', (80, 120)),
                noise_std_range=aug_cfg.get('noise_std_range', (0.02, 0.05)),
            )
            print(f"Augmentation enabled: rotation=±{aug_cfg.get('rotation_limit', 15)}°, "
                  f"scale=±{aug_cfg.get('scale_limit', 0.1)*100:.0f}%, elastic, intensity")
        else:
            self.spatial_transform = None
            self.intensity_transform = None

        # Setup CarveMix (only for single-label binary masks)
        self.carve_mix = carve_mix and self.random_coloring_nb == 0
        cm_cfg = carve_mix_config or {}
        self.carve_mix_p = cm_cfg.get("probability", 0.5)
        self.carve_mix_margin_range = tuple(cm_cfg.get("margin_range", [0.1, 0.5]))
        self.carve_mix_harmonize = cm_cfg.get("harmonize", True)
        self.carve_mix_harmonize_sigma = cm_cfg.get("harmonize_sigma", 5.0)
        if self.carve_mix:
            print(f"CarveMix enabled: p={self.carve_mix_p}, margin={self.carve_mix_margin_range}, "
                  f"harmonize={self.carve_mix_harmonize}, sigma={self.carve_mix_harmonize_sigma}")

        # Setup advanced augmentation (only for single-label binary masks)
        self.advanced_augmentation = advanced_augmentation and self.random_coloring_nb == 0
        adv = advanced_augmentation_config or {}
        self.adv_windowing_jitter = adv.get("windowing_jitter", 0)
        fg = adv.get("foreground_crop", {})
        self.adv_fg_crop_p = fg.get("probability", 0.3)
        self.adv_fg_crop_min = fg.get("min_crop_frac", 0.5)
        mp = adv.get("mask_perturbation", {})
        self.adv_mask_perturb_p = mp.get("probability", 0.3)
        self.adv_mask_perturb_kernel = mp.get("max_kernel", 5)
        rd = adv.get("resolution_degradation", {})
        self.adv_degrade_p = rd.get("probability", 0.2)
        self.adv_degrade_min_scale = rd.get("min_scale", 0.25)
        ai = adv.get("asymmetric_intensity", {})
        self.adv_asym_p = ai.get("probability", 0.5)
        self.adv_asym_brightness = ai.get("brightness_shift", 0.15)
        self.adv_asym_contrast = ai.get("contrast_scale", 0.15)
        self.adv_asym_gamma = tuple(ai.get("gamma_range", [0.7, 1.5]))
        if self.advanced_augmentation:
            print(f"Advanced augmentation enabled: windowing_jitter={self.adv_windowing_jitter}, "
                  f"fg_crop_p={self.adv_fg_crop_p}, mask_perturb_p={self.adv_mask_perturb_p}, "
                  f"degrade_p={self.adv_degrade_p}, asym_intensity_p={self.adv_asym_p}")

        # Load stats dict
        with open(stats_path, "rb") as f:
            self.stats = pickle.load(f)
        print(f"Loaded stats for {len(self.stats)} cases")

        # Convert label_id_list to set for faster lookup
        label_id_set = set(self.label_id_list)

        # Build label_id -> case_ids mapping
        self.label_to_cases: Dict[str, List[str]] = {}
        # Build case_id -> set of label_ids mapping (for multi-label queries)
        self.case_to_labels: Dict[str, set] = {}
        all_labels_in_stats = set()
        for case_id, labels in self.stats.items():
            self.case_to_labels[case_id] = set()
            for label_id in labels.keys():
                all_labels_in_stats.add(label_id)
                if label_id in label_id_set:
                    self.label_to_cases.setdefault(label_id, []).append(case_id)
                    self.case_to_labels[case_id].add(label_id)

        # Debug: show mismatch if no labels found
        if len(self.label_to_cases) == 0:
            sample_stats_labels = list(all_labels_in_stats)[:5]
            sample_requested_labels = self.label_id_list[:5]
            print(f"WARNING: No matching labels found!")
            print(f"  Sample labels in stats: {sample_stats_labels}")
            print(f"  Sample requested labels: {sample_requested_labels}")

        print(f"Built mapping for {len(self.label_to_cases)} labels (stats has {len(all_labels_in_stats)} unique labels)")

        # For random coloring mode, precompute color palette
        if self.random_coloring_nb > 0:
            self.color_palette = define_colors_by_mean_sep(
                num_colors=max(256, len(self.label_id_list)),
                channelsep=7
            )

        # Filter by split if provided
        if split is not None:
            self._filter_by_split(split)

        # Build list of all (case_id, label_id, axis) samples
        # Also build (label_id, axis) -> [valid_case_ids] mapping for fast context lookup
        self.samples = []
        self.valid_contexts: Dict[Tuple[str, str], List[str]] = {}  # (label_id, axis) -> [case_ids]

        for case_id, labels in self.stats.items():
            for label_id, label_stats in labels.items():
                if label_id in label_id_set:
                    # Check case exists in root_dir
                    case_dir = self.root_dir / case_id / label_id
                    if case_dir.exists():
                        for axis in self.axes:
                            # Use slice_coverage from stats if available (after re-preprocessing)
                            # Otherwise include all samples (will be filtered at runtime)
                            slice_coverage = label_stats.get("slice_coverage", {})
                            coverage = slice_coverage.get(axis, 1)  # default to 1 if not available

                            if coverage > 0:
                                self.samples.append((case_id, label_id, axis))
                                key = (label_id, axis)
                                self.valid_contexts.setdefault(key, []).append(case_id)

        print(f"Created {len(self.samples)} samples")
        if self.max_ds_len is not None and len(self.samples) > self.max_ds_len:
            random.shuffle(self.samples)
            print(f"Shuffled samples")

    def _filter_by_split(self, split: str):
        """Filter cases by train/val/test split using meta.csv."""
        import pandas as pd

        meta_path = self.root_dir.parent / "TotalSeg" / "meta.csv"
        if not meta_path.exists():
            print(f"Warning: meta.csv not found at {meta_path}, skipping split filter")
            return

        df = pd.read_csv(meta_path, sep=";")
        split_case_ids = set(df["image_id"][df["split"] == split].tolist())

        # Filter stats
        self.stats = {k: v for k, v in self.stats.items() if k in split_case_ids}

        # Filter label_to_cases
        for label_id in self.label_to_cases:
            self.label_to_cases[label_id] = [
                c for c in self.label_to_cases[label_id] if c in split_case_ids
            ]

        print(f"Filtered to {len(self.stats)} cases for split '{split}'")

        # Also filter case_to_labels
        self.case_to_labels = {k: v for k, v in self.case_to_labels.items() if k in split_case_ids}

    def _find_cases_with_labels(self, label_ids: List[str]) -> List[str]:
        """Find cases that have ALL specified labels."""
        label_set = set(label_ids)
        matching_cases = []
        for case_id, case_labels in self.case_to_labels.items():
            if label_set.issubset(case_labels):
                matching_cases.append(case_id)
        return matching_cases

    def _sample_random_colors(self, num_labels: int) -> Dict[int, Tuple[int, int, int]]:
        """Sample random colors for each label index."""
        # Sample random indices from the color palette
        color_indices = random.sample(range(len(self.color_palette)), num_labels)
        return {i: self.color_palette[idx] for i, idx in enumerate(color_indices)}

    def _create_colored_mask(
        self,
        case_id: str,
        label_ids: List[str],
        axis: str,
        color_map: Dict[int, Tuple[int, int, int]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load multiple labels and create a 3-channel RGB colored mask.

        Args:
            case_id: Case ID
            label_ids: List of label IDs to load
            axis: Slice axis
            color_map: Mapping from label index to RGB color

        Returns:
            img: [H, W] grayscale image
            colored_mask: [3, H, W] RGB colored mask (values 0-1)
        """
        img = None
        colored_mask = None

        for label_idx, label_id in enumerate(label_ids):
            slice_dir = self.root_dir / case_id / label_id

            # Load image (only once, from first label)
            if img is None:
                img_path_npy = slice_dir / f"{axis}_slice_img.npy"
                if img_path_npy.exists():
                    img = np.load(img_path_npy)
                else:
                    img_path = slice_dir / f"{axis}_slice_img.nii.gz"
                    img = nib.load(str(img_path)).get_fdata().astype(np.float32)

                # Initialize colored mask
                H, W = img.shape
                colored_mask = np.zeros((3, H, W), dtype=np.float32)

            # Load label mask
            label_path_npy = slice_dir / f"{axis}_slice.npy"
            if label_path_npy.exists():
                label = np.load(label_path_npy)
            else:
                label_path = slice_dir / f"{axis}_slice.nii.gz"
                label = nib.load(str(label_path)).get_fdata().astype(np.float32)

            # Apply color to this label's mask
            mask_binary = (label > 0.5).astype(np.float32)
            r, g, b = color_map[label_idx]
            if colored_mask is not None:
                colored_mask[0] += mask_binary * (r / 255.0)
                colored_mask[1] += mask_binary * (g / 255.0)
                colored_mask[2] += mask_binary * (b / 255.0)

        # Clip to [0, 1] in case of overlapping labels
        colored_mask = np.clip(colored_mask, 0, 1)

        return img, colored_mask

    def _load_slice(self, case_id: str, label_id: str, axis: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load image and label slice for a specific case/label/axis.

        Supports both .npy (fast) and .nii.gz (legacy) formats.
        """
        slice_dir = self.root_dir / case_id / label_id

        # Try .npy first (fast), fall back to .nii.gz (legacy)
        img_path_npy = slice_dir / f"{axis}_slice_img.npy"
        label_path_npy = slice_dir / f"{axis}_slice.npy"

        if img_path_npy.exists():
            img = np.load(img_path_npy, mmap_mode='r')
            label = np.load(label_path_npy, mmap_mode='r')
        else:
            # Legacy .nii.gz format
            img_path = slice_dir / f"{axis}_slice_img.nii.gz"
            label_path = slice_dir / f"{axis}_slice.nii.gz"
            img = nib.load(str(img_path)).get_fdata().astype(np.float32)
            label = nib.load(str(label_path)).get_fdata().astype(np.float32)

        return img, label

    def _load_features(
        self, case_id: str, label_id: str, axis: str, layer_idx: Optional[int] = None
    ) -> Optional[torch.Tensor]:
        """Load pre-computed DINOv3 features for a specific case/label/axis.

        Args:
            case_id: Case identifier
            label_id: Label identifier
            axis: Slice axis ('x', 'y', 'z')
            layer_idx: Specific layer to load (overrides self.feature_layer_idx)

        Returns:
            features: [N, D] where N = CLS + registers + patches, D = 768
                If self.feature_layers is set, returns dict with features from each layer.
        """
        slice_dir = self.root_dir / case_id / label_id
        features_path = slice_dir / f"{axis}_slice_img_meddino.npz"

        if not features_path.exists():
            return None

        data = np.load(features_path)

        # Determine which layer(s) to load
        if self.feature_layers is not None:
            # Multi-layer mode: return dict of features
            result = {}
            for lidx in self.feature_layers:
                cls = torch.from_numpy(data[f"layer_{lidx}_cls"])
                regs = torch.from_numpy(data[f"layer_{lidx}_registers"])
                patches = torch.from_numpy(data[f"layer_{lidx}_patches"])
                if cls.ndim == 1:
                    cls = cls.unsqueeze(0)
                if regs.ndim == 1:
                    regs = regs.unsqueeze(0)
                if patches.ndim == 1:
                    patches = patches.unsqueeze(0)
                full_sequence = torch.cat([cls, regs, patches], dim=0)
                result[f"layer_{lidx}"] = full_sequence
            return result

        # Single layer mode
        lidx = layer_idx if layer_idx is not None else self.feature_layer_idx
        cls = torch.from_numpy(data[f"layer_{lidx}_cls"])
        regs = torch.from_numpy(data[f"layer_{lidx}_registers"])
        patches = torch.from_numpy(data[f"layer_{lidx}_patches"])

        # Ensure consistent shape
        if cls.ndim == 1:
            cls = cls.unsqueeze(0)
        if regs.ndim == 1:
            regs = regs.unsqueeze(0)
        if patches.ndim == 1:
            patches = patches.unsqueeze(0)

        # Concatenate: CLS (1) + Registers (4) + Patches
        full_sequence = torch.cat([cls, regs, patches], dim=0)

        return full_sequence

    def _get_2d_bbox(self, case_id: str, label_id: str, axis: str) -> Tuple[int, int, int, int]:
        """
        Get 2D bounding box for a slice from the 3D bbox in stats.
        
        The 3D bbox is stored as ((zmin, zmax), (ymin, ymax), (xmin, xmax)).
        For each axis slice, we return the 2D bbox of the other two dimensions.
        
        Returns:
            (row_min, row_max, col_min, col_max) for the 2D slice
        """
        bbox_3d = self.stats[case_id][label_id]["bbox"]
        z_range, y_range, x_range = bbox_3d
        
        # Map axis to 2D bbox (row, col correspond to the slice dimensions)
        if axis == "z":  # slice along z, so 2D is (y, x)
            return (y_range[0], y_range[1], x_range[0], x_range[1])
        elif axis == "y":  # slice along y, so 2D is (z, x)
            return (z_range[0], z_range[1], x_range[0], x_range[1])
        else:  # axis == "x", slice along x, so 2D is (z, y)
            return (z_range[0], z_range[1], y_range[0], y_range[1])

    def _crop_to_bbox(self, img: np.ndarray, label: np.ndarray, 
                      bbox: Tuple[int, int, int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Crop image and label to bounding box with padding.
        
        Args:
            img: 2D image array
            label: 2D label array
            bbox: (row_min, row_max, col_min, col_max)
            
        Returns:
            Cropped (img, label) arrays
        """
        row_min, row_max, col_min, col_max = bbox
        h, w = img.shape
        
        # Add padding and clamp to image bounds
        row_min = max(0, row_min - self.bbox_padding)
        row_max = min(h, row_max + self.bbox_padding + 1)
        col_min = max(0, col_min - self.bbox_padding)
        col_max = min(w, col_max + self.bbox_padding + 1)
        
        return img[row_min:row_max, col_min:col_max], label[row_min:row_max, col_min:col_max]

    def _normalize_image(self, img: np.ndarray, a_min: float = -200, a_max: float = 300, jitter: float = 0) -> np.ndarray:
        """Normalize CT image to [0, 1] range with optional windowing jitter."""
        if jitter > 0:
            a_min = a_min + random.uniform(-jitter, jitter)
            a_max = a_max + random.uniform(-jitter, jitter)
            if a_max - a_min < 200:  # minimum window width for label visibility
                a_max = a_min + 200
        img = np.clip(img, a_min, a_max)
        img = (img - a_min) / (a_max - a_min)
        return img

    def _resize(self, arr: np.ndarray, size: Tuple[int, int], mode: str = "bilinear") -> np.ndarray:
        """Resize 2D array to target size."""
        tensor = torch.from_numpy(arr.copy()).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        if mode == "bilinear":
            resized = torch.nn.functional.interpolate(tensor, size=size, mode="bilinear", align_corners=False)
        else:
            resized = torch.nn.functional.interpolate(tensor, size=size, mode="nearest")
        return resized.squeeze().numpy()

    def __len__(self) -> int:
        if self.max_ds_len is not None:
            return min(self.max_ds_len, len(self.samples))
        return len(self.samples)

    def _resize_multichannel(self, arr: np.ndarray, size: Tuple[int, int], mode: str = "bilinear") -> np.ndarray:
        """Resize multi-channel array [C, H, W] to target size."""
        tensor = torch.from_numpy(arr).unsqueeze(0)  # [1, C, H, W]
        if mode == "bilinear":
            resized = torch.nn.functional.interpolate(tensor, size=size, mode="bilinear", align_corners=False)
        else:
            resized = torch.nn.functional.interpolate(tensor, size=size, mode="nearest")
        return resized.squeeze(0).numpy()  # [C, H, W]

    def _apply_augmentation(
        self,
        img: np.ndarray,
        mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply augmentation to image and mask.

        Args:
            img: [H, W] normalized image in [0, 1]
            mask: [H, W] binary mask

        Returns:
            Augmented image and mask
        """
        if not self.augment or self.spatial_transform is None:
            return img, mask

        # Convert image to uint8 for albumentations (expects 0-255)
        img_uint8 = (img * 255).astype(np.uint8)
        mask_uint8 = (mask * 255).astype(np.uint8)

        # Apply spatial transforms (same to image and mask)
        spatial_result = self.spatial_transform(image=img_uint8, mask=mask_uint8)
        img_aug = spatial_result['image']
        mask_aug = spatial_result['mask']

        # Apply intensity transforms (image only)
        if self.intensity_transform is not None:
            intensity_result = self.intensity_transform(image=img_aug)
            img_aug = intensity_result['image']

        # Convert back to float [0, 1]
        img_aug = img_aug.astype(np.float32) / 255.0
        mask_aug = (mask_aug > 127).astype(np.float32)  # Binarize mask

        return img_aug, mask_aug

    def _load_carve_mix_donor(
        self, label_id: str, axis: str, exclude_case_ids: List[str]
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Load a random donor image/mask for CarveMix, excluding specified cases."""
        context_key = (label_id, axis)
        candidates = self.valid_contexts.get(context_key, [])
        exclude_set = set(exclude_case_ids)
        candidates = [c for c in candidates if c not in exclude_set]
        if not candidates:
            return None

        random.shuffle(candidates)
        for donor_case_id in candidates[:3]:  # retry up to 3 times
            try:
                donor_img, donor_mask = self._load_slice(donor_case_id, label_id, axis)
                if donor_mask.max() == 0:
                    continue
                if self.crop_to_bbox:
                    bbox = self._get_2d_bbox(donor_case_id, label_id, axis)
                    donor_img, donor_mask = self._crop_to_bbox(donor_img, donor_mask, bbox)
                donor_img = self._normalize_image(donor_img)
                if self.image_size is not None:
                    donor_img = self._resize(donor_img, self.image_size, mode="bilinear")
                    donor_mask = self._resize(donor_mask, self.image_size, mode="nearest")
                return donor_img, donor_mask
            except Exception:
                continue
        return None

    def _apply_carve_mix(
        self,
        target_img: np.ndarray,
        target_mask: np.ndarray,
        label_id: str,
        axis: str,
        exclude_case_ids: List[str],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Conditionally apply CarveMix to target based on probability."""
        if not self.carve_mix or random.random() > self.carve_mix_p:
            return target_img, target_mask

        donor = self._load_carve_mix_donor(label_id, axis, exclude_case_ids)
        if donor is None:
            return target_img, target_mask

        donor_img, donor_mask = donor
        return carve_mix_2d(
            target_img, target_mask, donor_img, donor_mask,
            margin_range=self.carve_mix_margin_range,
            harmonize=self.carve_mix_harmonize,
            harmonize_sigma=self.carve_mix_harmonize_sigma,
        )

    def _apply_advanced_augmentation(
        self,
        target_img: np.ndarray,
        target_mask: np.ndarray,
        context_imgs: Optional[List[np.ndarray]] = None,
        context_masks: Optional[List[np.ndarray]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[List[np.ndarray]], Optional[List[np.ndarray]]]:
        """Apply post-resize advanced augmentations for generalization.

        Asymmetric intensity and resolution degradation are applied independently
        per image so target/context see different appearance. Context masks get
        morphological perturbation to simulate annotation noise.
        """
        if not self.advanced_augmentation:
            return target_img, target_mask, context_imgs, context_masks

        # Asymmetric intensity (different random params per image)
        if random.random() < self.adv_asym_p:
            target_img = random_intensity_shift(
                target_img, self.adv_asym_brightness, self.adv_asym_contrast, self.adv_asym_gamma
            )
        if context_imgs:
            for i in range(len(context_imgs)):
                if random.random() < self.adv_asym_p:
                    context_imgs[i] = random_intensity_shift(
                        context_imgs[i], self.adv_asym_brightness, self.adv_asym_contrast, self.adv_asym_gamma
                    )

        # Resolution degradation (independent per image)
        if random.random() < self.adv_degrade_p:
            target_img = degrade_resolution(target_img, self.adv_degrade_min_scale)
        if context_imgs:
            for i in range(len(context_imgs)):
                if random.random() < self.adv_degrade_p:
                    context_imgs[i] = degrade_resolution(context_imgs[i], self.adv_degrade_min_scale)

        # Context mask perturbation (context only — target mask is GT)
        if context_masks:
            for i in range(len(context_masks)):
                if random.random() < self.adv_mask_perturb_p:
                    context_masks[i] = perturb_mask(context_masks[i], self.adv_mask_perturb_kernel)

        return target_img, target_mask, context_imgs, context_masks

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample with context examples.

        Returns:
            Dictionary with:
            - 'image': [1, H, W] - Target image
            - 'label': [C, H, W] - Target label (C=1 for binary, C=3 for random coloring)
            - 'context_in': [k, 1, H, W] - Context images
            - 'context_out': [k, C, H, W] - Context labels
            - 'target_case_id': str
            - 'context_case_ids': List[str]
            - 'label_id': str or List[str] (list if random_coloring_nb > 0)
            - 'axis': str
            - 'color_map': Dict[int, Tuple[int,int,int]] (only if random_coloring_nb > 0)
            If load_dinov3_features=True, also includes:
            - 'target_features': [196, 1024] - Target DINOv3 patch features
            - 'context_features': [k, 196, 1024] - Context DINOv3 patch features
        """
        target_case_id, label_id, axis = self.samples[idx]

        # Random coloring mode: sample multiple labels and find cases with all of them
        if self.random_coloring_nb > 0:
            return self._getitem_random_coloring(idx)

        # Standard single-label mode
        # Load target
        target_img, target_label = self._load_slice(target_case_id, label_id, axis)

        # Warn if target mask is empty (shouldn't happen with proper preprocessing)
        if target_label.max() == 0:
            print(f"Warning: Empty target mask for {target_case_id}/{label_id}/{axis}")

        # Load target features if requested
        target_features = None
        if self.load_dinov3_features:
            target_features = self._load_features(target_case_id, label_id, axis)

        # Crop to bbox if enabled
        if self.crop_to_bbox:
            target_bbox = self._get_2d_bbox(target_case_id, label_id, axis)
            target_img, target_label = self._crop_to_bbox(target_img, target_label, target_bbox)

        jitter = self.adv_windowing_jitter if self.advanced_augmentation else 0
        target_img = self._normalize_image(target_img, jitter=jitter)

        # Foreground random crop (before resize, ensures FG stays visible)
        if self.advanced_augmentation and random.random() < self.adv_fg_crop_p:
            target_img, target_label = foreground_random_crop(
                target_img, target_label, self.adv_fg_crop_min
            )

        # Skip context loading if context_size is 0
        if self.context_size == 0:
            # Resize if needed
            if self.image_size is not None:
                target_img = self._resize(target_img, self.image_size, mode="bilinear")
                target_label = self._resize(target_label, self.image_size, mode="nearest")

            # CarveMix on target (after resize, before augmentation)
            target_img, target_label = self._apply_carve_mix(
                target_img, target_label, label_id, axis, [target_case_id]
            )

            # Advanced augmentation (asymmetric intensity, resolution degradation)
            target_img, target_label, _, _ = self._apply_advanced_augmentation(
                target_img, target_label
            )

            # Apply augmentation
            if self.augment:
                target_img, target_label = self._apply_augmentation(target_img, target_label)

            # Convert to tensors
            target_in = torch.from_numpy(target_img.copy()).unsqueeze(0).float()  # [1, H, W]
            target_out = torch.from_numpy(target_label.copy()).unsqueeze(0).float()  # [1, H, W]

            result = {
                "image": target_in,
                "label": target_out,
                "target_case_id": target_case_id,
                "label_id": label_id,
                "axis": axis,
            }

            if self.load_dinov3_features and target_features is not None:
                result["target_features"] = target_features

            return result

        # Get context cases with same label+axis that have non-empty masks (excluding target)
        # Use precomputed valid_contexts if available, otherwise fall back to label_to_cases
        context_key = (label_id, axis)
        if context_key in self.valid_contexts:
            available_contexts = [c for c in self.valid_contexts[context_key] if c != target_case_id]
        else:
            available_contexts = [c for c in self.label_to_cases.get(label_id, []) if c != target_case_id]

        if len(available_contexts) == 0:
            raise RuntimeError(f"No context cases for label '{label_id}' axis '{axis}'")

        # Shuffle all available contexts and try until we have enough valid ones
        if self.random_context:
            random.shuffle(available_contexts)

        # Load context slices - try all candidates until we have enough
        context_imgs = []
        context_labels = []
        context_features_list = []
        valid_context_ids = []

        for ctx_case_id in available_contexts:
            # Stop once we have enough valid contexts
            if len(context_imgs) >= self.context_size:
                break

            try:
                ctx_img, ctx_label = self._load_slice(ctx_case_id, label_id, axis)

                # Skip context if mask is empty (label doesn't exist in this slice)
                if ctx_label.max() == 0:
                    continue

                # Load context features if requested
                ctx_features = None
                if self.load_dinov3_features:
                    ctx_features = self._load_features(ctx_case_id, label_id, axis)

                # Crop to bbox if enabled
                if self.crop_to_bbox:
                    ctx_bbox = self._get_2d_bbox(ctx_case_id, label_id, axis)
                    ctx_img, ctx_label = self._crop_to_bbox(ctx_img, ctx_label, ctx_bbox)

                ctx_img = self._normalize_image(ctx_img, jitter=jitter)

                # Foreground random crop (independent per context)
                if self.advanced_augmentation and random.random() < self.adv_fg_crop_p:
                    ctx_img, ctx_label = foreground_random_crop(
                        ctx_img, ctx_label, self.adv_fg_crop_min
                    )

                context_imgs.append(ctx_img)
                context_labels.append(ctx_label)
                if ctx_features is not None:
                    context_features_list.append(ctx_features)
                valid_context_ids.append(ctx_case_id)
            except Exception as e:
                print(f"Warning: Failed to load {ctx_case_id}/{label_id}/{axis}: {e}")
                continue

        if len(context_imgs) == 0:
            raise RuntimeError(f"Failed to load any context for label '{label_id}' axis '{axis}' (tried {len(available_contexts)} cases, all had empty masks)")

        # Resize if needed
        if self.image_size is not None:
            target_img = self._resize(target_img, self.image_size, mode="bilinear")
            target_label = self._resize(target_label, self.image_size, mode="nearest")
            context_imgs = [self._resize(c, self.image_size, mode="bilinear") for c in context_imgs]
            context_labels = [self._resize(c, self.image_size, mode="nearest") for c in context_labels]

        # CarveMix on target only (after resize, before augmentation)
        exclude_ids = [target_case_id] + valid_context_ids
        target_img, target_label = self._apply_carve_mix(
            target_img, target_label, label_id, axis, exclude_ids
        )

        # Advanced augmentation (asymmetric intensity, resolution degradation, mask perturbation)
        target_img, target_label, context_imgs, context_labels = self._apply_advanced_augmentation(  # type: ignore[assignment]
            target_img, target_label, context_imgs, context_labels
        )

        # Apply augmentation (different random transform for target and each context)
        if self.augment:
            target_img, target_label = self._apply_augmentation(target_img, target_label)
            aug_context_imgs = []
            aug_context_labels = []
            for ctx_img, ctx_label in zip(context_imgs, context_labels):
                ctx_img_aug, ctx_label_aug = self._apply_augmentation(ctx_img, ctx_label)
                aug_context_imgs.append(ctx_img_aug)
                aug_context_labels.append(ctx_label_aug)
            context_imgs = aug_context_imgs
            context_labels = aug_context_labels

        # Convert to tensors
        target_in = torch.from_numpy(target_img.copy()).unsqueeze(0).float()  # [1, H, W]
        target_out = torch.from_numpy(target_label.copy()).unsqueeze(0).float()  # [1, H, W]
        context_in = torch.stack([torch.from_numpy(c.copy()).unsqueeze(0).float() for c in context_imgs])  # [k, 1, H, W]
        context_out = torch.stack([torch.from_numpy(c.copy()).unsqueeze(0).float() for c in context_labels])  # [k, 1, H, W]

        result = {
            "image": target_in,
            "label": target_out,
            "context_in": context_in,
            "context_out": context_out,
            "target_case_id": target_case_id,
            "context_case_ids": valid_context_ids,
            "label_id": label_id,
            "axis": axis,
        }

        # Add features if loaded
        if self.load_dinov3_features:
            if target_features is not None:
                result["target_features"] = target_features # [196, 1024]
            if context_features_list:
                result["context_features"] = torch.stack(
                    [f for f in context_features_list]
                )  # [k, 196, 1024]

        return result

    def _getitem_random_coloring(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample with random coloring (multiple labels with RGB colors).

        Samples random_coloring_nb labels, finds cases with all labels,
        assigns random RGB colors, and returns 3-channel colored masks.
        """
        # Get base sample info for axis selection
        _, _, axis = self.samples[idx]

        # Sample N random labels
        n_labels = min(self.random_coloring_nb, len(self.label_id_list))
        sampled_labels = random.sample(self.label_id_list, n_labels)

        # Find cases that have ALL sampled labels
        available_cases = self._find_cases_with_labels(sampled_labels)

        if len(available_cases) < 2:
            # Fallback: try with fewer labels
            for n in range(n_labels - 1, 0, -1):
                sampled_labels = random.sample(self.label_id_list, n)
                available_cases = self._find_cases_with_labels(sampled_labels)
                if len(available_cases) >= 2:
                    break

        if len(available_cases) < 2:
            raise RuntimeError(f"Not enough cases with labels {sampled_labels}")

        # Sample random colors for each label
        color_map = self._sample_random_colors(len(sampled_labels))

        # Sample target case
        target_case_id = random.choice(available_cases)

        # Load target with colored mask
        target_img, target_label = self._create_colored_mask(
            target_case_id, sampled_labels, axis, color_map
        )
        target_img = self._normalize_image(target_img)

        # Load target features if requested (use first label's features)
        target_features = None
        if self.load_dinov3_features:
            target_features = self._load_features(target_case_id, sampled_labels[0], axis)

        # Get context candidates (excluding target), shuffle if random
        context_candidates = [c for c in available_cases if c != target_case_id]
        if self.random_context:
            random.shuffle(context_candidates)

        # Load context slices with same color mapping - try all until we have enough
        context_imgs = []
        context_labels = []
        context_features_list = []
        valid_context_ids = []

        for ctx_case_id in context_candidates:
            # Stop once we have enough valid contexts
            if len(context_imgs) >= self.context_size:
                break

            try:
                ctx_img, ctx_label = self._create_colored_mask(
                    ctx_case_id, sampled_labels, axis, color_map
                )

                # Skip context if mask is empty (labels don't exist in this slice)
                if ctx_label.max() == 0:
                    continue

                ctx_img = self._normalize_image(ctx_img)

                # Load context features if requested
                ctx_features = None
                if self.load_dinov3_features:
                    ctx_features = self._load_features(ctx_case_id, sampled_labels[0], axis)

                context_imgs.append(ctx_img)
                context_labels.append(ctx_label)
                if ctx_features is not None:
                    context_features_list.append(ctx_features)
                valid_context_ids.append(ctx_case_id)
            except Exception as e:
                print(f"Warning: Failed to load colored mask for {ctx_case_id}: {e}")
                continue

        if len(context_imgs) == 0:
            raise RuntimeError(f"Failed to load any context for labels {sampled_labels} axis '{axis}' (tried {len(context_candidates)} cases)")

        # Resize if needed
        if self.image_size is not None:
            target_img = self._resize(target_img, self.image_size, mode="bilinear")
            target_label = self._resize_multichannel(target_label, self.image_size, mode="nearest")
            context_imgs = [self._resize(c, self.image_size, mode="bilinear") for c in context_imgs]
            context_labels = [self._resize_multichannel(c, self.image_size, mode="nearest") for c in context_labels]

        # Convert to tensors
        target_in = torch.from_numpy(target_img).unsqueeze(0)  # [1, H, W]
        target_out = torch.from_numpy(target_label)  # [3, H, W]
        context_in = torch.stack([torch.from_numpy(c).unsqueeze(0) for c in context_imgs])  # [k, 1, H, W]
        context_out = torch.stack([torch.from_numpy(c) for c in context_labels])  # [k, 3, H, W]

        result = {
            "image": target_in,
            "label": target_out,
            "context_in": context_in,
            "context_out": context_out,
            "target_case_id": target_case_id,
            "context_case_ids": valid_context_ids,
            "label_id": sampled_labels,  # List of label IDs
            "axis": axis,
            "color_map": color_map,  # For visualization/debugging
        }

        # Add features if loaded
        if self.load_dinov3_features:
            if target_features is not None:
                result["target_features"] = torch.from_numpy(target_features)
            if context_features_list:
                result["context_features"] = torch.stack(
                    [torch.from_numpy(f) for f in context_features_list]
                )

        return result


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Custom collate function for batching.

    Handles both single-label (1-channel) and random coloring (3-channel) modes.
    Also handles context_size=0 case where context_in/context_out are not present.
    """
    result = {
        "image": torch.stack([item["image"] for item in batch]),  # [B, 1, H, W]
        "label": torch.stack([item["label"] for item in batch]),  # [B, C, H, W] C=1 or 3
        "target_case_ids": [item["target_case_id"] for item in batch],
        "label_ids": [item["label_id"] for item in batch],  # str or List[str]
        "axes": [item["axis"] for item in batch],
    }

    # Add context data if present (context_size > 0)
    if "context_in" in batch[0]:
        result["context_in"] = torch.stack([item["context_in"] for item in batch])  # [B, k, 1, H, W]
        result["context_out"] = torch.stack([item["context_out"] for item in batch])  # [B, k, C, H, W]
        result["context_case_ids"] = [item["context_case_ids"] for item in batch]

    # Add color_map if present (random coloring mode)
    if "color_map" in batch[0]:
        result["color_maps"] = [item["color_map"] for item in batch]

    # Add features if present in batch
    if "target_features" in batch[0]:
        result["target_features"] = torch.stack(
            [item["target_features"] for item in batch]
        )  # [B, 196, 1024]
    if "context_features" in batch[0]:
        result["context_features"] = torch.stack(
            [item["context_features"] for item in batch]
        )  # [B, k, 196, 1024]

    return result


def get_dataloader(
    root_dir: str,
    stats_path: str,
    label_id_list: Union[List[str], str],
    context_size: int = 3,
    batch_size: int = 8,
    image_size: Optional[Tuple[int, int]] = (256, 256),
    crop_to_bbox: bool = False,
    bbox_padding: int = 10,
    num_workers: int = 4,
    split: Optional[str] = None,
    shuffle: bool = True,
    load_dinov3_features: bool = False,
    random_coloring_nb: int = 0,
    feature_layer_idx: int = 11,
    feature_layers: Optional[List[int]] = None,
    max_labels: Optional[int] = None,
    **dataset_kwargs,
) -> DataLoader:
    """
    Create a DataLoader for TotalSeg2D dataset.

    Args:
        root_dir: Path to TotalSeg2D directory
        stats_path: Path to totalseg_stats.pkl
        label_id_list: List of label IDs, or "train"/"val"/"all" for predefined splits
        context_size: Number of context examples
        batch_size: Batch size
        image_size: Target image size (H, W)
        crop_to_bbox: If True, crop images around label bounding box
        bbox_padding: Padding around bbox when cropping
        num_workers: Number of data loading workers
        split: 'train', 'val', or 'test'
        shuffle: Whether to shuffle
        load_dinov3_features: If True, load pre-computed DINOv3 features
        random_coloring_nb: Number of labels to sample for random coloring mode.
            If > 0, returns 3-channel RGB masks with random colors per label.
            If 0, uses standard single-label binary masks (default).
        feature_layer_idx: Which MedDINO layer to load features from (default: 11).
        feature_layers: If provided, load features from multiple layers.
        max_labels: If provided, only use the first n labels by total volume.
        **dataset_kwargs: Additional args for TotalSeg2DDataset

    Returns:
        DataLoader instance
    """
    dataset = TotalSeg2DDataset(
        root_dir=root_dir,
        stats_path=stats_path,
        label_id_list=label_id_list,
        context_size=context_size,
        image_size=image_size,
        crop_to_bbox=crop_to_bbox,
        bbox_padding=bbox_padding,
        split=split,
        load_dinov3_features=load_dinov3_features,
        random_coloring_nb=random_coloring_nb,
        feature_layer_idx=feature_layer_idx,
        feature_layers=feature_layers,
        max_labels=max_labels,
        **dataset_kwargs,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
