"""
TotalSegmentator 2D DataLoader (Fast HDF5 Version)

Loads pre-processed 2D slices from consolidated HDF5 files for speed.
Supports in-context learning by sampling k context examples with the same label.

Label splits (train/val) are defined in data/label_ids_totalseg.py.
"""

import pickle
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from data.label_ids_totalseg import get_label_ids
from src.dataloaders.augmentations import (
    create_spatial_only_transform,
    create_intensity_only_transform,
    carve_mix_2d,
    cut_mix_2d,
    foreground_random_crop,
    perturb_mask,
    degrade_resolution,
    apply_task_level_augmentation,
    apply_universeg_augmentation,
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
    Dataset for TotalSegmentator 2D slices from HDF5 files.

    Loads 2D slices on-the-fly from HDF5 files (one per case).
    This is much faster than loading from thousands of individual .npy files.

    Directory structure expected:
        TotalSeg2D_H5/
            s0001.h5
            s0002.h5
            ...

    Args:
        root_dir: Path to TotalSeg2D_H5 directory
        stats_path: Path to totalseg_stats.pkl file
        ... (same as original dataloader)
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
        split: Optional[Union[str, List[str]]] = None,
        random_context: bool = True,
        max_ds_len: Optional[int] = None,
        random_coloring_nb: int = 0,
        # Legacy augmentation parameters (backwards compatible)
        augment: bool = False,
        augment_config: Optional[Dict] = None,
        carve_mix: bool = False,
        carve_mix_config: Optional[Dict] = None,
        advanced_augmentation: bool = False,
        advanced_augmentation_config: Optional[Dict] = None,
        # New unified augmentation config
        augmentation_config: Optional[Dict] = None,
        max_labels: Optional[int] = None,
        max_cases: Optional[int] = None,
        class_balanced: bool = False,
        modality: str = "ct",  # "ct" or "mri"
        # Coverage filtering (unified with shared dataloader)
        min_coverage: int = 100,  # Minimum pixels for a slice to be valid
        min_coverage_ratio: float = 0.1,  # Minimum ratio of max coverage
        # Context diversity selection
        context_diversity: Optional[Dict] = None,  # {type: "farthest", num_candidates: 10}
    ):
        self.root_dir = Path(root_dir)
        self.modality = modality.lower()
        self.min_coverage = min_coverage
        self.min_coverage_ratio = min_coverage_ratio

        # Context diversity config
        div_cfg = context_diversity or {}
        self.context_diversity_type = div_cfg.get('type', 'random')  # 'random', 'farthest'
        self.context_diversity_candidates = div_cfg.get('num_candidates', 10)  # Sample pool size
        self.context_feature_key = div_cfg.get('feature_key', 'mean_features')  # Key in stats

        # Resolve label_id_list
        if isinstance(label_id_list, str):
            self.label_id_list = get_label_ids(label_id_list, max_labels=max_labels, modality=self.modality)
        else:
            self.label_id_list = label_id_list[:max_labels] if max_labels else label_id_list
        label_id_set = set(self.label_id_list)

        self.context_size = context_size
        self.axes = axes
        self.image_size = image_size
        self.crop_to_bbox = crop_to_bbox
        self.bbox_padding = bbox_padding
        self.random_context = random_context
        self.max_ds_len = max_ds_len
        self.random_coloring_nb = random_coloring_nb

        # Setup augmentation (unified config only)
        if augmentation_config is not None and augmentation_config.get("enabled", False):
            self._setup_unified_augmentation(augmentation_config)
        else:
            # No augmentation - set defaults
            self.augment = False
            self.use_unified_augmentation = True
            self.aug_type = "none"
            self.mix_type = "none"
            self.mix_probability = 0.0
            self.spatial_enabled = False
            self.spatial_transform = None
            self.intensity_enabled = False
            self.intensity_transform = None
            self.fg_crop_enabled = False
            self.degrade_enabled = False
            self.mask_perturb_enabled = False
            self.task_level_enabled = False
            self.adv_windowing_jitter = 0

        # Load stats dict
        with open(stats_path, "rb") as f:
            self.stats = pickle.load(f)
        print(f"Loaded stats for {len(self.stats)} cases")
        
        self.h5_files = sorted(list(self.root_dir.glob("*.h5")))

        # Filter by split if provided
        if split is not None:
            self._filter_by_split(split)

        # Limit number of cases (after split filtering)
        if max_cases is not None and len(self.h5_files) > max_cases:
            random.shuffle(self.h5_files)
            self.h5_files = sorted(self.h5_files[:max_cases])
            # Also filter stats to match
            valid_case_ids = {f.stem for f in self.h5_files}
            self.stats = {k: v for k, v in self.stats.items() if k in valid_case_ids}
            print(f"Limited to {max_cases} cases (after split filter)")

        # Build samples and context maps from HDF5 files
        self.samples = []
        self.label_to_cases: Dict[str, List[str]] = {}
        self.case_to_labels: Dict[str, set] = {}

        print(f"Scanning {len(self.h5_files)} HDF5 files...")
        for h5_file_path in tqdm(self.h5_files, desc="Scanning H5 files"):
            case_id = h5_file_path.stem
            if case_id not in self.stats:
                continue

            self.case_to_labels[case_id] = set()
            with h5py.File(h5_file_path, 'r') as h5f:
                for label_id in h5f.keys():
                    if label_id in label_id_set:
                        label_stats = self.stats.get(case_id, {}).get(label_id, {})
                        if not label_stats:
                            continue

                        # Support both old (slice_coverage) and new (num_slices) formats
                        num_slices = label_stats.get("num_slices", {})
                        slice_coverage = label_stats.get("slice_coverage", {})

                        # Check if any axis has slices
                        has_slices = False
                        for axis in self.axes:
                            n_slices = num_slices.get(axis, 0) if num_slices else (1 if slice_coverage.get(axis, 0) > 0 else 0)
                            if n_slices > 0:
                                has_slices = True
                                break

                        if has_slices:
                            self.label_to_cases.setdefault(label_id, []).append(case_id)
                            self.case_to_labels[case_id].add(label_id)

                            # Get slice coverages for filtering
                            slice_coverages = label_stats.get("slice_coverages", {})

                            for axis in self.axes:
                                n_slices = num_slices.get(axis, 0) if num_slices else (1 if slice_coverage.get(axis, 0) > 0 else 0)
                                axis_coverages = slice_coverages.get(axis, [])

                                # Filter slices by coverage (absolute min + ratio)
                                if axis_coverages:
                                    max_cov = max(axis_coverages) if axis_coverages else 0
                                    threshold = max(self.min_coverage, max_cov * self.min_coverage_ratio)
                                    for slice_idx in range(n_slices):
                                        if slice_idx < len(axis_coverages) and axis_coverages[slice_idx] >= threshold:
                                            self.samples.append((case_id, label_id, axis, slice_idx))
                                else:
                                    # No coverage data: include all slices
                                    for slice_idx in range(n_slices):
                                        self.samples.append((case_id, label_id, axis, slice_idx))

        # Map (label, axis) -> list of cases for context selection
        self.valid_contexts = {(label, axis): cases for label, cases in self.label_to_cases.items() for axis in self.axes}

        # Class-balanced sampling: group samples by label for uniform label selection
        self.class_balanced = class_balanced
        self.label_to_samples: Dict[str, List[Tuple[str, str, int]]] = {}
        for case_id, label_id, axis, slice_idx in self.samples:
            self.label_to_samples.setdefault(label_id, []).append((case_id, axis, slice_idx))
        self.active_labels = list(self.label_to_samples.keys())

        print(f"Built mapping for {len(self.label_to_cases)} labels.")
        print(f"Created {len(self.samples)} samples (modality={self.modality}, min_coverage={self.min_coverage}, min_coverage_ratio={self.min_coverage_ratio})")
        if self.class_balanced:
            label_counts = {l: len(s) for l, s in self.label_to_samples.items()}
            min_l = min(label_counts, key=label_counts.get)
            max_l = max(label_counts, key=label_counts.get)
            print(f"Class-balanced sampling: {len(self.active_labels)} labels, "
                  f"min={min_l}({label_counts[min_l]}), max={max_l}({label_counts[max_l]})")

        if self.max_ds_len is not None and len(self.samples) > self.max_ds_len:
            random.shuffle(self.samples)
            print(f"Shuffled samples")

        if self.random_coloring_nb > 0:
            self.color_palette = define_colors_by_mean_sep(num_colors=max(256, len(self.label_id_list)))

        # Per-worker H5 file cache (populated lazily, max 8 handles to bound RAM)
        self._h5_cache: Dict[str, h5py.File] = {}
        self._h5_cache_max = 8

    def _setup_unified_augmentation(self, cfg: Dict):
        """Setup augmentation from unified config format."""
        self.augment = True
        self.use_unified_augmentation = True

        # Check augmentation type
        aug_type = cfg.get("type", "legacy")
        self.aug_type = aug_type

        if aug_type == "universeg":
            self._setup_universeg_augmentation(cfg)
            return

        # Legacy unified augmentation
        self._setup_legacy_unified_augmentation(cfg)

    def _setup_universeg_augmentation(self, cfg: Dict):
        """Setup UniverSeg-style two-level augmentation."""
        self.universeg_full_config = cfg  # Store full config for apply_universeg_augmentation
        self.universeg_task_config = cfg.get("task", {})
        self.universeg_example_config = cfg.get("example", {})

        # Convert list configs to tuples for the augmentation functions
        for key in ["rotation_range", "scale_range", "elastic_alpha", "elastic_sigma",
                    "brightness_range", "contrast_range", "blur_sigma", "noise_mean", "noise_var"]:
            if key in self.universeg_task_config and isinstance(self.universeg_task_config[key], list):
                self.universeg_task_config[key] = tuple(self.universeg_task_config[key])
            if key in self.universeg_example_config and isinstance(self.universeg_example_config[key], list):
                self.universeg_example_config[key] = tuple(self.universeg_example_config[key])

        # Disable mix augmentation for UniverSeg style (uses its own augmentation)
        self.mix_type = "none"
        self.mix_probability = 0.0

        # Disable other legacy augmentation components
        self.spatial_enabled = False
        self.spatial_transform = None
        self.intensity_enabled = False
        self.intensity_transform = None
        self.fg_crop_enabled = False
        self.degrade_enabled = False
        self.mask_perturb_enabled = False
        self.task_level_enabled = False
        self.adv_windowing_jitter = 0

        print(f"UniverSeg-style augmentation enabled:")
        print(f"  Task-level: flip_intensities={self.universeg_task_config.get('flip_intensities_p', 0.5):.1f}, "
              f"flip_labels={self.universeg_task_config.get('flip_labels_p', 0.5):.1f}, "
              f"sobel_edge={self.universeg_task_config.get('sobel_edge_p', 0.5):.1f}")
        print(f"  Task-level: affine={self.universeg_task_config.get('affine_p', 0.5):.1f}, "
              f"rotation={self.universeg_task_config.get('rotation_range', (0, 360))}")
        print(f"  Example-level: enabled={self.universeg_example_config.get('enabled', True)}, "
              f"elastic_p={self.universeg_example_config.get('elastic_p', 0.8):.1f}")

    def _setup_legacy_unified_augmentation(self, cfg: Dict):
        """Setup legacy unified augmentation (pre-UniverSeg style)."""
        # Mix augmentation (CutMix or CarveMix)
        mix_cfg = cfg.get("mix", {})
        self.mix_type = mix_cfg.get("type", "none")  # "cutmix", "carve_mix", "none"
        self.mix_probability = mix_cfg.get("probability", 0.5)
        # CutMix params
        cutmix_cfg = mix_cfg.get("cutmix", {})
        self.cutmix_min_ratio = cutmix_cfg.get("min_ratio", 0.3)
        self.cutmix_max_ratio = cutmix_cfg.get("max_ratio", 0.7)
        # CarveMix params
        carve_cfg = mix_cfg.get("carve_mix", {})
        self.carve_mix_margin_range = tuple(carve_cfg.get("margin_range", [0.1, 0.5]))
        self.carve_mix_harmonize = carve_cfg.get("harmonize", True)
        self.carve_mix_harmonize_sigma = carve_cfg.get("harmonize_sigma", 5.0)

        # Spatial augmentation
        spatial_cfg = cfg.get("spatial", {})
        self.spatial_enabled = spatial_cfg.get("enabled", True)
        if self.spatial_enabled:
            self.spatial_transform = create_spatial_only_transform(
                rotation_limit=spatial_cfg.get("rotation_limit", 15.0),
                scale_limit=spatial_cfg.get("scale_limit", 0.1),
                elastic_alpha=spatial_cfg.get("elastic_alpha", 50.0),
                elastic_sigma=spatial_cfg.get("elastic_sigma", 5.0),
            )
        else:
            self.spatial_transform = None

        # Foreground crop config (within spatial)
        fg_crop_cfg = spatial_cfg.get("foreground_crop", {})
        self.fg_crop_enabled = fg_crop_cfg.get("enabled", True)
        self.fg_crop_probability = fg_crop_cfg.get("probability", 0.3)
        self.fg_crop_min_frac = fg_crop_cfg.get("min_crop_frac", 0.5)
        self.fg_crop_disable_with_mix = fg_crop_cfg.get("disable_with_mix", True)

        # Intensity augmentation (single unified pipeline)
        intensity_cfg = cfg.get("intensity", {})
        self.intensity_enabled = intensity_cfg.get("enabled", True)
        if self.intensity_enabled:
            img_size = self.image_size[0] if self.image_size else 512
            self.intensity_transform = create_intensity_only_transform(
                brightness_limit=intensity_cfg.get("brightness_limit", 0.1),
                contrast_limit=intensity_cfg.get("contrast_limit", 0.1),
                gamma_limit=tuple(intensity_cfg.get("gamma_limit", [80, 120])),
                noise_std_range=tuple(intensity_cfg.get("noise_std_range", [0.02, 0.05])),
                img_size=img_size,
            )
            self.intensity_asymmetric = intensity_cfg.get("asymmetric", True)
            self.intensity_asymmetric_p = intensity_cfg.get("asymmetric_probability", 0.5)
        else:
            self.intensity_transform = None
            self.intensity_asymmetric = False
            self.intensity_asymmetric_p = 0.0

        # Resolution degradation
        degrade_cfg = cfg.get("resolution_degradation", {})
        self.degrade_enabled = degrade_cfg.get("enabled", True)
        self.degrade_probability = degrade_cfg.get("probability", 0.2)
        self.degrade_min_scale = degrade_cfg.get("min_scale", 0.25)

        # Context mask perturbation
        context_cfg = cfg.get("context", {})
        mask_perturb_cfg = context_cfg.get("mask_perturbation", {})
        self.mask_perturb_enabled = mask_perturb_cfg.get("enabled", True)
        self.mask_perturb_probability = mask_perturb_cfg.get("probability", 0.3)
        self.mask_perturb_max_kernel = mask_perturb_cfg.get("max_kernel", 5)

        # Task-level augmentation (same transform to all images)
        task_cfg = cfg.get("task_level", {})
        self.task_level_enabled = task_cfg.get("enabled", True)
        self.task_level_probability = task_cfg.get("probability", 0.3)
        task_transforms = task_cfg.get("transforms", {})
        self.task_flip_h = task_transforms.get("flip_horizontal", True)
        self.task_flip_v = task_transforms.get("flip_vertical", True)
        self.task_rotate_90 = task_transforms.get("rotate_90", True)

        # Windowing jitter (CT-specific)
        self.adv_windowing_jitter = cfg.get("windowing_jitter", 0)

        # Disable mix for random_coloring mode
        if self.random_coloring_nb > 0:
            self.mix_type = "none"

        # Print configuration summary
        print(f"Legacy unified augmentation enabled:")
        print(f"  Mix: type={self.mix_type}, p={self.mix_probability}")
        print(f"  Spatial: enabled={self.spatial_enabled}")
        print(f"  Foreground crop: enabled={self.fg_crop_enabled}, p={self.fg_crop_probability}, disable_with_mix={self.fg_crop_disable_with_mix}")
        print(f"  Intensity: enabled={self.intensity_enabled}, asymmetric={self.intensity_asymmetric}")
        print(f"  Resolution degradation: enabled={self.degrade_enabled}, p={self.degrade_probability}")
        print(f"  Context mask perturbation: enabled={self.mask_perturb_enabled}, p={self.mask_perturb_probability}")
        print(f"  Task-level: enabled={self.task_level_enabled}, p={self.task_level_probability}")
    def _filter_by_split(self, split: Union[str, List[str]]):
        """Filter cases by train/val/test split using meta.csv."""
        import pandas as pd

        # Assuming meta.csv is in a standard location relative to the HDF5 dir
        meta_path = self.root_dir.parent / "totalseg" / "meta.csv"
        if not meta_path.exists():
            print(f"Warning: meta.csv not found at {meta_path}, skipping split filter")
            return

        df = pd.read_csv(meta_path, sep=";")
        if isinstance(split, str):
            split_case_ids = set(df["image_id"][df["split"] == split].tolist())
        else:
            split_case_ids = set(df["image_id"][df["split"].isin(split)].tolist())
        
        # Filter stats and h5_files
        self.stats = {k: v for k, v in self.stats.items() if k in split_case_ids}
        self.h5_files = [f for f in self.h5_files if f.stem in split_case_ids]
        print(f"Filtered to {len(self.h5_files)} cases for split '{split}'")
        # (This is a simplified version of the logic in __init__)

    def _get_h5_file(self, case_id: str) -> h5py.File:
        """Get cached H5 file handle (lazy initialization per worker, LRU-bounded)."""
        if case_id not in self._h5_cache:
            # Evict oldest entry if at capacity
            if len(self._h5_cache) >= self._h5_cache_max:
                oldest = next(iter(self._h5_cache))
                try:
                    self._h5_cache[oldest].close()
                except Exception:
                    pass
                del self._h5_cache[oldest]
            h5_path = self.root_dir / f"{case_id}.h5"
            self._h5_cache[case_id] = h5py.File(h5_path, 'r', swmr=True)
        else:
            # Move to end (most-recently-used) using dict insertion order
            self._h5_cache[case_id] = self._h5_cache.pop(case_id)
        return self._h5_cache[case_id]

    def _load_slice(self, case_id: str, label_id: str, axis: str, slice_idx: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """Load image and label slice from HDF5 file (cached handle).

        Supports both old format (2D arrays with keys like '{label}/{axis}_slice_img')
        and new format (3D arrays with keys like '{label}/{axis}_slices_img').
        """
        h5f = self._get_h5_file(case_id)

        # Try new format first (3D arrays)
        new_img_key = f"{label_id}/{axis}_slices_img"
        new_mask_key = f"{label_id}/{axis}_slices"

        if new_img_key in h5f:
            img = h5f[new_img_key][slice_idx]
            mask = h5f[new_mask_key][slice_idx]
        else:
            # Fall back to old format (2D arrays, ignore slice_idx)
            img = h5f[f"{label_id}/{axis}_slice_img"][:]
            mask = h5f[f"{label_id}/{axis}_slice"][:]

        return img, mask

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
        for donor_case_id in candidates[:3]:
            try:
                # Get number of slices for this donor
                donor_stats = self.stats.get(donor_case_id, {}).get(label_id, {})
                num_slices = donor_stats.get("num_slices", {})
                n_donor_slices = num_slices.get(axis, 1) if num_slices else 1
                donor_slice_idx = random.randint(0, n_donor_slices - 1) if n_donor_slices > 1 else 0

                donor_img, donor_mask = self._load_slice(donor_case_id, label_id, axis, donor_slice_idx)
                if donor_mask.max() == 0:
                    continue
                # Apply same preprocessing as __getitem__
                if self.crop_to_bbox:
                    bbox = self._get_2d_bbox(donor_case_id, label_id, axis)
                    donor_img, donor_mask = self._crop_to_bbox(donor_img, donor_mask, bbox)
                donor_img = self._normalize_image(donor_img)
                if self.image_size:
                    donor_img = self._resize(donor_img, self.image_size, "bilinear")
                    donor_mask = self._resize_mask(donor_mask, self.image_size)
                if donor_mask.max() == 0:
                    continue
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

    def _apply_mix_augmentation(
        self,
        target_img: np.ndarray,
        target_mask: np.ndarray,
        label_id: str,
        axis: str,
        exclude_case_ids: List[str],
    ) -> Tuple[np.ndarray, np.ndarray, bool]:
        """Apply mix augmentation (CutMix or CarveMix) based on unified config.

        Returns the mixed image/mask and a flag indicating if mix was applied.
        """
        if self.mix_type == "none" or random.random() > self.mix_probability:
            return target_img, target_mask, False

        donor = self._load_carve_mix_donor(label_id, axis, exclude_case_ids)
        if donor is None:
            return target_img, target_mask, False

        donor_img, donor_mask = donor

        if self.mix_type == "cutmix":
            mixed_img, mixed_mask = cut_mix_2d(
                target_img, target_mask, donor_img, donor_mask,
                min_ratio=self.cutmix_min_ratio,
                max_ratio=self.cutmix_max_ratio,
            )
        elif self.mix_type == "carve_mix":
            mixed_img, mixed_mask = carve_mix_2d(
                target_img, target_mask, donor_img, donor_mask,
                margin_range=self.carve_mix_margin_range,
                harmonize=self.carve_mix_harmonize,
                harmonize_sigma=self.carve_mix_harmonize_sigma,
            )
        else:
            return target_img, target_mask, False

        return mixed_img, mixed_mask, True

    def _apply_task_level_augmentation(
        self,
        target_img: np.ndarray,
        target_mask: np.ndarray,
        context_imgs: List[np.ndarray],
        context_masks: List[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], List[np.ndarray]]:
        """Apply the same random transform to ALL images (UniverSeg-style)."""
        if not self.task_level_enabled or random.random() > self.task_level_probability:
            return target_img, target_mask, context_imgs, context_masks

        all_imgs = [target_img] + context_imgs
        all_masks = [target_mask] + context_masks

        all_imgs, all_masks = apply_task_level_augmentation(
            all_imgs, all_masks,
            flip_horizontal=self.task_flip_h,
            flip_vertical=self.task_flip_v,
            rotate_90=self.task_rotate_90,
        )

        return all_imgs[0], all_masks[0], all_imgs[1:], all_masks[1:]

    def _apply_spatial_augmentation(
        self,
        img: np.ndarray,
        mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply spatial-only augmentation (no intensity changes)."""
        if self.spatial_transform is None:
            return img, mask

        img_uint8 = (img * 255).astype(np.uint8)
        mask_uint8 = (mask * 255).astype(np.uint8)
        result = self.spatial_transform(image=img_uint8, mask=mask_uint8)
        return result['image'].astype(np.float32) / 255.0, (result['mask'] > 127).astype(np.float32)

    def _apply_intensity_augmentation(
        self,
        img: np.ndarray,
    ) -> np.ndarray:
        """Apply intensity-only augmentation."""
        if self.intensity_transform is None:
            return img

        img_uint8 = (img * 255).astype(np.uint8)
        result = self.intensity_transform(image=img_uint8)
        return result['image'].astype(np.float32) / 255.0

    def _apply_universeg_augmentation(
        self,
        target_img: np.ndarray,
        target_mask: np.ndarray,
        context_imgs: List[np.ndarray],
        context_masks: List[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], List[np.ndarray]]:
        """Apply UniverSeg-style two-level augmentation.

        1. Task-level: Same transform applied to all images (target + context)
        2. Example-level: Different transform per context image
        """
        return apply_universeg_augmentation(
            target_img=target_img,
            target_mask=target_mask,
            context_imgs=context_imgs,
            context_masks=context_masks,
            full_config=self.universeg_full_config,
        )

    def _apply_unified_augmentation(
        self,
        target_img: np.ndarray,
        target_mask: np.ndarray,
        context_imgs: List[np.ndarray],
        context_masks: List[np.ndarray],
        label_id: str,
        axis: str,
        exclude_case_ids: List[str],
    ) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], List[np.ndarray]]:
        """Apply the unified augmentation pipeline.

        Flow (legacy):
        1. Mix augmentation (CutMix OR CarveMix) on target
        2. Foreground crop (only if mix not applied, based on config)
        3. Re-resize if shapes changed
        4. Task-level augmentation (same transform to ALL)
        5. Spatial augmentation (different per image)
        6. Intensity augmentation (different per image, or skip based on asymmetric setting)
        7. Context mask perturbation
        8. Resolution degradation

        Flow (UniverSeg):
        1. Task-level augmentation (same transform to ALL, includes intensity/spatial)
        2. Example-level augmentation (different per context image)
        """
        # Check if using UniverSeg-style augmentation
        if getattr(self, 'aug_type', 'legacy') == "universeg":
            return self._apply_universeg_augmentation(
                target_img, target_mask, context_imgs, context_masks
            )

        # Legacy unified augmentation flow
        # 1. Mix augmentation on target
        target_img, target_mask, mix_applied = self._apply_mix_augmentation(
            target_img, target_mask, label_id, axis, exclude_case_ids
        )

        # 2. Foreground crop (skip if mix applied and disable_with_mix is True)
        if self.fg_crop_enabled:
            should_crop = not (mix_applied and self.fg_crop_disable_with_mix)
            if should_crop and random.random() < self.fg_crop_probability:
                target_img, target_mask = foreground_random_crop(target_img, target_mask, self.fg_crop_min_frac)
                # Also apply to context (independent probability)
                for i in range(len(context_imgs)):
                    if random.random() < self.fg_crop_probability:
                        context_imgs[i], context_masks[i] = foreground_random_crop(
                            context_imgs[i], context_masks[i], self.fg_crop_min_frac
                        )

        # 3. Re-resize if shapes changed
        if self.image_size is not None:
            tgt_size = tuple(self.image_size)
            if target_img.shape[:2] != tgt_size:
                target_img = self._resize(target_img, self.image_size, mode="bilinear")
                target_mask = self._resize_mask(target_mask, self.image_size)
            context_imgs = [self._resize(c, self.image_size, "bilinear") if c.shape[:2] != tgt_size else c for c in context_imgs]
            context_masks = [self._resize_mask(c, self.image_size) if c.shape[:2] != tgt_size else c for c in context_masks]

        # 4. Task-level augmentation (same transform to ALL)
        target_img, target_mask, context_imgs, context_masks = self._apply_task_level_augmentation(
            target_img, target_mask, context_imgs, context_masks
        )

        # 5. Spatial augmentation (different per image)
        if self.spatial_enabled:
            target_img, target_mask = self._apply_spatial_augmentation(target_img, target_mask)
            for i in range(len(context_imgs)):
                context_imgs[i], context_masks[i] = self._apply_spatial_augmentation(context_imgs[i], context_masks[i])

        # 6. Intensity augmentation (single unified pipeline)
        if self.intensity_enabled:
            # Apply to target
            if not self.intensity_asymmetric or random.random() < self.intensity_asymmetric_p:
                target_img = self._apply_intensity_augmentation(target_img)
            # Apply to context (independent probability if asymmetric)
            for i in range(len(context_imgs)):
                if not self.intensity_asymmetric or random.random() < self.intensity_asymmetric_p:
                    context_imgs[i] = self._apply_intensity_augmentation(context_imgs[i])

        # 7. Context mask perturbation
        if self.mask_perturb_enabled:
            for i in range(len(context_masks)):
                if random.random() < self.mask_perturb_probability:
                    context_masks[i] = perturb_mask(context_masks[i], self.mask_perturb_max_kernel)

        # 8. Resolution degradation
        if self.degrade_enabled:
            if random.random() < self.degrade_probability:
                target_img = degrade_resolution(target_img, self.degrade_min_scale)
            for i in range(len(context_imgs)):
                if random.random() < self.degrade_probability:
                    context_imgs[i] = degrade_resolution(context_imgs[i], self.degrade_min_scale)

        return target_img, target_mask, context_imgs, context_masks

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sample with context examples (on-the-fly from HDF5)."""
        if self.class_balanced:
            # Two-stage sampling: pick label uniformly, then pick a random sample
            label_id = random.choice(self.active_labels)
            target_case_id, axis, slice_idx = random.choice(self.label_to_samples[label_id])
        else:
            target_case_id, label_id, axis, slice_idx = self.samples[idx]

        return self._getitem_unified(target_case_id, label_id, axis, slice_idx)

    def _getitem_unified(self, target_case_id: str, label_id: str, axis: str, slice_idx: int = 0) -> Dict[str, torch.Tensor]:
        """Get item using unified augmentation pipeline."""
        # 1. Load target slice
        img, mask = self._load_slice(target_case_id, label_id, axis, slice_idx)

        # 2. Bbox crop and normalization
        if self.crop_to_bbox:
            bbox = self._get_2d_bbox(target_case_id, label_id, axis)
            img, mask = self._crop_to_bbox(img, mask, bbox)

        jitter = getattr(self, 'adv_windowing_jitter', 0)
        img = self._normalize_image(img, jitter=jitter)

        # 3. Initial resize
        if self.image_size:
            img = self._resize(img, self.image_size, "bilinear")
            mask = self._resize_mask(mask, self.image_size)

        # 3b. Warn if mask is empty before augmentation (data issue, not augmentation)
        if mask.max() < 0.5:
            import warnings
            warnings.warn(
                f"Empty mask before augmentation: {target_case_id}/{label_id}/{axis}/slice{slice_idx}",
                stacklevel=2
            )
            retry_idx = random.randint(0, len(self.samples) - 1)
            return self.__getitem__(retry_idx)

        # 4. Load context
        context_imgs, context_labels, valid_context_ids = [], [], []
        if self.context_size > 0:
            context_key = (label_id, axis)
            available_contexts = [c for c in self.valid_contexts.get(context_key, []) if c != target_case_id]

            # Apply diversity selection or random shuffle
            if self.context_diversity_type == 'farthest':
                available_contexts = self._select_diverse_contexts(
                    available_contexts, target_case_id, label_id, self.context_size
                )
            elif self.random_context:
                random.shuffle(available_contexts)

            for ctx_case_id in available_contexts:
                if len(context_imgs) >= self.context_size:
                    break
                try:
                    # Get number of slices for this context
                    ctx_stats = self.stats.get(ctx_case_id, {}).get(label_id, {})
                    num_slices = ctx_stats.get("num_slices", {})
                    n_ctx_slices = num_slices.get(axis, 1) if num_slices else 1
                    ctx_slice_idx = random.randint(0, n_ctx_slices - 1) if n_ctx_slices > 1 else 0

                    ctx_img, ctx_label = self._load_slice(ctx_case_id, label_id, axis, ctx_slice_idx)
                    if ctx_label.max() == 0:
                        continue

                    if self.crop_to_bbox:
                        bbox = self._get_2d_bbox(ctx_case_id, label_id, axis)
                        ctx_img, ctx_label = self._crop_to_bbox(ctx_img, ctx_label, bbox)

                    ctx_img = self._normalize_image(ctx_img, jitter=jitter)

                    if self.image_size:
                        ctx_img = self._resize(ctx_img, self.image_size, "bilinear")
                        ctx_label = self._resize_mask(ctx_label, self.image_size)

                    context_imgs.append(ctx_img)
                    context_labels.append(ctx_label)
                    valid_context_ids.append(ctx_case_id)
                except Exception as e:
                    print(f"Warning: Failed to load context {ctx_case_id}/{label_id}/{axis}: {e}")
                    continue

            if len(context_imgs) == 0:
                raise RuntimeError(f"Failed to load any context for label '{label_id}' axis '{axis}'")

        # 5. Apply unified augmentation
        exclude_ids = [target_case_id] + valid_context_ids
        img, mask, context_imgs, context_labels = self._apply_unified_augmentation(
            img, mask, context_imgs, context_labels, label_id, axis, exclude_ids
        )

        # 5b. Retry if target mask became empty after augmentation (spatial transforms can push small objects out)
        if mask.max() < 0.5:
            retry_idx = random.randint(0, len(self.samples) - 1)
            return self.__getitem__(retry_idx)

        # 6. Convert to tensors
        target_in = torch.from_numpy(img.copy()).unsqueeze(0).float()
        target_out = torch.from_numpy(mask.copy()).unsqueeze(0).float()

        if self.context_size == 0:
            return {
                "image": target_in, "label": target_out,
                "target_case_id": target_case_id, "label_id": label_id, "axis": axis,
            }

        context_in = torch.stack([torch.from_numpy(c.copy()).unsqueeze(0).float() for c in context_imgs])
        context_out = torch.stack([torch.from_numpy(c.copy()).unsqueeze(0).float() for c in context_labels])

        return {
            "image": target_in, "label": target_out,
            "context_in": context_in, "context_out": context_out,
            "target_case_id": target_case_id, "context_case_ids": valid_context_ids,
            "label_id": label_id, "axis": axis,
        }

    def _select_diverse_contexts(
        self,
        available_contexts: List[str],
        target_case_id: str,
        label_id: str,
        n_contexts: int,
    ) -> List[str]:
        """Select diverse contexts using farthest-point sampling.

        Requires pre-computed mean features in stats file under `mean_features` key.
        Falls back to random selection if features unavailable.

        Args:
            available_contexts: List of candidate case IDs
            target_case_id: Current target case ID
            label_id: Label being segmented
            n_contexts: Number of contexts to select

        Returns:
            List of selected case IDs
        """
        if self.context_diversity_type == 'random' or len(available_contexts) <= n_contexts:
            # Random: shuffle and take first n
            random.shuffle(available_contexts)
            return available_contexts[:n_contexts]

        # Get features for candidates
        candidates = available_contexts[:self.context_diversity_candidates]
        features = []
        valid_candidates = []

        for case_id in candidates:
            case_stats = self.stats.get(case_id, {}).get(label_id, {})
            feat = case_stats.get(self.context_feature_key)
            if feat is not None:
                features.append(np.array(feat))
                valid_candidates.append(case_id)

        # Fall back to random if no features available
        if len(features) < n_contexts:
            random.shuffle(available_contexts)
            return available_contexts[:n_contexts]

        features = np.stack(features)  # [N, D]

        # Farthest-point sampling
        selected_indices = []
        distances = np.full(len(features), np.inf)

        # Start with random seed point
        first_idx = random.randint(0, len(features) - 1)
        selected_indices.append(first_idx)

        for _ in range(n_contexts - 1):
            if len(selected_indices) >= len(features):
                break
            # Update distances to nearest selected point
            last_feat = features[selected_indices[-1]]
            dists_to_last = np.linalg.norm(features - last_feat, axis=1)
            distances = np.minimum(distances, dists_to_last)
            distances[selected_indices] = -np.inf  # Already selected

            # Select point farthest from all selected points
            next_idx = np.argmax(distances)
            selected_indices.append(next_idx)

        return [valid_candidates[i] for i in selected_indices]

    def _get_2d_bbox(self, case_id: str, label_id: str, axis: str) -> Tuple[int, int, int, int]:
        bbox_3d = self.stats[case_id][label_id]["bbox"]
        z_range, y_range, x_range = bbox_3d
        if axis == "z": return (y_range[0], y_range[1], x_range[0], x_range[1])
        elif axis == "y": return (z_range[0], z_range[1], x_range[0], x_range[1])
        else: return (z_range[0], z_range[1], y_range[0], y_range[1])

    def _crop_to_bbox(self, img: np.ndarray, label: np.ndarray, bbox: Tuple[int, int, int, int]) -> Tuple[np.ndarray, np.ndarray]:
        row_min, row_max, col_min, col_max = bbox
        h, w = img.shape
        row_min = max(0, row_min - self.bbox_padding)
        row_max = min(h, row_max + self.bbox_padding + 1)
        col_min = max(0, col_min - self.bbox_padding)
        col_max = min(w, col_max + self.bbox_padding + 1)
        return img[row_min:row_max, col_min:col_max], label[row_min:row_max, col_min:col_max]

    def _normalize_image(self, img: np.ndarray, jitter: float = 0) -> np.ndarray:
        """Normalize image based on modality.

        CT: Clip to [-500, 1000], min-max normalize to [0, 1].
        MRI: Clip to [0.5, 99.5] percentiles of non-zero voxels, min-max normalize to [0, 1].
        """
        if self.modality == "mri":
            # MRI: percentile-based normalization on non-zero voxels
            nonzero_mask = img > 0
            if nonzero_mask.any():
                nonzero_vals = img[nonzero_mask]
                a_min = np.percentile(nonzero_vals, 0.5)
                a_max = np.percentile(nonzero_vals, 99.5)
            else:
                a_min, a_max = img.min(), img.max()
            # Ensure valid range
            if a_max - a_min < 1e-6:
                a_max = a_min + 1.0
        else:
            # CT: fixed window [-500, 1000]
            a_min, a_max = -500.0, 1000.0
            if jitter > 0:
                a_min = a_min + random.uniform(-jitter, jitter)
                a_max = a_max + random.uniform(-jitter, jitter)
                if a_max - a_min < 200:
                    a_max = a_min + 200

        img = np.clip(img, a_min, a_max)
        img = (img - a_min) / (a_max - a_min)
        return img

    def _resize(self, arr: np.ndarray, size: Tuple[int, int], mode: str = "bilinear") -> np.ndarray:
        tensor = torch.from_numpy(arr.copy()).unsqueeze(0).unsqueeze(0)
        resized = torch.nn.functional.interpolate(tensor, size=size, mode=mode, align_corners=False if mode == "bilinear" else None)
        return resized.squeeze().numpy()

    def _resize_mask(self, mask: np.ndarray, size: Tuple[int, int], min_value: float = 0.5) -> np.ndarray:
        """Resize mask with hybrid approach: preserves small objects.

        Uses area interpolation for soft targets, but ensures small objects get a minimum
        value via max pooling. This prevents tiny labels from vanishing when downscaling.
        """
        tensor = torch.from_numpy(mask.copy()).unsqueeze(0).unsqueeze(0).float()

        # Area interpolation: soft area fractions
        area_resized = torch.nn.functional.interpolate(tensor, size=size, mode='area')

        # Max pooling: preserves presence of small objects
        max_resized = torch.nn.functional.adaptive_max_pool2d(tensor, size)

        # Hybrid: use area value, but ensure minimum where max detects foreground
        result = torch.maximum(area_resized, max_resized * min_value)

        return result.squeeze().numpy()

    def __len__(self) -> int:
        if self.max_ds_len is not None:
            return min(self.max_ds_len, len(self.samples))
        return len(self.samples)

    def __del__(self):
        """Close cached H5 file handles on cleanup."""
        for h5f in self._h5_cache.values():
            try:
                h5f.close()
            except Exception:
                pass
        self._h5_cache.clear()

# Collate function
def collate_fn(batch: List[Dict]) -> Dict[str, Union[torch.Tensor, List]]:
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

    # Add context data if present, handling cases where some items might not have context
    if any("context_in" in item for item in batch):
        # Find an item with context to determine shapes
        prototype_item = next(item for item in batch if "context_in" in item)
        context_size, _, h, w = prototype_item["context_in"].shape
        _, c_out, _, _ = prototype_item["context_out"].shape

        # Create default tensors for padding
        default_in = torch.zeros(context_size, 1, h, w)
        default_out = torch.zeros(context_size, c_out, h, w)
        default_ids = ["PAD"] * context_size

        result["context_in"] = torch.stack([item.get("context_in", default_in) for item in batch])
        result["context_out"] = torch.stack([item.get("context_out", default_out) for item in batch])
        result["context_case_ids"] = [item.get("context_case_ids", default_ids) for item in batch]


    # Add color_map if present (random coloring mode)
    if "color_map" in batch[0]:
        result["color_map"] = [item["color_map"] for item in batch]

    return result

# get_dataloader needs to be updated to use the new Dataset class name if it was changed
# but since we are overwriting, we keep the name and it works.
def get_dataloader(
    root_dir: str,
    stats_path: str,
    label_id_list: Union[List[str], str],
    **kwargs,
) -> DataLoader:
    """Instantiates and returns a DataLoader for the TotalSeg2DDataset."""
    dataset_kwargs = kwargs.copy()
    batch_size = dataset_kwargs.pop("batch_size", 8)
    shuffle = dataset_kwargs.pop("shuffle", True)
    num_workers = dataset_kwargs.pop("num_workers", 4)

    dataset = TotalSeg2DDataset(
        root_dir=root_dir,
        stats_path=stats_path,
        label_id_list=label_id_list,
        **dataset_kwargs,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=False,
        prefetch_factor=1 if num_workers > 0 else None,  # halves shared-memory buffer vs default 2
    )
