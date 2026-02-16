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
        augment: bool = False,
        augment_config: Optional[Dict] = None,
        carve_mix: bool = False,
        carve_mix_config: Optional[Dict] = None,
        advanced_augmentation: bool = False,
        advanced_augmentation_config: Optional[Dict] = None,
        max_labels: Optional[int] = None,
        class_balanced: bool = False,
    ):
        self.root_dir = Path(root_dir)

        # Resolve label_id_list
        if isinstance(label_id_list, str):
            self.label_id_list = get_label_ids(label_id_list, max_labels=max_labels)
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

        # Augmentation setup (remains the same)
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
        
        self.h5_files = sorted(list(self.root_dir.glob("*.h5")))

        # Filter by split if provided
        if split is not None:
            self._filter_by_split(split)

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
                        # Check if this label has non-zero coverage from stats
                        coverage_ok = False
                        if self.stats.get(case_id, {}).get(label_id, {}):
                             slice_coverage = self.stats[case_id][label_id].get("slice_coverage", {})
                             if any(slice_coverage.get(axis, 0) > 0 for axis in self.axes):
                                coverage_ok = True

                        if coverage_ok:
                            self.label_to_cases.setdefault(label_id, []).append(case_id)
                            self.case_to_labels[case_id].add(label_id)
                            for axis in self.axes:
                                 if slice_coverage.get(axis, 0) > 0:
                                    self.samples.append((case_id, label_id, axis))

        # This is a simplified valid_contexts. Original checked coverage, which we now do above.
        self.valid_contexts = {(label, axis): cases for label, cases in self.label_to_cases.items() for axis in self.axes}

        # Class-balanced sampling: group samples by label for uniform label selection
        self.class_balanced = class_balanced
        self.label_to_samples: Dict[str, List[Tuple[str, str]]] = {}
        for case_id, label_id, axis in self.samples:
            self.label_to_samples.setdefault(label_id, []).append((case_id, axis))
        self.active_labels = list(self.label_to_samples.keys())

        print(f"Built mapping for {len(self.label_to_cases)} labels.")
        print(f"Created {len(self.samples)} samples")
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
    
    # --- Methods like _filter_by_split, _find_cases_with_labels, etc. remain largely the same ---
    def _filter_by_split(self, split: Union[str, List[str]]):
        """Filter cases by train/val/test split using meta.csv."""
        import pandas as pd

        # Assuming meta.csv is in a standard location relative to the HDF5 dir
        meta_path = self.root_dir.parent / "TotalSeg" / "meta.csv"
        if not meta_path.exists():
            print(f"Warning: meta.csv not found at {meta_path}, skipping split filter")
            return

        df = pd.read_csv(meta_path, sep=";")
        if isinstance(split, str):
            split_case_ids = set(df["image_id"][df["split"] == split].tolist())
        else:
            split_case_ids = set(df["image_id"][df["split"].isin(split)].tolist())
        
        # Filter stats, h5_files, and mappings
        self.stats = {k: v for k, v in self.stats.items() if k in split_case_ids}
        # Rebuild mappings after filtering
        # (This is a simplified version of the logic in __init__)

    def _load_slice(self, case_id: str, label_id: str, axis: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load image and label slice from HDF5 file on-the-fly."""
        h5_path = self.root_dir / f"{case_id}.h5"
        with h5py.File(h5_path, 'r') as h5f:
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
                donor_img, donor_mask = self._load_slice(donor_case_id, label_id, axis)
                if donor_mask.max() == 0:
                    continue
                # Apply same preprocessing as __getitem__
                if self.crop_to_bbox:
                    bbox = self._get_2d_bbox(donor_case_id, label_id, axis)
                    donor_img, donor_mask = self._crop_to_bbox(donor_img, donor_mask, bbox)
                donor_img = self._normalize_image(donor_img)
                if self.image_size:
                    donor_img = self._resize(donor_img, self.image_size, "bilinear")
                    donor_mask = self._resize(donor_mask, self.image_size, "nearest")
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
        """Get a sample with context examples (on-the-fly from HDF5)."""
        if self.class_balanced:
            # Two-stage sampling: pick label uniformly, then pick a random sample
            label_id = random.choice(self.active_labels)
            target_case_id, axis = random.choice(self.label_to_samples[label_id])
        else:
            target_case_id, label_id, axis = self.samples[idx]

        # Load target slice
        img, mask = self._load_slice(target_case_id, label_id, axis)

        # Bbox crop (if enabled)
        if self.crop_to_bbox:
            bbox = self._get_2d_bbox(target_case_id, label_id, axis)
            img, mask = self._crop_to_bbox(img, mask, bbox)

        # Normalize image
        jitter = self.adv_windowing_jitter if self.advanced_augmentation else 0
        img = self._normalize_image(img, jitter=jitter)

        # Resize (if enabled)
        if self.image_size:
            img = self._resize(img, self.image_size, "bilinear")
            mask = self._resize(mask, self.image_size, "nearest")

        # Foreground random crop
        if self.advanced_augmentation and random.random() < self.adv_fg_crop_p:
            img, mask = foreground_random_crop(img, mask, self.adv_fg_crop_min)

        # Skip context loading if context_size is 0
        if self.context_size == 0:
            # Re-resize if foreground crop changed shape
            if self.image_size is not None and img.shape[:2] != tuple(self.image_size):
                img = self._resize(img, self.image_size, mode="bilinear")
                mask = self._resize(mask, self.image_size, mode="nearest")

            # CarveMix on target
            img, mask = self._apply_carve_mix(img, mask, label_id, axis, [target_case_id])

            # Advanced augmentation
            img, mask, _, _ = self._apply_advanced_augmentation(img, mask)

            # Apply augmentation
            if self.augment:
                img, mask = self._apply_augmentation(img, mask)

            target_in = torch.from_numpy(img.copy()).unsqueeze(0).float()
            target_out = torch.from_numpy(mask.copy()).unsqueeze(0).float()
            return {
                "image": target_in, "label": target_out,
                "target_case_id": target_case_id, "label_id": label_id, "axis": axis,
            }

        # Load context
        context_imgs, context_labels, valid_context_ids = [], [], []
        context_key = (label_id, axis)
        available_contexts = [c for c in self.valid_contexts.get(context_key, []) if c != target_case_id]
        if self.random_context:
            random.shuffle(available_contexts)

        for ctx_case_id in available_contexts:
            if len(context_imgs) >= self.context_size:
                break
            try:
                ctx_img, ctx_label = self._load_slice(ctx_case_id, label_id, axis)
                if ctx_label.max() == 0:
                    continue

                if self.crop_to_bbox:
                    bbox = self._get_2d_bbox(ctx_case_id, label_id, axis)
                    ctx_img, ctx_label = self._crop_to_bbox(ctx_img, ctx_label, bbox)

                ctx_img = self._normalize_image(ctx_img, jitter=jitter)

                if self.image_size:
                    ctx_img = self._resize(ctx_img, self.image_size, "bilinear")
                    ctx_label = self._resize(ctx_label, self.image_size, "nearest")

                # Foreground random crop (independent per context)
                if self.advanced_augmentation and random.random() < self.adv_fg_crop_p:
                    ctx_img, ctx_label = foreground_random_crop(
                        ctx_img, ctx_label, self.adv_fg_crop_min
                    )

                context_imgs.append(ctx_img)
                context_labels.append(ctx_label)
                valid_context_ids.append(ctx_case_id)
            except Exception as e:
                print(f"Warning: Failed to load context {ctx_case_id}/{label_id}/{axis}: {e}")
                continue

        if len(context_imgs) == 0:
            raise RuntimeError(f"Failed to load any context for label '{label_id}' axis '{axis}'")

        # Re-resize if foreground crop changed any shapes
        if self.image_size is not None:
            tgt_size = tuple(self.image_size)
            if img.shape[:2] != tgt_size:
                img = self._resize(img, self.image_size, mode="bilinear")
                mask = self._resize(mask, self.image_size, mode="nearest")
            context_imgs = [self._resize(c, self.image_size, mode="bilinear") if c.shape[:2] != tgt_size else c for c in context_imgs]
            context_labels = [self._resize(c, self.image_size, mode="nearest") if c.shape[:2] != tgt_size else c for c in context_labels]

        # CarveMix on target only (after resize, before augmentation)
        exclude_ids = [target_case_id] + valid_context_ids
        img, mask = self._apply_carve_mix(img, mask, label_id, axis, exclude_ids)

        # Advanced augmentation (asymmetric intensity, resolution degradation, mask perturbation)
        img, mask, context_imgs, context_labels = self._apply_advanced_augmentation(
            img, mask, context_imgs, context_labels
        )

        # Apply augmentation (different random transform for target and each context)
        if self.augment:
            img, mask = self._apply_augmentation(img, mask)
            aug_context_imgs = []
            aug_context_labels = []
            for ctx_img, ctx_label in zip(context_imgs, context_labels):
                ctx_img_aug, ctx_label_aug = self._apply_augmentation(ctx_img, ctx_label)
                aug_context_imgs.append(ctx_img_aug)
                aug_context_labels.append(ctx_label_aug)
            context_imgs = aug_context_imgs
            context_labels = aug_context_labels

        # Convert to tensors
        target_in = torch.from_numpy(img.copy()).unsqueeze(0).float()
        target_out = torch.from_numpy(mask.copy()).unsqueeze(0).float()
        context_in = torch.stack([torch.from_numpy(c.copy()).unsqueeze(0).float() for c in context_imgs])
        context_out = torch.stack([torch.from_numpy(c.copy()).unsqueeze(0).float() for c in context_labels])

        return {
            "image": target_in, "label": target_out,
            "context_in": context_in, "context_out": context_out,
            "target_case_id": target_case_id, "context_case_ids": valid_context_ids,
            "label_id": label_id, "axis": axis,
        }

    # --- Other helper methods (_get_2d_bbox, _crop_to_bbox, _normalize_image, _resize, _apply_augmentation, etc.) ---
    # These methods are mostly unchanged as they operate on numpy arrays.
    # I will include them for completeness.

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

    def _normalize_image(self, img: np.ndarray, a_min: float = -200, a_max: float = 300, jitter: float = 0) -> np.ndarray:
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

    def __len__(self) -> int:
        if self.max_ds_len is not None:
            return min(self.max_ds_len, len(self.samples))
        return len(self.samples)

    def _apply_augmentation(self, img: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self.augment or self.spatial_transform is None:
            return img, mask
        img_uint8 = (img * 255).astype(np.uint8)
        mask_uint8 = (mask * 255).astype(np.uint8)
        spatial_result = self.spatial_transform(image=img_uint8, mask=mask_uint8)
        img_aug, mask_aug = spatial_result['image'], spatial_result['mask']
        if self.intensity_transform:
            img_aug = self.intensity_transform(image=img_aug)['image']
        return img_aug.astype(np.float32) / 255.0, (mask_aug > 127).astype(np.float32)

# Collate function remains the same
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
        persistent_workers=num_workers > 0,
    )
