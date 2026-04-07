"""
MedSegBench DataLoader

Fast dataloader for MedSegBench .npz datasets.
Loads entire datasets to RAM for speed.
Samples target, then k context examples with the same label.
"""

import glob
import os
import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

# Load medsegbench label name mapping if available
try:
    from medsegbench.info import INFO as _MSB_INFO
    _LABEL_NAMES: Dict[str, Dict[str, str]] = {
        ds: info.get("pixel_labels", {}) for ds, info in _MSB_INFO.items()
    }  # ds_name -> {str(label_value) -> label_name}
except ImportError:
    _LABEL_NAMES = {}
import torch
from src.dataloaders.augmentations import (
    apply_universeg_augmentation,
    create_augmentation_transforms,
    create_spatial_only_transform,
    create_intensity_only_transform,
    carve_mix_2d,
    foreground_random_crop,
    perturb_mask,
    degrade_resolution,
    random_intensity_shift,
)
from torch.utils.data import DataLoader, Dataset


def _find_keys(keys: List[str], split: str) -> Tuple[Optional[str], Optional[str]]:
    """Find image and label keys in npz file."""
    img_candidates = [f"{split}_images", f"{split}_img", "images", "img"]
    lbl_candidates = [
        f"{split}_labels",
        f"{split}_label",
        f"{split}_mask",
        "labels",
        "masks",
        "label",
    ]
    img_key = next((k for k in img_candidates if k in keys), None)
    lbl_key = next((k for k in lbl_candidates if k in keys), None)
    return img_key, lbl_key


class MedSegBenchDataset(Dataset):
    """
    Fast MedSegBench dataset that loads everything to RAM.

    Args:
        data_root: Path to medsegbench directory containing .npz files
        datasets: List of dataset names (without .npz) to load, or None for all
        split: Which split to use ('train', 'val', 'test')
        context_size: Number of context examples per sample
        image_size: Target size (H, W) for resizing
        augment: Whether to apply augmentation
        augment_config: Augmentation parameters
        max_samples_per_dataset: Limit samples per dataset (for debugging)
    """

    def __init__(
        self,
        data_root: str,
        datasets: Optional[List[str]] = None,
        split: str = "train",
        context_size: int = 3,
        image_size: Tuple[int, int] = (256, 256),
        augment: bool = False,
        augment_config: Optional[Dict] = None,
        augmentation_config: Optional[Dict] = None,
        max_samples_per_dataset: Optional[int] = None,
        max_ds_len: Optional[int] = None,
        carve_mix: bool = False,
        carve_mix_config: Optional[Dict] = None,
        advanced_augmentation: bool = False,
        advanced_augmentation_config: Optional[Dict] = None,
    ):
        self.data_root = data_root
        self.split = split
        self.context_size = context_size
        self.image_size = image_size
        self.augment = augment
        self.max_ds_len = max_ds_len

        # Setup augmentation
        augmentation_config = augmentation_config or augment_config
        self.augment = augment or (augmentation_config is not None and augmentation_config.get("enabled", False))
        if self.augment and augmentation_config is not None:
            self._setup_augmentation(augmentation_config)
        elif augment:
            # Legacy: augment=True but no config — use defaults
            self._setup_augmentation({})
        else:
            self.augmentation_type = "legacy"
            self.augmentation_config_full = None
            self.spatial_transform = None
            self.intensity_transform = None

        # Setup CarveMix
        self.carve_mix = carve_mix
        cm_cfg = carve_mix_config or {}
        self.carve_mix_p = cm_cfg.get("probability", 0.5)
        self.carve_mix_margin_range = tuple(cm_cfg.get("margin_range", [0.1, 0.5]))
        self.carve_mix_harmonize = cm_cfg.get("harmonize", True)
        self.carve_mix_harmonize_sigma = cm_cfg.get("harmonize_sigma", 5.0)
        if self.carve_mix:
            print(f"CarveMix enabled: p={self.carve_mix_p}, margin={self.carve_mix_margin_range}, "
                  f"harmonize={self.carve_mix_harmonize}, sigma={self.carve_mix_harmonize_sigma}")

        # Setup advanced augmentation
        self.advanced_augmentation = advanced_augmentation
        adv = advanced_augmentation_config or {}
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
            print(f"Advanced augmentation enabled: "
                  f"fg_crop_p={self.adv_fg_crop_p}, mask_perturb_p={self.adv_mask_perturb_p}, "
                  f"degrade_p={self.adv_degrade_p}, asym_intensity_p={self.adv_asym_p}")

        # Find dataset files
        # Filter by image_size suffix (e.g., abdomenus_256.npz for image_size=(256, 256))
        size_suffix = f"_{image_size[0]}"
        if datasets is None:
            npz_files = glob.glob(os.path.join(data_root, f"*{size_suffix}.npz"))
            datasets = [os.path.basename(f).replace(".npz", "") for f in npz_files]
        else:
            # Append size suffix to dataset names if not already present
            datasets = [
                ds if ds.endswith(size_suffix) else f"{ds}{size_suffix}"
                for ds in datasets
            ]

        # Load all data to RAM
        # Structure: self.data[dataset_name] = {"images": np.array, "labels": np.array}
        self.data: Dict[str, Dict[str, np.ndarray]] = {}
        # Index: (dataset_name, sample_idx) for all samples
        self.samples: List[Tuple[str, int]] = []
        # Label index: label_value -> [(dataset_name, sample_idx), ...]
        self.label_to_samples: Dict[int, List[Tuple[str, int]]] = defaultdict(list)

        print(f"Loading {len(datasets)} datasets (size={image_size[0]}) to RAM...")
        for ds_name in datasets:
            npz_path = os.path.join(data_root, f"{ds_name}.npz")
            if not os.path.exists(npz_path):
                print(f"  [Skip] {ds_name}: file not found")
                continue

            try:
                data = np.load(npz_path)
                keys = list(data.keys())

                # Find keys for requested split, fallback to other splits
                img_key, lbl_key = _find_keys(keys, split)
                if not img_key:
                    for fallback in ["train", "val", "test"]:
                        img_key, lbl_key = _find_keys(keys, fallback)
                        if img_key:
                            break

                if not img_key:
                    print(f"  [Skip] {ds_name}: no valid keys found")
                    continue

                images = data[img_key]

                # Handle per-class label keys (e.g. idrib: train_label_C1, train_label_C2, ...)
                if not lbl_key:
                    used_split = split
                    if not any(k.startswith(f"{split}_label_C") for k in keys):
                        used_split = next(
                            (
                                s
                                for s in ["train", "val", "test"]
                                if any(k.startswith(f"{s}_label_C") for k in keys)
                            ),
                            None,
                        )
                    if used_split:
                        class_keys = sorted(
                            k for k in keys if k.startswith(f"{used_split}_label_C")
                        )
                        # Stack per-class binary masks into a single multi-class label array
                        class_masks = [data[k] for k in class_keys]
                        labels = np.zeros_like(class_masks[0], dtype=np.uint8)
                        for ci, cm in enumerate(class_masks, start=1):
                            labels[cm > 0] = ci
                    else:
                        print(f"  [Skip] {ds_name}: no valid keys found")
                        continue
                else:
                    labels = data[lbl_key]

                # Limit samples if requested
                if max_samples_per_dataset and len(images) > max_samples_per_dataset:
                    indices = np.random.choice(
                        len(images), max_samples_per_dataset, replace=False
                    )
                    images = images[indices]
                    labels = labels[indices]

                # Store in RAM
                self.data[ds_name] = {"images": images, "labels": labels}

                # Build sample index
                for i in range(len(images)):
                    self.samples.append((ds_name, i))
                    # Get unique labels in this mask (excluding background 0)
                    unique_labels = np.unique(labels[i])
                    for lbl in unique_labels:
                        if lbl != 0:
                            self.label_to_samples[int(lbl)].append((ds_name, i))

                print(
                    f"  [OK] {ds_name}: {len(images)} samples, {len(np.unique(labels))} labels"
                )

            except Exception as e:
                print(f"  [Error] {ds_name}: {e}")
                continue

        print(
            f"Loaded {len(self.samples)} total samples from {len(self.data)} datasets"
        )
        print(f"Found {len(self.label_to_samples)} unique label values")

    def _setup_augmentation(self, cfg: Dict):
        """Setup augmentation from config (mirrors TotalSeg2DZOpt)."""
        aug_type = cfg.get("type", "legacy")
        self.augmentation_type = aug_type

        if aug_type in ("universeg", "custom"):
            self.augmentation_config_full = cfg
            self.spatial_transform = None
            self.intensity_transform = None
            example_cfg = cfg.get("example", {})
            medical_cfg = cfg.get("medical_specialty", {})
            print(f"Augmentation enabled: type=universeg "
                  f"(apply_to_target={example_cfg.get('apply_to_target', False)}, "
                  f"crop_p={medical_cfg.get('crop_p', 0)}, "
                  f"rotation={example_cfg.get('rotation_range', 'default')})")
            return

        # Legacy: albumentations spatial + intensity
        self.augmentation_config_full = None
        spatial_cfg = cfg.get("spatial", cfg)  # flat config for backwards compat
        intensity_cfg = cfg.get("intensity", cfg)
        try:
            if cfg.get("spatial", {}).get("enabled", True) if "spatial" in cfg else True:
                self.spatial_transform, self.intensity_transform = create_augmentation_transforms(
                    rotation_limit=spatial_cfg.get("rotation_limit", 15.0),
                    scale_limit=spatial_cfg.get("scale_limit", 0.1),
                    elastic_alpha=spatial_cfg.get("elastic_alpha", 50.0),
                    elastic_sigma=spatial_cfg.get("elastic_sigma", 5.0),
                    brightness_limit=intensity_cfg.get("brightness_limit", 0.1),
                    contrast_limit=intensity_cfg.get("contrast_limit", 0.1),
                    gamma_limit=tuple(intensity_cfg.get("gamma_limit", [80, 120])),
                    noise_std_range=tuple(intensity_cfg.get("noise_std_range", [0.02, 0.05])),
                )
            else:
                self.spatial_transform = None
                self.intensity_transform = None
            print(f"Augmentation enabled: type=legacy "
                  f"spatial={self.spatial_transform is not None}, "
                  f"intensity={self.intensity_transform is not None}")
        except Exception as e:
            print(f"Warning: Could not setup augmentation: {e}")
            self.spatial_transform = None
            self.intensity_transform = None

    def __len__(self) -> int:
        if self.max_ds_len is not None:
            return min(self.max_ds_len, len(self.samples))
        return len(self.samples)

    def _normalize_image(self, img: np.ndarray) -> np.ndarray:
        """Robust normalization: clip to [1st, 99th] percentile and scale to [0, 1]."""
        img = img.astype(np.float32)
        p_low = np.percentile(img, 1)
        p_high = np.percentile(img, 99)
        if p_high > p_low:
            img = np.clip(img, p_low, p_high)
            img = (img - p_low) / (p_high - p_low)
        else:
            # fallback: min-max over full range
            img_min = img.min()
            img_max = img.max()
            if img_max > img_min:
                img = (img - img_min) / (img_max - img_min)
            else:
                img = np.zeros_like(img)
        return img

    def _to_grayscale(self, img: np.ndarray) -> np.ndarray:
        """Convert image to single-channel grayscale [H, W]."""
        if img.ndim == 2:
            return img
        # Channels first (C, H, W)
        if img.ndim == 3 and img.shape[0] in [1, 3]:
            if img.shape[0] == 1:
                return img[0]
            return np.mean(img, axis=0)
        # Channels last (H, W, C)
        if img.ndim == 3 and img.shape[-1] in [1, 3]:
            if img.shape[-1] == 1:
                return img[..., 0]
            return np.mean(img, axis=-1)
        return img

    def _resize(self, arr: np.ndarray, mode: str = "bilinear") -> np.ndarray:
        """Resize 2D array to self.image_size."""
        if arr.shape == self.image_size:
            return arr
        tensor = torch.from_numpy(arr.copy()).unsqueeze(0).unsqueeze(0).float()
        if mode == "bilinear":
            resized = torch.nn.functional.interpolate(
                tensor, size=self.image_size, mode="bilinear", align_corners=False
            )
        else:
            resized = torch.nn.functional.interpolate(
                tensor, size=self.image_size, mode="nearest"
            )
        return resized.squeeze().numpy()

    def _create_binary_mask(self, mask: np.ndarray, label: int) -> np.ndarray:
        """Create binary mask for specific label."""
        return (mask == label).astype(np.float32)

    def _apply_augmentation(
        self, img: np.ndarray, mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply augmentation to image and mask."""
        if not self.augment or self.spatial_transform is None:
            return img, mask

        img_uint8 = (img * 255).astype(np.uint8)
        mask_uint8 = (mask * 255).astype(np.uint8)

        spatial_result = self.spatial_transform(image=img_uint8, mask=mask_uint8)
        img_aug = spatial_result["image"]
        mask_aug = spatial_result["mask"]

        if self.intensity_transform is not None:
            intensity_result = self.intensity_transform(image=img_aug)
            img_aug = intensity_result["image"]

        img_aug = img_aug.astype(np.float32) / 255.0
        mask_aug = (mask_aug > 127).astype(np.float32)
        return img_aug, mask_aug

    def _get_context_samples(
        self, target_ds: str, target_idx: int, label: int, k: int
    ) -> List[Tuple[str, int]]:
        """Get k context samples with same label from same dataset, excluding target."""
        candidates = self.label_to_samples.get(label, [])
        # Filter to same dataset and exclude target
        candidates = [
            (ds, idx)
            for ds, idx in candidates
            if ds == target_ds and idx != target_idx
        ]
        if len(candidates) == 0:
            return []
        # Random sample
        k = min(k, len(candidates))
        return random.sample(candidates, k)

    def _load_carve_mix_donor(
        self, ds_name: str, label_value: int, exclude_indices: List[int],
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Load a random donor image/mask for CarveMix from same dataset+label."""
        candidates = self.label_to_samples.get(label_value, [])
        exclude_set = set(exclude_indices)
        candidates = [
            (ds, idx) for ds, idx in candidates
            if ds == ds_name and idx not in exclude_set
        ]
        if not candidates:
            return None

        random.shuffle(candidates)
        for donor_ds, donor_idx in candidates[:3]:
            try:
                donor_img = self.data[donor_ds]["images"][donor_idx]
                donor_mask = self.data[donor_ds]["labels"][donor_idx]
                donor_img = self._to_grayscale(donor_img)
                donor_img = self._normalize_image(donor_img)
                donor_binary = self._create_binary_mask(donor_mask, label_value)
                if donor_binary.max() == 0:
                    continue
                donor_img = self._resize(donor_img, mode="bilinear")
                donor_binary = self._resize(donor_binary, mode="nearest")
                return donor_img, donor_binary
            except Exception:
                continue
        return None

    def _apply_carve_mix(
        self,
        target_img: np.ndarray,
        target_mask: np.ndarray,
        ds_name: str,
        label_value: int,
        exclude_indices: List[int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Conditionally apply CarveMix to target based on probability."""
        if not self.carve_mix or random.random() > self.carve_mix_p:
            return target_img, target_mask

        donor = self._load_carve_mix_donor(ds_name, label_value, exclude_indices)
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
        """Apply post-resize advanced augmentations.

        Asymmetric intensity and resolution degradation are applied independently
        per image. Context masks get morphological perturbation.
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

        # Context mask perturbation (context only - target mask is GT)
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
            - 'image': [1, H, W] target image
            - 'label': [1, H, W] target binary mask
            - 'context_in': [k, 1, H, W] context images
            - 'context_out': [k, 1, H, W] context masks
            - 'target_case_id': str, e.g. "datasetname_42"
            - 'context_case_ids': List[str], e.g. ["datasetname_10", ...]
            - 'label_id': str, e.g. "datasetname_3"
        """
        if self.max_ds_len is not None:
            ds_name, sample_idx = random.choice(self.samples)
        else:
            ds_name, sample_idx = self.samples[idx]
        images = self.data[ds_name]["images"]
        labels = self.data[ds_name]["labels"]

        # Get target image and mask
        target_img = images[sample_idx]
        target_mask = labels[sample_idx]

        # Convert to grayscale and normalize
        target_img = self._to_grayscale(target_img)
        target_img = self._normalize_image(target_img)

        # Sample a random label from this mask (excluding background)
        unique_labels = np.unique(target_mask)
        fg_labels = [l for l in unique_labels if l != 0]
        if len(fg_labels) == 0:
            # Fallback: return empty mask
            label_value = 1
        else:
            label_value = random.choice(fg_labels)

        # Create binary mask for sampled label
        target_binary = self._create_binary_mask(target_mask, label_value)

        # Foreground random crop (before resize, ensures FG stays visible)
        if self.advanced_augmentation and random.random() < self.adv_fg_crop_p:
            target_img, target_binary = foreground_random_crop(
                target_img, target_binary, self.adv_fg_crop_min
            )

        # Resize
        target_img = self._resize(target_img, mode="bilinear")
        target_binary = self._resize(target_binary, mode="nearest")

        # Get context samples
        context_samples = self._get_context_samples(
            ds_name, sample_idx, label_value, self.context_size
        )

        context_imgs = []
        context_masks = []
        for ctx_ds, ctx_idx in context_samples:
            ctx_img = self.data[ctx_ds]["images"][ctx_idx]
            ctx_mask = self.data[ctx_ds]["labels"][ctx_idx]

            ctx_img = self._to_grayscale(ctx_img)
            ctx_img = self._normalize_image(ctx_img)
            ctx_binary = self._create_binary_mask(ctx_mask, label_value)

            # Foreground random crop (independent per context)
            if self.advanced_augmentation and random.random() < self.adv_fg_crop_p:
                ctx_img, ctx_binary = foreground_random_crop(
                    ctx_img, ctx_binary, self.adv_fg_crop_min
                )

            ctx_img = self._resize(ctx_img, mode="bilinear")
            ctx_binary = self._resize(ctx_binary, mode="nearest")

            context_imgs.append(ctx_img)
            context_masks.append(ctx_binary)

        # CarveMix on target (after resize, before augmentation)
        exclude_indices = [sample_idx] + [idx for _, idx in context_samples]
        target_img, target_binary = self._apply_carve_mix(
            target_img, target_binary, ds_name, label_value, exclude_indices
        )

        # Advanced augmentation (asymmetric intensity, resolution degradation, mask perturbation)
        target_img, target_binary, context_imgs, context_masks = self._apply_advanced_augmentation(
            target_img, target_binary,
            context_imgs if context_imgs else None,
            context_masks if context_masks else None,
        )
        context_imgs = context_imgs or []
        context_masks = context_masks or []

        # Apply augmentation
        if self.augment:
            if self.augmentation_type in ("universeg", "custom"):
                aug_cfg = dict(self.augmentation_config_full)
                aug_cfg["img_size"] = self.image_size[0]
                target_img, target_binary, context_imgs, context_masks = apply_universeg_augmentation(
                    target_img, target_binary, context_imgs, context_masks,
                    full_config=aug_cfg,
                )
            else:
                # Legacy: per-image spatial + intensity
                target_img, target_binary = self._apply_augmentation(target_img, target_binary)
                aug_context_imgs, aug_context_masks = [], []
                for ci, cm in zip(context_imgs, context_masks):
                    ci_aug, cm_aug = self._apply_augmentation(ci, cm)
                    aug_context_imgs.append(ci_aug)
                    aug_context_masks.append(cm_aug)
                context_imgs = aug_context_imgs
                context_masks = aug_context_masks

        # Convert to tensors
        target_in = torch.from_numpy(target_img.copy()).unsqueeze(0).float()
        target_out = torch.from_numpy(target_binary.copy()).unsqueeze(0).float()

        # Build case/label IDs
        target_case_id = f"{ds_name}_{sample_idx}"
        label_name = _LABEL_NAMES.get(ds_name, {}).get(str(label_value), str(label_value))
        label_id = f"{ds_name}_{label_name}"
        context_case_ids = [
            f"{ctx_ds}_{ctx_idx}" for ctx_ds, ctx_idx in context_samples
        ]

        result = {
            "image": target_in,
            "label": target_out,
            "target_case_id": target_case_id,
            "context_case_ids": context_case_ids,
            "label_id": label_id,
        }

        if context_imgs:
            context_in = torch.stack(
                [torch.from_numpy(c.copy()).unsqueeze(0).float() for c in context_imgs]
            )
            context_out = torch.stack(
                [torch.from_numpy(c.copy()).unsqueeze(0).float() for c in context_masks]
            )
            result["context_in"] = context_in
            result["context_out"] = context_out

        return result


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate function for batching."""
    result = {
        "image": torch.stack([item["image"] for item in batch]),
        "label": torch.stack([item["label"] for item in batch]),
        "target_case_ids": [item["target_case_id"] for item in batch],
        "context_case_ids": [item["context_case_ids"] for item in batch],
        "label_ids": [item["label_id"] for item in batch],
    }

    if "context_in" in batch[0]:
        # Pad context to same size across batch
        max_k = max(item["context_in"].shape[0] for item in batch)
        h, w = batch[0]["image"].shape[1:]

        context_in_list = []
        context_out_list = []
        for item in batch:
            k = item["context_in"].shape[0]
            if k < max_k:
                # Pad with zeros
                pad_in = torch.zeros(max_k - k, 1, h, w)
                pad_out = torch.zeros(max_k - k, 1, h, w)
                context_in_list.append(torch.cat([item["context_in"], pad_in], dim=0))
                context_out_list.append(
                    torch.cat([item["context_out"], pad_out], dim=0)
                )
            else:
                context_in_list.append(item["context_in"])
                context_out_list.append(item["context_out"])

        result["context_in"] = torch.stack(context_in_list)
        result["context_out"] = torch.stack(context_out_list)

    return result


def get_dataloader(
    data_root: str,
    datasets: Optional[List[str]] = None,
    split: str = "train",
    context_size: int = 3,
    batch_size: int = 8,
    image_size: Tuple[int, int] = (256, 256),
    num_workers: int = 0,
    shuffle: bool = True,
    augment: bool = False,
    augment_config: Optional[Dict] = None,
    augmentation_config: Optional[Dict] = None,
    max_ds_len: Optional[int] = None,
    **kwargs,
) -> DataLoader:
    """Create a DataLoader for MedSegBench."""
    dataset = MedSegBenchDataset(
        data_root=data_root,
        datasets=datasets,
        split=split,
        context_size=context_size,
        image_size=image_size,
        augment=augment,
        augment_config=augment_config,
        augmentation_config=augmentation_config,
        max_ds_len=max_ds_len,
        **kwargs,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
