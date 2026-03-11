"""
TotalSegmentator 2D DataLoader for Z-Optimized 3D Format.

Works with HDF5 files created by scripts/totalseg_3d_to_2d_z_optimized.py.
Key features:
- Full 3D volumes with z-optimized chunking (1, H, W)
- ~40% smaller storage than shared slices format
- ~9x faster z-slice loading
- Z-axis only (no y/x slices)

HDF5 Structure expected:
    case.h5
    ├── ct: (D, H, W) float32, chunks=(1, H, W)
    ├── masks/{label_id}: (D, H, W) uint8, chunks=(1, H, W)
    └── meta/
        ├── shape, spacing, modality
        ├── z_indices: [0, 3, 6, ...]
        └── labels/{label_id}/z_coverage, bbox, volume
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


class TotalSeg2DZOptDataset(Dataset):
    """
    Dataset for TotalSegmentator 2D z-slices from z-optimized 3D volumes.

    Filters slices by coverage at runtime for flexibility.
    """

    def __init__(
        self,
        root_dir: str,
        stats_path: str,
        label_id_list: Union[List[str], str],
        context_size: int = 3,
        image_size: Optional[Tuple[int, int]] = None,
        crop_to_bbox: bool = False,
        bbox_padding: int = 10,
        min_coverage: int = 100,
        min_coverage_ratio: float = 0.1,
        split: Optional[Union[str, List[str]]] = None,
        random_context: bool = True,
        max_ds_len: Optional[int] = None,
        max_labels: Optional[int] = None,
        max_cases: Optional[int] = None,
        modality: str = "ct",
        same_case_context: bool = False,  # Deprecated, use same_case_context_ratio
        same_case_context_ratio: float = 0.0,  # Ratio of context from same case (0.0-1.0)
        class_balanced: bool = False,
        augment: bool = False,
        augment_config: Optional[Dict] = None,
        augmentation_config: Optional[Dict] = None,
        context_diversity: Optional[Dict] = None,
        max_slices_per_group: Optional[int] = None,
        slice_selection: str = "all",
    ):
        self.root_dir = Path(root_dir)
        self.modality = modality.lower()
        # Handle same_case_context as bool (legacy) or ratio
        if same_case_context and same_case_context_ratio == 0.0:
            # Legacy: same_case_context=True means ratio=1.0 (prefer same case)
            self.same_case_context_ratio = 1.0
        else:
            self.same_case_context_ratio = same_case_context_ratio
        self.max_slices_per_group = max_slices_per_group
        self.slice_selection = slice_selection
        self.min_coverage = min_coverage
        self.min_coverage_ratio = min_coverage_ratio
        self.context_size = context_size
        self.image_size = image_size
        self.crop_to_bbox = crop_to_bbox
        self.bbox_padding = bbox_padding
        self.random_context = random_context
        self.max_ds_len = max_ds_len
        self.class_balanced = class_balanced

        # Context diversity config
        div_cfg = context_diversity or {}
        self.context_diversity_type = div_cfg.get('type', 'random')
        self.context_diversity_candidates = div_cfg.get('num_candidates', 10)

        # Setup augmentation
        augmentation_config = augmentation_config or augment_config
        self.augment = augment or (augmentation_config is not None and augmentation_config.get("enabled", False))
        if self.augment and augmentation_config is not None:
            self._setup_augmentation(augmentation_config)
        else:
            self.augmentation_type = "legacy"
            self.augmentation_config_full = None
            self.windowing_jitter = 0
            self.spatial_transform = None
            self.intensity_transform = None

        # Resolve label list
        if isinstance(label_id_list, str):
            self.label_id_list = get_label_ids(label_id_list, max_labels=max_labels, modality=self.modality)
        else:
            self.label_id_list = label_id_list[:max_labels] if max_labels else label_id_list
        label_id_set = set(self.label_id_list)

        # Load stats
        with open(stats_path, "rb") as f:
            self.stats = pickle.load(f)
        print(f"Loaded stats for {len(self.stats)} cases")

        # Find HDF5 files
        self.h5_files = sorted(list(self.root_dir.glob("*.h5")))

        # Filter by split if needed
        if split is not None:
            self._filter_by_split(split)

        # Limit cases
        if max_cases is not None and len(self.h5_files) > max_cases:
            random.shuffle(self.h5_files)
            self.h5_files = sorted(self.h5_files[:max_cases])
            valid_case_ids = {f.stem for f in self.h5_files}
            self.stats = {k: v for k, v in self.stats.items() if k in valid_case_ids}
            print(f"Limited to {max_cases} cases")

        # Build sample index: (case_id, label_id, z_idx)
        self.samples = []
        self.label_to_cases: Dict[str, List[str]] = {}
        self.case_to_labels: Dict[str, set] = {}
        self.valid_slices_cache: Dict[Tuple[str, str], List[int]] = {}  # (case, label) -> [z_idx]

        print(f"Building sample index...")
        for h5_file in tqdm(self.h5_files, desc="Scanning"):
            case_id = h5_file.stem
            case_stats = self.stats.get(case_id)
            if case_stats is None:
                continue

            self.case_to_labels[case_id] = set()
            labels_stats = case_stats.get("labels", {})

            for label_id in labels_stats:
                if label_id not in label_id_set:
                    continue

                label_stats = labels_stats[label_id]
                coverage = label_stats.get("coverage", {})
                z_coverage = coverage.get("z", [])

                if not z_coverage:
                    continue

                # Filter slices by coverage
                max_cov = max(z_coverage) if z_coverage else 0
                threshold = max(self.min_coverage, max_cov * self.min_coverage_ratio)

                valid_slices = [
                    (zi, cov) for zi, cov in enumerate(z_coverage) if cov >= threshold
                ]
                if self.max_slices_per_group is not None:
                    valid_slices = self._select_slices(valid_slices, self.max_slices_per_group)

                # Cache valid z-indices for context sampling
                valid_indices = [zi for zi, _ in valid_slices]
                self.valid_slices_cache[(case_id, label_id)] = valid_indices

                for z_idx in valid_indices:
                    self.samples.append((case_id, label_id, z_idx))

                # Track which cases have this label
                if valid_slices:
                    self.label_to_cases.setdefault(label_id, []).append(case_id)
                    self.case_to_labels[case_id].add(label_id)

        # Deduplicate label_to_cases
        for label_id in self.label_to_cases:
            self.label_to_cases[label_id] = list(set(self.label_to_cases[label_id]))

        # Class-balanced sampling: group samples by label
        self.label_to_samples: Dict[str, List[Tuple]] = {}
        for case_id, label_id, z_idx in self.samples:
            self.label_to_samples.setdefault(label_id, []).append((case_id, z_idx))
        self.active_labels = list(self.label_to_samples.keys())

        print(f"Built {len(self.samples)} samples from {len(self.label_to_cases)} labels")
        if self.class_balanced:
            label_counts = {l: len(s) for l, s in self.label_to_samples.items()}
            min_l = min(label_counts, key=label_counts.get)
            max_l = max(label_counts, key=label_counts.get)
            print(f"Class-balanced sampling: {len(self.active_labels)} labels, "
                  f"min={min_l}({label_counts[min_l]}), max={max_l}({label_counts[max_l]})")

        if self.max_ds_len is not None and len(self.samples) > self.max_ds_len:
            random.shuffle(self.samples)
            print(f"Shuffled samples (max_ds_len={self.max_ds_len})")

        # H5 file cache
        self._h5_cache: Dict[str, h5py.File] = {}
        self._h5_cache_max = 32

    def _setup_augmentation(self, cfg: Dict):
        """Setup augmentation from config."""
        aug_type = cfg.get("type", "legacy")
        self.augmentation_type = aug_type

        if aug_type in ("universeg", "custom"):
            self.augmentation_config_full = cfg
            self.windowing_jitter = 0
            self.spatial_transform = None
            self.intensity_transform = None
            print(f"Augmentation enabled: type=universeg/custom (task+example level)")
            return

        # Legacy: build albumentations transforms
        self.augmentation_config_full = None
        self.windowing_jitter = cfg.get("windowing_jitter", 0)
        try:
            from src.dataloaders.augmentations import (
                create_spatial_only_transform,
                create_intensity_only_transform,
            )

            spatial_cfg = cfg.get("spatial", {})
            if spatial_cfg.get("enabled", True):
                self.spatial_transform = create_spatial_only_transform(
                    rotation_limit=spatial_cfg.get("rotation_limit", 15.0),
                    scale_limit=spatial_cfg.get("scale_limit", 0.1),
                    elastic_alpha=spatial_cfg.get("elastic_alpha", 50.0),
                    elastic_sigma=spatial_cfg.get("elastic_sigma", 5.0),
                )
            else:
                self.spatial_transform = None

            intensity_cfg = cfg.get("intensity", {})
            if intensity_cfg.get("enabled", True):
                img_size = self.image_size[0] if self.image_size else 512
                self.intensity_transform = create_intensity_only_transform(
                    brightness_limit=intensity_cfg.get("brightness_limit", 0.1),
                    contrast_limit=intensity_cfg.get("contrast_limit", 0.1),
                    gamma_limit=tuple(intensity_cfg.get("gamma_limit", [80, 120])),
                    noise_std_range=tuple(intensity_cfg.get("noise_std_range", [0.02, 0.05])),
                    img_size=img_size,
                )
            else:
                self.intensity_transform = None

            print(f"Augmentation enabled: spatial={self.spatial_transform is not None}, "
                  f"intensity={self.intensity_transform is not None}")
        except ImportError:
            print("Warning: Could not import augmentation functions, augmentation disabled")
            self.spatial_transform = None
            self.intensity_transform = None

    def _select_slices(
        self, valid_slices: List[Tuple[int, float]], max_slices: int
    ) -> List[Tuple[int, float]]:
        """Subsample valid slices per (case, label) group."""
        n = len(valid_slices)
        if n <= max_slices:
            return valid_slices

        if self.slice_selection == "random":
            return sorted(random.sample(valid_slices, max_slices), key=lambda x: x[0])

        elif self.slice_selection == "stride":
            indices = np.linspace(0, n - 1, max_slices, dtype=int)
            return [valid_slices[i] for i in indices]

        elif self.slice_selection == "stride_peak":
            peak_pos = max(range(n), key=lambda i: valid_slices[i][1])
            without_peak = [s for i, s in enumerate(valid_slices) if i != peak_pos]
            n_stride = max_slices - 1
            if n_stride == 0:
                return [valid_slices[peak_pos]]
            indices = np.linspace(0, len(without_peak) - 1, n_stride, dtype=int)
            selected = [without_peak[i] for i in indices] + [valid_slices[peak_pos]]
            return sorted(selected, key=lambda x: x[0])

        return valid_slices

    def _filter_by_split(self, split: Union[str, List[str]]):
        """Filter cases by train/val/test split."""
        import pandas as pd

        meta_path = self.root_dir.parent / "totalseg" / "meta.csv"
        if not meta_path.exists():
            print(f"Warning: meta.csv not found at {meta_path}")
            return

        df = pd.read_csv(meta_path, sep=";")
        if isinstance(split, str):
            split_case_ids = set(df["image_id"][df["split"] == split].tolist())
        else:
            split_case_ids = set(df["image_id"][df["split"].isin(split)].tolist())

        self.stats = {k: v for k, v in self.stats.items() if k in split_case_ids}
        self.h5_files = [f for f in self.h5_files if f.stem in split_case_ids]
        print(f"Filtered to {len(self.h5_files)} cases for split '{split}'")

    def _get_h5_file(self, case_id: str) -> h5py.File:
        """Get cached H5 file handle with LRU eviction."""
        if case_id not in self._h5_cache:
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
            self._h5_cache[case_id] = self._h5_cache.pop(case_id)

        return self._h5_cache[case_id]

    def _load_slice(
        self, case_id: str, label_id: str, z_idx: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Load CT and mask z-slice from z-optimized HDF5 format."""
        h5f = self._get_h5_file(case_id)

        # Direct 3D indexing - fast due to z-optimized chunking
        img = h5f["ct"][z_idx, :, :]
        mask = h5f[f"masks/{label_id}"][z_idx, :, :]

        return img.astype(np.float32), mask.astype(np.float32)

    def _get_2d_bbox(self, case_id: str, label_id: str) -> Tuple[int, int, int, int]:
        """Get 2D bounding box for a label (z-axis view: y, x dimensions)."""
        case_stats = self.stats.get(case_id, {})
        label_stats = case_stats.get("labels", {}).get(label_id, {})
        bbox_3d = label_stats.get("bbox", ((0, 0), (0, 0), (0, 0)))
        z_range, y_range, x_range = bbox_3d
        return (y_range[0], y_range[1], x_range[0], x_range[1])

    def _crop_to_bbox(
        self, img: np.ndarray, mask: np.ndarray, bbox: Tuple[int, int, int, int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Crop image and mask to bounding box with padding."""
        row_min, row_max, col_min, col_max = bbox
        h, w = img.shape
        row_min = max(0, row_min - self.bbox_padding)
        row_max = min(h, row_max + self.bbox_padding + 1)
        col_min = max(0, col_min - self.bbox_padding)
        col_max = min(w, col_max + self.bbox_padding + 1)
        return img[row_min:row_max, col_min:col_max], mask[row_min:row_max, col_min:col_max]

    def _normalize_image(self, img: np.ndarray, jitter: float = 0) -> np.ndarray:
        """Normalize image based on modality."""
        if self.modality == "mri":
            nonzero_mask = img > 0
            if nonzero_mask.any():
                nonzero_vals = img[nonzero_mask]
                a_min = np.percentile(nonzero_vals, 0.5)
                a_max = np.percentile(nonzero_vals, 99.5)
            else:
                a_min, a_max = img.min(), img.max()
            if a_max - a_min < 1e-6:
                a_max = a_min + 1.0
        else:
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
        """Resize array using torch interpolation."""
        tensor = torch.from_numpy(arr.copy()).unsqueeze(0).unsqueeze(0)
        resized = torch.nn.functional.interpolate(
            tensor, size=size, mode=mode,
            align_corners=False if mode == "bilinear" else None
        )
        return resized.squeeze().numpy()

    def _resize_mask(self, mask: np.ndarray, size: Tuple[int, int], min_value: float = 0.5) -> np.ndarray:
        """Resize mask preserving small objects (hybrid area + max pooling)."""
        tensor = torch.from_numpy(mask.copy()).unsqueeze(0).unsqueeze(0).float()
        area_resized = torch.nn.functional.interpolate(tensor, size=size, mode='area')
        max_resized = torch.nn.functional.adaptive_max_pool2d(tensor, size)
        result = torch.maximum(area_resized, max_resized * min_value)
        return result.squeeze().numpy()

    def _apply_spatial_augmentation(
        self, img: np.ndarray, mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply spatial-only augmentation."""
        if self.spatial_transform is None:
            return img, mask

        img_uint8 = (img * 255).astype(np.uint8)
        mask_uint8 = (mask * 255).astype(np.uint8)
        result = self.spatial_transform(image=img_uint8, mask=mask_uint8)
        return result['image'].astype(np.float32) / 255.0, (result['mask'] > 127).astype(np.float32)

    def _apply_intensity_augmentation(self, img: np.ndarray) -> np.ndarray:
        """Apply intensity-only augmentation."""
        if self.intensity_transform is None:
            return img

        img_uint8 = (img * 255).astype(np.uint8)
        result = self.intensity_transform(image=img_uint8)
        return result['image'].astype(np.float32) / 255.0

    def _get_context_slices(
        self, target_case_id: str, label_id: str, target_z_idx: int = -1,
        same_case_only: bool = True,
    ) -> List[Tuple[str, int]]:
        """Get valid context z-slice candidates for a label.

        Args:
            same_case_only: If True, return only same-case candidates.
                           If False, return only other-case candidates.
        """
        candidates = []

        for case_id in self.label_to_cases.get(label_id, []):
            is_same_case = (case_id == target_case_id)

            if same_case_only and not is_same_case:
                continue
            if not same_case_only and is_same_case:
                continue

            valid_indices = self.valid_slices_cache.get((case_id, label_id), [])
            for z_idx in valid_indices:
                if is_same_case and z_idx == target_z_idx:
                    continue
                candidates.append((case_id, z_idx))

        return candidates

    def __getitem__(self, idx: int, _retry: int = 0) -> Dict[str, torch.Tensor]:
        """Get a sample with context examples."""
        if self.class_balanced:
            label_id = random.choice(self.active_labels)
            target_case_id, z_idx = random.choice(self.label_to_samples[label_id])
        else:
            target_case_id, label_id, z_idx = self.samples[idx]

        # Load target
        img, mask = self._load_slice(target_case_id, label_id, z_idx)

        # Crop to bbox if enabled
        if self.crop_to_bbox:
            bbox = self._get_2d_bbox(target_case_id, label_id)
            img, mask = self._crop_to_bbox(img, mask, bbox)

        img = self._normalize_image(img, jitter=self.windowing_jitter if self.augment else 0)

        if self.image_size:
            img = self._resize(img, self.image_size, "bilinear")
            mask = self._resize_mask(mask, self.image_size)

        # Load context
        context_imgs, context_masks, context_case_ids = [], [], []

        if self.context_size > 0:
            # Calculate how many to sample from same case vs other cases
            max_same_case = int(np.ceil(self.same_case_context_ratio * self.context_size))
            max_other_case = self.context_size - max_same_case

            # Get candidates from same case and other cases
            same_case_candidates = self._get_context_slices(
                target_case_id, label_id, z_idx, same_case_only=True
            )
            other_case_candidates = self._get_context_slices(
                target_case_id, label_id, z_idx, same_case_only=False
            )

            if self.random_context:
                random.shuffle(same_case_candidates)
                random.shuffle(other_case_candidates)

            # Build ordered candidate list: same-case first (up to max), then other-case
            # If not enough same-case, other-case will fill the gap
            all_candidates = []
            all_candidates.extend(same_case_candidates[:max_same_case])
            all_candidates.extend(other_case_candidates[:max_other_case])
            # Fallback: if we don't have enough, use remaining from both pools
            if len(all_candidates) < self.context_size:
                remaining_same = same_case_candidates[max_same_case:]
                remaining_other = other_case_candidates[max_other_case:]
                all_candidates.extend(remaining_other + remaining_same)

            for ctx_case_id, ctx_z_idx in all_candidates:
                if len(context_imgs) >= self.context_size:
                    break

                try:
                    ctx_img, ctx_mask = self._load_slice(ctx_case_id, label_id, ctx_z_idx)

                    if ctx_mask.max() == 0:
                        continue

                    if self.crop_to_bbox:
                        bbox = self._get_2d_bbox(ctx_case_id, label_id)
                        ctx_img, ctx_mask = self._crop_to_bbox(ctx_img, ctx_mask, bbox)

                    ctx_img = self._normalize_image(ctx_img, jitter=self.windowing_jitter if self.augment else 0)

                    if self.image_size:
                        ctx_img = self._resize(ctx_img, self.image_size, "bilinear")
                        ctx_mask = self._resize_mask(ctx_mask, self.image_size)

                    context_imgs.append(ctx_img)
                    context_masks.append(ctx_mask)
                    context_case_ids.append(ctx_case_id)

                except Exception as e:
                    print(f"Warning: Failed to load context {ctx_case_id}: {e}")
                    continue

            if len(context_imgs) == 0:
                if _retry >= 10:
                    raise RuntimeError(f"Could not find valid context for label={label_id} after 10 retries")
                retry_idx = random.randint(0, len(self.samples) - 1)
                return self.__getitem__(retry_idx, _retry=_retry + 1)

            while len(context_imgs) < self.context_size and len(context_imgs) > 0:
                resample_idx = random.randint(0, len(context_imgs) - 1)
                context_imgs.append(context_imgs[resample_idx].copy())
                context_masks.append(context_masks[resample_idx].copy())
                context_case_ids.append(context_case_ids[resample_idx])

        # Apply augmentation
        if self.augment:
            if self.augmentation_type in ("universeg", "custom"):
                from src.dataloaders.augmentations import apply_universeg_augmentation
                aug_cfg = dict(self.augmentation_config_full)
                if self.image_size:
                    aug_cfg["img_size"] = self.image_size[0]
                img, mask, context_imgs, context_masks = apply_universeg_augmentation(
                    img, mask, context_imgs, context_masks,
                    full_config=aug_cfg,
                )
            else:
                img, mask = self._apply_spatial_augmentation(img, mask)
                img = self._apply_intensity_augmentation(img)
                for i in range(len(context_imgs)):
                    context_imgs[i], context_masks[i] = self._apply_spatial_augmentation(
                        context_imgs[i], context_masks[i]
                    )
                    context_imgs[i] = self._apply_intensity_augmentation(context_imgs[i])

        # Convert to tensors
        target_in = torch.from_numpy(img.copy()).unsqueeze(0).float()
        target_out = torch.from_numpy(mask.copy()).unsqueeze(0).float()

        if self.context_size == 0:
            return {
                "image": target_in,
                "label": target_out,
                "target_case_id": target_case_id,
                "label_id": label_id,
            }

        context_in = torch.stack([
            torch.from_numpy(c.copy()).unsqueeze(0).float() for c in context_imgs
        ])
        context_out = torch.stack([
            torch.from_numpy(c.copy()).unsqueeze(0).float() for c in context_masks
        ])

        return {
            "image": target_in,
            "label": target_out,
            "context_in": context_in,
            "context_out": context_out,
            "target_case_id": target_case_id,
            "context_case_ids": context_case_ids,
            "label_id": label_id,
        }

    def __len__(self) -> int:
        if self.max_ds_len is not None:
            return min(self.max_ds_len, len(self.samples))
        return len(self.samples)

    def __del__(self):
        """Close cached H5 file handles."""
        for h5f in self._h5_cache.values():
            try:
                h5f.close()
            except Exception:
                pass
        self._h5_cache.clear()


def collate_fn(batch: List[Dict]) -> Dict[str, Union[torch.Tensor, List]]:
    """Custom collate function for batching."""
    result = {
        "image": torch.stack([item["image"] for item in batch]),
        "label": torch.stack([item["label"] for item in batch]),
        "target_case_ids": [item["target_case_id"] for item in batch],
        "label_ids": [item["label_id"] for item in batch],
    }

    if any("context_in" in item for item in batch):
        prototype_item = next((item for item in batch if "context_in" in item), None)
        if prototype_item is not None:
            context_size, _, h, w = prototype_item["context_in"].shape

            default_in = torch.zeros(context_size, 1, h, w)
            default_out = torch.zeros(context_size, 1, h, w)
            default_ids = ["PAD"] * context_size

            result["context_in"] = torch.stack([
                item.get("context_in", default_in) for item in batch
            ])
            result["context_out"] = torch.stack([
                item.get("context_out", default_out) for item in batch
            ])
            result["context_case_ids"] = [
                item.get("context_case_ids", default_ids) for item in batch
            ]

    return result


def get_dataloader(
    root_dir: str,
    stats_path: str,
    label_id_list: Union[List[str], str],
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 4,
    **kwargs,
) -> DataLoader:
    """Create a DataLoader for the z-optimized dataset."""
    dataset = TotalSeg2DZOptDataset(
        root_dir=root_dir,
        stats_path=stats_path,
        label_id_list=label_id_list,
        **kwargs,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
        multiprocessing_context="spawn" if num_workers > 0 else None,
    )
