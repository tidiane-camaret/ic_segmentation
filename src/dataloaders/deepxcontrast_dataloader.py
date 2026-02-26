"""
DeepXcontrast 2D DataLoader

Loads pre-processed 2D slices from HDF5 files for in-context learning.
Supports CT and MRI modalities with appropriate normalization.

Labels:
    CT mode: c1 (gray matter), c2 (white matter), c3 (CSF), nuc (nuclei)
    MRI mode: c1, c2, c3, tissue (combined), nuc
"""
import pickle
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


# Available labels per modality
LABELS_CT = ["c1", "c2", "c3", "nuc"]
LABELS_MRI = ["c1", "c2", "c3", "tissue", "nuc"]


class DeepXcontrastDataset(Dataset):
    """
    Dataset for DeepXcontrast 2D slices from HDF5 files.

    Args:
        root_dir: Path to HDF5 directory (containing case_id.h5 files)
        stats_path: Path to stats.pkl file
        label_list: List of labels to include, or "all" for all available
        context_size: Number of context examples per sample
        axes: Tuple of axes to include ("x", "y", "z")
        image_size: Target image size (H, W) or None for original
        modality: "ct" or "mri" (affects normalization)
        split: Train/val split ratio (e.g., 0.8 for 80% train)
        is_train: Whether this is the training set
        max_cases: Limit number of cases
        augment: Enable basic augmentation
        seed: Random seed for reproducibility
    """

    def __init__(
        self,
        root_dir: str,
        stats_path: str,
        label_list: Union[List[str], str] = "all",
        context_size: int = 3,
        axes: Tuple[str, ...] = ("x", "y", "z"),
        image_size: Optional[Tuple[int, int]] = None,
        modality: str = "ct",
        split: float = 0.8,
        is_train: bool = True,
        max_cases: Optional[int] = None,
        augment: bool = False,
        seed: int = 42,
    ):
        self.root_dir = Path(root_dir)
        self.context_size = context_size
        self.axes = axes
        self.image_size = image_size
        self.modality = modality.lower()
        self.augment = augment and is_train
        self.is_train = is_train

        # Resolve label list
        available_labels = LABELS_CT if self.modality == "ct" else LABELS_MRI
        if label_list == "all":
            self.label_list = available_labels
        else:
            self.label_list = [l for l in label_list if l in available_labels]
        label_set = set(self.label_list)

        # Load stats
        with open(stats_path, "rb") as f:
            self.stats = pickle.load(f)
        print(f"Loaded stats for {len(self.stats)} cases")

        # Get H5 files
        self.h5_files = sorted(list(self.root_dir.glob("*.h5")))

        # Filter to cases with stats
        valid_case_ids = set(self.stats.keys())
        self.h5_files = [f for f in self.h5_files if f.stem in valid_case_ids]

        # Train/val split
        random.seed(seed)
        all_cases = list(self.h5_files)
        random.shuffle(all_cases)
        split_idx = int(len(all_cases) * split)
        if is_train:
            self.h5_files = sorted(all_cases[:split_idx])
        else:
            self.h5_files = sorted(all_cases[split_idx:])

        # Limit cases
        if max_cases and len(self.h5_files) > max_cases:
            random.shuffle(self.h5_files)
            self.h5_files = sorted(self.h5_files[:max_cases])

        # Build sample index: (case_id, label_id, axis)
        self.samples = []
        self.label_to_cases: Dict[str, List[str]] = {}

        print(f"Scanning {len(self.h5_files)} HDF5 files...")
        for h5_path in self.h5_files:
            case_id = h5_path.stem
            case_stats = self.stats.get(case_id, {})

            with h5py.File(h5_path, 'r') as h5f:
                for label_id in h5f.keys():
                    if label_id not in label_set:
                        continue
                    if label_id not in case_stats:
                        continue

                    self.label_to_cases.setdefault(label_id, []).append(case_id)

                    for axis in self.axes:
                        key = f"{label_id}/{axis}_slice"
                        if key in h5f:
                            self.samples.append((case_id, label_id, axis))

        # Context lookup: (label, axis) -> list of case_ids
        self.valid_contexts = {
            (label, axis): cases
            for label, cases in self.label_to_cases.items()
            for axis in self.axes
        }

        print(f"Dataset: {len(self.samples)} samples, {len(self.label_to_cases)} labels, "
              f"modality={self.modality}, is_train={is_train}")

        # H5 file cache (per-worker, bounded)
        self._h5_cache: Dict[str, h5py.File] = {}
        self._h5_cache_max = 8

    def _get_h5(self, case_id: str) -> h5py.File:
        """Get cached H5 file handle."""
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
        return self._h5_cache[case_id]

    def _load_slice(self, case_id: str, label_id: str, axis: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load image and mask slice from HDF5."""
        h5f = self._get_h5(case_id)
        img = h5f[f"{label_id}/{axis}_slice_img"][:]
        mask = h5f[f"{label_id}/{axis}_slice"][:]
        return img, mask

    def _normalize(self, img: np.ndarray) -> np.ndarray:
        """Normalize image based on modality."""
        if self.modality == "ct":
            # CT: window [-500, 1000] HU
            a_min, a_max = -500.0, 1000.0
        else:
            # MRI: percentile-based
            nonzero = img[img > 0]
            if len(nonzero) > 0:
                a_min = np.percentile(nonzero, 0.5)
                a_max = np.percentile(nonzero, 99.5)
            else:
                a_min, a_max = img.min(), img.max()
            if a_max - a_min < 1e-6:
                a_max = a_min + 1.0

        img = np.clip(img, a_min, a_max)
        img = (img - a_min) / (a_max - a_min)
        return img

    def _resize(self, arr: np.ndarray, mode: str = "bilinear") -> np.ndarray:
        """Resize array to target size."""
        if self.image_size is None:
            return arr
        if arr.shape[:2] == tuple(self.image_size):
            return arr
        tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).float()
        align = False if mode == "bilinear" else None
        resized = torch.nn.functional.interpolate(
            tensor, size=self.image_size, mode=mode,
            align_corners=align if mode == "bilinear" else None
        )
        return resized.squeeze().numpy()

    def _augment(self, img: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply basic augmentation (flips + rotation)."""
        if not self.augment:
            return img, mask

        # Random horizontal flip
        if random.random() > 0.5:
            img = np.flip(img, axis=1).copy()
            mask = np.flip(mask, axis=1).copy()

        # Random vertical flip
        if random.random() > 0.5:
            img = np.flip(img, axis=0).copy()
            mask = np.flip(mask, axis=0).copy()

        # Random 90° rotation
        k = random.randint(0, 3)
        if k > 0:
            img = np.rot90(img, k).copy()
            mask = np.rot90(mask, k).copy()

        return img, mask

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get sample with context examples."""
        case_id, label_id, axis = self.samples[idx]

        # Load and preprocess target
        img, mask = self._load_slice(case_id, label_id, axis)
        img = self._normalize(img)
        img = self._resize(img, "bilinear")
        mask = self._resize(mask, "nearest")
        mask = (mask > 0.5).astype(np.float32)  # Binarize

        # Load context
        context_imgs, context_masks, context_ids = [], [], []
        if self.context_size > 0:
            candidates = [c for c in self.valid_contexts.get((label_id, axis), []) if c != case_id]
            random.shuffle(candidates)

            for ctx_case_id in candidates:
                if len(context_imgs) >= self.context_size:
                    break
                try:
                    ctx_img, ctx_mask = self._load_slice(ctx_case_id, label_id, axis)
                    if ctx_mask.max() == 0:
                        continue

                    ctx_img = self._normalize(ctx_img)
                    ctx_img = self._resize(ctx_img, "bilinear")
                    ctx_mask = self._resize(ctx_mask, "nearest")
                    ctx_mask = (ctx_mask > 0.5).astype(np.float32)

                    # Augment context independently
                    ctx_img, ctx_mask = self._augment(ctx_img, ctx_mask)

                    context_imgs.append(ctx_img)
                    context_masks.append(ctx_mask)
                    context_ids.append(ctx_case_id)
                except Exception:
                    continue

            if len(context_imgs) == 0:
                raise RuntimeError(f"No context found for {label_id}/{axis}")

        # Augment target
        img, mask = self._augment(img, mask)

        # Convert to tensors
        target_in = torch.from_numpy(img.copy()).unsqueeze(0).float()
        target_out = torch.from_numpy(mask.copy()).unsqueeze(0).float()

        result = {
            "image": target_in,
            "label": target_out,
            "target_case_id": case_id,
            "label_id": label_id,
            "axis": axis,
        }

        if self.context_size > 0:
            result["context_in"] = torch.stack([
                torch.from_numpy(c.copy()).unsqueeze(0).float() for c in context_imgs
            ])
            result["context_out"] = torch.stack([
                torch.from_numpy(c.copy()).unsqueeze(0).float() for c in context_masks
            ])
            result["context_case_ids"] = context_ids

        return result

    def __len__(self) -> int:
        return len(self.samples)

    def __del__(self):
        for h5f in self._h5_cache.values():
            try:
                h5f.close()
            except Exception:
                pass


def collate_fn(batch: List[Dict]) -> Dict[str, Union[torch.Tensor, List]]:
    """Custom collate function for batching."""
    result = {
        "image": torch.stack([item["image"] for item in batch]),
        "label": torch.stack([item["label"] for item in batch]),
        "target_case_ids": [item["target_case_id"] for item in batch],
        "label_ids": [item["label_id"] for item in batch],
        "axes": [item["axis"] for item in batch],
    }

    if "context_in" in batch[0]:
        result["context_in"] = torch.stack([item["context_in"] for item in batch])
        result["context_out"] = torch.stack([item["context_out"] for item in batch])
        result["context_case_ids"] = [item["context_case_ids"] for item in batch]

    return result


def get_dataloader(
    root_dir: str,
    stats_path: str,
    batch_size: int = 8,
    num_workers: int = 4,
    **kwargs,
) -> DataLoader:
    """Create DataLoader for DeepXcontrast dataset."""
    shuffle = kwargs.pop("shuffle", kwargs.get("is_train", True))

    dataset = DeepXcontrastDataset(
        root_dir=root_dir,
        stats_path=stats_path,
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
    )
