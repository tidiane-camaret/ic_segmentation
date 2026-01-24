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

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from data.label_ids_totalseg import get_label_ids


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
    ):
        self.root_dir = Path(root_dir)

        # Resolve label_id_list if it's a string split name
        if isinstance(label_id_list, str):
            self.label_id_list = get_label_ids(label_id_list)
            print(f"Using {label_id_list} label split: {len(self.label_id_list)} labels")
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

        # Load stats dict
        with open(stats_path, "rb") as f:
            self.stats = pickle.load(f)
        print(f"Loaded stats for {len(self.stats)} cases")

        # Convert label_id_list to set for faster lookup
        label_id_set = set(self.label_id_list)

        # Build label_id -> case_ids mapping
        self.label_to_cases: Dict[str, List[str]] = {}
        all_labels_in_stats = set()
        for case_id, labels in self.stats.items():
            for label_id in labels.keys():
                all_labels_in_stats.add(label_id)
                if label_id in label_id_set:
                    self.label_to_cases.setdefault(label_id, []).append(case_id)

        # Debug: show mismatch if no labels found
        if len(self.label_to_cases) == 0:
            sample_stats_labels = list(all_labels_in_stats)[:5]
            sample_requested_labels = self.label_id_list[:5]
            print(f"WARNING: No matching labels found!")
            print(f"  Sample labels in stats: {sample_stats_labels}")
            print(f"  Sample requested labels: {sample_requested_labels}")

        print(f"Built mapping for {len(self.label_to_cases)} labels (stats has {len(all_labels_in_stats)} unique labels)")

        # Filter by split if provided
        if split is not None:
            self._filter_by_split(split)

        # Build list of all (case_id, label_id, axis) samples
        self.samples = []
        for case_id, labels in self.stats.items():
            for label_id in labels.keys():
                if label_id in label_id_set:
                    # Check case exists in root_dir
                    case_dir = self.root_dir / case_id / label_id
                    if case_dir.exists():
                        for axis in self.axes:
                            self.samples.append((case_id, label_id, axis))

        print(f"Created {len(self.samples)} samples")

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

    def _load_slice(self, case_id: str, label_id: str, axis: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load image and label slice for a specific case/label/axis.

        Supports both .npy (fast) and .nii.gz (legacy) formats.
        """
        slice_dir = self.root_dir / case_id / label_id

        # Try .npy first (fast), fall back to .nii.gz (legacy)
        img_path_npy = slice_dir / f"{axis}_slice_img.npy"
        label_path_npy = slice_dir / f"{axis}_slice.npy"

        if img_path_npy.exists():
            img = np.load(img_path_npy)
            label = np.load(label_path_npy)
        else:
            # Legacy .nii.gz format
            img_path = slice_dir / f"{axis}_slice_img.nii.gz"
            label_path = slice_dir / f"{axis}_slice.nii.gz"
            img = nib.load(str(img_path)).get_fdata().astype(np.float32)
            label = nib.load(str(label_path)).get_fdata().astype(np.float32)

        return img, label

    def _load_features(self, case_id: str, label_id: str, axis: str) -> Optional[np.ndarray]:
        """Load pre-computed DINOv3 features for a specific case/label/axis.

        Returns:
            features: [196, 1024] patch features (14x14 grid) or None if not found
        """
        slice_dir = self.root_dir / case_id / label_id
        features_path = slice_dir / f"{axis}_slice_img_dinov3.npz"

        if not features_path.exists():
            return None

        data = np.load(str(features_path))
        return data["features"].astype(np.float32)

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

    def _normalize_image(self, img: np.ndarray, a_min: float = -200, a_max: float = 300) -> np.ndarray:
        """Normalize CT image to [0, 1] range."""
        img = np.clip(img, a_min, a_max)
        img = (img - a_min) / (a_max - a_min)
        return img

    def _resize(self, arr: np.ndarray, size: Tuple[int, int], mode: str = "bilinear") -> np.ndarray:
        """Resize 2D array to target size."""
        tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        if mode == "bilinear":
            resized = torch.nn.functional.interpolate(tensor, size=size, mode="bilinear", align_corners=False)
        else:
            resized = torch.nn.functional.interpolate(tensor, size=size, mode="nearest")
        return resized.squeeze().numpy()

    def __len__(self) -> int:
        if self.max_ds_len is not None:
            return min(self.max_ds_len, len(self.samples))
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample with context examples.

        Returns:
            Dictionary with:
            - 'image': [1, H, W] - Target image
            - 'label': [1, H, W] - Target label
            - 'context_in': [k, 1, H, W] - Context images
            - 'context_out': [k, 1, H, W] - Context labels
            - 'target_case_id': str
            - 'context_case_ids': List[str]
            - 'label_id': str
            - 'axis': str
            If load_dinov3_features=True, also includes:
            - 'target_features': [196, 1024] - Target DINOv3 patch features
            - 'context_features': [k, 196, 1024] - Context DINOv3 patch features
        """
        target_case_id, label_id, axis = self.samples[idx]

        # Load target
        target_img, target_label = self._load_slice(target_case_id, label_id, axis)

        # Load target features if requested
        target_features = None
        if self.load_dinov3_features:
            target_features = self._load_features(target_case_id, label_id, axis)

        # Crop to bbox if enabled
        if self.crop_to_bbox:
            target_bbox = self._get_2d_bbox(target_case_id, label_id, axis)
            target_img, target_label = self._crop_to_bbox(target_img, target_label, target_bbox)

        target_img = self._normalize_image(target_img)

        # Get context cases with same label (excluding target)
        available_contexts = [c for c in self.label_to_cases.get(label_id, []) if c != target_case_id]

        if len(available_contexts) == 0:
            raise RuntimeError(f"No context cases for label '{label_id}'")

        # Sample context cases
        k = min(self.context_size, len(available_contexts))
        if self.random_context:
            context_case_ids = random.sample(available_contexts, k)
        else:
            context_case_ids = available_contexts[:k]

        # Load context slices
        context_imgs = []
        context_labels = []
        context_features_list = []
        valid_context_ids = []

        for ctx_case_id in context_case_ids:
            try:
                ctx_img, ctx_label = self._load_slice(ctx_case_id, label_id, axis)

                # Load context features if requested
                ctx_features = None
                if self.load_dinov3_features:
                    ctx_features = self._load_features(ctx_case_id, label_id, axis)

                # Crop to bbox if enabled
                if self.crop_to_bbox:
                    ctx_bbox = self._get_2d_bbox(ctx_case_id, label_id, axis)
                    ctx_img, ctx_label = self._crop_to_bbox(ctx_img, ctx_label, ctx_bbox)

                ctx_img = self._normalize_image(ctx_img)
                context_imgs.append(ctx_img)
                context_labels.append(ctx_label)
                if ctx_features is not None:
                    context_features_list.append(ctx_features)
                valid_context_ids.append(ctx_case_id)
            except Exception as e:
                print(f"Warning: Failed to load {ctx_case_id}/{label_id}/{axis}: {e}")
                continue

        if len(context_imgs) == 0:
            raise RuntimeError(f"Failed to load any context for label '{label_id}'")

        # Resize if needed
        if self.image_size is not None:
            target_img = self._resize(target_img, self.image_size, mode="bilinear")
            target_label = self._resize(target_label, self.image_size, mode="nearest")
            context_imgs = [self._resize(c, self.image_size, mode="bilinear") for c in context_imgs]
            context_labels = [self._resize(c, self.image_size, mode="nearest") for c in context_labels]

        # Convert to tensors
        target_in = torch.from_numpy(target_img).unsqueeze(0)  # [1, H, W]
        target_out = torch.from_numpy(target_label).unsqueeze(0)  # [1, H, W]
        context_in = torch.stack([torch.from_numpy(c).unsqueeze(0) for c in context_imgs])  # [k, 1, H, W]
        context_out = torch.stack([torch.from_numpy(c).unsqueeze(0) for c in context_labels])  # [k, 1, H, W]

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
                result["target_features"] = torch.from_numpy(target_features)  # [196, 1024]
            if context_features_list:
                result["context_features"] = torch.stack(
                    [torch.from_numpy(f) for f in context_features_list]
                )  # [k, 196, 1024]

        return result


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Custom collate function for batching."""
    result = {
        "image": torch.stack([item["image"] for item in batch]),  # [B, 1, H, W]
        "label": torch.stack([item["label"] for item in batch]),  # [B, 1, H, W]
        "context_in": torch.stack([item["context_in"] for item in batch]),  # [B, k, 1, H, W]
        "context_out": torch.stack([item["context_out"] for item in batch]),  # [B, k, 1, H, W]
        "target_case_ids": [item["target_case_id"] for item in batch],
        "context_case_ids": [item["context_case_ids"] for item in batch],
        "label_ids": [item["label_id"] for item in batch],
        "axes": [item["axis"] for item in batch],
    }

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
