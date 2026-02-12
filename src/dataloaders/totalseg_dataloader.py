"""
TotalSegmentator DataLoader for Medverse Model

This dataloader supports in-context learning by sampling k context examples
for each target image.
"""

import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import torch
from monai.transforms import (
    Compose,
    CropForeground,
    CropForegroundd,
    EnsureChannelFirst,
    EnsureChannelFirstd,
    LoadImaged,
    RandSpatialCropd,
    Resize,
    Resized,
    ScaleIntensityRange,
    ScaleIntensityRanged,
    Spacing,
    Spacingd,
    SpatialPad,
    SpatialPadd,
    ToTensord,
)
from torch.utils.data import DataLoader, Dataset


class TotalSegmentatorDataset(Dataset):
    """
    Dataset for TotalSegmentator in-context learning eval.

    Each sample focuses on label at a time:
    1. For each case, randomly samples one label from label_id_list that exists in that case
    2. Finds k context cases that have the SAME label (with non-zero voxels)
    3. Creates binary segmentation for only that specific label
    4. Context examples always match the target label for better in-context learning
    Example:
        If case s0001 has [liver, kidney_left, spleen]:
        - Sample might choose "liver"
        - Finds 5 other cases with liver segmentation
        - All images show liver (label=1) vs background (label=0)

    Case Filtering:
        Cases are automatically excluded if they have NO label_ids from label_id_list available.
        This happens after applying empty_segmentations exclusions.

    Args for empty_segmentations:
        Recommended: Get from scan_dataset() which checks actual voxel counts:
            from medverse.data.totalseg_utils import scan_dataset
            stats = scan_dataset("/path/to/TotalSegmentator")
            empty_segs = stats['empty_segmentations']

        Format:
            empty_segmentations = {
                'liver': ['s0010', 's0015'],      # Cases with zero liver voxels
                'kidney_left': ['s0020', 's0025'] # Cases with zero kidney_left voxels
            }

        These cases are automatically excluded when building label_id_to_cases mapping.
        Additionally, cases with NO valid label_ids from label_id_list are filtered out entirely.
        If None, assumes all cases have non-zero voxels for all label_ids (not recommended).

    If image_size is specified, all images are resized to that size.
    If image_size is None, context images are dynamically resized to match each target's size.

    Returns:
        Dictionary with:
        - 'target_in': [1, H, W, D] - Target CT scan
        - 'context_in': [k, 1, H, W, D] - k context CT scans (same label_id as target)
        - 'context_out': [k, 1, H, W, D] - k context segmentations (binary: 0=bg, 1=label_id)
        - 'target_out': [1, H, W, D] - Target segmentation (binary: 0=bg, 1=label_id)
        - 'target_case_id': str - Target case identifier
        - 'context_case_ids': List[str] - Context case identifiers
        - 'label_id': str - The specific label_id being segmented in this sample
    """

    def __init__(
        self,
        root_dir: str,
        label_id_list: List[str],
        empty_segmentations: Optional[Dict[str, List[str]]] = None,
        context_size: int = 3,
        image_size: Optional[Tuple[int, int, int]] = (128, 128, 128),
        spacing: Tuple[float, float, float] = (1.5, 1.5, 1.5),
        intensity_range: Tuple[float, float] = (-200, 300),  # HU units for CT
        cache_rate: float = 0.0,
        split: str = 'train',
        random_context: bool = True,
        fixed_context_indices: Optional[List[int]] = None,
        max_ds_len: Optional[int] = None,
    ):
        """
        Args:
            root_dir: Root directory containing s0000, s0001, ... folders
            label_id_list: List of label_ids to segment (e.g., ['liver', 'kidney_left'])
            empty_segmentations: Optional dict mapping label_id -> list of case_ids to exclude
                                 (cases where that label_id has zero voxels)
            context_size: Number of context examples (k)
            image_size: Target spatial size (H, W, D). If None, context images are resized to match target.
            spacing: Target voxel spacing in mm
            intensity_range: HU window for intensity normalization
            cache_rate: Fraction of dataset to cache in memory (0.0 = no cache)
            split: 'train', 'val', or 'test'
            random_context: If True, randomly sample context. If False, use fixed indices.
            fixed_context_indices: If provided and random_context=False, use these indices
            max_ds_len: If provided, limit dataset to this many samples (for debugging)
        """
        self.root_dir = Path(root_dir)
        self.label_id_list = label_id_list
        self.empty_segmentations = empty_segmentations or {}
        self.context_size = context_size
        self.image_size = image_size
        self.spacing = spacing
        self.split = split
        self.random_context = random_context
        self.fixed_context_indices = fixed_context_indices
        self.max_ds_len = max_ds_len

        # Warn if empty_segmentations not provided
        if not self.empty_segmentations:
            print("Warning: empty_segmentations not provided. Assuming all cases have non-zero voxels.")
            print("  Recommend running: stats = scan_dataset(root_dir) and passing stats['empty_segmentations']")

        # Find all case folders
        self.case_folders = sorted([
            f for f in self.root_dir.iterdir()
            if f.is_dir() and f.name.startswith('s')
        ])
        # keep cases from split
        desc_df_path = self.root_dir / "meta.csv"
        import pandas as pd
        df = pd.read_csv(desc_df_path, sep=';')
        split_case_ids = df["image_id"][df["split"]==split].tolist()
        self.case_folders = [f for f in self.case_folders if f.name in split_case_ids]

        self.valid_cases = self.case_folders
        print(f"Found {len(self.valid_cases)} cases in {root_dir}")

        # Build label_id to cases mapping from empty_segmentations
        self.label_id_to_cases = self._build_label_id_to_cases_mapping()
        print(f"Built label_id-to-cases mapping for {len(self.label_id_to_cases)} label_ids")

        # Filter valid_cases to only include cases with at least one label_id from label_id_list
        all_valid_case_ids = set()
        for label_id, case_ids in self.label_id_to_cases.items():
            all_valid_case_ids.update(case_ids)

        self.valid_cases = [
            c for c in self.valid_cases
            if c.name in all_valid_case_ids
        ]
        print(f"Filtered to {len(self.valid_cases)} cases with at least one label_id from label_id_list")

        # Build list of all valid (case_id, label_id) pairs
        # This ensures we process ALL label_ids for ALL cases in one epoch
        self.case_label_id_pairs = []
        for case_folder in self.valid_cases:
            case_id = case_folder.name
            for label_id in self.label_id_list:
                if case_id in self.label_id_to_cases.get(label_id, []):
                    self.case_label_id_pairs.append((case_id, label_id))
        print(f"Created {len(self.case_label_id_pairs)} (case, label_id) pairs for complete coverage")

        # Setup transforms
        self.transforms = self._get_transforms(intensity_range)

        # Cache for loaded data
        self.cache = {}
        self.cache_rate = cache_rate
        if cache_rate > 0:
            num_to_cache = int(len(self.valid_cases) * cache_rate)
            print(f"Caching {num_to_cache} cases...")
            for idx in range(num_to_cache):
                self.cache[idx] = self._load_case(idx)

    def _build_label_id_to_cases_mapping(self) -> Dict[str, List[str]]:
        """
        Build mapping from label_id name to list of case IDs that have that label_id.

        Uses empty_segmentations to efficiently exclude cases without loading files:
        label_id_to_cases[label_id] = all_cases - empty_segmentations[label_id]
        """
        label_id_to_cases = {}

        # Get all case IDs
        all_case_ids = [c.name for c in self.valid_cases]

        for label_id in self.label_id_list:
            # Get cases to exclude for this label_id
            exclude_cases = set(self.empty_segmentations.get(label_id, []))

            # All cases minus excluded ones
            valid_cases_for_label_id = [
                case_id for case_id in all_case_ids
                if case_id not in exclude_cases
            ]

            label_id_to_cases[label_id] = valid_cases_for_label_id

        return label_id_to_cases

    def _get_transforms(self, intensity_range: Tuple[float, float]):
        """Create MONAI transform pipeline."""
        transforms = [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Spacingd(
                keys=["image", "label"],
                pixdim=self.spacing,
                mode=("bilinear", "nearest"),
            ),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=intensity_range[0],
                a_max=intensity_range[1],
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
        ]

        # Only add padding and resizing if image_size is specified
        if self.image_size is not None:
            transforms.extend([
                SpatialPadd(
                    keys=["image", "label"],
                    spatial_size=self.image_size,
                    mode="constant",
                ),
                Resized(
                    keys=["image", "label"],
                    spatial_size=self.image_size,
                    mode=("trilinear", "nearest"),
                ),
            ])

        transforms.append(ToTensord(keys=["image", "label"]))

        return Compose(transforms)

    def _load_single_case(self, case_folder: Path, label_id: str) -> Dict[str, torch.Tensor]:
        """Load CT and segmentation for a single case and specific label_id."""
        ct_path = case_folder / "ct.nii.gz"
        seg_folder = case_folder / "segmentations"
        seg_path = seg_folder / f"{label_id}.nii.gz"

        if not seg_path.exists():
            raise ValueError(f"label_id {label_id} not found in case {case_folder.name}")

        # Apply transforms with file paths (not numpy arrays)
        data_dict = {
            "image": str(ct_path),
            "label": str(seg_path),
        }
        transformed = self.transforms(data_dict)

        return {
            "image": transformed["image"],
            "label": transformed["label"],
            "case_id": case_folder.name,
        }

    def _get_available_label_ids(self, case_folder: Path) -> List[str]:
        """
        Get list of label_ids from label_id_list that exist in this case.

        Uses label_id_to_cases mapping for efficient lookup without loading files.
        """
        case_id = case_folder.name
        available_label_ids = []

        for label_id in self.label_id_list:
            # Check if case_id is in the valid cases for this label_id
            if case_id in self.label_id_to_cases.get(label_id, []):
                available_label_ids.append(label_id)

        return available_label_ids

    def __len__(self) -> int:
        if self.max_ds_len is not None:
            return min(self.max_ds_len, len(self.case_label_id_pairs))
        return len(self.case_label_id_pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample with context examples for a specific label_id.

        Returns:
            Dictionary with:
            - 'target_in': [1, H, W, D]
            - 'context_in': [k, 1, H, W, D]
            - 'context_out': [k, 1, H, W, D] - Binary mask for sampled label_id
            - 'target_out': [1, H, W, D] - Binary mask for sampled label_id
            - 'target_case_id': str
            - 'context_case_ids': List[str]
            - 'label_id': str - The label_id being segmented
        """
        # Look up the (case_id, label_id) pair for this index
        target_case_id, sampled_label_id = self.case_label_id_pairs[idx]
        target_case_folder = self.root_dir / target_case_id

        # Load target case with sampled label_id
        target_data = self._load_single_case(target_case_folder, sampled_label_id)

        # Find context cases that have this same label_id
        cases_with_label_id = self.label_id_to_cases.get(sampled_label_id, [])

        # Filter out target case
        available_context_case_ids = [
            case_id for case_id in cases_with_label_id
            if case_id != target_case_id
        ]

        if len(available_context_case_ids) < self.context_size:
            # Not enough contexts with this label_id, sample fewer
            num_contexts = len(available_context_case_ids)
            if num_contexts == 0:
                raise RuntimeError(
                    f"No context cases available for label_id '{sampled_label_id}' "
                    f"(excluding target case {target_case_id})"
                )
            context_case_ids = available_context_case_ids
        else:
            # Randomly sample k context cases
            if self.random_context:
                context_case_ids = random.sample(available_context_case_ids, self.context_size)
            else:
                # Use first k cases for deterministic selection
                context_case_ids = available_context_case_ids[:self.context_size]

        # Load context cases with the same label_id
        context_images = []
        context_labels = []

        for context_case_id in context_case_ids:
            # Find the case folder for this case_id
            context_case_folder = self.root_dir / context_case_id

            if not context_case_folder.exists():
                # Skip if case folder doesn't exist
                continue

            try:
                ctx_data = self._load_single_case(context_case_folder, sampled_label_id)
                context_images.append(ctx_data["image"])
                context_labels.append(ctx_data["label"])
            except Exception as e:
                # Skip cases that fail to load
                print(f"Warning: Failed to load {context_case_id} for label_id {sampled_label_id}: {e}")
                continue

        # Ensure we have at least one context
        if len(context_images) == 0:
            raise RuntimeError(
                f"Failed to load any context cases for label_id '{sampled_label_id}'"
            )

        # If image_size is None, resize contexts to match target's spatial dimensions
        if self.image_size is None:
            # Get target spatial dimensions [C, H, W, D]
            target_shape = target_data["image"].shape
            target_spatial_size = target_shape[1:]  # (H, W, D)

            # Resize each context image and label to match target
            resized_context_images = []
            resized_context_labels = []

            for ctx_img, ctx_lbl in zip(context_images, context_labels):
                # Resize image with trilinear interpolation
                ctx_img_resized = torch.nn.functional.interpolate(
                    ctx_img.unsqueeze(0),  # Add batch dim: [1, C, H, W, D]
                    size=target_spatial_size,
                    mode='trilinear',
                    align_corners=False
                ).squeeze(0)  # Remove batch dim: [C, H, W, D]

                # Resize label with nearest neighbor interpolation
                ctx_lbl_resized = torch.nn.functional.interpolate(
                    ctx_lbl.unsqueeze(0),  # Add batch dim: [1, C, H, W, D]
                    size=target_spatial_size,
                    mode='nearest'
                ).squeeze(0)  # Remove batch dim: [C, H, W, D]

                resized_context_images.append(ctx_img_resized)
                resized_context_labels.append(ctx_lbl_resized)

            context_images = resized_context_images
            context_labels = resized_context_labels

        # Stack contexts
        context_in = torch.stack(context_images, dim=0)  # [k, C, H, W, D]
        context_out = torch.stack(context_labels, dim=0)  # [k, C, H, W, D]

        return {
            "target_in": target_data["image"],  # [C, H, W, D]
            "context_in": context_in,  # [k, C, H, W, D]
            "context_out": context_out,  # [k, C, H, W, D]
            "target_out": target_data["label"],  # [C, H, W, D]
            "target_case_id": target_data["case_id"],
            "context_case_ids": context_case_ids,
            "label_id": sampled_label_id,
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function to handle batching.

    Args:
        batch: List of samples from __getitem__

    Returns:
        Batched dictionary ready for Medverse model
    """
    # Stack batch dimension
    target_in = torch.stack([item["target_in"] for item in batch], dim=0)  # [B, 1, H, W, D]
    context_in = torch.stack([item["context_in"] for item in batch], dim=0)  # [B, k, 1, H, W, D]
    context_out = torch.stack([item["context_out"] for item in batch], dim=0)  # [B, k, 1, H, W, D]
    target_out = torch.stack([item["target_out"] for item in batch], dim=0)  # [B, 1, H, W, D]

    return {
        "target_in": target_in,
        "context_in": context_in,
        "context_out": context_out,
        "target_out": target_out,
        "target_case_ids": [item["target_case_id"] for item in batch],
        "context_case_ids": [item["context_case_ids"] for item in batch],
        "label_ids": [item["label_id"] for item in batch],
    }


def get_dataloader(
    root_dir: str,
    label_id_list: List[str],
    empty_segmentations: Optional[Dict[str, List[str]]] = None,
    context_size: int = 3,
    batch_size: int = 1,
    image_size: Optional[Tuple[int, int, int]] = (128, 128, 128),
    spacing: Tuple[float, float, float] = (1.5, 1.5, 1.5),
    num_workers: int = 4,
    split: str = 'train',
    shuffle: bool = True,
    random_context: bool = True,
    **dataset_kwargs
) -> DataLoader:
    """
    Create a DataLoader for TotalSegmentator dataset.

    Args:
        root_dir: Root directory of TotalSegmentator dataset
        label_id_list: List of label_ids to segment
        empty_segmentations: Optional dict mapping label_id -> list of case_ids to exclude
        context_size: Number of context examples
        batch_size: Batch size
        image_size: Target spatial dimensions. If None, contexts are resized to match each target.
        spacing: Target voxel spacing
        num_workers: Number of data loading workers
        split: 'train', 'val', or 'test'
        shuffle: Whether to shuffle data
        **dataset_kwargs: Additional arguments for TotalSegmentatorDataset

    Returns:
        DataLoader instance
    """
    dataset = TotalSegmentatorDataset(
        root_dir=root_dir,
        label_id_list=label_id_list,
        empty_segmentations=empty_segmentations,
        context_size=context_size,
        image_size=image_size,
        spacing=spacing,
        split=split,
        random_context=random_context,
        **dataset_kwargs
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    return dataloader
