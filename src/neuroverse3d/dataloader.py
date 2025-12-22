"""
Dataloader for Neuroverse3D training.

This module provides dataset classes for loading 3D medical imaging data
in nnUNet format and preparing it for in-context learning.
"""

import os
import random
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import nibabel as nib
import numpy as np
import torch
from scipy.ndimage import zoom
from torch.utils.data import Dataset


class Neuroverse3DDataset(Dataset):
    """
    Dataset for loading 3D medical images in nnUNet format for in-context learning.

    Expected directory structure:
        data_dir/
            imagesTr/
                case_001_0000.nii.gz
                case_002_0000.nii.gz
                ...
            labelsTr/
                case_001.nii.gz
                case_002.nii.gz
                ...
    """

    def __init__(
        self,
        data_dir: str,
        mode: str = 'train',
        train_val_split: float = 0.8,
        target_size: Tuple[int, int, int] = (128, 128, 128),
        normalize: bool = True,
        augment: bool = True,
    ):
        """
        Initialize dataset.

        Args:
            data_dir: Directory containing imagesTr and labelsTr folders
            mode: 'train' or 'val'
            train_val_split: Fraction of data to use for training
            target_size: Target volume size (H, W, D)
            normalize: Whether to normalize images
            augment: Whether to apply data augmentation (train only)
        """
        self.data_dir = Path(data_dir)
        self.mode = mode
        self.target_size = target_size
        self.normalize = normalize
        self.augment = augment and (mode == 'train')

        # Find all cases
        self.images_dir = self.data_dir / 'imagesTr'
        self.labels_dir = self.data_dir / 'labelsTr'

        if not self.images_dir.exists() or not self.labels_dir.exists():
            raise ValueError(
                f"Data directory must contain 'imagesTr' and 'labelsTr' folders. "
                f"Got: {self.data_dir}"
            )

        # Get all image files
        image_files = sorted(list(self.images_dir.glob('*_0000.nii.gz')))
        case_ids = [f.name.split('_0000')[0] for f in image_files]

        # Split train/val
        n_train = int(len(case_ids) * train_val_split)
        if mode == 'train':
            self.case_ids = case_ids[:n_train]
        else:
            self.case_ids = case_ids[n_train:]

        print(f"Loaded {len(self.case_ids)} cases for {mode}")

    def __len__(self):
        return len(self.case_ids)

    def load_nifti(self, path: Path) -> np.ndarray:
        """Load and preprocess a NIfTI file."""
        nii = nib.load(str(path))
        data = nii.get_fdata()
        return data

    def resize_volume(self, volume: np.ndarray) -> np.ndarray:
        """Resize volume to target size."""
        current_size = volume.shape
        zoom_factors = [t / c for t, c in zip(self.target_size, current_size)]

        # Use different interpolation for labels (order=0) vs images (order=1)
        order = 0 if volume.dtype == np.int64 or len(np.unique(volume)) < 50 else 1
        resized = zoom(volume, zoom_factors, order=order)

        return resized

    def normalize_volume(self, volume: np.ndarray) -> np.ndarray:
        """Normalize volume using z-score normalization."""
        # Clip outliers
        p1, p99 = np.percentile(volume, [1, 99])
        volume = np.clip(volume, p1, p99)

        # Z-score normalization
        mean = volume.mean()
        std = volume.std()
        if std > 0:
            volume = (volume - mean) / std

        return volume

    def augment_volume(self, image: np.ndarray, label: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply random augmentations."""
        # Random flip
        for axis in range(3):
            if random.random() < 0.5:
                image = np.flip(image, axis=axis).copy()
                label = np.flip(label, axis=axis).copy()

        # Random rotation (90-degree increments)
        if random.random() < 0.5:
            k = random.randint(1, 3)
            axes = random.choice([(0, 1), (0, 2), (1, 2)])
            image = np.rot90(image, k=k, axes=axes).copy()
            label = np.rot90(label, k=k, axes=axes).copy()

        return image, label

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single case."""
        case_id = self.case_ids[idx]

        # Load image and label
        image_path = self.images_dir / f"{case_id}_0000.nii.gz"
        label_path = self.labels_dir / f"{case_id}.nii.gz"

        image = self.load_nifti(image_path)
        label = self.load_nifti(label_path)

        # Resize
        image = self.resize_volume(image)
        label = self.resize_volume(label)

        # Normalize image
        if self.normalize:
            image = self.normalize_volume(image)

        # Augment
        if self.augment:
            image, label = self.augment_volume(image, label)

        # Convert to tensors
        image = torch.from_numpy(image).float().unsqueeze(0)  # Add channel dim
        label = torch.from_numpy(label).long().unsqueeze(0)

        return {
            'image': image,
            'label': label,
            'case_id': case_id,
        }


class MetaDataset_Multi_Extended(Dataset):
    """
    Extended dataset for in-context learning with variable context sizes.

    This dataset samples groups of images to create support sets (context)
    and query samples for in-context learning.
    """

    def __init__(
        self,
        mode: str,
        config,
        group_size: int = 4,  # Number of samples per batch (context + query)
        random_context: bool = False,
        min_context: int = 2,
        max_context: int = 9,
    ):
        """
        Initialize meta-dataset.

        Args:
            mode: 'train' or 'val'
            config: MultiDatasetConfig object
            group_size: Total number of samples (context examples + 1 query)
            random_context: Whether to use random context sizes
            min_context: Minimum context size (if random_context=True)
            max_context: Maximum context size (if random_context=True)
        """
        self.mode = mode
        self.config = config
        self.group_size = group_size
        self.random_context = random_context
        self.min_context = min_context
        self.max_context = max_context

        # Load all datasets
        self.datasets = []
        for dataset_config in config.datasets:
            dataset = Neuroverse3DDataset(
                data_dir=dataset_config.path,
                mode=mode,
                train_val_split=0.8,
            )
            # Sample according to sample_rate
            n_samples = int(len(dataset) * dataset_config.sample_rate)
            self.datasets.append((dataset, n_samples))

        # Calculate total samples
        self.total_samples = sum(n for _, n in self.datasets)
        print(f"Total {mode} samples across {len(self.datasets)} datasets: {self.total_samples}")

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx: int):
        """
        Get a group of samples for in-context learning.

        Returns:
            Dictionary with:
                - target_in: Query image (1, C, H, W, D)
                - target_out: Query label (1, C, H, W, D)
                - context_in: Context images (1, N, C, H, W, D)
                - context_out: Context labels (1, N, C, H, W, D)
        """
        # Determine context size
        if self.random_context:
            context_size = random.randint(self.min_context, self.max_context)
        else:
            context_size = self.group_size - 1

        # Randomly select a dataset
        dataset, _ = random.choice(self.datasets)

        # Sample context_size + 1 cases from this dataset
        indices = random.sample(range(len(dataset)), context_size + 1)

        # Load all samples
        samples = [dataset[i] for i in indices]

        # Last sample is the query
        query = samples[-1]
        context_samples = samples[:-1]

        # Stack context samples
        context_images = torch.stack([s['image'] for s in context_samples], dim=0)
        context_labels = torch.stack([s['label'] for s in context_samples], dim=0)

        return {
            'target_in': query['image'].unsqueeze(0),  # (1, C, H, W, D)
            'target_out': query['label'].unsqueeze(0),  # (1, C, H, W, D)
            'context_in': context_images.unsqueeze(0),  # (1, N, C, H, W, D)
            'context_out': context_labels.unsqueeze(0),  # (1, N, C, H, W, D)
        }


def create_dataloaders(
    config,
    batch_size: int = 1,
    num_workers: int = 4,
    context_size: int = 3,
    random_context: bool = False,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create train and validation dataloaders.

    Args:
        config: MultiDatasetConfig object
        batch_size: Batch size
        num_workers: Number of data loading workers
        context_size: Number of context examples
        random_context: Whether to use random context sizes (Stage 2)

    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create datasets
    train_dataset = MetaDataset_Multi_Extended(
        mode='train',
        config=config,
        group_size=context_size + 1,
        random_context=random_context,
        min_context=2 if random_context else context_size,
        max_context=9 if random_context else context_size,
    )

    val_dataset = MetaDataset_Multi_Extended(
        mode='val',
        config=config,
        group_size=context_size + 1,
        random_context=False,  # Always use fixed context for validation
    )

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader
