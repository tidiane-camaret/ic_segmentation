"""
SynthMorph Synthetic Dataloader

On-the-fly synthetic segmentation task generation for in-context learning.
Generates (target + k context) samples per iteration, matching MedSegBench interface.

Based on SynthMorph (Hoffmann et al., 2022) and UniverSeg (Butoi et al., 2023).
"""

import random
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.dataloaders.synthmorph_utils import (
    generate_base_label_map,
    generate_subject,
)
from src.dataloaders.augmentations import apply_universeg_augmentation


class LRUCache:
    """Simple LRU cache for base label maps."""

    def __init__(self, max_size: int = 500):
        self.max_size = max_size
        self._cache: OrderedDict = OrderedDict()

    def get(self, key):
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def put(self, key, value):
        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            if len(self._cache) >= self.max_size:
                self._cache.popitem(last=False)
            self._cache[key] = value


class SynthMorphDataset(Dataset):
    """
    On-the-fly synthetic segmentation dataset.

    Each task is defined by a seed that deterministically generates a base
    label map (anatomy). Subjects are created by applying random deformations
    and synthesizing intensity images.

    Returns same format as MedSegBenchDataset for training compatibility.

    Args:
        num_tasks: Number of unique anatomies (task seeds)
        num_labels: Number of regions per task
        context_size: Number of context examples per sample
        image_size: Target spatial size (H, W)
        epoch_length: Virtual dataset length per epoch
        master_seed: Seed for generating task seeds
        max_cache_size: LRU cache size for base label maps
        sigma_range: Range for base label smoothing (controls region size)
        sigma_def: Deformation magnitude
        sigma_smooth: Deformation smoothness
        augment: Whether to apply augmentation
        augmentation_config: Augmentation parameters (universeg-style)
    """

    def __init__(
        self,
        num_tasks: int = 1000,
        num_labels: int = 16,
        context_size: int = 3,
        image_size: Tuple[int, int] = (256, 256),
        epoch_length: int = 10000,
        master_seed: int = 42,
        max_cache_size: int = 500,
        # Generation parameters
        sigma_range: Tuple[float, float] = (5.0, 15.0),
        sigma_def: float = 2.0,
        sigma_smooth: float = 8.0,
        # Augmentation
        augment: bool = False,
        augmentation_config: Optional[Dict] = None,
    ):
        self.num_tasks = num_tasks
        self.num_labels = num_labels
        self.context_size = context_size
        self.image_size = image_size
        self.epoch_length = epoch_length
        self.sigma_range = sigma_range
        self.sigma_def = sigma_def
        self.sigma_smooth = sigma_smooth

        # Generate deterministic task seeds
        master_rng = np.random.default_rng(master_seed)
        self.task_seeds = [
            int(master_rng.integers(0, 2**31)) for _ in range(num_tasks)
        ]

        # LRU cache for base label maps
        self._label_cache = LRUCache(max_size=max_cache_size)

        # Augmentation setup
        self.augment = augment or (
            augmentation_config is not None
            and augmentation_config.get("enabled", False)
        )
        self.augmentation_config = augmentation_config

        print(
            f"SynthMorphDataset initialized: {num_tasks} tasks, {num_labels} labels, "
            f"context_size={context_size}, image_size={image_size}"
        )
        if self.augment:
            print(f"  Augmentation enabled: {augmentation_config}")

    def __len__(self) -> int:
        return self.epoch_length

    def _get_base_label(self, task_idx: int) -> np.ndarray:
        """Get or generate base label map for a task (cached)."""
        cached = self._label_cache.get(task_idx)
        if cached is not None:
            return cached

        # Generate deterministically from task seed
        rng = np.random.default_rng(self.task_seeds[task_idx])
        label = generate_base_label_map(
            shape=self.image_size,
            num_labels=self.num_labels,
            sigma_range=self.sigma_range,
            rng=rng,
        )
        self._label_cache.put(task_idx, label)
        return label

    def _resize(self, arr: np.ndarray, mode: str = "bilinear") -> np.ndarray:
        """Resize 2D array to self.image_size if needed."""
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

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Generate a synthetic sample with context.

        Returns:
            Dictionary with:
            - 'image': [1, H, W] target image
            - 'label': [1, H, W] target binary mask
            - 'context_in': [k, 1, H, W] context images
            - 'context_out': [k, 1, H, W] context masks
            - 'target_case_id': str
            - 'context_case_ids': List[str]
            - 'label_id': str
        """
        # Fresh RNG for each call (infinite subject diversity)
        rng = np.random.default_rng()

        # Pick random task
        task_idx = rng.integers(0, self.num_tasks)
        base_label = self._get_base_label(task_idx)

        # Pick random foreground label (exclude 0 which may be background-like)
        # Actually in synthmorph all labels are equal, but we exclude label 0
        # to be consistent with real data where 0 is background
        fg_label = rng.integers(1, self.num_labels)

        # Generate target subject
        target_img, target_label_map = generate_subject(
            base_label,
            num_labels=self.num_labels,
            sigma_def=self.sigma_def,
            sigma_smooth=self.sigma_smooth,
            rng=rng,
        )
        target_binary = (target_label_map == fg_label).astype(np.float32)

        # Generate context subjects
        context_imgs = []
        context_masks = []
        context_subject_ids = []

        for ctx_i in range(self.context_size):
            ctx_img, ctx_label_map = generate_subject(
                base_label,
                num_labels=self.num_labels,
                sigma_def=self.sigma_def,
                sigma_smooth=self.sigma_smooth,
                rng=rng,
            )
            ctx_binary = (ctx_label_map == fg_label).astype(np.float32)
            context_imgs.append(ctx_img)
            context_masks.append(ctx_binary)
            context_subject_ids.append(f"synth_t{task_idx}_l{fg_label}_s{ctx_i}")

        # Apply augmentation
        if self.augment and self.augmentation_config is not None:
            aug_cfg = dict(self.augmentation_config)
            aug_cfg["img_size"] = self.image_size[0]
            target_img, target_binary, context_imgs, context_masks = (
                apply_universeg_augmentation(
                    target_img,
                    target_binary,
                    context_imgs,
                    context_masks,
                    full_config=aug_cfg,
                )
            )

        # Convert to tensors
        target_in = torch.from_numpy(target_img.copy()).unsqueeze(0).float()
        target_out = torch.from_numpy(target_binary.copy()).unsqueeze(0).float()

        # Build case/label IDs
        target_case_id = f"synth_t{task_idx}_l{fg_label}_target"
        label_id = f"synth_t{task_idx}_l{fg_label}"

        result = {
            "image": target_in,
            "label": target_out,
            "target_case_id": target_case_id,
            "context_case_ids": context_subject_ids,
            "label_id": label_id,
        }

        if context_imgs:
            context_in = torch.stack(
                [
                    torch.from_numpy(c.copy()).unsqueeze(0).float()
                    for c in context_imgs
                ]
            )
            context_out = torch.stack(
                [
                    torch.from_numpy(c.copy()).unsqueeze(0).float()
                    for c in context_masks
                ]
            )
            result["context_in"] = context_in
            result["context_out"] = context_out

        return result


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate function for batching (same as MedSegBench)."""
    result = {
        "image": torch.stack([item["image"] for item in batch]),
        "label": torch.stack([item["label"] for item in batch]),
        "target_case_ids": [item["target_case_id"] for item in batch],
        "context_case_ids": [item["context_case_ids"] for item in batch],
        "label_ids": [item["label_id"] for item in batch],
    }

    if "context_in" in batch[0]:
        max_k = max(item["context_in"].shape[0] for item in batch)
        h, w = batch[0]["image"].shape[1:]

        context_in_list = []
        context_out_list = []
        for item in batch:
            k = item["context_in"].shape[0]
            if k < max_k:
                pad_in = torch.zeros(max_k - k, 1, h, w)
                pad_out = torch.zeros(max_k - k, 1, h, w)
                context_in_list.append(
                    torch.cat([item["context_in"], pad_in], dim=0)
                )
                context_out_list.append(
                    torch.cat([item["context_out"], pad_out], dim=0)
                )
            else:
                context_in_list.append(item["context_in"])
                context_out_list.append(item["context_out"])

        result["context_in"] = torch.stack(context_in_list)
        result["context_out"] = torch.stack(context_out_list)

    return result


def get_synthmorph_dataloader(
    num_tasks: int = 1000,
    num_labels: int = 16,
    context_size: int = 3,
    batch_size: int = 8,
    image_size: Tuple[int, int] = (256, 256),
    epoch_length: int = 10000,
    num_workers: int = 4,
    shuffle: bool = True,
    augment: bool = False,
    augmentation_config: Optional[Dict] = None,
    **kwargs,
) -> DataLoader:
    """Create a DataLoader for SynthMorph synthetic data."""
    dataset = SynthMorphDataset(
        num_tasks=num_tasks,
        num_labels=num_labels,
        context_size=context_size,
        image_size=image_size,
        epoch_length=epoch_length,
        augment=augment,
        augmentation_config=augmentation_config,
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


class MixedDataLoader:
    """
    Iterator that mixes synthetic and real data batches.

    Args:
        real_loader: DataLoader for real data (MedSegBench)
        synth_loader: DataLoader for synthetic data (SynthMorph)
        synth_ratio: Probability of sampling from synthetic data
        seed: Random seed for reproducibility
    """

    def __init__(
        self,
        real_loader: DataLoader,
        synth_loader: DataLoader,
        synth_ratio: float = 0.5,
        seed: Optional[int] = None,
    ):
        self.real_loader = real_loader
        self.synth_loader = synth_loader
        self.synth_ratio = synth_ratio
        self.rng = random.Random(seed)

        # Compute effective length
        self._length = max(len(real_loader), len(synth_loader))

    def __len__(self) -> int:
        return self._length

    def __iter__(self):
        real_iter = iter(self.real_loader)
        synth_iter = iter(self.synth_loader)

        for _ in range(self._length):
            if self.rng.random() < self.synth_ratio:
                try:
                    batch = next(synth_iter)
                except StopIteration:
                    synth_iter = iter(self.synth_loader)
                    batch = next(synth_iter)
            else:
                try:
                    batch = next(real_iter)
                except StopIteration:
                    real_iter = iter(self.real_loader)
                    batch = next(real_iter)
            yield batch
