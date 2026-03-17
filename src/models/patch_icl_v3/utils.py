"""Shared utilities for PatchICL v3.

Consolidated functions for entropy, confidence, and mask processing.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def compute_binary_entropy(p: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Compute normalized binary entropy H(p)/log(2) in [0, 1].

    Args:
        p: Probabilities in [0, 1]
        eps: Numerical stability epsilon

    Returns:
        Entropy values in [0, 1] - high at p=0.5, low at p=0 or 1
    """
    p = p.clamp(eps, 1 - eps)
    return -(p * p.log() + (1 - p) * (1 - p).log()) / math.log(2)


def compute_confidence_map(logits: torch.Tensor) -> torch.Tensor:
    """Compute confidence map as 1 - entropy.

    Args:
        logits: Raw prediction logits

    Returns:
        Confidence in [0, 1] - high where predictions are confident
    """
    p = torch.sigmoid(logits)
    return 1.0 - compute_binary_entropy(p)


def compute_entropy_sampling_map(
    logits: torch.Tensor,
    temperature: float | torch.Tensor = 1.0,
) -> torch.Tensor:
    """Compute sampling map from binary prediction entropy.

    Returns confidence (1 - entropy): high values where predictions are confident,
    low values where uncertain (good for sampling uncertain regions).

    Args:
        logits: [B, K, C, H, W] - raw prediction logits
        temperature: Temperature for logit scaling (default 1.0)

    Returns:
        sampling_map: [B, K, 1, H, W] - confidence values in [0, 1]
    """
    is_unit_temp = (temperature == 1.0) if isinstance(temperature, (int, float)) else False
    scaled_logits = logits if is_unit_temp else logits / temperature

    orig_dtype = scaled_logits.dtype
    p = torch.sigmoid(scaled_logits.float())
    confidence = 1.0 - compute_binary_entropy(p)

    sampling_map = confidence.to(orig_dtype)
    if sampling_map.shape[2] > 1:
        sampling_map = sampling_map.mean(dim=2, keepdim=True)
    return sampling_map


def downsample_mask(
    mask: torch.Tensor,
    resolution: int,
    min_value: float = 1.0,
) -> torch.Tensor:
    """Downsample mask with hybrid area + max pooling.

    Uses area interpolation for soft targets, but ensures small objects
    get a minimum value via max pooling to prevent tiny labels from vanishing.

    Args:
        mask: [B, C, H, W] input mask
        resolution: Target resolution
        min_value: Minimum value for pixels where max pooling detects foreground

    Returns:
        Downsampled mask [B, C, resolution, resolution]
    """
    if mask.shape[-1] == resolution and mask.shape[-2] == resolution:
        return mask

    mask_float = mask.float()

    # Area interpolation: soft area fractions
    if mask.shape[1] > 1:  # Multi-channel (RGB)
        area_pooled = F.interpolate(
            mask_float, size=(resolution, resolution),
            mode='bilinear', align_corners=False
        )
    else:
        area_pooled = F.interpolate(
            mask_float, size=(resolution, resolution), mode='area'
        )

    # Max pooling: preserves small objects
    max_pooled = F.adaptive_max_pool2d(mask_float, (resolution, resolution))

    # Hybrid: area value with minimum where max detects foreground
    return torch.maximum(area_pooled, max_pooled * min_value)


def resize_label(
    label: torch.Tensor,
    size: tuple[int, int],
    min_value: float = 1.0,
) -> torch.Tensor:
    """Resize label with hybrid area + max pooling.

    Same as downsample_mask but accepts size tuple.

    Args:
        label: [B, C, H, W] float tensor
        size: Target (H, W)
        min_value: Minimum foreground weight

    Returns:
        Resized label [B, C, *size]
    """
    if label.shape[-2:] == size:
        return label

    area = F.interpolate(label, size=size, mode='area')
    maxp = F.adaptive_max_pool2d(label, size)
    return torch.maximum(area, maxp * min_value)


def mask_to_weights(mask: torch.Tensor) -> torch.Tensor:
    """Convert multi-channel mask to single-channel weights.

    Args:
        mask: [B, C, H, W] mask tensor

    Returns:
        [B, 1, H, W] weight tensor
    """
    if mask.shape[1] == 1:
        return mask
    return mask.max(dim=1, keepdim=True)[0]


def create_coverage_mask(
    coords: torch.Tensor,
    patch_size: int,
    output_size: tuple[int, int],
) -> torch.Tensor:
    """Create coverage mask from patch coordinates.

    Uses scatter-based approach for memory efficiency.

    Args:
        coords: [B, K, 2] - Patch coordinates (h, w) at level resolution
        patch_size: Size of patches
        output_size: (H, W) output resolution

    Returns:
        coverage_mask: [B, 1, H, W] - 1 where patches cover, 0 elsewhere
    """
    B, K, _ = coords.shape
    H, W = output_size
    device = coords.device

    # Initialize flat coverage mask [B, H*W]
    coverage_flat = torch.zeros(B, H * W, device=device)

    # Create patch pixel offsets: [patch_size^2, 2]
    ph = torch.arange(patch_size, device=device)
    pw = torch.arange(patch_size, device=device)
    offset_h, offset_w = torch.meshgrid(ph, pw, indexing='ij')
    offsets_h = offset_h.flatten()
    offsets_w = offset_w.flatten()

    # Expand coords to all pixels in each patch
    h_coords = coords[:, :, 0:1] + offsets_h.view(1, 1, -1)
    w_coords = coords[:, :, 1:2] + offsets_w.view(1, 1, -1)

    # Clip to valid range and compute flat indices
    h_coords = h_coords.clamp(0, H - 1).long()
    w_coords = w_coords.clamp(0, W - 1).long()
    flat_indices = h_coords * W + w_coords

    # Scatter ones to coverage mask
    batch_idx = torch.arange(B, device=device).view(B, 1, 1).expand_as(flat_indices)
    coverage_flat[batch_idx.flatten(), flat_indices.flatten()] = 1.0

    return coverage_flat.view(B, 1, H, W)


def extract_patch_features(
    features: torch.Tensor,
    coords: torch.Tensor,
    patch_size: int,
    level_resolution: int,
    feature_grid_size: int = 16,
    target_patch_grid_size: int | None = None,
) -> torch.Tensor:
    """Extract patch features from pre-computed full-image feature maps.

    Uses grid_sample with memory-efficient per-batch processing.

    Args:
        features: [B, N, D] - Pre-computed features (N = grid_size^2)
        coords: [B, K, 2] - Patch coordinates (h, w) at level resolution
        patch_size: Size of patches at level resolution
        level_resolution: Resolution of the current level
        feature_grid_size: Size of feature grid
        target_patch_grid_size: Desired spatial size per patch for encoder

    Returns:
        patch_features: [B, K, tokens_per_patch, D]
    """
    B, K, _ = coords.shape
    device = features.device
    N = features.shape[1]
    D = features.shape[2]

    # Infer actual grid size from feature count
    actual_grid_size = int(N ** 0.5)
    assert actual_grid_size * actual_grid_size == N, f"Features must be square grid, got N={N}"

    scale = actual_grid_size / level_resolution
    patch_size_in_features = max(1, int(patch_size * scale))

    # Use target grid size for extraction if specified
    extract_size = target_patch_grid_size if target_patch_grid_size else patch_size_in_features

    # Reshape features to spatial format: [B, D, H, W]
    features_spatial = features.view(B, actual_grid_size, actual_grid_size, D)
    features_spatial = features_spatial.permute(0, 3, 1, 2)

    # Compute patch centers in normalized coordinates [-1, 1]
    fh = coords[:, :, 0].float() * scale + patch_size_in_features / 2
    fw = coords[:, :, 1].float() * scale + patch_size_in_features / 2

    center_h = (fh / actual_grid_size) * 2 - 1
    center_w = (fw / actual_grid_size) * 2 - 1

    # Create sampling grid offsets
    half_patch = (patch_size_in_features / actual_grid_size)
    offset = torch.linspace(-half_patch, half_patch, extract_size, device=device)
    grid_y, grid_x = torch.meshgrid(offset, offset, indexing='ij')
    grid_offsets = torch.stack([grid_x, grid_y], dim=-1)

    # Build grid: [B, K, extract_size, extract_size, 2]
    centers = torch.stack([center_w, center_h], dim=-1)
    grid = centers.view(B, K, 1, 1, 2) + grid_offsets.view(1, 1, extract_size, extract_size, 2)

    # Memory-efficient extraction: process per batch
    patch_features_list = []
    for b in range(B):
        feat_b = features_spatial[b:b+1].expand(K, -1, -1, -1)
        grid_b = grid[b]
        patches_b = F.grid_sample(
            feat_b, grid_b, mode='bilinear',
            padding_mode='border', align_corners=False
        )
        patch_features_list.append(patches_b)

    # Stack and reshape: [B, K, D, extract_size^2] -> [B, K, tokens, D]
    patch_features = torch.stack(patch_features_list, dim=0)
    patch_features = patch_features.view(B, K, D, extract_size * extract_size)
    patch_features = patch_features.permute(0, 1, 3, 2)

    return patch_features


def extract_mask_patches(
    mask: torch.Tensor,
    coords: torch.Tensor,
    patch_size: int,
    level_resolution: int,
    target_size: int = 16,
) -> torch.Tensor:
    """Extract mask patches using grid_sample.

    Args:
        mask: [B, 1, H, W] - Logits or probabilities at level resolution
        coords: [B, K, 2] - Patch coordinates (h, w) at level resolution
        patch_size: Size of patches at level resolution
        level_resolution: Resolution of the current level
        target_size: Desired spatial size for output patches

    Returns:
        mask_patches: [B, K, 1, target_size, target_size]
    """
    B, C, H, W = mask.shape
    K = coords.shape[1]
    device = mask.device

    # Compute patch centers in normalized coordinates [-1, 1]
    fh = coords[:, :, 0].float() + patch_size / 2
    fw = coords[:, :, 1].float() + patch_size / 2

    center_h = (fh / level_resolution) * 2 - 1
    center_w = (fw / level_resolution) * 2 - 1

    # Create sampling grid offsets
    half_patch = patch_size / level_resolution
    offset = torch.linspace(-half_patch, half_patch, target_size, device=device)
    grid_y, grid_x = torch.meshgrid(offset, offset, indexing='ij')
    grid_offsets = torch.stack([grid_x, grid_y], dim=-1)

    # Build grid: [B, K, target_size, target_size, 2]
    centers = torch.stack([center_w, center_h], dim=-1)
    grid = centers.view(B, K, 1, 1, 2) + grid_offsets.view(1, 1, target_size, target_size, 2)

    # Memory-efficient extraction: process per batch
    mask_patches_list = []
    for b in range(B):
        mask_b = mask[b:b+1].expand(K, -1, -1, -1)
        grid_b = grid[b]
        patches_b = F.grid_sample(
            mask_b, grid_b, mode='bilinear',
            padding_mode='border', align_corners=False
        )
        mask_patches_list.append(patches_b)

    return torch.stack(mask_patches_list, dim=0)
