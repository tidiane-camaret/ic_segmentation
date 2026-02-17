import pickle
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import albumentations as A
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import cv2
from data.label_ids_totalseg import get_label_ids

def create_augmentation_transforms(
    img_size: int = 512,
    rotation_limit: float = 30.0,
    scale_limit: float = 0.2,
    elastic_alpha: float = 120.0,
    elastic_sigma: float = 6.0,
    brightness_limit: float = 0.2,
    contrast_limit: float = 0.2,
    gamma_limit: Tuple[float, float] = (80, 120),
    noise_std_range: Tuple[float, float] = (0.02, 0.05),
) -> Tuple[A.Compose, A.Compose]:
    
    spatial_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.RandomRotate90(p=0.5),
        
        A.OneOf([
            A.Affine(
                scale=(1.0 - scale_limit, 1.0 + scale_limit),
                rotate=(-rotation_limit, rotation_limit),
                shear=(-10, 10),
                border_mode=0,
                fill=0,
                fill_mask=0,
                p=0.8,
            ),
            A.GridDistortion(
                num_steps=5, 
                distort_limit=0.3, 
                border_mode=0, 
                p=0.2
            ),
        ], p=1.0),

        # FIX: Removed 'alpha_affine' (use A.Affine separately if needed)
        A.ElasticTransform(
            alpha=elastic_alpha,
            sigma=elastic_sigma,
            border_mode=0,
            fill=0,
            fill_mask=0,
            p=0.3,
        ),
    ])

    intensity_transform = A.Compose([
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=brightness_limit,
                contrast_limit=contrast_limit,
                p=0.7,
            ),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.3),
        ], p=1.0),

        A.RandomGamma(gamma_limit=gamma_limit, p=0.3),
        
        A.OneOf([
            A.GaussNoise(std_range=noise_std_range, p=0.5),
            A.GaussianBlur(blur_limit=(3, 5), p=0.5),
        ], p=0.4),

        # FIX: Updated CoarseDropout parameters
        # max_holes -> num_holes_range
        # max_height/width -> hole_height_range / hole_width_range
        A.CoarseDropout(
            num_holes_range=(1, 6), 
            hole_height_range=(img_size//20, img_size//10), 
            hole_width_range=(img_size//20, img_size//10), 
            fill_value=0, 
            p=0.3
        ),
    ])

    return spatial_transform, intensity_transform


def carve_mix_2d(
    target_img: np.ndarray,
    target_mask: np.ndarray,
    donor_img: np.ndarray,
    donor_mask: np.ndarray,
    margin_range: Tuple[float, float] = (0.1, 0.5),
    harmonize: bool = True,
    harmonize_sigma: float = 5.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Carve foreground ROI from donor and paste into target image.

    Args:
        target_img: [H, W] float32 in [0, 1]
        target_mask: [H, W] float32 binary
        donor_img: [H, W] float32 in [0, 1]
        donor_mask: [H, W] float32 binary
        margin_range: Range of margin fraction to expand bbox
        harmonize: Match donor intensity stats to target
        harmonize_sigma: Sigma for Gaussian blur on blending mask
    """
    # Find donor foreground bounding box
    fg_coords = np.argwhere(donor_mask > 0.5)
    if len(fg_coords) == 0:
        return target_img.copy(), target_mask.copy()

    rmin, cmin = fg_coords.min(axis=0)
    rmax, cmax = fg_coords.max(axis=0)
    bbox_h = rmax - rmin + 1
    bbox_w = cmax - cmin + 1
    H, W = target_img.shape

    # Expand bbox by random margin
    margin_frac = random.uniform(*margin_range)
    margin_px = int(margin_frac * max(bbox_h, bbox_w))
    rmin = max(0, rmin - margin_px)
    rmax = min(H - 1, rmax + margin_px)
    cmin = max(0, cmin - margin_px)
    cmax = min(W - 1, cmax + margin_px)

    # Extract ROI from donor
    donor_roi_img = donor_img[rmin:rmax + 1, cmin:cmax + 1].copy()
    donor_roi_mask = donor_mask[rmin:rmax + 1, cmin:cmax + 1].copy()

    # Intensity harmonization: match donor ROI stats to target ROI stats
    if harmonize:
        target_fg = target_img[target_mask > 0.5] if target_mask.max() > 0.5 else target_img.ravel()
        donor_fg = donor_roi_img[donor_roi_mask > 0.5] if donor_roi_mask.max() > 0.5 else donor_roi_img.ravel()

        t_mean, t_std = target_fg.mean(), max(target_fg.std(), 1e-6)
        d_mean, d_std = donor_fg.mean(), max(donor_fg.std(), 1e-6)

        donor_roi_img = (donor_roi_img - d_mean) * (t_std / d_std) + t_mean
        donor_roi_img = np.clip(donor_roi_img, 0.0, 1.0)

    # Soft blending mask from donor foreground within ROI
    blend_mask = (donor_roi_mask > 0.5).astype(np.float32)
    ksize = int(harmonize_sigma * 6) | 1  # ensure odd
    blend_mask = cv2.GaussianBlur(blend_mask, (ksize, ksize), harmonize_sigma)

    # Composite image (soft blend)
    mixed_img = target_img.copy()
    roi_slice = (slice(rmin, rmax + 1), slice(cmin, cmax + 1))
    mixed_img[roi_slice] = mixed_img[roi_slice] * (1 - blend_mask) + donor_roi_img * blend_mask

    # Composite mask (hard replacement)
    mixed_mask = target_mask.copy()
    mixed_mask[roi_slice] = np.where(donor_roi_mask > 0.5, donor_roi_mask, mixed_mask[roi_slice])

    return mixed_img, mixed_mask


def foreground_random_crop(
    img: np.ndarray,
    mask: np.ndarray,
    min_crop_frac: float = 0.5,
    max_crop_frac: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Random crop guaranteed to contain the full foreground region.

    Selects a random sub-image whose size is a random fraction of the original,
    positioned so the entire foreground bounding box remains inside.
    """
    fg = np.argwhere(mask > 0.5)
    if len(fg) == 0:
        return img, mask

    rmin, cmin = fg.min(axis=0)
    rmax, cmax = fg.max(axis=0)
    fg_h, fg_w = rmax - rmin + 1, cmax - cmin + 1
    H, W = img.shape

    # Crop size: random fraction of image, at least as big as foreground + margin
    crop_frac = random.uniform(min_crop_frac, max_crop_frac)
    crop_h = min(H, max(int(H * crop_frac), fg_h + 4))
    crop_w = min(W, max(int(W * crop_frac), fg_w + 4))

    # Random top-left that keeps foreground fully inside
    r_lo = max(0, rmax - crop_h + 1)
    r_hi = min(rmin, H - crop_h)
    c_lo = max(0, cmax - crop_w + 1)
    c_hi = min(cmin, W - crop_w)

    if r_lo > r_hi or c_lo > c_hi:
        return img, mask  # cannot fit, return unchanged

    r0 = random.randint(r_lo, r_hi)
    c0 = random.randint(c_lo, c_hi)
    return (
        img[r0:r0 + crop_h, c0:c0 + crop_w].copy(),
        mask[r0:r0 + crop_h, c0:c0 + crop_w].copy(),
    )


def perturb_mask(
    mask: np.ndarray,
    max_kernel: int = 5,
) -> np.ndarray:
    """Randomly dilate or erode mask to simulate annotation variability.

    Reverts to original if erosion eliminates all foreground.
    """
    if mask.max() < 0.5:
        return mask

    ksize = random.randint(2, max(2, max_kernel))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    binary = (mask > 0.5).astype(np.uint8)

    if random.random() < 0.5:
        result = cv2.dilate(binary, kernel, iterations=1)
    else:
        result = cv2.erode(binary, kernel, iterations=1)
        if result.max() == 0:
            return mask  # erosion killed foreground, keep original

    return result.astype(np.float32)


def degrade_resolution(
    img: np.ndarray,
    min_scale: float = 0.25,
    max_scale: float = 0.75,
) -> np.ndarray:
    """Downsample then upsample to simulate low-resolution acquisition."""
    H, W = img.shape
    scale = random.uniform(min_scale, max_scale)
    small_h = max(4, int(H * scale))
    small_w = max(4, int(W * scale))
    small = cv2.resize(img, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR)


def random_intensity_shift(
    img: np.ndarray,
    brightness_shift: float = 0.15,
    contrast_scale: float = 0.15,
    gamma_range: Tuple[float, float] = (0.7, 1.5),
) -> np.ndarray:
    """Random intensity transform for cross-image appearance diversity.

    DEPRECATED: Use the unified augmentation pipeline with intensity transforms
    in albumentations instead. This function is kept for backwards compatibility.
    """
    import warnings
    warnings.warn(
        "random_intensity_shift is deprecated. Use the unified augmentation pipeline "
        "with intensity.asymmetric=True instead.",
        DeprecationWarning,
        stacklevel=2
    )
    b = random.uniform(-brightness_shift, brightness_shift)
    c = random.uniform(1 - contrast_scale, 1 + contrast_scale)
    g = random.uniform(*gamma_range)
    img = np.clip(img * c + b, 0, 1)
    img = np.power(img, g)
    return img.astype(np.float32)


def cut_mix_2d(
    target_img: np.ndarray,
    target_mask: np.ndarray,
    donor_img: np.ndarray,
    donor_mask: np.ndarray,
    min_ratio: float = 0.3,
    max_ratio: float = 0.7,
) -> Tuple[np.ndarray, np.ndarray]:
    """Simple CutMix: paste a random rectangular region from donor into target.

    Args:
        target_img: [H, W] float32 in [0, 1]
        target_mask: [H, W] float32 binary
        donor_img: [H, W] float32 in [0, 1]
        donor_mask: [H, W] float32 binary
        min_ratio: Minimum ratio of cut region (0-1)
        max_ratio: Maximum ratio of cut region (0-1)

    Returns:
        mixed_img: [H, W] image with rectangular region from donor
        mixed_mask: [H, W] mask with rectangular region from donor
    """
    H, W = target_img.shape

    # Random cut size
    cut_ratio_h = random.uniform(min_ratio, max_ratio)
    cut_ratio_w = random.uniform(min_ratio, max_ratio)
    cut_h = int(H * cut_ratio_h)
    cut_w = int(W * cut_ratio_w)

    # Random position in target
    r0 = random.randint(0, H - cut_h)
    c0 = random.randint(0, W - cut_w)

    # Random position in donor (for source region)
    donor_H, donor_W = donor_img.shape
    dr0 = random.randint(0, max(0, donor_H - cut_h))
    dc0 = random.randint(0, max(0, donor_W - cut_w))

    # Handle size mismatch
    actual_h = min(cut_h, donor_H - dr0, H - r0)
    actual_w = min(cut_w, donor_W - dc0, W - c0)

    # Paste donor region into target
    mixed_img = target_img.copy()
    mixed_mask = target_mask.copy()
    mixed_img[r0:r0+actual_h, c0:c0+actual_w] = donor_img[dr0:dr0+actual_h, dc0:dc0+actual_w]
    mixed_mask[r0:r0+actual_h, c0:c0+actual_w] = donor_mask[dr0:dr0+actual_h, dc0:dc0+actual_w]

    return mixed_img, mixed_mask


def apply_task_level_augmentation(
    images: List[np.ndarray],
    masks: List[np.ndarray],
    flip_horizontal: bool = True,
    flip_vertical: bool = True,
    rotate_90: bool = True,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Apply the same random transform to ALL images/masks (UniverSeg-style).

    This ensures spatial consistency across target and context images,
    simulating different orientations of the same anatomical structure.

    Args:
        images: List of [H, W] images
        masks: List of [H, W] masks
        flip_horizontal: Allow horizontal flip
        flip_vertical: Allow vertical flip
        rotate_90: Allow 90/180/270 degree rotations

    Returns:
        Transformed images and masks (same transform applied to all)
    """
    if not images:
        return images, masks

    # Decide on transforms (same for all images)
    do_hflip = flip_horizontal and random.random() < 0.5
    do_vflip = flip_vertical and random.random() < 0.5
    rot_k = random.choice([0, 1, 2, 3]) if rotate_90 else 0

    transformed_images = []
    transformed_masks = []

    for img, mask in zip(images, masks):
        # Apply horizontal flip
        if do_hflip:
            img = np.fliplr(img).copy()
            mask = np.fliplr(mask).copy()

        # Apply vertical flip
        if do_vflip:
            img = np.flipud(img).copy()
            mask = np.flipud(mask).copy()

        # Apply rotation (k * 90 degrees)
        if rot_k > 0:
            img = np.rot90(img, k=rot_k).copy()
            mask = np.rot90(mask, k=rot_k).copy()

        transformed_images.append(img)
        transformed_masks.append(mask)

    return transformed_images, transformed_masks


def create_spatial_only_transform(
    rotation_limit: float = 15.0,
    scale_limit: float = 0.1,
    elastic_alpha: float = 50.0,
    elastic_sigma: float = 5.0,
) -> A.Compose:
    """Create spatial-only augmentation transform (no intensity changes)."""
    return A.Compose([
        A.OneOf([
            A.Affine(
                scale=(1.0 - scale_limit, 1.0 + scale_limit),
                rotate=(-rotation_limit, rotation_limit),
                shear=(-10, 10),
                border_mode=0,
                fill=0,
                fill_mask=0,
                p=0.8,
            ),
            A.GridDistortion(
                num_steps=5,
                distort_limit=0.3,
                border_mode=0,
                p=0.2
            ),
        ], p=0.8),
        A.ElasticTransform(
            alpha=elastic_alpha,
            sigma=elastic_sigma,
            border_mode=0,
            fill=0,
            fill_mask=0,
            p=0.3,
        ),
    ])


def create_intensity_only_transform(
    brightness_limit: float = 0.1,
    contrast_limit: float = 0.1,
    gamma_limit: Tuple[float, float] = (80, 120),
    noise_std_range: Tuple[float, float] = (0.02, 0.05),
    img_size: int = 512,
) -> A.Compose:
    """Create intensity-only augmentation transform (no spatial changes)."""
    return A.Compose([
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=brightness_limit,
                contrast_limit=contrast_limit,
                p=0.7,
            ),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.3),
        ], p=0.8),
        A.RandomGamma(gamma_limit=gamma_limit, p=0.3),
        A.OneOf([
            A.GaussNoise(std_range=noise_std_range, p=0.5),
            A.GaussianBlur(blur_limit=(3, 5), p=0.5),
        ], p=0.4),
        A.CoarseDropout(
            num_holes_range=(1, 6),
            hole_height_range=(img_size//20, img_size//10),
            hole_width_range=(img_size//20, img_size//10),
            fill_value=0,
            p=0.2
        ),
    ])