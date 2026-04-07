"""
SynthMorph Synthetic Data Generation Utilities

Core functions for generating synthetic segmentation tasks on-the-fly,
based on SynthMorph (Hoffmann et al., 2022) and UniverSeg (Butoi et al., 2023).

Pipeline:
1. Base label map: smoothed noise → argmax → discrete regions
2. Deformation: random smooth displacement field → subject variation
3. Image synthesis: per-region intensity + noise + smooth texture
"""

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from typing import Optional, Tuple


def generate_base_label_map(
    shape: Tuple[int, ...],
    num_labels: int = 16,
    sigma_range: Tuple[float, float] = (5.0, 15.0),
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Generate a discrete label map with blob-like regions.

    Algorithm (SynthMorph Section II-B):
    1. Sample num_labels independent noise volumes from N(0, 1)
    2. Smooth each with Gaussian kernel (sigma sampled per-volume)
    3. Argmax across volumes → discrete label map

    Args:
        shape: Spatial dimensions (H, W) or (D, H, W)
        num_labels: Number of distinct regions
        sigma_range: Range for Gaussian smoothing sigma (controls region size)
        rng: Random generator for reproducibility

    Returns:
        Label map with values in {0, ..., num_labels-1}, dtype int16
    """
    if rng is None:
        rng = np.random.default_rng()

    noise_stack = np.zeros((num_labels, *shape), dtype=np.float32)
    for j in range(num_labels):
        raw = rng.standard_normal(shape).astype(np.float32)
        sigma = rng.uniform(*sigma_range)
        noise_stack[j] = gaussian_filter(raw, sigma=sigma)

    return np.argmax(noise_stack, axis=0).astype(np.int16)


def random_deformation_field(
    shape: Tuple[int, ...],
    sigma_def: float = 2.0,
    sigma_smooth: float = 8.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Generate a random smooth deformation field.

    Args:
        shape: Spatial dimensions (H, W) or (D, H, W)
        sigma_def: Displacement magnitude (std of initial noise)
        sigma_smooth: Smoothness of deformation (Gaussian kernel sigma)
        rng: Random generator

    Returns:
        Flow field of shape (ndim, *shape) in voxel units
    """
    if rng is None:
        rng = np.random.default_rng()

    ndim = len(shape)
    flow = rng.normal(0, sigma_def, size=(ndim, *shape)).astype(np.float32)
    for d in range(ndim):
        flow[d] = gaussian_filter(flow[d], sigma=sigma_smooth)

    return flow


def apply_deformation(
    volume: np.ndarray,
    flow: np.ndarray,
    order: int = 0,
) -> np.ndarray:
    """
    Apply deformation field to a volume using scipy map_coordinates.

    Args:
        volume: Input array of shape (*shape)
        flow: Displacement field of shape (ndim, *shape)
        order: Interpolation order (0=nearest, 1=linear)

    Returns:
        Warped volume with same shape and dtype as input
    """
    shape = volume.shape
    ndim = len(shape)

    # Build coordinate grid
    grid = np.mgrid[tuple(slice(0, s) for s in shape)].astype(np.float64)
    coords = grid + flow.astype(np.float64)

    return map_coordinates(
        volume.astype(np.float64), coords, order=order, mode='reflect'
    ).astype(volume.dtype)


def synthesize_image(
    label_map: np.ndarray,
    num_labels: int = 16,
    intensity_range: Tuple[float, float] = (0.0, 1.0),
    region_noise_range: Tuple[float, float] = (0.01, 0.05),
    global_noise_range: Tuple[float, float] = (0.0, 0.05),
    smooth_noise_sigma: float = 15.0,
    smooth_noise_amplitude_range: Tuple[float, float] = (0.0, 0.1),
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Synthesize intensity image from label map.

    Each region gets a random mean intensity with intra-region variation.
    Global noise and smooth spatial texture are added.

    Args:
        label_map: Discrete label map with values in {0, ..., num_labels-1}
        num_labels: Number of possible labels
        intensity_range: Range for per-region mean intensity
        region_noise_range: Range for per-region intensity std
        global_noise_range: Range for global Gaussian noise std
        smooth_noise_sigma: Sigma for smooth spatial texture
        smooth_noise_amplitude_range: Range for smooth noise amplitude
        rng: Random generator

    Returns:
        Intensity image in [0, 1], dtype float32
    """
    if rng is None:
        rng = np.random.default_rng()

    shape = label_map.shape
    image = np.zeros(shape, dtype=np.float32)

    # Per-region intensity (GMM-like)
    for l in range(num_labels):
        mask = (label_map == l)
        if not mask.any():
            continue
        mu = rng.uniform(*intensity_range)
        sigma = rng.uniform(*region_noise_range)
        image[mask] = rng.normal(mu, sigma, size=mask.sum()).astype(np.float32)

    # Global Gaussian noise
    noise_std = rng.uniform(*global_noise_range)
    if noise_std > 0:
        image += rng.normal(0, noise_std, size=shape).astype(np.float32)

    # Smooth spatial texture (Perlin-like)
    smooth_noise = rng.standard_normal(shape).astype(np.float32)
    smooth_noise = gaussian_filter(smooth_noise, sigma=smooth_noise_sigma)
    sn_range = smooth_noise.max() - smooth_noise.min()
    if sn_range > 0:
        smooth_noise = 2.0 * (smooth_noise - smooth_noise.min()) / sn_range - 1.0
    amplitude = rng.uniform(*smooth_noise_amplitude_range)
    image += amplitude * smooth_noise

    return np.clip(image, 0.0, 1.0)


def generate_subject(
    base_label: np.ndarray,
    num_labels: int = 16,
    sigma_def: float = 2.0,
    sigma_smooth: float = 8.0,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a single subject from a base label map.

    Applies random deformation and synthesizes intensity image.

    Args:
        base_label: Base label map (shared anatomy)
        num_labels: Number of labels for intensity synthesis
        sigma_def: Deformation magnitude
        sigma_smooth: Deformation smoothness
        rng: Random generator

    Returns:
        (image, label_map) tuple, both same spatial shape
    """
    if rng is None:
        rng = np.random.default_rng()

    # Apply random deformation
    flow = random_deformation_field(
        base_label.shape, sigma_def=sigma_def, sigma_smooth=sigma_smooth, rng=rng
    )
    subject_label = apply_deformation(base_label, flow, order=0)

    # Synthesize intensity image
    image = synthesize_image(subject_label, num_labels=num_labels, rng=rng)

    return image, subject_label
