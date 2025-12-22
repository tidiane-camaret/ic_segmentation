"""
Create mock 3D medical imaging data for testing the Neuroverse3D pipeline.

This script generates synthetic NIfTI files with simple geometric patterns
to test the training pipeline without requiring real medical data.
"""

import argparse
import os
from pathlib import Path

import nibabel as nib
import numpy as np


def create_synthetic_volume(size=(64, 64, 64), pattern='sphere'):
    """
    Create a synthetic 3D volume with a simple pattern.

    Args:
        size: Tuple of (H, W, D) for volume dimensions
        pattern: Type of pattern ('sphere', 'cube', 'cylinder')

    Returns:
        Tuple of (image, mask) as numpy arrays
    """
    H, W, D = size
    image = np.zeros(size, dtype=np.float32)
    mask = np.zeros(size, dtype=np.uint8)

    # Create coordinate grids
    z, y, x = np.meshgrid(
        np.arange(H),
        np.arange(W),
        np.arange(D),
        indexing='ij'
    )

    # Center coordinates
    center_z, center_y, center_x = H // 2, W // 2, D // 2

    if pattern == 'sphere':
        # Create a sphere
        radius = min(H, W, D) // 4
        distance = np.sqrt(
            (z - center_z)**2 +
            (y - center_y)**2 +
            (x - center_x)**2
        )
        sphere_mask = distance <= radius

        # Image: gradient intensity
        image = np.maximum(0, radius - distance) / radius
        # Add some noise
        image += np.random.normal(0, 0.1, size)
        image = np.clip(image, 0, 1)

        # Mask: binary sphere
        mask[sphere_mask] = 1

    elif pattern == 'cube':
        # Create a cube
        size_cube = min(H, W, D) // 3
        z_range = (center_z - size_cube//2, center_z + size_cube//2)
        y_range = (center_y - size_cube//2, center_y + size_cube//2)
        x_range = (center_x - size_cube//2, center_x + size_cube//2)

        cube_mask = (
            (z >= z_range[0]) & (z < z_range[1]) &
            (y >= y_range[0]) & (y < y_range[1]) &
            (x >= x_range[0]) & (x < x_range[1])
        )

        # Image: constant intensity in cube with noise
        image = np.random.normal(0.3, 0.1, size)
        image[cube_mask] = np.random.normal(0.8, 0.1, np.sum(cube_mask))
        image = np.clip(image, 0, 1)

        # Mask: binary cube
        mask[cube_mask] = 1

    elif pattern == 'cylinder':
        # Create a cylinder along z-axis
        radius = min(H, W) // 4
        distance_xy = np.sqrt(
            (y - center_y)**2 +
            (x - center_x)**2
        )
        cylinder_mask = distance_xy <= radius

        # Image: gradient from center
        image = np.maximum(0, radius - distance_xy) / radius
        image += np.random.normal(0, 0.1, size)
        image = np.clip(image, 0, 1)

        # Mask: binary cylinder
        mask[cylinder_mask] = 1

    # Normalize image to typical medical imaging range
    image = image * 1000  # Scale to typical MRI intensity range

    return image, mask


def create_mock_dataset(output_dir, num_cases=5, size=(64, 64, 64)):
    """
    Create a mock dataset with multiple cases.

    Args:
        output_dir: Directory to save the mock data
        num_cases: Number of cases to generate
        size: Size of each volume
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    patterns = ['sphere', 'cube', 'cylinder']

    print(f"Creating {num_cases} mock cases in {output_dir}...")

    for i in range(num_cases):
        # Vary the pattern
        pattern = patterns[i % len(patterns)]

        # Add some variation in size
        var_size = tuple(int(s * np.random.uniform(0.9, 1.1)) for s in size)

        # Create synthetic data
        image, mask = create_synthetic_volume(size=var_size, pattern=pattern)

        # Create affine matrix (identity for simplicity)
        affine = np.eye(4)

        # Create NIfTI images
        image_nii = nib.Nifti1Image(image, affine=affine)
        mask_nii = nib.Nifti1Image(mask, affine=affine)

        # Save files
        case_id = f"mock_case_{i+1:03d}"
        image_path = output_dir / f"{case_id}_img.nii.gz"
        mask_path = output_dir / f"{case_id}_gt.nii.gz"

        nib.save(image_nii, str(image_path))
        nib.save(mask_nii, str(mask_path))

        print(f"  Created {case_id} ({pattern}, shape={var_size})")

    print(f"\nMock dataset created successfully!")
    print(f"Total files: {num_cases * 2} ({num_cases} images + {num_cases} masks)")


def main():
    parser = argparse.ArgumentParser(
        description='Create mock 3D medical imaging data for testing'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/mock_raw',
        help='Directory to save mock data'
    )
    parser.add_argument(
        '--num-cases',
        type=int,
        default=6,
        help='Number of mock cases to create'
    )
    parser.add_argument(
        '--size',
        type=int,
        nargs=3,
        default=[64, 64, 64],
        help='Volume size (H W D)'
    )

    args = parser.parse_args()

    create_mock_dataset(
        output_dir=args.output_dir,
        num_cases=args.num_cases,
        size=tuple(args.size)
    )


if __name__ == '__main__':
    main()
