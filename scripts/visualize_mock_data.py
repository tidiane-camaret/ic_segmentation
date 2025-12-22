"""
Visualize mock data to verify it was created correctly.

This script creates simple text-based visualizations of the 3D volumes.
"""

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np


def print_slice_ascii(slice_2d, width=60):
    """Print a 2D slice as ASCII art."""
    # Normalize to 0-9 range
    slice_normalized = slice_2d / (slice_2d.max() + 1e-8)
    slice_quantized = (slice_normalized * 9).astype(int)

    # Resize to fit terminal width
    h, w = slice_2d.shape
    if w > width:
        # Simple downsampling
        step = w // width
        slice_quantized = slice_quantized[:, ::step]

    # Characters for different intensities
    chars = ' .:-=+*#%@'

    for row in slice_quantized:
        print(''.join(chars[val] for val in row))


def visualize_case(image_path, mask_path):
    """Visualize a single case."""
    # Load data
    img_nii = nib.load(str(image_path))
    mask_nii = nib.load(str(mask_path))

    img_data = img_nii.get_fdata()
    mask_data = mask_nii.get_fdata()

    print(f"\nCase: {image_path.stem}")
    print(f"{'='*60}")
    print(f"Image shape: {img_data.shape}")
    print(f"Mask shape: {mask_data.shape}")
    print(f"Image range: [{img_data.min():.2f}, {img_data.max():.2f}]")
    print(f"Mask values: {np.unique(mask_data)}")
    print(f"Foreground voxels: {(mask_data > 0).sum()} / {mask_data.size} "
          f"({(mask_data > 0).mean()*100:.1f}%)")

    # Show middle slices
    mid_z = img_data.shape[2] // 2
    mid_y = img_data.shape[1] // 2
    mid_x = img_data.shape[0] // 2

    print(f"\nAxial slice (z={mid_z}):")
    print("Image:")
    print_slice_ascii(img_data[:, :, mid_z], width=50)
    print("\nMask:")
    print_slice_ascii(mask_data[:, :, mid_z] * 1000, width=50)  # Scale mask for visibility


def main():
    parser = argparse.ArgumentParser(description='Visualize mock 3D medical data')
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/mock_neuroverse3d',
        help='Directory containing nnUNet-formatted data'
    )
    parser.add_argument(
        '--num-cases',
        type=int,
        default=3,
        help='Number of cases to visualize'
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    images_dir = data_dir / 'imagesTr'
    labels_dir = data_dir / 'labelsTr'

    if not images_dir.exists():
        print(f"Error: Directory not found: {images_dir}")
        return

    # Get all image files
    image_files = sorted(list(images_dir.glob('case_*_0000.nii.gz')))[:args.num_cases]

    print("="*60)
    print("Mock Data Visualization")
    print("="*60)

    for img_path in image_files:
        # Extract case ID from filename like "case_001_0000.nii.gz" -> "case_001"
        case_id = img_path.name.replace('_0000.nii.gz', '')
        mask_path = labels_dir / f"{case_id}.nii.gz"

        if not mask_path.exists():
            print(f"Warning: Mask not found for {case_id}")
            continue

        visualize_case(img_path, mask_path)

    print("\n" + "="*60)
    print("Visualization complete!")
    print("="*60)


if __name__ == '__main__':
    main()
