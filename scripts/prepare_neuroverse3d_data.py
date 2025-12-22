"""
Data preparation script for Neuroverse3D training.

This script helps convert your medical imaging data into the nnUNet format
required by Neuroverse3D.

Expected input format:
    - NIfTI files (.nii.gz) with images and corresponding segmentation masks

Expected output format (nnUNet):
    data_dir/
        imagesTr/
            case_001_0000.nii.gz
            case_002_0000.nii.gz
            ...
        labelsTr/
            case_001.nii.gz
            case_002.nii.gz
            ...

Usage:
    python scripts/prepare_neuroverse3d_data.py \
        --input-dir /path/to/raw/data \
        --output-dir data/neuroverse3d \
        --pattern "*.nii.gz"
"""

import argparse
import os
import shutil
from pathlib import Path
from typing import List, Tuple

import nibabel as nib
import numpy as np


def find_image_mask_pairs(
    input_dir: Path,
    image_pattern: str = "*_img.nii.gz",
    mask_pattern: str = "*_gt.nii.gz"
) -> List[Tuple[Path, Path]]:
    """
    Find pairs of images and masks in the input directory.

    Args:
        input_dir: Input directory containing images and masks
        image_pattern: Glob pattern for image files
        mask_pattern: Glob pattern for mask files

    Returns:
        List of (image_path, mask_path) tuples
    """
    image_files = sorted(list(input_dir.glob(image_pattern)))
    pairs = []

    for img_path in image_files:
        # Derive mask path from image path
        mask_path = img_path.parent / img_path.name.replace('_img.nii.gz', '_gt.nii.gz')

        if not mask_path.exists():
            print(f"Warning: No mask found for {img_path.name}")
            continue

        pairs.append((img_path, mask_path))

    return pairs


def convert_to_nnunet_format(
    pairs: List[Tuple[Path, Path]],
    output_dir: Path,
    copy: bool = True,
    validate: bool = True,
):
    """
    Convert image-mask pairs to nnUNet format.

    Args:
        pairs: List of (image_path, mask_path) tuples
        output_dir: Output directory for nnUNet-formatted data
        copy: Whether to copy files (True) or create symlinks (False)
        validate: Whether to validate the data
    """
    # Create output directories
    images_dir = output_dir / 'imagesTr'
    labels_dir = output_dir / 'labelsTr'
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nConverting {len(pairs)} cases to nnUNet format...")
    print(f"Output directory: {output_dir}")

    for idx, (img_path, mask_path) in enumerate(pairs, start=1):
        # Create nnUNet-style filenames
        case_id = f"{idx:03d}"
        img_out = images_dir / f"case_{case_id}_0000.nii.gz"
        mask_out = labels_dir / f"case_{case_id}.nii.gz"

        # Copy or symlink files
        if copy:
            shutil.copy2(img_path, img_out)
            shutil.copy2(mask_path, mask_out)
        else:
            img_out.symlink_to(img_path.absolute())
            mask_out.symlink_to(mask_path.absolute())

        # Validate if requested
        if validate:
            try:
                img_nii = nib.load(str(img_out))
                mask_nii = nib.load(str(mask_out))

                img_shape = img_nii.shape
                mask_shape = mask_nii.shape

                if img_shape != mask_shape:
                    print(f"Warning: Shape mismatch for case_{case_id}: "
                          f"image {img_shape} vs mask {mask_shape}")

                # Check label values
                mask_data = mask_nii.get_fdata()
                unique_labels = np.unique(mask_data)
                if len(unique_labels) > 10:
                    print(f"Warning: case_{case_id} has {len(unique_labels)} unique labels: {unique_labels}")

            except Exception as e:
                print(f"Error validating case_{case_id}: {e}")

        if idx % 10 == 0:
            print(f"Processed {idx}/{len(pairs)} cases...")

    print(f"\nConversion complete!")
    print(f"Images: {images_dir}")
    print(f"Labels: {labels_dir}")


def create_dataset_json(
    output_dir: Path,
    dataset_name: str,
    modality: str = "MRI",
    labels: dict = None,
):
    """
    Create dataset.json file for nnUNet format.

    Args:
        output_dir: Output directory
        dataset_name: Name of the dataset
        modality: Imaging modality
        labels: Dictionary of label indices to names
    """
    import json

    if labels is None:
        labels = {
            "0": "background",
            "1": "foreground",
        }

    dataset_json = {
        "name": dataset_name,
        "description": f"{dataset_name} dataset for Neuroverse3D",
        "tensorImageSize": "3D",
        "modality": {
            "0": modality
        },
        "labels": labels,
        "numTraining": len(list((output_dir / 'imagesTr').glob('*.nii.gz'))),
        "numTest": 0,
        "training": [
            {
                "image": f"./imagesTr/{f.name}",
                "label": f"./labelsTr/{f.name.replace('_0000.nii.gz', '.nii.gz')}"
            }
            for f in sorted((output_dir / 'imagesTr').glob('*.nii.gz'))
        ]
    }

    json_path = output_dir / 'dataset.json'
    with open(json_path, 'w') as f:
        json.dump(dataset_json, f, indent=2)

    print(f"\nCreated dataset.json: {json_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Prepare data for Neuroverse3D training in nnUNet format'
    )

    parser.add_argument(
        '--input-dir',
        type=str,
        required=True,
        help='Input directory containing images and masks'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Output directory for nnUNet-formatted data'
    )
    parser.add_argument(
        '--image-pattern',
        type=str,
        default='*_img.nii.gz',
        help='Glob pattern for image files'
    )
    parser.add_argument(
        '--mask-pattern',
        type=str,
        default='*_gt.nii.gz',
        help='Glob pattern for mask files'
    )
    parser.add_argument(
        '--dataset-name',
        type=str,
        default='CustomDataset',
        help='Name of the dataset'
    )
    parser.add_argument(
        '--modality',
        type=str,
        default='MRI',
        help='Imaging modality (MRI, CT, etc.)'
    )
    parser.add_argument(
        '--copy',
        action='store_true',
        default=True,
        help='Copy files instead of creating symlinks'
    )
    parser.add_argument(
        '--symlink',
        action='store_true',
        help='Create symlinks instead of copying files'
    )
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='Skip validation of converted data'
    )

    args = parser.parse_args()

    # Convert paths
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")

    # Find image-mask pairs
    print(f"Searching for image-mask pairs in: {input_dir}")
    pairs = find_image_mask_pairs(
        input_dir,
        image_pattern=args.image_pattern,
        mask_pattern=args.mask_pattern
    )

    if not pairs:
        raise ValueError(f"No image-mask pairs found in {input_dir}")

    print(f"Found {len(pairs)} image-mask pairs")

    # Convert to nnUNet format
    convert_to_nnunet_format(
        pairs,
        output_dir,
        copy=not args.symlink,
        validate=not args.no_validate
    )

    # Create dataset.json
    create_dataset_json(
        output_dir,
        dataset_name=args.dataset_name,
        modality=args.modality
    )

    print("\n" + "="*60)
    print("Data preparation complete!")
    print("="*60)
    print(f"\nYou can now train with:")
    print(f"python scripts/train_neuroverse3d.py --data-dir {output_dir}")


if __name__ == '__main__':
    main()
