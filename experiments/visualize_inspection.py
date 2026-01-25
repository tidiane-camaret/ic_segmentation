#!/usr/bin/env python
"""
Visualize patch inspection data from foreground sampling evaluation.

Usage:
    python scripts/visualize_inspection.py --case s0001 --label heart --level 3 --patch 0
"""

import argparse
from pathlib import Path
import json
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
from src.config import config


def visualize_patch(inspection_dir, case_id, label_id, level, patch_id, slice_axis='z'):
    """
    Visualize a single patch from inspection data.

    Args:
        inspection_dir: Base inspection directory
        case_id: Case ID (e.g., 's0032')
        label_id: Label ID (e.g., 'heart')
        level: Autoregressive level (1, 2, 3, ...)
        patch_id: Patch ID (0, 1, 2, ...)
        slice_axis: Axis to slice ('x', 'y', or 'z')
    """
    case_dir = Path(inspection_dir) / case_id

    if not case_dir.exists():
        print(f"Case directory not found: {case_dir}")
        return

    # Load metadata
    metadata_file = case_dir / f'{label_id}_metadata.json'
    if not metadata_file.exists():
        print(f"Metadata file not found: {metadata_file}")
        return

    with open(metadata_file) as f:
        all_metadata = json.load(f)

    # Find metadata for this level
    level_metadata = None
    for level_meta in all_metadata['levels']:
        if level_meta['level'] == level:
            level_metadata = level_meta
            break

    if level_metadata is None:
        print(f"Level {level} not found in metadata")
        return

    metadata = level_metadata

    print(f"\n{'='*60}")
    print(f"Case: {case_id}, Label: {label_id}")
    print(f"Level: {level}, Patch: {patch_id}")
    print(f"Spatial shape: {metadata['spatial_shape']}")
    print(f"ROI size: {metadata['roi_size']}")
    print(f"{'='*60}\n")

    # File prefix for flat hierarchy
    prefix = f"{label_id}_level_{level}_patch_{patch_id:04d}"

    # Load target and prediction
    target_file = case_dir / f'{prefix}_target_in.nii.gz'
    pred_file = case_dir / f'{prefix}_prediction.nii.gz'

    if not target_file.exists():
        print(f"Target file not found: {target_file}")
        return

    target = nib.load(target_file).get_fdata()
    prediction = nib.load(pred_file).get_fdata()

    # Load context coordinates if available
    coords_file = case_dir / f'{prefix}_context_coordinates.json'
    if coords_file.exists():
        with open(coords_file) as f:
            coords = json.load(f)
    else:
        coords = None

    # Count context examples
    context_files = sorted(case_dir.glob(f'{prefix}_context_*_img.nii.gz'))
    num_contexts = len(context_files)

    # Determine slice index
    axis_map = {'x': 0, 'y': 1, 'z': 2}
    axis_idx = axis_map.get(slice_axis, 2)
    slice_idx = target.shape[axis_idx] // 2

    # Create figure
    num_cols = 2 + num_contexts * 2  # target, pred, + contexts (img + mask each)
    fig, axes = plt.subplots(1, num_cols, figsize=(4 * num_cols, 4))
    if num_cols == 1:
        axes = [axes]

    col_idx = 0

    # Plot target
    target_slice = np.take(target, slice_idx, axis=axis_idx)
    axes[col_idx].imshow(target_slice, cmap='gray')
    axes[col_idx].set_title('Target Image')
    axes[col_idx].axis('off')
    col_idx += 1

    # Plot prediction
    pred_slice = np.take(prediction, slice_idx, axis=axis_idx)
    axes[col_idx].imshow(pred_slice, cmap='hot', vmin=0, vmax=1)
    axes[col_idx].set_title('Prediction')
    axes[col_idx].axis('off')
    col_idx += 1

    # Plot contexts
    for ctx_idx in range(num_contexts):
        ctx_img = nib.load(case_dir / f'{prefix}_context_{ctx_idx}_img.nii.gz').get_fdata()
        ctx_mask = nib.load(case_dir / f'{prefix}_context_{ctx_idx}_mask.nii.gz').get_fdata()

        ctx_img_slice = np.take(ctx_img, slice_idx, axis=axis_idx)
        ctx_mask_slice = np.take(ctx_mask, slice_idx, axis=axis_idx)

        # Image
        axes[col_idx].imshow(ctx_img_slice, cmap='gray')
        if coords:
            coord_str = coords.get(f'context_{ctx_idx}', {})
            title = f'Context {ctx_idx}\n@ {coord_str}'
        else:
            title = f'Context {ctx_idx} Image'
        axes[col_idx].set_title(title, fontsize=10)
        axes[col_idx].axis('off')
        col_idx += 1

        # Mask
        axes[col_idx].imshow(ctx_mask_slice, cmap='hot', vmin=0, vmax=1)
        fg_count = int((ctx_mask > 0).sum())
        axes[col_idx].set_title(f'Context {ctx_idx} Mask\n{fg_count} FG voxels', fontsize=10)
        axes[col_idx].axis('off')
        col_idx += 1

    plt.suptitle(f'{case_id}/{label_id} - Level {level}, Patch {patch_id} - {slice_axis.upper()} slice {slice_idx}',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()

    # Print statistics
    print("\nPatch Statistics:")
    print(f"  Target shape: {target.shape}")
    print(f"  Target range: [{target.min():.3f}, {target.max():.3f}]")
    print(f"  Prediction range: [{prediction.min():.3f}, {prediction.max():.3f}]")
    print(f"  Predicted foreground: {(prediction > 0.5).sum()} voxels")

    if coords:
        print("\nContext Sampling Coordinates:")
        for ctx_id, coord in coords.items():
            ctx_idx = int(ctx_id.split('_')[1])
            ctx_mask = nib.load(case_dir / f'{prefix}_context_{ctx_idx}_mask.nii.gz').get_fdata()
            fg_count = int((ctx_mask > 0).sum())
            print(f"  {ctx_id}: {coord} - {fg_count} foreground voxels")


def list_available(inspection_dir):
    """List all available inspection data with flat hierarchy."""
    inspection_path = Path(inspection_dir)

    if not inspection_path.exists():
        print(f"Inspection directory not found: {inspection_path}")
        return

    print(f"\nAvailable inspection data in {inspection_path}:\n")

    for case_dir in sorted(inspection_path.glob('*')):
        if not case_dir.is_dir():
            continue
        case_id = case_dir.name

        # Find all metadata files (one per label)
        metadata_files = sorted(case_dir.glob('*_metadata.json'))

        if not metadata_files:
            continue

        print(f"{case_id}:")

        for metadata_file in metadata_files:
            # Extract label_id from filename (e.g., "heart_metadata.json" -> "heart")
            label_id = metadata_file.stem.replace('_metadata', '')

            with open(metadata_file) as f:
                metadata = json.load(f)

            for level_meta in metadata['levels']:
                level = level_meta['level']
                num_patches = level_meta['num_patches']

                print(f"  {label_id} - Level {level}: {num_patches} patches, "
                      f"shape {level_meta['spatial_shape']}, ROI {level_meta['roi_size']}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize patch inspection data from foreground sampling"
    )
    parser.add_argument(
        '--inspection-dir',
        type=str,
        default=str(Path(config['RESULTS_DIR']) / 'totalseg_inspection'),
        help='Inspection data directory'
    )
    parser.add_argument(
        '--case',
        type=str,
        help='Case ID (e.g., s0001)'
    )
    parser.add_argument(
        '--label',
        type=str,
        help='Label ID (e.g., heart)'
    )
    parser.add_argument(
        '--level',
        type=int,
        default=3,
        help='Autoregressive level to visualize (default: 3)'
    )
    parser.add_argument(
        '--patch',
        type=int,
        default=0,
        help='Patch ID to visualize (default: 0)'
    )
    parser.add_argument(
        '--slice-axis',
        type=str,
        default='z',
        choices=['x', 'y', 'z'],
        help='Axis to slice for visualization (default: z)'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List available inspection data'
    )

    args = parser.parse_args()

    if args.list:
        list_available(args.inspection_dir)
    elif args.case and args.label:
        visualize_patch(
            args.inspection_dir,
            args.case,
            args.label,
            args.level,
            args.patch,
            args.slice_axis
        )
    else:
        print("Please provide --case and --label, or use --list to see available data")
        parser.print_help()


if __name__ == '__main__':
    main()
