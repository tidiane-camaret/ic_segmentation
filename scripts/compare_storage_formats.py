"""
Compare storage and loading times: 2D sliced vs 3D volume formats.

Current approach: Store 2D slices for z/y/x axes (every 3rd slice)
Proposed approach: Store full 3D volume once, extract slices at runtime
"""

import time
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import nibabel as nib
import numpy as np


def get_slice_format_stats(h5_path: Path) -> Dict:
    """Analyze existing 2D sliced HDF5 file."""
    file_size_mb = h5_path.stat().st_size / (1024 * 1024)

    with h5py.File(h5_path, 'r') as h5f:
        shape = tuple(h5f['meta'].attrs['shape'])

        # CT data sizes
        ct_z_shape = h5f['ct/z'].shape
        ct_y_shape = h5f['ct/y'].shape
        ct_x_shape = h5f['ct/x'].shape

        ct_total_elements = np.prod(ct_z_shape) + np.prod(ct_y_shape) + np.prod(ct_x_shape)

        # Count labels and mask sizes
        num_labels = len(h5f['masks'].keys())
        mask_total_elements = 0
        for label_id in h5f['masks'].keys():
            for axis in ['z', 'y', 'x']:
                mask_total_elements += np.prod(h5f[f'masks/{label_id}/{axis}'].shape)

    return {
        'file_size_mb': file_size_mb,
        'volume_shape': shape,
        'volume_elements': np.prod(shape),
        'ct_slice_shapes': {'z': ct_z_shape, 'y': ct_y_shape, 'x': ct_x_shape},
        'ct_total_elements': ct_total_elements,
        'num_labels': num_labels,
        'mask_total_elements': mask_total_elements,
    }


def create_3d_volume_format(
    case_dir: Path,
    output_path: Path,
    step_size: int = 3,
) -> Dict:
    """Create alternative 3D volume format and return stats."""

    # Load CT
    img_file = case_dir / "ct.nii.gz"
    modality = "ct"
    if not img_file.exists():
        img_file = case_dir / "mri.nii.gz"
        modality = "mri"

    img_nii = nib.load(str(img_file))
    img_data = img_nii.get_fdata().astype(np.float32)
    spacing = tuple(float(s) for s in nib.affines.voxel_sizes(img_nii.affine))

    D, H, W = img_data.shape

    # Generate slice indices (same as 2D format)
    z_indices = list(range(0, D, step_size))
    y_indices = list(range(0, H, step_size))
    x_indices = list(range(0, W, step_size))

    # Load all labels
    labels_dir = case_dir / "segmentations"
    labels_data = {}

    for label_file in labels_dir.glob("*.nii.gz"):
        label_id = label_file.stem.replace(".nii", "")
        label_nii = nib.load(str(label_file))
        label_data = label_nii.get_fdata()

        if label_data.max() == 0:
            continue

        labels_data[label_id] = label_data.astype(np.uint8)

    # Write 3D format
    with h5py.File(output_path, 'w') as h5f:
        # Store full 3D CT volume (compressed)
        h5f.create_dataset("ct", data=img_data, compression="gzip", compression_opts=4)

        # Store slice indices
        h5f.create_dataset("indices/z", data=z_indices)
        h5f.create_dataset("indices/y", data=y_indices)
        h5f.create_dataset("indices/x", data=x_indices)

        # Metadata
        meta_grp = h5f.create_group("meta")
        meta_grp.attrs["shape"] = (D, H, W)
        meta_grp.attrs["spacing"] = spacing
        meta_grp.attrs["modality"] = modality

        # Store full 3D masks
        masks_grp = h5f.create_group("masks")
        for label_id, mask_data in labels_data.items():
            masks_grp.create_dataset(
                label_id, data=mask_data,
                compression="gzip", compression_opts=4
            )

    file_size_mb = output_path.stat().st_size / (1024 * 1024)

    return {
        'file_size_mb': file_size_mb,
        'volume_shape': (D, H, W),
        'volume_elements': D * H * W,
        'ct_total_elements': D * H * W,
        'num_labels': len(labels_data),
        'mask_total_elements': len(labels_data) * D * H * W,
    }


def benchmark_loading_2d_format(h5_path: Path, num_samples: int = 100) -> Dict:
    """Benchmark loading random slices from 2D format."""

    with h5py.File(h5_path, 'r') as h5f:
        labels = list(h5f['masks'].keys())
        if not labels:
            return {'error': 'No labels found'}

        axes = ['z', 'y', 'x']

        # Warmup
        _ = h5f['ct/z'][0]

        times = []
        for _ in range(num_samples):
            label_id = np.random.choice(labels)
            axis = np.random.choice(axes)

            n_slices = h5f[f'ct/{axis}'].shape[0]
            slice_idx = np.random.randint(0, n_slices)

            start = time.perf_counter()
            ct_slice = h5f[f'ct/{axis}'][slice_idx]
            mask_slice = h5f[f'masks/{label_id}/{axis}'][slice_idx]
            elapsed = time.perf_counter() - start
            times.append(elapsed)

    return {
        'mean_ms': np.mean(times) * 1000,
        'std_ms': np.std(times) * 1000,
        'min_ms': np.min(times) * 1000,
        'max_ms': np.max(times) * 1000,
        'p95_ms': np.percentile(times, 95) * 1000,
    }


def benchmark_loading_3d_format(h5_path: Path, num_samples: int = 100, step_size: int = 3) -> Dict:
    """Benchmark loading slices from 3D volume format."""

    with h5py.File(h5_path, 'r') as h5f:
        labels = list(h5f['masks'].keys())
        if not labels:
            return {'error': 'No labels found'}

        D, H, W = h5f['meta'].attrs['shape']
        z_indices = list(range(0, D, step_size))
        y_indices = list(range(0, H, step_size))
        x_indices = list(range(0, W, step_size))

        # Warmup
        _ = h5f['ct'][0, :, :]

        times = []
        for _ in range(num_samples):
            label_id = np.random.choice(labels)
            axis = np.random.choice(['z', 'y', 'x'])

            if axis == 'z':
                indices = z_indices
            elif axis == 'y':
                indices = y_indices
            else:
                indices = x_indices

            idx = np.random.choice(indices)

            start = time.perf_counter()
            if axis == 'z':
                ct_slice = h5f['ct'][idx, :, :]
                mask_slice = h5f[f'masks/{label_id}'][idx, :, :]
            elif axis == 'y':
                ct_slice = h5f['ct'][:, idx, :]
                mask_slice = h5f[f'masks/{label_id}'][:, idx, :]
            else:
                ct_slice = h5f['ct'][:, :, idx]
                mask_slice = h5f[f'masks/{label_id}'][:, :, idx]
            elapsed = time.perf_counter() - start
            times.append(elapsed)

    return {
        'mean_ms': np.mean(times) * 1000,
        'std_ms': np.std(times) * 1000,
        'min_ms': np.min(times) * 1000,
        'max_ms': np.max(times) * 1000,
        'p95_ms': np.percentile(times, 95) * 1000,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input-3d", type=str, required=True, help="Path to original 3D dataset")
    parser.add_argument("--input-2d", type=str, required=True, help="Path to 2D sliced HDF5 files")
    parser.add_argument("--output", type=str, default="/tmp/3d_format_test", help="Output dir for 3D format test")
    parser.add_argument("--num-cases", type=int, default=5, help="Number of cases to test")
    parser.add_argument("--num-samples", type=int, default=200, help="Number of loading samples per benchmark")
    args = parser.parse_args()

    input_3d = Path(args.input_3d)
    input_2d = Path(args.input_2d)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find cases that exist in both formats
    h5_files = sorted(list(input_2d.glob("*.h5")))[:args.num_cases]

    print(f"=== Storage & Loading Comparison ===")
    print(f"Input 3D: {input_3d}")
    print(f"Input 2D: {input_2d}")
    print(f"Testing {len(h5_files)} cases\n")

    results = []

    for h5_file in h5_files:
        case_id = h5_file.stem
        case_dir = input_3d / case_id

        if not case_dir.exists():
            print(f"Skipping {case_id}: 3D source not found")
            continue

        print(f"--- {case_id} ---")

        # Analyze existing 2D format
        stats_2d = get_slice_format_stats(h5_file)
        print(f"2D Format: {stats_2d['file_size_mb']:.2f} MB, "
              f"shape={stats_2d['volume_shape']}, "
              f"{stats_2d['num_labels']} labels")

        # Create and analyze 3D format
        output_3d = output_dir / f"{case_id}_3d.h5"
        stats_3d = create_3d_volume_format(case_dir, output_3d)
        print(f"3D Format: {stats_3d['file_size_mb']:.2f} MB, "
              f"{stats_3d['num_labels']} labels")

        # Storage comparison
        ratio = stats_3d['file_size_mb'] / stats_2d['file_size_mb']
        print(f"Storage ratio (3D/2D): {ratio:.2f}x")

        # Benchmark loading
        print(f"Benchmarking {args.num_samples} random slice loads...")

        load_2d = benchmark_loading_2d_format(h5_file, args.num_samples)
        load_3d = benchmark_loading_3d_format(output_3d, args.num_samples)

        print(f"2D Load: mean={load_2d['mean_ms']:.2f}ms, p95={load_2d['p95_ms']:.2f}ms")
        print(f"3D Load: mean={load_3d['mean_ms']:.2f}ms, p95={load_3d['p95_ms']:.2f}ms")

        speedup = load_3d['mean_ms'] / load_2d['mean_ms']
        print(f"Load time ratio (3D/2D): {speedup:.2f}x")
        print()

        results.append({
            'case_id': case_id,
            'size_2d_mb': stats_2d['file_size_mb'],
            'size_3d_mb': stats_3d['file_size_mb'],
            'size_ratio': ratio,
            'load_2d_ms': load_2d['mean_ms'],
            'load_3d_ms': load_3d['mean_ms'],
            'load_ratio': speedup,
            'num_labels': stats_2d['num_labels'],
        })

    # Summary
    print("=== Summary ===")
    avg_size_ratio = np.mean([r['size_ratio'] for r in results])
    avg_load_ratio = np.mean([r['load_ratio'] for r in results])
    total_2d = sum(r['size_2d_mb'] for r in results)
    total_3d = sum(r['size_3d_mb'] for r in results)

    print(f"Total storage 2D: {total_2d:.1f} MB")
    print(f"Total storage 3D: {total_3d:.1f} MB")
    print(f"Average size ratio (3D/2D): {avg_size_ratio:.2f}x")
    print(f"Average load time ratio (3D/2D): {avg_load_ratio:.2f}x")

    if avg_size_ratio < 1.0:
        print(f"\n3D format saves {(1-avg_size_ratio)*100:.1f}% storage")
    else:
        print(f"\n3D format uses {(avg_size_ratio-1)*100:.1f}% more storage")

    if avg_load_ratio > 1.0:
        print(f"3D format is {avg_load_ratio:.1f}x slower to load slices")
    else:
        print(f"3D format is {1/avg_load_ratio:.1f}x faster to load slices")


if __name__ == "__main__":
    main()
