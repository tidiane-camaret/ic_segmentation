"""
Test z-axis optimized 3D storage: (1, H, W) chunks for fast z-slice access.
"""

import time
from pathlib import Path
from typing import Dict

import h5py
import nibabel as nib
import numpy as np


def create_z_optimized_format(case_dir: Path, output_path: Path) -> Dict:
    """Create 3D volume with z-slice optimized chunking."""

    img_file = case_dir / "ct.nii.gz"
    modality = "ct"
    if not img_file.exists():
        img_file = case_dir / "mri.nii.gz"
        modality = "mri"

    img_nii = nib.load(str(img_file))
    img_data = img_nii.get_fdata().astype(np.float32)
    spacing = tuple(float(s) for s in nib.affines.voxel_sizes(img_nii.affine))

    D, H, W = img_data.shape
    # Z-optimized: each z-slice is one chunk
    chunks = (1, H, W)

    labels_dir = case_dir / "segmentations"
    labels_data = {}

    for label_file in labels_dir.glob("*.nii.gz"):
        label_id = label_file.stem.replace(".nii", "")
        label_nii = nib.load(str(label_file))
        label_data = label_nii.get_fdata()
        if label_data.max() == 0:
            continue
        labels_data[label_id] = label_data.astype(np.uint8)

    with h5py.File(output_path, 'w') as h5f:
        h5f.create_dataset("ct", data=img_data, compression="gzip", compression_opts=4, chunks=chunks)

        meta_grp = h5f.create_group("meta")
        meta_grp.attrs["shape"] = (D, H, W)
        meta_grp.attrs["spacing"] = spacing
        meta_grp.attrs["modality"] = modality

        masks_grp = h5f.create_group("masks")
        for label_id, mask_data in labels_data.items():
            masks_grp.create_dataset(label_id, data=mask_data, compression="gzip", compression_opts=4, chunks=chunks)

    return {
        'file_size_mb': output_path.stat().st_size / (1024 * 1024),
        'volume_shape': (D, H, W),
        'chunks': chunks,
        'num_labels': len(labels_data),
    }


def benchmark_z_only(h5_path: Path, num_samples: int = 100, step_size: int = 3) -> Dict:
    """Benchmark z-slice loading only."""

    with h5py.File(h5_path, 'r') as h5f:
        labels = list(h5f['masks'].keys())
        D, H, W = h5f['meta'].attrs['shape']
        z_indices = list(range(0, D, step_size))

        _ = h5f['ct'][0, :, :]  # warmup

        times = []
        for _ in range(num_samples):
            label_id = np.random.choice(labels)
            idx = np.random.choice(z_indices)

            start = time.perf_counter()
            ct_slice = h5f['ct'][idx, :, :]
            mask_slice = h5f[f'masks/{label_id}'][idx, :, :]
            elapsed = time.perf_counter() - start
            times.append(elapsed)

    return {
        'mean_ms': np.mean(times) * 1000,
        'p95_ms': np.percentile(times, 95) * 1000,
    }


def benchmark_2d_z_only(h5_path: Path, num_samples: int = 100) -> Dict:
    """Benchmark z-slice loading from 2D format."""

    with h5py.File(h5_path, 'r') as h5f:
        labels = list(h5f['masks'].keys())
        n_slices = h5f['ct/z'].shape[0]

        _ = h5f['ct/z'][0]  # warmup

        times = []
        for _ in range(num_samples):
            label_id = np.random.choice(labels)
            slice_idx = np.random.randint(0, n_slices)

            start = time.perf_counter()
            ct_slice = h5f['ct/z'][slice_idx]
            mask_slice = h5f[f'masks/{label_id}/z'][slice_idx]
            elapsed = time.perf_counter() - start
            times.append(elapsed)

    return {
        'mean_ms': np.mean(times) * 1000,
        'p95_ms': np.percentile(times, 95) * 1000,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input-3d", type=str, required=True)
    parser.add_argument("--input-2d", type=str, required=True)
    parser.add_argument("--output", type=str, default="/tmp/z_optimized_test")
    parser.add_argument("--num-cases", type=int, default=5)
    parser.add_argument("--num-samples", type=int, default=200)
    args = parser.parse_args()

    input_3d = Path(args.input_3d)
    input_2d = Path(args.input_2d)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    h5_files = sorted(list(input_2d.glob("*.h5")))[:args.num_cases]

    print(f"=== Z-Optimized 3D vs 2D Slices (z-axis only) ===\n")

    results = []

    for h5_file in h5_files:
        case_id = h5_file.stem
        case_dir = input_3d / case_id

        if not case_dir.exists():
            continue

        print(f"--- {case_id} ---")

        size_2d_mb = h5_file.stat().st_size / (1024 * 1024)

        # Create z-optimized 3D format
        output_3d = output_dir / f"{case_id}_z_opt.h5"
        stats_3d = create_z_optimized_format(case_dir, output_3d)

        ratio = stats_3d['file_size_mb'] / size_2d_mb
        print(f"2D: {size_2d_mb:.2f} MB | 3D z-opt: {stats_3d['file_size_mb']:.2f} MB ({ratio:.2f}x)")

        # Benchmark z-slices only
        load_2d = benchmark_2d_z_only(h5_file, args.num_samples)
        load_3d = benchmark_z_only(output_3d, args.num_samples)

        speedup = load_3d['mean_ms'] / load_2d['mean_ms']
        print(f"Z-slice load: 2D={load_2d['mean_ms']:.2f}ms | 3D={load_3d['mean_ms']:.2f}ms ({speedup:.2f}x)\n")

        results.append({
            'case_id': case_id,
            'size_2d_mb': size_2d_mb,
            'size_3d_mb': stats_3d['file_size_mb'],
            'size_ratio': ratio,
            'load_2d_ms': load_2d['mean_ms'],
            'load_3d_ms': load_3d['mean_ms'],
            'load_ratio': speedup,
        })

    # Summary
    print("=== Summary (z-slices only) ===")
    avg_size_ratio = np.mean([r['size_ratio'] for r in results])
    avg_load_ratio = np.mean([r['load_ratio'] for r in results])

    print(f"Average storage ratio (3D/2D): {avg_size_ratio:.2f}x")
    print(f"Average z-slice load ratio (3D/2D): {avg_load_ratio:.2f}x")

    if avg_load_ratio < 1.0:
        print(f"\n3D z-optimized is {1/avg_load_ratio:.1f}x FASTER for z-slices!")
    else:
        print(f"\n3D z-optimized is {avg_load_ratio:.1f}x slower for z-slices")


if __name__ == "__main__":
    main()
