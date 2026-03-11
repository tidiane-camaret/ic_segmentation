"""
Compare storage and loading times: 2D sliced vs 3D volume with chunking.

The key insight: HDF5 with appropriate chunking can make slice access fast
by only decompressing the relevant chunks.
"""

import time
from pathlib import Path
from typing import Dict

import h5py
import nibabel as nib
import numpy as np


def create_3d_chunked_format(
    case_dir: Path,
    output_path: Path,
    chunk_strategy: str = "slices",  # "slices" or "auto"
) -> Dict:
    """Create 3D volume format with chunking optimized for slice access."""

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

    # Chunk sizes optimized for slice access
    # For z-slices: (1, H, W) - one full z-slice per chunk
    # For y-slices: (D, 1, W) - one full y-slice per chunk
    # For x-slices: (D, H, 1) - one full x-slice per chunk
    # Compromise: use small chunks in each dimension
    if chunk_strategy == "slices":
        # Chunks that work well for any axis
        # Small in each dim so any slice only needs a few chunks
        chunk_d = min(8, D)
        chunk_h = min(32, H)
        chunk_w = min(32, W)
        chunks = (chunk_d, chunk_h, chunk_w)
    else:
        chunks = True  # Let h5py decide

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

    # Write chunked 3D format
    with h5py.File(output_path, 'w') as h5f:
        # Store full 3D CT volume with chunking
        h5f.create_dataset(
            "ct", data=img_data,
            compression="gzip", compression_opts=4,
            chunks=chunks
        )

        # Metadata
        meta_grp = h5f.create_group("meta")
        meta_grp.attrs["shape"] = (D, H, W)
        meta_grp.attrs["spacing"] = spacing
        meta_grp.attrs["modality"] = modality

        # Store full 3D masks with chunking
        masks_grp = h5f.create_group("masks")
        for label_id, mask_data in labels_data.items():
            masks_grp.create_dataset(
                label_id, data=mask_data,
                compression="gzip", compression_opts=4,
                chunks=chunks
            )

    file_size_mb = output_path.stat().st_size / (1024 * 1024)

    return {
        'file_size_mb': file_size_mb,
        'volume_shape': (D, H, W),
        'chunks': chunks,
        'num_labels': len(labels_data),
    }


def benchmark_loading_3d_chunked(h5_path: Path, num_samples: int = 100, step_size: int = 3) -> Dict:
    """Benchmark loading slices from chunked 3D volume format."""

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

        times = {'z': [], 'y': [], 'x': []}

        for _ in range(num_samples):
            label_id = np.random.choice(labels)
            axis = np.random.choice(['z', 'y', 'x'])

            if axis == 'z':
                idx = np.random.choice(z_indices)
            elif axis == 'y':
                idx = np.random.choice(y_indices)
            else:
                idx = np.random.choice(x_indices)

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
            times[axis].append(elapsed)

    all_times = times['z'] + times['y'] + times['x']
    return {
        'mean_ms': np.mean(all_times) * 1000,
        'std_ms': np.std(all_times) * 1000,
        'p95_ms': np.percentile(all_times, 95) * 1000,
        'z_mean_ms': np.mean(times['z']) * 1000 if times['z'] else 0,
        'y_mean_ms': np.mean(times['y']) * 1000 if times['y'] else 0,
        'x_mean_ms': np.mean(times['x']) * 1000 if times['x'] else 0,
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

        times = {'z': [], 'y': [], 'x': []}
        for _ in range(num_samples):
            label_id = np.random.choice(labels)
            axis = np.random.choice(axes)

            n_slices = h5f[f'ct/{axis}'].shape[0]
            slice_idx = np.random.randint(0, n_slices)

            start = time.perf_counter()
            ct_slice = h5f[f'ct/{axis}'][slice_idx]
            mask_slice = h5f[f'masks/{label_id}/{axis}'][slice_idx]
            elapsed = time.perf_counter() - start
            times[axis].append(elapsed)

    all_times = times['z'] + times['y'] + times['x']
    return {
        'mean_ms': np.mean(all_times) * 1000,
        'std_ms': np.std(all_times) * 1000,
        'p95_ms': np.percentile(all_times, 95) * 1000,
        'z_mean_ms': np.mean(times['z']) * 1000 if times['z'] else 0,
        'y_mean_ms': np.mean(times['y']) * 1000 if times['y'] else 0,
        'x_mean_ms': np.mean(times['x']) * 1000 if times['x'] else 0,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input-3d", type=str, required=True)
    parser.add_argument("--input-2d", type=str, required=True)
    parser.add_argument("--output", type=str, default="/tmp/3d_chunked_test")
    parser.add_argument("--num-cases", type=int, default=5)
    parser.add_argument("--num-samples", type=int, default=200)
    args = parser.parse_args()

    input_3d = Path(args.input_3d)
    input_2d = Path(args.input_2d)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    h5_files = sorted(list(input_2d.glob("*.h5")))[:args.num_cases]

    print(f"=== Storage & Loading: 2D Slices vs 3D Chunked ===")
    print(f"Testing {len(h5_files)} cases\n")

    results = []

    for h5_file in h5_files:
        case_id = h5_file.stem
        case_dir = input_3d / case_id

        if not case_dir.exists():
            print(f"Skipping {case_id}: 3D source not found")
            continue

        print(f"--- {case_id} ---")

        # 2D format stats
        size_2d_mb = h5_file.stat().st_size / (1024 * 1024)
        print(f"2D Format: {size_2d_mb:.2f} MB")

        # Create chunked 3D format
        output_3d = output_dir / f"{case_id}_3d_chunked.h5"
        stats_3d = create_3d_chunked_format(case_dir, output_3d)
        print(f"3D Chunked: {stats_3d['file_size_mb']:.2f} MB (chunks={stats_3d['chunks']})")

        ratio = stats_3d['file_size_mb'] / size_2d_mb
        print(f"Storage ratio: {ratio:.2f}x")

        # Benchmark
        print(f"Benchmarking {args.num_samples} random slice loads...")

        load_2d = benchmark_loading_2d_format(h5_file, args.num_samples)
        load_3d = benchmark_loading_3d_chunked(output_3d, args.num_samples)

        print(f"2D Load: mean={load_2d['mean_ms']:.2f}ms (z={load_2d['z_mean_ms']:.1f}, y={load_2d['y_mean_ms']:.1f}, x={load_2d['x_mean_ms']:.1f})")
        print(f"3D Load: mean={load_3d['mean_ms']:.2f}ms (z={load_3d['z_mean_ms']:.1f}, y={load_3d['y_mean_ms']:.1f}, x={load_3d['x_mean_ms']:.1f})")

        speedup = load_3d['mean_ms'] / load_2d['mean_ms']
        print(f"Load time ratio (3D/2D): {speedup:.2f}x\n")

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
    print("=== Summary ===")
    avg_size_ratio = np.mean([r['size_ratio'] for r in results])
    avg_load_ratio = np.mean([r['load_ratio'] for r in results])
    total_2d = sum(r['size_2d_mb'] for r in results)
    total_3d = sum(r['size_3d_mb'] for r in results)

    print(f"Total storage 2D: {total_2d:.1f} MB")
    print(f"Total storage 3D chunked: {total_3d:.1f} MB ({avg_size_ratio:.2f}x)")
    print(f"Average load time ratio (3D/2D): {avg_load_ratio:.2f}x")

    if avg_size_ratio < 1.0:
        print(f"\n3D chunked format saves {(1-avg_size_ratio)*100:.1f}% storage")
    if avg_load_ratio > 1.0:
        print(f"3D chunked format is {avg_load_ratio:.1f}x slower to load")
    else:
        print(f"3D chunked format is {1/avg_load_ratio:.1f}x faster to load")


if __name__ == "__main__":
    main()
