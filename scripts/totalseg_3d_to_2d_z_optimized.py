"""
Extract z-optimized 3D volumes from TotalSegmentator for fast z-slice access.

Key features:
- Store full 3D volumes (not pre-sliced)
- Z-axis optimized chunking: (1, H, W) for instant z-slice access
- ~40% smaller than 2D sliced format
- ~9x faster z-slice loading

HDF5 Structure:
    case.h5
    ├── ct: (D, H, W) float32, chunks=(1, H, W)
    ├── masks/
    │   ├── {label_id}: (D, H, W) uint8, chunks=(1, H, W)
    │   └── ...
    └── meta/
        ├── shape: (D, H, W)
        ├── spacing: (sz, sy, sx)
        ├── modality: "ct" or "mri"
        ├── z_indices: [0, 3, 6, ...] (every n-th slice)
        └── labels/
            ├── {label_id}/
            │   ├── z_coverage: [pixels per slice]
            │   ├── bbox: ((z0,z1), (y0,y1), (x0,x1))
            │   └── volume: total pixels
            └── ...
"""

import multiprocessing as mp
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Optional, Tuple

import h5py
import hydra
import nibabel as nib
import numpy as np
from omegaconf import DictConfig


def extract_volume_for_case(
    case_dir: Path,
    step_size: int = 3,
) -> Tuple[Optional[Dict], Optional[Dict], Optional[str]]:
    """
    Extract CT volume and all label masks with metadata.

    Returns:
        ct_data: Dict with 'volume', 'shape', 'spacing', 'modality', 'z_indices'
        labels_data: Dict[label_id] -> {'volume': array, 'z_coverage': [...], 'bbox': ..., 'total_volume': int}
        error: Error message if failed, None otherwise
    """
    try:
        # Find and load CT/MRI
        img_file = case_dir / "ct.nii.gz"
        modality = "ct"
        if not img_file.exists():
            img_file = case_dir / "mri.nii.gz"
            modality = "mri"
        if not img_file.exists():
            return None, None, f"No ct.nii.gz or mri.nii.gz in {case_dir}"

        img_nii = nib.load(str(img_file))
        img_data = img_nii.get_fdata().astype(np.float32)
        spacing = tuple(float(s) for s in nib.affines.voxel_sizes(img_nii.affine))

        D, H, W = img_data.shape

        # Generate z-slice indices (every n-th slice)
        z_indices = list(range(0, D, step_size))

        ct_data = {
            "volume": img_data,
            "shape": (D, H, W),
            "spacing": spacing,
            "modality": modality,
            "z_indices": z_indices,
        }

        # Process all labels
        labels_dir = case_dir / "segmentations"
        if not labels_dir.exists():
            return None, None, f"No segmentations directory in {case_dir}"

        labels_data = {}

        for label_file in labels_dir.glob("*.nii.gz"):
            label_id = label_file.stem.replace(".nii", "")

            try:
                label_nii = nib.load(str(label_file))
                label_data = label_nii.get_fdata()

                # Skip empty labels
                if label_data.max() == 0:
                    continue

                # Compute z-slice coverage at sampled positions
                z_coverage = [int(label_data[zi, :, :].sum()) for zi in z_indices]

                # Skip if no coverage at any sampled z-slice
                if max(z_coverage) == 0:
                    continue

                # Compute bounding box
                coords = np.array(label_data.nonzero())
                bbox = (
                    (int(coords[0].min()), int(coords[0].max())),
                    (int(coords[1].min()), int(coords[1].max())),
                    (int(coords[2].min()), int(coords[2].max())),
                )

                labels_data[label_id] = {
                    "volume": label_data.astype(np.uint8),
                    "z_coverage": z_coverage,
                    "bbox": bbox,
                    "total_volume": int(coords.shape[1]),
                }

            except Exception as e:
                print(f"  Warning: Failed to process label {label_id}: {e}")
                continue

        if not labels_data:
            return None, None, f"No valid labels found in {case_dir}"

        return ct_data, labels_data, None

    except Exception as e:
        return None, None, str(e)


def process_case(args) -> Tuple[str, Dict, Optional[str]]:
    """Process a single case and save to HDF5."""
    case_dir_str, output_dir_str, step_size = args
    case_dir = Path(case_dir_str)
    output_dir = Path(output_dir_str)
    case_name = case_dir.name

    h5_path = output_dir / f"{case_name}.h5"

    # Extract data
    ct_data, labels_data, error = extract_volume_for_case(case_dir, step_size)

    if error:
        return case_name, {}, error

    try:
        D, H, W = ct_data["shape"]
        # Z-optimized chunks: each z-slice is one chunk
        chunks = (1, H, W)

        with h5py.File(h5_path, 'w') as h5f:
            # Save full 3D CT volume with z-optimized chunking
            h5f.create_dataset(
                "ct",
                data=ct_data["volume"],
                compression="gzip",
                compression_opts=4,
                chunks=chunks,
            )

            # Save metadata
            meta_grp = h5f.create_group("meta")
            meta_grp.attrs["shape"] = ct_data["shape"]
            meta_grp.attrs["spacing"] = ct_data["spacing"]
            meta_grp.attrs["modality"] = ct_data["modality"]
            meta_grp.create_dataset("z_indices", data=ct_data["z_indices"])

            # Save label metadata
            labels_meta_grp = meta_grp.create_group("labels")

            # Save full 3D masks with z-optimized chunking
            masks_grp = h5f.create_group("masks")

            for label_id, label_info in labels_data.items():
                # Save mask volume
                masks_grp.create_dataset(
                    label_id,
                    data=label_info["volume"],
                    compression="gzip",
                    compression_opts=4,
                    chunks=chunks,
                )

                # Save label metadata
                label_meta = labels_meta_grp.create_group(label_id)
                label_meta.create_dataset("z_coverage", data=label_info["z_coverage"])
                label_meta.attrs["bbox"] = label_info["bbox"]
                label_meta.attrs["volume"] = label_info["total_volume"]

        # Build stats dict for return (compatible with dataloader)
        case_stats = {
            "shape": ct_data["shape"],
            "spacing": ct_data["spacing"],
            "modality": ct_data["modality"],
            "num_slices": {"z": len(ct_data["z_indices"])},
            "labels": {},
        }

        for label_id, label_info in labels_data.items():
            case_stats["labels"][label_id] = {
                "bbox": label_info["bbox"],
                "volume": label_info["total_volume"],
                "coverage": {"z": label_info["z_coverage"]},
            }

        return case_name, case_stats, None

    except Exception as e:
        if h5_path.exists():
            h5_path.unlink()
        return case_name, {}, str(e)


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    import pickle
    from tqdm import tqdm

    # Configuration
    max_files = getattr(cfg, 'max_files_3d_to_2d', None)
    step_size = getattr(cfg, 'slice_step_size', 1)

    dataset_dir = Path(cfg.paths.base_dataset)
    output_dir = Path(str(cfg.paths.base_dataset) + "_3d_zopt")
    stats_path = output_dir / "stats.pkl"

    print("=== TotalSeg 3D to Z-Optimized ===")
    print(f"Input dir: {dataset_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Stats path: {stats_path}")
    print(f"Slice step size: {step_size}")

    if not dataset_dir.exists():
        print(f"ERROR: Dataset directory not found: {dataset_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Build task list
    tasks = [
        (str(case_dir), str(output_dir), step_size)
        for case_dir in dataset_dir.iterdir()
        if case_dir.is_dir() and not case_dir.name.endswith('.csv')
    ]

    # Randomize order
    seed = cfg.get("seed", cfg.training.get("seed", 42))
    random.seed(seed)
    random.shuffle(tasks)
    print(f"Randomized case order with seed: {seed}")

    if max_files is not None:
        tasks = tasks[:max_files]
        print(f"Limiting to max_files={max_files}")

    print(f"Total cases to process: {len(tasks)}")

    # Process in parallel
    n_workers = min(mp.cpu_count(), 20)
    print(f"Using {n_workers} workers...")

    dataset_stats = {}
    errors = []

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(process_case, task) for task in tasks}

        for future in tqdm(as_completed(futures), total=len(tasks), desc="Processing"):
            case_name, case_stats, error = future.result()

            if case_stats:
                dataset_stats[case_name] = case_stats

            if error:
                errors.append((case_name, error))

    # Summary
    print(f"\nCompleted processing for {len(dataset_stats)} cases.")

    if dataset_stats:
        all_labels = set()
        for case_data in dataset_stats.values():
            all_labels.update(case_data.get("labels", {}).keys())
        print(f"Unique labels found: {len(all_labels)}")

        # Estimate storage
        sample_case = next(iter(dataset_stats.values()))
        n_slices = sample_case["num_slices"]["z"]
        n_labels = len(sample_case.get("labels", {}))
        print(f"Sample case: {n_slices} z-slices, {n_labels} labels")

    if errors:
        print(f"\nErrors ({len(errors)}):")
        for name, err in errors[:10]:
            print(f"  {name}: {err}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")

    # Save stats
    if dataset_stats:
        print(f"\nSaving stats to {stats_path}...")
        with open(stats_path, "wb") as f:
            pickle.dump(dataset_stats, f)
        print(f"Saved stats for {len(dataset_stats)} cases.")


if __name__ == "__main__":
    main()
