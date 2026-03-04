"""
Extract 2D slices from 3D TotalSegmentator volumes with shared CT slices.

Key differences from previous approaches:
- CT slices stored ONCE per case (not duplicated per label)
- Every n-th slice across full volume (not bounding box)
- Masks stored as uint8 (not float32)
- Coverage metadata stored for dataloader filtering

HDF5 Structure:
    case.h5
    ├── ct/
    │   ├── z: (N_z, H, W) float32
    │   ├── y: (N_y, D, W) float32
    │   ├── x: (N_x, D, H) float32
    │   └── indices: {z: [...], y: [...], x: [...]}
    ├── masks/
    │   ├── {label_id}/
    │   │   ├── z: (N_z, H, W) uint8
    │   │   ├── y: (N_y, D, W) uint8
    │   │   ├── x: (N_x, D, H) uint8
    │   │   └── coverage: {z: [...], y: [...], x: [...]}
    │   └── ...
    └── meta/
        ├── shape: (D, H, W)
        ├── spacing: (sz, sy, sx)
        └── modality: "ct" or "mri"
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


def extract_slices_for_case(
    case_dir: Path,
    step_size: int = 3,
) -> Tuple[Optional[Dict], Optional[Dict], Optional[str]]:
    """
    Extract every n-th slice from CT and all label masks.

    Returns:
        ct_data: Dict with 'z', 'y', 'x' slices and 'indices'
        labels_data: Dict[label_id] -> {axis: mask_slices, coverage: {axis: [...]}}
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

        # Generate slice indices (every n-th slice)
        z_indices = list(range(0, D, step_size))
        y_indices = list(range(0, H, step_size))
        x_indices = list(range(0, W, step_size))

        # Extract CT slices
        ct_data = {
            "z": img_data[z_indices, :, :],  # (N_z, H, W)
            "y": np.moveaxis(img_data[:, y_indices, :], 1, 0),  # (N_y, D, W)
            "x": np.moveaxis(img_data[:, :, x_indices], 2, 0),  # (N_x, D, H)
            "indices": {"z": z_indices, "y": y_indices, "x": x_indices},
            "shape": (D, H, W),
            "spacing": spacing,
            "modality": modality,
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

                # Extract mask slices at same positions as CT
                z_masks = label_data[z_indices, :, :]
                y_masks = np.moveaxis(label_data[:, y_indices, :], 1, 0)
                x_masks = np.moveaxis(label_data[:, :, x_indices], 2, 0)

                # Compute per-slice coverage (number of foreground pixels)
                z_coverage = [int(z_masks[i].sum()) for i in range(len(z_indices))]
                y_coverage = [int(y_masks[i].sum()) for i in range(len(y_indices))]
                x_coverage = [int(x_masks[i].sum()) for i in range(len(x_indices))]

                # Skip if no coverage at any slice position
                if max(z_coverage + y_coverage + x_coverage) == 0:
                    continue

                # Compute bounding box for stats
                coords = np.array(label_data.nonzero())
                bbox = (
                    (int(coords[0].min()), int(coords[0].max())),
                    (int(coords[1].min()), int(coords[1].max())),
                    (int(coords[2].min()), int(coords[2].max())),
                )

                labels_data[label_id] = {
                    "z": z_masks.astype(np.uint8),
                    "y": y_masks.astype(np.uint8),
                    "x": x_masks.astype(np.uint8),
                    "coverage": {"z": z_coverage, "y": y_coverage, "x": x_coverage},
                    "bbox": bbox,
                    "volume": int(coords.shape[1]),
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
    ct_data, labels_data, error = extract_slices_for_case(case_dir, step_size)

    if error:
        return case_name, {}, error

    try:
        with h5py.File(h5_path, 'w') as h5f:
            # Save CT slices (stored once!)
            ct_grp = h5f.create_group("ct")
            ct_grp.create_dataset("z", data=ct_data["z"], compression="gzip", compression_opts=4)
            ct_grp.create_dataset("y", data=ct_data["y"], compression="gzip", compression_opts=4)
            ct_grp.create_dataset("x", data=ct_data["x"], compression="gzip", compression_opts=4)

            # Save slice indices as attributes
            ct_grp.attrs["z_indices"] = ct_data["indices"]["z"]
            ct_grp.attrs["y_indices"] = ct_data["indices"]["y"]
            ct_grp.attrs["x_indices"] = ct_data["indices"]["x"]

            # Save metadata
            meta_grp = h5f.create_group("meta")
            meta_grp.attrs["shape"] = ct_data["shape"]
            meta_grp.attrs["spacing"] = ct_data["spacing"]
            meta_grp.attrs["modality"] = ct_data["modality"]

            # Save label masks
            masks_grp = h5f.create_group("masks")

            for label_id, label_info in labels_data.items():
                label_grp = masks_grp.create_group(label_id)

                # Save mask arrays as uint8 with compression
                label_grp.create_dataset("z", data=label_info["z"], compression="gzip", compression_opts=4)
                label_grp.create_dataset("y", data=label_info["y"], compression="gzip", compression_opts=4)
                label_grp.create_dataset("x", data=label_info["x"], compression="gzip", compression_opts=4)

                # Save coverage as attributes
                label_grp.attrs["z_coverage"] = label_info["coverage"]["z"]
                label_grp.attrs["y_coverage"] = label_info["coverage"]["y"]
                label_grp.attrs["x_coverage"] = label_info["coverage"]["x"]

                # Save bbox and volume
                label_grp.attrs["bbox"] = label_info["bbox"]
                label_grp.attrs["volume"] = label_info["volume"]

        # Build stats dict for return
        case_stats = {
            "shape": ct_data["shape"],
            "spacing": ct_data["spacing"],
            "modality": ct_data["modality"],
            "num_slices": {
                "z": len(ct_data["indices"]["z"]),
                "y": len(ct_data["indices"]["y"]),
                "x": len(ct_data["indices"]["x"]),
            },
            "labels": {},
        }

        for label_id, label_info in labels_data.items():
            case_stats["labels"][label_id] = {
                "bbox": label_info["bbox"],
                "volume": label_info["volume"],
                "coverage": label_info["coverage"],
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
    step_size = getattr(cfg, 'slice_step_size', 3)

    dataset_dir = Path(cfg.paths.dataset)
    output_dir = Path(str(cfg.paths.dataset) + "_2d_shared")
    stats_path = output_dir / "stats.pkl"

    print("=== TotalSeg 3D to 2D (Shared Slices) ===")
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
        n_slices = sum(sample_case["num_slices"].values())
        n_labels = len(sample_case.get("labels", {}))
        print(f"Sample case: {n_slices} slices/axis, {n_labels} labels")

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
