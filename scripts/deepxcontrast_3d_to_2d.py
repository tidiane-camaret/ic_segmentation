"""
Convert DeepXcontrast 3D volumes to 2D slices for in-context segmentation training.

Supports two modalities:
- MRI (T1): Uses GT/ labels which are in T1 space (256³, 1mm isotropic)
- CT: Uses CT/ labels which are warped to CT space (512×512×161, 0.54mm×0.54mm×1mm)

Output HDF5 structure per case:
    {label_name}/{axis}_slice_img   - image slice
    {label_name}/{axis}_slice       - mask slice

Labels processed:
    MRI mode: c1, c2, c3, tissue, nuc (from GT/)
    CT mode: c1, c2, c3, nuc (from CT/out_CTtissue_*.nii.gz)
"""
import multiprocessing as mp
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np
from tqdm import tqdm


# Label configurations per modality
LABELS_MRI = {
    "c1": "GT/c1.nii",
    "c2": "GT/c2.nii",
    "c3": "GT/c3.nii",
    "tissue": "GT/tissue.nii",
    "nuc": "GT/nuc.nii",
}

LABELS_CT = {
    "c1": "CT/out_CTtissue_c1.nii.gz",
    "c2": "CT/out_CTtissue_c2.nii.gz",
    "c3": "CT/out_CTtissue_c3.nii.gz",
    "nuc": "CT/out_CTnuclei.nii.gz",
}


def load_nii(path: Path):
    """Load NIfTI file, return (data, affine, spacing)."""
    nii = nib.load(str(path))
    data = nii.get_fdata()
    spacing = tuple(float(s) for s in nib.affines.voxel_sizes(nii.affine))
    return data, nii.affine, spacing


def resample_to_shape(data: np.ndarray, target_shape: tuple, order: int = 1) -> np.ndarray:
    """Resample 3D array to target shape using scipy zoom."""
    from scipy.ndimage import zoom
    factors = [t / s for t, s in zip(target_shape, data.shape)]
    return zoom(data, factors, order=order)


def extract_label_slices(label_file: Path, img_data: np.ndarray, spacing_3d: tuple):
    """
    For a single label file, compute stats and extract 2D slices.
    Returns (img_slices, mask_slices), stats or (None, None) if empty.

    If label and image have different shapes, resamples image to label shape.
    """
    try:
        label_nii = nib.load(str(label_file))
        label_data = label_nii.get_fdata()

        # Resample image to label shape if needed (CT labels are often in different space)
        if img_data.shape != label_data.shape:
            img_resampled = resample_to_shape(img_data, label_data.shape, order=1)
        else:
            img_resampled = img_data

        # For multi-class labels (tissue), binarize for coverage computation
        label_binary = (label_data > 0).astype(np.float32)

        # Compute stats
        coords = np.array(label_binary.nonzero())
        if coords.shape[1] == 0:
            return None, None  # Empty label

        # Bounding box
        mins = coords.min(axis=1)
        maxs = coords.max(axis=1)
        volume = int(coords.shape[1])
        center = tuple(int((mins[i] + maxs[i]) // 2) for i in range(3))

        # Find slices with max coverage per axis
        x_counts = label_binary.sum(axis=(1, 2))
        y_counts = label_binary.sum(axis=(0, 2))
        z_counts = label_binary.sum(axis=(0, 1))

        xc = int(x_counts.argmax())
        yc = int(y_counts.argmax())
        zc = int(z_counts.argmax())

        stats = {
            "bbox": ((int(mins[0]), int(maxs[0])),
                     (int(mins[1]), int(maxs[1])),
                     (int(mins[2]), int(maxs[2]))),
            "volume": volume,
            "center": center,
            "spacing_3d": spacing_3d,
            "label_shape": label_data.shape,
            "slice_indices": {"x": xc, "y": yc, "z": zc},
            "slice_coverage": {
                "x": int(x_counts[xc].item()),
                "y": int(y_counts[yc].item()),
                "z": int(z_counts[zc].item())
            },
        }

        # Extract slices (keep original label values for multi-class)
        mask_slices = {
            "x": label_data[xc, :, :],
            "y": label_data[:, yc, :],
            "z": label_data[:, :, zc],
        }
        img_slices = {
            "x": img_resampled[xc, :, :],
            "y": img_resampled[:, yc, :],
            "z": img_resampled[:, :, zc],
        }

        return (img_slices, mask_slices), stats

    except Exception as e:
        print(f"    Error processing {label_file.name}: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def process_case(args):
    """Process a single DeepXcontrast case."""
    case_dir_str, output_dir_str, modality = args
    case_dir = Path(case_dir_str)
    output_dir = Path(output_dir_str)
    case_name = case_dir.parent.name  # CASE_ID is parent of "0"

    h5_path = output_dir / f"{case_name}.h5"
    case_stats = {}

    # Select image and labels based on modality
    if modality == "ct":
        img_path = case_dir / "CT.nii"
        labels_config = LABELS_CT
    else:  # mri
        img_path = case_dir / "T1.nii"
        labels_config = LABELS_MRI

    if not img_path.exists():
        return case_name, {}, f"No {img_path.name} in {case_dir}"

    try:
        img_data, _, spacing_3d = load_nii(img_path)

        with h5py.File(h5_path, 'w') as h5f:
            # Store case metadata
            h5f.attrs['case_id'] = case_name
            h5f.attrs['modality'] = modality
            h5f.attrs['spacing'] = spacing_3d

            for label_name, label_relpath in labels_config.items():
                label_file = case_dir / label_relpath
                if not label_file.exists():
                    # Try .nii.gz variant for MRI labels
                    if modality == "mri":
                        label_file = case_dir / (label_relpath + ".gz")
                    if not label_file.exists():
                        continue

                slice_data, stats = extract_label_slices(label_file, img_data, spacing_3d)

                if stats is None or slice_data is None:
                    continue

                img_slices, mask_slices = slice_data
                case_stats[label_name] = stats

                # Save slices to HDF5
                grp = h5f.create_group(label_name)
                for axis in ['x', 'y', 'z']:
                    grp.create_dataset(f"{axis}_slice_img",
                                       data=img_slices[axis].astype(np.float32),
                                       compression='gzip')
                    grp.create_dataset(f"{axis}_slice",
                                       data=mask_slices[axis].astype(np.float32),
                                       compression='gzip')

        if not case_stats:
            h5_path.unlink()
            return case_name, {}, f"No valid labels found for case {case_name}"

        return case_name, case_stats, None

    except Exception as e:
        if h5_path.exists():
            h5_path.unlink()
        return case_name, {}, str(e)


def main():
    import argparse
    import pickle

    parser = argparse.ArgumentParser(description="Convert DeepXcontrast 3D to 2D slices")
    parser.add_argument("--input", type=str,
                        default="/nfs/data/nii/data1/DeepXcontrast",
                        help="Input dataset directory")
    parser.add_argument("--output", type=str,
                        default="/nfs/data/nii/data1/Analysis/camaret___in_context_segmentation/ANALYSIS_20251122/data/deepxcontrast",
                        help="Output HDF5 directory")
    parser.add_argument("--stats", type=str, default=None,
                        help="Output stats pickle file (default: {output}/stats_{modality}.pkl)")
    parser.add_argument("--modality", type=str, default="ct", choices=["ct", "mri"],
                        help="Modality to extract (ct or mri)")
    parser.add_argument("--max-cases", type=int, default=None,
                        help="Limit number of cases to process")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for case ordering")
    args = parser.parse_args()

    dataset_dir = Path(args.input)
    output_dir = Path(args.output) / args.modality
    stats_path = Path(args.stats) if args.stats else output_dir / f"stats.pkl"

    print(f"Input dir:  {dataset_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Stats path: {stats_path}")
    print(f"Modality:   {args.modality}")

    if not dataset_dir.exists():
        print(f"ERROR: dataset directory not found: {dataset_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all cases (structure: CASE_ID/0/)
    tasks = []
    for case_id_dir in dataset_dir.iterdir():
        if case_id_dir.is_dir():
            case_subdir = case_id_dir / "0"
            if case_subdir.exists():
                tasks.append((str(case_subdir), str(output_dir), args.modality))

    if not tasks:
        print("ERROR: No cases found")
        return

    # Randomize order
    random.seed(args.seed)
    random.shuffle(tasks)
    print(f"Found {len(tasks)} cases (seed={args.seed})")

    if args.max_cases:
        tasks = tasks[:args.max_cases]
        print(f"Limiting to {args.max_cases} cases")

    # Process in parallel
    n_workers = args.workers or min(mp.cpu_count(), 16)
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
    print(f"\nCompleted: {len(dataset_stats)} cases")

    label_counts = {}
    for case_data in dataset_stats.values():
        for label in case_data.keys():
            label_counts[label] = label_counts.get(label, 0) + 1
    print("Labels found:")
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count} cases")

    if errors:
        print(f"\nErrors ({len(errors)}):")
        for name, err in errors[:10]:
            print(f"  {name}: {err}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")

    # Save stats
    if dataset_stats:
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        with open(stats_path, "wb") as f:
            pickle.dump(dataset_stats, f)
        print(f"\nSaved stats for {len(dataset_stats)} cases to {stats_path}")


if __name__ == "__main__":
    main()
