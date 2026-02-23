import random
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import h5py
import hydra
import nibabel as nib
import numpy as np
from omegaconf import DictConfig

class_list = [
        "spinal_cord",  # Added
        "rib_left_1", "rib_right_1", "rib_left_2", "rib_right_2", 
        "rib_left_3", "rib_right_3", "rib_left_4", "rib_right_4", 
        "rib_left_5", "rib_right_5", "rib_left_6", "rib_right_6",
        "lung_upper_lobe_left", "lung_upper_lobe_right", 
        "lung_lower_lobe_left", "lung_lower_lobe_right", 
        "lung_middle_lobe_right", 
        "heart", "trachea", "thyroid_gland", 
        "stomach", "spleen", "pancreas", "gallbladder", 
        "adrenal_gland_left", "adrenal_gland_right", 
        "clavicula_left", "clavicula_right", 
        "scapula_left", "scapula_right", 
        "humerus_left", "humerus_right", 
        "hip_left", "hip_right", 
        "gluteus_maximus_left", "gluteus_maximus_right", 
        "gluteus_medius_left", "gluteus_medius_right", 
        "gluteus_minimus_left", "gluteus_minimus_right", 
        "aorta", "pulmonary_vein", 
        "subclavian_artery_left", "subclavian_artery_right", 
        "common_carotid_artery_left", "common_carotid_artery_right", 
        "vertebrae_C1", "vertebrae_C2", "vertebrae_C3", "vertebrae_C4", 
        "vertebrae_C5", "vertebrae_C6", "vertebrae_C7", 
        "vertebrae_T1", "vertebrae_T2", "vertebrae_T3", "vertebrae_T4", "vertebrae_T5",
        "vertebrae_S1", # Added
        "rib_left_7", "rib_right_7", "rib_left_8", "rib_right_8", 
        "rib_left_9", "rib_right_9", "rib_left_10", "rib_right_10", 
        "rib_left_11", "rib_right_11", "rib_left_12", "rib_right_12", 
        "sternum", "costal_cartilages", 
        "liver", 
        "kidney_left", "kidney_right", "kidney_cyst_left", "kidney_cyst_right", 
        "colon", "small_bowel", "duodenum", 
        "urinary_bladder", "prostate", "esophagus", 
        "brain", "atrial_appendage_left", 
        "femur_left", "femur_right", "skull", 
        "iliopsoas_left", "iliopsoas_right", 
        "autochthon_left", "autochthon_right", 
        "inferior_vena_cava", "superior_vena_cava", 
        "portal_vein_and_splenic_vein", 
        "iliac_artery_left", "iliac_artery_right", 
        "iliac_vena_left", "iliac_vena_right", 
        "brachiocephalic_vein_left", "brachiocephalic_vein_right", "brachiocephalic_trunk", 
        "vertebrae_T6", "vertebrae_T7", "vertebrae_T8", "vertebrae_T9", 
        "vertebrae_T10", "vertebrae_T11", "vertebrae_T12", 
        "vertebrae_L1", "vertebrae_L2", "vertebrae_L3", "vertebrae_L4", "vertebrae_L5", 
        "sacrum"
    ]

def extract_label_slices(label_file: Path, case_path: Path, step_size: int, min_coverage_ratio: float = 0.1):
    """
    For a single label file, compute stats and extract every n-th 2D slice
    within the bounding box. Returns shape (N, H, W) for each axis.

    Filters out slices with coverage below min_coverage_ratio of the max coverage slice.
    """
    try:
        # Load label data
        label_nii = nib.load(str(label_file))
        label_data = label_nii.get_fdata()

        # Compute stats
        coords = np.array(label_data.nonzero())
        if coords.shape[1] == 0:
            return None, None  # Empty label, skip

        zmin, zmax = int(coords[0].min()), int(coords[0].max())
        ymin, ymax = int(coords[1].min()), int(coords[1].max())
        xmin, xmax = int(coords[2].min()), int(coords[2].max())
        volume = coords.shape[1]
        center = ((zmin+zmax)//2, (ymin+ymax)//2, (xmin+xmax)//2)

        # Extract spacing from affine
        spacing_3d = tuple(float(s) for s in nib.affines.voxel_sizes(label_nii.affine))

        # Compute per-slice coverage for filtering
        z_counts = label_data.sum(axis=(1, 2))  # Coverage per z-slice
        y_counts = label_data.sum(axis=(0, 2))  # Coverage per y-slice
        x_counts = label_data.sum(axis=(0, 1))  # Coverage per x-slice

        def filter_indices_by_coverage(indices, counts, min_ratio):
            """Keep only slices with coverage >= min_ratio * max_coverage."""
            if not indices:
                return indices, []
            coverages = [int(counts[i]) for i in indices]
            max_cov = max(coverages)
            threshold = max_cov * min_ratio
            filtered = [(idx, cov) for idx, cov in zip(indices, coverages) if cov >= threshold]
            if not filtered:
                # Fallback: keep slice with max coverage
                best_idx = indices[coverages.index(max_cov)]
                return [best_idx], [max_cov]
            return [f[0] for f in filtered], [f[1] for f in filtered]

        # Generate candidate indices for every step_size slice
        z_candidates = list(range(zmin, zmax + 1, step_size))
        y_candidates = list(range(ymin, ymax + 1, step_size))
        x_candidates = list(range(xmin, xmax + 1, step_size))

        # Failsafe: if bounding box is thinner than step_size, at least grab the center
        if not z_candidates: z_candidates = [(zmin+zmax)//2]
        if not y_candidates: y_candidates = [(ymin+ymax)//2]
        if not x_candidates: x_candidates = [(xmin+xmax)//2]

        # Filter by coverage
        z_indices, z_coverages = filter_indices_by_coverage(z_candidates, z_counts, min_coverage_ratio)
        y_indices, y_coverages = filter_indices_by_coverage(y_candidates, y_counts, min_coverage_ratio)
        x_indices, x_coverages = filter_indices_by_coverage(x_candidates, x_counts, min_coverage_ratio)

        stats = {
            "bbox": ((zmin, zmax), (ymin, ymax), (xmin, xmax)),
            "volume": volume,
            "center": center,
            "spacing_3d": spacing_3d,
            "slice_indices": {"z": z_indices, "y": y_indices, "x": x_indices},
            "slice_coverages": {"z": z_coverages, "y": y_coverages, "x": x_coverages},
            "num_slices": {"z": len(z_indices), "y": len(y_indices), "x": len(x_indices)},
        }

        # Extract slices and move the extracted axis to position 0 to standardize to (N, H, W)
        mask_slices = {
            "z": label_data[z_indices, :, :],                                 # (N, Y, X)
            "y": np.moveaxis(label_data[:, y_indices, :], 1, 0),              # (Z, N, X) -> (N, Z, X)
            "x": np.moveaxis(label_data[:, :, x_indices], 2, 0)               # (Z, Y, N) -> (N, Z, Y)
        }

        # Load image data
        img_file = case_path / "ct.nii.gz"
        if not img_file.exists():
            img_file = case_path / "mri.nii.gz"
        img_nii = nib.load(str(img_file))
        img_data = img_nii.get_fdata()
        
        img_slices = {
            "z": img_data[z_indices, :, :],
            "y": np.moveaxis(img_data[:, y_indices, :], 1, 0),
            "x": np.moveaxis(img_data[:, :, x_indices], 2, 0)
        }

        return (img_slices, mask_slices), stats
    except Exception as e:
        print(f"    Error processing label {label_file.name}: {e}")
        return None, None


def process_case(args):
    """
    Process a single case:
    1. Find all its labels.
    2. For each label, extract multiple 2D slices based on step size.
    3. Save all slices (as 3D stacks) to a single HDF5 file.
    4. Return stats for the case.
    """
    case_dir_str, output_dir_str, step_size, min_coverage_ratio = args
    case_dir = Path(case_dir_str)
    output_dir = Path(output_dir_str)
    case_name = case_dir.name
    
    h5_path = output_dir / f"{case_name}.h5"
    case_stats = {}

    labels_dir = case_dir / "segmentations"
    if not labels_dir.exists():
        return case_name, {}, f"No 'segmentations' directory in {case_dir}"

    try:
        with h5py.File(h5_path, 'w') as h5f:
            for label_file in labels_dir.glob("*.nii.gz"):
                label_id = label_file.name.split(".")[0]
                
                slice_data, stats = extract_label_slices(label_file, case_dir, step_size, min_coverage_ratio)
                
                if stats is None or slice_data is None:
                    continue

                img_slices, mask_slices = slice_data
                case_stats[label_id] = stats
                
                # Save slice stacks to HDF5. Note the datasets are now 3D arrays (N, H, W)
                for axis in ['z', 'y', 'x']:
                    h5f.create_dataset(f"{label_id}/{axis}_slices_img", data=img_slices[axis].astype(np.float32))
                    h5f.create_dataset(f"{label_id}/{axis}_slices", data=mask_slices[axis].astype(np.float32))
                    
        if not case_stats:
            h5_path.unlink()
            return case_name, {}, f"No valid labels found for case {case_name}"

        return case_name, case_stats, None
    except Exception as e:
        if h5_path.exists():
            h5_path.unlink()
        return case_name, {}, str(e)


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    import pickle
    from tqdm import tqdm

    max_files = getattr(cfg, 'max_files_3d_to_2d', None)
    step_size = getattr(cfg, 'slice_step_size', 3)
    min_coverage_ratio = getattr(cfg, 'min_coverage_ratio', 0.1)  # Filter slices with < 10% of max coverage
    print(cfg.dataset)
    orig_dir = Path(cfg.paths.dataset)
    output_dir = Path("/work/dlclarge2/ndirt-SegFM3D/data/totalseg2d")
    #Path(cfg.paths.totalseg2d_every_n_slice)
    stats_path = output_dir / "stats.pkl"

    print(f"Current Stats Path: {stats_path}")
    print(f"Input dir: {orig_dir}")
    print(f"HDF5 Output dir: {output_dir}")
    print(f"Stats path: {stats_path}")
    print(f"Slice step size (n): {step_size}")
    print(f"Min coverage ratio: {min_coverage_ratio}")

    if not orig_dir.exists():
        print(f"ERROR: dataset directory not found: {orig_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Pass step_size and min_coverage_ratio into the task arguments
    tasks = [
        (str(case_dir), str(output_dir), step_size, min_coverage_ratio)
        for case_dir in orig_dir.iterdir() if case_dir.is_dir()
    ]

    seed = cfg.get("seed", cfg.training.get("seed", 42))
    random.seed(seed)
    random.shuffle(tasks)
    print(f"Randomized case order with seed: {seed}")

    if max_files is not None:
        tasks = tasks[:max_files]
        print(f"Limiting to max_files={max_files}")
    print(f"Total cases to process: {len(tasks)}")

    n_workers = min(mp.cpu_count(), 29)
    print(f"Using {n_workers} workers...")

    dataset_stats = {}
    errors = []

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(process_case, task) for task in tasks}
        
        for future in tqdm(as_completed(futures), total=len(tasks), desc="Processing cases"):
            case_name, case_stats, error = future.result()
            
            if case_stats:
                dataset_stats[case_name] = case_stats
            
            if error:
                errors.append((case_name, error))

    print(f"\nCompleted processing for {len(dataset_stats)} cases.")
    unique_labels = set()
    for case_data in dataset_stats.values():
        unique_labels.update(case_data.keys())
    print(f"Unique labels found: {len(unique_labels)}")
    
    if errors:
        print(f"Errors ({len(errors)}):")
        for name, err in errors[:10]:
            print(f"  {name}: {err}")

    if dataset_stats:
        print(f"\nSaving stats to {stats_path}...")
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        with open(stats_path, "wb") as f:
            pickle.dump(dataset_stats, f)
        print(f"Saved stats for {len(dataset_stats)} cases.")
    else:
        print("\nNo stats generated. Skipping stat file saving.")


if __name__ == "__main__":
    main()