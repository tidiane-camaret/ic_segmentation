from pathlib import Path
import nibabel as nib
import hydra
from omegaconf import DictConfig
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import h5py
import numpy as np

def extract_label_slices(label_file: Path, case_path: Path):
    """
    For a single label file, compute stats and extract 2D slices.
    Does NOT save to disk, returns slices and stats.
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

        # Find slices with max coverage
        z_counts = label_data.sum(axis=(1, 2))
        y_counts = label_data.sum(axis=(0, 2))
        x_counts = label_data.sum(axis=(0, 1))

        zc = int(z_counts.argmax())
        yc = int(y_counts.argmax())
        xc = int(x_counts.argmax())

        stats = {
            "bbox": ((zmin, zmax), (ymin, ymax), (xmin, xmax)),
            "volume": volume,
            "center": center,
            "spacing_3d": spacing_3d,
            "slice_indices": {"z": zc, "y": yc, "x": xc},
            "slice_coverage": {"z": int(z_counts[zc]), "y": int(y_counts[yc]), "x": int(x_counts[xc])},
        }

        mask_slices = {"z": label_data[zc, :, :],
                       "y": label_data[:, yc, :],
                       "x": label_data[:, :, xc]}

        # Load image data
        img_nii = nib.load(str(case_path / "mri.nii.gz"))
        img_data = img_nii.get_fdata()
        img_slices = {"z": img_data[zc, :, :],
                      "y": img_data[:, yc, :],
                      "x": img_data[:, :, xc]}

        return (img_slices, mask_slices), stats
    except Exception:
        return None, None


def process_case(args):
    """
    Process a single case:
    1. Find all its labels.
    2. For each label, extract best 2D slices for each axis.
    3. Save all slices to a single HDF5 file.
    4. Return stats for the case.
    """
    case_dir_str, output_dir_str = args
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
                
                slice_data, stats = extract_label_slices(label_file, case_dir)
                
                if stats is None or slice_data is None:
                    continue

                img_slices, mask_slices = slice_data
                case_stats[label_id] = stats
                
                # Save slices to HDF5
                for axis in ['z', 'y', 'x']:
                    h5f.create_dataset(f"{label_id}/{axis}_slice_img", data=img_slices[axis].astype(np.float32))
                    h5f.create_dataset(f"{label_id}/{axis}_slice", data=mask_slices[axis].astype(np.float32))
                    
        if not case_stats:
            # If no labels were successfully processed, the h5 file is empty and can be removed.
            h5_path.unlink()
            return case_name, {}, f"No valid labels found for case {case_name}"

        return case_name, case_stats, None
    except Exception as e:
        # Clean up partially created file on error
        if h5_path.exists():
            h5_path.unlink()
        return case_name, {}, str(e)


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    import pickle
    from tqdm import tqdm

    totalseg_dir = Path(cfg.paths.totalseg)
    # New output dir for HDF5 files
    totalseg_2d_dir_h5 = Path(cfg.paths.totalseg2d_h5) 
    stats_path = Path(cfg.paths.totalseg_stats)

    print(f"Input dir: {totalseg_dir}")
    print(f"HDF5 Output dir: {totalseg_2d_dir_h5}")
    print(f"Stats path: {stats_path}")

    if not totalseg_dir.exists():
        print(f"ERROR: TotalSeg directory not found: {totalseg_dir}")
        return

    totalseg_2d_dir_h5.mkdir(parents=True, exist_ok=True)

    # Build list of all cases
    tasks = [
        (str(case_dir), str(totalseg_2d_dir_h5))
        for case_dir in totalseg_dir.iterdir() if case_dir.is_dir()
    ]
    print(f"Total cases to process: {len(tasks)}")

    # Process in parallel
    n_workers = min(mp.cpu_count(), 20)
    print(f"Using {n_workers} workers...")

    totalseg_stats = {}
    errors = []

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Use tqdm to show progress for futures
        futures = {executor.submit(process_case, task) for task in tasks}
        
        for future in tqdm(as_completed(futures), total=len(tasks), desc="Processing cases"):
            case_name, case_stats, error = future.result()
            
            if case_stats:
                totalseg_stats[case_name] = case_stats
            
            if error:
                errors.append((case_name, error))

    print(f"\nCompleted processing for {len(totalseg_stats)} cases.")
    unique_labels = set()
    for case_data in totalseg_stats.values():
        unique_labels.update(case_data.keys())
    print(f"Unique labels found: {len(unique_labels)}")
    
    if errors:
        print(f"Errors ({len(errors)}):")
        for name, err in errors[:10]:
            print(f"  {name}: {err}")

    # Save stats to pickle
    if totalseg_stats:
        print(f"\nSaving stats to {stats_path}...")
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        with open(stats_path, "wb") as f:
            pickle.dump(totalseg_stats, f)
        print(f"Saved stats for {len(totalseg_stats)} cases.")
    else:
        print("\nNo stats generated. Skipping stat file saving.")


if __name__ == "__main__":
    main()
