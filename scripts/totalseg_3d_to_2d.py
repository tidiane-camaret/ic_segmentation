from pathlib import Path
import nibabel as nib
import hydra
from omegaconf import DictConfig, OmegaConf
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

def process_label_full(args):
    """Compute stats and extract 2D slices in a single pass.

    Saves slices as .npy files for fast loading (10-50x faster than .nii.gz).
    Spacing info is stored in the returned stats dict.
    """
    import nibabel as nib
    import numpy as np
    from pathlib import Path

    case_name, label_file_str, case_path_str, output_dir_str = args
    label_file = Path(label_file_str)
    case_path = Path(case_path_str)
    output_dir = Path(output_dir_str)
    label_id = label_file.name.split(".")[0]

    try:
        # Load label data
        label_nii = nib.load(str(label_file))
        label_data = label_nii.get_fdata()

        # Compute stats
        coords = label_data.nonzero()
        if len(coords[0]) == 0:
            return case_name, label_id, None, True, None  # Empty label, skip

        zmin, zmax = int(coords[0].min()), int(coords[0].max())
        ymin, ymax = int(coords[1].min()), int(coords[1].max())
        xmin, xmax = int(coords[2].min()), int(coords[2].max())
        volume = len(coords[0])
        center = ((zmin+zmax)//2, (ymin+ymax)//2, (xmin+xmax)//2)

        # Extract spacing from affine (diagonal elements give voxel size)
        spacing_3d = tuple(float(s) for s in nib.affines.voxel_sizes(label_nii.affine))

        # Extract slices at position with maximum label coverage (not center)
        # This handles discontinuous labels like costal_cartilages where center may be empty
        z_counts = label_data.sum(axis=(1, 2))  # sum over y,x for each z
        y_counts = label_data.sum(axis=(0, 2))  # sum over z,x for each y
        x_counts = label_data.sum(axis=(0, 1))  # sum over z,y for each x

        zc = int(z_counts.argmax())
        yc = int(y_counts.argmax())
        xc = int(x_counts.argmax())

        stats = {
            "bbox": ((zmin, zmax), (ymin, ymax), (xmin, xmax)),
            "volume": volume,
            "center": center,
            "spacing_3d": spacing_3d,  # (sz, sy, sx) in mm
            "slice_indices": {"z": zc, "y": yc, "x": xc},
            "slice_coverage": {"z": int(z_counts[zc]), "y": int(y_counts[yc]), "x": int(x_counts[xc])},
        }

        slices = {"z": label_data[zc, :, :],
                  "y": label_data[:, yc, :],
                  "x": label_data[:, :, xc]}

        # Load image data (only once per label)
        img_nii = nib.load(str(case_path / "mri.nii.gz"))
        img_data = img_nii.get_fdata()
        img_slices = {"z": img_data[zc, :, :],
                      "y": img_data[:, yc, :],
                      "x": img_data[:, :, xc]}

        # Save slices as .npy (much faster to load than .nii.gz)
        for axis, slice_data in slices.items():
            save_dir = output_dir / case_name / label_id
            save_dir.mkdir(parents=True, exist_ok=True)
            np.save(save_dir / f"{axis}_slice.npy", slice_data.astype(np.float32))
            np.save(save_dir / f"{axis}_slice_img.npy", img_slices[axis].astype(np.float32))

        return case_name, label_id, stats, True, None
    except Exception as e:
        return case_name, label_id, None, False, str(e)


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    import pickle

    totalseg_dir = Path(cfg.paths.totalseg)
    totalseg_2d_dir = Path(cfg.paths.totalsegmri2d)
    stats_path = Path(cfg.paths.totalseg_stats)

    print(f"Input dir: {totalseg_dir}")
    print(f"Output dir: {totalseg_2d_dir}")
    print(f"Stats path: {stats_path}")

    if not totalseg_dir.exists():
        print(f"ERROR: TotalSeg directory not found: {totalseg_dir}")
        return

    totalseg_2d_dir.mkdir(parents=True, exist_ok=True)

    # Build list of all tasks
    case_dirs = list(totalseg_dir.iterdir())
    tasks = []
    for case_dir in case_dirs:
        if case_dir.is_dir():
            labels_dir = case_dir / "segmentations"
            for label_file in labels_dir.glob("*.nii.gz"):
                tasks.append((case_dir.name, str(label_file), str(case_dir), str(totalseg_2d_dir)))

    print(f"Total label files to process: {len(tasks)}")

    # Process in parallel
    n_workers = min(mp.cpu_count(), 20)
    print(f"Using {n_workers} workers...")

    totalseg_stats = {}
    label_ids = []
    completed = 0
    errors = []

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(process_label_full, task): task for task in tasks}
        
        for future in as_completed(futures):
            case_name, label_id, stats, success, error = future.result()
            completed += 1
            
            if stats is not None:
                totalseg_stats.setdefault(case_name, {})[label_id] = stats
                if label_id not in label_ids:
                    label_ids.append(label_id)
            
            if not success and error is not None:
                errors.append((f"{case_name}/{label_id}", error))
            
            if completed % 100 == 0:
                print(f"Progress: {completed}/{len(tasks)} ({100*completed/len(tasks):.1f}%)")

    print(f"\nCompleted: {completed}/{len(tasks)}")
    print(f"Cases with stats: {len(totalseg_stats)}")
    print(f"Unique labels: {len(label_ids)}")
    if errors:
        print(f"Errors ({len(errors)}):")
        for name, err in errors[:10]:
            print(f"  {name}: {err}")

    # Save stats to pickle
    print(f"\nSaving stats to {stats_path}...")
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "wb") as f:
        pickle.dump(totalseg_stats, f)
    print(f"Saved stats for {len(totalseg_stats)} cases.")


if __name__ == "__main__":
    main()