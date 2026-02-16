"""Rebuild totalseg_stats.pkl from existing HDF5 files.

Only populates slice_coverage (per-axis mask sum), which is the only
field needed when crop_to_bbox=False.
"""

import pickle
from pathlib import Path

import h5py
import hydra
import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    h5_dir = Path(cfg.paths.totalseg2d_h5)
    stats_path = Path(cfg.paths.totalseg_stats)

    h5_files = sorted(h5_dir.glob("*.h5"))
    print(f"Found {len(h5_files)} HDF5 files in {h5_dir}")

    totalseg_stats = {}
    for h5_path in tqdm(h5_files, desc="Reading H5 files"):
        case_id = h5_path.stem
        case_stats = {}
        try:
            with h5py.File(h5_path, "r") as h5f:
                for label_id in h5f.keys():
                    coverage = {}
                    for axis in ("z", "y", "x"):
                        key = f"{label_id}/{axis}_slice"
                        if key in h5f:
                            coverage[axis] = int(np.sum(h5f[key][:] > 0))
                        else:
                            coverage[axis] = 0
                    case_stats[label_id] = {"slice_coverage": coverage}
        except Exception as e:
            print(f"Error reading {h5_path}: {e}")
            continue

        if case_stats:
            totalseg_stats[case_id] = case_stats

    print(f"\nRebuilt stats for {len(totalseg_stats)} cases")
    unique_labels = set()
    for case_data in totalseg_stats.values():
        unique_labels.update(case_data.keys())
    print(f"Unique labels: {len(unique_labels)}")

    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "wb") as f:
        pickle.dump(totalseg_stats, f)
    print(f"Saved to {stats_path}")


if __name__ == "__main__":
    main()
