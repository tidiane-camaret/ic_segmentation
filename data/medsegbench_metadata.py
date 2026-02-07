import os
import glob
import pickle
import numpy as np
from tqdm import tqdm

# Configuration
DATA_ROOT = "/work/dlclarge2/ndirt-SegFM3D/data/medsegbench"
OUTPUT_PICKLE = "data/medsegbench_metadata.pkl"

def scan_npz_files():
    # 1. Find all .npz files matching the pattern
    npz_files = glob.glob(os.path.join(DATA_ROOT, "*_256.npz"))
    
    metadata = {}
    print(f"Found {len(npz_files)} .npz files in {DATA_ROOT}...")

    # 2. Iterate and inspect headers (fast load)
    for file_path in tqdm(npz_files):
        filename = os.path.basename(file_path)
        dataset_name = filename.replace("_256.npz", "") # e.g., 'abdomenus'
        
        try:
            # Load with mmap_mode='r' to read headers without loading full data to RAM
            with np.load(file_path, mmap_mode='r') as data:
                # Check available splits
                keys = data.files
                stats = {
                    "path": file_path,
                    "splits": {},
                    "total_samples": 0,
                    "num_classes": 0
                }
                
                # Count samples per split
                for split in ['train', 'val', 'test']:
                    img_key = f"{split}_img"
                    lbl_key = f"{split}_label"
                    
                    if img_key in keys:
                        count = data[img_key].shape[0]
                        stats["splits"][split] = count
                        stats["total_samples"] += count
                        
                        # Heuristic for num_classes: check the first mask of training
                        # (Fast check, avoiding full dataset scan)
                        if stats["num_classes"] == 0 and lbl_key in keys:
                            # We load just one mask to check max value
                            sample_mask = data[lbl_key][0]
                            stats["num_classes"] = int(np.max(sample_mask)) + 1

                metadata[dataset_name] = stats
                
        except Exception as e:
            print(f"Failed to read {filename}: {e}")

    return metadata

if __name__ == "__main__":
    if not os.path.exists(DATA_ROOT):
        print(f"Error: {DATA_ROOT} does not exist.")
    else:
        data = scan_npz_files()
        
        print(f"\nScan complete. indexed {len(data)} datasets.")
        with open(OUTPUT_PICKLE, 'wb') as f:
            pickle.dump(data, f)
        print(f"Metadata saved to {OUTPUT_PICKLE}")