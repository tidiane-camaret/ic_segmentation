import os
import glob
import random
import numpy as np
import matplotlib.pyplot as plt

# Configuration
DATA_ROOT = "/work/dlclarge2/ndirt-SegFM3D/data/medsegbench"
SAVE_PATH = "experiments/medsegbench_sample.png"
NUM_SAMPLES = 5

def find_keys(keys, split='train'):
    """Finds image/label keys robustly (handles 'img' vs 'images')."""
    # Priority list for image keys
    img_candidates = [f"{split}_images", f"{split}_img", "images", "img"]
    # Priority list for label keys
    lbl_candidates = [f"{split}_label", f"{split}_mask", "labels", "masks", "label"]
    
    img_key = next((k for k in img_candidates if k in keys), None)
    lbl_key = next((k for k in lbl_candidates if k in keys), None)
    return img_key, lbl_key

def visualize_grid():
    # 1. Find all .npz files
    all_files = glob.glob(os.path.join(DATA_ROOT, "*.npz"))
    if not all_files:
        print(f"[Error] No .npz files found in {DATA_ROOT}")
        return

    # 2. Select 5 Unique Datasets (or fewer if not enough exist)
    sample_count = min(NUM_SAMPLES, len(all_files))
    selected_files = random.sample(all_files, sample_count)
    
    print(f"Selected {sample_count} datasets for visualization:")
    
    # 3. Setup Grid Plot
    fig, axes = plt.subplots(sample_count, 3, figsize=(12, 4 * sample_count))
    if sample_count == 1: axes = [axes] # Handle single row case

    for i, file_path in enumerate(selected_files):
        dataset_name = os.path.basename(file_path).replace(".npz", "").replace("_256", "")
        print(f" {i+1}. Processing {dataset_name}...")
        
        try:
            with np.load(file_path) as data:
                keys = list(data.keys())
                
                # Find keys (try 'train', fallback to 'test' or 'val')
                split = 'train'
                img_key, lbl_key = find_keys(keys, split)
                if not img_key: 
                    split = 'test'
                    img_key, lbl_key = find_keys(keys, split)
                
                if not img_key or not lbl_key:
                    print(f"    [Skipping] Could not find keys in {dataset_name}")
                    continue

                # Load full arrays to pick a random index
                images = data[img_key]
                labels = data[lbl_key]
                
                idx = random.randint(0, len(images) - 1)
                img = images[idx]
                mask = labels[idx]

                # --- Handle Shape & Color ---
                viz_img = img
                cmap = 'gray'
                
                # If RGB (Channels First: 3, H, W) -> Transpose to (H, W, 3)
                if img.ndim == 3 and img.shape[0] == 3:
                    viz_img = np.transpose(img, (1, 2, 0))
                    cmap = None
                # If RGB (Channels Last: H, W, 3) -> Keep as is
                elif img.ndim == 3 and img.shape[-1] == 3:
                    cmap = None
                
                # Normalize if needed (float 0..1 or uint8 0..255)
                if viz_img.max() > 1 and viz_img.dtype != np.uint8:
                    viz_img = (viz_img / viz_img.max() * 255).astype(np.uint8)

                # --- Plotting ---
                ax = axes[i]
                
                # Image
                ax[0].imshow(viz_img, cmap=cmap)
                ax[0].set_title(f"{dataset_name}\n({split}, idx={idx})")
                ax[0].axis('off')

                # Mask
                ax[1].imshow(mask, cmap='viridis', interpolation='nearest')
                ax[1].set_title(f"Mask (Max: {mask.max()})")
                ax[1].axis('off')

                # Overlay
                ax[2].imshow(viz_img, cmap=cmap)
                ax[2].imshow(mask, cmap='jet', alpha=0.4, interpolation='nearest')
                ax[2].set_title("Overlay")
                ax[2].axis('off')

        except Exception as e:
            print(f"    [Error] {dataset_name}: {e}")

    plt.tight_layout()
    plt.savefig(SAVE_PATH)
    print(f"\n[Success] Visualization saved to: {os.path.abspath(SAVE_PATH)}")

if __name__ == "__main__":
    visualize_grid()