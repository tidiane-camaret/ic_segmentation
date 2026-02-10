import pandas as pd
import os
import inspect
import sys
from pathlib import Path

import medsegbench
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

# Load config to get paths.medsegbench
config_path = Path(__file__).resolve().parents[1] / "configs"
with initialize_config_dir(config_dir=str(config_path), version_base=None):
    cfg = compose(config_name="train")

DOWNLOAD_ROOT = cfg.paths.medsegbench
IMG_SIZE = 256
stats_list = []

print(f"Loaded path from config: {DOWNLOAD_ROOT}")

# --- CRITICAL FIX: Create the directory manually first ---
if not os.path.exists(DOWNLOAD_ROOT):
    print(f"Creating directory: {DOWNLOAD_ROOT}")
    os.makedirs(DOWNLOAD_ROOT, exist_ok=True)
# ---------------------------------------------------------

# 2. Dynamically find all dataset classes
dataset_classes = []
for name, obj in inspect.getmembers(medsegbench):
    if inspect.isclass(obj) and name.endswith("MSBench"):
        dataset_classes.append(obj)

dataset_classes.sort(key=lambda x: x.__name__)

print(f"Found {len(dataset_classes)} datasets.")
print(f"Data will be saved to: {DOWNLOAD_ROOT}\n")

# 3. Iterate, Download, and Log
for cls in dataset_classes:
    name = cls.__name__
    try:
        print(f"Processing {name}...", end=" ", flush=True)
        
        # Initialize (This triggers download)
        # We assume the library needs the directory to exist already
        train_ds = cls(root=DOWNLOAD_ROOT, split='train', size=IMG_SIZE, download=True)
        val_ds   = cls(root=DOWNLOAD_ROOT, split='val',   size=IMG_SIZE, download=True)
        test_ds  = cls(root=DOWNLOAD_ROOT, split='test',  size=IMG_SIZE, download=True)
        
        total_images = len(train_ds) + len(val_ds) + len(test_ds)
        
        # Simple modality heuristic
        modality = "Other"
        n_lower = name.lower()
        if any(x in n_lower for x in ['us', 'busi', 'nerve', 'fhps']): modality = "Ultrasound"
        elif any(x in n_lower for x in ['kvasir', 'polyp', 'bkai', 'm2cai', 'robotool']): modality = "Endoscopy/Optical"
        elif any(x in n_lower for x in ['isic', 'skin', 'derm']): modality = "Dermatology"
        elif any(x in n_lower for x in ['nuclei', 'cell', 'wbc', 'yeaz', 'bacs', 'monusac']): modality = "Microscopy"
        elif any(x in n_lower for x in ['covid', 'radio', 'mosmed', 'pan', 'xray']): modality = "X-Ray/CT"
        elif any(x in n_lower for x in ['drive', 'chase', 'idrib', 'chuac', 'dca1', 'cysto']): modality = "Retina/OCT"
        elif any(x in n_lower for x in ['promise', 'mri']): modality = "MRI"

        stats_list.append({
            "Dataset": name,
            "Modality_Guess": modality,
            "Total_Images": total_images,
            "Train_Size": len(train_ds),
            "Test_Size": len(test_ds),
            "Num_Classes": train_ds.num_classes if hasattr(train_ds, 'num_classes') else "Unknown",
            "Resolution": f"{IMG_SIZE}x{IMG_SIZE}",
            "Status": "OK"
        })
        print(f"Done! ({total_images} imgs)")

    except Exception as e:
        print(f"\n[!] SKIPPED {name}: {e}")
        # Append minimal info so we can see what failed in the CSV
        stats_list.append({
            "Dataset": name,
            "Modality_Guess": "Error",
            "Total_Images": 0,
            "Status": "Error",
            "Error_Msg": str(e)
        })

# 4. Save safely
df = pd.DataFrame(stats_list)

csv_save_path = os.path.join(DOWNLOAD_ROOT, "medsegbench_stats.csv")
df.to_csv(csv_save_path, index=False)
print(f"\nCompleted! Stats saved to {csv_save_path}")

# Only try to print the summary if we actually have data
if not df.empty and 'Total_Images' in df.columns:
    # Sort for better viewing
    df = df.sort_values(by='Total_Images', ascending=False)
    
    # Filter out errors for the console display
    success_df = df[df['Status'] == 'OK']
    if not success_df.empty:
        print("\nSuccessful Downloads:")
        print(success_df[["Dataset", "Modality_Guess", "Total_Images"]].head(10).to_markdown(index=False))
    else:
        print("\nNo datasets were downloaded successfully. Check 'Error_Msg' in the CSV.")
else:
    print("DataFrame is empty or malformed.")