import h5py
from pathlib import Path

root_dir = Path("/work/dlclarge2/ndirt-SegFM3D/data/TotalSeg2D_H5")
h5_files = sorted(list(root_dir.glob("*.h5")))

for h5_file_path in h5_files:
    print(f"Processing file: {h5_file_path.name}")
    try:
        with h5py.File(h5_file_path, 'r') as h5f:
            for label_id in h5f.keys():
                print(f"Label ID: {label_id}")
                for axis in ['z', 'y', 'x']:
                    img_slice = h5f[f"{label_id}/{axis}_slice_img"][:]
                    mask_slice = h5f[f"{label_id}/{axis}_slice"][:]
                    print(f"  Axis: {axis}, Image Slice Shape: {img_slice.shape}, Mask Slice Shape: {mask_slice.shape}")
    except Exception as e:
        print(f"Error processing {h5_file_path.name}: {e}")