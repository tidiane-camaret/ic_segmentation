"""Visualize a few samples from the MedSegBench dataloader."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
from src.dataloaders.medsegbench_dataloader import MedSegBenchDataset

DATA_ROOT = "/work/dlclarge2/ndirt-SegFM3D/data/medsegbench"
SAVE_PATH = "experiments/dataloading/medsegbench_samples.png"
NUM_SAMPLES = 5
CONTEXT_SIZE = 3

ds = MedSegBenchDataset(
    data_root=DATA_ROOT,
    split="train",
    context_size=CONTEXT_SIZE,
    image_size=(256, 256),
    max_samples_per_dataset=50,
)

# Columns: target image, target mask, context_1 img, context_1 mask, ...
ncols = 2 + CONTEXT_SIZE * 2
fig, axes = plt.subplots(NUM_SAMPLES, ncols, figsize=(2.5 * ncols, 3 * NUM_SAMPLES))

for row in range(NUM_SAMPLES):
    sample = ds[row * (len(ds) // NUM_SAMPLES)]  # spread across dataset
    img = sample["image"][0].numpy()
    mask = sample["label"][0].numpy()
    ds_name = sample["dataset"]
    lbl = sample["label_value"]

    axes[row, 0].imshow(img, cmap="gray", vmin=0, vmax=1)
    axes[row, 0].set_title(f"{ds_name}\nlabel={lbl}", fontsize=8)
    axes[row, 0].axis("off")

    axes[row, 1].imshow(mask, cmap="gray", vmin=0, vmax=1)
    axes[row, 1].set_title("target mask", fontsize=8)
    axes[row, 1].axis("off")

    if "context_in" in sample:
        for k in range(sample["context_in"].shape[0]):
            ctx_img = sample["context_in"][k, 0].numpy()
            ctx_mask = sample["context_out"][k, 0].numpy()

            axes[row, 2 + k * 2].imshow(ctx_img, cmap="gray", vmin=0, vmax=1)
            axes[row, 2 + k * 2].set_title(f"ctx {k+1} img", fontsize=8)
            axes[row, 2 + k * 2].axis("off")

            axes[row, 2 + k * 2 + 1].imshow(ctx_mask, cmap="gray", vmin=0, vmax=1)
            axes[row, 2 + k * 2 + 1].set_title(f"ctx {k+1} mask", fontsize=8)
            axes[row, 2 + k * 2 + 1].axis("off")

plt.tight_layout()
plt.savefig(SAVE_PATH, dpi=150)
print(f"Saved to {SAVE_PATH}")
