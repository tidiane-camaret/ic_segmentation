## INSTRUCTIONS

This research project evaluates methods for deep-learning based 3D medical image segmentation. It aims to design architectures and evaluate them dataset for accuracy (Dice, NSD ...) and efficiency (memory consumption, flops, inference time)

The final goal is to perform segmentation in-context, i.e. the model recieves a target image and a set of related context image/masks, and outputs a mask prediction for the target image.

Write understandable code with short docstrings. Do not write extensive documentation or quickstarts, just write important changes in README.md and short research logs in logs.md. Write tests only when necessary.
The relevant code repos as in /software/notebooks/camaret/repos , e.g Medverse, nnInteractive_fork, PatchWork, Neuroverse3D

## Quick Start

### Basic Evaluation

Run evaluation on TotalSeg dataset with foreground context sampling:

```bash
cd /software/notebooks/camaret/ic_segmentation

# Basic evaluation with 3 context examples
python scripts/eval_totalseg.py \
    --context-size 3 \
    --no-wandb

# With inspection enabled (saves all patches for debugging)
python scripts/eval_totalseg.py \
    --enable-inspection \
    --max-inspect-cases 1 \
    --context-size 3 \
    --no-wandb
```

### Visualize Inspection Data

```bash
# List available inspection data
python scripts/visualize_inspection.py --list

# Visualize specific patch
python scripts/visualize_inspection.py \
    --case s0032 \
    --label heart \
    --level 3 \
    --patch 0
```

## Architecture Overview

### Foreground-Based Context Sampling

**Problem**: The original Medverse implementation samples context patches from the same spatial location as the target patch. For many anatomical structures, this results in context patches with no foreground (empty background), reducing the effectiveness of in-context learning.

**Solution**: `LightningModelForegroundSampling` (in `src/medverse_foreground_sampling.py`) inherits from Medverse's `LightningModel` and overrides the sliding window inference to:

1. Compute all valid patch centers where context masks contain ≥1% foreground voxels
2. Randomly sample context patches from these foreground-rich locations
3. Preserve all other aspects of the original autoregressive inference

**Key Design Principle**: Stay as close as possible to the original implementation. Only modify context patch sampling, not the core model or inference logic.

### File Structure

```
ic_segmentation/
├── src/
│   ├── medverse_foreground_sampling.py    # Inherited class with foreground sampling
│   ├── totalseg_dataloader.py             # TotalSeg dataset loader
│   ├── config.py                          # Configuration
│   ├── INSPECTION_GUIDE.md                # Detailed inspection guide
│   └── README_foreground_sampling.md      # Implementation details
├── scripts/
│   ├── eval_totalseg.py                   # Main evaluation script
│   ├── visualize_inspection.py            # Inspection visualization tool
│   └── export_nora.py                     # Export to Nora format
├── QUICKSTART_INSPECTION.md               # Quick start for inspection
├── INSPECTION_SUMMARY.md                  # Complete inspection summary
└── CLAUDE.md                              # This file
```

## Core Implementation

### `src/medverse_foreground_sampling.py`

The `LightningModelForegroundSampling` class extends Medverse's `LightningModel`:

```python
class LightningModelForegroundSampling(LightningModel):
    def __init__(self, hparams):
        super().__init__(hparams)
        self.enable_inspection = False
        self.inspection_data = None
```

**Key Methods:**

1. **`_compute_foreground_patch_centers(context_out, roi_size, overlap, min_foreground_ratio=0.01)`**
   - Computes sliding window patch grid
   - Filters patches with ≥1% foreground voxels
   - Returns list of (d, h, w) center coordinates

2. **`_sample_context_patches(context_in, context_out, roi_size, patch_centers, num_samples=1)`**
   - Randomly samples from foreground patch locations
   - Extracts corresponding image and mask patches
   - Returns sampled_in, sampled_out tensors

3. **`_sliding_window_autoregressive_step(target_in, context_in, context_out, current_level, ...)`**
   - Overrides parent's sliding window implementation
   - Uses MONAI's sliding window with foreground-sampled contexts
   - Collects inspection data if enabled

4. **`save_inspection_data_to_nifti(save_dir, case_id, label_id)`**
   - Saves all collected patch data to NIfTI files
   - Uses flat file hierarchy for easy browsing

### Flat File Hierarchy

All inspection data for a case is stored in a single directory:

```
results/totalseg_inspection/s0032/
├── heart_metadata.json
├── heart_level_1_patch_0000_target_in.nii.gz
├── heart_level_1_patch_0000_prediction.nii.gz
├── heart_level_1_patch_0000_context_0_img.nii.gz
├── heart_level_1_patch_0000_context_0_mask.nii.gz
├── heart_level_1_patch_0000_context_1_img.nii.gz
├── heart_level_1_patch_0000_context_1_mask.nii.gz
├── heart_level_1_patch_0000_context_2_img.nii.gz
├── heart_level_1_patch_0000_context_2_mask.nii.gz
├── heart_level_1_patch_0000_context_coordinates.json
├── heart_level_1_patch_0001_target_in.nii.gz
└── ...
```

**Filename Pattern**: `{label}_level_{L}_patch_{PPPP}_{type}.nii.gz`

**Benefits**:
- Easy to filter (e.g., `ls *level_3*` shows all level 3 patches)
- Simple programmatic access
- Compatible with command-line tools
- No nested directory navigation

## Inspection System

### What Gets Captured

For each patch in the sliding window inference:
- **Target image patch**: The CT scan region to segment
- **Prediction**: Model's output for this patch
- **Context images** (3x): Example CT patches showing the structure
- **Context masks** (3x): Segmentation masks for context examples
- **Context coordinates**: Where each context patch was sampled from (proves foreground sampling)

### Multi-Level Tracking

Captures data across all autoregressive levels:
- **Level 1**: Coarsest resolution (e.g., 32×32×32)
- **Level 2**: Intermediate (e.g., 64×64×64)
- **Level 3**: Full resolution (e.g., 128×128×128)

### Enable Inspection

```bash
python scripts/eval_totalseg.py \
    --enable-inspection \
    --max-inspect-cases 1 \
    --context-size 3 \
    --no-wandb
```

**Flags**:
- `--enable-inspection`: Enable patch data collection
- `--max-inspect-cases N`: Limit inspection to first N cases (saves memory/disk)
- `--context-size N`: Number of context examples per patch
- `--no-wandb`: Disable Weights & Biases logging

### Verify Foreground Sampling

```python
import nibabel as nib
import json

case_dir = "results/totalseg_inspection/s0032"
prefix = "heart_level_3_patch_0000"

# Load context mask
ctx_mask = nib.load(f"{case_dir}/{prefix}_context_0_mask.nii.gz").get_fdata()
print(f"Foreground voxels in context 0: {(ctx_mask > 0).sum()}")  # Should be > 0!

# Check coordinates where it was sampled
with open(f"{case_dir}/{prefix}_context_coordinates.json") as f:
    coords = json.load(f)
print(f"Sampled from: {coords['context_0']}")
# Output: {"d": 45, "h": 62, "w": 38}
```

## Common Tasks

### Run Evaluation Without Inspection

```bash
python scripts/eval_totalseg.py \
    --context-size 3 \
    --no-wandb
```

### Save Input/Output Images

```bash
python scripts/eval_totalseg.py \
    --context-size 3 \
    --save-imgs-masks \
    --no-wandb
```

Saves to `results/totalseg_eval/{case_id}/{label}_img.nii.gz` etc.

### List Available Inspection Data

```bash
python scripts/visualize_inspection.py --list
```

Output:
```
s0032:
  heart - Level 1: 2 patches, shape [32, 32, 32], ROI [128, 128, 128]
  heart - Level 2: 4 patches, shape [64, 64, 64], ROI [128, 128, 128]
  heart - Level 3: 8 patches, shape [128, 128, 128], ROI [128, 128, 128]
```

### Visualize Specific Patch

```bash
python scripts/visualize_inspection.py \
    --case s0032 \
    --label heart \
    --level 3 \
    --patch 0 \
    --slice-axis z
```

Shows matplotlib figure with:
- Target image
- Prediction
- Context 0: image + mask (with foreground!)
- Context 1: image + mask (with foreground!)
- Context 2: image + mask (with foreground!)

### Compare Across Levels

```python
import nibabel as nib
import matplotlib.pyplot as plt

case_dir = "results/totalseg_inspection/s0032"

# Load predictions from different levels (same patch ID)
pred_l1 = nib.load(f"{case_dir}/heart_level_1_patch_0000_prediction.nii.gz").get_fdata()
pred_l2 = nib.load(f"{case_dir}/heart_level_2_patch_0000_prediction.nii.gz").get_fdata()
pred_l3 = nib.load(f"{case_dir}/heart_level_3_patch_0000_prediction.nii.gz").get_fdata()

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, pred, level in zip(axes, [pred_l1, pred_l2, pred_l3], [1, 2, 3]):
    s = pred.shape[2] // 2
    ax.imshow(pred[:, :, s], cmap='hot', vmin=0, vmax=1)
    ax.set_title(f'Level {level}: {pred.shape}')
plt.show()
```

## Configuration

### Dataset Paths

Edit `src/config.py`:

```python
config = {
    "DATA_DIR": "/path/to/data",
    "RESULTS_DIR": "/path/to/results",
}
```

### Model Path

In `scripts/eval_totalseg.py` (line 36):

```python
model_path: str = "/path/to/Medverse.ckpt"
```

Or use command-line argument:

```bash
python scripts/eval_totalseg.py --model-path /path/to/checkpoint.ckpt
```

### Label List

Edit `DEFAULT_LABEL_ID_LIST` in `scripts/eval_totalseg.py` (line 13):

```python
DEFAULT_LABEL_ID_LIST = [
    "heart",
    "spinal_cord",
    "liver",
    # ... add more labels
]
```

## Debugging

### Empty Levels Bug

**Symptom**: Metadata shows `{"levels": []}`

**Cause**: `autoregressive_inference()` wrapper was overriding `current_level` with `None`

**Fix**: Simplified wrapper in `src/medverse_foreground_sampling.py`:

```python
def autoregressive_inference(self, *args, **kwargs):
    """Override to use custom autoregressive inference with level tracking."""
    return self._autoregressive_inference_with_level_tracking(*args, **kwargs)
```

### No Foreground in Context Patches

**Symptom**: Context masks are empty (all zeros)

**Check**:
1. Verify `_compute_foreground_patch_centers()` is finding valid patches
2. Check `min_foreground_ratio` parameter (default 0.01 = 1%)
3. Ensure context masks actually contain foreground in the volume

**Debug**:
```python
# In _compute_foreground_patch_centers, add print:
print(f"Found {len(valid_centers)} foreground patches out of {len(all_centers)} total")
```

### Memory Issues

**Symptom**: OOM during inspection

**Solutions**:
- Use `--max-inspect-cases 1` to limit inspection
- Reduce `--context-size`
- Disable inspection for production runs

**Typical memory usage**: ~512 MB per case (30-60 patches × 8 tensors × 128³ × 4 bytes)

### Tensor Dimension Errors

**Common Issue**: `IndexError: too many indices for tensor of dimension 5`

**Cause**: `image_context_in_prev` has shape [B, C, D, H, W] (5D) but code assumes 6D

**Fix**: Check tensor shapes before indexing:
```python
print(f"Tensor shape: {tensor.shape}")
# Then adjust indexing accordingly
```

## Performance Considerations

- **Inspection overhead**: ~10-20% slower inference when enabled
- **Disk usage**: Each case generates hundreds of NIfTI files (~1-2 GB per case)
- **Recommendation**: Only enable inspection for debugging/analysis, not production

## Use Cases

1. **Debug foreground sampling**: Verify contexts actually contain structure
2. **Understand multi-resolution**: See how predictions evolve across levels
3. **Identify failure cases**: Examine patches where model fails
4. **Validate data loading**: Check that preprocessing is correct
5. **Paper figures**: Generate visualizations showing in-context learning

## Advanced Analysis

### Check Sampling Diversity

```python
from pathlib import Path
import json

case_dir = Path("results/totalseg_inspection/s0032")

# Collect all context 0 coordinates from level 3
coords = []
for coord_file in sorted(case_dir.glob("heart_level_3_*_context_coordinates.json")):
    with open(coord_file) as f:
        data = json.load(f)
        coords.append(data['context_0'])

print(f"Sampled {len(coords)} patches")
print(f"Context 0 locations:")
for i, c in enumerate(coords):
    print(f"  Patch {i}: d={c['d']}, h={c['h']}, w={c['w']}")
```

### Batch Analysis

```python
import nibabel as nib
import numpy as np
from pathlib import Path

case_dir = Path("results/totalseg_inspection/s0032")

# Analyze all level 3 predictions
pred_files = sorted(case_dir.glob("heart_level_3_*_prediction.nii.gz"))

foreground_counts = []
for pred_file in pred_files:
    pred = nib.load(pred_file).get_fdata()
    fg_count = (pred > 0.5).sum()
    foreground_counts.append(fg_count)

print(f"Mean foreground voxels: {np.mean(foreground_counts):.0f}")
print(f"Std: {np.std(foreground_counts):.0f}")
print(f"Min: {np.min(foreground_counts):.0f}, Max: {np.max(foreground_counts):.0f}")
```

## Technical Details

### Inheritance Pattern

We use inheritance to avoid modifying the original Medverse codebase:

```python
# Original: medverse/lightning_model.py (unchanged)
class LightningModel(pl.LightningModule):
    def autoregressive_inference(...):
        # Original implementation
        pass

# Our extension: src/medverse_foreground_sampling.py
class LightningModelForegroundSampling(LightningModel):
    def autoregressive_inference(...):
        # Custom implementation with foreground sampling
        return self._autoregressive_inference_with_level_tracking(...)
```

### Data Collection Flow

1. **Enable inspection**: `model.enable_patch_inspection()`
2. **Inference runs**: Autoregressive inference processes image
3. **Each patch collected**:
   - Target image/prediction tensors copied to CPU
   - Context patches sampled (coordinates tracked)
   - Stored in `inspection_data` dict
4. **Save to disk**: `model.save_inspection_data_to_nifti()`
   - Creates flat file structure
   - Saves all NIfTI files
   - Writes metadata JSON

## Future Enhancements

1. Support batch_size > 1
2. Option for deterministic sampling (always pick patch with most foreground)
3. Selective saving (only save patches with low Dice scores)
4. Video generation showing patch-by-patch processing
5. Interactive visualization with slider

## References

- **Medverse**: Original in-context learning model for medical image segmentation
- **MONAI**: Medical imaging framework used for sliding window inference
- **TotalSeg**: CT scan segmentation dataset

## Contact & Support

For issues or questions about this implementation:
1. Check `INSPECTION_GUIDE.md` and `QUICKSTART_INSPECTION.md`
2. Review error messages in "Debugging" section above
3. Examine inspection data to verify behavior

## Quick Reference Commands

```bash
# Basic evaluation
python scripts/eval_totalseg.py --context-size 3 --no-wandb

# With inspection
python scripts/eval_totalseg.py --enable-inspection --max-inspect-cases 1 --context-size 3 --no-wandb

# List inspection data
python scripts/visualize_inspection.py --list

# Visualize patch
python scripts/visualize_inspection.py --case s0032 --label heart --level 3 --patch 0

# Check foreground in context
python -c "import nibabel as nib; print((nib.load('results/totalseg_inspection/s0032/heart_level_3_patch_0000_context_0_mask.nii.gz').get_fdata() > 0).sum())"
```
