# Foreground-Based Context Patch Sampling

## Overview

This module extends the Medverse LightningModel with foreground-based context patch sampling for improved in-context segmentation learning.

## Key Modification

**Original behavior:** During sliding window inference, context patches are extracted from the same spatial location as the target patch.

**New behavior:** Context patches are randomly sampled from locations where the context segmentation masks have foreground (≥1% foreground voxels).

## Architecture

### Class Hierarchy

```
LightningModel (from medverse.lightning_model)
    └── LightningModelForegroundSampling (src/medverse_foreground_sampling.py)
            └── Overrides: _sliding_window_autoregressive_step()
            └── Adds: _compute_foreground_patch_centers()
            └── Adds: _sample_context_patches()
```

### File Structure

```
ic_segmentation/
├── src/
│   ├── medverse_foreground_sampling.py   # Extended LightningModel class
│   └── README_foreground_sampling.md     # This file
└── scripts/
    └── eval_totalseg.py                  # Evaluation script (uses extended class)
```

## Implementation Details

### 1. `_compute_foreground_patch_centers()`

Computes all valid patch locations where a context mask has foreground.

**Parameters:**
- `context_out`: [C, D, H, W] - Single context segmentation mask
- `roi_size`: (D_roi, H_roi, W_roi) - Patch size
- `overlap`: Overlap ratio for patch sampling
- `min_foreground_ratio`: Minimum ratio of foreground voxels (default 0.01 = 1%)

**Returns:**
- List of (d, h, w) patch top-left corner coordinates with foreground

**Behavior:**
- Iterates over grid with stride based on overlap
- Counts foreground voxels in each patch
- Keeps patches with ≥ threshold foreground
- Falls back to all patches if no foreground found

### 2. `_sample_context_patches()`

Samples patches from context images at foreground locations.

**Parameters:**
- `context_in`: [L, C, D, H, W] - Context images
- `context_out`: [L, C, D, H, W] - Context masks
- `roi_size`: (D_roi, H_roi, W_roi) - Patch size
- `patch_centers`: List of lists of (d, h, w) centers for each context
- `num_samples`: Number of patches to sample from each context (default 1)

**Returns:**
- Tuple of (sampled_context_in, sampled_context_out), each [L, C, D_roi, H_roi, W_roi]

**Behavior:**
- For each context example, randomly selects one foreground patch
- Extracts corresponding image and mask patches
- Returns stacked samples

### 3. `_sliding_window_autoregressive_step()` (Override)

Modified sliding window inference with foreground-based context sampling.

**Key Changes:**
1. **Pre-processing** (lines 251-264):
   - Computes foreground patch centers for each context mask once
   - Only enabled when batch_size=1 (current limitation)

2. **Context handling** (lines 283-296):
   - Context tensors NOT stacked into input (different from original)
   - Passed to predictor via closure instead

3. **Inside predictor** (lines 332-349):
   - Target patch: extracted from same location (unchanged)
   - Context patches: sampled from foreground locations (NEW)
   - Image context: extracted from same location (unchanged, for autoregressive)

4. **MONAI sliding window** (lines 388-402):
   - Uses original MONAI `sliding_window_inference` (unchanged)
   - All blending, padding, overlap logic preserved

## Benefits

1. **Better in-context learning**: Context examples always show relevant anatomy with the structure of interest
2. **Robust to misalignment**: Works even when target and context have different anatomy at the same spatial location
3. **Minimal code changes**: Only overrides one method, keeps all other functionality
4. **No Medverse modifications**: Original Medverse code remains untouched

## Limitations

1. **Batch size restriction**: Currently only supports `batch_size=1` during inference
2. **Random sampling**: Different context patches each run (introduces variability)
3. **Compute overhead**: Small overhead for computing foreground centers per context

## Usage

### In eval_totalseg.py

```python
from src.medverse_foreground_sampling import LightningModelForegroundSampling

# Load model with foreground sampling
model = LightningModelForegroundSampling.load_from_checkpoint(
    model_path,
    map_location=device
).to(device).eval()

# Use as normal
prediction = model.autoregressive_inference(
    target_in=target_in,
    context_in=context_in,
    context_out=context_out,
    level=None,
    forward_l_arg=3,
    sw_roi_size=(128, 128, 128),
    sw_overlap=0.25,
    sw_batch_size_val=1,
)
```

### To use original behavior

Simply use the original LightningModel:

```python
from medverse.lightning_model import LightningModel

model = LightningModel.load_from_checkpoint(
    model_path,
    map_location=device
).to(device).eval()
```

## Testing

Run the evaluation:

```bash
cd /software/notebooks/camaret/ic_segmentation
python scripts/eval_totalseg.py --context-size 3 --save-imgs-masks
```

Unit tests:

```bash
python test_foreground_sampling.py
```

## Expected Results

- **Dice scores**: Should be comparable or better than original (context patches now informative)
- **Inference time**: Slightly slower due to foreground computation overhead
- **Variability**: Higher between runs due to random sampling (can set seed for reproducibility)

## Future Improvements

1. Support batch_size > 1
2. Option to use deterministic sampling (e.g., select patch with most foreground)
3. Option to sample multiple patches per context and ensemble predictions
4. Adaptive foreground threshold based on total foreground volume

## Author

Implementation created for TotalSeg in-context segmentation evaluation (2026-01-10)
