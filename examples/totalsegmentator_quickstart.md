# TotalSegmentator DataLoader - Quick Start Guide

## Step 1: Scan Your Dataset

First, scan your dataset to see which organs are available and have non-zero voxels:

```bash
cd /software/notebooks/camaret/repos/Medverse

python medverse/data/totalseg_utils.py \
    --root_dir /path/to/TotalSegmentator \
    --action scan \
    --show_empty
```

This will:
- Check all cases for CT scans and segmentations
- Count non-zero voxels for each organ
- Show statistics including:
  - Number of valid cases
  - Organs sorted by frequency
  - Average voxel counts
  - Empty segmentations (files that exist but are all zeros)

Output example:
```
TotalSegmentator Dataset Statistics
================================================================================

Total cases scanned: 1429
Valid cases (with CT + non-empty segmentations): 1420
Cases missing CT: 5
Cases missing segmentation folder: 4

Total unique organs (with non-zero voxels): 117

Organs sorted by frequency:
Organ Name                                  Cases      %      Avg Voxels
--------------------------------------------------------------------------------
liver                                        1400   98.6%         250,000
kidney_left                                  1395   98.2%          45,000
kidney_right                                 1392   98.0%          43,000
spleen                                       1385   97.5%          80,000
...
```

## Step 2: Choose Organs

### Option A: Use Predefined Groups

```bash
python medverse/data/totalseg_utils.py \
    --root_dir /path/to/TotalSegmentator \
    --action show_groups
```

This shows organ groups like:
- Major Abdominal (liver, spleen, kidneys, etc.)
- Cardiovascular (heart, aorta, etc.)
- Thoracic (lungs)
- etc.

### Option B: Filter by Availability

```bash
python medverse/data/totalseg_utils.py \
    --root_dir /path/to/TotalSegmentator \
    --action filter_group \
    --group "Major Abdominal" \
    --min_cases 1000 \
    --min_voxels 5000
```

This will show which organs in the group meet your criteria.

## Step 3: List Organs in a Specific Case

```bash
python medverse/data/totalseg_utils.py \
    --root_dir /path/to/TotalSegmentator \
    --action list_organs
```

This lists all organs in case `s0000` that have non-zero voxels.

## Step 4: Create Train/Val Split

```bash
python medverse/data/totalseg_utils.py \
    --root_dir /path/to/TotalSegmentator \
    --action create_split \
    --output_dir ./splits
```

This creates:
- `splits/train_split.json`
- `splits/val_split.json`

## Step 5: Create DataLoader in Python

```python
from medverse.data.totalseg_dataloader import get_dataloader

# Define organs (from your scan results)
# Each sample will focus on ONE organ at a time
organ_list = [
    "liver",
    "kidney_left",
    "kidney_right",
    "spleen",
]

# Optional: Load empty_segmentations from scan
# This excludes cases where organs have zero voxels
from medverse.data.totalseg_utils import scan_dataset
stats = scan_dataset("/path/to/TotalSegmentator")
empty_segs = stats['empty_segmentations']

# Create training dataloader
train_loader = get_dataloader(
    root_dir="/path/to/TotalSegmentator",
    organ_list=organ_list,
    empty_segmentations=empty_segs,  # Exclude empty cases!
    context_size=5,              # Use 5 example pairs
    batch_size=1,                # Usually 1 for 3D medical images
    image_size=(128, 128, 128),  # Resize to 128³
    spacing=(1.5, 1.5, 1.5),     # Resample to 1.5mm isotropic
    num_workers=4,
    mode='train',
    shuffle=True,
    random_context=True,         # Randomly sample contexts each epoch
)

# Create validation dataloader
val_loader = get_dataloader(
    root_dir="/path/to/TotalSegmentator",
    organ_list=organ_list,
    empty_segmentations=empty_segs,  # Same exclusions
    context_size=5,
    batch_size=1,
    image_size=(128, 128, 128),
    spacing=(1.5, 1.5, 1.5),
    num_workers=4,
    mode='val',
    shuffle=False,
    random_context=False,        # Use same contexts for reproducibility
)
```

## Step 6: Use with Medverse Model

```python
from medverse.lightning_model import LightningModel
import torch

# Load model
model = LightningModel.load_from_checkpoint("Medverse.ckpt")
model.eval()
model.cuda()

# Get a batch
batch = next(iter(val_loader))

# Extract inputs
target_in = batch['target_in'].cuda()        # [1, 1, 128, 128, 128]
context_in = batch['context_in'].cuda()      # [1, 5, 1, 128, 128, 128]
context_out = batch['context_out'].cuda()    # [1, 5, 1, 128, 128, 128]

# Run inference
with torch.no_grad():
    prediction = model.autoregressive_inference(
        target_in=target_in,
        context_in=context_in,
        context_out=context_out,
        level=None,                    # Auto-calculate levels
        forward_l_arg=3,               # Process 3 contexts at a time
        sw_roi_size=(128, 128, 128),
        sw_overlap=0.25,
        sw_batch_size_val=1,
    )

print(f"Prediction shape: {prediction.shape}")  # [1, 1, 128, 128, 128]
```

## Common Use Cases

### High-Resolution Inference

For larger images (e.g., 256³ or 512³):

```python
prediction = model.autoregressive_inference(
    target_in=target_in,
    context_in=context_in,
    context_out=context_out,
    level=3,                       # Use 3 resolution levels
    sw_roi_size=(128, 128, 128),   # Process in 128³ windows
    sw_overlap=0.5,                # 50% overlap for smooth blending
)
```

### Memory-Efficient Context Processing

For many context examples:

```python
# Use smaller mini-batch size
prediction = model.autoregressive_inference(
    target_in=target_in,
    context_in=context_in,          # e.g., 20 contexts
    context_out=context_out,
    forward_l_arg=2,                # Process only 2 at a time
)
```

### Single Organ Segmentation

```python
organ_list = ["liver"]  # Segment only liver

dataloader = get_dataloader(
    root_dir="/path/to/TotalSegmentator",
    organ_list=organ_list,
    context_size=10,     # Use more contexts for single organ
    ...
)
```

### Multi-Organ Segmentation

```python
# All organs are combined into binary segmentation (label=1)
organ_list = [
    "liver",
    "kidney_left",
    "kidney_right",
    "spleen",
]

dataloader = get_dataloader(
    root_dir="/path/to/TotalSegmentator",
    organ_list=organ_list,
    ...
)
```

The segmentation mask will have **binary values**:
- 0 = background
- 1 = any organ from organ_list

All organs in `organ_list` are combined into a single binary mask.

## Dataset Statistics Interpretation

When you scan the dataset, pay attention to:

1. **Case count**: Organs appearing in >90% of cases are most reliable
2. **Average voxels**: Very small organs (<1000 voxels) may be harder to segment
3. **Empty segmentations**: High number of empty files may indicate:
   - Organ not present in some patients (normal variation)
   - Annotation issues
   - Challenging anatomical regions

## Troubleshooting

### Issue: "No valid cases found"

```python
# Check if organ names are correct
from medverse.data.totalseg_utils import get_available_organs

organs = get_available_organs("/path/to/TotalSegmentator", case_id="s0000")
print(organs)
```

### Issue: Out of memory

```python
# Reduce image size
image_size=(96, 96, 96)  # Instead of (128, 128, 128)

# Or reduce context size
context_size=3  # Instead of 5

# Or use smaller ROI for sliding window
sw_roi_size=(96, 96, 96)
```

### Issue: Slow data loading

```python
# Enable caching for small datasets
from medverse.data.totalseg_dataloader import TotalSegmentatorDataset

dataset = TotalSegmentatorDataset(
    ...,
    cache_rate=0.5,  # Cache 50% of data in memory
)

# Or increase workers
num_workers=8  # Instead of 4
```

## Performance Tips

1. **Start small**: Test with `num_samples=10` first
2. **Cache if possible**: Use `cache_rate > 0` for datasets that fit in RAM
3. **Adjust image size**: 128³ is a good balance, use 96³ for faster iteration
4. **Monitor voxel counts**: Organs with <1000 voxels may need special handling
5. **Check empty segmentations**: Filter out organs with many empty files

## Example: Complete Pipeline

```python
# 1. Scan dataset
from medverse.data.totalseg_utils import scan_dataset, print_dataset_stats

stats = scan_dataset("/path/to/TotalSegmentator")
print_dataset_stats(stats, show_empty=True)

# 2. Choose organs based on statistics
from medverse.data.totalseg_utils import get_organs_by_min_cases

good_organs = get_organs_by_min_cases(
    stats,
    min_cases=90,      # Appears in at least 90 cases
    min_avg_voxels=5000  # At least 5000 voxels on average
)
print(f"Selected organs: {good_organs}")

# 3. Get empty segmentations to exclude problematic cases
empty_segs = stats['empty_segmentations']
print(f"\nEmpty cases to exclude:")
for organ, cases in empty_segs.items():
    if organ in good_organs:
        print(f"  {organ}: {len(cases)} empty cases")

# 4. Create dataloader
from medverse.data.totalseg_dataloader import get_dataloader

dataloader = get_dataloader(
    root_dir="/path/to/TotalSegmentator",
    organ_list=good_organs,
    empty_segmentations=empty_segs,  # Exclude empty cases!
    context_size=5,
    batch_size=1,
    image_size=(128, 128, 128),
)

# 5. Run inference
# ... (see Step 6 above)

# 6. Check which organs are being sampled
for batch in dataloader:
    print(f"Segmenting organ: {batch['organs'][0]}")
    # Model sees context examples with the same organ
    break
```
