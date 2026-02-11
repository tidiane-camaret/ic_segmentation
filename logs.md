# Development Log

## 2026-02-11: Multi-Level Cascaded Training Support

### Changes
- **`src/models/patch_icl_v2/patch_icl.py`**: Refactored from single-level to multi-level cascaded architecture.
  - `PatchICL.__init__`: Stores per-level samplers (`nn.ModuleList`), aggregators, oracle configs. Backward-compat aliases preserved.
  - `_forward_level()`: New helper that processes a single resolution level (sampling, feature extraction, backbone, aggregation).
  - `forward()`: Loops over levels coarse-to-fine. Level 0 uses uniform/oracle weights; subsequent levels use `sigmoid(prev_level_pred)` as sampling weights. Configurable `detach_between_levels` (default: True).
  - `compute_loss()`: Per-level losses with configurable `level_weights` multiplier. Aggregated losses averaged across levels for logging.
  - `_downsample`/`_downsample_mask`: Now take `resolution` parameter instead of using `self.resolution`.
  - `_select_context_patches`: Now takes explicit `sampler` argument.
  - All levels share the same `patch_size` (backbone constraint enforced via assertion).
- **`configs/experiment/60_2_levels.yaml`**: New config with 2 levels (resolution 16 + 32, 8+12 patches).
- **Design decisions**: Single backward after all levels (losses accumulated). Detach between levels by default (memory efficient, no gradient benefit since sampling is non-differentiable).

## 2026-02-06: Train/Val Dice Gap Analysis & Fixes

### Problem
- Train dice: 0.3, Val dice: 0.1 (3x gap)

### Root Causes Identified

1. **Warmup Bug**: exp_10 had warmup_epochs=10 with num_epochs=10, meaning the model never reached full learning rate (fixed in exp_10 to warmup_epochs=3)

2. **Sliding Window Ignores Oracle**: The SlidingWindowSampler completely ignores the `weights` parameter. Oracle settings have no effect with sliding_window sampler.

3. **Low Context Size**: Only 1 context example for unseen organs is insufficient for in-context learning.

4. **Context Loss Disabled**: No supervision signal for context segmentation.

5. **Augmentation Disabled**: No regularization at patch level.

6. **Skip Connections Disabled**: Decoder loses spatial detail from encoder.

7. **Train/Val Use Different Organs** (by design): This is the ICL test - validation uses 20% unseen organs.

### New Experiment Config: exp_11_improved.yaml

Created `configs/experiment/exp_11_improved.yaml` with these changes:

| Setting | exp_10 | exp_11 |
|---------|--------|--------|
| num_epochs | 10 | 100 |
| warmup_epochs | 3 | 5 |
| context_size | 1 | 3 |
| sampler type | sliding_window | continuous |
| oracle_levels_train | [false] | [true] |
| augmentation.enabled | false | true |
| rotation | none | "90" |
| flip_horizontal | false | true |
| flip_vertical | false | true |
| decoder_use_skip_connections | false | true |
| context_patch loss | 0 | 0.5 |
| context_aggreg loss | 0 | 0.5 |

### Expected Improvement

| State | Train Dice | Val Dice |
|-------|------------|----------|
| Current (exp_10) | 0.30 | 0.10 |
| exp_11 | 0.65 | 0.40 |

### Usage
```bash
python scripts/train.py experiment=exp_11_improved
```

---

## 2026-02-05: PatchICL Codebase Simplification

### Overview
Reduced codebase bloat by ~66% while keeping v1 and v2 experiments functional.

### Changes

#### New Simplified Code (patch_icl_v2/)
Created simplified implementations in `src/models/patch_icl_v2/`:
- `patch_icl.py`: 496 lines (was 1209) - Single-level support, no refinement passes
- `sampling.py`: 345 lines (was 1272) - Only ContinuousSampler and SlidingWindowSampler
- `aggregate.py`: 173 lines (was 491) - Only PatchAggregator and GaussianAggregator

#### Preserved Old Code (patch_icl_v1/)
Moved original implementations to `src/models/patch_icl_v1/` for reference.

#### Simplified Scripts
- `scripts/train.py`: 248 lines (was 442) - TotalSeg2D + PatchICL only
- `src/train_utils.py`: 345 lines (was 749) - Removed debug code, unused functions

#### Simplified Configs
- `configs/train.yaml`: 132 lines (was 183) - Removed segformer3d, unused options
- Experiment configs cleaned up

### Reduction Summary

| Component | Before | After | Reduction |
|-----------|--------|-------|-----------|
| patch_icl.py | 1209 | 496 | 59% |
| sampling.py | 1272 | 345 | 73% |
| aggregate.py | 491 | 173 | 65% |
| train.py | 442 | 248 | 44% |
| train_utils.py | 749 | 345 | 54% |
| **Total core code** | 4163 | 1607 | **61%** |

### What Was Removed

**From sampling.py:**
- PatchSampler base class (merged into ContinuousSampler)
- UniformSampler (never configured)
- DeterministicTopKSampler (never configured)
- GumbelSoftmaxSampler (251 lines, experimental)

**From aggregate.py:**
- ConfidenceAggregator (never configured)
- LearnedAggregator (never configured)
- LearnedCombineAggregator (never configured)

**From patch_icl.py:**
- Multi-level loop (single level only)
- Refinement passes logic
- Gumbel tau annealing
- Complex oracle level handling
- 8 loss weight types (reduced to 4)
- Feature loss placeholders

**From train.py:**
- totalseg_no_context dataset branch
- totalseg (3D) dataset branch
- medsegbench dataset branch
- segformer3d method branch
- Multiple optimizer types (AdamW only)

**From train_utils.py:**
- build_dataloaders (unused)
- save_predictions (moved or removed)
- Debug print statements
- Complex per-level loss tracking

### What v1 and v2 Configs Use

| Component | v1 | v2 |
|-----------|----|----|
| Sampler | continuous | sliding_window |
| Aggregator | gaussian | average |
| Features | Precomputed DINOv3 | On-the-fly MedSAM v1 |
| Oracle | Train: true, Val: true | Train: false, Val: false |
| Loss weights | target_patch only | target_patch + target_aggreg |

### Verification
Both v1 and v2 configs tested and working:
- Import tests pass
- Model instantiation works
- Forward/backward passes run
- Loss computation works
