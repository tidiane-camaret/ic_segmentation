# PatchICL Training Workflow Analysis

## Overview

PatchICL is a multi-resolution in-context learning architecture for medical image segmentation. It processes images through multiple resolution levels (coarse-to-fine), sampling patches at each level based on weight maps from previous predictions, and uses a DINO-based backbone for feature extraction.

## Architecture Summary

```
Input Image → [Level 1 (64×64)] → [Level 2 (128×128)] → [Level 3 (224×224)] → Final Prediction
                    ↓                      ↓                      ↓
               Patch Selection        Patch Selection        Patch Selection
               (weighted by GT)    (weighted by L1 pred) (weighted by L2 pred)
                    ↓                      ↓                      ↓
               DINO Backbone          DINO Backbone          DINO Backbone
                    ↓                      ↓                      ↓
               Aggregation            Aggregation            Aggregation
```

---

## Identified Issues and Potential Flaws

### 1. Oracle Dependency at First Level (Critical)

**Location**: `src/models/patch_icl.py:314-319`

```python
if prev_pred is not None:
    weights = self.downsample(prev_pred)
elif labels is not None:
    weights = labels_ds  # <-- Uses GT mask as sampling weights!
else:
    weights = torch.ones(...)
```

**Problem**: During training, the first level uses the **ground truth mask** as weights for patch sampling. This creates a significant train/test distribution shift:
- **Training**: Patches are perfectly sampled from foreground regions
- **Inference**: Without GT, the model must rely on uniform weights or an initial prediction

**Impact**: The model never learns to find relevant regions from scratch - it's always guided by GT.

**Update**: Added separate `oracle_levels_train` and `oracle_levels_valid` configs to control oracle per-level and per-mode. This allows GT-guided sampling during training but uniform sampling during validation to avoid train/test mismatch.

```yaml
oracle_levels_train: [true]   # Training: use GT mask for focused sampling
oracle_levels_valid: [false]  # Validation: uniform sampling (realistic test scenario)
```

**TODO**:
- [x] Add per-level oracle config (`oracle_levels` in config.yaml)
- [x] Separate oracle settings for train vs validation (`oracle_levels_train`, `oracle_levels_valid`)
- [ ] Train a lightweight coarse segmentation head for the first level
- [ ] Or implement a separate "proposal network" for initial patch selection

---

### 2. Non-Differentiable Patch Selection

**Location**: `src/models/sampling.py:PatchSampler.sample_indices()`

```python
indices = torch.multinomial(probs, k_safe, replacement=False)
```

**Problem**: `torch.multinomial` is non-differentiable. Gradients cannot flow through the patch selection process, so the model cannot learn to improve *which* patches to select.

**Impact**: The patch sampling strategy remains fixed (temperature-based softmax over avg-pooled weights). The model only learns to segment given patches, not which patches matter.

**Update**: Patch sampling has been refactored into a modular `PatchSampler` class in `src/models/sampling.py`. Three sampler types available:
- `weighted` (default): Temperature-scaled softmax + multinomial sampling
- `uniform`: Ignores weights, samples uniformly at random
- `topk`: Deterministic top-K selection (for reproducible inference)

Config in `config.yaml`:
```yaml
sampler:
  type: "weighted"  # Options: "weighted", "uniform", "topk"
  exploration_noise: 0.5
  stride_divisor: 4
```

**TODO**:
- [x] Refactor patch sampling into separate `PatchSampler` class
- [x] Add configurable sampler types (weighted, uniform, topk)
- [x] Implement `GumbelSoftmaxSampler` for differentiable sampling
- [ ] Or use REINFORCE/policy gradient methods (alternative approach)
- [ ] Or use attention-based soft patch weighting instead of hard selection (alternative approach)

---

### 3. Context Integration Architecture

**Location**: `src/models/patch_icl.py:357-367`

```python
# Concatenate target and context patches for joint processing
all_patches = torch.cat([patches, context_patches], dim=1)
all_coords = torch.cat([coords_for_backbone, context_coords_for_backbone], dim=1)

# Process all patches through backbone
all_logits = self.backbone(all_patches, coords=all_coords)

# Split back: target predictions are first K
patch_logits = all_logits[:, :K]  # Only use target logits
```

**Problem**: Context patches influence targets only through transformer self-attention. The context logits are computed but discarded. There's no explicit mechanism to:
- Cross-attend from target to context
- Use context masks as explicit guidance
- Weight context importance

**Impact**: In-context learning relies entirely on implicit attention patterns, which may not be optimal for segmentation tasks.

**TODO**:
- [ ] Add explicit cross-attention layers from target tokens to context tokens
- [ ] Incorporate context masks as additional input (e.g., concatenate with context images)
- [ ] Experiment with prototype-based approaches (average context features as class prototype)

---

### 4. Naive Patch Aggregation

**Location**: `src/models/patch_icl.py:194-221`

```python
# Average overlapping regions
counts = counts.clamp(min=1)
aggregated = output / counts

# Optionally combine with previous prediction
if prev_pred is not None:
    prev_resized = F.interpolate(prev_pred, ...)
    aggregated = (aggregated + prev_resized) / 2  # Simple average
```

**Problems**:
1. Simple averaging of overlapping patches can blur boundaries
2. Equal weighting of current and previous predictions ignores confidence
3. No learned aggregation mechanism

**Impact**: Boundary precision suffers; coarse predictions have equal weight to fine ones.

**Update**: Aggregation has been refactored into a modular `PatchAggregator` class in `src/models/aggregate.py`. Multiple aggregator types available:
- `average` (default): Simple averaging with configurable prev-level combination
- `gaussian`: Gaussian weighting from patch centers (reduces boundary blur)
- `confidence`: Confidence-based weighting (certain predictions contribute more)
- `learned`: Small CNN predicts per-pixel weights
- `learned_combine`: Learned spatially-varying blend with previous level

Config in `config.yaml`:
```yaml
aggregator:
  type: "average"  # Options: "average", "gaussian", "confidence", "learned", "learned_combine"
  combine_mode: "average"  # How to combine with prev level
  combine_weight: 0.5      # Weight for current prediction
  sigma_ratio: 0.3         # For gaussian aggregator
  confidence_temperature: 2.0  # For confidence aggregator
  hidden_dim: 32           # For learned aggregators
```

**TODO**:
- [x] Implement learned aggregation (small CNN or attention over overlapping regions)
- [x] Use confidence-based weighting (higher confidence patches contribute more)
- [x] Consider Gaussian weighting from patch centers
- [ ] Ablate different aggregation strategies on validation set

---

### 5. 2D-Only Implementation

**Location**: Throughout `src/models/patch_icl.py`

The code uses:
- `F.interpolate(..., mode='bilinear')`
- `F.avg_pool2d(...)`
- 2D convolutions in backbone

**Problem**: The project goal (per CLAUDE.md) is 3D medical image segmentation, but PatchICL is 2D-only.

**Impact**: Cannot be directly applied to volumetric CT/MRI data without significant modifications.

**TODO**:
- [ ] Create PatchICL3D variant with trilinear interpolation and 3D pooling
- [ ] Adapt backbone to handle 3D patches (or use 2.5D slice-wise approach)
- [ ] Update dataloader to provide 3D patches

---

### 6. Loss Function Design Issues

**Location**: `src/models/patch_icl.py:563-633`

```python
# Patch-level loss
patch_loss = criterion(
    patch_logits.reshape(B * K, -1),  # Flattens spatial structure
    patch_labels.reshape(B * K, -1),
)

# Level prediction loss
labels_ds = F.interpolate(labels.float(), ..., mode='nearest')  # Lossy for small structures
level_loss = criterion(pred, labels_ds)

# Total loss: unweighted sum
total_loss = total_loss + weight * level_total  # level_total = patch_loss + level_loss
total_loss = total_loss + final_loss  # No relative weighting
```

**Problems**:
1. **Flattening patches** loses spatial structure and treats all pixels equally
2. **Nearest-neighbor downsampling** of labels can lose small anatomical structures
3. **No relative weighting** between patch-level, level-prediction, and final losses
4. **Double counting**: Final prediction loss overlaps with finest level loss

**TODO**:
- [ ] Keep spatial structure in loss computation (don't flatten)
- [ ] Use area-preserving label downsampling or soft labels
- [ ] Add configurable loss weights for different components
- [ ] Consider boundary-focused losses (e.g., boundary dice, HD loss)

---

### 7. Zero-Padding for Missing Patches

**Location**: `src/models/patch_icl.py:154-157`

```python
# Pad if fewer candidates
while len(batch_patches) < K:
    batch_patches.append(torch.zeros(C, ps, ps, device=image.device))
    batch_labels.append(torch.zeros(1, ps, ps, device=image.device))
    batch_coords.append([0, 0])
```

**Problem**: Zero-filled patches with coordinates [0,0] are passed to the backbone and included in loss computation.

**Impact**:
- Wastes computation on meaningless patches
- May confuse the model (zeros at position [0,0])
- Loss includes terms for empty patches

**TODO**:
- [ ] Track valid patch count and mask out padded patches in loss
- [ ] Use attention masking to ignore padded patches in transformer
- [ ] Or dynamically adjust K based on available candidates

---

### 8. Training Data Flow Issues - PARTIAL

**Location**: `scripts/train.py` and `src/train_utils.py`

**Fixed**:
- Features now properly passed to model: `target_features` and `context_features` from batch are sent to model forward()
- Validation uses correct `val_batch_size` instead of `train_batch_size`
- Warmup scheduler properly implemented with LinearLR + SequentialLR

**Remaining Problems**:
1. **Random label per sample**: Each sample randomly selects which label to segment, creating inconsistent batches
2. **Dataloader coupling**: Code assumes specific dataloader structure

**TODO**:
- [x] Pass precomputed features to model (DONE - train_utils.py)
- [x] Fix validation batch size (DONE - train.py)
- [x] Implement warmup scheduler (DONE - train.py)
- [ ] Consider episode-based sampling (consistent label within episode)
- [ ] Verify batch size is feasible with context loading overhead

---

### 9. Single-Level Configuration

**Location**: `config.yaml:113-125`

```yaml
levels:
  - resolution: 64
    patch_size: 16
    num_patches: 16
    sampling_temperature: 0.3
  #- resolution: 128  # COMMENTED OUT
  #- resolution: 224  # COMMENTED OUT
```

**Problem**: Only one level is active, defeating the purpose of multi-resolution processing.

**Impact**: The architecture is effectively single-resolution, losing coarse-to-fine benefits.

**TODO**:
- [ ] Enable multi-level configuration and test coarse-to-fine benefits
- [ ] Ablate number of levels (1 vs 2 vs 3)
- [ ] Tune resolution/patch_size/num_patches per level

---

### 10. Backbone Coordinate Handling - RESOLVED

**Location**: `src/models/backbone.py`

**Problem**: Coordinate scaling assumed coordinates fit within `image_size` (224). But actual image coordinates were in 512 space, causing position embeddings to be incorrectly scaled.

**Solution**: Added `actual_image_size` parameter to backbone methods:
```python
def forward(self, target_patches, context_patches=None, coords=None,
            context_coords=None, actual_image_size=None):
    # Scale coordinates from actual image space to backbone space
    img_size = actual_image_size if actual_image_size is not None else self.image_size
    coord_scale = self.image_size / img_size
    scaled_coords = coords.float() * coord_scale
```

**TODO**:
- [x] Add coordinate clamping or proper normalization (DONE - actual_image_size param)
- [ ] Verify coordinate ranges during training with assertions
- [ ] Document expected coordinate ranges

---

### 11. Exploration Noise Only During Training

**Location**: `src/models/patch_icl.py:114-116`

```python
if self.training:
    noise = torch.rand_like(scaled_scores) * 0.5
    scaled_scores = scaled_scores + noise
```

**Problem**: Exploration noise helps training but creates another train/test gap. The model sees noisy sampling during training but deterministic sampling during inference.

**TODO**:
- [ ] Consider test-time augmentation with noise
- [ ] Or gradually anneal noise during training
- [ ] Or use dropout-style noise that's consistent train/test

---

### 12. Validation Uses Oracle Guidance - RESOLVED

**Location**: `src/train_utils.py` and `src/models/patch_icl.py`

**Problem**: Validation was passing GT labels to the model, enabling oracle patch sampling even during evaluation.

**Solution**:
1. Added `oracle_levels_train` and `oracle_levels_valid` config options
2. PatchICL.forward() selects appropriate oracle based on `self.training` mode
3. Validation now uses uniform sampling by default (`oracle_levels_valid: [false]`)

**TODO**:
- [x] Remove labels from validation forward pass (DONE - `train_utils.py`)
- [x] Add separate "oracle" vs "realistic" validation modes (DONE - separate train/valid oracle configs)
- [ ] Track both metrics to measure oracle gap

---

## Summary Table

| Issue | Severity | Category | Status |
|-------|----------|----------|--------|
| Oracle dependency at first level | **Critical** | Train/test gap | DONE (separate train/valid oracle configs) |
| Non-differentiable patch selection | **High** | Architecture | DONE (GumbelSoftmaxSampler implemented) |
| Validation uses GT labels | **High** | Evaluation | DONE (oracle_levels_valid) |
| Position embedding scale mismatch | **High** | Bug | DONE (actual_image_size param) |
| Output size mismatch | **High** | Bug | DONE (target_size in SegmentationHead) |
| embed_dim mismatch | **High** | Config | DONE (768→1024 for DINOv3 ViT-L) |
| 2D-only (project needs 3D) | **High** | Scope | TODO |
| Single level active in config | **Medium** | Configuration | TODO |
| Naive aggregation | **Medium** | Architecture | DONE (modular PatchAggregator) |
| Loss design issues | **Medium** | Optimization | TODO |
| Context integration is implicit | **Medium** | Architecture | TODO |
| Slow feature extraction | **Medium** | Performance | DONE (vectorized) |
| Zero-padding for missing patches | **Low** | Efficiency | TODO |
| Exploration noise train/test gap | **Low** | Train/test gap | TODO |
| Coordinate handling edge cases | **Low** | Robustness | DONE (actual_image_size) |
| Training data flow issues | **Low** | Code quality | PARTIAL (features now passed) |

---

## Recent Bug Fixes (2026-01-21)

### Config/Setup Fixes
- **embed_dim**: Fixed 768→1024 to match DINOv3 ViT-L feature dimension
- **val_batch_size**: Fixed train.py to use `val_batch_size` instead of `train_batch_size`
- **Warmup scheduler**: Implemented LinearLR warmup + SequentialLR in train.py
- **Features passed to model**: Fixed train_utils.py to pass `target_features` and `context_features`

### Position Embedding / Coordinate Fixes
- **actual_image_size**: Added parameter to backbone.py to correctly scale coordinates
  - Coords in 512 space but backbone assumed 224 → now scales correctly
- **target_size in SegmentationHead**: Output was tokens_h×16, now resizes to actual patch_size

### Performance
- **Vectorized feature extraction**: Replaced Python loops with tensor operations in `extract_patch_features()`

---

## Priority Action Items

### Immediate (before next training run)
1. [x] Fix validation to not use GT labels
2. [x] Add per-level oracle config (`oracle_levels` in config.yaml)
3. [x] Separate train/valid oracle settings (`oracle_levels_train`, `oracle_levels_valid`)
4. [x] Refactor patch sampling into modular `PatchSampler` class
5. [x] Fix embed_dim mismatch (768→1024)
6. [x] Fix position embedding scaling (actual_image_size)
7. [x] Fix SegmentationHead output size (target_size)
8. [x] Implement warmup scheduler
9. [ ] Enable multi-level configuration
10. [ ] Add loss weighting configuration

### Short-term (next iteration)
11. [x] Improve aggregation mechanism (DONE - `src/models/aggregate.py`)
12. [ ] Add proper padding masking
13. [ ] Ablate aggregation strategies

### Medium-term (architecture improvements)
14. [x] Implement `GumbelSoftmaxSampler` for differentiable patch selection
15. [ ] Add explicit context cross-attention
16. [ ] Extend to 3D

---

*Analysis generated: 2026-01-20*
*Updated: 2026-01-20 - Added PatchSampler refactoring, per-level oracle config*
*Updated: 2026-01-20 - Implemented GumbelSoftmaxSampler for differentiable patch selection*
*Updated: 2026-01-20 - Added PatchAugmenter with rotation/flip/scale augmentation*
*Updated: 2026-01-20 - Implemented modular PatchAggregator with gaussian/confidence/learned options*
*Updated: 2026-01-21 - Fixed embed_dim, position embedding, output size, warmup scheduler, feature passing, vectorized extraction*
*Updated: 2026-01-21 - Separated oracle_levels into train/valid configs*
