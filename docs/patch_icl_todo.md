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

**TODO**:
- [ ] Train a lightweight coarse segmentation head for the first level
- [ ] Or use uniform sampling initially and let the model learn from mistakes
- [ ] Or implement a separate "proposal network" for initial patch selection

---

### 2. Non-Differentiable Patch Selection

**Location**: `src/models/patch_icl.py:122`

```python
top_indices = torch.multinomial(probs, k_safe, replacement=False)
```

**Problem**: `torch.multinomial` is non-differentiable. Gradients cannot flow through the patch selection process, so the model cannot learn to improve *which* patches to select.

**Impact**: The patch sampling strategy remains fixed (temperature-based softmax over avg-pooled weights). The model only learns to segment given patches, not which patches matter.

**TODO**:
- [ ] Implement Gumbel-softmax for differentiable sampling
- [ ] Or use REINFORCE/policy gradient methods
- [ ] Or use attention-based soft patch weighting instead of hard selection

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

**TODO**:
- [ ] Implement learned aggregation (small CNN or attention over overlapping regions)
- [ ] Use confidence-based weighting (higher confidence patches contribute more)
- [ ] Consider Gaussian weighting from patch centers

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

### 8. Training Data Flow Issues

**Location**: `scripts/train.py:65-82` and `src/dataloaders/medsegbench_dataloader.py`

```python
# In train.py - hardcoded to medsegbench
train_loader = get_dataloader(
    dataset_name=train_config["dataset"],
    ...
)

# In dataloader - random label selection per sample
if len(available_labels) > 0:
    label_id = random.choice(available_labels)
```

**Problems**:
1. **Hardcoded dataloader**: The commented-out `build_dataloaders` is unused; code is tied to MedSegBench
2. **Random label per sample**: Each sample randomly selects which label to segment, creating inconsistent batches
3. **Config mismatch**: `train_batch_size: 64` but context examples require loading multiple images per sample

**TODO**:
- [ ] Generalize dataloader selection based on config
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

### 10. Backbone Coordinate Handling

**Location**: `src/models/patch_icl.py:329` and `src/models/local.py:243-244`

```python
# In patch_icl.py
coords_for_backbone = coords.float() * original_coords_scale

# In LocalDino.compute_rope_embeddings
token_coords = token_coords / self.image_size  # Normalize to [0,1]
token_coords = 2.0 * token_coords - 1.0  # Map to [-1,+1]
```

**Problem**: Coordinate scaling assumes coordinates fit within `image_size`. If `original_coords_scale` produces coordinates > `image_size`, the RoPE normalization breaks (values exceed [-1, +1]).

**TODO**:
- [ ] Add coordinate clamping or proper normalization
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

### 12. Validation Uses Oracle Guidance

**Location**: `src/train_utils.py:265`

```python
# Pass labels for oracle global branch (for now - later replace with learned global)
outputs = model(images, labels=labels, context_in=context_in, context_out=context_out, mode="test")
```

**Problem**: Validation passes GT labels to the model, enabling oracle patch sampling even during evaluation.

**Impact**: Validation metrics are overly optimistic and don't reflect true inference performance.

**TODO**:
- [ ] Remove labels from validation forward pass
- [ ] Add separate "oracle" vs "realistic" validation modes
- [ ] Track both metrics to measure oracle gap

---

## Summary Table

| Issue | Severity | Category | Status |
|-------|----------|----------|--------|
| Oracle dependency at first level | **Critical** | Train/test gap | TODO |
| Non-differentiable patch selection | **High** | Architecture | TODO |
| Validation uses GT labels | **High** | Evaluation | TODO |
| 2D-only (project needs 3D) | **High** | Scope | TODO |
| Single level active in config | **Medium** | Configuration | TODO |
| Naive aggregation | **Medium** | Architecture | TODO |
| Loss design issues | **Medium** | Optimization | TODO |
| Context integration is implicit | **Medium** | Architecture | TODO |
| Zero-padding for missing patches | **Low** | Efficiency | TODO |
| Exploration noise train/test gap | **Low** | Train/test gap | TODO |
| Coordinate handling edge cases | **Low** | Robustness | TODO |
| Training data flow issues | **Low** | Code quality | TODO |

---

## Priority Action Items

### Immediate (before next training run)
1. [ ] Fix validation to not use GT labels
2. [ ] Enable multi-level configuration
3. [ ] Add loss weighting configuration

### Short-term (next iteration)
4. [ ] Implement non-oracle first-level sampling
5. [ ] Improve aggregation mechanism
6. [ ] Add proper padding masking

### Medium-term (architecture improvements)
7. [ ] Implement differentiable patch selection
8. [ ] Add explicit context cross-attention
9. [ ] Extend to 3D

---

*Analysis generated: 2026-01-20*
