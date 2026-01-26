# Research Logs

## 2026-01-11: Improved Context Sampling with PatchWork Method

**Implemented label-balanced context patch sampling:**
- Replaced threshold-based foreground sampling with PatchWork's probability-weighted method
- Samples from all valid patch positions (not just sliding window grid), enabling much denser sampling
- Uses formula `background_p = (1 - ratio) * pos / (numvx * ratio - pos)` to achieve 50% foreground/background ratio
- Fixed coordinate tracking - now saves actual sampled positions instead of (0,0,0)

**Extended inspection system:**
- Added saving of image context inputs (`prev_lvl_img.nii.gz`, `prev_lvl_pred_mask.nii.gz`) - predictions from previous autoregressive levels
- Added full stitched predictions for each level (`{label}_level_{L}_full_pred_mask.nii.gz`)
- Allows visualization of prediction refinement from coarse to fine resolution

**Files modified:**
- `src/medverse_foreground_sampling.py`: Implemented `_sample_patch_center_balanced()`, removed old `_compute_foreground_patch_centers()` and `_sample_context_patches()`

**Testing:** Run with `python scripts/eval_totalseg.py --enable-inspection --max-inspect-cases 1 --context-size 3 --no-wandb`

## 2026-01-16: Token-based NMSW Architecture

**Goal:** Replace sliding window inference with selective patch sampling to reduce compute.

**Analyzed NMSW (No More Sliding Window)** from `/software/notebooks/camaret/repos/open_nmsw`:
- Global branch produces coarse prediction + objectness scores
- Gumbel top-k selects important patches differentiably
- Local branch processes only selected patches
- Aggregation combines global + local predictions

**Initial implementation:** Standard NMSW with SegFormer3D
- Created `src/nmsw_sampling.py` (GumbelTopK, PatchSampler, PatchExtractor)
- Created `src/nmsw_aggregation.py` (PatchAggregator with Gaussian weighting)
- Created `src/nmsw_segformer.py` (NMSWSegFormer3D wrapper)

**Redesigned to Token-based approach:** User insight - patches should be very small (8³) and attend to each other via transformer.

**Final architecture (TokenNMSW):**
```
Input [128³] → GlobalBranch (UNet) → Objectness scores
     ↓
Extract all 8³ patches → Gumbel top-k selects 200 patches
     ↓
PatchTokenizer (CNN: 8³→4³→2³→1³→embed) → [B, 200, 512]
     ↓
+ Positional encoding (3 options: sinusoidal/learnable/relative)
     ↓
CrossPatchTransformer (12 layers, 8 heads) → patches attend to each other
     ↓
PatchDecoder (CNN: 1³→8³) → [B, 200, 1, 8, 8, 8]
     ↓
Aggregate → Final prediction [128³]
```

**Key design choices:**
- Patch size: 8³ (fine granularity, ~4000 patches for 128³ volume)
- Select 200 patches via Gumbel-softmax (differentiable)
- Full transformer enables cross-patch communication
- 3 positional encoding options for ablation

**Files created:**
- `src/token_nmsw.py`: TokenNMSW, PatchTokenizer, CrossPatchTransformer, PatchDecoder, GlobalBranch, 3 positional encodings
- `scripts/train_token_nmsw.py`: Training script with wandb logging

**GPU optimization (49GB available, was using 1.8GB):**
- `train_batch_size`: 2 → 16
- `num_patches`: 100 → 200
- `embed_dim`: 256 → 512
- `num_layers`: 8 → 12

**Config:** All parameters in `config.yaml` under `train_totalseg.token_nmsw`

**Run:** `python scripts/train_token_nmsw.py` or `python scripts/train_token_nmsw.py --pos-encoding relative`

## 2026-01-21: PatchICL Training Pipeline Bug Fixes

**Fixed critical bugs preventing proper training:**

### Config Fixes
- `embed_dim`: 768 → 1024 (DINOv3 ViT-L produces 1024-dim features, not 768)
- `val_batch_size`: Was using `train_batch_size` for validation loader

### Training Pipeline
- **Features not passed to model**: `train_utils.py` extracted `target_features` and `context_features` from batch but didn't pass them to `model()`. Now properly passed.
- **Warmup scheduler**: Config specified warmup but train.py used plain CosineAnnealingLR. Implemented `LinearLR` warmup with `SequentialLR` to chain warmup → main scheduler.

### Position Embedding / Coordinate Bugs
- **Scale mismatch**: Patch coordinates were in 512 space but backbone assumed 224. Added `actual_image_size` parameter to `PrecomputedFeatureBackbone` and `PrecomputedDinoBackbone` for correct scaling.
- **Output size mismatch**: `SegmentationHead` output was `tokens_h × 16` but needed `patch_size`. Added `target_size` parameter with bilinear resize.

### Train/Test Distribution Mismatch
- **Oracle sampling**: Training used GT mask for patch sampling but validation had no alternative. Added separate configs:
  - `oracle_levels_train: [true]` - GT-guided sampling during training
  - `oracle_levels_valid: [false]` - Uniform sampling during validation
- PatchICL.forward() now selects oracle based on `self.training` mode

### Performance
- **Vectorized feature extraction**: `extract_patch_features()` used Python loops over B×K patches. Replaced with tensor operations (batch indexing).

**Files modified:**
- `config.yaml`: embed_dim, oracle_levels_train/valid
- `scripts/train.py`: val_batch_size, warmup scheduler
- `src/train_utils.py`: Pass features to model
- `src/models/backbone.py`: actual_image_size, target_size
- `src/models/patch_icl.py`: Separate train/valid oracle, vectorized extraction

**Training now runs successfully.**

## 2026-01-26: CrossPatchAttentionBackbone Improvements

### 2D RoPE (Rotary Position Embeddings)

**Problem:** Original RoPE used sequential patch indices (0, 1, 2, ...) regardless of actual spatial location. Patches at the same (x, y) position but different sequence positions got different embeddings.

**Solution:** Implemented 2D RoPE that uses actual patch coordinates:
- `build_rope_cache_2d(max_pos, dim)`: Precomputes sin/cos for spatial positions
- `apply_rope_2d(x, coords, rope_cache, image_size)`: Applies 2D rotations based on coordinates
  - First half of embedding rotated by x-coordinate
  - Second half rotated by y-coordinate
  - Coordinates normalized to [0, max_pos) range

**Usage:** Enabled by default (`use_rope_2d=True`). Falls back to 1D RoPE if `coords=None`.

```python
# In forward():
if self.use_rope_2d and coords is not None:
    combined = apply_rope_2d(combined, coords, self.rope_cache, self.image_size)
else:
    combined = apply_rope(combined, self.rope_cache)
```

### Spatial Feature Grid for Segmentation Head

**Problem:** After cross-patch attention, features were flattened `[B, K, nb_features*D]`. The `SegmentationHead` started from 1×1 spatial and upsampled, losing the inherent 7×7 spatial structure of DINO features.

**Solution:** Reshape features to spatial grid before segmentation:
```
Attention output: [B, K, 2548]         # nb_features * D = 49 * 52
Reshaped:         [B, K, 49, 52]       # Separate spatial and channel dims
Spatial:          [B, K, 52, 7, 7]     # Permute to [B, K, D, h, w]
```

**SegmentationHead changes:**
- New parameter `feature_grid_size` (default 7 for 7×7 = 49 features)
- Input changed from `[B, K, embed_dim]` to `[B, K, D, h, w]`
- Dynamic upsampling: calculates blocks needed for `patch_size / feature_grid_size`
- Preserves spatial structure throughout decoding

**Files modified:**
- `src/models/backbone.py`:
  - Added `build_rope_cache_2d()`, `apply_rope_2d()`
  - `CrossPatchAttentionBackbone`: Added `use_rope_2d` param, `feature_grid_size` computation
  - `SegmentationHead`: Accepts spatial input `[B, K, D, h, w]`
