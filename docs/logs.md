# Research Logs

## 2026-02-02: On-the-fly Feature Extraction (MedDINO + MedSAM2)

### Overview

Added support for computing features on-the-fly during training/inference, as an alternative to loading precomputed features from disk. Supports two feature extractors:
- **MedDINOv3**: ViT-base model (768-dim features)
- **MedSAM2**: SAM2.1-based Hiera backbone (256-dim features, from HuggingFace wanglab/MedSAM2)

### New Files

**`src/models/meddino_extractor.py`**: Feature extraction module

**MedDINO classes:**
- `MedDINOProcessor`: Preprocesses images (percentile clipping, [0,1] rescale, RGB, ImageNet norm)
- `MedDINOFeatureExtractor`: Wraps MedDINOv3 ViT-base, outputs [B, N, 768]
- `create_meddino_extractor()`: Factory function

**MedSAM2 classes:**
- `MedSAM2Processor`: Preprocesses images (percentile clipping, [0,255] rescale, RGB, SAM norm)
- `MedSAM2FeatureExtractor`: Wraps SAM2.1 Hiera encoder, outputs [B, N, 256]
  - Supports Hiera tiny/small/base+/large variants via config
  - Auto-downloads from HuggingFace (wanglab/MedSAM2) if no local checkpoint
- `create_medsam2_extractor()`: Factory function
- `create_feature_extractor(type, ...)`: Generic factory for either extractor

### Config Changes

New options in `configs/train.yaml`:
```yaml
feature_mode: "precomputed"  # or "on_the_fly"
feature_extractor_type: "meddino"  # or "medsam2"
feature_extraction_resolution: 256  # Use 256 for MedDINO, 1024 for MedSAM2
meddino_layer_idx: 11  # MedDINO: which transformer layer
medsam2_config: "sam2.1_hiera_l.yaml"  # MedSAM2: t/s/b+/l variants
```

### Model Changes

**`src/models/patch_icl.py`**:
- `PatchICL.__init__()` now accepts optional `feature_extractor` parameter
- `PatchICL.set_feature_extractor()` method to set/update extractor post-init
- `PatchICL._extract_features()` internal method for on-the-fly extraction
- `PatchICL.forward()` automatically extracts features if none provided but extractor available

### Training Script Changes

**`scripts/train.py`**:
- Added `feature_mode` and `feature_extractor_type` config parsing
- When `feature_mode="on_the_fly"`:
  - Creates MedDINO or MedSAM2 extractor based on `feature_extractor_type`
  - Passes extractor to PatchICL
  - Dataloader skips loading precomputed `.npz` files
- When `feature_mode="precomputed"` (default): unchanged

### Usage

**Precomputed features (default):**
```bash
python scripts/train.py
```

**On-the-fly MedDINO:**
```bash
python scripts/train.py feature_mode=on_the_fly feature_extractor_type=meddino
```

**On-the-fly MedSAM2:**
```bash
python scripts/train.py feature_mode=on_the_fly feature_extractor_type=medsam2 feature_extraction_resolution=1024
```

### Feature Extractor Comparison

| Aspect | MedDINO | MedSAM2 |
|--------|---------|---------|
| Architecture | ViT-base | Hiera (t/s/b+/l) |
| Feature dim | 768 | 256 |
| Recommended resolution | 256 | 1024 |
| GPU memory | ~350 MB | ~150-900 MB |
| Source | Local checkpoint | HuggingFace |

### Performance Tradeoffs

| Aspect | Precomputed | On-the-fly |
|--------|-------------|------------|
| Training speed | Faster | 10-30% slower |
| Disk storage | ~300KB/image | ~100KB/image |
| GPU memory | Lower | +1.5-2.5 GB |
| Flexibility | Fixed | Can experiment |

### Files Modified

- `src/models/meddino_extractor.py` (NEW): MedDINO + MedSAM2 extractors
- `src/models/patch_icl.py`: Added feature_extractor support
- `scripts/train.py`: Feature mode + extractor type handling
- `configs/train.yaml`: New config options
- `configs/cluster/nfs.yaml`: Added medsam2 checkpoint path option

## 2026-02-01: Eval Script Improvements + Attention Visualization

### Checkpoint Loading in eval.py

Added checkpoint loading before evaluation:
```python
ckpt_path = cfg.paths.ckpts.get(str(cfg.method), None)  # e.g., paths.ckpts.patch_icl
if ckpt_path:
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
```

### Per-Case and Per-Label Dice Logging

Modified `validate()` in `train_utils.py` to track individual dice scores:
- Returns 5-tuple: `(loss, local_dice, final_dice, context_dice, detailed_results)`
- `detailed_results["per_case"]`: List of `{case_id, label_id, axis, dice}` for each sample
- `detailed_results["per_label"]`: Dict of average dice per label_id

Updated `eval.py` to log to wandb:
- Per-label dice as `dice_label/{label_id}` metrics
- Per-case results as a wandb Table with columns `[case_id, label_id, axis, dice]`

### Axis-Based File Naming

Changed output folder naming from `{case_id}_{label_id}_b{batch}_s{sample}` to `{case_id}_{label_id}_{axis}`:
```
Before: s0001_liver_b00_s02/
After:  s0001_liver_z/
```
- `axis` is x, y, or z (the slice plane)
- Extracted from batch via `batch.get("axes")`

### Attention Weights and Register Token Saving

Modified `simple_backbone.py` to optionally return attention internals:

**AttentionBlock.forward:**
- Added `return_attn_weights` parameter
- When True, computes attention manually to capture weights
- Returns `(x, attn_weights)` where `attn_weights` is `[B, H, K, K]`

**CrossPatchAttention.forward:**
- Collects attention from all layers into `all_attn_weights` list
- Captures `register_tokens` after attention, before removal
- Returns `(x, extras)` where `extras = {'attn_weights': [...], 'register_tokens': [B, R, D]}`

**SimpleBackbone.forward:**
- Passes through `return_attn_weights` flag
- Includes `attn_weights` and `register_tokens` in output dict

**PatchICL propagation:**
- `PatchICL_Level.forward`: Added `return_attn_weights` param, captures backbone outputs
- `PatchICL.forward`: Passes flag through, includes in final output

**Saving in train_utils.py:**
- `validate()`: Passes `return_attn_weights=True` only when saving outputs
- `save_predictions()`: Saves per-layer attention as `.npy`:
  - `level{idx}_layer{idx}_attn_weights.npy` - shape `[H, K_total, K_total]`
  - `level{idx}_register_tokens.npy` - shape `[R, D]`

**Attention weight structure:**
```
K_total = num_registers + num_target_patches + num_context_patches

Sequence order:
[0 : R]              → Register tokens
[R : R + K_target]   → Target patches
[R + K_target : end] → Context patches
```

**Files modified:**
- `scripts/eval.py`: Checkpoint loading, per-case/label logging, axis naming
- `src/train_utils.py`: `validate()` returns detailed_results, `save_predictions()` saves attention
- `src/models/simple_backbone.py`: `return_attn_weights` through all layers
- `src/models/patch_icl.py`: Propagate flag through PatchICL_Level and PatchICL

## 2027-01-27: SimpleBackbone + High-Capacity Training Config

### SimpleBackbone Implementation

Created `src/models/simple_backbone.py` - a clean, focused replacement for the complex backbone.py (2,539 lines → 630 lines).

**Architecture:**
```
SimpleCNNEncoder → CrossPatchAttention → SimpleCNNDecoder
[B,K,49,1024]        [B,K,D]              [B,K,D] + skips
     │                  │                      │
     ▼                  ▼                      ▼
Linear(1024→D)      + type_embed          TransConv + skips
Reshape [B*K,D,7,7] + 2D RoPE             Bilinear upsample
Conv layers         + registers            → [B,K,C,ps,ps]
Pool [B,K,D]        Multi-layer attention
```

**Key features:**
- **SimpleCNNEncoder**: Projects DINO features, 3-level CNN with skip connections (7×7 → 4×4 → 2×2 → pool)
- **CrossPatchAttention**: Multi-layer attention with:
  - Type embeddings (context vs target)
  - 2D RoPE for spatial position encoding
  - Register tokens for global context
  - Configurable target self-attention
  - Masked attention pattern (context↔context, target→context)
- **SimpleCNNDecoder**: U-Net style with skip fusion, bilinear upsampling to any patch_size

**Config example:**
```yaml
backbone:
  type: "simple"
  encoder:
    embed_dim: 1024        # DINO input dim
    embed_proj_dim: 512    # Working dimension
  cross_attention:
    num_heads: 8
    num_layers: 4          # Stackable attention layers
    num_registers: 8
    target_self_attention: true
    dropout: 0.1
```

### High-Capacity Training Config

Created `configs/experiment/high_capacity.yaml` to improve training dice and GPU utilization.

**Analysis of original config issues:**
- Only 3.8GB / 49GB VRAM used (8%)
- 16 patches at resolution 32 = poor coverage
- embed_dim=128 = underpowered model
- smoothL1 loss = wrong for segmentation
- Context supervision disabled

**New high-capacity config:**
| Setting | Before | After | Impact |
|---------|--------|-------|--------|
| Parameters | ~2M | 37.7M | 19x capacity |
| VRAM | 3.8GB (8%) | 21GB (43%) | 5.5x utilization |
| Batch size | 50-64 | 192 | Better gradients |
| Patches | 16 | 96 | 6x coverage |
| Embed dim | 128 | 512 | 4x richer features |
| Attention layers | 1 | 4 | Deeper reasoning |
| Loss | smoothL1 | diceCE | Proper segmentation |
| Context loss | 0 | 0.5 | Extra supervision |

**VRAM scaling tests:**
```
num_layers=1: 21.7GB (44%), 28.2M params
num_layers=2: 19.7GB (40%), 31.4M params
num_layers=4: 20.5GB (42%), 37.7M params
```
More layers add ~3M params each but minimal VRAM (attention mask shared).

**Files created/modified:**
- `src/models/simple_backbone.py` (NEW): SimpleBackbone with 4 classes
- `src/models/patch_icl.py`: Added SimpleBackbone import and factory case
- `configs/experiment/high_capacity.yaml` (NEW): Optimized training config

**Run:** `python scripts/train.py experiment=high_capacity`

## 2026-01-26: Fixed Patch Rotation with Precomputed Features

**Problem:** When using precomputed DINO features with patch augmentation (rotation/flip), there was a mismatch:
- Image patches were extracted and rotated
- Patch labels were rotated to match
- But feature patches were extracted from precomputed maps at *original* positions (not rotated)
- This caused features to not match the rotated visual content

**Solution:** Apply the same augmentation to extracted features:
1. Sample patch coordinates
2. Extract image patches and labels at coords (with augmentation → get aug_params)
3. Extract features at the same coords
4. Apply the *same* augmentation (from aug_params) to features
5. Process rotated features in backbone
6. Inverse-rotate predictions before aggregation

**Files modified:**
- `src/models/sampling.py`:
  - Added `_rotate_features_90()`, `_rotate_features_continuous()` for feature tensor rotation
  - Added `_flip_features()`, `_scale_features()` for other augmentations
  - Added `augment_features_only()` method to apply pre-determined aug_params to features
  - Updated sampler forward methods to return augmented features
- `src/models/patch_icl.py`:
  - Updated `PatchICL_Level.forward()` to extract features after sampling, then augment them
  - Updated `_select_context_patches()` to return aug_params for each context
  - Added inverse augmentation to context patch logits before aggregation

**Key insight:** Features are `[B, K, tokens, D]` where tokens=h×w (e.g., 7×7=49). Reshape to spatial `[B, K, D, h, w]`, apply same rotation as patches, reshape back.

**Known limitation - Continuous rotation corner artifacts:**
With continuous rotation (arbitrary angles), corners of the rotated feature patch sample from positions *outside* the extracted region, getting zeros (padding). This is because we only have features for the patch region itself, not neighboring regions.

```
Original features (7x7):     After rotation (e.g., 30°):
┌─────────┐                  ┌─────────┐
│ f f f f │                  │ 0 f f 0 │  ← corners are ZEROS
│ f f f f │   rotate →       │ f f f f │     (no features to interpolate)
│ f f f f │                  │ f f f f │
│ f f f f │                  │ 0 f f 0 │
└─────────┘                  └─────────┘
```

**Recommended workarounds:**
1. **Use 90° rotation only** (`rotation: "90"`) - no interpolation needed, no artifacts
2. **Small rotation angles** - for `rotation_range ≈ ±0.3 rad` (±17°), corner artifacts are minimal
3. **Extract with margin** (future work) - extract larger feature region, rotate, then crop

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
