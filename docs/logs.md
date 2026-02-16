# Research Log

Consolidated project log. Previous logs in `logs.md` and `configs/experiment/logs.md` have been merged here.

---

## 2026-02-15: Context-First Sequential Attention

**Problem:** The backbone uses fully bidirectional attention — context and target patches attend equally. No explicit mechanism forces context patches to first establish inter-context relationships before targets read from them. Using attention masks (to enforce context→target flow) disables the Flash SDP backend.

**Solution:** Two-stage sequential attention, both stages fully bidirectional (Flash-compatible):
1. **Stage 1 (context-only):** Context patches self-attend through dedicated `context_layers` with separate `context_registers`. This enriches context representations with inter-example relationships.
2. **Stage 2 (joint):** All patches (target + enriched context) attend together through the main `layers`, as before.

When `num_context_layers: 0` (default), behavior is identical to before. The context stage is also skipped when there are no context patches or `num_target_patches` is not provided.

**Config:**
```yaml
backbone:
  num_context_layers: 1  # 0 = disabled (backward compatible)
  num_layers: 2          # Joint stage layers (unchanged)
```

**Cost:** With `embed_dim=256`, one context layer adds ~530K params (~50K at dim=64). Negligible memory overhead since context-only attention operates on fewer tokens than the joint stage.

**Bugfix:** Fixed infinite recursion in `ICLEncoder.train()` when `freeze=True` — `self.eval()` called `self.train(False)` which called `self.eval()` again.

**Files modified:**
| File | Changes |
|------|---------|
| `src/models/simple_backbone.py` | `CrossPatchAttention`: added `num_context_layers`, `context_layers`, `context_registers`, context-first stage in `forward()`. `SimpleBackbone`: passes `num_context_layers` and `num_target_patches` through. |
| `src/models/patch_icl_v2/patch_icl.py` | Reads `num_context_layers` from config, passes `num_target_patches=K` to backbone. |
| `src/models/icl_encoder.py` | Fixed `train()` recursion bug when frozen. |
| `configs/experiment/70_attention.yaml` | Added `num_context_layers: 1`. |

---

## 2026-02-14: Unmasked Flash Attention + torch.compile

**Changes to `simple_backbone.py`:**
1. **Removed attention mask** — Previously blocked context→target attention (~17% of matrix) using a float mask with `-inf`, which disabled Flash SDP. Now uses full bidirectional attention, enabling Flash Attention backend.
2. **Optimized RoPE** — Replaced manual sin/cos rotation with `torch.view_as_complex` multiplication (~1.3x faster). Removed `.item()` calls that break `torch.compile` graphs.
3. **Added `torch.compile`** — New `backbone.compile: true` config flag (default false) compiles encoder, attention, and decoder submodules after DDP wrapping.

**Config**: `target_self_attention` is now ignored (kept for backward compat). Set `backbone.compile: true` to enable compilation.

---

## 2026-02-11: Multi-Level Cascaded Training Support

Refactored PatchICL from single-level to multi-level cascaded architecture.

- `PatchICL.__init__`: Per-level samplers (`nn.ModuleList`), aggregators, oracle configs. All levels share `patch_size` (backbone constraint).
- `_forward_level()`: Processes a single resolution level (sampling, feature extraction, backbone, aggregation).
- `forward()`: Loops coarse-to-fine. Level 0 uses uniform/oracle weights; subsequent levels use `sigmoid(prev_level_pred)` as sampling weights. `detach_between_levels` (default True).
- `compute_loss()`: Per-level losses with configurable `level_weights`. Averaged across levels for logging.
- New config: `configs/experiment/60_2_levels.yaml` (resolution 16 + 32, 8+12 patches).

**Design decisions:** Single backward after all levels (losses accumulated). Detach between levels by default (memory efficient, no gradient benefit since sampling is non-differentiable).

---

## 2026-02-10: Center-Based Patch Sampling + Train/Val Patch Counts

### Center-based sampling

**Problem:** Patches could only be sampled fully inside the image. For a 32x32 feature map with patch_size=8, only 25x25=625 valid positions existed, under-representing borders.

**Solution:** Allow any patch whose **center** is inside the image. Top-left ranges from `-ps//2` to `H - 1 - ps//2`, giving HxW valid positions.

**Implementation:**
- **Samplers** (`sampling.py`): Pad image/labels/weights. Sample from padded space, convert coords back (can be negative).
- **Validity mask**: `[B, K, 1, ps, ps]` tracking real vs padding pixels. Concatenated with labels during augmentation for consistent transforms.
- **Aggregator** (`aggregate.py`): Two-sided coordinate clipping handles negative coords.
- **Loss** (`patch_icl.py`): `_masked_patch_loss()` sets invalid pixels to logit=-100 / label=0.

### Train/val patch counts

Added `num_patches_val` per level config. `ContinuousSampler` picks K based on `self.training`. Defaults to `num_patches` if omitted.

---

## 2026-02-09: Unified Soft GT Downsampling (max_pool -> avg_pool)

**Problem:** GT masks downsampled inconsistently — `max_pool2d` (dilated), `F.interpolate` nearest (misses small FG), and `avg_pool2d > 0.25` (metrics) in three different places. Conflicting supervision.

**Solution:** All GT downsampling now uses `avg_pool2d`, producing soft [0, 1] area-fraction targets throughout. `SoftDiceBCE` handles continuous targets natively. Consistent with existing dice metrics.

---

## 2026-02-06: Train/Val Dice Gap Analysis

**Problem:** Train dice 0.3, val dice 0.1 (3x gap).

**Root causes identified:**
1. Warmup bug: warmup_epochs=10 with num_epochs=10 (model never reached full LR)
2. SlidingWindowSampler ignores `weights` — oracle settings have no effect
3. Low context size (1 example)
4. Context loss disabled
5. No patch-level augmentation
6. Train/val use different organs (by design — ICL generalization test)

**New config `exp_11_improved.yaml`:** 100 epochs, warmup=5, context_size=3, continuous sampler, oracle train, 90-deg rotation + flips, context loss=0.5.

---

## 2026-02-05: Codebase Simplification + Architecture Experiments

### Codebase simplification (v1 -> v2)

Reduced core code by 61% (4163 -> 1607 lines). Created `src/models/patch_icl_v2/` with only the samplers and aggregators actually used. Moved originals to `patch_icl_v1/`.

| Component | Before | After |
|-----------|--------|-------|
| patch_icl.py | 1209 | 496 |
| sampling.py | 1272 | 345 |
| aggregate.py | 491 | 173 |
| train.py | 442 | 248 |
| train_utils.py | 749 | 345 |

Removed: UniformSampler, DeterministicTopKSampler, GumbelSoftmaxSampler, ConfidenceAggregator, LearnedAggregator, LearnedCombineAggregator, 3D dataset branches, SegFormer3D.

### Architecture experiments

Iterative experiments with baseline val_final_dice ~0.1:

| Experiment | Key Change | val_final_dice | Conclusion |
|:-----------|:-----------|:---------------|:-----------|
| Exp 1: Context Loss | Enabled context loss (0.5) | ~0.087 | Failure — overfit |
| Exp 2: Shallow Backbone | 4 -> 2 attention layers | ~0.16 | Success — less overfitting |
| Exp 3: Shallow + No Skips | 2 layers + no skip connections | ~0.25 | Major success — information bottleneck |

**Key finding:** Removing skip connections forces the decoder to rely solely on attention output, creating a bottleneck that produces better context-aware representations. This is now the default architecture.

### Train/val gap investigation

| Config | Train Dice | Val Dice | Gap |
|--------|------------|----------|-----|
| exp_03 (no skips + ctx loss) | 0.40 | 0.07 | 0.33 |
| exp_03 (train labels for val) | 0.15 | 0.08 | 0.07 |

Using same labels for train/val doesn't close the gap — the issue is not purely organ diversity. Likely causes: model memorizes spatial patch positions, sliding window creates train-specific patterns.

---

## 2026-02-04: Feature Extraction Experiments Infrastructure

Added experiment infrastructure in `Experiments/feature_extraction/` for comparing feature extractors:
- **Layer comparison**: MedDINO layers [2, 5, 8, 11]
- **Multi-layer fusion**: average, learned_weighted, concat_proj
- **MedSAM v1**: Alternative extractor (ViT-B, 1.5M medical images)

Added `MultiLayerFeatureExtractor` to `meddino_extractor.py`. Fixed `feature_grid_size` default: 16 -> 14 (correct for MedDINO 14x14 grid).

---

## 2026-02-03: Refinement Bug Fixes + Sampling Improvements

Fixed 4 critical bugs in iterative refinement:

1. **Confident predictions destroyed**: Uncovered regions filled with -10 logits. Fix: `prev_pred_for_agg` + `combine_mode: "coverage"` preserves uncovered regions.
2. **False confidence**: Logits < -5 had sigmoid ~ 0, interpreted as "confident background". Fix: Detect and mark as high uncertainty.
3. **Double-counting gradients**: Previous predictions flowed gradients through refinement. Fix: Detach before refinement.
4. **Resolution ping-pong**: Uncertainty upsampled then immediately downsampled. Fix: Pass at level resolution.

**Sampling improvement:** Changed weight pooling from `avg_pool2d` to `max_pool2d` — boundary patches (50% FG) now score equally to interior patches (100% FG).

---

## 2026-02-02: On-the-fly Feature Extraction (MedDINO + MedSAM2)

Added on-the-fly feature extraction as alternative to precomputed features. Two extractors:
- **MedDINOv3**: ViT-base, 768-dim, 256px input, ~350MB GPU
- **MedSAM2**: Hiera (t/s/b+/l), 256-dim, 1024px input, ~150-900MB GPU

Config: `feature_mode: "on_the_fly"`, `feature_extractor_type: "meddino"/"medsam2"`. Dataloader skips `.npz` loading. ~10-30% slower than precomputed but enables experimentation without re-extracting features.

---

## 2026-02-01: Eval Script Improvements + Attention Visualization

- **Checkpoint loading** in `eval.py` before evaluation
- **Per-case/per-label dice**: `validate()` returns `detailed_results` with per-sample tracking
- **Axis-based naming**: Output folders now `{case}_{label}_{axis}` instead of `{case}_{label}_b{batch}_s{sample}`
- **Attention weight saving**: `return_attn_weights` flag through AttentionBlock -> CrossPatchAttention -> SimpleBackbone -> PatchICL. Saves per-layer `[H, K_total, K_total]` attention weights and `[R, D]` register tokens as `.npy`.

---

## 2026-01-27: SimpleBackbone + High-Capacity Config

### SimpleBackbone

Created `src/models/simple_backbone.py` — clean replacement for backbone.py (2539 -> 630 lines).

```
SimpleCNNEncoder -> CrossPatchAttention -> SimpleCNNDecoder
[B,K,tokens,D]      [B,K,D]               [B,K,C,ps,ps]
```

Key features: 3-level CNN encoder with skips, multi-layer attention with type embeddings + 2D RoPE + registers, U-Net decoder with skip fusion.

### High-capacity config

Identified original config was only using 8% of 49GB VRAM:

| Setting | Before | After |
|---------|--------|-------|
| Parameters | ~2M | 37.7M |
| VRAM | 3.8GB (8%) | 21GB (43%) |
| Embed dim | 128 | 512 |
| Attention layers | 1 | 4 |
| Loss | smoothL1 | diceCE |

---

## 2026-01-26: 2D RoPE + Feature Rotation for Precomputed Features

### 2D RoPE

**Problem:** Original RoPE used sequential patch indices regardless of spatial location.

**Solution:** `apply_rope_2d()` uses actual patch (y, x) coordinates. First half of embedding rotated by x-coordinate, second half by y-coordinate.

### Feature rotation

**Problem:** With precomputed features + patch augmentation, features were extracted at original positions but images/labels were rotated — mismatch.

**Solution:** Apply same augmentation to features: reshape `[B, K, tokens, D]` to `[B, K, D, h, w]`, apply rotation/flip, reshape back. Added `augment_features_only()` and inverse augmentation for predictions before aggregation.

**Known limitation:** Continuous rotation introduces corner artifacts on low-res feature grids (7x7). Recommend 90-degree rotation only.

---

## 2026-01-21: PatchICL Training Pipeline Bug Fixes

Fixed critical bugs preventing proper training:
- **embed_dim**: 768 -> 1024 (DINOv3 ViT-L)
- **Features not passed to model**: `train_utils.py` extracted features but didn't pass to `model()`
- **Warmup scheduler**: Implemented `LinearLR` + `SequentialLR` chain
- **Coordinate scale mismatch**: Patch coords in 512 space but backbone assumed 224
- **Oracle sampling**: Added separate `oracle_levels_train`/`oracle_levels_valid` configs
- **Vectorized feature extraction**: Replaced Python loops with batch indexing

---

## 2026-01-16: Token-based NMSW Architecture (Exploratory)

Explored selective patch sampling via NMSW (No More Sliding Window). Evolved from standard NMSW to token-based approach:

```
Input [128^3] -> GlobalBranch (UNet) -> Objectness scores
-> Gumbel top-k selects 200 of ~4000 patches (8^3 each)
-> PatchTokenizer (CNN) -> [B, 200, 512]
-> CrossPatchTransformer (12 layers, 8 heads)
-> PatchDecoder -> Aggregate -> Final prediction
```

This exploration informed the PatchICL architecture — the key insight that small patches attending to each other via transformer is effective.

---

## 2026-01-11: Context Sampling with PatchWork Method

Replaced threshold-based foreground sampling with PatchWork's probability-weighted method. Achieves 50% foreground/background ratio using `background_p = (1 - ratio) * pos / (numvx * ratio - pos)`. Added inspection system for visualizing predictions at each autoregressive level.
