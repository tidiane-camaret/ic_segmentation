# Research Log

Consolidated project log. Previous logs in `logs.md` and `configs/experiment/logs.md` have been merged here.

---

## 2026-02-26: Scheduled Oracle Sampling

**Goal:** Reduce train/val distribution mismatch caused by oracle sampling during training.

### Problem

| Phase | Level 1 Sampling | Distribution |
|-------|------------------|--------------|
| Train | Oracle (GT foreground) | Easy patches, always correct regions |
| Val | Model predictions | Harder, includes errors and uncertain regions |

This is "exposure bias" — the model never learns to recover from its own mistakes.

### Solution: Scheduled Sampling

Gradually transition from oracle (GT-guided) to model predictions during training:

```
Epoch 0-10:  100% oracle (warmup)
Epoch 10-60: Linear decay from 100% → 30%
Epoch 60+:   30% oracle (maintain some GT signal)
```

Per-sample stochastic mixing ensures the model sees both oracle and self-generated distributions.

### Config

```yaml
oracle_scheduling:
  enabled: true
  schedule: "linear"        # linear, exponential, inverse_sigmoid
  start_prob: 1.0           # Start with full oracle
  end_prob: 0.3             # End with 30% oracle
  warmup_epochs: 10         # Full oracle during warmup
  decay_epochs: 50          # Decay period
```

### Implementation

Added to `patch_icl.py`:
- `_get_oracle_probability(level_idx)`: Computes oracle prob based on epoch and schedule
- Modified sampling weight computation to stochastically mix oracle and model weights
- Logging: `level_{i}_oracle_prob` tracked during training

### Files Modified

| File | Changes |
|------|---------|
| `src/models/patch_icl_v2/patch_icl.py` | Added `oracle_scheduling` config, `_get_oracle_probability()`, modified forward sampling logic |
| `configs/experiment/103_2_lvls.yaml` | Added `oracle_scheduling` config block |

---

## 2026-02-26: Resolution-Conditioned Normalization (FiLM)

**Goal:** Prevent gradient interference between levels in multi-level training while maintaining resolution-agnostic architecture.

### Problem

When training with multiple levels (e.g., level 0 at 18×18, level 1 at 36×36), shared backbone weights receive conflicting gradients:
- Level 0 (uniform sampling) → learns general features
- Level 1 (oracle sampling) → learns foreground-specific features
- Both update the same BatchNorm statistics → interference

Even with `detach_between_levels: true`, gradients from both levels flow through the shared encoder/decoder.

### Solution: FiLM-style Resolution Conditioning

Replaced `BatchNorm2d` with `ResolutionConditionedNorm` — GroupNorm with scale (γ) and shift (β) predicted from continuous resolution embedding:

```
out = γ(resolution) × GroupNorm(x) + β(resolution)
```

**Key properties:**
- **Resolution-agnostic**: Works with any resolution (continuous embedding, not discrete)
- **Partial gradient isolation**: Different resolutions → different γ/β paths
- **Generalizes to unseen resolutions**: Can test on resolutions not seen during training
- **No fixed level count**: Architecture doesn't hardcode number of levels

### Implementation

New class `ResolutionConditionedNorm` in `simple_backbone.py`:
```python
class ResolutionConditionedNorm(nn.Module):
    def __init__(self, num_channels, scale_embed_dim, num_groups=8):
        self.norm = nn.GroupNorm(num_groups, num_channels, affine=False)
        self.gamma_proj = nn.Linear(scale_embed_dim, num_channels)
        self.beta_proj = nn.Linear(scale_embed_dim, num_channels)

    def forward(self, x, scale_embed):
        x = self.norm(x)
        gamma = self.gamma_proj(scale_embed).view(1, -1, 1, 1)
        beta = self.beta_proj(scale_embed).view(1, -1, 1, 1)
        return gamma * x + beta
```

### Files Modified

| File | Changes |
|------|---------|
| `src/models/simple_backbone.py` | Added `ResolutionConditionedNorm`, updated `SimpleCNNEncoder` and `SimpleCNNDecoder` to use it, pass `scale_embed` through `SimpleBackbone.forward()` |

### Parameter Overhead

FiLM projector params account for ~9% of backbone:
- FiLM params: ~200K
- Total backbone: ~2.2M

### Usage

No config changes needed — automatically uses resolution passed to backbone:

```python
# In PatchICL._forward_level():
backbone_out = self.backbone(..., resolution=level_cfg['resolution'])
```

---

## 2026-02-23: Generalization Improvements

**Goal:** Address train/val gap and improve generalization for in-context segmentation.

### Problem Analysis

Identified key issues affecting generalization:
1. **Train/val sampling gap**: Model learns to rely on specific sampling distributions that differ at validation
2. **Context selection**: Random context selection ignores feature diversity
3. **Confidence calibration**: Entropy-based confidence can be overconfident
4. **Level weighting**: Uniform level weights may undertrain finer levels

### Implemented Improvements

#### A1 & A2: Sampling Robustness (`src/models/patch_icl_v2/patch_icl.py`)

Added sampling dropout and temperature annealing to reduce dependence on specific sampling patterns:

```yaml
sampling_robustness:
  dropout: 0.2           # 20% chance of uniform sampling
  temperature_start: 2.0  # Start more uniform
  temperature_end: 1.0    # Anneal to normal
  temperature_epochs: 50  # Anneal over 50 epochs
```

- **Dropout**: Randomly replaces confidence/oracle weights with uniform sampling during training
- **Temperature annealing**: `weights^(1/T)` where T anneals from high (uniform) to low (sharp)

#### A3: Feature-Based Context Diversity (`src/dataloaders/totalseg2d_dataloader_fast.py`)

Implemented farthest-point sampling for context selection:

```yaml
context_diversity:
  type: "farthest"       # or "random"
  num_candidates: 10     # Pool size for selection
  feature_key: "mean_features"  # Key in stats.pkl
```

Requires pre-computed mean features per case/label in stats file.

#### A4: Confidence Temperature Calibration (`src/models/patch_icl_v2/patch_icl.py`)

Added temperature scaling for entropy-based confidence to calibrate overconfident predictions:

```yaml
confidence:
  method: "entropy"
  temperature: 1.5  # T > 1 softens, T < 1 sharpens
```

Formula: `conf = 1 - H(sigmoid(logits/T))` where T is the temperature.

#### A5: Progressive Level Weights

Added `"progressive"` option for level weights that linearly increases from 0.3 to 1.0:

```yaml
loss:
  level_weights: "progressive"  # or [0.3, 0.65, 1.0]
```

#### A6: Dual Mask Pathway Ablation

The existing `use_context_mask: false` option can ablate the backbone mask fusion when UniverSeg already provides mask-conditioned features.

### Experiment Configs

| Config | Description |
|--------|-------------|
| `89_generalization.yaml` | Combined best settings |
| `89a_sampling_dropout.yaml` | A1: Sampling dropout only |
| `89b_temp_anneal.yaml` | A2: Temperature annealing only |
| `89c_progressive_weights.yaml` | A5: Progressive level weights |
| `89d_conf_temp.yaml` | A4: Confidence temperature |
| `89e_no_dual_mask.yaml` | A6: Disable dual mask pathway |

### Files Changed

| File | Changes |
|------|---------|
| `src/models/patch_icl_v2/patch_icl.py` | Added `sampling_robustness` config, `set_epoch()`, `_apply_sampling_robustness()`, temperature param to `compute_entropy_confidence()`, progressive level weights |
| `src/dataloaders/totalseg2d_dataloader_fast.py` | Added `context_diversity` config, `_select_diverse_contexts()` |
| `src/train_utils.py` | Call `set_epoch()` on model for temperature annealing |
| `configs/experiment/89*.yaml` | New ablation configs |

---

## 2026-02-18: Confidence Head Implementation

**Goal:** Add patch-level confidence prediction to enable confidence-weighted loss, confidence-aware aggregation, and multi-level blending.

### Architecture

Added a confidence head to `SimpleCNNDecoder` that predicts pixel-level confidence alongside segmentation. The decoder now uses a shared trunk with separate output heads:

```
Decoder Trunk (U-Net style) → [B*K, D, h, h]
    ├─ seg_head → [B, K, C, ps, ps]    # Segmentation logits
    └─ conf_head → [B, K, 1, ps, ps]   # Confidence in [0, 1] (sigmoid)
```

### New Loss Functions (`src/losses.py`)

| Loss | Description |
|------|-------------|
| `ConfidenceSupervisionLoss` | MSE between predicted confidence and `1 - |pred_prob - gt|`. Trains confidence to match prediction accuracy. |
| `BoundaryConfidenceLoss` | Penalizes high confidence at patch borders (artifact-prone regions) |
| `ConfidenceWeightedDiceLoss` | Dice loss weighted by confidence (high-confidence regions contribute more) |

### Aggregation Enhancements (`src/models/patch_icl_v2/aggregate.py`)

- Added `use_confidence` and `confidence_mode` parameters to aggregators
- `confidence_mode="multiply"`: Multiplies base weights (uniform/Gaussian) by confidence
- `confidence_mode="replace"`: Uses confidence as sole weighting
- New `combine_mode="confidence"`: Uses aggregated confidence for blending with previous level

### Multi-Level Confidence Blending

When `cascade.confidence_blend=true`, levels are blended using confidence:
```
combined = conf × current + (1 - conf) × previous
```
Uncertain regions defer to coarser level predictions.

### Config Options

```yaml
backbone:
  predict_confidence: true    # Enable confidence head

aggregator:
  use_confidence: true        # Modulate weights by confidence
  confidence_mode: "multiply" # or "replace"

loss:
  confidence:
    enabled: true
    supervision_weight: 0.5   # MSE(conf, 1-error) weight
    boundary_weight: 0.1      # Penalty for border confidence
    boundary_width: 2         # Border pixels to penalize
    weighted_seg: false       # Use confidence-weighted Dice

cascade:
  confidence_blend: true      # Multi-level confidence blending
```

### Files Modified

| File | Changes |
|------|---------|
| `src/models/simple_backbone.py` | Added `predict_confidence` param, refactored decoder to use shared trunk with separate heads |
| `src/losses.py` | Added `ConfidenceSupervisionLoss`, `BoundaryConfidenceLoss`, `ConfidenceWeightedDiceLoss` |
| `src/models/patch_icl_v2/aggregate.py` | Added confidence parameter, multiply/replace modes, confidence combine mode |
| `src/models/patch_icl_v2/patch_icl.py` | Integrated confidence in loss computation, multi-level blending |
| `configs/experiment/85_rad_dino.yaml` | Added confidence config options (disabled by default) |

---

## 2026-02-18: RAD-DINO Pretrained Multi-Modality Encoder

**Goal:** Replace trainable ICLEncoder with pretrained multi-modality encoder for better CT→MRI generalization.

### Background

The current `ICLEncoder` (trainable CNN) achieves good results on CT datasets but accuracy drops on MRI due to domain shift. Pretrained encoders trained on diverse medical imaging data can provide better cross-modality generalization.

### Implementation

**RAD-DINO** (Microsoft, HuggingFace: `microsoft/rad-dino`):
- DINOv2-based ViT-B/14 trained on RadImageNet (1.35M images)
- Covers CT, MRI, ultrasound, X-ray (11 anatomical regions)
- 768-dim features, 16x16 output grid for 224x224 input
- Self-supervised (no text supervision) — robust features

**Architecture change**: Context masks are NOT passed to the feature extractor. Pretrained encoders only process images. Mask information is instead handled at the backbone/transformer level via `use_context_mask`, which passes patch-level masks directly to attention.

### Files

| File | Description |
|------|-------------|
| `src/models/rad_dino_extractor.py` | New RAD-DINO feature extractor (RADDINOExtractor class) |
| `scripts/train.py` | Added `rad_dino` extractor type registration |
| `configs/experiment/85_rad_dino.yaml` | Config using RAD-DINO with adapted backbone settings |

### Key Config Changes

```yaml
feature_extractor_type: "rad_dino"

model:
  patch_icl:
    feature_extractor:
      type: "rad_dino"
      model_name: "microsoft/rad-dino"
      target_size: 224      # ViT native input
      output_grid_size: 16  # ViT-B/14: 224/14=16
      freeze: true          # Frozen for cross-modality

    backbone:
      embed_dim: 768        # Match RAD-DINO output
      embed_proj_dim: 256   # Project down for efficiency
      feature_grid_size: 16 # Match extractor grid
      num_heads: 8          # More heads for 768-dim
      use_context_mask: true  # Masks at attention level
```

### Usage

```bash
# Train with RAD-DINO
python scripts/train.py --config configs/experiment/85_rad_dino.yaml

# Test encoder loading
python -c "from src.models.rad_dino_extractor import RADDINOExtractor; e = RADDINOExtractor(device='cpu'); print(e.get_feature_info())"
```

### Dependencies

```bash
pip install transformers  # For HuggingFace model loading
```

---

## 2026-02-17: Unified Augmentation Pipeline Implementation

**Goal:** Refactor augmentation to fix issues identified in the literature review.

### Issues Fixed

1. **Double intensity augmentation** — Old pipeline had `random_intensity_shift()` in advanced_augmentation AND `intensity_transform` in standard augmentation. Now uses single unified intensity pipeline.

2. **No task-level augmentation** — Added UniverSeg-style `apply_task_level_augmentation()` that applies the same transform (flip/rotate90) to all images in a batch.

3. **CarveMix only** — Added simpler `cut_mix_2d()` as alternative (literature shows CutMix > CarveMix).

4. **Foreground crop timing** — Now disabled when mix is applied to avoid scale mismatch.

### New Config Structure

```yaml
augmentation:
  enabled: true
  mix:
    type: "cutmix"  # "cutmix", "carve_mix", "none"
    probability: 0.5
  spatial:
    enabled: true
    foreground_crop:
      disable_with_mix: true  # Key fix
  intensity:
    enabled: true
    asymmetric: true  # Different per image
  task_level:
    enabled: true
    probability: 0.3
```

### Files Changed

- `src/dataloaders/augmentations.py` — Added `cut_mix_2d()`, `apply_task_level_augmentation()`, deprecated `random_intensity_shift()`
- `src/dataloaders/totalseg2d_dataloader_fast.py` — Added unified config support, refactored `__getitem__` with new flow
- `scripts/train.py` — Parse unified `augmentation` config key
- `configs/experiment/83_unified_augmentation.yaml` — Example config

### Backwards Compatibility

Legacy configs (`image_augmentation`, `carve_mix`, `advanced_augmentation`) still work. Unified config takes precedence when `augmentation.enabled: true`.

---

## 2026-02-16: Augmentation Literature Review for In-Context Segmentation

**Goal:** Survey SOTA augmentation strategies for in-context learning (ICL) segmentation, with focus on medical imaging (2024–2026).

### Key Methods Reviewed

**UniverSeg (ICCV 2023)** — Two-tier augmentation:
- *In-Task Augmentation*: Standard transforms (affine, elastic, noise) applied **independently** to query and each support image with different random params.
- *Task Augmentation*: Same transform applied **uniformly** to all query+support images (e.g., flip all, edge-detect all masks). Helps generalize to unseen tasks.
- Support set size: 64 in most experiments.

**SegGPT / Painter (ICCV 2023)** — Random coloring as core strategy:
- Random color mapping per sample prevents class-color memorization.
- Standard augmentations: random resize crop, color jitter, horizontal flip.
- 50% probability of using augmented view as in-context example (semantic seg); 100% for instance seg.
- No asymmetric support/query augmentation — the random coloring itself is the key trick.

**SegICL (2024)** — Minimal augmentation details published. Relies on SA-Med2D-20M preprocessing. No explicit support/query augmentation distinction.

**SAM2 + Augmentative Prompting (March 2025)** — Training-free, augmentation at inference:
- Geometric: affine (scale, rotation, shear, translation) applied to support image + mask jointly.
- Photometric: color jitter (brightness, contrast, saturation, hue) on image only.
- NT=2 augmented versions per support image.
- Dynamic matching via LPIPS: for each query slice, select augmented support with lowest perceptual distance.

**Visual Prompt Selection (ECCV 2024)** — Support diversity > similarity:
- Combining nearest AND farthest examples improves IoU by ~2.4 over nearest-only.
- Random selection shows >5.6 IoU gap between best/worst.
- Dissimilar contexts outperform in ~40% of cases.

### Mix-Based Augmentation (MICCAI 2024)

"Cut to the Mix" compared CutMix, CarveMix, ObjectAug, AnatoMix on organ segmentation:
- **CutMix (+4.9 Dice) > CarveMix (+2.0) > AnatoMix (+1.9)**
- Counterintuitive: simple CutMix outperforms anatomy-aware methods.
- Anatomical plausibility doesn't help — raw diversity matters more.

### nnUNet Augmentation (Gold Standard Reference)

Fixed pipeline, universally robust:
- Spatial: rotation ±30°, elastic deformation (alpha=1000, sigma=10), scaling
- Intensity: gamma, brightness, contrast, Gaussian noise, Gaussian blur, low-res simulation
- Mirroring on all axes

### Emerging Trends (2025–2026)

| Trend | Description |
|-------|-------------|
| Adaptive augmentation | Per-sample intensity (MICCAI 2025 ADA framework) |
| Diffusion-based augmentation | Generative models for realistic training samples (ICCV 2025) |
| Test-time augmentation as prompting | Augment support images at inference for prompt diversity |
| Support set optimization | Learn which supports to select rather than augment blindly |
| Simple > complex mixing | CutMix outperforms CarveMix — diminishing returns on anatomical realism |

### Assessment of Our Current Pipeline (exp 81)

| Our approach | SOTA consensus | Assessment |
|---|---|---|
| CarveMix (p=0.5) | CutMix may be simpler and equally/more effective | Consider ablating CarveMix vs CutMix |
| Independent spatial aug per image | UniverSeg does same — standard | Good |
| Asymmetric intensity (per image) | Aligned with UniverSeg in-task aug | Good and novel |
| Mask perturbation on context only | Not found in any SOTA method | Worth ablating |
| Double intensity augmentation | No SOTA method stacks two intensity pipelines | Likely redundant, risk of saturation |
| Foreground crop + resolution degradation | nnUNet does low-res sim; fg crop unusual for ICL | Risky with CarveMix (scale mismatch) |
| Random coloring | SegGPT core strategy | Well-motivated |
| No task augmentation (uniform to all) | UniverSeg uses both in-task AND task aug | **Gap** — consider adding |

### Recommendations

1. **Add task-level augmentation** (UniverSeg-style): occasionally apply same transform to all images to teach task-invariance.
2. **Remove or reduce double intensity augmentation** — no SOTA method stacks two independent intensity pipelines.
3. **Consider replacing CarveMix with simpler CutMix** based on MICCAI 2024 findings.
4. **Support set diversity** matters as much as augmentation quality.
5. **Mask perturbation** on context is unique — worth keeping but needs ablation.

### References

- UniverSeg (ICCV 2023): https://arxiv.org/abs/2304.06131
- SegGPT (ICCV 2023): https://arxiv.org/abs/2304.03284
- SegICL (2024): https://arxiv.org/abs/2403.16578
- SAM2 Augmentative Prompting (2025): https://arxiv.org/abs/2503.04826
- Visual Prompt Selection (ECCV 2024): https://arxiv.org/abs/2407.10233
- Cut to the Mix (MICCAI 2024): https://papers.miccai.org/miccai-2024/185-Paper0674.html
- MICCAI 2025 ADA Framework: https://papers.miccai.org/miccai-2025/0039-Paper0315.html
- DINOv2 Few-Shot Med Seg (ISBI 2024): https://arxiv.org/abs/2403.03273

---

## 2026-02-16: Mask Prior Fusion Before Attention

**Goal:** Use `combined_pred` from level i-1 to guide attention at level i, not just for sampling. Extract mask patches and fuse them with encoded image features before attention.

**Approach:** SAM-style additive fusion with optional gated or concatenation modes.

**Implementation:**
1. `extract_mask_patches()` in `patch_icl.py` — Extracts mask patches using `grid_sample` (same logic as `extract_patch_features` but for 1-channel mask input).
2. `LayerNorm2d` in `simple_backbone.py` — SAM-style 2D layer normalization.
3. `MaskPriorEncoder` in `simple_backbone.py` — 3-layer CNN (SAM mask_downscaling architecture) that encodes `[B*K, 1, h, h]` mask patches to `[B, K, embed_dim]`.
4. `SimpleBackbone` — New params `use_mask_prior` and `mask_fusion_type`. Fuses mask prior embeddings with target patch encodings before attention.
5. `PatchICL._forward_level()` — Accepts `mask_prior`, extracts mask patches, passes to backbone.
6. `PatchICL.forward()` — Passes `combined_pred` from previous level as `mask_prior` (level 0 gets None).

**Fusion modes:**
- `additive`: `encoded += mask_encoded` (SAM-style, default)
- `gated`: `encoded += sigmoid(gate) * mask_encoded` (learnable scale, MAIS-inspired)
- `concat`: `encoded = proj(cat(encoded, mask_encoded))` (Medverse-style)

**Config:**
```yaml
backbone:
  use_mask_prior: true
  mask_fusion_type: "additive"  # or "gated", "concat"
```

**Tensor flow (level 1+):**
```
combined_pred [B, 1, prev_res, prev_res]
    ↓ interpolate
mask_prior_ds [B, 1, resolution, resolution]
    ↓ extract_mask_patches
mask_prior_patches [B, K_target, 1, h, h]
    ↓ MaskPriorEncoder
mask_encoded [B, K_target, embed_dim]
    ↓ additive fusion
encoded[:, :K_target] += mask_encoded
    ↓ attention (now conditioned on mask prior)
```

**Extension: Context Mask Fusion**

Also added option to fuse GT context masks into context patch embeddings. Since context masks are GT at both train and test time (user-provided), no train-test asymmetry concern.

- `use_context_mask: true` — Extracts mask patches from context GT masks and fuses them into context patch embeddings.
- Uses same `MaskPriorEncoder` and fusion type as target mask prior.
- For gated fusion, uses separate `context_mask_gate` parameter.

**Config:**
```yaml
backbone:
  use_mask_prior: true      # Target: fuse previous level prediction
  use_context_mask: true    # Context: fuse GT masks
  mask_fusion_type: "additive"
```

**Files modified:**
| File | Changes |
|------|---------|
| `src/models/patch_icl_v2/patch_icl.py` | Added `extract_mask_patches()`, modified `_forward_level()` to extract mask patches for both target (from `combined_pred`) and context (from `context_out_ds`), modified `forward()` to pass `combined_pred` as `mask_prior`. |
| `src/models/simple_backbone.py` | Added `LayerNorm2d`, `MaskPriorEncoder`, modified `SimpleBackbone.__init__()` with `use_mask_prior` and `use_context_mask`, modified `forward()` to fuse both target and context mask patches. |
| `configs/experiment/70_attention.yaml` | Added `use_mask_prior: true`, `use_context_mask: true`, `mask_fusion_type: "additive"`. |

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
