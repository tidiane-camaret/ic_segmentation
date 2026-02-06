# PatchICL

In-context medical image segmentation using patch-based learning. The model receives a target image and context image/mask pairs, then outputs a segmentation for the target.

## Architecture

**PatchICL** samples patches from target and context images, processes them through a shared backbone with cross-attention, and aggregates predictions back to full resolution.

```
Target Image + Context Pairs → Sample Patches → Cross-Patch Attention → Aggregate → Segmentation
```

### Key Components

- **Sampler**: Selects patch locations (continuous, weighted, sliding window, etc.)
- **Backbone**: Cross-attention between target and context patches (SimpleBackbone, CrossPatchAttention)
- **Aggregator**: Combines patch predictions into full mask (Gaussian weighting, coverage-based)
- **Refinement**: Iteratively re-samples uncertain regions to improve predictions

## Experiments

### `patch_icl_v1` - DINO Features + Oracle Sampling

Baseline configuration using DINOv2 features (768-dim) with oracle sampling during training.

| Setting | Value |
|---------|-------|
| Feature extractor | DINOv2 ViT-B (768-dim) |
| Resolution | 32×32 |
| Patches | 32 patches, 16×16 each |
| Oracle sampling | Enabled (train & val) |
| Refinement | Disabled |

```bash
python scripts/train.py experiment=patch_icl_v1
```

### `patch_icl_v2` - MedSAM2 Features + Refinement

MedSAM2 features (256-dim) and iterative refinement

| Setting | Value |
|---------|-------|
| Feature extractor | MedSAM2 (256-dim) |
| Resolution | 32×32 |
| Patches | 32 patches, 16×16 each |
| Oracle sampling | Disabled |
| Refinement | 2 passes (uncertainty-based) |

```bash
python scripts/train.py experiment=patch_icl_v2
```

### `exp_11_improved` - Optimized for Train/Val Gap

Addresses train/val dice gap with optimized settings:

| Setting | exp_10 | exp_11_improved |
|---------|--------|-----------------|
| Epochs | 10 | 100 |
| Context size | 1 | 3 |
| Sampler | sliding_window | continuous |
| Oracle (train) | false | true |
| Augmentation | disabled | rotation + flips |
| Skip connections | disabled | enabled |
| Context loss | 0 | 0.5 |

```bash
python scripts/train.py experiment=exp_11_improved
```

## Refinement

The refinement mechanism iteratively improves predictions by re-sampling uncertain regions:

```
Pass 1: Sample patches uniformly/weighted → Initial prediction
Pass 2: Compute uncertainty → Sample from uncertain regions → Merge with confident regions
Pass 3+: Repeat until convergence
```

### How It Works

1. **Uncertainty Computation**: After each pass, compute pixel-wise uncertainty using:
   - `confidence`: `1 - |sigmoid(logit) - 0.5| * 2`
   - `entropy`: Binary entropy of predictions
   - `margin`: Distance between top predictions

2. **Selective Re-sampling**: High-uncertainty regions get new patches; confident regions are preserved.

3. **Coverage-based Aggregation**: The aggregator uses `combine_mode: "coverage"` to:
   - Use new predictions where patches were sampled
   - Preserve previous predictions where no patches were sampled

### Configuration

```yaml
refinement:
  passes: 3                    # 1 = disabled, 2+ = iterative
  uncertainty_type: "confidence"  # "confidence", "entropy", "margin"

aggregator:
  type: "gaussian"
  combine_mode: "coverage"     # Essential for refinement
  min_coverage: 0.01           # Threshold for "covered" pixels
```

### Saved Outputs

Patch positions are saved separately for each pass:
```
level0_pass0_patch_positions_mask.nii.gz  # Initial pass
level0_pass1_patch_positions_mask.nii.gz  # Refinement pass 1
level0_pass2_patch_positions_mask.nii.gz  # Refinement pass 2
```


## Evaluation

```bash
python scripts/eval.py experiment=<experiment_name> 
```

Checkpoint path is set in cluster config under `paths.ckpts.patch_icl`.

## Configuration Reference

### Model Config

```yaml
model:
  patch_icl:
    levels:
      - resolution: 32        # Processing resolution
        patch_size: 16        # Patch dimensions
        num_patches: 32       # Patches per image
        sampling_temperature: 0.001

    oracle_levels_train: [false]  # Use GT for sampling (train)
    oracle_levels_valid: [false]  # Use GT for sampling (val)

    sampler:
      type: "continuous"      # "continuous", "weighted", "topk", "sliding_window"
      stride_divisor: 4

    aggregator:
      type: "gaussian"        # "gaussian", "average", "confidence", "learned"
      combine_mode: "coverage"  # "coverage", "average", "replace"
      sigma_ratio: 0.125
      min_coverage: 0.01

    backbone:
      type: "simple"          # "simple", "perceiver", "multilayer", "cnn"
      embed_dim: 256
      cross_attention:
        num_heads: 32
        num_layers: 4
        num_registers: 8
        dropout: 0.1
```

### Loss Config

```yaml
loss:
  patch_loss:
    type: "diceCE"            # Loss on individual patches
  aggreg_loss:
    type: "diceCE"            # Loss on aggregated mask
  weights:
    default:
      target_patch: 1.0       # Weight for target patch loss
      target_aggreg: 0.0      # Weight for target aggregated loss
      context_patch: 0.0      # Weight for context patch loss
      context_aggreg: 0.0     # Weight for context aggregated loss
```

## Directory Structure

```
configs/
├── experiment/           # Experiment configs
│   ├── patch_icl_v1.yaml
│   └── patch_icl_v2.yaml
├── cluster/              # Cluster-specific paths
└── train.yaml            # Base training config

src/
├── models/
│   ├── patch_icl.py      # Main PatchICL model
│   ├── aggregate.py      # Patch aggregation strategies
│   ├── sampling.py       # Patch sampling strategies
│   └── backbone.py       # Cross-attention backbones
└── train_utils.py        # Training utilities
```
