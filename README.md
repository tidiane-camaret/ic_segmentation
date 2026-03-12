# PatchICL

In-context medical image segmentation using patch-based learning. The model receives a target image and context image/mask pairs, then outputs a segmentation for the target.

## Architecture

**PatchICL** uses a multi-level coarse-to-fine architecture. Each level samples patches, processes them through cross-attention with context, and aggregates predictions. Finer levels use coarse predictions to guide sampling toward uncertain regions.

```
Level 0 (24×24):   Uniform sampling → Cross-attention → Coarse prediction
Level 1 (128×128): Sample uncertain regions → Refine → Blend with level 0
Level 2 (256×256): Sample remaining errors → Refine → Final prediction
```

### Key Components

| Component | Description |
|-----------|-------------|
| **Sampler** | Selects patch locations using GT (oracle) or model uncertainty |
| **Backbone** | Cross-attention between target and context patches with registers |
| **Aggregator** | Combines patch predictions into full mask (Gaussian/average weighting) |
| **Confidence** | Predicts pixel-wise confidence for uncertainty-guided sampling |

## Quick Start

```bash
# Train
python scripts/train.py experiment=112_res_256

# Evaluate
python scripts/eval.py experiment=112_res_256
```

## Configuration

### Multi-Level Architecture

```yaml
model:
  patch_icl:
    levels:
      - resolution: 24    # Coarse level
        patch_size: 8
        num_patches: 9
      - resolution: 128   # Mid level
        patch_size: 8
        num_patches: 40
        spread_sigma: 2.0  # Blur sampling weights
      - resolution: 256   # Fine level
        patch_size: 8
        num_patches: 40
        spread_sigma: 2.0
```

### Oracle Scheduling

Gradually transitions from GT-guided (oracle) to model-guided sampling during training:

```yaml
oracle_levels_train: [false, true, true]  # Level 0: uniform, levels 1-2: oracle
oracle_levels_valid: [false, false, false]  # All uniform at validation

oracle_scheduling:
  enabled: true
  schedule: "linear"
  start_prob: 0.3      # Start with 30% oracle
  end_prob: 0.0        # End with 0% oracle
  warmup_epochs: 1
  decay_epochs: 30
```

### Sampling Modes

```yaml
sampler:
  target_sampling: "gt_entropy"           # Oracle: sample from GT borders
  target_model_sampling: "predicted_uncertainty"  # Model: sample low confidence
  context_sampling: "gt_entropy"          # Context: sample from GT borders
```

| Mode | Source | Description |
|------|--------|-------------|
| `gt_foreground` | GT mask | Sample from foreground |
| `gt_entropy` | GT mask | Sample from high-entropy (border) regions |
| `predicted_uncertainty` | Model | Sample from 1 - confidence |

### Confidence Prediction

```yaml
backbone:
  predict_confidence: true  # Enable confidence head

loss:
  confidence:
    method: "learned"       # "learned" (CNN head) or "entropy" (from logits)
    supervision_weight: 0.2  # MSE(conf, 1 - max(gt_entropy, pred_error))
```

### Backbone

```yaml
backbone:
  embed_dim: 256
  num_heads: 4
  num_layers: 2
  num_context_layers: 1    # Context-first attention stage
  num_registers: 1
  use_context_mask: true   # Fuse GT masks into context embeddings
  decoder_use_skip_connections: false  # Information bottleneck
```

## Metrics

### Per-Level Metrics
- `level_{i}_dice`, `level_{i}_soft_dice` — Dice at native resolution

### Hierarchical Comparison (fair, area-only GT)
- `level_{i}_level_improvement` — Does level i improve on covered regions?
- `level_{i}_combination_effect` — Does blending help?
- `level_{i}_error_recall` — Fraction of level i-1 errors covered by level i

### Final Output
- `final_dice`, `final_soft_dice` — Full resolution dice

## Directory Structure

```
configs/
├── experiment/           # Experiment configs (112_res_256.yaml, etc.)
├── method/               # Base method configs (patch_icl.yaml)
└── cluster/              # Cluster-specific paths

src/
├── models/
│   ├── patch_icl_v2/     # Main implementation
│   │   ├── patch_icl.py  # PatchICL model
│   │   ├── sampling.py   # Patch samplers
│   │   ├── aggregate.py  # Patch aggregators
│   │   └── metrics.py    # Dice and hierarchical metrics
│   └── simple_backbone.py  # Cross-attention backbone
├── train_utils.py        # Training/validation loops
└── losses.py             # Loss functions

scripts/
├── train.py              # Training script
└── eval.py               # Evaluation script
```
