## Project Overview

In-context medical image segmentation research. The model receives a target image and context image/mask pairs, then outputs a segmentation for the target.

## Code Guidelines

- Write understandable code with short docstrings
- Log changes to README.md and research notes to logs.md
- Write tests only when necessary
- Related repos in `/software/notebooks/camaret/repos`: Medverse, nnInteractive_fork, PatchWork, Neuroverse3D

## File Structure

```
src/
├── models/
│   ├── patch_icl.py          # Main PatchICL model (multi-resolution patch-based ICL)
│   ├── backbone.py           # DINOv3 feature extraction and transformers
│   ├── aggregate.py          # Patch aggregation strategies (gaussian, confidence, etc.)
│   ├── sampler.py            # Patch sampling (gumbel, weighted, uniform)
│   └── local.py              # Local branch transformer
├── dataloaders/
│   ├── totalseg2d_dataloader.py  # 2D TotalSeg dataset with precomputed DINO features
│   └── totalseg_dataloader.py    # 3D TotalSeg dataset
├── train_utils.py            # Training/validation loops
└── medverse_foreground_sampling.py  # Medverse with foreground context sampling

scripts/
├── train.py                  # Training script
├── eval_totalseg.py          # Evaluation script
└── precompute_dino_features.py  # Precompute DINO features for training
```

## Configuration

All settings in `config.yaml`:

- **paths**: Data directories, checkpoint paths
- **train**: Dataset, batch sizes, optimizer, scheduler, loss
- **model_params.patch_icl**: Multi-resolution levels, sampling, aggregation, backbone
- **eval**: Evaluation dataset and settings

Key PatchICL settings:
```yaml
model_params:
  patch_icl:
    levels:                    # Multi-resolution pyramid
      - resolution: 32
        patch_size: 16
        num_patches: 16
    oracle_levels_train: [true]   # GT-guided sampling during training
    oracle_levels_valid: [false]  # Uniform sampling during validation
    sampler:
      type: "gumbel"           # Differentiable patch selection
    backbone:
      type: "precomputed"      # Use precomputed DINOv3 features
      embed_dim: 1024          # DINOv3 ViT-L feature dimension
```

## Commands

```bash
# Train PatchICL
python scripts/train.py

# Evaluate on TotalSeg
python scripts/eval_totalseg.py --context-size 3 --no-wandb

# Precompute DINO features (required for precomputed backbone)
python scripts/precompute_dino_features.py
```

## Architecture Notes

**PatchICL**: Multi-resolution patch-based in-context learning
1. Downsample target/context to coarse resolution
2. Sample K patches using gumbel-softmax (differentiable)
3. Extract DINOv3 features for each patch
4. Process with transformer (target + context patches)
5. Aggregate patch predictions back to full resolution
6. Repeat at finer resolutions (optional)

**Precomputed Features**: DINOv3 ViT-L extracts 14x14 grid of 1024-dim tokens from 224x224 input. Features are precomputed and stored to speed up training.

**Oracle vs Uniform Sampling**: Training uses GT mask to guide patch sampling (oracle). Validation uses uniform sampling to simulate test conditions.
