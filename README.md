# PatchICL

In-context medical image segmentation using patch-based learning. The model receives a target image and context image/mask pairs, then outputs a segmentation for the target.

## Architecture

**PatchICL** samples patches from target and context images, processes them through a shared backbone with cross-attention, and aggregates predictions back to full resolution.

```
Target Image + Context Pairs → Sample Patches → Cross-Patch Attention → Aggregate → Segmentation
```

## Training

```bash
python scripts/train.py experiment=<experiment_name> cluster=<cluster_name>
```

Example:
```bash
python scripts/train.py experiment=patchicl_v1 cluster=dlclarge
```

## Evaluation

```bash
python scripts/eval.py experiment=<experiment_name> cluster=<cluster_name>
```

Checkpoint path is set in cluster config under `paths.ckpts.patch_icl`.

## Config

- `configs/experiment/` - experiment configs
- `configs/cluster/` - cluster-specific paths
- `configs/train.yaml` - base training config
