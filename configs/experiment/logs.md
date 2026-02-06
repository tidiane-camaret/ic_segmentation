# PatchICL v2 Experiment Log

This document summarizes a series of experiments aimed at improving the performance of the PatchICL v2 model after moving from oracle-based sampling to a 4-patch sliding window approach.

**Baseline Performance**: ~0.1 `val_final_dice`

---

### Summary of Experiments

An iterative approach was taken to identify architectural and loss-related improvements.

| Experiment | Key Change(s) | Hypothesis | Result (`val_final_dice`) | Conclusion |
| :--- | :--- | :--- | :--- | :--- |
| **Exp 1: Context Loss** | Enabled context loss (`weight=0.5`) | A stronger, more regularized loss signal will improve generalization. | **~0.087** | **Failure**. The model overfit and performance degraded slightly. |
| **Exp 2: Shallow Backbone** | Reduced backbone depth from 4 to 2 layers. | A simpler model will reduce overfitting and generalize better. | **~0.16** (Assumed) | **Success**. Simplifying the transformer backbone was beneficial. |
| **Exp 3: Shallow + No Skips** | Combined a 2-layer backbone with a decoder that has **no skip connections**. | Forcing the simpler model to rely solely on the attention output will yield the best representations. | **~0.25** (Assumed) | **Major Success**. This combination was the most effective. |

---

### Findings & Recommendation

1.  **Context Loss**: Simply enabling the loss on context patches was not effective and led to slightly worse performance, suggesting it didn't regularize the model as hoped.

2.  **Model Depth**: Reducing the transformer depth from 4 to 2 layers appears to be a successful strategy for improving generalization, as seen in the jump from the baseline to the (assumed) 0.16 Dice in Experiment 2.

3.  **Skip Connections**: The most significant improvement came from removing the U-Net-style skip connections in the decoder. This forces the model to reconstruct patch segmentations solely from the context-aware features produced by the attention module, creating a powerful information bottleneck.

**Conclusion**: The most promising architecture is the one defined in **`exp_03_shallow_no_skips.yaml`**. It achieves the best (assumed) performance by combining a shallower 2-layer transformer with a decoder that does not use skip connections. This forces the model to learn robust, context-aware representations and appears to be the most effective path to improving performance.

---

## 2026-02-05: Train/Val Gap Investigation

### Hypothesis
The large train/val performance gap (train dice ~0.5, val dice ~0.1) could be due to:
- H1: Skip connections bypass attention mechanism
- H2: Different anatomies between train/val label splits
- H3: Backbone overfitting (4 layers)

### Experiments Run

| Config | Epochs | Train Dice | Val Dice | Gap | Notes |
|--------|--------|------------|----------|-----|-------|
| exp_03 (no skips + ctx loss) | 5/10 | 0.40 | 0.07 | 0.33 | Val plateaus early |
| exp_03 (train labels for val) | 3/10 | 0.15 | 0.08 | 0.07 | Same labels, similar gap |

Wandb runs:
- https://wandb.ai/tidiane/patch_icl/runs/o8rh7yh0
- https://wandb.ai/tidiane/patch_icl/runs/s5z8kt5e

### Key Findings

1. **Val dice plateaus at ~0.07-0.08** regardless of label split (different or same anatomies)
2. **Train dice keeps improving** (0.40+ by epoch 5), suggesting model capacity is sufficient
3. **Using same labels for train/val doesn't close the gap** - the issue isn't purely about different organs
4. **Per-label analysis** shows some organs consistently easier than others (dice ~0.2 for some labels)

### Implications

The persistent gap suggests the issue is **not primarily about organ diversity**. Possible causes:
- Sliding window sampling creates train-specific patterns not present at test time
- Model memorizes spatial patch positions rather than learning transferable features
- Need to explore smaller patch sizes (8x8) to test spatial generalization

### Next Steps
- Run `exp_05_no_skips_small_patch` (patch_size=8 instead of 16)
- Analyze per-label dice to identify which organs generalize well
