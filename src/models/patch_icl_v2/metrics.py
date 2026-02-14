"""Centralized metrics computation for PatchICL.

All dice and metric calculations should go through this module to ensure
consistency in thresholds and naming.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


# Thresholds for hard-dice metric binarization
PRED_THRESHOLD = 0.5      # sigmoid probability -> binary prediction
GT_AREA_THRESHOLD = 0.25  # soft avg-pooled GT -> binary (>=25% coverage = foreground)


def compute_dice(
    pred: torch.Tensor,
    target: torch.Tensor,
    pred_threshold: float = PRED_THRESHOLD,
    gt_threshold: float = GT_AREA_THRESHOLD,
    spatial_dims: tuple[int, ...] | None = None,
    return_soft: bool = True,
    apply_sigmoid: bool = True,
) -> dict[str, torch.Tensor]:
    """Compute hard and soft dice scores.

    Args:
        pred: Prediction logits or probabilities [B, C, ...] or [B, K, C, ...]
        target: Ground truth [B, C, ...] or [B, K, C, ...]
        pred_threshold: Threshold for binarizing predictions
        gt_threshold: Threshold for binarizing GT (for area-based soft labels)
        spatial_dims: Dimensions to sum over. If None, auto-detected as all dims after first 2.
        return_soft: Whether to also compute soft dice
        apply_sigmoid: Whether to apply sigmoid to pred (set False if already probabilities)

    Returns:
        Dict with 'dice' and optionally 'soft_dice', each [B] or [B, K]
    """
    if apply_sigmoid:
        pred_probs = torch.sigmoid(pred)
    else:
        pred_probs = pred

    pred_binary = (pred_probs > pred_threshold).float()
    target_float = target.float()
    target_binary = (target_float > gt_threshold).float()

    if spatial_dims is None:
        spatial_dims = tuple(range(2, pred.dim()))

    # Hard dice
    intersection = (pred_binary * target_binary).sum(dim=spatial_dims)
    union = pred_binary.sum(dim=spatial_dims) + target_binary.sum(dim=spatial_dims)
    dice = (2 * intersection + 1e-6) / (union + 1e-6)

    result = {'dice': dice}

    # Soft dice
    if return_soft:
        # Cast to float32 for stable calculation, especially with fp16
        pred_probs_f32 = pred_probs.float()
        target_float_f32 = target_float.float()

        soft_intersection = (pred_probs_f32 * target_float_f32).sum(dim=spatial_dims)
        soft_denom = pred_probs_f32.sum(dim=spatial_dims) + target_float_f32.sum(dim=spatial_dims)
        soft_dice = (2 * soft_intersection + 1e-6) / (soft_denom + 1e-6)
        result['soft_dice'] = soft_dice

    return result


def compute_level_metrics(
    level_outputs: list[dict],
    labels: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Compute per-level dice metrics.

    Args:
        level_outputs: List of level output dicts from model forward
        labels: Full-resolution ground truth [B, C, H, W]

    Returns:
        Dict with level_{i}_dice, level_{i}_soft_dice for each level
    """
    metrics = {}

    for li, level_out in enumerate(level_outputs):
        level_pred = level_out['pred']
        level_res = level_pred.shape[-1]
        scale_factor = labels.shape[-1] // level_res
        labels_ds = F.avg_pool2d(labels.float(), kernel_size=scale_factor, stride=scale_factor)

        dice_result = compute_dice(level_pred, labels_ds)
        metrics[f'level_{li}_dice'] = dice_result['dice'].mean()
        metrics[f'level_{li}_soft_dice'] = dice_result['soft_dice'].mean()

        # Refined probs dice (sampling guidance quality from progressive refinement)
        refined_probs = level_out.get('refined_probs')
        if refined_probs is not None:
            rp_res = refined_probs.shape[-1]
            rp_sf = labels.shape[-1] // rp_res
            rp_gt = F.avg_pool2d(labels.float(), kernel_size=rp_sf, stride=rp_sf)

            # Soft dice (refined_probs is already probabilities)
            rp_dice = compute_dice(refined_probs, rp_gt, apply_sigmoid=False)
            metrics[f'level_{li}_refined_probs_dice'] = rp_dice['dice'].mean()
            metrics[f'level_{li}_refined_probs_soft_dice'] = rp_dice['soft_dice'].mean()

    return metrics


def compute_patch_metrics(
    patch_logits: torch.Tensor,
    patch_labels: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Compute patch-level (local) dice metrics.

    Args:
        patch_logits: [B, K, C, H, W] patch predictions
        patch_labels: [B, K, C, H, W] patch ground truth

    Returns:
        Dict with local_dice, local_soft_dice
    """
    # Sum over (C, H, W) within each patch, then mean over K
    spatial_dims = (2, 3, 4)
    dice_result = compute_dice(patch_logits, patch_labels, spatial_dims=spatial_dims)

    return {
        'local_dice': dice_result['dice'].mean(),
        'local_soft_dice': dice_result['soft_dice'].mean(),
    }


def compute_context_metrics(
    context_pred: torch.Tensor,
    context_labels: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Compute context prediction dice metrics.

    Args:
        context_pred: [B, k, C, H, W] context predictions
        context_labels: [B, k, C, H, W] context ground truth

    Returns:
        Dict with context_dice, context_soft_dice
    """
    # Sum over (C, H, W), keep batch and context dims
    spatial_dims = (2, 3, 4)
    dice_result = compute_dice(context_pred, context_labels, spatial_dims=spatial_dims)

    return {
        'context_dice': dice_result['dice'].mean(),
        'context_soft_dice': dice_result['soft_dice'].mean(),
    }


def compute_all_metrics(
    outputs: dict[str, torch.Tensor],
    labels: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Compute all dice metrics from model outputs.

    Args:
        outputs: Model output dict from forward()
        labels: Full-resolution ground truth [B, C, H, W]

    Returns:
        Dict with all dice metrics
    """
    if labels.dim() == 3:
        labels = labels.unsqueeze(1)

    metrics = {}

    # Patch-level (local) metrics
    patch_logits = outputs.get('patch_logits')
    patch_labels = outputs.get('patch_labels')
    if patch_logits is not None and patch_labels is not None:
        metrics.update(compute_patch_metrics(patch_logits, patch_labels))

    # Per-level metrics
    level_outputs = outputs.get('level_outputs', [])
    if level_outputs:
        metrics.update(compute_level_metrics(level_outputs, labels))

        # Final dice at last level's resolution (matches training loss resolution)
        last_pred = level_outputs[-1]['pred']
        last_res = last_pred.shape[-1]
        sf = labels.shape[-1] // last_res
        labels_ds = F.avg_pool2d(labels.float(), kernel_size=sf, stride=sf)
        dice_result = compute_dice(last_pred, labels_ds)
        metrics['final_dice'] = dice_result['dice'].mean()
        metrics['final_soft_dice'] = dice_result['soft_dice'].mean()

        # Context metrics from last level
        last_level = level_outputs[-1]
        context_pred = last_level.get('context_pred')
        context_labels = last_level.get('context_labels')
        if context_pred is not None and context_labels is not None:
            metrics.update(compute_context_metrics(context_pred, context_labels))
    else:
        # Fallback for non-level outputs
        final_logit = outputs.get('final_logit')
        if final_logit is not None:
            dice_result = compute_dice(final_logit, labels)
            metrics['final_dice'] = dice_result['dice'].mean()
            metrics['final_soft_dice'] = dice_result['soft_dice'].mean()

    return metrics


def compute_per_sample_dice(
    outputs: dict[str, torch.Tensor],
    labels: torch.Tensor,
) -> torch.Tensor:
    """Compute per-sample final dice for per-label tracking.

    Args:
        outputs: Model output dict
        labels: Ground truth [B, C, H, W]

    Returns:
        Per-sample dice tensor [B]
    """
    if labels.dim() == 3:
        labels = labels.unsqueeze(1)

    level_outputs = outputs.get('level_outputs', [])
    if level_outputs:
        last_pred = level_outputs[-1]['pred']
        last_res = last_pred.shape[-1]
        sf = labels.shape[-1] // last_res
        labels_ds = F.avg_pool2d(labels.float(), kernel_size=sf, stride=sf)
        dice_result = compute_dice(last_pred, labels_ds)
    else:
        final_logit = outputs.get('final_logit')
        if final_logit is not None:
            dice_result = compute_dice(final_logit, labels)
        else:
            raise ValueError("No level_outputs or final_logit in model outputs")

    return dice_result['dice']
