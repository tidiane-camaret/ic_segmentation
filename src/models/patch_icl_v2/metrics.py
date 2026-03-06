"""Centralized metrics computation for PatchICL.

All dice and metric calculations should go through this module to ensure
consistency in thresholds and naming.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


# Thresholds for hard-dice metric binarization
PRED_THRESHOLD = 0.5      # sigmoid probability -> binary 
GT_AREA_THRESHOLD = 0.5  # soft avg-pooled GT -> binary 


def _resize_label(label: torch.Tensor, size: tuple[int, int], min_value: float = 1) -> torch.Tensor:
    """Hybrid label resize: area interpolation + max pooling.

    Mirrors the _resize_mask approach in the dataloader: area pooling gives soft
    coverage fractions but erases small objects; max pooling preserves their
    presence. The hybrid ensures any detected foreground contributes >= min_value.

    Args:
        label: [B, C, H, W] float tensor
        size: target (H, W)
        min_value: minimum foreground weight when max pool detects presence

    Returns:
        Resized label [B, C, *size]
    """
    area = F.interpolate(label, size=size, mode='area')
    maxp = F.adaptive_max_pool2d(label, size)
    return torch.maximum(area, maxp * min_value)


def compute_pixel_mae(
    pred: torch.Tensor,
    target: torch.Tensor,
    apply_sigmoid: bool = True,
) -> torch.Tensor:
    """Mean absolute error between soft pred and soft GT, per sample.

    Works at any resolution without binarization. At coarse resolutions (8x8),
    this is more informative than dice which requires binarization.

    Args:
        pred: Prediction logits [B, C, H, W]
        target: Soft ground truth [B, C, H, W] (e.g. from avg_pool2d)
        apply_sigmoid: Whether to apply sigmoid to pred

    Returns:
        Per-sample MAE tensor [B] (lower is better)
    """
    pred_probs = torch.sigmoid(pred).float() if apply_sigmoid else pred.float()
    target_f = target.float()
    spatial_dims = tuple(range(1, pred.dim()))
    return (pred_probs - target_f).abs().mean(dim=spatial_dims)


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

    # Detect empty GT from continuous values (threshold-free)
    gt_has_fg = target_float.sum(dim=spatial_dims) > 0

    # Hard dice
    intersection = (pred_binary * target_binary).sum(dim=spatial_dims)
    union = pred_binary.sum(dim=spatial_dims) + target_binary.sum(dim=spatial_dims)
    dice = (2 * intersection + 1e-6) / (union + 1e-6)
    dice = torch.where(gt_has_fg, dice, torch.zeros_like(dice))

    result = {'dice': dice, 'gt_has_foreground': gt_has_fg}

    # Soft dice (no binarization — uses continuous pred and GT directly)
    if return_soft:
        pred_probs_f32 = pred_probs.float()
        target_float_f32 = target_float.float()

        soft_intersection = (pred_probs_f32 * target_float_f32).sum(dim=spatial_dims)
        soft_denom = pred_probs_f32.sum(dim=spatial_dims) + target_float_f32.sum(dim=spatial_dims)
        soft_dice = (2 * soft_intersection + 1e-6) / (soft_denom + 1e-6)
        soft_dice = torch.where(gt_has_fg, soft_dice, torch.zeros_like(soft_dice))
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
        labels_ds = _resize_label(labels.float(), size=level_pred.shape[-2:])

        dice_result = compute_dice(level_pred, labels_ds)
        metrics[f'level_{li}_dice'] = dice_result['dice'].mean()
        metrics[f'level_{li}_soft_dice'] = dice_result['soft_dice'].mean()
        metrics[f'level_{li}_pixel_mae'] = compute_pixel_mae(level_pred, labels_ds).mean()

        # Refined probs dice (sampling guidance quality from progressive refinement)
        refined_probs = level_out.get('refined_probs')
        if refined_probs is not None:
            rp_gt = _resize_label(labels.float(), size=refined_probs.shape[-2:])

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
    return_per_sample: bool = False,
) -> dict[str, torch.Tensor]:
    """Compute all dice metrics from model outputs.

    Args:
        outputs: Model output dict from forward()
        labels: Full-resolution ground truth [B, C, H, W]
        return_per_sample: If True, also return per-sample dice in 'per_sample_dice'

    Returns:
        Dict with all dice metrics, and optionally 'per_sample_dice': [B]
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

        # Final metrics: use final_logit (upsampled combined_pred, not raw last level pred)
        # combined_pred blends all levels via confidence-weighted alpha, which is the actual model output
        final_logit = outputs.get('final_logit')
        if final_logit is not None:
            dice_result = compute_dice(final_logit, labels)
            metrics['final_dice'] = dice_result['dice'].mean()
            metrics['final_soft_dice'] = dice_result['soft_dice'].mean()
            metrics['final_pixel_mae'] = compute_pixel_mae(final_logit, labels).mean()
        else:
            # Fallback: upsample last level pred (legacy behavior)
            last_pred = level_outputs[-1]['pred']
            last_res = last_pred.shape[-1]
            full_res = labels.shape[-1]
            if last_res < full_res:
                pred_upsampled = F.interpolate(
                    last_pred, size=(full_res, full_res),
                    mode='bilinear', align_corners=False,
                )
            else:
                pred_upsampled = last_pred
            dice_result = compute_dice(pred_upsampled, labels)
            metrics['final_dice'] = dice_result['dice'].mean()
            metrics['final_soft_dice'] = dice_result['soft_dice'].mean()
            metrics['final_pixel_mae'] = compute_pixel_mae(pred_upsampled, labels).mean()

        # Store per-sample dice to avoid redundant interpolation
        if return_per_sample:
            metrics['per_sample_dice'] = dice_result['dice']

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
            if return_per_sample:
                metrics['per_sample_dice'] = dice_result['dice']

    # Uncertainty metrics from final_conf or final_logit
    metrics.update(compute_uncertainty_metrics(outputs, labels))

    return metrics


def compute_uncertainty_metrics(
    outputs: dict[str, torch.Tensor],
    labels: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Compute uncertainty/confidence metrics from model outputs.

    Computes entropy-based metrics from final_conf (if provided) or from
    final_logit. Reports mean confidence, mean entropy, and error-uncertainty
    correlation (AUCO-like: whether uncertain pixels are also wrong).

    Args:
        outputs: Model output dict (needs 'final_logit', optionally 'final_conf')
        labels: Ground truth [B, C, H, W]

    Returns:
        Dict with uncertainty metrics (all scalar tensors)
    """
    metrics = {}
    final_logit = outputs.get('final_logit')
    if final_logit is None:
        return metrics

    if labels.dim() == 3:
        labels = labels.unsqueeze(1)

    with torch.no_grad():
        p = torch.sigmoid(final_logit).clamp(1e-6, 1 - 1e-6)
        entropy = -(p * p.log() + (1 - p) * (1 - p).log()) / math.log(2)
        confidence = 1.0 - entropy  # [0, 1]

        # Mean entropy and confidence across all pixels
        metrics['mean_entropy'] = entropy.mean()
        metrics['mean_confidence'] = confidence.mean()

        # Error-weighted entropy: avg entropy where predictions are wrong
        pred_binary = (p > PRED_THRESHOLD).float()
        gt_binary = (labels.float() > GT_AREA_THRESHOLD).float()
        errors = (pred_binary != gt_binary).float()
        n_errors = errors.sum()
        if n_errors > 0:
            metrics['mean_entropy_on_errors'] = (entropy * errors).sum() / n_errors
            metrics['mean_entropy_on_correct'] = (
                (entropy * (1 - errors)).sum() / (1 - errors).sum().clamp(min=1)
            )

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

    # Use final_logit (upsampled combined_pred) which is the actual model output
    final_logit = outputs.get('final_logit')
    if final_logit is not None:
        dice_result = compute_dice(final_logit, labels)
    else:
        # Fallback: upsample last level pred (legacy behavior)
        level_outputs = outputs.get('level_outputs', [])
        if level_outputs:
            last_pred = level_outputs[-1]['pred']
            last_res = last_pred.shape[-1]
            full_res = labels.shape[-1]
            if last_res < full_res:
                pred_upsampled = F.interpolate(
                    last_pred, size=(full_res, full_res),
                    mode='bilinear', align_corners=False,
                )
            else:
                pred_upsampled = last_pred
            dice_result = compute_dice(pred_upsampled, labels)
        else:
            raise ValueError("No final_logit or level_outputs in model outputs")

    return dice_result['dice']
