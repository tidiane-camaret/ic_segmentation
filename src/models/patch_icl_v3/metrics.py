"""Centralized metrics computation for PatchICL v3.

Essential metrics only - focused on what matters for segmentation and sampling quality.

ESSENTIAL METRICS:
- final_dice, final_soft_dice: Ultimate segmentation quality
- level_{i}_dice: Per-level segmentation quality
- level_{i}_error_targeting_lift: Sampling quality (>1 = better than random)
- level_{i}_level_improvement: Multi-level contribution

SECONDARY (logged less frequently):
- level_{i}_error_recall/precision: Detailed sampling analysis
- level_{i}_coverage_ratio: Patch budget check
- mean_entropy: Uncertainty calibration
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .utils import resize_label


# Thresholds for hard-dice metric binarization
PRED_THRESHOLD = 0.5
GT_AREA_THRESHOLD = 0.5


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
        gt_threshold: Threshold for binarizing GT
        spatial_dims: Dimensions to sum over. If None, auto-detected.
        return_soft: Whether to also compute soft dice
        apply_sigmoid: Whether to apply sigmoid to pred

    Returns:
        Dict with 'dice' and optionally 'soft_dice'
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

    gt_has_fg = target_float.sum(dim=spatial_dims) > 0

    # Hard dice
    intersection = (pred_binary * target_binary).sum(dim=spatial_dims)
    union = pred_binary.sum(dim=spatial_dims) + target_binary.sum(dim=spatial_dims)
    dice = (2 * intersection + 1e-6) / (union + 1e-6)
    dice = torch.where(gt_has_fg, dice, torch.zeros_like(dice))

    result = {'dice': dice, 'gt_has_foreground': gt_has_fg}

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

    Note: level_{i}_dice uses resize_label (max-pooled GT) - biased for coarse levels.
    Use level_{i}_dice_fair for unbiased comparison (upsamples pred to full res).
    """
    metrics = {}
    full_res = labels.shape[-2:]

    for li, level_out in enumerate(level_outputs):
        level_pred = level_out['pred']
        level_res = level_pred.shape[-2:]

        # Standard metric (uses max-pooled GT at level resolution) - BIASED
        labels_ds = resize_label(labels.float(), size=level_res)
        dice_result = compute_dice(level_pred, labels_ds)
        metrics[f'level_{li}_dice'] = dice_result['dice'].mean()
        metrics[f'level_{li}_soft_dice'] = dice_result['soft_dice'].mean()

        # Fair metric: upsample pred to full res, compare with original GT - UNBIASED
        if level_res != full_res:
            pred_upsampled = F.interpolate(
                level_pred, size=full_res, mode='bilinear', align_corners=False
            )
            dice_fair = compute_dice(pred_upsampled, labels.float())
            metrics[f'level_{li}_dice_fair'] = dice_fair['dice'].mean()
            metrics[f'level_{li}_soft_dice_fair'] = dice_fair['soft_dice'].mean()
        else:
            # At full resolution, fair = standard
            metrics[f'level_{li}_dice_fair'] = metrics[f'level_{li}_dice']
            metrics[f'level_{li}_soft_dice_fair'] = metrics[f'level_{li}_soft_dice']

    return metrics


def compute_context_metrics(
    context_pred: torch.Tensor,
    context_labels: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Compute context prediction dice metrics."""
    spatial_dims = (2, 3, 4)
    dice_result = compute_dice(context_pred, context_labels, spatial_dims=spatial_dims)

    return {
        'context_dice': dice_result['dice'].mean(),
        'context_soft_dice': dice_result['soft_dice'].mean(),
    }


def compute_hierarchical_metrics(
    level_outputs: list[dict],
    labels: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Compute hierarchical metrics: level improvement and sampling quality."""
    metrics = {}

    def masked_softdice(pred, gt, mask):
        pred_prob = torch.sigmoid(pred)
        pred_masked = pred_prob * mask.float()
        gt_masked = gt * mask.float()
        intersection = (pred_masked * gt_masked).sum(dim=(1, 2, 3))
        denom = pred_masked.sum(dim=(1, 2, 3)) + gt_masked.sum(dim=(1, 2, 3))
        mask_sum = mask.float().sum(dim=(1, 2, 3))
        valid = mask_sum > 0
        dice = torch.where(valid, (2 * intersection + 1e-6) / (denom + 1e-6), torch.zeros_like(intersection))
        return dice.mean()

    def compute_entropy(pred):
        p = torch.sigmoid(pred).clamp(1e-6, 1 - 1e-6)
        return -(p * p.log() + (1 - p) * (1 - p).log()) / math.log(2)

    full_res = labels.shape[-2:]

    with torch.no_grad():
        # Per-level uncertainty calibration metrics
        for li, level_out in enumerate(level_outputs):
            level_pred = level_out['pred']
            level_res = level_pred.shape[-2:]

            # Standard calibration (uses max-pooled GT) - BIASED for coarse levels
            labels_li = resize_label(labels.float(), size=level_res)
            entropy = compute_entropy(level_pred)
            pred_binary = (torch.sigmoid(level_pred) > 0.5).float()
            gt_binary = (labels_li > 0.5).float()
            error_map = (pred_binary != gt_binary).float()

            mean_entropy = entropy.mean()
            metrics[f'level_{li}_mean_entropy'] = mean_entropy

            # Standard calibration correlation (biased)
            entropy_flat = entropy.view(-1)
            error_flat = error_map.view(-1)
            ent_centered = entropy_flat - entropy_flat.mean()
            err_centered = error_flat - error_flat.mean()
            corr_num = (ent_centered * err_centered).sum()
            corr_den = (ent_centered.pow(2).sum() * err_centered.pow(2).sum()).sqrt() + 1e-8
            calibration_corr = corr_num / corr_den
            metrics[f'level_{li}_uncertainty_calibration'] = calibration_corr

            n_errors = error_map.sum().clamp(min=1)
            n_correct = (1 - error_map).sum().clamp(min=1)
            entropy_on_errors = (entropy * error_map).sum() / n_errors
            entropy_on_correct = (entropy * (1 - error_map)).sum() / n_correct
            metrics[f'level_{li}_entropy_ratio'] = entropy_on_errors / (entropy_on_correct + 1e-6)

            # FAIR calibration: upsample pred to full res, compare with original GT
            if level_res != full_res:
                pred_up = F.interpolate(level_pred, size=full_res, mode='bilinear', align_corners=False)
                entropy_up = F.interpolate(entropy, size=full_res, mode='bilinear', align_corners=False)
                pred_binary_up = (torch.sigmoid(pred_up) > 0.5).float()
                gt_binary_full = (labels.float() > 0.5).float()
                error_map_fair = (pred_binary_up != gt_binary_full).float()

                # Fair calibration correlation
                entropy_flat_fair = entropy_up.view(-1)
                error_flat_fair = error_map_fair.view(-1)
                ent_c = entropy_flat_fair - entropy_flat_fair.mean()
                err_c = error_flat_fair - error_flat_fair.mean()
                corr_fair = (ent_c * err_c).sum() / ((ent_c.pow(2).sum() * err_c.pow(2).sum()).sqrt() + 1e-8)
                metrics[f'level_{li}_uncertainty_calibration_fair'] = corr_fair

                n_err_fair = error_map_fair.sum().clamp(min=1)
                n_cor_fair = (1 - error_map_fair).sum().clamp(min=1)
                ent_on_err_fair = (entropy_up * error_map_fair).sum() / n_err_fair
                ent_on_cor_fair = (entropy_up * (1 - error_map_fair)).sum() / n_cor_fair
                metrics[f'level_{li}_entropy_ratio_fair'] = ent_on_err_fair / (ent_on_cor_fair + 1e-6)
            else:
                metrics[f'level_{li}_uncertainty_calibration_fair'] = calibration_corr
                metrics[f'level_{li}_entropy_ratio_fair'] = metrics[f'level_{li}_entropy_ratio']
        for i in range(1, len(level_outputs)):
            prev_level = level_outputs[i - 1]
            curr_level = level_outputs[i]

            prev_pred = prev_level['pred']
            curr_pred = curr_level['pred']
            coverage_mask = curr_level.get('coverage_mask')

            if coverage_mask is None:
                continue

            curr_res = curr_pred.shape[-2:]

            prev_pred_up = F.interpolate(prev_pred, size=curr_res, mode='bilinear', align_corners=False)
            labels_curr = F.interpolate(labels.float(), size=curr_res, mode='area')

            covered = coverage_mask > 0.5
            covered_exp = covered.expand_as(curr_pred)

            # Level improvement
            prev_dice = masked_softdice(prev_pred_up, labels_curr, covered_exp)
            curr_dice = masked_softdice(curr_pred, labels_curr, covered_exp)
            metrics[f'level_{i}_level_improvement'] = curr_dice - prev_dice

            # Coverage ratio
            metrics[f'level_{i}_coverage_ratio'] = covered.float().mean()

            # Error targeting
            prev_prob = torch.sigmoid(prev_pred_up)
            prev_binary = (prev_prob > 0.5).float()
            gt_binary = (labels_curr > 0.5).float()
            prev_error = (prev_binary != gt_binary).float()
            prev_error_any = prev_error.max(dim=1, keepdim=True)[0]
            covered_f = covered.float()

            error_pixels = prev_error_any.sum(dim=(2, 3)) + 1e-6
            covered_error_pixels = (prev_error_any * covered_f).sum(dim=(2, 3))
            covered_pixels = covered_f.sum(dim=(2, 3)) + 1e-6

            error_recall = (covered_error_pixels / error_pixels).mean()
            error_precision = (covered_error_pixels / covered_pixels).mean()
            error_ratio = prev_error_any.mean()

            metrics[f'level_{i}_error_recall'] = error_recall
            metrics[f'level_{i}_error_precision'] = error_precision
            metrics[f'level_{i}_error_targeting_lift'] = error_precision / (error_ratio + 1e-6)

    return metrics


def compute_all_metrics(
    outputs: dict[str, torch.Tensor],
    labels: torch.Tensor,
    return_per_sample: bool = False,
) -> dict[str, torch.Tensor]:
    """Compute essential metrics from model outputs."""
    if labels.dim() == 3:
        labels = labels.unsqueeze(1)

    metrics = {}

    level_outputs = outputs.get('level_outputs', [])
    if level_outputs:
        metrics.update(compute_level_metrics(level_outputs, labels))

        if len(level_outputs) > 1:
            metrics.update(compute_hierarchical_metrics(level_outputs, labels))

        final_logit = outputs.get('final_logit')
        if final_logit is not None:
            dice_result = compute_dice(final_logit, labels)
            metrics['final_dice'] = dice_result['dice'].mean()
            metrics['final_soft_dice'] = dice_result['soft_dice'].mean()
        else:
            last_pred = level_outputs[-1]['pred']
            full_res = labels.shape[-1]
            if last_pred.shape[-1] < full_res:
                pred_upsampled = F.interpolate(
                    last_pred, size=(full_res, full_res),
                    mode='bilinear', align_corners=False,
                )
            else:
                pred_upsampled = last_pred
            dice_result = compute_dice(pred_upsampled, labels)
            metrics['final_dice'] = dice_result['dice'].mean()
            metrics['final_soft_dice'] = dice_result['soft_dice'].mean()

        if return_per_sample:
            metrics['per_sample_dice'] = dice_result['dice']

        last_level = level_outputs[-1]
        context_pred = last_level.get('context_pred')
        context_labels = last_level.get('context_labels')
        if context_pred is not None and context_labels is not None:
            metrics.update(compute_context_metrics(context_pred, context_labels))
    else:
        final_logit = outputs.get('final_logit')
        if final_logit is not None:
            dice_result = compute_dice(final_logit, labels)
            metrics['final_dice'] = dice_result['dice'].mean()
            metrics['final_soft_dice'] = dice_result['soft_dice'].mean()
            if return_per_sample:
                metrics['per_sample_dice'] = dice_result['dice']

    metrics.update(compute_uncertainty_metrics(outputs, labels))

    return metrics


def compute_uncertainty_metrics(
    outputs: dict[str, torch.Tensor],
    labels: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Compute uncertainty/confidence metrics."""
    metrics = {}
    final_logit = outputs.get('final_logit')
    if final_logit is None:
        return metrics

    if labels.dim() == 3:
        labels = labels.unsqueeze(1)

    with torch.no_grad():
        p = torch.sigmoid(final_logit).clamp(1e-6, 1 - 1e-6)
        entropy = -(p * p.log() + (1 - p) * (1 - p).log()) / math.log(2)
        confidence = 1.0 - entropy

        metrics['mean_entropy'] = entropy.mean()
        metrics['mean_confidence'] = confidence.mean()

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
    """Compute per-sample final dice for per-label tracking."""
    if labels.dim() == 3:
        labels = labels.unsqueeze(1)

    final_logit = outputs.get('final_logit')
    if final_logit is not None:
        dice_result = compute_dice(final_logit, labels)
    else:
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
