"""Single level processing for PatchICL v3.

Extracts level-specific forward logic from the main model.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .aggregate import PatchAggregator
from .sampling import ContinuousSampler, compute_sampling_weights
from .utils import (
    compute_entropy_sampling_map,
    downsample_mask,
    extract_mask_patches,
    extract_patch_features,
    mask_to_weights,
)


@dataclass
class LevelConfig:
    """Configuration for a single level."""
    resolution: int
    patch_size: int
    num_patches: int
    num_patches_val: int | None = None
    num_context_patches: int | None = None
    num_context_patches_val: int | None = None
    stride: int | None = None
    context_stride: int | None = None
    spread_sigma: float = 0.0
    spread_sigma_target: float | None = None
    spread_sigma_context: float | None = None
    sampling_temperature: float = 0.3
    sampling_method: str = "continuous"  # "continuous" or "sliding_window"
    pad_before: int | None = None
    pad_after: int | None = None


@dataclass
class LevelOutput:
    """Output from processing a single level."""
    pred: torch.Tensor  # [B, C, res, res]
    patch_logits: torch.Tensor  # [B, K, C, ps, ps]
    patch_labels: torch.Tensor  # [B, K, C, ps, ps]
    coords: torch.Tensor  # [B, K, 2]
    context_patch_logits: torch.Tensor | None
    context_patch_labels: torch.Tensor | None
    context_coords: torch.Tensor | None
    context_pred: torch.Tensor | None  # [B, k, C, res, res]
    context_labels: torch.Tensor | None  # Downsampled context labels
    context_labels_fullres: torch.Tensor | None
    target_validity: torch.Tensor | None
    context_validity: torch.Tensor | None
    attn_weights: list | None
    register_tokens: torch.Tensor | None
    patch_sampling_map: torch.Tensor | None
    aggregated_sampling_map: torch.Tensor | None
    selection_probs: torch.Tensor | None
    coverage_mask: torch.Tensor | None


def process_level(
    level_idx: int,
    level_cfg: LevelConfig,
    image: torch.Tensor,
    labels: torch.Tensor | None,
    labels_ds: torch.Tensor,
    context_in: torch.Tensor | None,
    context_out: torch.Tensor | None,
    target_features: torch.Tensor | None,
    context_features: torch.Tensor | None,
    sampling_weights: torch.Tensor,
    sampler: nn.Module,
    context_sampler: nn.Module,
    aggregator: PatchAggregator,
    backbone: nn.Module,
    augmenter: nn.Module | None,
    feature_grid_size: int,
    patch_feature_grid_size: int,
    context_sampling_mode: str,
    sampling_map_source: str,
    sampling_map_temperature: float | torch.Tensor,
    differentiable_sampling: bool,
    mask_prior: torch.Tensor | None = None,
    prev_register_tokens: torch.Tensor | None = None,
    return_attn_weights: bool = False,
) -> LevelOutput:
    """Process a single resolution level.

    Args:
        level_idx: Level index for backbone conditioning
        level_cfg: Level configuration
        image: [B, C, H, W] - Input image
        labels: [B, C, H, W] - Full resolution labels (or None)
        labels_ds: [B, C, res, res] - Pre-downsampled labels
        context_in/out: Context images and masks
        target/context_features: Pre-computed DINO features
        sampling_weights: [B, 1, res, res] - Weights for patch sampling
        sampler/context_sampler/aggregator: Level-specific modules
        backbone: Shared backbone
        augmenter: Optional patch augmenter
        feature_grid_size: Feature extractor grid size
        patch_feature_grid_size: Backbone encoder grid size
        context_sampling_mode: Mode for context patch selection
        sampling_map_source: "entropy", "learned", or "none"
        sampling_map_temperature: Temperature for entropy sampling map
        differentiable_sampling: Whether sampling is differentiable
        mask_prior: [B, 1, H_prev, W_prev] - Previous level prediction
        prev_register_tokens: [B, R, D] - Registers from previous level
        return_attn_weights: Whether to return attention weights

    Returns:
        LevelOutput with all level outputs
    """
    resolution = level_cfg.resolution
    patch_size = level_cfg.patch_size

    B = image.shape[0]
    H, W = image.shape[2], image.shape[3]
    device = image.device

    # Downsample image to level resolution
    image_ds = F.interpolate(
        image, size=(resolution, resolution),
        mode='bilinear', align_corners=False
    )

    # Sample target patches
    patches, patch_labels, coords, _, aug_params, target_validity, K, selection_probs = sampler(
        image_ds, labels_ds, sampling_weights, None
    )
    coord_scale = H / resolution

    # Extract mask prior patches if available
    mask_prior_patches = None
    if mask_prior is not None and getattr(backbone, 'use_mask_prior', False):
        mask_prior_ds = F.interpolate(
            mask_prior, size=(resolution, resolution),
            mode='bilinear', align_corners=False
        )
        mask_prior_patches = extract_mask_patches(
            mask=mask_prior_ds,
            coords=coords,
            patch_size=patch_size,
            level_resolution=resolution,
            target_size=patch_feature_grid_size,
        )

    # Extract target patch features
    target_patch_features = None
    if target_features is not None:
        target_patch_features = extract_patch_features(
            features=target_features,
            coords=coords,
            patch_size=patch_size,
            level_resolution=resolution,
            feature_grid_size=feature_grid_size,
            target_patch_grid_size=patch_feature_grid_size,
        )
        if augmenter is not None and aug_params:
            target_patch_features = augmenter.augment_features_only(target_patch_features, aug_params)

    # Process context
    context_patch_labels, context_coords = None, None
    context_patch_logits, context_pred, context_out_ds = None, None, None
    context_aug_params = None
    context_validity = None

    if context_in is not None and context_out is not None:
        k = context_in.shape[1]
        context_in_flat = context_in.view(B * k, *context_in.shape[2:])
        context_out_flat = context_out.view(B * k, *context_out.shape[2:])
        context_in_ds = F.interpolate(
            context_in_flat, size=(resolution, resolution),
            mode='bilinear', align_corners=False
        ).view(B, k, -1, resolution, resolution)
        context_out_ds = downsample_mask(context_out_flat, resolution).view(
            B, k, context_out.shape[2], resolution, resolution
        )

        # Compute context sampling weights
        context_mask_flat = mask_to_weights(
            context_out_ds.view(B * k, *context_out_ds.shape[2:])
        )
        context_weights = compute_sampling_weights(
            mode=context_sampling_mode,
            gt_mask=context_mask_flat,
        ).view(B, k, 1, resolution, resolution)

        # Select context patches (batched)
        context_patch_labels, context_coords, context_aug_params, context_validity, K_per_ctx = (
            _select_context_patches(context_in_ds, context_out_ds, context_weights, context_sampler)
        )
        K_ctx = K_per_ctx * k

        # Extract context patch features
        context_patch_features = None
        if context_features is not None:
            ctx_feats_flat = context_features.view(B * k, *context_features.shape[2:])
            ctx_coords_flat = context_coords.view(B, k, K_per_ctx, 2).reshape(B * k, K_per_ctx, 2)

            ctx_extracted_flat = extract_patch_features(
                features=ctx_feats_flat,
                coords=ctx_coords_flat,
                patch_size=patch_size,
                level_resolution=resolution,
                feature_grid_size=feature_grid_size,
                target_patch_grid_size=patch_feature_grid_size,
            )

            ctx_extracted = ctx_extracted_flat.view(B, k, K_per_ctx, *ctx_extracted_flat.shape[2:])

            # Apply augmentation per context image
            if augmenter is not None and context_aug_params is not None:
                for ctx_idx in range(k):
                    for b in range(B):
                        ctx_aug = context_aug_params[b][ctx_idx]
                        if ctx_aug and any(v is not None for v in ctx_aug.values()):
                            ctx_extracted[b, ctx_idx] = augmenter.augment_features_only(
                                ctx_extracted[b:b+1, ctx_idx], ctx_aug
                            ).squeeze(0)

            context_patch_features = ctx_extracted.view(B, K_ctx, *ctx_extracted.shape[3:])

        # Extract context mask patches if needed
        context_mask_patches = None
        if getattr(backbone, 'use_context_mask', False):
            ctx_mask_flat = context_out_ds[:, :, :1].reshape(B * k, 1, resolution, resolution)
            ctx_coords_flat = context_coords.view(B, k, K_per_ctx, 2).reshape(B * k, K_per_ctx, 2)

            ctx_mask_extracted_flat = extract_mask_patches(
                mask=ctx_mask_flat,
                coords=ctx_coords_flat,
                patch_size=patch_size,
                level_resolution=resolution,
                target_size=patch_feature_grid_size,
            )
            context_mask_patches = ctx_mask_extracted_flat.view(B, K_ctx, *ctx_mask_extracted_flat.shape[2:])

        # Prepare backbone inputs
        img_patches = torch.cat([target_patch_features, context_patch_features], dim=1)
        all_coords = torch.cat([
            coords.float() * coord_scale,
            context_coords.float() * coord_scale
        ], dim=1)
        ctx_id_labels = torch.zeros(B, K + K_ctx, dtype=torch.long, device=device)
        for ctx_idx in range(k):
            start = K + ctx_idx * K_per_ctx
            end = K + (ctx_idx + 1) * K_per_ctx
            ctx_id_labels[:, start:end] = ctx_idx + 1

        backbone_out = backbone(
            img_patches=img_patches,
            coords=all_coords,
            ctx_id_labels=ctx_id_labels,
            return_attn_weights=return_attn_weights,
            level_idx=level_idx,
            resolution=float(resolution),
            num_target_patches=K,
            mask_prior_patches=mask_prior_patches,
            context_mask_patches=context_mask_patches,
            prev_register_tokens=prev_register_tokens,
        )

        all_logits = backbone_out['mask_patch_logit_preds']
        patch_logits = all_logits[:, :K]
        context_patch_logits = all_logits[:, K:]

        # Get sampling map
        patch_sampling_map = _get_patch_sampling_map(
            backbone_out, patch_logits, K,
            sampling_map_source, sampling_map_temperature
        )

        # Aggregate context predictions
        context_preds = []
        for ctx_idx in range(k):
            start_idx = ctx_idx * K_per_ctx
            end_idx = (ctx_idx + 1) * K_per_ctx
            ctx_logits = context_patch_logits[:, start_idx:end_idx]
            ctx_coords_slice = context_coords[:, start_idx:end_idx]
            if augmenter is not None and context_aug_params is not None:
                for b in range(B):
                    ctx_aug = context_aug_params[b][ctx_idx]
                    if ctx_aug and any(v is not None for v in ctx_aug.values()):
                        ctx_logits[b:b+1] = augmenter.inverse(ctx_logits[b:b+1], ctx_aug)
            agg_result = aggregator(ctx_logits, ctx_coords_slice, (resolution, resolution))
            ctx_pred = agg_result[0] if isinstance(agg_result, tuple) else agg_result
            context_preds.append(ctx_pred)
        context_pred = torch.stack(context_preds, dim=1)
    else:
        # No context
        ctx_id_labels = torch.zeros(B, K, dtype=torch.long, device=device)
        backbone_out = backbone(
            img_patches=target_patch_features,
            coords=coords.float() * coord_scale,
            ctx_id_labels=ctx_id_labels,
            return_attn_weights=return_attn_weights,
            level_idx=level_idx,
            resolution=float(resolution),
            mask_prior_patches=mask_prior_patches,
            prev_register_tokens=prev_register_tokens,
        )
        patch_logits = backbone_out['mask_patch_logit_preds']
        patch_sampling_map = _get_patch_sampling_map(
            backbone_out, patch_logits, K,
            sampling_map_source, sampling_map_temperature
        )

    # Apply inverse augmentation and aggregate
    patch_logits_for_agg = patch_logits
    if augmenter is not None and aug_params:
        patch_logits_for_agg = augmenter.inverse(patch_logits, aug_params)

    # Differentiable sampling: weight patches by selection probability
    if selection_probs is not None:
        selection_weight = selection_probs.view(B, K, 1, 1, 1)
        patch_logits_for_agg = patch_logits_for_agg * selection_weight * K

    # Aggregate
    aggregated_sampling_map = None
    if patch_sampling_map is not None:
        pred, aggregated_sampling_map = aggregator(
            patch_logits_for_agg, coords, (resolution, resolution),
            sampling_map=patch_sampling_map
        )
    else:
        pred = aggregator(patch_logits_for_agg, coords, (resolution, resolution))

    # Create coverage mask
    coverage_mask = _create_coverage_mask(coords, patch_size, (resolution, resolution))

    return LevelOutput(
        pred=pred,
        patch_logits=patch_logits,
        patch_labels=patch_labels,
        coords=coords,
        context_patch_logits=context_patch_logits,
        context_patch_labels=context_patch_labels,
        context_coords=context_coords,
        context_pred=context_pred,
        context_labels=context_out_ds,
        context_labels_fullres=context_out,
        target_validity=target_validity,
        context_validity=context_validity,
        attn_weights=backbone_out.get('attn_weights'),
        register_tokens=backbone_out.get('register_tokens'),
        patch_sampling_map=patch_sampling_map,
        aggregated_sampling_map=aggregated_sampling_map,
        selection_probs=selection_probs,
        coverage_mask=coverage_mask,
    )


def _select_context_patches(
    context_in: torch.Tensor,
    context_out: torch.Tensor,
    context_weights: torch.Tensor,
    sampler: nn.Module,
) -> tuple[torch.Tensor, torch.Tensor, list[list[dict]] | None, torch.Tensor, int]:
    """Select patches from all context images in a single batched call."""
    B, k = context_in.shape[:2]

    ctx_in_flat = context_in.reshape(B * k, *context_in.shape[2:])
    ctx_out_flat = context_out.reshape(B * k, *context_out.shape[2:])
    ctx_w_flat = context_weights.reshape(B * k, *context_weights.shape[2:])

    _, labels, coords, _, aug_params, validity, K_per, _ = sampler(
        ctx_in_flat, ctx_out_flat, ctx_w_flat
    )

    labels = labels.reshape(B, k * K_per, *labels.shape[2:])
    coords = coords.reshape(B, k * K_per, *coords.shape[2:])
    validity = validity.reshape(B, k * K_per, *validity.shape[2:])

    ctx_aug_params: list[list[dict]] | None = None
    if aug_params and any(v is not None for v in aug_params.values()):
        ctx_aug_params = []
        for b in range(B):
            batch_aug = []
            for c in range(k):
                idx = b * k + c
                single = {key: (val[idx:idx+1] if val is not None else None)
                          for key, val in aug_params.items()}
                batch_aug.append(single)
            ctx_aug_params.append(batch_aug)

    return labels, coords, ctx_aug_params, validity, K_per


def _get_patch_sampling_map(
    backbone_out: dict,
    patch_logits: torch.Tensor,
    K: int,
    source: str,
    temperature: float | torch.Tensor,
) -> torch.Tensor | None:
    """Get per-patch sampling map based on configured source."""
    if source == 'none':
        return None
    elif source == 'entropy':
        return compute_entropy_sampling_map(patch_logits, temperature=temperature)
    else:  # 'learned'
        learned_map = backbone_out.get('sampling_map')
        if learned_map is not None:
            return learned_map[:, :K]
        return None


def _create_coverage_mask(
    coords: torch.Tensor,
    patch_size: int,
    output_size: tuple[int, int],
) -> torch.Tensor:
    """Create coverage mask from patch coordinates."""
    B, K, _ = coords.shape
    H, W = output_size
    device = coords.device

    coverage_flat = torch.zeros(B, H * W, device=device)

    ph = torch.arange(patch_size, device=device)
    pw = torch.arange(patch_size, device=device)
    offset_h, offset_w = torch.meshgrid(ph, pw, indexing='ij')
    offsets_h = offset_h.flatten()
    offsets_w = offset_w.flatten()

    h_coords = coords[:, :, 0:1] + offsets_h.view(1, 1, -1)
    w_coords = coords[:, :, 1:2] + offsets_w.view(1, 1, -1)

    h_coords = h_coords.clamp(0, H - 1).long()
    w_coords = w_coords.clamp(0, W - 1).long()
    flat_indices = h_coords * W + w_coords

    batch_idx = torch.arange(B, device=device).view(B, 1, 1).expand_as(flat_indices)
    coverage_flat[batch_idx.flatten(), flat_indices.flatten()] = 1.0

    return coverage_flat.view(B, 1, H, W)
