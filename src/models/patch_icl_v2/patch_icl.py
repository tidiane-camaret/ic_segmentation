"""
PatchICL Architecture v2 with multi-level cascaded support.

Supports coarse-to-fine prediction: predict at level 0, use predictions
to guide patch sampling at finer levels.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.patch_icl_v2.aggregate import PatchAggregator, GaussianAggregator, create_aggregator
from src.models.patch_icl_v2.sampling import ContinuousSampler, SlidingWindowSampler, PatchAugmenter, create_sampler
from src.models.simple_backbone import SimpleBackbone


def extract_patch_features(
    features: torch.Tensor,
    coords: torch.Tensor,
    patch_size: int,
    level_resolution: int,
    feature_grid_size: int = 16,
    target_patch_grid_size: int | None = None,
) -> torch.Tensor:
    """Extract patch features from pre-computed full-image feature maps.

    Extracts at full feature resolution, then downsizes to target_patch_grid_size
    if it differs from the native patch size in feature space.

    Args:
        features: [B, N, D] - Pre-computed features (N = feature_grid_size^2)
        coords: [B, K, 2] - Patch coordinates (h, w) at level resolution
        patch_size: Size of patches at level resolution
        level_resolution: Resolution of the current level
        feature_grid_size: Size of feature grid (e.g., 128 for ICLEncoder layer 0)
        target_patch_grid_size: Desired spatial size per patch for the backbone encoder.
            If None or equal to native size, no resizing is done.

    Returns:
        patch_features: [B, K, tokens_per_patch, D]
    """
    B, K, _ = coords.shape
    device = features.device
    N = features.shape[1]
    D = features.shape[2]

    scale = feature_grid_size / level_resolution
    patch_size_in_features = max(1, int(patch_size * scale))
    tokens_per_patch = patch_size_in_features * patch_size_in_features

    fh = (coords[:, :, 0].float() * scale).long()
    fw = (coords[:, :, 1].float() * scale).long()
    max_start = feature_grid_size - patch_size_in_features
    fh = fh.clamp(0, max_start)
    fw = fw.clamp(0, max_start)

    offset_h = torch.arange(patch_size_in_features, device=device)
    offset_w = torch.arange(patch_size_in_features, device=device)
    offset_grid_h, offset_grid_w = torch.meshgrid(offset_h, offset_w, indexing='ij')
    offset_grid_h = offset_grid_h.reshape(-1)
    offset_grid_w = offset_grid_w.reshape(-1)

    fh_expanded = fh.unsqueeze(-1) + offset_grid_h.view(1, 1, -1)
    fw_expanded = fw.unsqueeze(-1) + offset_grid_w.view(1, 1, -1)
    flat_indices = fh_expanded * feature_grid_size + fw_expanded
    flat_indices = flat_indices.clamp(0, N - 1)

    batch_indices = torch.arange(B, device=device).view(B, 1, 1).expand(B, K, tokens_per_patch)
    patch_features = features[batch_indices, flat_indices]

    # Downsample patches to target grid size if needed
    if (target_patch_grid_size is not None
            and patch_size_in_features != target_patch_grid_size):
        patch_features = patch_features.view(
            B * K, patch_size_in_features, patch_size_in_features, D
        ).permute(0, 3, 1, 2)  # [B*K, D, h, w]
        patch_features = F.adaptive_avg_pool2d(
            patch_features, (target_patch_grid_size, target_patch_grid_size)
        )
        patch_features = patch_features.permute(0, 2, 3, 1).contiguous()
        patch_features = patch_features.view(
            B, K, target_patch_grid_size * target_patch_grid_size, D
        )

    return patch_features


class PatchICL(nn.Module):
    """Multi-level cascaded PatchICL model.

    Processes levels from coarse to fine. Each level's predictions can guide
    patch sampling at the next finer level.
    """

    def __init__(self, config: dict, context_size: int = 0, feature_extractor: nn.Module = None):
        super().__init__()
        self.context_size = context_size
        self.feature_extractor = feature_extractor
        self._feature_extractor_cfg = config.get('feature_extractor', None)

        # Level configs
        levels_cfg = config.get('levels', [{'resolution': 32, 'patch_size': 16, 'num_patches': 16}])
        self.levels = levels_cfg
        self.num_levels = len(levels_cfg)

        # All levels must share the same patch_size (backbone constraint)
        patch_sizes = set(lc['patch_size'] for lc in levels_cfg)
        assert len(patch_sizes) == 1, f"All levels must share the same patch_size, got {patch_sizes}"

        # Backward compat: expose first level's attributes
        self.resolution = levels_cfg[0]['resolution']
        self.patch_size = levels_cfg[0]['patch_size']
        self.num_patches = levels_cfg[0]['num_patches']
        self.num_patches_val = levels_cfg[0].get('num_patches_val', self.num_patches)

        # Oracle config (per level)
        self.oracle_train = list(config.get('oracle_levels_train', [True] * self.num_levels))
        self.oracle_valid = list(config.get('oracle_levels_valid', [False] * self.num_levels))
        while len(self.oracle_train) < self.num_levels:
            self.oracle_train.append(False)
        while len(self.oracle_valid) < self.num_levels:
            self.oracle_valid.append(False)

        # Cascade config
        cascade_cfg = config.get('cascade', {})
        self.detach_between_levels = cascade_cfg.get('detach_between_levels', True)

        # Sampler config (type shared, per-level instances)
        sampler_cfg = config.get('sampler', {})
        self.sampler_type = sampler_cfg.get('type', 'continuous')
        self.sliding_window_stride = sampler_cfg.get('sliding_window_stride', None)

        # Augmenter (shared)
        aug_cfg = sampler_cfg.get('augmentation', {})
        if aug_cfg.get('enabled', False):
            self.augmenter = PatchAugmenter(
                rotation=aug_cfg.get('rotation', 'none'),
                rotation_range=aug_cfg.get('rotation_range', 0.5),
                flip_horizontal=aug_cfg.get('flip_horizontal', False),
                flip_vertical=aug_cfg.get('flip_vertical', False),
            )
        else:
            self.augmenter = None

        # Per-level samplers
        self.samplers = nn.ModuleList()
        for level_cfg in levels_cfg:
            self.samplers.append(create_sampler(
                sampler_type=self.sampler_type,
                patch_size=level_cfg['patch_size'],
                num_patches=level_cfg['num_patches'],
                num_patches_val=level_cfg.get('num_patches_val', level_cfg['num_patches']),
                temperature=level_cfg.get('sampling_temperature', 0.3),
                stride=self.sliding_window_stride,
                augmenter=self.augmenter,
            ))
        self.sampler = self.samplers[0]  # backward compat

        # Per-level aggregators
        self.aggregator_cfg = config.get('aggregator', {})
        self.aggregator_type = self.aggregator_cfg.get('type', 'average')
        self.aggregators = nn.ModuleList()
        for level_cfg in levels_cfg:
            self.aggregators.append(create_aggregator(
                aggregator_type=self.aggregator_type,
                patch_size=level_cfg['patch_size'],
                **self.aggregator_cfg,
            ))
        self.aggregator = self.aggregators[0]  # backward compat

        # Mask channels (1 for binary, 3 for RGB)
        self.num_mask_channels = config.get('num_mask_channels', 1)

        # Loss config
        loss_cfg = config.get('loss', {})
        self.patch_criterion = None
        self.aggreg_criterion = None
        weights_cfg = loss_cfg.get('weights', {}).get('default', {})
        self.loss_weights = {
            'target_patch': weights_cfg.get('target_patch', 1.0),
            'target_aggreg': weights_cfg.get('target_aggreg', 1.0),
            'context_patch': weights_cfg.get('context_patch', 1.0),
            'context_aggreg': weights_cfg.get('context_aggreg', 1.0),
        }
        self.level_loss_weights = list(loss_cfg.get('level_weights', [1.0] * self.num_levels))
        while len(self.level_loss_weights) < self.num_levels:
            self.level_loss_weights.append(1.0)

        # Backbone (shared across levels, with level embedding)
        backbone_cfg = config.get('backbone', {})
        self.feature_grid_size = backbone_cfg.get('feature_grid_size', 16)
        self.patch_feature_grid_size = backbone_cfg.get('patch_feature_grid_size', 16)

        self.backbone = SimpleBackbone(
            embed_dim=backbone_cfg.get('embed_proj_dim', 128),
            num_heads=backbone_cfg.get('num_heads', 8),
            num_layers=backbone_cfg.get('num_layers', 1),
            num_registers=backbone_cfg.get('num_registers', 4),
            num_classes=self.num_mask_channels,
            patch_size=self.patch_size,
            image_size=backbone_cfg.get('image_size', 224),
            input_dim=backbone_cfg.get('embed_dim', 1024),
            target_self_attention=backbone_cfg.get('target_self_attention', False),
            dropout=backbone_cfg.get('dropout', 0.0),
            feature_grid_size=self.patch_feature_grid_size,
            decoder_use_skip_connections=backbone_cfg.get('decoder_use_skip_connections', True),
            append_zero_attn=backbone_cfg.get('append_zero_attn', False),
            max_levels=self.num_levels,
        )

    def set_loss_functions(self, patch_criterion: nn.Module, aggreg_criterion: nn.Module):
        """Set the loss functions for patch and aggreg losses."""
        self.patch_criterion = patch_criterion
        self.aggreg_criterion = aggreg_criterion

    def set_feature_extractor(self, feature_extractor: nn.Module):
        """Set or update the feature extractor."""
        self.feature_extractor = feature_extractor

    def _downsample(self, x: torch.Tensor, resolution: int) -> torch.Tensor:
        """Downsample tensor to given resolution."""
        if x.shape[-1] == resolution and x.shape[-2] == resolution:
            return x
        return F.interpolate(x, size=(resolution, resolution), mode='bilinear', align_corners=False)

    def _downsample_mask(self, mask: torch.Tensor, resolution: int) -> torch.Tensor:
        """Downsample mask using avg pooling (soft area-fraction targets)."""
        if mask.shape[-1] == resolution and mask.shape[-2] == resolution:
            return mask
        if mask.shape[1] > 1:  # Multi-channel (RGB)
            return F.interpolate(mask.float(), size=(resolution, resolution), mode='bilinear', align_corners=False)
        scale_factor = mask.shape[-1] // resolution
        if scale_factor > 1:
            return F.avg_pool2d(mask.float(), kernel_size=scale_factor, stride=scale_factor)
        return F.interpolate(mask.float(), size=(resolution, resolution), mode='area')

    def _mask_to_weights(self, mask: torch.Tensor) -> torch.Tensor:
        """Convert multi-channel mask to single-channel weights for sampling."""
        if mask.shape[1] == 1:
            return mask
        return mask.max(dim=1, keepdim=True)[0]

    def _extract_features(
        self,
        target_images: torch.Tensor,
        context_images: torch.Tensor | None = None,
        context_masks: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Extract features on-the-fly using the feature extractor."""
        if self.feature_extractor is None:
            raise RuntimeError("No feature extractor available.")
        if getattr(self.feature_extractor, '_frozen', False):
            with torch.no_grad():
                return self.feature_extractor.extract_batch(target_images, context_images, context_masks)
        return self.feature_extractor.extract_batch(target_images, context_images, context_masks)

    def _select_context_patches(
        self,
        context_in: torch.Tensor,
        context_out: torch.Tensor,
        context_weights: torch.Tensor,
        sampler: nn.Module,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[list[dict]], torch.Tensor]:
        """Select patches from each context image using the given sampler."""
        B, k = context_in.shape[:2]
        all_ctx_patches, all_ctx_labels, all_ctx_coords, all_aug_params, all_ctx_validity = [], [], [], [], []

        for b in range(B):
            batch_patches, batch_labels, batch_coords, batch_aug_params, batch_validity = [], [], [], [], []
            for ctx_idx in range(k):
                ctx_img = context_in[b, ctx_idx].unsqueeze(0)
                ctx_mask = context_out[b, ctx_idx].unsqueeze(0)
                ctx_weight = context_weights[b, ctx_idx].unsqueeze(0)
                patches, labels, coords, _, aug_params, validity = sampler(ctx_img, ctx_mask, ctx_weight)
                batch_patches.append(patches.squeeze(0))
                batch_labels.append(labels.squeeze(0))
                batch_coords.append(coords.squeeze(0))
                batch_aug_params.append(aug_params)
                batch_validity.append(validity.squeeze(0))
            all_ctx_patches.append(torch.cat(batch_patches, dim=0))
            all_ctx_labels.append(torch.cat(batch_labels, dim=0))
            all_ctx_coords.append(torch.cat(batch_coords, dim=0))
            all_aug_params.append(batch_aug_params)
            all_ctx_validity.append(torch.cat(batch_validity, dim=0))

        return (torch.stack(all_ctx_patches), torch.stack(all_ctx_labels),
                torch.stack(all_ctx_coords), all_aug_params, torch.stack(all_ctx_validity))

    def _forward_level(
        self,
        level_idx: int,
        image: torch.Tensor,
        labels: torch.Tensor | None,
        context_in: torch.Tensor | None,
        context_out: torch.Tensor | None,
        target_features: torch.Tensor | None,
        context_features: torch.Tensor | None,
        weights: torch.Tensor,
        H: int,
        W: int,
        return_attn_weights: bool = False,
    ) -> tuple[dict, torch.Tensor]:
        """Process a single resolution level.

        Returns:
            level_out: Dict with all level outputs (patches, logits, coords, etc.)
            pred: Aggregated prediction logits [B, C, resolution, resolution]
        """
        level_cfg = self.levels[level_idx]
        resolution = level_cfg['resolution']
        patch_size = level_cfg['patch_size']
        sampler = self.samplers[level_idx]
        aggregator = self.aggregators[level_idx]

        B = image.shape[0]
        device = image.device

        # Downsample to level resolution
        image_ds = self._downsample(image, resolution)
        labels_ds = self._downsample_mask(labels, resolution) if labels is not None else torch.zeros(
            B, self.num_mask_channels, resolution, resolution, device=device)

        # Select target patches
        patches, patch_labels, coords, _, aug_params, target_validity = sampler(image_ds, labels_ds, weights, None)
        K = patches.shape[1]
        coord_scale = H / resolution

        # Extract target patch features
        target_patch_features = None
        if target_features is not None:
            target_patch_features = extract_patch_features(
                features=target_features,
                coords=coords,
                patch_size=patch_size,
                level_resolution=resolution,
                feature_grid_size=self.feature_grid_size,
                target_patch_grid_size=self.patch_feature_grid_size,
            )
            if self.augmenter is not None and aug_params:
                target_patch_features = self.augmenter.augment_features_only(target_patch_features, aug_params)

        # Process context
        context_patches, context_patch_labels, context_coords = None, None, None
        context_patch_logits, context_pred, context_out_ds = None, None, None
        context_aug_params = None
        context_validity = None

        if context_in is not None and context_out is not None:
            k = context_in.shape[1]
            context_in_flat = context_in.view(B * k, *context_in.shape[2:])
            context_out_flat = context_out.view(B * k, *context_out.shape[2:])
            context_in_ds = self._downsample(context_in_flat, resolution).view(B, k, -1, resolution, resolution)
            context_out_ds = self._downsample_mask(context_out_flat, resolution).view(B, k, context_out.shape[2], resolution, resolution)
            context_weights = self._mask_to_weights(
                context_out_ds.view(B * k, *context_out_ds.shape[2:])
            ).view(B, k, 1, resolution, resolution)

            context_patches, context_patch_labels, context_coords, context_aug_params, context_validity = (
                self._select_context_patches(context_in_ds, context_out_ds, context_weights, sampler)
            )
            K_ctx = context_patches.shape[1]

            # Extract context patch features
            context_patch_features = None
            if context_features is not None:
                K_per_ctx = K_ctx // k
                ctx_features_list = []
                for ctx_idx in range(k):
                    ctx_feats = context_features[:, ctx_idx]
                    ctx_coords = context_coords[:, ctx_idx * K_per_ctx:(ctx_idx + 1) * K_per_ctx]
                    extracted = extract_patch_features(
                        features=ctx_feats,
                        coords=ctx_coords,
                        patch_size=patch_size,
                        level_resolution=resolution,
                        feature_grid_size=self.feature_grid_size,
                        target_patch_grid_size=self.patch_feature_grid_size,
                    )
                    if self.augmenter is not None:
                        for b in range(B):
                            ctx_aug = context_aug_params[b][ctx_idx]
                            if ctx_aug and any(v is not None for v in ctx_aug.values()):
                                extracted[b:b+1] = self.augmenter.augment_features_only(extracted[b:b+1], ctx_aug)
                    ctx_features_list.append(extracted)
                context_patch_features = torch.cat(ctx_features_list, dim=1)

            # Prepare backbone inputs
            img_patches = torch.cat([target_patch_features, context_patch_features], dim=1)
            all_coords = torch.cat([coords.float() * coord_scale, context_coords.float() * coord_scale], dim=1)
            ctx_id_labels = torch.zeros(B, K + K_ctx, dtype=torch.long, device=device)
            K_per_ctx = K_ctx // k
            for ctx_idx in range(k):
                start = K + ctx_idx * K_per_ctx
                end = K + (ctx_idx + 1) * K_per_ctx
                ctx_id_labels[:, start:end] = ctx_idx + 1

            backbone_out = self.backbone(
                img_patches=img_patches, coords=all_coords, ctx_id_labels=ctx_id_labels,
                return_attn_weights=return_attn_weights, level_idx=level_idx,
            )

            all_logits = backbone_out['mask_patch_logit_preds']
            patch_logits = all_logits[:, :K]
            context_patch_logits = all_logits[:, K:]

            # Aggregate context predictions
            context_preds = []
            for ctx_idx in range(k):
                start_idx = ctx_idx * K_per_ctx
                end_idx = (ctx_idx + 1) * K_per_ctx
                ctx_logits = context_patch_logits[:, start_idx:end_idx]
                ctx_coords_slice = context_coords[:, start_idx:end_idx]
                if self.augmenter is not None:
                    for b in range(B):
                        ctx_aug = context_aug_params[b][ctx_idx]
                        if ctx_aug and any(v is not None for v in ctx_aug.values()):
                            ctx_logits[b:b+1] = self.augmenter.inverse(ctx_logits[b:b+1], ctx_aug)
                ctx_pred = aggregator(ctx_logits, ctx_coords_slice, (resolution, resolution))
                context_preds.append(ctx_pred)
            context_pred = torch.stack(context_preds, dim=1)
        else:
            # No context
            ctx_id_labels = torch.zeros(B, K, dtype=torch.long, device=device)
            backbone_out = self.backbone(
                img_patches=target_patch_features, coords=coords.float() * coord_scale,
                ctx_id_labels=ctx_id_labels, return_attn_weights=return_attn_weights,
                level_idx=level_idx,
            )
            patch_logits = backbone_out['mask_patch_logit_preds']

        # Apply inverse augmentation and aggregate
        patch_logits_for_agg = patch_logits
        if self.augmenter is not None and aug_params:
            patch_logits_for_agg = self.augmenter.inverse(patch_logits, aug_params)

        pred = aggregator(patch_logits_for_agg, coords, (resolution, resolution))

        level_out = {
            'pred': pred,
            'patches': patches,
            'patch_labels': patch_labels,
            'patch_logits': patch_logits,
            'coords': coords,
            'context_patches': context_patches,
            'context_patch_labels': context_patch_labels,
            'context_patch_logits': context_patch_logits,
            'context_coords': context_coords,
            'context_pred': context_pred,
            'context_labels': context_out_ds,
            'context_labels_fullres': context_out,
            'target_validity': target_validity,
            'context_validity': context_validity,
            'attn_weights': backbone_out.get('attn_weights'),
            'register_tokens': backbone_out.get('register_tokens'),
            'patch_size': patch_size,
            'level_res': resolution,
        }

        return level_out, pred

    def forward(
        self,
        image: torch.Tensor,
        labels: torch.Tensor = None,
        context_in: torch.Tensor = None,
        context_out: torch.Tensor = None,
        target_features: torch.Tensor = None,
        context_features: torch.Tensor = None,
        mode: str = "train",
        return_attn_weights: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Multi-level cascaded forward pass."""
        training = (mode == "train")
        B, _, H, W = image.shape
        device = image.device

        if labels is not None and labels.dim() == 3:
            labels = labels.unsqueeze(1)

        # On-the-fly feature extraction if needed
        if target_features is None and self.feature_extractor is not None:
            target_features, context_features = self._extract_features(image, context_in, context_out)

        # Process each level
        prev_pred = None
        level_outputs = []

        for i in range(self.num_levels):
            resolution = self.levels[i]['resolution']

            # Determine sampling weights for this level
            use_oracle = self.oracle_train[i] if training else self.oracle_valid[i]
            if use_oracle and labels is not None:
                labels_ds = self._downsample_mask(labels, resolution)
                weights = self._mask_to_weights(labels_ds)
            elif prev_pred is not None:
                # Use previous level's prediction to guide sampling
                prev_probs = torch.sigmoid(prev_pred)
                if self.detach_between_levels:
                    prev_probs = prev_probs.detach()
                weights = F.interpolate(prev_probs, size=(resolution, resolution), mode='bilinear', align_corners=False)
            else:
                weights = torch.ones(B, 1, resolution, resolution, device=device)

            level_out, pred = self._forward_level(
                level_idx=i,
                image=image,
                labels=labels,
                context_in=context_in,
                context_out=context_out,
                target_features=target_features,
                context_features=context_features,
                weights=weights,
                H=H, W=W,
                return_attn_weights=return_attn_weights,
            )
            level_outputs.append(level_out)
            prev_pred = pred

        # Final prediction from last level
        final_pred = F.interpolate(prev_pred, size=(H, W), mode='bilinear', align_corners=False)
        coarse_pred = F.interpolate(level_outputs[0]['pred'], size=(H, W), mode='bilinear', align_corners=False)

        # Backward compat aliases from last level
        last = level_outputs[-1]
        return {
            'final_pred': final_pred,
            'final_logit': final_pred,
            'coarse_pred': coarse_pred,
            'level_outputs': level_outputs,
            'patches': last['patches'],
            'patch_labels': last['patch_labels'],
            'patch_logits': last['patch_logits'],
            'patch_coords': last['coords'],
            'context_patches': last['context_patches'],
            'context_patch_labels': last['context_patch_labels'],
            'context_patch_logits': last['context_patch_logits'],
            'context_coords': last['context_coords'],
            'attn_weights': last.get('attn_weights'),
            'register_tokens': last.get('register_tokens'),
        }

    def _masked_patch_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        validity: torch.Tensor | None,
    ) -> torch.Tensor:
        """Compute patch loss, masking out invalid (out-of-image) pixels."""
        if validity is None:
            return self.patch_criterion(logits, labels)
        # For invalid pixels: set logit to -100 (sigmoid~0) and label to 0,
        # so the loss contribution is ~0 for any standard criterion (BCE, dice).
        invalid = ~validity.expand_as(logits).bool()
        masked_logits = torch.where(invalid, torch.tensor(-100.0, device=logits.device), logits)
        masked_labels = torch.where(invalid, torch.zeros_like(labels), labels)
        return self.patch_criterion(masked_logits, masked_labels)

    def compute_loss(
        self,
        outputs: dict[str, torch.Tensor],
        labels: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute multi-level losses with configurable per-level weights."""
        if self.patch_criterion is None or self.aggreg_criterion is None:
            raise RuntimeError("Loss functions not set. Call set_loss_functions() first.")

        if labels.dim() == 3:
            labels = labels.unsqueeze(1)

        B = labels.shape[0]
        device = outputs['patch_logits'].device
        losses = {}
        total_loss = 0.0

        # Accumulators for logging (averaged across levels)
        sum_target_patch = 0.0
        sum_target_aggreg = 0.0
        sum_context_patch = 0.0
        sum_context_aggreg = 0.0

        for i, level_out in enumerate(outputs['level_outputs']):
            lw = self.level_loss_weights[i]

            # Target patch loss
            patch_logits = level_out['patch_logits']
            patch_labels = level_out['patch_labels']
            target_validity = level_out.get('target_validity')
            K = patch_logits.shape[1]
            target_patch_loss = self._masked_patch_loss(
                patch_logits.reshape(B * K, -1),
                patch_labels.reshape(B * K, -1),
                target_validity.reshape(B * K, -1) if target_validity is not None else None,
            )
            losses[f'level_{i}_target_patch_loss'] = target_patch_loss
            losses[f'level_{i}_patch_loss'] = target_patch_loss
            total_loss = total_loss + lw * self.loss_weights['target_patch'] * target_patch_loss
            sum_target_patch = sum_target_patch + target_patch_loss

            # Target aggreg loss (soft avg_pool targets)
            pred = level_out['pred']
            scale_factor = labels.shape[-1] // pred.shape[-1]
            labels_ds = F.avg_pool2d(labels.float(), kernel_size=scale_factor, stride=scale_factor)
            target_aggreg_loss = self.aggreg_criterion(pred, labels_ds)
            losses[f'level_{i}_target_aggreg_loss'] = target_aggreg_loss
            losses[f'level_{i}_pred_loss'] = target_aggreg_loss
            total_loss = total_loss + lw * self.loss_weights['target_aggreg'] * target_aggreg_loss
            sum_target_aggreg = sum_target_aggreg + target_aggreg_loss

            # Context patch loss
            ctx_patch_logits = level_out.get('context_patch_logits')
            ctx_patch_labels = level_out.get('context_patch_labels')
            ctx_validity = level_out.get('context_validity')
            if ctx_patch_logits is not None and ctx_patch_labels is not None:
                K_ctx = ctx_patch_logits.shape[1]
                context_patch_loss = self._masked_patch_loss(
                    ctx_patch_logits.reshape(B * K_ctx, -1),
                    ctx_patch_labels.reshape(B * K_ctx, -1),
                    ctx_validity.reshape(B * K_ctx, -1) if ctx_validity is not None else None,
                )
                losses[f'level_{i}_context_patch_loss'] = context_patch_loss
                total_loss = total_loss + lw * self.loss_weights['context_patch'] * context_patch_loss
                sum_context_patch = sum_context_patch + context_patch_loss
            else:
                losses[f'level_{i}_context_patch_loss'] = torch.tensor(0.0, device=device)

            # Context aggreg loss
            context_pred = level_out.get('context_pred')
            context_labels_fullres = level_out.get('context_labels_fullres')
            if context_pred is not None and context_labels_fullres is not None:
                B_ctx, k_ctx = context_pred.shape[:2]
                ctx_flat = context_labels_fullres.view(B_ctx * k_ctx, *context_labels_fullres.shape[2:])
                ctx_scale = ctx_flat.shape[-1] // context_pred.shape[-1]
                context_labels_ds = F.avg_pool2d(ctx_flat.float(), kernel_size=ctx_scale, stride=ctx_scale)
                context_aggreg_loss = self.aggreg_criterion(
                    context_pred.reshape(B_ctx * k_ctx, -1),
                    context_labels_ds.reshape(B_ctx * k_ctx, -1),
                )
                losses[f'level_{i}_context_aggreg_loss'] = context_aggreg_loss
                total_loss = total_loss + lw * self.loss_weights['context_aggreg'] * context_aggreg_loss
                sum_context_aggreg = sum_context_aggreg + context_aggreg_loss
            else:
                losses[f'level_{i}_context_aggreg_loss'] = torch.tensor(0.0, device=device)

        # Final prediction loss (from last level, upsampled to full res)
        final_loss = self.aggreg_criterion(outputs['final_pred'], labels)
        losses['final_loss'] = final_loss
        total_loss = total_loss + final_loss
        losses['total_loss'] = total_loss

        # Aggregated losses for logging (mean across levels)
        n = self.num_levels
        losses['target_patch_loss'] = sum_target_patch / n
        losses['target_aggreg_loss'] = sum_target_aggreg / n
        losses['context_patch_loss'] = sum_context_patch / n
        losses['context_aggreg_loss'] = sum_context_aggreg / n
        losses['target_loss'] = losses['target_patch_loss'] + losses['target_aggreg_loss']
        losses['context_loss'] = losses['context_patch_loss'] + losses['context_aggreg_loss']
        losses['patch_loss_total'] = losses['target_patch_loss'] + losses['context_patch_loss']
        losses['aggreg_loss_total'] = losses['target_aggreg_loss'] + losses['context_aggreg_loss']

        # Legacy compatibility
        losses['aggreg_loss'] = losses['target_aggreg_loss']
        losses['local_loss'] = losses['target_patch_loss']
        losses['agg_loss'] = final_loss

        return losses
