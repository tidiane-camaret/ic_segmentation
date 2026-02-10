"""
Simplified PatchICL Architecture v2.

Single-level in-context learning for segmentation.
Removes: multi-level support, refinement passes, gumbel sampling, feature losses.
Keeps: continuous and sliding_window samplers, gaussian and average aggregators.
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
) -> torch.Tensor:
    """
    Extract patch features from pre-computed full-image feature maps.

    Args:
        features: [B, N, D] - Pre-computed features (N = feature_grid_size^2)
        coords: [B, K, 2] - Patch coordinates (h, w) at level resolution
        patch_size: Size of patches at level resolution
        level_resolution: Resolution of the current level
        feature_grid_size: Size of feature grid (e.g., 14 for ViT)

    Returns:
        patch_features: [B, K, tokens_per_patch, D]
    """
    B, K, _ = coords.shape
    device = features.device
    N = features.shape[1]

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
    return patch_features


class PatchICL(nn.Module):
    """
    Simplified single-level PatchICL model.

    Supports both precomputed features and on-the-fly feature extraction.
    """

    def __init__(self, config: dict, context_size: int = 0, feature_extractor: nn.Module = None):
        super().__init__()
        self.context_size = context_size
        self.feature_extractor = feature_extractor
        self._feature_extractor_cfg = config.get('feature_extractor', None)

        # Level config (single level only)
        levels_cfg = config.get('levels', [{'resolution': 32, 'patch_size': 16, 'num_patches': 16}])
        level_cfg = levels_cfg[0]  # Use first level only
        self.resolution = level_cfg['resolution']
        self.patch_size = level_cfg['patch_size']
        self.num_patches = level_cfg['num_patches']
        self.num_patches_val = level_cfg.get('num_patches_val', self.num_patches)

        # Oracle config
        self.oracle_train = config.get('oracle_levels_train', [True])[0]
        self.oracle_valid = config.get('oracle_levels_valid', [False])[0]

        # Sampler config
        sampler_cfg = config.get('sampler', {})
        self.sampler_type = sampler_cfg.get('type', 'continuous')
        self.sliding_window_stride = sampler_cfg.get('sliding_window_stride', None)

        # Augmenter
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

        # Create sampler
        self.sampler = create_sampler(
            sampler_type=self.sampler_type,
            patch_size=self.patch_size,
            num_patches=self.num_patches,
            num_patches_val=self.num_patches_val,
            temperature=level_cfg.get('sampling_temperature', 0.3),
            stride=self.sliding_window_stride,
            augmenter=self.augmenter,
        )

        # Aggregator config
        self.aggregator_cfg = config.get('aggregator', {})
        self.aggregator_type = self.aggregator_cfg.get('type', 'average')
        self.aggregator = create_aggregator(
            aggregator_type=self.aggregator_type,
            patch_size=self.patch_size,
            **self.aggregator_cfg,
        )

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

        # Backbone config
        backbone_cfg = config.get('backbone', {})
        self.feature_grid_size = backbone_cfg.get('feature_grid_size', 16)
        patch_feature_grid_size = backbone_cfg.get('patch_feature_grid_size', 16)

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
            feature_grid_size=patch_feature_grid_size,
            decoder_use_skip_connections=backbone_cfg.get('decoder_use_skip_connections', True),
            append_zero_attn=backbone_cfg.get('append_zero_attn', False),
        )

    def set_loss_functions(self, patch_criterion: nn.Module, aggreg_criterion: nn.Module):
        """Set the loss functions for patch and aggreg losses."""
        self.patch_criterion = patch_criterion
        self.aggreg_criterion = aggreg_criterion

    def set_feature_extractor(self, feature_extractor: nn.Module):
        """Set or update the feature extractor."""
        self.feature_extractor = feature_extractor

    def _downsample(self, x: torch.Tensor) -> torch.Tensor:
        """Downsample tensor to level resolution."""
        if x.shape[-1] == self.resolution and x.shape[-2] == self.resolution:
            return x
        return F.interpolate(x, size=(self.resolution, self.resolution), mode='bilinear', align_corners=False)

    def _downsample_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """Downsample mask to level resolution using avg pooling (soft area-fraction targets)."""
        if mask.shape[-1] == self.resolution and mask.shape[-2] == self.resolution:
            return mask
        if mask.shape[1] > 1:  # Multi-channel (RGB)
            return F.interpolate(mask.float(), size=(self.resolution, self.resolution), mode='bilinear', align_corners=False)
        scale_factor = mask.shape[-1] // self.resolution
        if scale_factor > 1:
            return F.avg_pool2d(mask.float(), kernel_size=scale_factor, stride=scale_factor)
        return F.interpolate(mask.float(), size=(self.resolution, self.resolution), mode='area')

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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[list[dict]], torch.Tensor]:
        """Select patches from each context image."""
        B, k = context_in.shape[:2]
        all_ctx_patches, all_ctx_labels, all_ctx_coords, all_aug_params, all_ctx_validity = [], [], [], [], []

        for b in range(B):
            batch_patches, batch_labels, batch_coords, batch_aug_params, batch_validity = [], [], [], [], []
            for ctx_idx in range(k):
                ctx_img = context_in[b, ctx_idx].unsqueeze(0)
                ctx_mask = context_out[b, ctx_idx].unsqueeze(0)
                ctx_weight = context_weights[b, ctx_idx].unsqueeze(0)
                patches, labels, coords, _, aug_params, validity = self.sampler(ctx_img, ctx_mask, ctx_weight)
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
        """Forward pass."""
        training = (mode == "train")
        B, _, H, W = image.shape
        device = image.device

        if labels is not None and labels.dim() == 3:
            labels = labels.unsqueeze(1)

        # On-the-fly feature extraction if needed
        if target_features is None and self.feature_extractor is not None:
            target_features, context_features = self._extract_features(image, context_in, context_out)

        # Downsample to level resolution
        image_ds = self._downsample(image)
        labels_ds = self._downsample_mask(labels) if labels is not None else torch.zeros(B, self.num_mask_channels, self.resolution, self.resolution, device=device)

        # Create sampling weights
        use_oracle = self.oracle_train if training else self.oracle_valid
        if use_oracle and labels is not None:
            weights = self._mask_to_weights(labels_ds)
        else:
            weights = torch.ones(B, 1, self.resolution, self.resolution, device=device)

        # Select target patches
        patches, patch_labels, coords, _, aug_params, target_validity = self.sampler(image_ds, labels_ds, weights, None)
        K = patches.shape[1]
        coord_scale = H / self.resolution

        # Extract target patch features
        target_patch_features = None
        if target_features is not None:
            target_patch_features = extract_patch_features(
                features=target_features,
                coords=coords,
                patch_size=self.patch_size,
                level_resolution=self.resolution,
                feature_grid_size=self.feature_grid_size,
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
            context_in_ds = self._downsample(context_in_flat).view(B, k, -1, self.resolution, self.resolution)
            context_out_ds = self._downsample_mask(context_out_flat).view(B, k, context_out.shape[2], self.resolution, self.resolution)
            context_weights = self._mask_to_weights(context_out_ds.view(B * k, *context_out_ds.shape[2:])).view(B, k, 1, self.resolution, self.resolution)

            context_patches, context_patch_labels, context_coords, context_aug_params, context_validity = self._select_context_patches(
                context_in_ds, context_out_ds, context_weights
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
                        patch_size=self.patch_size,
                        level_resolution=self.resolution,
                        feature_grid_size=self.feature_grid_size,
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
                return_attn_weights=return_attn_weights
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
                ctx_pred = self.aggregator(ctx_logits, ctx_coords_slice, (self.resolution, self.resolution))
                context_preds.append(ctx_pred)
            context_pred = torch.stack(context_preds, dim=1)
        else:
            # No context
            ctx_id_labels = torch.zeros(B, K, dtype=torch.long, device=device)
            backbone_out = self.backbone(
                img_patches=target_patch_features, coords=coords.float() * coord_scale,
                ctx_id_labels=ctx_id_labels, return_attn_weights=return_attn_weights
            )
            patch_logits = backbone_out['mask_patch_logit_preds']

        # Apply inverse augmentation and aggregate
        patch_logits_for_agg = patch_logits
        if self.augmenter is not None and aug_params:
            patch_logits_for_agg = self.augmenter.inverse(patch_logits, aug_params)

        pred = self.aggregator(patch_logits_for_agg, coords, (self.resolution, self.resolution))
        final_pred = F.interpolate(pred, size=(H, W), mode='bilinear', align_corners=False)

        return {
            'final_pred': final_pred,
            'final_logit': final_pred,
            'coarse_pred': final_pred,
            'level_outputs': [{
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
                'context_labels': context_out_ds,  # max_pool2d downsampled (used for sampling)
                'context_labels_fullres': context_out,  # original full-res masks for loss
                'target_validity': target_validity,
                'context_validity': context_validity,
                'attn_weights': backbone_out.get('attn_weights'),
                'register_tokens': backbone_out.get('register_tokens'),
                'patch_size': self.patch_size,
                'level_res': self.resolution,
            }],
            'patches': patches,
            'patch_labels': patch_labels,
            'patch_logits': patch_logits,
            'patch_coords': coords,
            'context_patches': context_patches,
            'context_patch_labels': context_patch_labels,
            'context_patch_logits': context_patch_logits,
            'context_coords': context_coords,
            'attn_weights': backbone_out.get('attn_weights'),
            'register_tokens': backbone_out.get('register_tokens'),
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
        # For invalid pixels: set logit to -100 (sigmoid≈0) and label to 0,
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
        """Compute losses with configurable weights."""
        if self.patch_criterion is None or self.aggreg_criterion is None:
            raise RuntimeError("Loss functions not set. Call set_loss_functions() first.")

        if labels.dim() == 3:
            labels = labels.unsqueeze(1)

        B = labels.shape[0]
        device = outputs['patch_logits'].device
        losses = {}
        total_loss = 0.0

        level_out = outputs['level_outputs'][0]

        # Target patch loss (masked by validity for border patches)
        patch_logits = level_out['patch_logits']
        patch_labels = level_out['patch_labels']
        target_validity = level_out.get('target_validity')
        K = patch_logits.shape[1]
        target_patch_loss = self._masked_patch_loss(
            patch_logits.reshape(B * K, -1),
            patch_labels.reshape(B * K, -1),
            target_validity.reshape(B * K, -1) if target_validity is not None else None,
        )
        losses['level_0_target_patch_loss'] = target_patch_loss
        losses['level_0_patch_loss'] = target_patch_loss
        total_loss = total_loss + self.loss_weights['target_patch'] * target_patch_loss

        # Target aggreg loss (soft avg_pool targets — preserves area fraction)
        pred = level_out['pred']
        scale_factor = labels.shape[-1] // pred.shape[-1]
        labels_ds = F.avg_pool2d(labels.float(), kernel_size=scale_factor, stride=scale_factor)
        target_aggreg_loss = self.aggreg_criterion(pred, labels_ds)
        losses['level_0_target_aggreg_loss'] = target_aggreg_loss
        losses['level_0_pred_loss'] = target_aggreg_loss
        total_loss = total_loss + self.loss_weights['target_aggreg'] * target_aggreg_loss

        # Context losses
        context_patch_logits = level_out.get('context_patch_logits')
        context_patch_labels = level_out.get('context_patch_labels')
        context_validity = level_out.get('context_validity')
        if context_patch_logits is not None and context_patch_labels is not None:
            K_ctx = context_patch_logits.shape[1]
            context_patch_loss = self._masked_patch_loss(
                context_patch_logits.reshape(B * K_ctx, -1),
                context_patch_labels.reshape(B * K_ctx, -1),
                context_validity.reshape(B * K_ctx, -1) if context_validity is not None else None,
            )
            losses['level_0_context_patch_loss'] = context_patch_loss
            total_loss = total_loss + self.loss_weights['context_patch'] * context_patch_loss
        else:
            losses['level_0_context_patch_loss'] = torch.tensor(0.0, device=device)

        context_pred = level_out.get('context_pred')
        context_labels_fullres = level_out.get('context_labels_fullres')
        if context_pred is not None and context_labels_fullres is not None:
            B_ctx, k_ctx = context_pred.shape[:2]
            # Soft avg_pool targets for context (same as target aggreg loss)
            ctx_flat = context_labels_fullres.view(B_ctx * k_ctx, *context_labels_fullres.shape[2:])
            ctx_scale = ctx_flat.shape[-1] // context_pred.shape[-1]
            context_labels_ds = F.avg_pool2d(ctx_flat.float(), kernel_size=ctx_scale, stride=ctx_scale)
            context_aggreg_loss = self.aggreg_criterion(
                context_pred.reshape(B_ctx * k_ctx, -1),
                context_labels_ds.reshape(B_ctx * k_ctx, -1),
            )
            losses['level_0_context_aggreg_loss'] = context_aggreg_loss
            total_loss = total_loss + self.loss_weights['context_aggreg'] * context_aggreg_loss
        else:
            losses['level_0_context_aggreg_loss'] = torch.tensor(0.0, device=device)

        # Final prediction loss
        final_loss = self.aggreg_criterion(outputs['final_pred'], labels)
        losses['final_loss'] = final_loss
        total_loss = total_loss + final_loss
        losses['total_loss'] = total_loss

        # Aggregated losses for logging
        losses['target_patch_loss'] = target_patch_loss
        losses['target_aggreg_loss'] = target_aggreg_loss
        losses['context_patch_loss'] = losses['level_0_context_patch_loss']
        losses['context_aggreg_loss'] = losses['level_0_context_aggreg_loss']
        losses['target_loss'] = target_patch_loss + target_aggreg_loss
        losses['context_loss'] = losses['context_patch_loss'] + losses['context_aggreg_loss']
        losses['patch_loss_total'] = target_patch_loss + losses['context_patch_loss']
        losses['aggreg_loss_total'] = target_aggreg_loss + losses['context_aggreg_loss']

        # Legacy compatibility
        losses['aggreg_loss'] = target_aggreg_loss
        losses['local_loss'] = target_patch_loss
        losses['agg_loss'] = final_loss

        return losses
