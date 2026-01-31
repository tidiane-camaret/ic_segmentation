"""
Patch-ICL Architecture: Multi-resolution in-context learning for segmentation.

Generalizes global_local to arbitrary number of resolution levels.
Each level processes patches at progressively finer resolution.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.aggregate import PatchAggregator, create_aggregator
from src.models.backbone import (
    CNNCrossPatchBackbone,
    CrossPatchAttentionBackbone,
    MultiLayerCrossPatchBackbone,
    build_modular_backbone,
)
from src.models.sampling import (
    ContinuousSampler,
    DeterministicTopKSampler,
    GumbelSoftmaxSampler,
    PatchAugmenter,
    PatchSampler,
    SlidingWindowSampler,
    UniformSampler,
)
from src.models.simple_backbone import SimpleBackbone


def extract_patch_features(
    features: torch.Tensor,
    coords: torch.Tensor,
    patch_size: int,
    level_resolution: int,
    feature_grid_size: int = 16,
) -> torch.Tensor:
    """
    Extract patch features from pre-computed full-image feature maps (vectorized).

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
    batch_indices = torch.arange(B, device=device).view(B, 1, 1).expand(B, K, tokens_per_patch)

    patch_features = features[batch_indices, flat_indices]
    return patch_features


class PatchICL_Level(nn.Module):
    """
    Single resolution level for PatchICL.

    Each level:
    1. Downsamples inputs to its resolution
    2. Samples K patches weighted by previous prediction
    3. Processes patches through backbone
    4. Aggregates predictions back to a mask
    """

    def __init__(
        self,
        resolution: int,
        patch_size: int,
        num_patches: int,
        backbone: nn.Module,
        level_idx: int = 0,
        sampling_temperature: float = 0.3,
        sampler: PatchSampler | None = None,
        aggregator: PatchAggregator | None = None,
        target_mask_ratio: float = 0.0,
        context_mask_ratio: float = 0.0,
        num_mask_channels: int = 1,
    ):
        super().__init__()
        self.resolution = resolution
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.backbone = backbone
        self.level_idx = level_idx
        self.sampling_temperature = sampling_temperature
        self.target_mask_ratio = target_mask_ratio
        self.context_mask_ratio = context_mask_ratio
        self.num_mask_channels = num_mask_channels

        if sampler is not None:
            self.sampler = sampler
        else:
            self.sampler = PatchSampler(
                patch_size=patch_size,
                num_patches=num_patches,
                temperature=sampling_temperature,
            )

        if aggregator is not None:
            self.aggregator = aggregator
        else:
            self.aggregator = PatchAggregator(patch_size=patch_size)

    def downsample(self, x: torch.Tensor) -> torch.Tensor:
        """Downsample tensor to this level's resolution."""
        if x.shape[-1] == self.resolution and x.shape[-2] == self.resolution:
            return x
        return F.interpolate(x, size=(self.resolution, self.resolution), mode='bilinear', align_corners=False)

    def downsample_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """Downsample mask to this level's resolution.
        
        For single-channel binary masks, uses nearest neighbor to preserve labels.
        For multi-channel (e.g., RGB) masks, uses bilinear to preserve color values.
        """
        if mask.shape[-1] == self.resolution and mask.shape[-2] == self.resolution:
            return mask
        # Use bilinear for multi-channel masks (RGB colors), nearest for binary
        mode = 'bilinear' if mask.shape[1] > 1 else 'nearest'
        if mode == 'bilinear':
            return F.interpolate(mask.float(), size=(self.resolution, self.resolution), mode='bilinear', align_corners=False)
        return F.interpolate(mask.float(), size=(self.resolution, self.resolution), mode='nearest')

    def _mask_to_weights(self, mask: torch.Tensor) -> torch.Tensor:
        """Convert mask to single-channel weight map for sampling.
        
        For single-channel masks, returns as-is.
        For multi-channel masks (e.g., RGB), returns max across channels.
        This ensures any colored region gets sampled.
        
        Args:
            mask: [B, C, H, W] - mask tensor (C=1 for binary, C=3 for RGB)
            
        Returns:
            weights: [B, 1, H, W] - single-channel weight map
        """
        if mask.shape[1] == 1:
            return mask
        # For multi-channel masks, take max across channels
        # This way any non-zero color channel contributes to sampling
        return mask.max(dim=1, keepdim=True)[0]

    def select_patches(
        self,
        image: torch.Tensor,
        labels: torch.Tensor,
        weights: torch.Tensor,
        patch_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, dict]:
        """Select K patches based on weight map, optionally augmenting features too."""
        return self.sampler(image, labels, weights, patch_features)

    def aggregate(
        self,
        patch_logits: torch.Tensor,
        coords: torch.Tensor,
        output_size: tuple[int, int],
        prev_pred: torch.Tensor = None,
    ) -> torch.Tensor:
        """Aggregate patch predictions back to a full mask."""
        return self.aggregator(patch_logits, coords, output_size, prev_pred)

    def _select_context_patches(
        self,
        context_in: torch.Tensor,
        context_out: torch.Tensor,
        context_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[list[dict]]]:
        """Select K patches from each context image.

        Returns:
            context_patches: [B, K*k, C, ps, ps]
            context_labels: [B, K*k, C_mask, ps, ps]
            context_coords: [B, K*k, 2]
            all_aug_params: list[list[dict]] - aug_params for each (batch, context) pair
        """
        B, k = context_in.shape[:2]

        all_ctx_patches = []
        all_ctx_labels = []
        all_ctx_coords = []
        all_aug_params = []  # [B][k] list of aug_params dicts

        for b in range(B):
            batch_patches = []
            batch_labels = []
            batch_coords = []
            batch_aug_params = []

            for ctx_idx in range(k):
                ctx_img = context_in[b, ctx_idx].unsqueeze(0)
                ctx_mask = context_out[b, ctx_idx].unsqueeze(0)
                ctx_weight = context_weights[b, ctx_idx].unsqueeze(0)

                patches, labels, coords, _, aug_params = self.select_patches(ctx_img, ctx_mask, ctx_weight)

                batch_patches.append(patches.squeeze(0))
                batch_labels.append(labels.squeeze(0))
                batch_coords.append(coords.squeeze(0))
                batch_aug_params.append(aug_params)

            all_ctx_patches.append(torch.cat(batch_patches, dim=0))
            all_ctx_labels.append(torch.cat(batch_labels, dim=0))
            all_ctx_coords.append(torch.cat(batch_coords, dim=0))
            all_aug_params.append(batch_aug_params)

        return (
            torch.stack(all_ctx_patches),
            torch.stack(all_ctx_labels),
            torch.stack(all_ctx_coords),
            all_aug_params,
        )

    def _prepare_backbone_inputs(
        self,
        target_features: torch.Tensor,
        target_coords: torch.Tensor,
        context_features: torch.Tensor = None,
        context_coords: torch.Tensor = None,
        num_context_images: int = 0,
    ) -> dict[str, torch.Tensor]:
        """
        Prepare inputs for CrossPatchAttentionBackbone.

        Args:
            target_features: [B, K, tokens, D] - target patch features
            target_coords: [B, K, 2] - target coordinates
            context_features: [B, K_ctx, tokens, D] - context patch features
            context_coords: [B, K_ctx, 2] - context coordinates
            num_context_images: k - number of context images

        Returns:
            Dict with img_patches, coords, ctx_id_labels
        """
        B = target_features.shape[0]
        K = target_features.shape[1]
        device = target_features.device

        if context_features is not None:
            K_ctx = context_features.shape[1]

            # Concatenate target and context features
            img_patches = torch.cat([target_features, context_features], dim=1)
            coords = torch.cat([target_coords, context_coords], dim=1)

            # Build ctx_id_labels: 0 for target, 1..k for each context image
            ctx_id_labels = torch.zeros(B, K + K_ctx, dtype=torch.long, device=device)
            K_per_ctx = K_ctx // num_context_images
            for ctx_idx in range(num_context_images):
                start = K + ctx_idx * K_per_ctx
                end = K + (ctx_idx + 1) * K_per_ctx
                ctx_id_labels[:, start:end] = ctx_idx + 1
        else:
            img_patches = target_features
            coords = target_coords
            ctx_id_labels = torch.zeros(B, K, dtype=torch.long, device=device)

        return {
            'img_patches': img_patches,
            'coords': coords,
            'ctx_id_labels': ctx_id_labels,
        }

    def forward(
        self,
        image: torch.Tensor,
        labels: torch.Tensor = None,
        prev_pred: torch.Tensor = None,
        use_oracle: bool = False,
        original_coords_scale: float = 1.0,
        context_in: torch.Tensor = None,
        context_out: torch.Tensor = None,
        target_features: torch.Tensor = None,
        context_features: torch.Tensor = None,
        training: bool = False,
        return_attn_weights: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass for this level.

        Args:
            image: [B, C, H, W] - original resolution image
            labels: [B, C_mask, H, W] - GT mask (C_mask=1 for binary, C_mask=3 for RGB)
            prev_pred: [B, C_mask, H, W] - prediction from previous level
            use_oracle: If True, use GT mask for sampling
            original_coords_scale: Scale factor for coordinates
            context_in: [B, k, C, H, W] - context images
            context_out: [B, k, C_mask, H, W] - context masks
            return_attn_weights: If True, return attention weights and register tokens
            target_features: [B, N, D] - Pre-computed features for target
            context_features: [B, k, N, D] - Pre-computed features for context
            training: Whether in training mode

        Returns:
            Dict with pred, patches, patch_labels, patch_logits, coords, context_*
        """
        B = image.shape[0]
        device = image.device

        if labels is not None and labels.dim() == 3:
            labels = labels.unsqueeze(1)

        # Downsample to this level's resolution
        image_ds = self.downsample(image)
        labels_ds = self.downsample_mask(labels) if labels is not None else None

        # Create weight map for sampling (always single-channel)
        if use_oracle and labels is not None:
            # Convert multi-channel mask to single-channel weights
            weights = self._mask_to_weights(labels_ds)
        elif prev_pred is not None:
            weights = self._mask_to_weights(self.downsample(prev_pred))
        else:
            weights = torch.ones(B, 1, self.resolution, self.resolution, device=device)

        # Initialize empty labels with correct number of channels
        if labels_ds is None:
            labels_ds = torch.zeros(B, self.num_mask_channels, self.resolution, self.resolution, device=device)

        # Select target patches (without features for now - we'll extract and augment features after)
        patches, patch_labels, coords, _, aug_params = self.select_patches(
            image_ds, labels_ds, weights, None
        )
        K = patches.shape[1]

        # Scale coordinates for backbone
        coords_scaled = coords.float() * original_coords_scale

        # Extract target patch features from pre-computed features at sampled coords
        target_patch_features = None
        if target_features is not None:
            target_patch_features = extract_patch_features(
                features=target_features,
                coords=coords,
                patch_size=self.patch_size,
                level_resolution=self.resolution,
            )
            # Apply the SAME augmentation to features that was applied to patches
            if self.sampler.augmenter is not None and aug_params:
                target_patch_features = self.sampler.augmenter.augment_features_only(
                    target_patch_features, aug_params
                )

        # Process context if provided
        context_patches = None
        context_patch_labels = None
        context_coords = None
        context_patch_logits = None
        context_pred = None
        context_out_ds = None

        if context_in is not None and context_out is not None:
            k = context_in.shape[1]

            # Downsample context
            context_in_flat = context_in.view(B * k, *context_in.shape[2:])
            context_out_flat = context_out.view(B * k, *context_out.shape[2:])

            context_in_ds = self.downsample(context_in_flat).view(B, k, -1, self.resolution, self.resolution)
            context_out_ds = self.downsample_mask(context_out_flat).view(B, k, context_out.shape[2], self.resolution, self.resolution)

            # Create single-channel weights from context masks for patch selection
            context_weights = self._mask_to_weights(context_out_ds.view(B * k, *context_out_ds.shape[2:])).view(B, k, 1, self.resolution, self.resolution)

            # Select patches from context (use single-channel weights for sampling)
            context_patches, context_patch_labels, context_coords, context_aug_params = self._select_context_patches(
                context_in_ds, context_out_ds, context_weights
            )
            K_ctx = context_patches.shape[1]

            # Scale context coordinates
            context_coords_scaled = context_coords.float() * original_coords_scale

            # Extract context patch features and apply the same augmentation
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
                    )
                    # Apply the SAME augmentation to context features that was applied to context patches
                    if self.sampler.augmenter is not None:
                        # context_aug_params is [B][k] list, we need to handle per-batch, per-context
                        # For simplicity, apply augmentation per-batch
                        for b in range(B):
                            ctx_aug = context_aug_params[b][ctx_idx]
                            if ctx_aug:
                                # Create single-sample aug_params tensors
                                single_aug = {}
                                for key, val in ctx_aug.items():
                                    if val is not None:
                                        single_aug[key] = val  # Already [1, K_per_ctx] shape
                                    else:
                                        single_aug[key] = None
                                if any(v is not None for v in single_aug.values()):
                                    extracted[b:b+1] = self.sampler.augmenter.augment_features_only(
                                        extracted[b:b+1], single_aug
                                    )
                    ctx_features_list.append(extracted)
                context_patch_features = torch.cat(ctx_features_list, dim=1)

            # Prepare backbone inputs
            backbone_inputs = self._prepare_backbone_inputs(
                target_features=target_patch_features,
                target_coords=coords_scaled,
                context_features=context_patch_features,
                context_coords=context_coords_scaled,
                num_context_images=k,
            )

            # Forward through backbone
            backbone_out = self.backbone(**backbone_inputs, return_attn_weights=return_attn_weights)

            # Split predictions
            all_logits = backbone_out['mask_patch_logit_preds']
            patch_logits = all_logits[:, :K]
            context_patch_logits = all_logits[:, K:]

            # Capture attention outputs if requested
            attn_weights = backbone_out.get('attn_weights', None)
            register_tokens = backbone_out.get('register_tokens', None)

            # Aggregate context predictions (apply inverse augmentation first)
            K_per_ctx = K_ctx // k
            context_preds = []
            for ctx_idx in range(k):
                start_idx = ctx_idx * K_per_ctx
                end_idx = (ctx_idx + 1) * K_per_ctx
                ctx_logits = context_patch_logits[:, start_idx:end_idx]
                ctx_coords_slice = context_coords[:, start_idx:end_idx]

                # Apply inverse augmentation to context logits before aggregation
                if self.sampler.augmenter is not None:
                    for b in range(B):
                        ctx_aug = context_aug_params[b][ctx_idx]
                        if ctx_aug and any(v is not None for v in ctx_aug.values()):
                            ctx_logits[b:b+1] = self.sampler.augmenter.inverse(
                                ctx_logits[b:b+1], ctx_aug
                            )

                ctx_pred = self.aggregate(ctx_logits, ctx_coords_slice, (self.resolution, self.resolution))
                context_preds.append(ctx_pred)
            context_pred = torch.stack(context_preds, dim=1)

        else:
            # No context - target only
            backbone_inputs = self._prepare_backbone_inputs(
                target_features=target_patch_features,
                target_coords=coords_scaled,
            )

            backbone_out = self.backbone(**backbone_inputs, return_attn_weights=return_attn_weights)
            patch_logits = backbone_out['mask_patch_logit_preds']

            # Capture attention outputs if requested
            attn_weights = backbone_out.get('attn_weights', None)
            register_tokens = backbone_out.get('register_tokens', None)

        # Apply inverse augmentation before aggregation
        patch_logits_for_agg = patch_logits
        if self.sampler.augmenter is not None and aug_params:
            patch_logits_for_agg = self.sampler.augmenter.inverse(patch_logits, aug_params)

        #print("PatchICL_Level forward: patch_logits_for_agg shape:", patch_logits_for_agg.shape)
        # Aggregate to full prediction
        pred = self.aggregate(patch_logits_for_agg, coords, (self.resolution, self.resolution))

        return {
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
            'attn_weights': attn_weights,
            'register_tokens': register_tokens,
        }


class PatchICL(nn.Module):
    """
    Multi-resolution PatchICL model.

    Chains multiple PatchICL_Level modules at progressively finer resolutions.
    """

    def __init__(self, config: dict, context_size: int = 0):
        super().__init__()
        self.context_size = context_size

        levels_cfg = config.get('levels', [
            {'resolution': 64, 'patch_size': 16, 'num_patches': 16},
            {'resolution': 128, 'patch_size': 16, 'num_patches': 32},
            {'resolution': 224, 'patch_size': 16, 'num_patches': 64},
        ])

        # Oracle config: separate settings for train vs validation
        num_levels = len(levels_cfg)
        default_oracle_train = [True] + [False] * (num_levels - 1)
        default_oracle_valid = [False] * num_levels

        if 'oracle_levels_train' in config or 'oracle_levels_valid' in config:
            self.oracle_levels_train = config.get('oracle_levels_train', default_oracle_train)
            self.oracle_levels_valid = config.get('oracle_levels_valid', default_oracle_valid)
        else:
            oracle = config.get('oracle_levels', default_oracle_train)
            self.oracle_levels_train = oracle
            self.oracle_levels_valid = oracle

        # Sampler config
        sampler_cfg = config.get('sampler', {})
        self.sampler_type = sampler_cfg.get('type', 'weighted')
        self.exploration_noise = sampler_cfg.get('exploration_noise', 0.5)
        self.stride_divisor = sampler_cfg.get('stride_divisor', 4)

        # Gumbel-Softmax specific config
        self.gumbel_tau = sampler_cfg.get('gumbel_tau', 1.0)
        self.gumbel_tau_min = sampler_cfg.get('gumbel_tau_min', 0.1)
        self.gumbel_hard = sampler_cfg.get('gumbel_hard', True)

        # Sliding window specific config
        self.sliding_window_stride = sampler_cfg.get('sliding_window_stride', None)

        # Augmenter config
        aug_cfg = sampler_cfg.get('augmentation', {})
        if aug_cfg.get('enabled', False):
            self.augmenter = PatchAugmenter(
                rotation=aug_cfg.get('rotation', 'none'),
                rotation_range=aug_cfg.get('rotation_range', 0.5),
                flip_horizontal=aug_cfg.get('flip_horizontal', False),
                flip_vertical=aug_cfg.get('flip_vertical', False),
                scale_range=aug_cfg.get('scale_range', None),
            )
        else:
            self.augmenter = None

        # Aggregator config
        self.aggregator_cfg = config.get('aggregator', {})
        self.aggregator_type = self.aggregator_cfg.get('type', 'average')

        # Masking config
        masking_cfg = config.get('masking', {})
        self.target_mask_ratio = masking_cfg.get('target_mask_ratio', 0.0)
        self.context_mask_ratio = masking_cfg.get('context_mask_ratio', 0.0)

        # Mask channels config (1 for binary, 3 for RGB)
        self.num_mask_channels = config.get('num_mask_channels', 1)

        # Loss config
        loss_cfg = config.get('loss', {})
        self.patch_loss_cfg = loss_cfg.get('patch_loss', {'type': 'dice', 'args': None})
        self.aggreg_loss_cfg = loss_cfg.get('aggreg_loss', {'type': 'dice', 'args': None})
        self.patch_criterion = None
        self.aggreg_criterion = None

        # Loss weights config
        weights_cfg = loss_cfg.get('weights', {})
        default_weights = weights_cfg.get('default', {})
        self.default_loss_weights = {
            'target_patch': default_weights.get('target_patch', 1.0),
            'target_aggreg': default_weights.get('target_aggreg', 1.0),
            'context_patch': default_weights.get('context_patch', 1.0),
            'context_aggreg': default_weights.get('context_aggreg', 1.0),
            'feature_patch': default_weights.get('feature_patch', 0.0),
            'feature_aggreg': default_weights.get('feature_aggreg', 0.0),
            'context_feature_patch': default_weights.get('context_feature_patch', 0.0),
            'context_feature_aggreg': default_weights.get('context_feature_aggreg', 0.0),
        }
        self.level_loss_weights = weights_cfg.get('levels', None)

        # Create backbone
        backbone_cfg = config.get('backbone', {})

        # Parse nested config sections
        enc_cfg = backbone_cfg.get('encoder', {})
        attn_cfg = backbone_cfg.get('cross_attention', {})
        dec_cfg = backbone_cfg.get('decoder', {})

        # Check if using new modular config (encoder_type + decoder_type)
        encoder_type = backbone_cfg.get('encoder_type')
        decoder_type = backbone_cfg.get('decoder_type')

        if encoder_type is not None and decoder_type is not None:
            # New modular architecture: separate encoder/decoder selection
            backbone = build_modular_backbone(
                encoder_type=encoder_type,
                decoder_type=decoder_type,
                embed_dim=enc_cfg.get('embed_dim', 1024),
                embed_proj_dim=enc_cfg.get('embed_proj_dim', 128),
                bottleneck_dim=attn_cfg.get('bottleneck_dim', 128),
                num_classes=self.num_mask_channels,
                patch_size=levels_cfg[0]['patch_size'],
                image_size=enc_cfg.get('image_size', 224),
                num_heads=attn_cfg.get('num_heads', 8),
                num_registers=attn_cfg.get('num_registers', 4),
                num_latents=enc_cfg.get('num_latents', 4),
                decoder_hidden_dim=dec_cfg.get('hidden_dim', 64),
                use_rope_2d=attn_cfg.get('use_rope_2d', True),
            )
        else:
            # Legacy config: 'type' selects coupled encoder+decoder
            backbone_type = backbone_cfg.get('type', 'perceiver').lower()

            backbone_kwargs = dict(
                embed_dim=enc_cfg.get('embed_dim', backbone_cfg.get('embed_dim', 1024)),
                embed_proj_dim=enc_cfg.get('embed_proj_dim', backbone_cfg.get('embed_proj_dim', 128)),
                bottleneck_dim=attn_cfg.get('bottleneck_dim', backbone_cfg.get('bottleneck_dim', 128)),
                num_heads=attn_cfg.get('num_heads', backbone_cfg.get('num_heads', 8)),
                num_registers=attn_cfg.get('num_registers', backbone_cfg.get('num_registers', 4)),
                use_rope_2d=attn_cfg.get('use_rope_2d', backbone_cfg.get('use_rope_2d', True)),
                decoder_hidden_dim=dec_cfg.get('hidden_dim', backbone_cfg.get('decoder_hidden_dim', 64)),
                patch_size=levels_cfg[0]['patch_size'],
                image_size=enc_cfg.get('image_size', backbone_cfg.get('image_size', 224)),
                num_classes=self.num_mask_channels,
            )

            if backbone_type == 'cnn':
                backbone = CNNCrossPatchBackbone(**backbone_kwargs)
            elif backbone_type == 'simple':
                # Simplified backbone: CNN encoder + modular attention + CNN decoder
                backbone = SimpleBackbone(
                    embed_dim=backbone_kwargs['embed_proj_dim'],
                    num_heads=backbone_kwargs['num_heads'],
                    num_layers=attn_cfg.get(
                        'num_layers', backbone_cfg.get('num_layers', 1)
                    ),
                    num_registers=backbone_kwargs['num_registers'],
                    num_classes=backbone_kwargs['num_classes'],
                    patch_size=backbone_kwargs['patch_size'],
                    image_size=backbone_kwargs['image_size'],
                    input_dim=backbone_kwargs['embed_dim'],
                    target_self_attention=attn_cfg.get(
                        'target_self_attention', backbone_cfg.get('target_self_attention', False)
                    ),
                    dropout=attn_cfg.get('dropout', backbone_cfg.get('dropout', 0.0)),
                )
            elif backbone_type == 'multilayer':
                # New multi-layer cross-patch attention backbone
                backbone_kwargs['num_latents_per_patch'] = enc_cfg.get(
                    'num_latents_per_patch', backbone_cfg.get('num_latents_per_patch', 4)
                )
                backbone_kwargs['num_layers'] = attn_cfg.get(
                    'num_layers', backbone_cfg.get('num_layers', 3)
                )
                backbone_kwargs['dropout'] = attn_cfg.get(
                    'dropout', backbone_cfg.get('dropout', 0.1)
                )
                backbone_kwargs['use_spatial_attn'] = attn_cfg.get(
                    'use_spatial_attn', backbone_cfg.get('use_spatial_attn', True)
                )
                backbone_kwargs.pop('bottleneck_dim', None)
                backbone = MultiLayerCrossPatchBackbone(**backbone_kwargs)
            else:
                # Default: single-layer perceiver-style backbone
                backbone_kwargs['num_latents_per_patch'] = enc_cfg.get(
                    'num_latents_per_patch', backbone_cfg.get('num_latents_per_patch', 4)
                )
                backbone_kwargs.pop('bottleneck_dim', None)
                backbone = CrossPatchAttentionBackbone(**backbone_kwargs)

        # Create levels
        self.levels = nn.ModuleList()
        for i, lcfg in enumerate(levels_cfg):
            sampler = self._create_sampler(
                patch_size=lcfg['patch_size'],
                num_patches=lcfg['num_patches'],
                temperature=lcfg.get('sampling_temperature', 0.3),
            )
            aggregator = self._create_aggregator(patch_size=lcfg['patch_size'])

            level = PatchICL_Level(
                resolution=lcfg['resolution'],
                patch_size=lcfg['patch_size'],
                num_patches=lcfg['num_patches'],
                backbone=backbone,
                level_idx=i,
                sampling_temperature=lcfg.get('sampling_temperature', 0.3),
                sampler=sampler,
                aggregator=aggregator,
                target_mask_ratio=self.target_mask_ratio,
                context_mask_ratio=self.context_mask_ratio,
                num_mask_channels=self.num_mask_channels,
            )
            self.levels.append(level)

        self.num_levels = len(self.levels)
        self.original_size = None

    def _create_sampler(
        self,
        patch_size: int,
        num_patches: int,
        temperature: float,
    ) -> PatchSampler:
        """Create sampler based on config."""
        if self.sampler_type == 'uniform':
            return UniformSampler(
                patch_size=patch_size,
                num_patches=num_patches,
                temperature=temperature,
                exploration_noise=self.exploration_noise,
                stride_divisor=self.stride_divisor,
                augmenter=self.augmenter,
            )
        elif self.sampler_type == 'topk':
            return DeterministicTopKSampler(
                patch_size=patch_size,
                num_patches=num_patches,
                temperature=temperature,
                exploration_noise=0.0,
                stride_divisor=self.stride_divisor,
                augmenter=self.augmenter,
            )
        elif self.sampler_type == 'gumbel':
            return GumbelSoftmaxSampler(
                patch_size=patch_size,
                num_patches=num_patches,
                temperature=temperature,
                tau=self.gumbel_tau,
                tau_min=self.gumbel_tau_min,
                hard=self.gumbel_hard,
                stride_divisor=self.stride_divisor,
                augmenter=self.augmenter,
            )
        elif self.sampler_type == 'sliding_window':
            return SlidingWindowSampler(
                patch_size=patch_size,
                stride=self.sliding_window_stride,
                augmenter=self.augmenter,
            )
        elif self.sampler_type == 'continuous':
            return ContinuousSampler(
                patch_size=patch_size,
                num_patches=num_patches,
                temperature=temperature,
                augmenter=self.augmenter,
            )
        else:
            return PatchSampler(
                patch_size=patch_size,
                num_patches=num_patches,
                temperature=temperature,
                exploration_noise=self.exploration_noise,
                stride_divisor=self.stride_divisor,
                augmenter=self.augmenter,
            )

    def _create_aggregator(self, patch_size: int) -> PatchAggregator:
        """Create aggregator based on config."""
        return create_aggregator(
            aggregator_type=self.aggregator_type,
            patch_size=patch_size,
            num_mask_channels=self.num_mask_channels,
            **self.aggregator_cfg,
        )

    def _get_loss_weights(self, level_idx: int) -> dict[str, float]:
        """Get loss weights for a specific level."""
        weights = self.default_loss_weights.copy()
        if self.level_loss_weights is not None and level_idx < len(self.level_loss_weights):
            level_weights = self.level_loss_weights[level_idx]
            if level_weights is not None:
                for key in weights:
                    if key in level_weights:
                        weights[key] = level_weights[key]
        return weights

    def set_loss_functions(self, patch_criterion: nn.Module, aggreg_criterion: nn.Module):
        """Set the loss functions for patch and aggreg losses."""
        self.patch_criterion = patch_criterion
        self.aggreg_criterion = aggreg_criterion

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
        """Forward pass through all levels.

        Args:
            return_attn_weights: If True, return attention weights and register tokens
        """
        training = (mode == "train")
        _, _, H, W = image.shape
        self.original_size = (H, W)

        if labels is not None and labels.dim() == 3:
            labels = labels.unsqueeze(1)

        prev_pred = None
        level_outputs = []

        for i, level in enumerate(self.levels):
            coord_scale = H / level.resolution
            oracle_levels = self.oracle_levels_train if training else self.oracle_levels_valid
            use_oracle = oracle_levels[i] if i < len(oracle_levels) else False

            level_out = level(
                image=image,
                labels=labels,
                prev_pred=prev_pred,
                use_oracle=use_oracle,
                original_coords_scale=coord_scale,
                context_in=context_in,
                context_out=context_out,
                target_features=target_features,
                context_features=context_features,
                training=training,
                return_attn_weights=return_attn_weights,
            )
            level_outputs.append(level_out)

            prev_pred = F.interpolate(
                level_out['pred'], size=(H, W), mode='bilinear', align_corners=False
            )

        final_pred = prev_pred
        finest_level = level_outputs[-1]
        coarse_pred = F.interpolate(
            level_outputs[0]['pred'], size=(H, W), mode='bilinear', align_corners=False
        )

        return {
            'final_pred': final_pred,
            'level_outputs': level_outputs,
            'final_logit': final_pred,
            'coarse_pred': coarse_pred,
            'patches': finest_level['patches'],
            'patch_labels': finest_level['patch_labels'],
            'patch_logits': finest_level['patch_logits'],
            'patch_coords': finest_level['coords'],
            'context_patches': finest_level.get('context_patches'),
            'context_patch_labels': finest_level.get('context_patch_labels'),
            'context_patch_logits': finest_level.get('context_patch_logits'),
            'context_coords': finest_level.get('context_coords'),
            'target_mask': finest_level.get('target_mask'),
            'context_mask': finest_level.get('context_mask'),
            'attn_weights': finest_level.get('attn_weights'),
            'register_tokens': finest_level.get('register_tokens'),
        }

    def compute_loss(
        self,
        outputs: dict[str, torch.Tensor],
        labels: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute losses for all levels with configurable weights."""
        if self.patch_criterion is None or self.aggreg_criterion is None:
            raise RuntimeError("Loss functions not set. Call set_loss_functions() first.")

        if labels.dim() == 3:
            labels = labels.unsqueeze(1)

        B = labels.shape[0]
        losses = {}
        total_loss = 0.0

        for i, level_out in enumerate(outputs['level_outputs']):
            weights = self._get_loss_weights(i)
            device = level_out['patch_logits'].device

            # Target patch loss
            patch_logits = level_out['patch_logits']
            patch_labels = level_out['patch_labels']
            K = patch_logits.shape[1]

            target_patch_loss = self.patch_criterion(
                patch_logits.reshape(B * K, -1),
                patch_labels.reshape(B * K, -1),
            )
            losses[f'level_{i}_target_patch_loss'] = target_patch_loss
            losses[f'level_{i}_patch_loss'] = target_patch_loss
            total_loss = total_loss + weights['target_patch'] * target_patch_loss

            # Target aggreg loss
            pred = level_out['pred']
            labels_ds = F.interpolate(labels.float(), size=pred.shape[-2:], mode='nearest')
            target_aggreg_loss = self.aggreg_criterion(pred, labels_ds)
            losses[f'level_{i}_target_aggreg_loss'] = target_aggreg_loss
            losses[f'level_{i}_pred_loss'] = target_aggreg_loss
            total_loss = total_loss + weights['target_aggreg'] * target_aggreg_loss

            # Context patch loss
            context_patch_logits = level_out.get('context_patch_logits')
            context_patch_labels = level_out.get('context_patch_labels')

            if context_patch_logits is not None and context_patch_labels is not None:
                K_ctx = context_patch_logits.shape[1]
                context_patch_loss = self.patch_criterion(
                    context_patch_logits.reshape(B * K_ctx, -1),
                    context_patch_labels.reshape(B * K_ctx, -1),
                )
                losses[f'level_{i}_context_patch_loss'] = context_patch_loss
                total_loss = total_loss + weights['context_patch'] * context_patch_loss
            else:
                losses[f'level_{i}_context_patch_loss'] = torch.tensor(0.0, device=device)

            # Context aggreg loss
            context_pred = level_out.get('context_pred')
            context_labels = level_out.get('context_labels')

            if context_pred is not None and context_labels is not None:
                B_ctx, k_ctx = context_pred.shape[:2]
                context_aggreg_loss = self.aggreg_criterion(
                    context_pred.reshape(B_ctx * k_ctx, -1),
                    context_labels.reshape(B_ctx * k_ctx, -1),
                )
                losses[f'level_{i}_context_aggreg_loss'] = context_aggreg_loss
                total_loss = total_loss + weights['context_aggreg'] * context_aggreg_loss
            else:
                losses[f'level_{i}_context_aggreg_loss'] = torch.tensor(0.0, device=device)

            # Feature losses (placeholders - backbone doesn't return these)
            losses[f'level_{i}_target_feature_patch_loss'] = torch.tensor(0.0, device=device)
            losses[f'level_{i}_target_feature_aggreg_loss'] = torch.tensor(0.0, device=device)
            losses[f'level_{i}_context_feature_patch_loss'] = torch.tensor(0.0, device=device)
            losses[f'level_{i}_context_feature_aggreg_loss'] = torch.tensor(0.0, device=device)

            # Per-level totals
            losses[f'level_{i}_target_loss'] = target_patch_loss + target_aggreg_loss
            losses[f'level_{i}_context_loss'] = (
                losses[f'level_{i}_context_patch_loss'] + losses[f'level_{i}_context_aggreg_loss']
            )
            losses[f'level_{i}_patch_loss_total'] = (
                target_patch_loss + losses[f'level_{i}_context_patch_loss']
            )
            losses[f'level_{i}_aggreg_loss_total'] = (
                target_aggreg_loss + losses[f'level_{i}_context_aggreg_loss']
            )
            losses[f'level_{i}_loss'] = (
                losses[f'level_{i}_target_loss'] + losses[f'level_{i}_context_loss']
            )

        # Final prediction loss
        final_loss = self.aggreg_criterion(outputs['final_pred'], labels)
        losses['final_loss'] = final_loss
        total_loss = total_loss + final_loss
        losses['total_loss'] = total_loss

        # Aggregate losses across levels for logging
        device = outputs['final_pred'].device

        def avg_losses(key):
            vals = [losses.get(f'level_{i}_{key}', torch.tensor(0.0, device=device))
                    for i in range(self.num_levels)]
            return sum(vals) / len(vals)

        losses['target_patch_loss'] = avg_losses('target_patch_loss')
        losses['target_aggreg_loss'] = avg_losses('target_aggreg_loss')
        losses['context_patch_loss'] = avg_losses('context_patch_loss')
        losses['context_aggreg_loss'] = avg_losses('context_aggreg_loss')
        losses['target_feature_patch_loss'] = avg_losses('target_feature_patch_loss')
        losses['target_feature_aggreg_loss'] = avg_losses('target_feature_aggreg_loss')
        losses['context_feature_patch_loss'] = avg_losses('context_feature_patch_loss')
        losses['context_feature_aggreg_loss'] = avg_losses('context_feature_aggreg_loss')

        # Combined totals
        losses['target_loss'] = losses['target_patch_loss'] + losses['target_aggreg_loss']
        losses['context_loss'] = losses['context_patch_loss'] + losses['context_aggreg_loss']
        losses['patch_loss_total'] = losses['target_patch_loss'] + losses['context_patch_loss']
        losses['aggreg_loss_total'] = losses['target_aggreg_loss'] + losses['context_aggreg_loss']

        # Legacy compatibility
        losses['aggreg_loss'] = losses.get('level_0_target_aggreg_loss', torch.tensor(0.0, device=device))
        losses['local_loss'] = losses.get(f'level_{self.num_levels - 1}_target_patch_loss', torch.tensor(0.0, device=device))
        losses['agg_loss'] = final_loss

        return losses
