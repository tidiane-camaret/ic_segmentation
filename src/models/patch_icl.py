"""
Patch-ICL Architecture: Multi-resolution in-context learning for segmentation.

Generalizes global_local to arbitrary number of resolution levels.
Each level processes patches at progressively finer resolution.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.local import LocalDino, LocalDinoLight
from src.models.backbone import PrecomputedFeatureBackbone, PrecomputedDinoBackbone
from src.models.sampling import (
    PatchAugmenter,
    PatchSampler,
    UniformSampler,
    DeterministicTopKSampler,
    GumbelSoftmaxSampler,
)
from src.models.aggregate import PatchAggregator, create_aggregator


def extract_patch_features(
    features: torch.Tensor,
    coords: torch.Tensor,
    patch_size: int,
    level_resolution: int,
    feature_grid_size: int = 14,
) -> torch.Tensor:
    """
    Extract patch features from pre-computed full-image feature maps (vectorized).

    Maps patch coordinates from level resolution to the feature grid and extracts
    the corresponding feature tokens.

    Args:
        features: [B, 196, 1024] - Pre-computed DINOv3 features (14x14 grid)
        coords: [B, K, 2] - Patch coordinates (h, w) at level resolution
        patch_size: Size of patches at level resolution
        level_resolution: Resolution of the current level (e.g., 64, 128, 224)
        feature_grid_size: Size of feature grid (14 for DINOv3 with 224x224 input)

    Returns:
        patch_features: [B, K, tokens_per_patch, 1024] - Extracted features for each patch
    """
    B, K, _ = coords.shape
    device = features.device

    # Map patch coordinates from level resolution to feature grid
    scale = feature_grid_size / level_resolution
    patch_size_in_features = max(1, int(patch_size * scale))
    tokens_per_patch = patch_size_in_features * patch_size_in_features

    # Compute feature grid coordinates: [B, K]
    fh = (coords[:, :, 0].float() * scale).long()
    fw = (coords[:, :, 1].float() * scale).long()

    # Clamp to valid range
    max_start = feature_grid_size - patch_size_in_features
    fh = fh.clamp(0, max_start)
    fw = fw.clamp(0, max_start)

    # Create offset grid for patch extraction: [patch_size_in_features, patch_size_in_features]
    offset_h = torch.arange(patch_size_in_features, device=device)
    offset_w = torch.arange(patch_size_in_features, device=device)
    offset_grid_h, offset_grid_w = torch.meshgrid(offset_h, offset_w, indexing='ij')
    # Flatten offsets: [tokens_per_patch]
    offset_grid_h = offset_grid_h.reshape(-1)
    offset_grid_w = offset_grid_w.reshape(-1)

    # Expand fh, fw to include all tokens in each patch
    # fh: [B, K] -> [B, K, tokens_per_patch]
    fh_expanded = fh.unsqueeze(-1) + offset_grid_h.view(1, 1, -1)
    fw_expanded = fw.unsqueeze(-1) + offset_grid_w.view(1, 1, -1)

    # Convert 2D indices to flat indices: [B, K, tokens_per_patch]
    flat_indices = fh_expanded * feature_grid_size + fw_expanded

    # Expand batch indices: [B, K, tokens_per_patch]
    batch_indices = torch.arange(B, device=device).view(B, 1, 1).expand(B, K, tokens_per_patch)

    # Gather features using advanced indexing
    # features: [B, 196, embed_dim] -> index with [B, K, tokens_per_patch]
    patch_features = features[batch_indices, flat_indices]  # [B, K, tokens_per_patch, embed_dim]

    return patch_features


class PatchICL_Level(nn.Module):
    """
    Single resolution level for PatchICL.

    Each level:
    1. Downsamples inputs to its resolution
    2. Samples K patches weighted by previous prediction
    3. Processes patches through backbone (with optional masking)
    4. Aggregates predictions back to a mask

    Supports masked training where random patches are masked and the model
    must predict their segmentation from context.
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
    ):
        """
        Args:
            resolution: Target resolution for this level (image will be resized to resolution x resolution)
            patch_size: Size of sampled patches (patch_size x patch_size)
            num_patches: Number of patches to sample
            backbone: Shared or level-specific backbone (e.g., LocalDino)
            level_idx: Index of this level (0 = coarsest)
            sampling_temperature: Temperature for patch sampling (lower = sharper distribution)
            sampler: Custom patch sampler (if None, creates default PatchSampler)
            aggregator: Custom patch aggregator (if None, creates default PatchAggregator)
            target_mask_ratio: Ratio of target patches to mask during training (0.0 = no masking)
            context_mask_ratio: Ratio of context patches to mask during training (0.0 = no masking)
        """
        super().__init__()
        self.resolution = resolution
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.backbone = backbone
        self.level_idx = level_idx
        self.sampling_temperature = sampling_temperature
        self.target_mask_ratio = target_mask_ratio
        self.context_mask_ratio = context_mask_ratio

        # Initialize sampler
        if sampler is not None:
            self.sampler = sampler
        else:
            self.sampler = PatchSampler(
                patch_size=patch_size,
                num_patches=num_patches,
                temperature=sampling_temperature,
            )

        # Initialize aggregator
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
        """Downsample mask using nearest neighbor to preserve labels."""
        if mask.shape[-1] == self.resolution and mask.shape[-2] == self.resolution:
            return mask
        return F.interpolate(mask.float(), size=(self.resolution, self.resolution), mode='nearest')

    def select_patches(
        self,
        image: torch.Tensor,
        labels: torch.Tensor,
        weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """
        Select K patches based on weight map.

        Delegates to the sampler module.

        Args:
            image: [B, C, H, W] - image at this level's resolution
            labels: [B, 1, H, W] - GT mask at this level's resolution
            weights: [B, 1, H, W] - weight map for sampling (e.g., prev_pred)

        Returns:
            patches: [B, K, C, ps, ps]
            patch_labels: [B, K, 1, ps, ps]
            coords: [B, K, 2] - (h, w) coordinates
            aug_params: dict with augmentation parameters
        """
        return self.sampler(image, labels, weights)

    def aggregate(
        self,
        patch_logits: torch.Tensor,
        coords: torch.Tensor,
        output_size: tuple[int, int],
        prev_pred: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Aggregate patch predictions back to a full mask.

        Delegates to the aggregator module.

        Args:
            patch_logits: [B, K, 1, ps, ps] - predictions for each patch
            coords: [B, K, 2] - patch coordinates
            output_size: (H, W) - output mask size
            prev_pred: [B, 1, H, W] - previous level prediction (optional, for combining)

        Returns:
            aggregated: [B, 1, H, W] - aggregated prediction at output_size
        """
        return self.aggregator(patch_logits, coords, output_size, prev_pred)

    def _select_context_patches(
        self,
        context_in: torch.Tensor,
        context_out: torch.Tensor,
        context_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Select K patches from each context image.

        Args:
            context_in: [B, k, C, H, W] - context images at this level's resolution
            context_out: [B, k, 1, H, W] - context masks at this level's resolution
            context_weights: [B, k, 1, H, W] - weights for sampling

        Returns:
            context_patches: [B, K*k, C, ps, ps]
            context_patch_labels: [B, K*k, 1, ps, ps]
            context_coords: [B, K*k, 2]
        """
        B, k = context_in.shape[:2]

        all_ctx_patches = []
        all_ctx_labels = []
        all_ctx_coords = []

        for b in range(B):
            batch_patches = []
            batch_labels = []
            batch_coords = []

            for ctx_idx in range(k):
                # Get single context image and mask
                ctx_img = context_in[b, ctx_idx].unsqueeze(0)  # [1, C, H, W]
                ctx_mask = context_out[b, ctx_idx].unsqueeze(0)  # [1, 1, H, W]
                ctx_weight = context_weights[b, ctx_idx].unsqueeze(0)  # [1, 1, H, W]

                # Select K patches from this context (ignore aug_params for context)
                patches, labels, coords, _ = self.select_patches(ctx_img, ctx_mask, ctx_weight)
                # patches: [1, K, C, ps, ps], coords: [1, K, 2]

                batch_patches.append(patches.squeeze(0))  # [K, C, ps, ps]
                batch_labels.append(labels.squeeze(0))  # [K, 1, ps, ps]
                batch_coords.append(coords.squeeze(0))  # [K, 2]

            # Concatenate all context patches: [K*k, C, ps, ps]
            all_ctx_patches.append(torch.cat(batch_patches, dim=0))
            all_ctx_labels.append(torch.cat(batch_labels, dim=0))
            all_ctx_coords.append(torch.cat(batch_coords, dim=0))

        return (
            torch.stack(all_ctx_patches),  # [B, K*k, C, ps, ps]
            torch.stack(all_ctx_labels),   # [B, K*k, 1, ps, ps]
            torch.stack(all_ctx_coords),   # [B, K*k, 2]
        )

    def _generate_patch_mask(
        self,
        num_patches: int,
        mask_ratio: float,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Generate random mask for patches.

        Args:
            num_patches: Number of patches (K or K*k)
            mask_ratio: Fraction of patches to mask
            batch_size: Batch size
            device: Device for tensor

        Returns:
            mask: [B, num_patches] - Boolean tensor, True = masked
        """
        if mask_ratio <= 0.0:
            return torch.zeros(batch_size, num_patches, dtype=torch.bool, device=device)

        num_to_mask = max(1, int(num_patches * mask_ratio))
        mask = torch.zeros(batch_size, num_patches, dtype=torch.bool, device=device)

        for b in range(batch_size):
            # Randomly select patches to mask
            indices = torch.randperm(num_patches, device=device)[:num_to_mask]
            mask[b, indices] = True

        return mask

    def forward(
        self,
        image: torch.Tensor,
        labels: torch.Tensor = None,
        prev_pred: torch.Tensor = None,
        original_coords_scale: float = 1.0,
        context_in: torch.Tensor = None,
        context_out: torch.Tensor = None,
        target_features: torch.Tensor = None,
        context_features: torch.Tensor = None,
        training: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass for this level.

        Args:
            image: [B, C, H, W] - original resolution image
            labels: [B, 1, H, W] - GT mask (for training)
            prev_pred: [B, 1, H, W] - prediction from previous level (at original resolution)
            original_coords_scale: Scale factor for coordinates (to map back to original image)
            context_in: [B, k, C, H, W] - context images (optional)
            context_out: [B, k, 1, H, W] - context masks (optional)
            target_features: [B, 196, 1024] - Pre-computed DINOv3 features for target (optional)
            context_features: [B, k, 196, 1024] - Pre-computed DINOv3 features for context (optional)
            training: Whether in training mode (enables masking)

        Returns:
            Dict with: pred, patches, patch_labels, patch_logits, coords, context_*, masks
        """
        B = image.shape[0]
        device = image.device

        # Ensure labels have channel dim
        if labels is not None and labels.dim() == 3:
            labels = labels.unsqueeze(1)

        # Downsample to this level's resolution
        image_ds = self.downsample(image)
        labels_ds = self.downsample_mask(labels) if labels is not None else None

        # Create weight map for sampling
        if prev_pred is not None:
            weights = self.downsample(prev_pred)
        elif labels is not None:
            weights = labels_ds
        else:
            weights = torch.ones(B, 1, self.resolution, self.resolution, device=device)

        # Default labels if not provided
        if labels_ds is None:
            labels_ds = torch.zeros(B, 1, self.resolution, self.resolution, device=device)

        # Select target patches
        patches, patch_labels, coords, aug_params = self.select_patches(image_ds, labels_ds, weights)

        # Scale coordinates for backbone
        coords_for_backbone = coords.float() * original_coords_scale

        # Extract patch features from pre-computed features if provided
        precomputed_patch_features = None
        if target_features is not None:
            precomputed_patch_features = extract_patch_features(
                features=target_features,
                coords=coords,
                patch_size=self.patch_size,
                level_resolution=self.resolution,
            )

        # Process context if provided
        context_patches = None
        context_patch_labels = None
        context_coords = None

        if context_in is not None and context_out is not None:
            B, k, C, H_ctx, W_ctx = context_in.shape

            # Downsample context images and masks to this level's resolution
            context_in_flat = context_in.view(B * k, C, H_ctx, W_ctx)
            context_out_flat = context_out.view(B * k, 1, H_ctx, W_ctx)

            context_in_ds = self.downsample(context_in_flat).view(B, k, C, self.resolution, self.resolution)
            context_out_ds = self.downsample_mask(context_out_flat).view(B, k, 1, self.resolution, self.resolution)

            # Use context masks as weights for sampling
            context_weights = context_out_ds

            # Select patches from context
            context_patches, context_patch_labels, context_coords = self._select_context_patches(
                context_in_ds, context_out_ds, context_weights
            )

            # Scale context coordinates
            context_coords_for_backbone = context_coords.float() * original_coords_scale

            # Extract context patch features if pre-computed features provided
            precomputed_context_features = None
            if context_features is not None:
                # context_features: [B, k, 196, 1024]
                # context_coords: [B, K*k, 2]
                # We need to extract features for each context image separately
                K_per_ctx = context_coords.shape[1] // k
                ctx_features_list = []
                for ctx_idx in range(k):
                    ctx_feats = context_features[:, ctx_idx]  # [B, 196, 1024]
                    ctx_coords = context_coords[:, ctx_idx * K_per_ctx:(ctx_idx + 1) * K_per_ctx]  # [B, K, 2]
                    extracted = extract_patch_features(
                        features=ctx_feats,
                        coords=ctx_coords,
                        patch_size=self.patch_size,
                        level_resolution=self.resolution,
                    )
                    ctx_features_list.append(extracted)
                # Concatenate: [B, K*k, tokens, embed_dim]
                precomputed_context_features = torch.cat(ctx_features_list, dim=1)

            # Concatenate target and context patches for joint processing
            all_patches = torch.cat([patches, context_patches], dim=1)
            all_coords = torch.cat([coords_for_backbone, context_coords_for_backbone], dim=1)

            # Combine pre-computed features if available
            all_precomputed = None
            if precomputed_patch_features is not None:
                if precomputed_context_features is not None:
                    all_precomputed = torch.cat([precomputed_patch_features, precomputed_context_features], dim=1)
                else:
                    all_precomputed = precomputed_patch_features

            # Generate masks for training (only if backbone supports masking)
            K = patches.shape[1]
            K_ctx = context_patches.shape[1]
            target_mask = None
            context_mask = None
            all_mask = None

            supports_masking = hasattr(self.backbone, 'mask_token') and self.backbone.mask_token is not None
            if training and supports_masking and (self.target_mask_ratio > 0 or self.context_mask_ratio > 0):
                target_mask = self._generate_patch_mask(K, self.target_mask_ratio, B, device)
                context_mask = self._generate_patch_mask(K_ctx, self.context_mask_ratio, B, device)
                all_mask = torch.cat([target_mask, context_mask], dim=1)

            # Compute actual image size from scale factor
            actual_image_size = int(original_coords_scale * self.resolution)

            # Process all patches through backbone
            if supports_masking:
                all_logits = self.backbone(
                    all_patches,
                    coords=all_coords,
                    precomputed_features=all_precomputed,
                    patch_mask=all_mask,
                    actual_image_size=actual_image_size,
                )
            else:
                all_logits = self.backbone(
                    all_patches,
                    coords=all_coords,
                    precomputed_features=all_precomputed,
                    actual_image_size=actual_image_size,
                )

            # Split back: target predictions are first K, context are the rest
            patch_logits = all_logits[:, :K]  # [B, K, 1, ps, ps]
            context_patch_logits = all_logits[:, K:]  # [B, K_ctx, 1, ps, ps]

        else:
            # No context: process target patches only
            K = patches.shape[1]
            target_mask = None
            context_mask = None
            context_patch_logits = None

            supports_masking = hasattr(self.backbone, 'mask_token') and self.backbone.mask_token is not None
            if training and supports_masking and self.target_mask_ratio > 0:
                target_mask = self._generate_patch_mask(K, self.target_mask_ratio, B, device)

            # Compute actual image size from scale factor
            actual_image_size = int(original_coords_scale * self.resolution)

            if supports_masking:
                patch_logits = self.backbone(
                    patches,
                    coords=coords_for_backbone,
                    precomputed_features=precomputed_patch_features,
                    patch_mask=target_mask,
                    actual_image_size=actual_image_size,
                )
            else:
                patch_logits = self.backbone(
                    patches,
                    coords=coords_for_backbone,
                    precomputed_features=precomputed_patch_features,
                    actual_image_size=actual_image_size,
                )

        # Apply inverse augmentation to predictions before aggregation
        # This ensures predictions are in the original orientation for correct placement
        patch_logits_for_agg = patch_logits
        if self.sampler.augmenter is not None and aug_params:
            patch_logits_for_agg = self.sampler.augmenter.inverse(patch_logits, aug_params)

        # Aggregate to full prediction at this level's resolution
        pred = self.aggregate(
            patch_logits_for_agg, coords,
            output_size=(self.resolution, self.resolution),
            # prev_pred=self.downsample(prev_pred) if prev_pred is not None else None,
        )

        return {
            'pred': pred,  # [B, 1, res, res]
            'patches': patches,  # [B, K, C, ps, ps]
            'patch_labels': patch_labels,  # [B, K, 1, ps, ps]
            'patch_logits': patch_logits,  # [B, K, 1, ps, ps] - augmented (for patch-level loss)
            'coords': coords,  # [B, K, 2]
            'context_patches': context_patches,  # [B, K*k, C, ps, ps] or None
            'context_patch_labels': context_patch_labels,  # [B, K*k, 1, ps, ps] or None
            'context_patch_logits': context_patch_logits,  # [B, K*k, 1, ps, ps] or None
            'context_coords': context_coords,  # [B, K*k, 2] or None
            'target_mask': target_mask,  # [B, K] or None - which target patches were masked
            'context_mask': context_mask,  # [B, K*k] or None - which context patches were masked
        }


class PatchICL(nn.Module):
    """
    Multi-resolution PatchICL model.

    Chains multiple PatchICL_Level modules, each operating at progressively
    finer resolution. Predictions flow from coarse to fine levels.

    Oracle mode: Each level can optionally use GT masks instead of previous
    level predictions for patch sampling. Controlled by `oracle_levels` config.
    """

    def __init__(self, config: dict, context_size: int = 0):
        """
        Args:
            config: Config dict with 'levels' list and 'backbone' params
                levels: List of dicts with {resolution, patch_size, num_patches}
                backbone: Backbone config (type, pretrained_path, etc.)
                oracle_levels: List of bools, one per level. True = use GT mask
                    for patch sampling, False = use previous level prediction.
            context_size: Number of context examples (0 = no context)
        """
        super().__init__()
        self.context_size = context_size

        levels_cfg = config.get('levels', [
            {'resolution': 64, 'patch_size': 16, 'num_patches': 16},
            {'resolution': 128, 'patch_size': 16, 'num_patches': 32},
            {'resolution': 224, 'patch_size': 16, 'num_patches': 64},
        ])

        # Oracle config: separate settings for train vs validation
        # This avoids train/test distribution mismatch when using oracle during training
        num_levels = len(levels_cfg)
        default_oracle_train = [True] + [False] * (num_levels - 1)
        default_oracle_valid = [False] * num_levels  # Uniform sampling during validation

        # Support both old single oracle_levels and new separate train/valid
        if 'oracle_levels_train' in config or 'oracle_levels_valid' in config:
            self.oracle_levels_train = config.get('oracle_levels_train', default_oracle_train)
            self.oracle_levels_valid = config.get('oracle_levels_valid', default_oracle_valid)
        else:
            # Backward compatibility: use oracle_levels for both
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

        # Masking config for masked training
        masking_cfg = config.get('masking', {})
        self.target_mask_ratio = masking_cfg.get('target_mask_ratio', 0.0)
        self.context_mask_ratio = masking_cfg.get('context_mask_ratio', 0.0)

        backbone_cfg = config.get('backbone', {})

        # Create shared or per-level backbone
        share_backbone = backbone_cfg.get('share_backbone', True)

        if share_backbone:
            backbone = self._create_backbone(backbone_cfg, levels_cfg[0]['patch_size'])
            backbones = [backbone] * len(levels_cfg)
        else:
            backbones = [
                self._create_backbone(backbone_cfg, lcfg['patch_size'])
                for lcfg in levels_cfg
            ]

        # Create levels
        self.levels = nn.ModuleList()
        for i, lcfg in enumerate(levels_cfg):
            # Create sampler for this level
            sampler = self._create_sampler(
                patch_size=lcfg['patch_size'],
                num_patches=lcfg['num_patches'],
                temperature=lcfg.get('sampling_temperature', 0.3),
            )

            # Create aggregator for this level
            aggregator = self._create_aggregator(patch_size=lcfg['patch_size'])

            level = PatchICL_Level(
                resolution=lcfg['resolution'],
                patch_size=lcfg['patch_size'],
                num_patches=lcfg['num_patches'],
                backbone=backbones[i],
                level_idx=i,
                sampling_temperature=lcfg.get('sampling_temperature', 0.3),
                sampler=sampler,
                aggregator=aggregator,
                target_mask_ratio=self.target_mask_ratio,
                context_mask_ratio=self.context_mask_ratio,
            )
            self.levels.append(level)

        self.num_levels = len(self.levels)

        # Store original image size (will be set from input)
        self.original_size = None

    def _create_backbone(self, backbone_cfg: dict, patch_size: int) -> nn.Module:
        """Create backbone module based on config."""
        backbone_type = backbone_cfg.get('type', 'dino_light')

        if backbone_type == 'dino':
            return LocalDino(
                pretrained_path=backbone_cfg.get('pretrained_path', ''),
                patch_size=patch_size,
                image_size=backbone_cfg.get('image_size', 224),
                freeze_backbone=backbone_cfg.get('freeze', True),
            )
        elif backbone_type == 'dino_light':
            return LocalDinoLight(
                pretrained_path=backbone_cfg.get('pretrained_path'),
                patch_size=patch_size,
                image_size=backbone_cfg.get('image_size', 224),
                embed_dim=backbone_cfg.get('embed_dim', 768),
                num_heads=backbone_cfg.get('num_heads', 8),
                num_layers=backbone_cfg.get('num_layers', 4),
            )
        elif backbone_type == 'precomputed':
            # Lightweight backbone for pre-computed features
            return PrecomputedFeatureBackbone(
                embed_dim=backbone_cfg.get('embed_dim', 1024),
                num_heads=backbone_cfg.get('num_heads', 8),
                num_layers=backbone_cfg.get('num_layers', 4),
                patch_size=patch_size,
                image_size=backbone_cfg.get('image_size', 224),
            )
        elif backbone_type == 'precomputed_dino':
            # Full DINO transformer with pre-computed features
            return PrecomputedDinoBackbone(
                pretrained_path=backbone_cfg.get('pretrained_path', ''),
                patch_size=patch_size,
                image_size=backbone_cfg.get('image_size', 224),
                freeze_backbone=backbone_cfg.get('freeze', True),
            )
        else:
            raise ValueError(f"Unknown backbone type: {backbone_type}")

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
                exploration_noise=0.0,  # No noise for deterministic
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
        else:  # Default: 'weighted'
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
            **self.aggregator_cfg,
        )

    def forward(
        self,
        image: torch.Tensor,
        labels: torch.Tensor = None,
        context_in: torch.Tensor = None,
        context_out: torch.Tensor = None,
        target_features: torch.Tensor = None,
        context_features: torch.Tensor = None,
        mode: str = "train",
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass through all levels.

        Args:
            image: [B, C, H, W] - input image
            labels: [B, 1, H, W] or [B, H, W] - GT mask
            context_in: [B, k, C, H, W] - context images
            context_out: [B, k, 1, H, W] - context masks
            target_features: [B, 196, 1024] - Pre-computed DINOv3 features for target (optional)
            context_features: [B, k, 196, 1024] - Pre-computed DINOv3 features for context (optional)
            mode: "train" or "test" - enables masking during training

        Returns:
            Dict with predictions and intermediate outputs per level
        """
        training = (mode == "train")
        _, _, H, W = image.shape
        self.original_size = (H, W)

        # Ensure labels have channel dim
        if labels is not None and labels.dim() == 3:
            labels = labels.unsqueeze(1)

        # Initialize prev_pred as None for first level
        prev_pred = None

        # Collect outputs from all levels
        level_outputs = []

        for i, level in enumerate(self.levels):
            # Compute coordinate scale factor (for position encoding)
            coord_scale = H / level.resolution

            # Determine whether to use oracle (GT) or previous prediction
            # Use separate oracle settings for train vs validation
            oracle_levels = self.oracle_levels_train if training else self.oracle_levels_valid
            use_oracle = oracle_levels[i] if i < len(oracle_levels) else False
            level_prev_pred = None if use_oracle else prev_pred

            # Forward through level with context and features
            level_out = level(
                image=image,
                labels=labels,
                prev_pred=level_prev_pred,
                original_coords_scale=coord_scale,
                context_in=context_in,
                context_out=context_out,
                target_features=target_features,
                context_features=context_features,
                training=training,
            )
            level_outputs.append(level_out)

            # Upsample prediction to original resolution for next level
            prev_pred = F.interpolate(
                level_out['pred'],
                size=(H, W),
                mode='bilinear',
                align_corners=False,
            )

        # Final prediction is from the finest level
        final_pred = prev_pred

        # Get finest level outputs for compatibility with train_utils
        finest_level = level_outputs[-1]

        # Coarse prediction from first level (upsampled to original size)
        coarse_pred = F.interpolate(
            level_outputs[0]['pred'],
            size=(H, W),
            mode='bilinear',
            align_corners=False,
        )

        return {
            # PatchICL-specific outputs
            'final_pred': final_pred,  # [B, 1, H, W]
            'level_outputs': level_outputs,  # List of level output dicts

            # Compatibility with GlobalLocalModel / train_utils
            'final_logit': final_pred,  # Alias for final_pred
            'coarse_pred': coarse_pred,  # From first (coarsest) level
            'patches': finest_level['patches'],  # [B, K, C, ps, ps]
            'patch_labels': finest_level['patch_labels'],  # [B, K, 1, ps, ps]
            'patch_logits': finest_level['patch_logits'],  # [B, K, 1, ps, ps]
            'patch_coords': finest_level['coords'],  # [B, K, 2]

            # Context patches from finest level
            'context_patches': finest_level.get('context_patches'),
            'context_patch_labels': finest_level.get('context_patch_labels'),
            'context_patch_logits': finest_level.get('context_patch_logits'),
            'context_coords': finest_level.get('context_coords'),

            # Masking info from finest level
            'target_mask': finest_level.get('target_mask'),
            'context_mask': finest_level.get('context_mask'),
        }

    def compute_loss(
        self,
        outputs: dict[str, torch.Tensor],
        labels: torch.Tensor,
        criterion,
        level_weights: list[float] | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Compute losses for all levels.

        Args:
            outputs: Forward pass outputs
            labels: [B, 1, H, W] - GT mask
            criterion: Loss function
            level_weights: Optional weights per level (default: equal)

        Returns:
            Dict with total_loss and per-level losses
        """
        if labels.dim() == 3:
            labels = labels.unsqueeze(1)

        B = labels.shape[0]

        if level_weights is None:
            level_weights = [1.0] * self.num_levels

        losses = {}
        total_loss = 0.0

        total_context_loss = 0.0
        context_loss_count = 0

        for i, (level_out, weight) in enumerate(zip(outputs['level_outputs'], level_weights)):
            # Target patch loss
            patch_logits = level_out['patch_logits']  # [B, K, 1, ps, ps]
            patch_labels = level_out['patch_labels']  # [B, K, 1, ps, ps]
            K = patch_logits.shape[1]

            target_patch_loss = criterion(
                patch_logits.reshape(B * K, -1),
                patch_labels.reshape(B * K, -1),
            )
            losses[f'level_{i}_target_patch_loss'] = target_patch_loss
            # Keep 'patch_loss' as alias for backward compatibility
            losses[f'level_{i}_patch_loss'] = target_patch_loss

            # Context patch loss (if context patches exist)
            context_patch_logits = level_out.get('context_patch_logits')
            context_patch_labels = level_out.get('context_patch_labels')

            if context_patch_logits is not None and context_patch_labels is not None:
                K_ctx = context_patch_logits.shape[1]
                context_patch_loss = criterion(
                    context_patch_logits.reshape(B * K_ctx, -1),
                    context_patch_labels.reshape(B * K_ctx, -1),
                )
                losses[f'level_{i}_context_patch_loss'] = context_patch_loss
                total_context_loss = total_context_loss + weight * context_patch_loss
                context_loss_count += 1
            else:
                losses[f'level_{i}_context_patch_loss'] = torch.tensor(0.0, device=patch_logits.device)

            # Level prediction loss (vs downsampled GT)
            pred = level_out['pred']  # [B, 1, res, res]
            labels_ds = F.interpolate(
                labels.float(),
                size=pred.shape[-2:],
                mode='nearest',
            )
            level_loss = criterion(pred, labels_ds)
            losses[f'level_{i}_pred_loss'] = level_loss

            # Combined level loss (target patches + prediction)
            level_total = target_patch_loss + level_loss
            losses[f'level_{i}_loss'] = level_total
            total_loss = total_loss + weight * level_total

        # Final prediction loss
        final_loss = criterion(outputs['final_pred'], labels)
        losses['final_loss'] = final_loss
        total_loss = total_loss + final_loss

        # Add context loss to total (weighted equally to target loss)
        if context_loss_count > 0:
            total_loss = total_loss + total_context_loss

        losses['total_loss'] = total_loss

        # Aggregate context loss across levels
        losses['context_loss'] = total_context_loss if context_loss_count > 0 else torch.tensor(0.0)

        # Aggregate target patch loss across levels for logging
        target_patch_losses = [losses.get(f'level_{i}_target_patch_loss', torch.tensor(0.0))
                              for i in range(self.num_levels)]
        losses['target_patch_loss'] = sum(target_patch_losses) / len(target_patch_losses)

        # Compatibility with GlobalLocalModel / train_utils
        # Map first level pred_loss to global_loss, last level patch_loss to local_loss
        losses['global_loss'] = losses.get('level_0_pred_loss', torch.tensor(0.0))
        losses['local_loss'] = losses.get(f'level_{self.num_levels - 1}_patch_loss', torch.tensor(0.0))
        losses['agg_loss'] = final_loss  # Aggregation loss = final prediction loss

        return losses
