"""
PatchICL Architecture v2 with multi-level cascaded support.

Supports coarse-to-fine prediction: predict at level 0, use predictions
to guide patch sampling at finer levels.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.patch_icl_v2.aggregate import (
    GaussianAggregator,
    PatchAggregator,
    create_aggregator,
)
from src.models.patch_icl_v2.metrics import compute_dice, GT_AREA_THRESHOLD
from src.models.patch_icl_v2.sampling import (
    ContinuousSampler,
    PatchAugmenter,
    SlidingWindowSampler,
    create_sampler,
)
from src.models.simple_backbone import SimpleBackbone


def extract_mask_patches(
    mask: torch.Tensor,
    coords: torch.Tensor,
    patch_size: int,
    level_resolution: int,
    target_size: int = 16,
) -> torch.Tensor:
    """Extract mask patches using grid_sample (memory-efficient version).

    Args:
        mask: [B, 1, H, W] - Logits or probabilities at level resolution
        coords: [B, K, 2] - Patch coordinates (h, w) at level resolution
        patch_size: Size of patches at level resolution
        level_resolution: Resolution of the current level
        target_size: Desired spatial size for output patches

    Returns:
        mask_patches: [B, K, 1, target_size, target_size]
    """
    B, C, H, W = mask.shape
    K = coords.shape[1]
    device = mask.device

    # Compute patch centers in normalized coordinates [-1, 1]
    fh = coords[:, :, 0].float() + patch_size / 2
    fw = coords[:, :, 1].float() + patch_size / 2

    center_h = (fh / level_resolution) * 2 - 1
    center_w = (fw / level_resolution) * 2 - 1

    # Create sampling grid offsets (shared across all patches)
    half_patch = patch_size / level_resolution
    offset = torch.linspace(-half_patch, half_patch, target_size, device=device)
    grid_y, grid_x = torch.meshgrid(offset, offset, indexing='ij')
    grid_offsets = torch.stack([grid_x, grid_y], dim=-1)  # [target_size, target_size, 2]

    # Build grid: [B, K, target_size, target_size, 2]
    centers = torch.stack([center_w, center_h], dim=-1)  # [B, K, 2]
    grid = centers.view(B, K, 1, 1, 2) + grid_offsets.view(1, 1, target_size, target_size, 2)

    # Memory-efficient extraction: process per batch
    mask_patches_list = []
    for b in range(B):
        # [1, C, H, W] -> [K, C, H, W] via expand (no copy)
        mask_b = mask[b:b+1].expand(K, -1, -1, -1)
        grid_b = grid[b]  # [K, target_size, target_size, 2]
        patches_b = F.grid_sample(
            mask_b, grid_b, mode='bilinear', padding_mode='border', align_corners=False
        )  # [K, C, target_size, target_size]
        mask_patches_list.append(patches_b)

    # Stack: [B, K, C, target_size, target_size]
    mask_patches = torch.stack(mask_patches_list, dim=0)

    return mask_patches


def extract_patch_features(
    features: torch.Tensor,
    coords: torch.Tensor,
    patch_size: int,
    level_resolution: int,
    feature_grid_size: int = 16,
    target_patch_grid_size: int | None = None,
) -> torch.Tensor:
    """Extract patch features from pre-computed full-image feature maps.

    Uses grid_sample with memory-efficient batching (processes one batch at a time
    instead of expanding to B*K copies).

    Args:
        features: [B, N, D] - Pre-computed features (N = actual_grid_size^2)
        coords: [B, K, 2] - Patch coordinates (h, w) at level resolution
        patch_size: Size of patches at level resolution
        level_resolution: Resolution of the current level
        feature_grid_size: Size of feature grid (may differ from actual, will be inferred)
        target_patch_grid_size: Desired spatial size per patch for the backbone encoder.
            If None or equal to native size, no resizing is done.

    Returns:
        patch_features: [B, K, tokens_per_patch, D]
    """
    B, K, _ = coords.shape
    device = features.device
    N = features.shape[1]
    D = features.shape[2]

    # Infer actual grid size from feature count
    actual_grid_size = int(N ** 0.5)
    assert actual_grid_size * actual_grid_size == N, f"Features must be square grid, got N={N}"

    scale = actual_grid_size / level_resolution
    patch_size_in_features = max(1, int(patch_size * scale))

    # Use target grid size for extraction if specified
    extract_size = target_patch_grid_size if target_patch_grid_size else patch_size_in_features

    # Reshape features to spatial format: [B, D, H, W]
    features_spatial = features.view(B, actual_grid_size, actual_grid_size, D)
    features_spatial = features_spatial.permute(0, 3, 1, 2)  # [B, D, H, W]

    # Compute patch centers in normalized coordinates [-1, 1]
    fh = coords[:, :, 0].float() * scale + patch_size_in_features / 2
    fw = coords[:, :, 1].float() * scale + patch_size_in_features / 2

    # Normalize to [-1, 1] for grid_sample
    center_h = (fh / actual_grid_size) * 2 - 1
    center_w = (fw / actual_grid_size) * 2 - 1

    # Create sampling grid offsets (shared across all patches)
    half_patch = (patch_size_in_features / actual_grid_size)
    offset = torch.linspace(-half_patch, half_patch, extract_size, device=device)
    grid_y, grid_x = torch.meshgrid(offset, offset, indexing='ij')
    grid_offsets = torch.stack([grid_x, grid_y], dim=-1)  # [extract_size, extract_size, 2]

    # Build grid: [B, K, extract_size, extract_size, 2]
    centers = torch.stack([center_w, center_h], dim=-1)  # [B, K, 2]
    grid = centers.view(B, K, 1, 1, 2) + grid_offsets.view(1, 1, extract_size, extract_size, 2)

    # Memory-efficient extraction: process per batch instead of expanding B*K
    # This avoids creating K copies of the feature map
    patch_features_list = []
    for b in range(B):
        # [1, D, H, W] -> [K, D, H, W] via expand (no copy)
        feat_b = features_spatial[b:b+1].expand(K, -1, -1, -1)
        grid_b = grid[b]  # [K, extract_size, extract_size, 2]
        patches_b = F.grid_sample(
            feat_b, grid_b, mode='bilinear', padding_mode='border', align_corners=False
        )  # [K, D, extract_size, extract_size]
        patch_features_list.append(patches_b)

    # Stack and reshape: [B, K, D, extract_size^2] -> [B, K, tokens, D]
    patch_features = torch.stack(patch_features_list, dim=0)  # [B, K, D, h, w]
    patch_features = patch_features.view(B, K, D, extract_size * extract_size)
    patch_features = patch_features.permute(0, 1, 3, 2)  # [B, K, tokens, D]

    return patch_features


def compute_entropy_confidence(
    logits: torch.Tensor,
    temperature: float | torch.Tensor = 1.0,
) -> torch.Tensor:
    """Compute confidence from binary prediction entropy.

    Confidence = 1 - H(p)/log(2), where H(p) = -(p*log(p) + (1-p)*log(1-p)).
    Fully differentiable. Returns 1 near p=0 or p=1, 0 at p=0.5.

    Temperature scaling improves calibration:
    - T > 1.0: softens logits -> lower confidence (good for overconfident models)
    - T < 1.0: sharpens logits -> higher confidence
    - T = 1.0: no scaling (default)

    Args:
        logits: [B, K, C, H, W] - raw prediction logits
        temperature: temperature for logit scaling (default 1.0), can be learnable tensor

    Returns:
        confidence: [B, K, 1, H, W] - confidence in [0, 1]
    """
    # Apply temperature scaling before sigmoid (supports learnable tensor)
    is_unit_temp = (temperature == 1.0) if isinstance(temperature, (int, float)) else False
    scaled_logits = logits if is_unit_temp else logits / temperature
    p = torch.sigmoid(scaled_logits)
    p = p.clamp(1e-6, 1 - 1e-6)
    entropy = -(p * p.log() + (1 - p) * (1 - p).log())
    # Normalize by max entropy (log(2)) so output is in [0, 1]
    normalized_entropy = entropy / math.log(2)
    confidence = 1.0 - normalized_entropy
    # If multi-channel, take mean across channels -> [B, K, 1, H, W]
    if confidence.shape[2] > 1:
        confidence = confidence.mean(dim=2, keepdim=True)
    return confidence


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

        # Sampling robustness config (improves train/val generalization gap)
        sampling_robustness_cfg = config.get('sampling_robustness', {})
        self.sampling_dropout = sampling_robustness_cfg.get('dropout', 0.0)  # Probability of using uniform sampling
        self.sampling_temp_start = sampling_robustness_cfg.get('temperature_start', 1.0)  # Start temperature
        self.sampling_temp_end = sampling_robustness_cfg.get('temperature_end', 1.0)  # End temperature
        self.sampling_temp_epochs = sampling_robustness_cfg.get('temperature_epochs', 100)  # Epochs for annealing
        self._current_epoch = 0  # Updated by train loop

        # Sampler config (type shared, per-level instances)
        sampler_cfg = config.get('sampler', {})
        self.sampler_type = sampler_cfg.get('type', 'continuous')
        self.default_stride = sampler_cfg.get('stride', None)

        # Context sampling mode: align with target sampling strategy
        # - "foreground": Original behavior, sample from GT foreground (default)
        # - "border": Sample from object borders (where GT is uncertain after downsampling)
        # - "entropy": Sample from high-entropy regions of soft GT
        self.context_sampling_mode = sampler_cfg.get('context_sampling', 'foreground')
        self.context_border_weight = sampler_cfg.get('context_border_weight', 0.5)

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

        # Per-level samplers (each level can override sampling_method)
        self.samplers = nn.ModuleList()
        for level_cfg in levels_cfg:
            level_sampler_type = level_cfg.get('sampling_method', self.sampler_type)
            level_stride = level_cfg.get('stride', self.default_stride)
            self.samplers.append(create_sampler(
                sampler_type=level_sampler_type,
                patch_size=level_cfg['patch_size'],
                num_patches=level_cfg['num_patches'],
                num_patches_val=level_cfg.get('num_patches_val', level_cfg['num_patches']),
                temperature=level_cfg.get('sampling_temperature', 0.3),
                stride=level_stride,
                augmenter=self.augmenter,
                pad_before=level_cfg.get('pad_before'),
                pad_after=level_cfg.get('pad_after'),
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
            'target_combined': weights_cfg.get('target_combined', 0.0),  # Loss on combined pred (for alpha grad)
            'context_patch': weights_cfg.get('context_patch', 1.0),
            'context_aggreg': weights_cfg.get('context_aggreg', 1.0),
        }
        # Per-level loss weights. Options:
        # - "uniform": all 1.0 (default for backward compatibility)
        # - "progressive": [0.3, 0.7, 1.0] - finer levels weighted more
        # - list of floats: explicit weights
        level_weights_cfg = loss_cfg.get('level_weights', 'uniform')
        if level_weights_cfg == 'uniform':
            self.level_loss_weights = [1.0] * self.num_levels
        elif level_weights_cfg == 'progressive':
            # Linear progression from 0.3 to 1.0
            if self.num_levels == 1:
                self.level_loss_weights = [1.0]
            else:
                self.level_loss_weights = [
                    0.3 + 0.7 * i / (self.num_levels - 1)
                    for i in range(self.num_levels)
                ]
        else:
            self.level_loss_weights = list(level_weights_cfg)
        while len(self.level_loss_weights) < self.num_levels:
            self.level_loss_weights.append(1.0)

        # Confidence method: "learned" (CNN head) or "entropy" (from prediction logits)
        conf_cfg = loss_cfg.get('confidence', {})
        self.confidence_method = conf_cfg.get('method', 'learned')
        assert self.confidence_method in ('learned', 'entropy'), \
            f"confidence.method must be 'learned' or 'entropy', got '{self.confidence_method}'"

        # Temperature scaling for entropy confidence (calibration)
        # T > 1.0 softens overconfident predictions, T < 1.0 sharpens
        # Can be learned via learn_temperature: true
        init_temp = conf_cfg.get('temperature', 1.0)
        self.learn_temperature = conf_cfg.get('learn_temperature', False)
        if self.learn_temperature:
            # Store log(T) as learnable parameter; T = exp(log_T) ensures T > 0
            self._log_temperature = nn.Parameter(torch.tensor(math.log(init_temp)))
        else:
            self.register_buffer('_log_temperature', torch.tensor(math.log(init_temp)))

        # Confidence loss config
        self.confidence_enabled = conf_cfg.get('enabled', False)
        self.confidence_supervision_weight = conf_cfg.get('supervision_weight', 0.0)
        self.confidence_boundary_weight = conf_cfg.get('boundary_weight', 0.0)
        self.confidence_boundary_width = conf_cfg.get('boundary_width', 2)
        self.use_confidence_weighted_seg = conf_cfg.get('weighted_seg', False)

        # Initialize confidence loss functions (only for learned method)
        if self.confidence_enabled and self.confidence_method == 'learned':
            from src.losses import ConfidenceSupervisionLoss, BoundaryConfidenceLoss
            self.conf_supervision_loss = ConfidenceSupervisionLoss()
            self.conf_boundary_loss = BoundaryConfidenceLoss(
                patch_size=self.patch_size,
                border_width=self.confidence_boundary_width,
            )

        # Cascade confidence blending config
        cascade_cfg = config.get('cascade', {})
        self.confidence_blend = cascade_cfg.get('confidence_blend', False)
        self.use_learned_alpha = cascade_cfg.get('learned_alpha', False)
        self.conf_weighted_aggreg = cascade_cfg.get('conf_weighted_aggreg', False)

        # Initialize confidence-weighted segmentation loss if enabled (for patch or aggreg)
        if self.use_confidence_weighted_seg or self.conf_weighted_aggreg:
            from src.losses import ConfidenceWeightedDiceLoss
            self.conf_weighted_seg_loss = ConfidenceWeightedDiceLoss()

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
            gradient_checkpointing=backbone_cfg.get('gradient_checkpointing', False),
            num_context_layers=backbone_cfg.get('num_context_layers', 0),
            use_mask_prior=backbone_cfg.get('use_mask_prior', False),
            mask_fusion_type=backbone_cfg.get('mask_fusion_type', 'additive'),
            use_context_mask=backbone_cfg.get('use_context_mask', False),
            predict_confidence=backbone_cfg.get('predict_confidence', False),
        )

        # Alpha predictor for learned level combination (simple linear on pooled registers)
        if self.use_learned_alpha:
            embed_proj_dim = backbone_cfg.get('embed_proj_dim', 128)
            self.alpha_head = nn.Linear(embed_proj_dim, 1)
            # Initialize with positive bias to favor current level initially
            nn.init.constant_(self.alpha_head.bias, 1.0)  # sigmoid(1) ≈ 0.73

    def compute_alpha(self, register_tokens: torch.Tensor) -> torch.Tensor:
        """Compute combination weight α from register tokens.

        Args:
            register_tokens: [B, num_registers, D] from backbone

        Returns:
            alpha: [B, 1, 1, 1] for broadcasting to spatial dimensions
        """
        if not self.use_learned_alpha or register_tokens is None:
            return None
        # Pool registers and project to scalar
        pooled = register_tokens.mean(dim=1)  # [B, D]
        alpha = torch.sigmoid(self.alpha_head(pooled))  # [B, 1]
        return alpha.view(-1, 1, 1, 1)  # [B, 1, 1, 1]

    @property
    def confidence_temperature(self) -> torch.Tensor:
        """Get current temperature value (always positive via exp)."""
        return self._log_temperature.exp()

    def set_loss_functions(self, patch_criterion: nn.Module, aggreg_criterion: nn.Module):
        """Set the loss functions for patch and aggreg losses."""
        self.patch_criterion = patch_criterion
        self.aggreg_criterion = aggreg_criterion

    def set_feature_extractor(self, feature_extractor: nn.Module):
        """Set or update the feature extractor."""
        self.feature_extractor = feature_extractor

    def set_epoch(self, epoch: int):
        """Set current epoch for temperature annealing."""
        self._current_epoch = epoch

    def _get_sampling_temperature(self) -> float:
        """Compute current sampling temperature based on epoch-based annealing."""
        if self.sampling_temp_epochs <= 0:
            return self.sampling_temp_end
        progress = min(1.0, self._current_epoch / self.sampling_temp_epochs)
        # Linear interpolation from start to end
        return self.sampling_temp_start + progress * (self.sampling_temp_end - self.sampling_temp_start)

    def _apply_sampling_robustness(
        self,
        sampling_weights: torch.Tensor,
        training: bool,
    ) -> torch.Tensor:
        """Apply sampling dropout and temperature scaling for robustness.

        During training:
        - With probability `sampling_dropout`, replace weights with uniform
        - Apply temperature scaling: weights^(1/T) where T is annealed

        Args:
            sampling_weights: [B, 1, H, W] - raw sampling weights
            training: whether in training mode

        Returns:
            Adjusted sampling weights
        """
        if not training:
            return sampling_weights

        B = sampling_weights.shape[0]
        device = sampling_weights.device

        # Sampling dropout: replace some samples with uniform weights
        if self.sampling_dropout > 0:
            dropout_mask = torch.rand(B, 1, 1, 1, device=device) < self.sampling_dropout
            uniform_weights = torch.ones_like(sampling_weights)
            sampling_weights = torch.where(dropout_mask, uniform_weights, sampling_weights)

        # Temperature scaling (annealed)
        temp = self._get_sampling_temperature()
        if temp != 1.0 and temp > 0:
            # Normalize to [0, 1] range, apply temp, renormalize
            w_min = sampling_weights.amin(dim=(2, 3), keepdim=True)
            w_max = sampling_weights.amax(dim=(2, 3), keepdim=True)
            w_range = (w_max - w_min).clamp(min=1e-6)
            w_normalized = (sampling_weights - w_min) / w_range
            # Higher temp -> more uniform; lower temp -> sharper
            w_tempered = w_normalized.pow(1.0 / temp)
            # Renormalize back to original scale (optional, samplers handle unnormalized)
            sampling_weights = w_tempered

        return sampling_weights

    def _downsample(self, x: torch.Tensor, resolution: int) -> torch.Tensor:
        """Downsample tensor to given resolution."""
        if x.shape[-1] == resolution and x.shape[-2] == resolution:
            return x
        return F.interpolate(x, size=(resolution, resolution), mode='bilinear', align_corners=False)

    def _downsample_mask(self, mask: torch.Tensor, resolution: int) -> torch.Tensor:
        """Downsample mask using area interpolation (soft area-fraction targets)."""
        if mask.shape[-1] == resolution and mask.shape[-2] == resolution:
            return mask
        if mask.shape[1] > 1:  # Multi-channel (RGB)
            return F.interpolate(mask.float(), size=(resolution, resolution), mode='bilinear', align_corners=False)
        # Use area interpolation for exact output size (avg_pool2d doesn't guarantee exact resolution)
        return F.interpolate(mask.float(), size=(resolution, resolution), mode='area')

    def _mask_to_weights(self, mask: torch.Tensor) -> torch.Tensor:
        """Convert multi-channel mask to single-channel weights for sampling."""
        if mask.shape[1] == 1:
            return mask
        return mask.max(dim=1, keepdim=True)[0]

    def _compute_context_sampling_weights(
        self,
        context_mask: torch.Tensor,
        mode: str = 'foreground',
        border_weight: float = 0.5,
    ) -> torch.Tensor:
        """Compute sampling weights for context patches.

        Aligns context sampling with target sampling strategy (e.g., borders).

        Args:
            context_mask: [B, 1, H, W] - soft GT mask (already downsampled with area mode)
            mode: Sampling strategy:
                - "foreground": Sample from foreground (original behavior)
                - "border": Sample from object borders
                - "entropy": Sample from high-entropy regions
            border_weight: Blend factor for border mode (0=foreground only, 1=border only)

        Returns:
            sampling_weights: [B, 1, H, W] - weights for patch sampling
        """
        if mode == 'foreground':
            return context_mask

        # Soft mask has values in [0, 1], with ~0.5 at borders
        soft_mask = context_mask.clamp(1e-6, 1 - 1e-6)

        if mode == 'entropy':
            # High entropy where probability is near 0.5 (borders)
            entropy = -(soft_mask * soft_mask.log() + (1 - soft_mask) * (1 - soft_mask).log())
            entropy = entropy / math.log(2)  # Normalize to [0, 1]
            return entropy

        elif mode == 'border':
            # Border score: peaks at 0.5, falls off toward 0 and 1
            # border_score = 1 - 2 * |mask - 0.5| = 1 at 0.5, 0 at 0 or 1
            border_score = 1.0 - 2.0 * torch.abs(soft_mask - 0.5)

            # Blend foreground and border weights
            # Ensure we still sample from foreground regions (not background borders)
            fg_mask = (soft_mask > 0.1).float()  # Only consider regions with some foreground
            border_weights = border_score * fg_mask

            # Blend: (1-α)*foreground + α*border
            weights = (1 - border_weight) * soft_mask + border_weight * border_weights
            return weights

        else:
            # Unknown mode, fallback to foreground
            return context_mask

    def _create_coverage_mask(
        self,
        coords: torch.Tensor,
        patch_size: int,
        output_size: tuple[int, int],
    ) -> torch.Tensor:
        """Create coverage mask from patch coordinates (memory-efficient).

        Uses index_put_ with flattened indices to avoid large intermediate tensors.

        Args:
            coords: [B, K, 2] - Patch coordinates (h, w) at level resolution
            patch_size: Size of patches
            output_size: (H, W) output resolution

        Returns:
            coverage_mask: [B, 1, H, W] - 1 where patches cover, 0 elsewhere
        """
        B, _K, _ = coords.shape
        H, W = output_size
        device = coords.device

        # Initialize flat coverage mask [B, H*W]
        coverage_flat = torch.zeros(B, H * W, device=device)

        # Create patch pixel offsets: [patch_size^2, 2]
        ph = torch.arange(patch_size, device=device)
        pw = torch.arange(patch_size, device=device)
        offset_h, offset_w = torch.meshgrid(ph, pw, indexing='ij')
        offsets_h = offset_h.flatten()  # [patch_size^2]
        offsets_w = offset_w.flatten()  # [patch_size^2]

        # Expand coords to all pixels in each patch
        # coords: [B, K, 2], offsets: [patch_size^2]
        # Result: pixel positions [B, K, patch_size^2]
        h_coords = coords[:, :, 0:1] + offsets_h.view(1, 1, -1)  # [B, K, patch_size^2]
        w_coords = coords[:, :, 1:2] + offsets_w.view(1, 1, -1)  # [B, K, patch_size^2]

        # Clip to valid range and compute flat indices
        h_coords = h_coords.clamp(0, H - 1).long()
        w_coords = w_coords.clamp(0, W - 1).long()
        flat_indices = h_coords * W + w_coords  # [B, K, patch_size^2]

        # Scatter ones to coverage mask
        # Use batch index expansion for scatter
        batch_idx = torch.arange(B, device=device).view(B, 1, 1).expand_as(flat_indices)
        coverage_flat[batch_idx.flatten(), flat_indices.flatten()] = 1.0

        return coverage_flat.view(B, 1, H, W)

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
    ) -> tuple[torch.Tensor, torch.Tensor, list[list[dict]] | None, torch.Tensor, int]:
        """Select patches from all context images in a single batched sampler call."""
        B, k = context_in.shape[:2]

        # Flatten [B, k, ...] -> [B*k, ...] and process all context images at once
        ctx_in_flat = context_in.reshape(B * k, *context_in.shape[2:])
        ctx_out_flat = context_out.reshape(B * k, *context_out.shape[2:])
        ctx_w_flat = context_weights.reshape(B * k, *context_weights.shape[2:])

        _, labels, coords, _, aug_params, validity, K_per = sampler(
            ctx_in_flat, ctx_out_flat, ctx_w_flat
        )

        # Reshape [B*k, K_per, ...] -> [B, k*K_per, ...]
        labels = labels.reshape(B, k * K_per, *labels.shape[2:])
        coords = coords.reshape(B, k * K_per, *coords.shape[2:])
        validity = validity.reshape(B, k * K_per, *validity.shape[2:])

        # Restructure aug_params into [B][k] list-of-lists for downstream compatibility
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

    def _forward_level(
        self,
        level_idx: int,
        image: torch.Tensor,
        labels: torch.Tensor | None,
        context_in: torch.Tensor | None,
        context_out: torch.Tensor | None,
        target_features: torch.Tensor | None,
        context_features: torch.Tensor | None,
        sampling_weights: torch.Tensor,
        H: int,
        W: int,
        return_attn_weights: bool = False,
        mask_prior: torch.Tensor | None = None,
    ) -> tuple[dict, torch.Tensor, torch.Tensor | None]:
        """Process a single resolution level.

        Args:
            mask_prior: [B, 1, H_prev, W_prev] - logits from previous level, used to
                condition attention via mask prior fusion (if backbone supports it).

        Returns:
            level_out: Dict with all level outputs (patches, logits, coords, etc.)
            pred: Aggregated prediction logits [B, C, resolution, resolution]
            aggregated_conf: Aggregated confidence map or None
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

        # Select target patches using sampling_weights (patches=None since extract_patches=False)
        patches, patch_labels, coords, _, aug_params, target_validity, K = sampler(image_ds, labels_ds, sampling_weights, None)
        coord_scale = H / resolution

        # Extract mask prior patches for target patches (if available)
        mask_prior_patches = None
        if mask_prior is not None and getattr(self.backbone, 'use_mask_prior', False):
            # Resize mask prior to current level resolution
            mask_prior_ds = F.interpolate(
                mask_prior, size=(resolution, resolution),
                mode='bilinear', align_corners=False
            )
            mask_prior_patches = extract_mask_patches(
                mask=mask_prior_ds,
                coords=coords,
                patch_size=patch_size,
                level_resolution=resolution,
                target_size=self.patch_feature_grid_size,
            )

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
        context_patch_labels, context_coords = None, None
        context_patch_logits, context_pred, context_out_ds = None, None, None
        context_aug_params = None
        context_validity = None

        if context_in is not None and context_out is not None:
            k = context_in.shape[1]
            context_in_flat = context_in.view(B * k, *context_in.shape[2:])
            context_out_flat = context_out.view(B * k, *context_out.shape[2:])
            context_in_ds = self._downsample(context_in_flat, resolution).view(B, k, -1, resolution, resolution)
            context_out_ds = self._downsample_mask(context_out_flat, resolution).view(B, k, context_out.shape[2], resolution, resolution)

            # Compute context sampling weights (align with target sampling strategy)
            context_mask_flat = self._mask_to_weights(
                context_out_ds.view(B * k, *context_out_ds.shape[2:])
            )
            context_weights = self._compute_context_sampling_weights(
                context_mask_flat,
                mode=self.context_sampling_mode,
                border_weight=self.context_border_weight,
            ).view(B, k, 1, resolution, resolution)

            context_patch_labels, context_coords, context_aug_params, context_validity, K_per_ctx = (
                self._select_context_patches(context_in_ds, context_out_ds, context_weights, sampler)
            )
            K_ctx = K_per_ctx * k

            # Extract context patch features (batched for efficiency)
            context_patch_features = None
            if context_features is not None:
                # Reshape: [B, k, N, D] -> [B*k, N, D] and coords accordingly
                ctx_feats_flat = context_features.view(B * k, *context_features.shape[2:])
                # context_coords: [B, K_ctx] where K_ctx = k * K_per_ctx
                # Reshape to [B*k, K_per_ctx, 2]
                ctx_coords_reshaped = context_coords.view(B, k, K_per_ctx, 2).permute(0, 1, 2, 3)
                ctx_coords_flat = ctx_coords_reshaped.reshape(B * k, K_per_ctx, 2)

                # Single batched extraction
                ctx_extracted_flat = extract_patch_features(
                    features=ctx_feats_flat,
                    coords=ctx_coords_flat,
                    patch_size=patch_size,
                    level_resolution=resolution,
                    feature_grid_size=self.feature_grid_size,
                    target_patch_grid_size=self.patch_feature_grid_size,
                )  # [B*k, K_per_ctx, tokens, D]

                # Reshape back: [B*k, K_per_ctx, tokens, D] -> [B, k*K_per_ctx, tokens, D]
                ctx_extracted = ctx_extracted_flat.view(B, k, K_per_ctx, *ctx_extracted_flat.shape[2:])
                # Apply augmentation per context image if needed
                if self.augmenter is not None and context_aug_params is not None:
                    for ctx_idx in range(k):
                        for b in range(B):
                            ctx_aug = context_aug_params[b][ctx_idx]
                            if ctx_aug and any(v is not None for v in ctx_aug.values()):
                                ctx_extracted[b, ctx_idx] = self.augmenter.augment_features_only(
                                    ctx_extracted[b:b+1, ctx_idx], ctx_aug
                                ).squeeze(0)
                # Merge k dimension into K: [B, k, K_per_ctx, tokens, D] -> [B, K_ctx, tokens, D]
                context_patch_features = ctx_extracted.view(B, K_ctx, *ctx_extracted.shape[3:])

            # Extract context mask patches (batched, if use_context_mask is enabled)
            context_mask_patches = None
            if getattr(self.backbone, 'use_context_mask', False):
                K_per_ctx = K_ctx // k
                # context_out_ds: [B, k, C, resolution, resolution] -> [B*k, 1, resolution, resolution]
                ctx_mask_flat = context_out_ds[:, :, :1].reshape(B * k, 1, resolution, resolution)
                ctx_coords_flat = context_coords.view(B, k, K_per_ctx, 2).reshape(B * k, K_per_ctx, 2)

                # Single batched extraction
                ctx_mask_extracted_flat = extract_mask_patches(
                    mask=ctx_mask_flat,
                    coords=ctx_coords_flat,
                    patch_size=patch_size,
                    level_resolution=resolution,
                    target_size=self.patch_feature_grid_size,
                )  # [B*k, K_per_ctx, 1, h, h]

                # Reshape: [B*k, K_per_ctx, 1, h, h] -> [B, K_ctx, 1, h, h]
                context_mask_patches = ctx_mask_extracted_flat.view(B, K_ctx, *ctx_mask_extracted_flat.shape[2:])

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
                num_target_patches=K,
                mask_prior_patches=mask_prior_patches,
                context_mask_patches=context_mask_patches,
            )

            all_logits = backbone_out['mask_patch_logit_preds']
            patch_logits = all_logits[:, :K]
            context_patch_logits = all_logits[:, K:]

            # Extract confidence predictions (only for target patches - context doesn't use confidence)
            all_confidence = backbone_out.get('confidence_preds')
            patch_confidence = None
            if self.confidence_method == 'entropy':
                patch_confidence = compute_entropy_confidence(
                    patch_logits, temperature=self.confidence_temperature
                )
            elif all_confidence is not None:
                patch_confidence = all_confidence[:, :K]

            # Aggregate context predictions
            context_preds = []
            for ctx_idx in range(k):
                start_idx = ctx_idx * K_per_ctx
                end_idx = (ctx_idx + 1) * K_per_ctx
                ctx_logits = context_patch_logits[:, start_idx:end_idx]
                ctx_coords_slice = context_coords[:, start_idx:end_idx]
                if self.augmenter is not None and context_aug_params is not None:
                    for b in range(B):
                        ctx_aug = context_aug_params[b][ctx_idx]
                        if ctx_aug and any(v is not None for v in ctx_aug.values()):
                            ctx_logits[b:b+1] = self.augmenter.inverse(ctx_logits[b:b+1], ctx_aug)
                agg_result = aggregator(ctx_logits, ctx_coords_slice, (resolution, resolution))
                # Handle tuple return when confidence is used
                ctx_pred = agg_result[0] if isinstance(agg_result, tuple) else agg_result
                context_preds.append(ctx_pred)
            context_pred = torch.stack(context_preds, dim=1)
        else:
            # No context
            ctx_id_labels = torch.zeros(B, K, dtype=torch.long, device=device)
            backbone_out = self.backbone(
                img_patches=target_patch_features, coords=coords.float() * coord_scale,
                ctx_id_labels=ctx_id_labels, return_attn_weights=return_attn_weights,
                level_idx=level_idx,
                mask_prior_patches=mask_prior_patches,
            )
            patch_logits = backbone_out['mask_patch_logit_preds']
            if self.confidence_method == 'entropy':
                patch_confidence = compute_entropy_confidence(
                    patch_logits, temperature=self.confidence_temperature
                )
            else:
                patch_confidence = backbone_out.get('confidence_preds')

        # Apply inverse augmentation and aggregate
        patch_logits_for_agg = patch_logits
        if self.augmenter is not None and aug_params:
            patch_logits_for_agg = self.augmenter.inverse(patch_logits, aug_params)

        # Aggregate with confidence if available
        aggregated_conf = None
        if patch_confidence is not None:
            agg_result = aggregator(
                patch_logits_for_agg, coords, (resolution, resolution),
                confidence=patch_confidence
            )
            pred, aggregated_conf = agg_result
        else:
            pred = aggregator(patch_logits_for_agg, coords, (resolution, resolution))

        level_out = {
            'pred': pred,
            'patches': patches,  # None (not extracted for efficiency)
            'patch_labels': patch_labels,
            'patch_logits': patch_logits,
            'coords': coords,
            'context_patches': None,  # Not extracted for efficiency
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
            'confidence_preds': patch_confidence,
            'aggregated_conf': aggregated_conf,
        }

        return level_out, pred, aggregated_conf

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

        # Process each level with progressive refinement
        combined_pred = None  # Progressively refined prediction
        combined_conf = None  # Progressively refined confidence
        level_outputs = []

        for i in range(self.num_levels):
            level_cfg = self.levels[i]
            resolution = level_cfg['resolution']
            patch_size = level_cfg['patch_size']

            # Determine sampling weights for this level (guides where patches are sampled)
            use_oracle = self.oracle_train[i] if training else self.oracle_valid[i]
            if use_oracle and labels is not None:
                labels_ds = self._downsample_mask(labels, resolution)
                sampling_weights = self._mask_to_weights(labels_ds)
                refined_probs = None
            elif combined_pred is not None:
                # Use previous level outputs for sampling weights
                if combined_conf is not None:
                    # Sample from LOW confidence regions (where model is uncertain)
                    conf_weights = 1.0 - combined_conf
                    if self.detach_between_levels:
                        conf_weights = conf_weights.detach()
                    sampling_weights = F.interpolate(
                        conf_weights, size=(resolution, resolution),
                        mode='bilinear', align_corners=False
                    )
                else:
                    # Fallback: use prediction probabilities
                    combined_prob = torch.sigmoid(combined_pred)
                    if self.detach_between_levels:
                        combined_prob = combined_prob.detach()
                    sampling_weights = F.interpolate(
                        combined_prob, size=(resolution, resolution),
                        mode='bilinear', align_corners=False
                    )
                refined_probs = sampling_weights.clone()
            else:
                sampling_weights = torch.ones(B, 1, resolution, resolution, device=device)
                refined_probs = None

            # Apply sampling robustness (dropout + temperature annealing)
            sampling_weights = self._apply_sampling_robustness(sampling_weights, training)

            # Pass previous prediction as mask prior (level 0 has None)
            mask_prior = None
            if i > 0 and combined_pred is not None:
                mask_prior = combined_pred.detach() if self.detach_between_levels else combined_pred

            level_out, pred, aggregated_conf = self._forward_level(
                level_idx=i,
                image=image,
                labels=labels,
                context_in=context_in,
                context_out=context_out,
                target_features=target_features,
                context_features=context_features,
                sampling_weights=sampling_weights,
                H=H, W=W,
                return_attn_weights=return_attn_weights,
                mask_prior=mask_prior,
            )

            # Progressive refinement: blend current prediction with previous combined
            coords = level_out['coords']
            register_tokens = level_out.get('register_tokens')

            # Compute coverage mask once (reused for blending and level_out)
            coverage = self._create_coverage_mask(coords, patch_size, (resolution, resolution))

            if combined_pred is not None:
                # Upsample previous combined prediction and confidence to current resolution
                # Detach to prevent level N+1 loss from affecting level N representations
                prev_pred = combined_pred.detach() if self.detach_between_levels else combined_pred
                combined_upsampled = F.interpolate(
                    prev_pred, size=(resolution, resolution),
                    mode='bilinear', align_corners=False
                )
                conf_upsampled = None
                if combined_conf is not None:
                    prev_conf = combined_conf.detach() if self.detach_between_levels else combined_conf
                    conf_upsampled = F.interpolate(
                        prev_conf, size=(resolution, resolution),
                        mode='bilinear', align_corners=False
                    )

                eps = 1e-6

                # Learned alpha combination (new approach)
                if self.use_learned_alpha and aggregated_conf is not None and conf_upsampled is not None:
                    # Compute α from current level's register tokens
                    alpha = self.compute_alpha(register_tokens)
                    if alpha is None:
                        alpha = torch.tensor(0.5, device=pred.device)

                    # Store alpha for logging
                    level_out['alpha'] = alpha

                    # Detach predictions, keep gradients for confidence
                    pred_detached = pred.detach()
                    prev_pred_detached = combined_upsampled.detach()

                    # Confidence-weighted combination:
                    # pred_comb = (α × pred_l × conf_l + (1-α) × pred_prev × conf_prev) / (α × conf_l + (1-α) × conf_prev)
                    weighted_curr = alpha * pred_detached * aggregated_conf
                    weighted_prev = (1 - alpha) * prev_pred_detached * conf_upsampled
                    conf_weight_sum = alpha * aggregated_conf + (1 - alpha) * conf_upsampled + eps

                    # In covered regions: use weighted combination; uncovered: use previous
                    blended_pred = (weighted_curr + weighted_prev) / conf_weight_sum
                    combined_pred = coverage * blended_pred + (1 - coverage) * combined_upsampled

                    # Confidence combination: α × conf_l + (1-α) × conf_prev
                    blended_conf = alpha * aggregated_conf + (1 - alpha) * conf_upsampled
                    combined_conf = coverage * blended_conf + (1 - coverage) * conf_upsampled

                # Legacy confidence blend (relative confidence weighting)
                # Detach everything: blend is a heuristic, not an optimization target
                elif self.confidence_blend and combined_conf is not None and aggregated_conf is not None:
                    # Relative weight: current_conf / (current_conf + prev_conf)
                    # Detach confidences to prevent blend from affecting confidence learning
                    conf_curr = aggregated_conf.detach()
                    conf_prev = conf_upsampled.detach()
                    relative_current = conf_curr / (conf_curr + conf_prev + eps)
                    blend_weight = coverage * relative_current
                    combined_pred = blend_weight * pred + (1 - blend_weight) * combined_upsampled
                    combined_conf = coverage * torch.max(aggregated_conf, conf_upsampled) + (1 - coverage) * conf_upsampled

                else:
                    # Standard coverage-based blending
                    combined_pred = coverage * pred + (1 - coverage) * combined_upsampled
                    if aggregated_conf is not None:
                        if conf_upsampled is not None:
                            combined_conf = coverage * aggregated_conf + (1 - coverage) * conf_upsampled
                        else:
                            combined_conf = aggregated_conf
            else:
                combined_pred = pred
                combined_conf = aggregated_conf

            # Store combined prediction for loss computation (provides gradient for alpha)
            level_out['combined_pred'] = combined_pred
            level_out['combined_conf'] = combined_conf

            level_out['refined_probs'] = refined_probs  # Probs used for sampling (after refinement)
            level_out['coverage_mask'] = coverage  # Reuse coverage computed earlier
            level_outputs.append(level_out)

        # Final prediction from refined combined prediction
        final_pred = F.interpolate(combined_pred, size=(H, W), mode='bilinear', align_corners=False)
        coarse_pred = F.interpolate(level_outputs[0]['pred'], size=(H, W), mode='bilinear', align_corners=False)

        # Final confidence map
        final_conf = None
        if combined_conf is not None:
            final_conf = F.interpolate(combined_conf, size=(H, W), mode='bilinear', align_corners=False)

        # Backward compat aliases from last level
        last = level_outputs[-1]
        return {
            'final_pred': final_pred,
            'final_logit': final_pred,
            'final_conf': final_conf,
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
        assert self.patch_criterion is not None, "Call set_loss_functions() first"
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
        sum_target_combined = 0.0
        sum_context_patch = 0.0
        sum_context_aggreg = 0.0

        for i, level_out in enumerate(outputs['level_outputs']):
            lw = self.level_loss_weights[i]

            # Target patch loss
            patch_logits = level_out['patch_logits']
            patch_labels = level_out['patch_labels']
            target_validity = level_out.get('target_validity')
            conf_preds = level_out.get('confidence_preds')
            K = patch_logits.shape[1]

            # Use confidence-weighted loss if enabled and confidence available
            if self.use_confidence_weighted_seg and conf_preds is not None:
                # Reshape for loss: [B*K, C, ps, ps]
                logits_flat = patch_logits.reshape(B * K, *patch_logits.shape[2:])
                labels_flat = patch_labels.reshape(B * K, *patch_labels.shape[2:])
                conf_flat = conf_preds.reshape(B * K, *conf_preds.shape[2:])
                target_patch_loss = self.conf_weighted_seg_loss(
                    logits_flat, labels_flat, conf_flat
                )
            else:
                target_patch_loss = self._masked_patch_loss(
                    patch_logits.reshape(B * K, -1),
                    patch_labels.reshape(B * K, -1),
                    target_validity.reshape(B * K, -1) if target_validity is not None else None,
                )
            losses[f'level_{i}_target_patch_loss'] = target_patch_loss
            total_loss = total_loss + lw * self.loss_weights['target_patch'] * target_patch_loss
            sum_target_patch = sum_target_patch + target_patch_loss

            # Target aggreg loss (area-interpolated targets for exact size match)
            pred = level_out['pred']
            aggregated_conf = level_out.get('aggregated_conf')
            labels_ds = F.interpolate(
                labels.float(), size=pred.shape[-2:], mode='area'
            )

            # Confidence-weighted aggreg loss: model penalized more where confident
            if self.conf_weighted_aggreg and aggregated_conf is not None:
                target_aggreg_loss = self.conf_weighted_seg_loss(pred, labels_ds, aggregated_conf)
            else:
                target_aggreg_loss = self.aggreg_criterion(pred, labels_ds)

            losses[f'level_{i}_target_aggreg_loss'] = target_aggreg_loss
            total_loss = total_loss + lw * self.loss_weights['target_aggreg'] * target_aggreg_loss
            sum_target_aggreg = sum_target_aggreg + target_aggreg_loss

            # Combined prediction loss (provides gradient for alpha in learned combination)
            # Only for levels > 0 where combination actually happens
            combined_pred = level_out.get('combined_pred')
            if i > 0 and self.use_learned_alpha and combined_pred is not None:
                combined_loss_weight = self.loss_weights['target_combined']
                if combined_loss_weight > 0:
                    # Downsample GT to combined_pred resolution
                    labels_combined = F.interpolate(
                        labels.float(), size=combined_pred.shape[-2:], mode='area'
                    )
                    target_combined_loss = self.aggreg_criterion(combined_pred, labels_combined)
                    losses[f'level_{i}_target_combined_loss'] = target_combined_loss
                    total_loss = total_loss + lw * combined_loss_weight * target_combined_loss
                    sum_target_combined = sum_target_combined + target_combined_loss

            # Confidence supervision loss: teach confidence to correlate with correctness
            if self.confidence_supervision_weight > 0 and conf_preds is not None:
                # conf_preds: [B, K, 1, ps, ps], patch_logits: [B, K, C, ps, ps]
                conf_target = 1.0 - torch.abs(torch.sigmoid(patch_logits.detach()) - patch_labels)
                if conf_target.shape[2] > 1:  # Multi-channel: average
                    conf_target = conf_target.mean(dim=2, keepdim=True)
                conf_sup_loss = F.mse_loss(conf_preds, conf_target.detach())
                losses[f'level_{i}_conf_supervision_loss'] = conf_sup_loss
                total_loss = total_loss + lw * self.confidence_supervision_weight * conf_sup_loss

            # Log alpha if using learned combination
            alpha = level_out.get('alpha')
            if alpha is not None:
                losses[f'level_{i}_alpha'] = alpha.mean().detach()

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
                context_labels_ds = F.interpolate(
                    ctx_flat.float(), size=context_pred.shape[-2:], mode='area'
                )
                context_aggreg_loss = self.aggreg_criterion(
                    context_pred.reshape(B_ctx * k_ctx, -1),
                    context_labels_ds.reshape(B_ctx * k_ctx, -1),
                )
                losses[f'level_{i}_context_aggreg_loss'] = context_aggreg_loss
                total_loss = total_loss + lw * self.loss_weights['context_aggreg'] * context_aggreg_loss
                sum_context_aggreg = sum_context_aggreg + context_aggreg_loss
            else:
                losses[f'level_{i}_context_aggreg_loss'] = torch.tensor(0.0, device=device)

        # Dice / soft-dice of refined probs vs GT (logging only, measures sampling guidance quality)
        for i, level_out in enumerate(outputs['level_outputs']):
            refined_probs = level_out.get('refined_probs')
            if refined_probs is not None:
                resolution = refined_probs.shape[-1]
                gt_ds = self._downsample_mask(labels, resolution)
                gt_ds = self._mask_to_weights(gt_ds)
                # Use centralized dice computation (refined_probs already probabilities)
                dice_result = compute_dice(
                    refined_probs, gt_ds, gt_threshold=GT_AREA_THRESHOLD, apply_sigmoid=False
                )
                losses[f'level_{i}_refined_probs_dice'] = dice_result['dice'].mean()
                losses[f'level_{i}_refined_probs_soft_dice'] = dice_result['soft_dice'].mean()

        # Confidence losses (only for learned method - entropy has no learned params)
        sum_conf_supervision = 0.0
        sum_conf_boundary = 0.0
        if self.confidence_enabled and self.confidence_method == 'learned':
            for i, level_out in enumerate(outputs['level_outputs']):
                lw = self.level_loss_weights[i]
                conf_preds = level_out.get('confidence_preds')

                if conf_preds is not None:
                    patch_logits = level_out['patch_logits']
                    patch_labels = level_out['patch_labels']

                    # Confidence supervision loss: MSE(conf, 1 - |pred - gt|)
                    conf_sup_loss = self.conf_supervision_loss(
                        conf_preds, patch_logits, patch_labels
                    )
                    losses[f'level_{i}_conf_supervision_loss'] = conf_sup_loss
                    total_loss = total_loss + lw * self.confidence_supervision_weight * conf_sup_loss
                    sum_conf_supervision = sum_conf_supervision + conf_sup_loss

                    # Boundary confidence penalty
                    if self.confidence_boundary_weight > 0:
                        conf_boundary_loss = self.conf_boundary_loss(conf_preds)
                        losses[f'level_{i}_conf_boundary_loss'] = conf_boundary_loss
                        total_loss = total_loss + lw * self.confidence_boundary_weight * conf_boundary_loss
                        sum_conf_boundary = sum_conf_boundary + conf_boundary_loss

            # Aggregated confidence losses
            losses['conf_supervision_loss'] = sum_conf_supervision / self.num_levels
            losses['conf_boundary_loss'] = sum_conf_boundary / self.num_levels

        losses['total_loss'] = total_loss

        # Log learned temperature if applicable
        if self.confidence_method == 'entropy':
            losses['confidence_temperature'] = self.confidence_temperature.detach()

        # Aggregated losses for logging (mean across levels)
        n = self.num_levels
        losses['target_patch_loss'] = sum_target_patch / n
        losses['target_aggreg_loss'] = sum_target_aggreg / n
        losses['target_combined_loss'] = sum_target_combined / max(n - 1, 1)  # Only levels > 0
        losses['context_patch_loss'] = sum_context_patch / n
        losses['context_aggreg_loss'] = sum_context_aggreg / n
        losses['target_loss'] = losses['target_patch_loss'] + losses['target_aggreg_loss']
        losses['context_loss'] = losses['context_patch_loss'] + losses['context_aggreg_loss']
        losses['patch_loss_total'] = losses['target_patch_loss'] + losses['context_patch_loss']
        losses['aggreg_loss_total'] = losses['target_aggreg_loss'] + losses['context_aggreg_loss']

        return losses
