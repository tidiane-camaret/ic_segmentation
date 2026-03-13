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
from src.models.patch_icl_v2.metrics import compute_dice, GT_AREA_THRESHOLD, _resize_label
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


def compute_entropy_sampling_map(
    logits: torch.Tensor,
    temperature: float | torch.Tensor = 1.0,
) -> torch.Tensor:
    """Compute sampling map from binary prediction entropy.

    Returns 1 - H(p)/log(2), where H(p) = -(p*log(p) + (1-p)*log(1-p)).
    High values (near 1) where predictions are confident (p near 0 or 1).
    Low values (near 0) where predictions are uncertain (p near 0.5).

    Args:
        logits: [B, K, C, H, W] - raw prediction logits
        temperature: temperature for logit scaling (default 1.0)

    Returns:
        sampling_map: [B, K, 1, H, W] - values in [0, 1]
    """
    is_unit_temp = (temperature == 1.0) if isinstance(temperature, (int, float)) else False
    scaled_logits = logits if is_unit_temp else logits / temperature

    orig_dtype = scaled_logits.dtype
    p = torch.sigmoid(scaled_logits.float())
    p = p.clamp(1e-6, 1 - 1e-6)
    entropy = -(p * p.log() + (1 - p) * (1 - p).log())
    normalized_entropy = entropy / math.log(2)
    sampling_map = (1.0 - normalized_entropy).to(orig_dtype)
    if sampling_map.shape[2] > 1:
        sampling_map = sampling_map.mean(dim=2, keepdim=True)
    return sampling_map


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

        # Oracle scheduling config (reduces train/val distribution mismatch)
        # Gradually transitions from oracle (GT) to model predictions during training
        oracle_sched_cfg = config.get('oracle_scheduling', {})
        self.oracle_sched_enabled = oracle_sched_cfg.get('enabled', False)
        self.oracle_sched_schedule = oracle_sched_cfg.get('schedule', 'linear')  # linear, exponential, inverse_sigmoid
        self.oracle_sched_start = oracle_sched_cfg.get('start_prob', 1.0)
        self.oracle_sched_end = oracle_sched_cfg.get('end_prob', 0.3)
        self.oracle_sched_warmup = oracle_sched_cfg.get('warmup_epochs', 10)
        self.oracle_sched_decay = oracle_sched_cfg.get('decay_epochs', 50)

        # Cascade config
        cascade_cfg = config.get('cascade', {})
        self.detach_between_levels = cascade_cfg.get('detach_between_levels', True)
        # Pass register tokens from level N as extra context tokens to level N+1
        self.cascade_registers = cascade_cfg.get('cascade_registers', False)

        self._current_epoch = 0  # Updated by train loop

        # Sampler config (type shared, per-level instances)
        sampler_cfg = config.get('sampler', {})
        self.sampler_type = sampler_cfg.get('type', 'continuous')
        self.default_stride = sampler_cfg.get('stride', None)

        # Sampling modes - clearly separated by source:
        #
        # GT-based modes (for oracle weights and context):
        #   - "gt_foreground": Sample from GT mask foreground (default)
        #   - "gt_border": Sample from GT mask borders (soft mask ≈ 0.5)
        #   - "gt_entropy": Sample from high-entropy regions of soft GT
        #
        # Prediction-based modes (for model weights at levels > 0):
        #   - "predicted_uncertainty": Sample from 1 - confidence (low confidence regions)
        #   - "predicted_entropy": Sample from entropy of prediction logits
        #
        # Target sampling uses oracle mode for oracle weights, model mode for model weights
        # Context sampling always uses GT-based modes (context has GT masks)
        self.target_oracle_mode = sampler_cfg.get('target_sampling', 'gt_foreground')
        self.target_model_mode = sampler_cfg.get('target_model_sampling', 'predicted_uncertainty')
        self.context_sampling_mode = sampler_cfg.get('context_sampling', 'gt_foreground')

        # Backward compatibility: map old mode names to new
        mode_map = {'foreground': 'gt_foreground', 'border': 'gt_border', 'entropy': 'gt_entropy'}
        self.target_oracle_mode = mode_map.get(self.target_oracle_mode, self.target_oracle_mode)
        self.context_sampling_mode = mode_map.get(self.context_sampling_mode, self.context_sampling_mode)

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
        self.context_samplers = nn.ModuleList()  # Separate samplers for context (fewer patches)
        for level_cfg in levels_cfg:
            level_sampler_type = level_cfg.get('sampling_method', self.sampler_type)
            level_stride = level_cfg.get('stride', self.default_stride)
            num_patches = level_cfg['num_patches']
            num_patches_val = level_cfg.get('num_patches_val', num_patches)
            # Target sampler
            # spread_sigma_target overrides spread_sigma for target patches
            target_spread = level_cfg.get('spread_sigma_target', level_cfg.get('spread_sigma', 0.0))
            self.samplers.append(create_sampler(
                sampler_type=level_sampler_type,
                patch_size=level_cfg['patch_size'],
                num_patches=num_patches,
                num_patches_val=num_patches_val,
                temperature=level_cfg.get('sampling_temperature', 0.3),
                stride=level_stride,
                augmenter=self.augmenter,
                pad_before=level_cfg.get('pad_before'),
                pad_after=level_cfg.get('pad_after'),
                spread_sigma=target_spread,
            ))
            # Context sampler (uses num_context_patches if specified, else same as target)
            # spread_sigma_context overrides spread_sigma for context patches
            context_spread = level_cfg.get('spread_sigma_context', level_cfg.get('spread_sigma', 0.0))
            num_context_patches = level_cfg.get('num_context_patches', num_patches)
            num_context_patches_val = level_cfg.get('num_context_patches_val', num_context_patches)
            context_stride = level_cfg.get('context_stride', level_stride)
            self.context_samplers.append(create_sampler(
                sampler_type=level_sampler_type,
                patch_size=level_cfg['patch_size'],
                num_patches=num_context_patches,
                num_patches_val=num_context_patches_val,
                temperature=level_cfg.get('sampling_temperature', 0.3),
                stride=context_stride,
                augmenter=self.augmenter,
                pad_before=level_cfg.get('pad_before'),
                pad_after=level_cfg.get('pad_after'),
                spread_sigma=context_spread,
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

        # Sampling map config - controls the auxiliary head that guides next-level sampling
        # Source: "learned" (CNN head), "entropy" (from logits), "none" (disabled)
        sampling_map_cfg = config.get('sampling_map', {})
        self.sampling_map_source = sampling_map_cfg.get('source', 'none')
        assert self.sampling_map_source in ('learned', 'entropy', 'none'), \
            f"sampling_map.source must be 'learned', 'entropy', or 'none', got '{self.sampling_map_source}'"

        # Temperature for entropy-based sampling map
        init_temp = sampling_map_cfg.get('temperature', 1.0)
        self.learn_temperature = sampling_map_cfg.get('learn_temperature', False)
        if self.learn_temperature:
            self._log_temperature = nn.Parameter(torch.tensor(math.log(init_temp)))
        else:
            self.register_buffer('_log_temperature', torch.tensor(math.log(init_temp)))

        # Training objectives for sampling map (only when source="learned")
        objectives_cfg = sampling_map_cfg.get('objectives', {})
        self.sampling_map_uncertainty_weight = objectives_cfg.get('uncertainty', 0.0)
        self.sampling_map_boundary_weight = objectives_cfg.get('boundary_penalty', 0.0)
        self.sampling_map_boundary_width = objectives_cfg.get('boundary_width', 2)

        # Initialize boundary loss if needed
        if self.sampling_map_source == 'learned' and self.sampling_map_boundary_weight > 0:
            from src.losses import BoundaryConfidenceLoss
            self.sampling_map_boundary_loss = BoundaryConfidenceLoss(
                patch_size=self.patch_size,
                border_width=self.sampling_map_boundary_width,
            )

        # Cascade config
        cascade_cfg = config.get('cascade', {})
        self.sampling_map_blend = cascade_cfg.get('sampling_map_blend', False)
        self.use_learned_alpha = cascade_cfg.get('learned_alpha', False)
        self.additive_fusion = cascade_cfg.get('additive_fusion', False)

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
            predict_sampling_map=(self.sampling_map_source == 'learned'),
            detach_sampling_features=sampling_map_cfg.get('detach_features', True),
        )

        # Alpha predictor for learned level combination (simple linear on pooled registers)
        if self.use_learned_alpha:
            embed_proj_dim = backbone_cfg.get('embed_proj_dim', 128)
            self.alpha_head = nn.Linear(embed_proj_dim, 1)
            # Initialize with positive bias to favor current level initially
            nn.init.constant_(self.alpha_head.bias, 1.0)  # sigmoid(1) ≈ 0.73

    def compute_alpha(self, register_tokens: torch.Tensor) -> torch.Tensor:
        """Compute combination weight α from register tokens.

        Detaches register tokens to prevent target_combined loss from flowing
        gradients back through the backbone. This isolates alpha_head training
        and lets the backbone focus on segmentation quality.

        Args:
            register_tokens: [B, num_registers, D] from backbone

        Returns:
            alpha: [B, 1, 1, 1] for broadcasting to spatial dimensions
        """
        if not self.use_learned_alpha or register_tokens is None:
            return None
        # Pool registers and project to scalar
        # Detach to prevent target_combined gradients from affecting backbone
        pooled = register_tokens.detach().mean(dim=1)  # [B, D]
        alpha = torch.sigmoid(self.alpha_head(pooled))  # [B, 1]
        return alpha.view(-1, 1, 1, 1)  # [B, 1, 1, 1]

    @property
    def sampling_map_temperature(self) -> torch.Tensor:
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
        """Set current epoch for oracle scheduling."""
        self._current_epoch = epoch

    def _get_patch_sampling_map(
        self,
        backbone_out: dict,
        patch_logits: torch.Tensor,
        K: int,
    ) -> torch.Tensor | None:
        """Get per-patch sampling map based on configured source.

        Args:
            backbone_out: Output dict from backbone
            patch_logits: [B, K, C, ps, ps] - patch prediction logits
            K: Number of target patches

        Returns:
            patch_sampling_map: [B, K, 1, ps, ps] or None if source='none'
        """
        if self.sampling_map_source == 'none':
            return None
        elif self.sampling_map_source == 'entropy':
            return compute_entropy_sampling_map(
                patch_logits, temperature=self.sampling_map_temperature
            )
        else:  # 'learned'
            learned_map = backbone_out.get('sampling_map')
            if learned_map is not None:
                return learned_map[:, :K]
            return None

    def _get_oracle_probability(self, level_idx: int) -> float:
        """Compute oracle sampling probability with scheduled decay.

        Implements scheduled sampling to reduce train/val distribution mismatch.
        Gradually transitions from oracle (GT-guided) to model predictions.

        Args:
            level_idx: Level index (scheduling only applies to oracle-enabled levels)

        Returns:
            Probability of using oracle sampling (0.0 to 1.0)
        """
        # Only apply scheduling to levels with oracle_train=True
        if not self.oracle_train[level_idx]:
            return 0.0

        # If scheduling disabled, return 1.0 (full oracle)
        if not self.oracle_sched_enabled:
            return 1.0

        # Warmup period: full oracle
        if self._current_epoch < self.oracle_sched_warmup:
            return self.oracle_sched_start

        # Compute progress through decay period
        decay_progress = min(1.0, (self._current_epoch - self.oracle_sched_warmup) / max(1, self.oracle_sched_decay))

        start = self.oracle_sched_start
        end = self.oracle_sched_end

        if self.oracle_sched_schedule == 'linear':
            return start + decay_progress * (end - start)
        elif self.oracle_sched_schedule == 'exponential':
            # Exponential decay: start * (end/start)^progress
            if end <= 0 or start <= 0:
                return end
            return start * (end / start) ** decay_progress
        elif self.oracle_sched_schedule == 'inverse_sigmoid':
            # Inverse sigmoid: smooth S-curve transition
            k = 5.0  # Controls steepness
            sigmoid_val = k / (k + math.exp(decay_progress * 2 * k - k))
            return end + (start - end) * sigmoid_val
        else:
            return end

    def _downsample(self, x: torch.Tensor, resolution: int) -> torch.Tensor:
        """Downsample tensor to given resolution."""
        if x.shape[-1] == resolution and x.shape[-2] == resolution:
            return x
        return F.interpolate(x, size=(resolution, resolution), mode='bilinear', align_corners=False)

    def _downsample_mask(
        self,
        mask: torch.Tensor,
        resolution: int,
        min_value: float = 1,
    ) -> torch.Tensor:
        """Downsample mask with hybrid approach: soft area fractions + small object preservation.

        Uses area interpolation for soft targets, but ensures small objects get a minimum
        value via max pooling. This prevents tiny labels from vanishing at coarse resolutions.

        Args:
            mask: [B, C, H, W] input mask
            resolution: target resolution
            min_value: minimum value for pixels where max pooling detects foreground
        """
        if mask.shape[-1] == resolution and mask.shape[-2] == resolution:
            return mask

        mask_float = mask.float()

        # Area interpolation: soft area fractions (but small objects get diluted)
        if mask.shape[1] > 1:  # Multi-channel (RGB)
            area_pooled = F.interpolate(mask_float, size=(resolution, resolution), mode='bilinear', align_corners=False)
        else:
            area_pooled = F.interpolate(mask_float, size=(resolution, resolution), mode='area')

        # Max pooling: preserves presence of small objects (binary-ish)
        max_pooled = F.adaptive_max_pool2d(mask_float, (resolution, resolution))

        # Hybrid: use area value, but ensure minimum where max detects foreground
        # This keeps soft boundaries for large objects while preserving small ones
        result = torch.maximum(area_pooled, max_pooled * min_value)


        return result

    def _mask_to_weights(self, mask: torch.Tensor) -> torch.Tensor:
        """Convert multi-channel mask to single-channel weights for sampling."""
        if mask.shape[1] == 1:
            return mask
        return mask.max(dim=1, keepdim=True)[0]

    def _compute_gt_sampling_weights(
        self,
        gt_mask: torch.Tensor,
        mode: str = 'gt_foreground',
    ) -> torch.Tensor:
        """Compute sampling weights from GT mask.

        Args:
            gt_mask: [B, 1, H, W] - soft GT mask (already downsampled with area mode)
            mode: GT-based sampling strategy:
                - "gt_foreground": Sample from foreground (default)
                - "gt_border": Sample from object borders (soft mask ≈ 0.5)
                - "gt_entropy": Sample from high-entropy regions of soft GT
                - "gt_foreground_entropy_balanced": Equal weight to center (foreground) and border (entropy)

        Returns:
            sampling_weights: [B, 1, H, W] - weights for patch sampling
        """
        if mode == 'gt_foreground':
            return gt_mask

        # Compute in fp32 for numerical stability
        orig_dtype = gt_mask.dtype
        soft_mask = gt_mask.float().clamp(1e-6, 1 - 1e-6)

        if mode == 'gt_entropy':
            # High entropy where probability is near 0.5 (borders)
            entropy = -(soft_mask * soft_mask.log() + (1 - soft_mask) * (1 - soft_mask).log())
            entropy = entropy / math.log(2)  # Normalize to [0, 1]
            return entropy.to(orig_dtype)

        elif mode == 'gt_border':
            # Border score: peaks at 0.5, falls off toward 0 and 1
            border_score = 1.0 - 2.0 * torch.abs(soft_mask - 0.5)
            # Only consider regions with some foreground (not background borders)
            fg_mask = (soft_mask > 0.1).float()
            return (border_score * fg_mask).to(orig_dtype)

        elif mode == 'gt_foreground_entropy_balanced':
            # Equal weight to center (foreground) and border (entropy)
            center_weight = soft_mask  # High where p≈1
            entropy = -(soft_mask * soft_mask.log() + (1 - soft_mask) * (1 - soft_mask).log())
            entropy = entropy / math.log(2)  # Normalize to [0, 1]
            combined = 0.5 * center_weight + 0.5 * entropy
            return combined.to(orig_dtype)

        else:
            # Unknown mode, fallback to foreground
            return gt_mask

    def _compute_prediction_sampling_weights(
        self,
        pred_logits: torch.Tensor,
        sampling_map: torch.Tensor | None,
        mode: str = 'predicted_uncertainty',
    ) -> torch.Tensor:
        """Compute sampling weights from model predictions.

        Args:
            pred_logits: [B, C, H, W] - prediction logits from previous level
            sampling_map: [B, 1, H, W] - sampling map from previous level (or None)
            mode: Prediction-based sampling strategy:
                - "sampling_map": Use 1 - sampling_map directly (sample uncertain regions)
                - "predicted_entropy": Compute entropy from prediction logits
                - "predicted_foreground_entropy_balanced": 50% foreground + 50% entropy

        Returns:
            sampling_weights: [B, 1, H, W] - weights for patch sampling
        """
        orig_dtype = pred_logits.dtype

        if mode == 'sampling_map':
            if sampling_map is not None:
                # Sample from low-value regions (high uncertainty)
                return (1.0 - sampling_map).to(orig_dtype)
            else:
                # Fallback: use prediction probabilities
                return torch.sigmoid(pred_logits).to(orig_dtype)

        elif mode == 'predicted_entropy':
            pred_prob = torch.sigmoid(pred_logits.float()).clamp(1e-6, 1 - 1e-6)
            entropy = -(pred_prob * pred_prob.log() + (1 - pred_prob) * (1 - pred_prob).log())
            entropy = entropy / math.log(2)
            if entropy.shape[1] > 1:
                entropy = entropy.max(dim=1, keepdim=True)[0]
            return entropy.to(orig_dtype)

        elif mode == 'predicted_foreground_entropy_balanced':
            pred_prob = torch.sigmoid(pred_logits.float()).clamp(1e-6, 1 - 1e-6)
            center_weight = pred_prob
            entropy = -(pred_prob * pred_prob.log() + (1 - pred_prob) * (1 - pred_prob).log())
            entropy = entropy / math.log(2)
            if center_weight.shape[1] > 1:
                center_weight = center_weight.max(dim=1, keepdim=True)[0]
                entropy = entropy.max(dim=1, keepdim=True)[0]
            combined = 0.5 * center_weight + 0.5 * entropy
            return combined.to(orig_dtype)

        else:
            # Unknown mode, fallback to sampling_map or entropy
            if sampling_map is not None:
                return (1.0 - sampling_map).to(orig_dtype)
            return torch.sigmoid(pred_logits).to(orig_dtype)

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
        labels_ds: torch.Tensor,
        context_in: torch.Tensor | None,
        context_out: torch.Tensor | None,
        target_features: torch.Tensor | None,
        context_features: torch.Tensor | None,
        sampling_weights: torch.Tensor,
        H: int,
        W: int,
        return_attn_weights: bool = False,
        mask_prior: torch.Tensor | None = None,
        prev_register_tokens: torch.Tensor | None = None,
    ) -> tuple[dict, torch.Tensor, torch.Tensor | None]:
        """Process a single resolution level.

        Args:
            labels_ds: [B, C, resolution, resolution] - pre-downsampled labels at level resolution
            mask_prior: [B, 1, H_prev, W_prev] - logits from previous level, used to
                condition attention via mask prior fusion (if backbone supports it).

        Returns:
            level_out: Dict with all level outputs (patches, logits, coords, etc.)
            pred: Aggregated prediction logits [B, C, resolution, resolution]
            aggregated_sampling_map: Aggregated sampling map or None
        """
        level_cfg = self.levels[level_idx]
        resolution = level_cfg['resolution']
        patch_size = level_cfg['patch_size']
        sampler = self.samplers[level_idx]
        context_sampler = self.context_samplers[level_idx]
        aggregator = self.aggregators[level_idx]

        B = image.shape[0]
        device = image.device

        # Downsample image to level resolution (labels_ds passed in from caller)
        image_ds = self._downsample(image, resolution)

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
            context_weights = self._compute_gt_sampling_weights(
                context_mask_flat,
                mode=self.context_sampling_mode,
            ).view(B, k, 1, resolution, resolution)

            context_patch_labels, context_coords, context_aug_params, context_validity, K_per_ctx = (
                self._select_context_patches(context_in_ds, context_out_ds, context_weights, context_sampler)
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
                resolution=float(resolution),  # Pass actual resolution for FiLM conditioning
                num_target_patches=K,
                mask_prior_patches=mask_prior_patches,
                context_mask_patches=context_mask_patches,
                prev_register_tokens=prev_register_tokens,
            )

            all_logits = backbone_out['mask_patch_logit_preds']
            patch_logits = all_logits[:, :K]
            context_patch_logits = all_logits[:, K:]

            # Extract sampling map (only for target patches)
            patch_sampling_map = self._get_patch_sampling_map(backbone_out, patch_logits, K)

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
                resolution=float(resolution),  # Pass actual resolution for FiLM conditioning
                mask_prior_patches=mask_prior_patches,
                prev_register_tokens=prev_register_tokens,
            )
            patch_logits = backbone_out['mask_patch_logit_preds']
            patch_sampling_map = self._get_patch_sampling_map(backbone_out, patch_logits, K)

        # Apply inverse augmentation and aggregate
        patch_logits_for_agg = patch_logits
        if self.augmenter is not None and aug_params:
            patch_logits_for_agg = self.augmenter.inverse(patch_logits, aug_params)

        # Aggregate with sampling_map if available
        aggregated_sampling_map = None
        if patch_sampling_map is not None:
            agg_result = aggregator(
                patch_logits_for_agg, coords, (resolution, resolution),
                sampling_map=patch_sampling_map
            )
            pred, aggregated_sampling_map = agg_result
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
            'patch_sampling_map': patch_sampling_map,
            'aggregated_sampling_map': aggregated_sampling_map,
        }

        return level_out, pred, aggregated_sampling_map

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
        combined_sampling_map = None  # Progressively refined sampling map
        prev_register_tokens = None  # Register tokens from previous level (for cascade_registers)
        level_outputs = []

        for i in range(self.num_levels):
            level_cfg = self.levels[i]
            resolution = level_cfg['resolution']
            patch_size = level_cfg['patch_size']

            # Determine sampling weights for this level (guides where patches are sampled)
            # Supports scheduled sampling: gradual transition from oracle to model predictions
            if training:
                oracle_prob = self._get_oracle_probability(i)
            else:
                oracle_prob = 1.0 if self.oracle_valid[i] else 0.0

            # Downsample labels to level resolution (cached for _forward_level)
            if labels is not None:
                labels_ds = self._downsample_mask(labels, resolution)
            else:
                labels_ds = torch.zeros(B, self.num_mask_channels, resolution, resolution, device=device)

            # Compute oracle weights (from GT labels)
            oracle_weights = None
            if labels is not None:
                oracle_weights = self._compute_gt_sampling_weights(
                    self._mask_to_weights(labels_ds),
                    mode=self.target_oracle_mode,
                )

            # Compute model weights (from previous level predictions)
            model_weights = None
            if combined_pred is not None:
                # Detach to prevent sampling from affecting prediction gradients
                pred_detached = combined_pred.detach() if self.detach_between_levels else combined_pred
                map_detached = combined_sampling_map.detach() if (combined_sampling_map is not None and self.detach_between_levels) else combined_sampling_map

                # Compute weights at combined_pred resolution, then resize
                model_weights_fullres = self._compute_prediction_sampling_weights(
                    pred_detached, map_detached, mode=self.target_model_mode,
                )
                model_weights = F.interpolate(
                    model_weights_fullres, size=(resolution, resolution),
                    mode='bilinear', align_corners=False
                )

            # Mix oracle and model weights based on scheduled probability
            refined_probs = None
            if oracle_prob >= 1.0 and oracle_weights is not None:
                # Full oracle mode
                sampling_weights = oracle_weights
            elif oracle_prob <= 0.0 or oracle_weights is None:
                # Full model mode (or no oracle available)
                if model_weights is not None:
                    sampling_weights = model_weights
                    refined_probs = sampling_weights.clone()
                else:
                    sampling_weights = torch.ones(B, 1, resolution, resolution, device=device)
            else:
                # Scheduled sampling: stochastic mix per sample
                if model_weights is None:
                    model_weights = torch.ones(B, 1, resolution, resolution, device=device)
                # Per-sample decision: use oracle or model weights
                use_oracle_mask = torch.rand(B, 1, 1, 1, device=device) < oracle_prob
                sampling_weights = torch.where(use_oracle_mask, oracle_weights, model_weights)
                refined_probs = model_weights.clone()  # Track model weights for logging

            # Pass previous prediction as mask prior (level 0 has None)
            mask_prior = None
            if i > 0 and combined_pred is not None:
                mask_prior = combined_pred.detach() if self.detach_between_levels else combined_pred

            level_out, pred, aggregated_sampling_map = self._forward_level(
                level_idx=i,
                image=image,
                labels=labels,
                labels_ds=labels_ds,
                context_in=context_in,
                context_out=context_out,
                target_features=target_features,
                context_features=context_features,
                sampling_weights=sampling_weights,
                H=H, W=W,
                return_attn_weights=return_attn_weights,
                mask_prior=mask_prior,
                prev_register_tokens=prev_register_tokens if self.cascade_registers else None,
            )

            # Progressive refinement: blend current prediction with previous combined
            coords = level_out['coords']
            register_tokens = level_out.get('register_tokens')

            # Update cascade register tokens for the next level
            if self.cascade_registers and register_tokens is not None:
                prev_register_tokens = register_tokens.detach() if self.detach_between_levels else register_tokens

            # Compute coverage mask once (reused for blending and level_out)
            coverage = self._create_coverage_mask(coords, patch_size, (resolution, resolution))

            if combined_pred is not None:
                # Upsample previous combined prediction and sampling_map to current resolution
                prev_pred = combined_pred.detach() if self.detach_between_levels else combined_pred
                combined_upsampled = F.interpolate(
                    prev_pred, size=(resolution, resolution),
                    mode='bilinear', align_corners=False
                )
                map_upsampled = None
                if combined_sampling_map is not None:
                    prev_map = combined_sampling_map.detach() if self.detach_between_levels else combined_sampling_map
                    map_upsampled = F.interpolate(
                        prev_map, size=(resolution, resolution),
                        mode='bilinear', align_corners=False
                    )

                # Use larger epsilon for fp16 numerical stability
                eps = 1e-4 if pred.dtype == torch.float16 else 1e-6

                # Learned alpha combination
                if self.use_learned_alpha and aggregated_sampling_map is not None and map_upsampled is not None:
                    alpha = self.compute_alpha(register_tokens)
                    if alpha is None:
                        alpha = torch.tensor(0.5, device=pred.device)

                    level_out['alpha'] = alpha

                    pred_detached = pred.detach()
                    prev_pred_detached = combined_upsampled.detach()

                    # Sampling-map-weighted combination
                    weighted_curr = alpha * pred_detached * aggregated_sampling_map
                    weighted_prev = (1 - alpha) * prev_pred_detached * map_upsampled
                    map_weight_sum = alpha * aggregated_sampling_map + (1 - alpha) * map_upsampled + eps

                    blended_pred = (weighted_curr + weighted_prev) / map_weight_sum
                    combined_pred = coverage * blended_pred + (1 - coverage) * combined_upsampled

                    blended_map = alpha * aggregated_sampling_map + (1 - alpha) * map_upsampled
                    combined_sampling_map = coverage * blended_map + (1 - coverage) * map_upsampled

                # Additive fusion: direct logit addition
                elif self.additive_fusion:
                    combined_pred = combined_upsampled + coverage * pred
                    if aggregated_sampling_map is not None:
                        if map_upsampled is not None:
                            # Weighted average: current level overrides previous in covered regions
                            combined_sampling_map = coverage * aggregated_sampling_map + (1 - coverage) * map_upsampled
                        else:
                            combined_sampling_map = aggregated_sampling_map

                # Sampling map blend (relative weighting)
                elif self.sampling_map_blend and combined_sampling_map is not None and aggregated_sampling_map is not None:
                    map_curr = aggregated_sampling_map.detach()
                    map_prev = map_upsampled.detach()
                    relative_current = map_curr / (map_curr + map_prev + eps)
                    blend_weight = coverage * relative_current
                    combined_pred = blend_weight * pred + (1 - blend_weight) * combined_upsampled
                    combined_sampling_map = coverage * torch.max(aggregated_sampling_map, map_upsampled) + (1 - coverage) * map_upsampled

                else:
                    # Standard coverage-based blending (default)
                    combined_pred = coverage * pred + (1 - coverage) * combined_upsampled
                    if aggregated_sampling_map is not None:
                        if map_upsampled is not None:
                            combined_sampling_map = coverage * aggregated_sampling_map + (1 - coverage) * map_upsampled
                        else:
                            combined_sampling_map = aggregated_sampling_map
            else:
                combined_pred = pred
                combined_sampling_map = aggregated_sampling_map

            # Store combined prediction for loss computation
            level_out['combined_pred'] = combined_pred
            level_out['combined_sampling_map'] = combined_sampling_map

            level_out['refined_probs'] = refined_probs  # Probs used for sampling (after refinement)
            level_out['coverage_mask'] = coverage  # Reuse coverage computed earlier
            level_outputs.append(level_out)

        # Final prediction from refined combined prediction
        final_pred = F.interpolate(combined_pred, size=(H, W), mode='bilinear', align_corners=False)
        coarse_pred = F.interpolate(level_outputs[0]['pred'], size=(H, W), mode='bilinear', align_corners=False)

        # Final sampling map
        final_sampling_map = None
        if combined_sampling_map is not None:
            final_sampling_map = F.interpolate(combined_sampling_map, size=(H, W), mode='bilinear', align_corners=False)

        # Backward compat aliases from last level
        last = level_outputs[-1]
        return {
            'final_pred': final_pred,
            'final_logit': final_pred,
            'final_sampling_map': final_sampling_map,
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
            patch_sampling_map = level_out.get('patch_sampling_map')
            K = patch_logits.shape[1]

            target_patch_loss = self._masked_patch_loss(
                patch_logits.reshape(B * K, -1),
                patch_labels.reshape(B * K, -1),
                target_validity.reshape(B * K, -1) if target_validity is not None else None,
            )
            losses[f'level_{i}_target_patch_loss'] = target_patch_loss
            total_loss = total_loss + lw * self.loss_weights['target_patch'] * target_patch_loss
            sum_target_patch = sum_target_patch + target_patch_loss

            # Target aggreg loss
            pred = level_out['pred']
            level_res = pred.shape[-2:]
            labels_ds = _resize_label(labels.float(), size=level_res)
            target_aggreg_loss = self.aggreg_criterion(pred, labels_ds)

            losses[f'level_{i}_target_aggreg_loss'] = target_aggreg_loss
            total_loss = total_loss + lw * self.loss_weights['target_aggreg'] * target_aggreg_loss
            sum_target_aggreg = sum_target_aggreg + target_aggreg_loss

            # Combined prediction loss (supervises the fused multi-level output)
            # For additive_fusion: provides gradient through combined = prev + coverage * pred
            # For learned_alpha: provides gradient for alpha head
            # Only for levels > 0 where combination actually happens
            combined_pred = level_out.get('combined_pred')
            if i > 0 and (self.additive_fusion or self.use_learned_alpha) and combined_pred is not None:
                combined_loss_weight = self.loss_weights['target_combined']
                if combined_loss_weight > 0:
                    # Reuse labels_ds if combined_pred is at same resolution (common case)
                    combined_res = combined_pred.shape[-2:]
                    labels_combined = labels_ds if combined_res == level_res else _resize_label(labels.float(), size=combined_res)
                    target_combined_loss = self.aggreg_criterion(combined_pred, labels_combined)
                    losses[f'level_{i}_target_combined_loss'] = target_combined_loss
                    total_loss = total_loss + lw * combined_loss_weight * target_combined_loss
                    sum_target_combined = sum_target_combined + target_combined_loss

            # Sampling map training objectives (only for source='learned')
            if self.sampling_map_source == 'learned' and patch_sampling_map is not None:
                # Uncertainty objective: map should be high where prediction is correct
                if self.sampling_map_uncertainty_weight > 0:
                    # Target: high value where confident (low GT entropy AND low pred error)
                    soft_gt = patch_labels.float().clamp(1e-6, 1 - 1e-6)
                    gt_entropy = -(soft_gt * soft_gt.log() + (1 - soft_gt) * (1 - soft_gt).log())
                    gt_entropy = gt_entropy / math.log(2)

                    pred_error = torch.abs(torch.sigmoid(patch_logits.detach()) - patch_labels)
                    uncertainty = torch.maximum(gt_entropy, pred_error)
                    map_target = 1.0 - uncertainty

                    if map_target.shape[2] > 1:
                        map_target = map_target.mean(dim=2, keepdim=True)

                    uncertainty_loss = F.mse_loss(patch_sampling_map, map_target.detach())
                    losses[f'level_{i}_sampling_map_uncertainty_loss'] = uncertainty_loss
                    total_loss = total_loss + lw * self.sampling_map_uncertainty_weight * uncertainty_loss

                # Boundary penalty: penalize high values at patch borders
                if self.sampling_map_boundary_weight > 0:
                    boundary_loss = self.sampling_map_boundary_loss(patch_sampling_map)
                    losses[f'level_{i}_sampling_map_boundary_loss'] = boundary_loss
                    total_loss = total_loss + lw * self.sampling_map_boundary_weight * boundary_loss

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
                context_labels_ds = _resize_label(ctx_flat.float(), size=context_pred.shape[-2:])
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

        losses['total_loss'] = total_loss

        # Log learned temperature if applicable
        if self.sampling_map_source == 'entropy':
            losses['sampling_map_temperature'] = self.sampling_map_temperature.detach()

        # Log oracle scheduling probabilities if enabled
        if self.oracle_sched_enabled:
            for i in range(self.num_levels):
                if self.oracle_train[i]:
                    losses[f'level_{i}_oracle_prob'] = self._get_oracle_probability(i)

        # Hierarchical diagnostic metrics: analyze level l improvement over level l-1
        # Helps identify why levels > 0 might underperform
        level_outputs = outputs.get('level_outputs', [])
        for i in range(1, len(level_outputs)):
            prev_level = level_outputs[i - 1]
            curr_level = level_outputs[i]

            # Get predictions and coverage
            prev_pred = prev_level['pred']  # [B, C, prev_res, prev_res]
            curr_pred = curr_level['pred']  # [B, C, curr_res, curr_res]
            combined_pred = curr_level.get('combined_pred')  # [B, C, curr_res, curr_res]
            coverage_mask = curr_level.get('coverage_mask')  # [B, 1, curr_res, curr_res]

            if coverage_mask is None:
                continue

            curr_res = curr_pred.shape[-2:]

            # Upsample prev_pred to current resolution for comparison
            prev_pred_up = F.interpolate(prev_pred, size=curr_res, mode='bilinear', align_corners=False)

            # Downsample GT to current resolution
            labels_curr = _resize_label(labels.float(), size=curr_res)

            # Compute softdice on covered vs uncovered regions
            with torch.no_grad():
                covered = coverage_mask > 0.5  # [B, 1, H, W]
                uncovered = ~covered

                # Expand coverage mask to match prediction channels
                covered_exp = covered.expand_as(curr_pred)
                uncovered_exp = uncovered.expand_as(curr_pred)

                # Softdice helper for masked regions
                def masked_softdice(pred, gt, mask):
                    """Compute softdice only on masked region."""
                    pred_prob = torch.sigmoid(pred)
                    # Mask both pred and gt
                    pred_masked = pred_prob * mask.float()
                    gt_masked = gt * mask.float()
                    # Soft intersection and union
                    intersection = (pred_masked * gt_masked).sum(dim=(1, 2, 3))
                    denom = pred_masked.sum(dim=(1, 2, 3)) + gt_masked.sum(dim=(1, 2, 3))
                    # Check if mask has any pixels
                    mask_sum = mask.float().sum(dim=(1, 2, 3))
                    valid = mask_sum > 0
                    dice = torch.where(valid, (2 * intersection + 1e-6) / (denom + 1e-6), torch.zeros_like(intersection))
                    return dice.mean(), valid.float().mean()

                # 1. Level l-1 pred vs GT on region COVERED by level l
                prev_covered_dice, prev_covered_valid = masked_softdice(prev_pred_up, labels_curr, covered_exp)
                losses[f'level_{i}_prev_covered_softdice'] = prev_covered_dice

                # 2. Level l pred vs GT on region COVERED by level l
                curr_covered_dice, curr_covered_valid = masked_softdice(curr_pred, labels_curr, covered_exp)
                losses[f'level_{i}_curr_covered_softdice'] = curr_covered_dice

                # 3. Level l-1 pred vs GT on region UNCOVERED by level l
                prev_uncovered_dice, prev_uncovered_valid = masked_softdice(prev_pred_up, labels_curr, uncovered_exp)
                losses[f'level_{i}_prev_uncovered_softdice'] = prev_uncovered_dice

                # 4. Level l pred vs GT on region UNCOVERED by level l (should match prev since no update)
                curr_uncovered_dice, _ = masked_softdice(curr_pred, labels_curr, uncovered_exp)
                losses[f'level_{i}_curr_uncovered_softdice'] = curr_uncovered_dice

                # 5. Combined pred vs GT on region COVERED by level l
                if combined_pred is not None:
                    combined_covered_dice, _ = masked_softdice(combined_pred, labels_curr, covered_exp)
                    losses[f'level_{i}_combined_covered_softdice'] = combined_covered_dice

                    # Improvement metrics (positive = better)
                    losses[f'level_{i}_level_improvement'] = curr_covered_dice - prev_covered_dice
                    losses[f'level_{i}_combination_effect'] = combined_covered_dice - curr_covered_dice

                # Coverage statistics
                losses[f'level_{i}_coverage_ratio'] = covered.float().mean()

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
