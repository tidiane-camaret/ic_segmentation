"""PatchICL v3 - Clean multi-level cascaded model.

Simplified from v2 with:
- Single level combination mode (additive_fusion)
- Unified sampling weight computation
- Config via dataclasses
- Removed dead code (learned_alpha, sampling_map_blend, gt_border, etc.)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from .aggregate import create_aggregator
from .backbone import SimpleBackbone
from .level import LevelConfig, LevelOutput, process_level
from .sampling import PatchAugmenter, compute_sampling_weights, create_sampler
from .utils import downsample_mask, mask_to_weights, resize_label


@dataclass
class PatchICLConfig:
    """Configuration for PatchICL model."""
    # Level configs
    levels: list[dict] = field(default_factory=lambda: [
        {'resolution': 32, 'patch_size': 16, 'num_patches': 16}
    ])

    # Oracle config
    oracle_levels_train: list[bool] = field(default_factory=lambda: [True])
    oracle_levels_valid: list[bool] = field(default_factory=lambda: [False])

    # Oracle scheduling
    oracle_scheduling_enabled: bool = False
    oracle_scheduling_warmup: int = 10
    oracle_scheduling_decay: int = 50
    oracle_scheduling_start: float = 1.0
    oracle_scheduling_end: float = 0.3

    # Cascade config
    detach_between_levels: bool = True
    cascade_registers: bool = False
    skip_context_decoding: bool = False  # Skip decoder for context patches (~20% speedup)

    # Sampling config
    sampler_type: str = "continuous"
    target_sampling_oracle: str = "gt_foreground"
    target_model_sampling: str = "predicted_entropy"
    context_sampling: str = "gt_foreground"
    differentiable_sampling: bool = False

    # Sampling map
    sampling_map_source: str = "none"  # "entropy", "learned", or "none"
    sampling_map_temperature: float = 1.0
    sampling_map_learn_temperature: bool = False

    # Aggregator
    aggregator_type: str = "average"

    # Loss weights
    loss_target_patch: float = 0.0
    loss_target_aggreg: float = 0.0
    loss_target_combined: float = 1.0
    loss_context_patch: float = 0.0
    loss_context_aggreg: float = 1.0
    level_weights: list[float] | str = field(default_factory=lambda: "uniform")

    # Augmentation
    augmentation_enabled: bool = False
    augmentation_rotation: str = "none"
    augmentation_rotation_range: float = 0.5
    augmentation_flip_horizontal: bool = False
    augmentation_flip_vertical: bool = False

    # Backbone
    backbone_embed_dim: int = 1024
    backbone_embed_proj_dim: int = 128
    backbone_num_heads: int = 8
    backbone_num_layers: int = 1
    backbone_num_registers: int = 4
    backbone_feature_grid_size: int = 16
    backbone_patch_feature_grid_size: int = 16
    backbone_dropout: float = 0.0
    backbone_decoder_use_skip_connections: bool = True
    backbone_gradient_checkpointing: bool = False
    backbone_append_zero_attn: bool = False
    backbone_num_context_layers: int = 0
    backbone_use_mask_prior: bool = False
    backbone_use_context_mask: bool = False
    backbone_mask_fusion_type: str = "additive"
    backbone_image_size: int = 224

    # Mask channels
    num_mask_channels: int = 1


class PatchICL(nn.Module):
    """Multi-level cascaded PatchICL model.

    Processes levels from coarse to fine. Each level's predictions guide
    patch sampling at finer levels via additive fusion.
    """

    def __init__(
        self,
        config: dict | PatchICLConfig,
        context_size: int = 0,
        feature_extractor: nn.Module = None,
    ):
        super().__init__()
        self.context_size = context_size
        self.feature_extractor = feature_extractor

        # Parse config
        if isinstance(config, dict):
            cfg = self._parse_config(config)
        else:
            cfg = config
        self._config = cfg

        # Level configs
        self.levels = [LevelConfig(**lc) if isinstance(lc, dict) else lc for lc in cfg.levels]
        self.num_levels = len(self.levels)

        # Verify all levels share patch_size
        patch_sizes = set(lc.patch_size for lc in self.levels)
        assert len(patch_sizes) == 1, f"All levels must share patch_size, got {patch_sizes}"

        # Oracle config
        self.oracle_train = list(cfg.oracle_levels_train)
        self.oracle_valid = list(cfg.oracle_levels_valid)
        while len(self.oracle_train) < self.num_levels:
            self.oracle_train.append(False)
        while len(self.oracle_valid) < self.num_levels:
            self.oracle_valid.append(False)

        # Oracle scheduling
        self.oracle_sched_enabled = cfg.oracle_scheduling_enabled
        self.oracle_sched_warmup = cfg.oracle_scheduling_warmup
        self.oracle_sched_decay = cfg.oracle_scheduling_decay
        self.oracle_sched_start = cfg.oracle_scheduling_start
        self.oracle_sched_end = cfg.oracle_scheduling_end
        self._current_epoch = 0

        # Cascade config
        self.detach_between_levels = cfg.detach_between_levels
        self.cascade_registers = cfg.cascade_registers
        self.skip_context_decoding = cfg.skip_context_decoding

        # Sampling config
        self.target_oracle_mode = cfg.target_sampling_oracle
        self.target_model_mode = cfg.target_model_sampling
        self.context_sampling_mode = cfg.context_sampling
        self.differentiable_sampling = cfg.differentiable_sampling

        # Sampling map
        self.sampling_map_source = cfg.sampling_map_source
        init_temp = cfg.sampling_map_temperature
        if cfg.sampling_map_learn_temperature:
            self._log_temperature = nn.Parameter(torch.tensor(math.log(init_temp)))
        else:
            self.register_buffer('_log_temperature', torch.tensor(math.log(init_temp)))

        # Mask channels
        self.num_mask_channels = cfg.num_mask_channels

        # Loss weights
        self.loss_weights = {
            'target_patch': cfg.loss_target_patch,
            'target_aggreg': cfg.loss_target_aggreg,
            'target_combined': cfg.loss_target_combined,
            'context_patch': cfg.loss_context_patch,
            'context_aggreg': cfg.loss_context_aggreg,
        }
        self.patch_criterion = None
        self.aggreg_criterion = None

        # Per-level loss weights
        if cfg.level_weights == 'uniform':
            self.level_weights = [1.0] * self.num_levels
        elif cfg.level_weights == 'progressive':
            if self.num_levels == 1:
                self.level_weights = [1.0]
            else:
                self.level_weights = [
                    0.3 + 0.7 * i / (self.num_levels - 1)
                    for i in range(self.num_levels)
                ]
        else:
            self.level_weights = list(cfg.level_weights)
        while len(self.level_weights) < self.num_levels:
            self.level_weights.append(1.0)

        # Feature extractor config
        fe_cfg = config.get('feature_extractor', {}) if isinstance(config, dict) else {}
        self.feature_image_size = fe_cfg.get('output_grid_size', None)

        # Augmenter
        if cfg.augmentation_enabled:
            self.augmenter = PatchAugmenter(
                rotation=cfg.augmentation_rotation,
                rotation_range=cfg.augmentation_rotation_range,
                flip_horizontal=cfg.augmentation_flip_horizontal,
                flip_vertical=cfg.augmentation_flip_vertical,
            )
        else:
            self.augmenter = None

        # Per-level samplers
        self.samplers = nn.ModuleList()
        self.context_samplers = nn.ModuleList()
        for lc in self.levels:
            target_spread = lc.spread_sigma_target if lc.spread_sigma_target is not None else lc.spread_sigma
            self.samplers.append(create_sampler(
                sampler_type=cfg.sampler_type,
                patch_size=lc.patch_size,
                num_patches=lc.num_patches,
                num_patches_val=lc.num_patches_val,
                temperature=lc.sampling_temperature,
                stride=lc.stride,
                augmenter=self.augmenter,
                pad_before=lc.pad_before,
                pad_after=lc.pad_after,
                spread_sigma=target_spread,
                differentiable=self.differentiable_sampling,
            ))

            context_spread = lc.spread_sigma_context if lc.spread_sigma_context is not None else lc.spread_sigma
            num_context = lc.num_context_patches if lc.num_context_patches is not None else lc.num_patches
            num_context_val = lc.num_context_patches_val if lc.num_context_patches_val is not None else num_context
            context_stride = lc.context_stride if lc.context_stride is not None else lc.stride
            self.context_samplers.append(create_sampler(
                sampler_type=cfg.sampler_type,
                patch_size=lc.patch_size,
                num_patches=num_context,
                num_patches_val=num_context_val,
                temperature=lc.sampling_temperature,
                stride=context_stride,
                augmenter=self.augmenter,
                pad_before=lc.pad_before,
                pad_after=lc.pad_after,
                spread_sigma=context_spread,
                differentiable=False,  # Context doesn't need differentiable
            ))

        # Per-level aggregators
        aggregator_cfg = config.get('aggregator', {}) if isinstance(config, dict) else {}
        self.aggregators = nn.ModuleList()
        for lc in self.levels:
            self.aggregators.append(create_aggregator(
                aggregator_type=cfg.aggregator_type,
                patch_size=lc.patch_size,
                **aggregator_cfg,
            ))

        # Backbone
        self.feature_grid_size = cfg.backbone_feature_grid_size
        self.patch_feature_grid_size = cfg.backbone_patch_feature_grid_size

        self.backbone = SimpleBackbone(
            embed_dim=cfg.backbone_embed_proj_dim,
            num_heads=cfg.backbone_num_heads,
            num_layers=cfg.backbone_num_layers,
            num_registers=cfg.backbone_num_registers,
            num_classes=self.num_mask_channels,
            patch_size=self.levels[0].patch_size,
            image_size=cfg.backbone_image_size,
            input_dim=cfg.backbone_embed_dim,
            feature_grid_size=self.patch_feature_grid_size,
            dropout=cfg.backbone_dropout,
            decoder_use_skip_connections=cfg.backbone_decoder_use_skip_connections,
            gradient_checkpointing=cfg.backbone_gradient_checkpointing,
            append_zero_attn=cfg.backbone_append_zero_attn,
            use_mask_prior=cfg.backbone_use_mask_prior,
            mask_fusion_type=cfg.backbone_mask_fusion_type,
            predict_sampling_map=(self.sampling_map_source == 'learned'),
            num_context_layers=cfg.backbone_num_context_layers,
            use_context_mask=cfg.backbone_use_context_mask,
        )

    def _parse_config(self, config: dict) -> PatchICLConfig:
        """Parse dict config to PatchICLConfig."""
        # Extract nested configs
        levels_cfg = config.get('levels', [{'resolution': 32, 'patch_size': 16, 'num_patches': 16}])
        oracle_sched = config.get('oracle_scheduling', {})
        sampler_cfg = config.get('sampler', {})
        aug_cfg = sampler_cfg.get('augmentation', {})
        sampling_map_cfg = config.get('sampling_map', {})
        cascade_cfg = config.get('cascade', {})
        backbone_cfg = config.get('backbone', {})

        return PatchICLConfig(
            levels=levels_cfg,
            oracle_levels_train=config.get('oracle_levels_train', [True]),
            oracle_levels_valid=config.get('oracle_levels_valid', [False]),
            oracle_scheduling_enabled=oracle_sched.get('enabled', False),
            oracle_scheduling_warmup=oracle_sched.get('warmup_epochs', 10),
            oracle_scheduling_decay=oracle_sched.get('decay_epochs', 50),
            oracle_scheduling_start=oracle_sched.get('start_prob', 1.0),
            oracle_scheduling_end=oracle_sched.get('end_prob', 0.3),
            detach_between_levels=cascade_cfg.get('detach_between_levels', True),
            cascade_registers=cascade_cfg.get('cascade_registers', False),
            skip_context_decoding=cascade_cfg.get('skip_context_decoding', False),
            sampler_type=sampler_cfg.get('type', 'continuous'),
            target_sampling_oracle=sampler_cfg.get('target_sampling_oracle', 'gt_foreground'),
            target_model_sampling=sampler_cfg.get('target_model_sampling', 'predicted_entropy'),
            context_sampling=sampler_cfg.get('context_sampling', 'gt_foreground'),
            differentiable_sampling=sampler_cfg.get('differentiable', False),
            sampling_map_source=sampling_map_cfg.get('source', 'none'),
            sampling_map_temperature=sampling_map_cfg.get('temperature', 1.0),
            sampling_map_learn_temperature=sampling_map_cfg.get('learn_temperature', False),
            aggregator_type=config.get('aggregator', {}).get('type', 'average'),
            loss_target_patch=config.get('loss', {}).get('weights', {}).get('target_patch', 0.0),
            loss_target_aggreg=config.get('loss', {}).get('weights', {}).get('target_aggreg', 0.0),
            loss_target_combined=config.get('loss', {}).get('weights', {}).get('target_combined', 1.0),
            loss_context_patch=config.get('loss', {}).get('weights', {}).get('context_patch', 0.0),
            loss_context_aggreg=config.get('loss', {}).get('weights', {}).get('context_aggreg', 1.0),
            level_weights=config.get('loss', {}).get('level_weights', 'uniform'),
            augmentation_enabled=aug_cfg.get('enabled', False),
            augmentation_rotation=aug_cfg.get('rotation', 'none'),
            augmentation_rotation_range=aug_cfg.get('rotation_range', 0.5),
            augmentation_flip_horizontal=aug_cfg.get('flip_horizontal', False),
            augmentation_flip_vertical=aug_cfg.get('flip_vertical', False),
            backbone_embed_dim=backbone_cfg.get('embed_dim', 1024),
            backbone_embed_proj_dim=backbone_cfg.get('embed_proj_dim', 128),
            backbone_num_heads=backbone_cfg.get('num_heads', 8),
            backbone_num_layers=backbone_cfg.get('num_layers', 1),
            backbone_num_registers=backbone_cfg.get('num_registers', 4),
            backbone_feature_grid_size=backbone_cfg.get('feature_grid_size', 16),
            backbone_patch_feature_grid_size=backbone_cfg.get('patch_feature_grid_size', 16),
            backbone_dropout=backbone_cfg.get('dropout', 0.0),
            backbone_decoder_use_skip_connections=backbone_cfg.get('decoder_use_skip_connections', True),
            backbone_gradient_checkpointing=backbone_cfg.get('gradient_checkpointing', False),
            backbone_append_zero_attn=backbone_cfg.get('append_zero_attn', False),
            backbone_num_context_layers=backbone_cfg.get('num_context_layers', 0),
            backbone_use_mask_prior=backbone_cfg.get('use_mask_prior', False),
            backbone_use_context_mask=backbone_cfg.get('use_context_mask', False),
            backbone_mask_fusion_type=backbone_cfg.get('mask_fusion_type', 'additive'),
            backbone_image_size=backbone_cfg.get('image_size', 224),
            num_mask_channels=config.get('num_mask_channels', 1),
        )

    @property
    def sampling_map_temperature(self) -> torch.Tensor:
        """Get current temperature value."""
        return self._log_temperature.exp()

    def set_loss_functions(self, patch_criterion: nn.Module, aggreg_criterion: nn.Module):
        """Set the loss functions."""
        self.patch_criterion = patch_criterion
        self.aggreg_criterion = aggreg_criterion

    def set_feature_extractor(self, feature_extractor: nn.Module):
        """Set or update the feature extractor."""
        self.feature_extractor = feature_extractor

    def set_epoch(self, epoch: int):
        """Set current epoch for oracle scheduling."""
        self._current_epoch = epoch

    def _get_oracle_probability(self, level_idx: int) -> float:
        """Compute oracle sampling probability with linear decay."""
        if not self.oracle_train[level_idx]:
            return 0.0
        if not self.oracle_sched_enabled:
            return 1.0

        if self._current_epoch < self.oracle_sched_warmup:
            return self.oracle_sched_start

        decay_progress = min(
            1.0,
            (self._current_epoch - self.oracle_sched_warmup) / max(1, self.oracle_sched_decay)
        )
        return self.oracle_sched_start + decay_progress * (self.oracle_sched_end - self.oracle_sched_start)

    def _extract_features(
        self,
        target_images: torch.Tensor,
        context_images: torch.Tensor | None = None,
        context_masks: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Extract features on-the-fly using the feature extractor."""
        if self.feature_extractor is None:
            raise RuntimeError("No feature extractor available.")

        input_size = target_images.shape[-1]
        if self.feature_image_size is not None and self.feature_image_size != input_size:
            fe_size = self.feature_image_size
            target_images = F.interpolate(
                target_images, size=(fe_size, fe_size),
                mode='bilinear', align_corners=False
            )
            if context_images is not None:
                B, k = context_images.shape[:2]
                ctx_flat = context_images.view(B * k, *context_images.shape[2:])
                ctx_flat = F.interpolate(
                    ctx_flat, size=(fe_size, fe_size),
                    mode='bilinear', align_corners=False
                )
                context_images = ctx_flat.view(B, k, *ctx_flat.shape[1:])

        if getattr(self.feature_extractor, '_frozen', False):
            with torch.no_grad():
                return self.feature_extractor.extract_batch(target_images, context_images, context_masks)
        return self.feature_extractor.extract_batch(target_images, context_images, context_masks)

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

        # Extract features if needed
        if target_features is None and self.feature_extractor is not None:
            target_features, context_features = self._extract_features(
                image, context_in, context_out
            )

        # Process each level
        combined_pred = None
        combined_sampling_map = None
        prev_register_tokens = None
        level_outputs = []

        for i in range(self.num_levels):
            level_cfg = self.levels[i]
            resolution = level_cfg.resolution
            patch_size = level_cfg.patch_size

            # Determine sampling weights
            if training:
                oracle_prob = self._get_oracle_probability(i)
            else:
                oracle_prob = 1.0 if self.oracle_valid[i] else 0.0

            # Downsample labels
            if labels is not None:
                labels_ds = downsample_mask(labels, resolution)
            else:
                labels_ds = torch.zeros(
                    B, self.num_mask_channels, resolution, resolution, device=device
                )

            # Compute oracle weights
            oracle_weights = None
            if labels is not None:
                prev_pred_for_oracle = None
                if combined_pred is not None:
                    prev_pred_for_oracle = F.interpolate(
                        combined_pred.detach(), size=(resolution, resolution),
                        mode='bilinear', align_corners=False
                    )
                oracle_weights = compute_sampling_weights(
                    mode=self.target_oracle_mode,
                    gt_mask=mask_to_weights(labels_ds),
                    prev_pred=prev_pred_for_oracle,
                )

            # Compute model weights
            model_weights = None
            if combined_pred is not None:
                pred_detached = combined_pred.detach() if self.detach_between_levels else combined_pred
                map_detached = (
                    combined_sampling_map.detach()
                    if (combined_sampling_map is not None and self.detach_between_levels)
                    else combined_sampling_map
                )
                model_weights_fullres = compute_sampling_weights(
                    mode=self.target_model_mode,
                    pred_logits=pred_detached,
                    sampling_map=map_detached,
                )
                model_weights = F.interpolate(
                    model_weights_fullres, size=(resolution, resolution),
                    mode='bilinear', align_corners=False
                )

            # Mix oracle and model weights
            if oracle_prob >= 1.0 and oracle_weights is not None:
                sampling_weights = oracle_weights
            elif oracle_prob <= 0.0 or oracle_weights is None:
                if model_weights is not None:
                    sampling_weights = model_weights
                else:
                    sampling_weights = torch.ones(B, 1, resolution, resolution, device=device)
            else:
                if model_weights is None:
                    model_weights = torch.ones(B, 1, resolution, resolution, device=device)
                use_oracle_mask = torch.rand(B, 1, 1, 1, device=device) < oracle_prob
                sampling_weights = torch.where(use_oracle_mask, oracle_weights, model_weights)

            # Mask prior
            mask_prior = None
            if i > 0 and combined_pred is not None:
                mask_prior = combined_pred.detach() if self.detach_between_levels else combined_pred

            # Process level
            level_out = process_level(
                level_idx=i,
                level_cfg=level_cfg,
                image=image,
                labels=labels,
                labels_ds=labels_ds,
                context_in=context_in,
                context_out=context_out,
                target_features=target_features,
                context_features=context_features,
                sampling_weights=sampling_weights,
                sampler=self.samplers[i],
                context_sampler=self.context_samplers[i],
                aggregator=self.aggregators[i],
                backbone=self.backbone,
                augmenter=self.augmenter,
                feature_grid_size=self.feature_grid_size,
                patch_feature_grid_size=self.patch_feature_grid_size,
                context_sampling_mode=self.context_sampling_mode,
                sampling_map_source=self.sampling_map_source,
                sampling_map_temperature=self.sampling_map_temperature,
                differentiable_sampling=self.differentiable_sampling,
                mask_prior=mask_prior,
                prev_register_tokens=prev_register_tokens if self.cascade_registers else None,
                return_attn_weights=return_attn_weights,
                skip_context_decoding=self.skip_context_decoding,
            )

            pred = level_out.pred
            coverage = level_out.coverage_mask
            register_tokens = level_out.register_tokens
            aggregated_sampling_map = level_out.aggregated_sampling_map

            # Update cascade registers
            if self.cascade_registers and register_tokens is not None:
                prev_register_tokens = (
                    register_tokens.detach() if self.detach_between_levels else register_tokens
                )

            # Combine predictions (additive fusion only)
            if combined_pred is not None:
                prev_pred = combined_pred.detach() if self.detach_between_levels else combined_pred
                combined_upsampled = F.interpolate(
                    prev_pred, size=(resolution, resolution),
                    mode='bilinear', align_corners=False
                )

                # Additive fusion
                combined_pred = combined_upsampled + coverage * pred

                # Update sampling map
                if aggregated_sampling_map is not None:
                    if combined_sampling_map is not None:
                        prev_map = (
                            combined_sampling_map.detach()
                            if self.detach_between_levels else combined_sampling_map
                        )
                        map_upsampled = F.interpolate(
                            prev_map, size=(resolution, resolution),
                            mode='bilinear', align_corners=False
                        )
                        combined_sampling_map = coverage * aggregated_sampling_map + (1 - coverage) * map_upsampled
                    else:
                        combined_sampling_map = aggregated_sampling_map
            else:
                combined_pred = pred
                combined_sampling_map = aggregated_sampling_map

            # Store level output as dict
            level_out_dict = {
                'pred': pred,
                'patch_labels': level_out.patch_labels,
                'patch_logits': level_out.patch_logits,
                'coords': level_out.coords,
                'context_patch_labels': level_out.context_patch_labels,
                'context_patch_logits': level_out.context_patch_logits,
                'context_coords': level_out.context_coords,
                'context_pred': level_out.context_pred,
                'context_labels': level_out.context_labels,
                'context_labels_fullres': level_out.context_labels_fullres,
                'target_validity': level_out.target_validity,
                'context_validity': level_out.context_validity,
                'attn_weights': level_out.attn_weights,
                'register_tokens': level_out.register_tokens,
                'patch_sampling_map': level_out.patch_sampling_map,
                'aggregated_sampling_map': level_out.aggregated_sampling_map,
                'selection_probs': level_out.selection_probs,
                'coverage_mask': coverage,
                'combined_pred': combined_pred,
                'combined_sampling_map': combined_sampling_map,
                'patch_size': patch_size,
                'level_res': resolution,
            }
            level_outputs.append(level_out_dict)

        # Final prediction
        final_pred = F.interpolate(
            combined_pred, size=(H, W), mode='bilinear', align_corners=False
        )
        coarse_pred = F.interpolate(
            level_outputs[0]['pred'], size=(H, W), mode='bilinear', align_corners=False
        )

        final_sampling_map = None
        if combined_sampling_map is not None:
            final_sampling_map = F.interpolate(
                combined_sampling_map, size=(H, W), mode='bilinear', align_corners=False
            )

        last = level_outputs[-1]
        return {
            'final_pred': final_pred,
            'final_logit': final_pred,
            'final_sampling_map': final_sampling_map,
            'coarse_pred': coarse_pred,
            'level_outputs': level_outputs,
            'patch_labels': last['patch_labels'],
            'patch_logits': last['patch_logits'],
            'patch_coords': last['coords'],
            'context_patch_labels': last['context_patch_labels'],
            'context_patch_logits': last['context_patch_logits'],
            'context_coords': last['context_coords'],
            'attn_weights': last.get('attn_weights'),
            'register_tokens': last.get('register_tokens'),
        }

    def compute_loss(
        self,
        outputs: dict[str, torch.Tensor],
        labels: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute multi-level losses."""
        if self.patch_criterion is None or self.aggreg_criterion is None:
            raise RuntimeError("Call set_loss_functions() first.")

        if labels.dim() == 3:
            labels = labels.unsqueeze(1)

        B = labels.shape[0]
        losses = {}
        total_loss = 0.0

        for i, level_out in enumerate(outputs['level_outputs']):
            lw = self.level_weights[i]
            patch_logits = level_out['patch_logits']
            patch_labels = level_out['patch_labels']
            target_validity = level_out.get('target_validity')
            K = patch_logits.shape[1]

            # Target patch loss
            if self.loss_weights['target_patch'] > 0:
                target_patch_loss = self._masked_patch_loss(
                    patch_logits.reshape(B * K, -1),
                    patch_labels.reshape(B * K, -1),
                    target_validity.reshape(B * K, -1) if target_validity is not None else None,
                )
                total_loss = total_loss + lw * self.loss_weights['target_patch'] * target_patch_loss

            # Target aggreg loss
            pred = level_out['pred']
            level_res = pred.shape[-2:]
            labels_ds = resize_label(labels.float(), size=level_res)

            if self.loss_weights['target_aggreg'] > 0:
                target_aggreg_loss = self.aggreg_criterion(pred, labels_ds)
                total_loss = total_loss + lw * self.loss_weights['target_aggreg'] * target_aggreg_loss

            # Combined prediction loss
            combined_pred = level_out.get('combined_pred')
            if i > 0 and combined_pred is not None and self.loss_weights['target_combined'] > 0:
                combined_res = combined_pred.shape[-2:]
                labels_combined = (
                    labels_ds if combined_res == level_res
                    else resize_label(labels.float(), size=combined_res)
                )
                target_combined_loss = self.aggreg_criterion(combined_pred, labels_combined)
                total_loss = total_loss + lw * self.loss_weights['target_combined'] * target_combined_loss

            # Context losses
            ctx_patch_logits = level_out.get('context_patch_logits')
            ctx_patch_labels = level_out.get('context_patch_labels')
            ctx_validity = level_out.get('context_validity')

            if ctx_patch_logits is not None and ctx_patch_labels is not None:
                if self.loss_weights['context_patch'] > 0:
                    K_ctx = ctx_patch_logits.shape[1]
                    context_patch_loss = self._masked_patch_loss(
                        ctx_patch_logits.reshape(B * K_ctx, -1),
                        ctx_patch_labels.reshape(B * K_ctx, -1),
                        ctx_validity.reshape(B * K_ctx, -1) if ctx_validity is not None else None,
                    )
                    total_loss = total_loss + lw * self.loss_weights['context_patch'] * context_patch_loss

            context_pred = level_out.get('context_pred')
            context_labels_fullres = level_out.get('context_labels_fullres')

            if context_pred is not None and context_labels_fullres is not None:
                if self.loss_weights['context_aggreg'] > 0:
                    B_ctx, k_ctx = context_pred.shape[:2]
                    ctx_flat = context_labels_fullres.view(B_ctx * k_ctx, *context_labels_fullres.shape[2:])
                    context_labels_ds = resize_label(ctx_flat.float(), size=context_pred.shape[-2:])
                    context_aggreg_loss = self.aggreg_criterion(
                        context_pred.reshape(B_ctx * k_ctx, -1),
                        context_labels_ds.reshape(B_ctx * k_ctx, -1),
                    )
                    total_loss = total_loss + lw * self.loss_weights['context_aggreg'] * context_aggreg_loss

        losses['total_loss'] = total_loss

        # Log oracle scheduling
        if self.oracle_sched_enabled:
            for i in range(self.num_levels):
                if self.oracle_train[i]:
                    losses[f'level_{i}_oracle_prob'] = self._get_oracle_probability(i)

        # Log temperature
        if self.sampling_map_source == 'entropy':
            losses['sampling_map_temperature'] = self.sampling_map_temperature.detach()

        return losses

    def _masked_patch_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        validity: torch.Tensor | None,
    ) -> torch.Tensor:
        """Compute patch loss, masking out invalid pixels."""
        assert self.patch_criterion is not None
        if validity is None:
            return self.patch_criterion(logits, labels)
        invalid = ~validity.expand_as(logits).bool()
        masked_logits = torch.where(invalid, torch.tensor(-100.0, device=logits.device), logits)
        masked_labels = torch.where(invalid, torch.zeros_like(labels), labels)
        return self.patch_criterion(masked_logits, masked_labels)
