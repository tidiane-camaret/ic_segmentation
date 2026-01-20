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
    ):
        """
        Args:
            resolution: Target resolution for this level (image will be resized to resolution x resolution)
            patch_size: Size of sampled patches (patch_size x patch_size)
            num_patches: Number of patches to sample
            backbone: Shared or level-specific backbone (e.g., LocalDino)
            level_idx: Index of this level (0 = coarsest)
            sampling_temperature: Temperature for patch sampling (lower = sharper distribution)
        """
        super().__init__()
        self.resolution = resolution
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.backbone = backbone
        self.level_idx = level_idx
        self.sampling_temperature = sampling_temperature

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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Select K patches based on weight map.

        Args:
            image: [B, C, H, W] - image at this level's resolution
            labels: [B, 1, H, W] - GT mask at this level's resolution
            weights: [B, 1, H, W] - weight map for sampling (e.g., prev_pred)

        Returns:
            patches: [B, K, C, ps, ps]
            patch_labels: [B, K, 1, ps, ps]
            coords: [B, K, 2] - (h, w) coordinates
        """
        B, C, H, W = image.shape
        ps = self.patch_size
        K = self.num_patches

        # Compute stride for dense patch scoring
        stride = max(1, ps // 4)

        # Compute patch grid dimensions
        grid_h = max(1, (H - ps) // stride + 1)
        grid_w = max(1, (W - ps) // stride + 1)

        # Compute scores using avg pooling on weights
        # Map patch locations to weight map
        scores_map = F.avg_pool2d(weights, kernel_size=ps, stride=stride, padding=0)

        # Resize to grid dimensions if needed
        if scores_map.shape[-2:] != (grid_h, grid_w):
            scores_map = F.interpolate(scores_map, size=(grid_h, grid_w), mode='bilinear', align_corners=False)

        # Flatten scores: [B, N_candidates]
        flat_scores = scores_map.flatten(2).squeeze(1)

        # Normalize and apply temperature
        score_min = flat_scores.min(dim=1, keepdim=True)[0]
        score_max = flat_scores.max(dim=1, keepdim=True)[0]
        score_range = (score_max - score_min).clamp(min=1e-6)
        normalized_scores = (flat_scores - score_min) / score_range
        scaled_scores = normalized_scores / self.sampling_temperature

        # Add noise during training for exploration
        if self.training:
            noise = torch.rand_like(scaled_scores) * 0.5
            scaled_scores = scaled_scores + noise

        # Sample patches using multinomial
        num_candidates = flat_scores.shape[1]
        k_safe = min(K, num_candidates)
        probs = F.softmax(scaled_scores, dim=1)
        top_indices = torch.multinomial(probs, k_safe, replacement=False)

        # Convert indices to coordinates
        row_indices = top_indices // grid_w
        col_indices = top_indices % grid_w
        h_starts = row_indices * stride
        w_starts = col_indices * stride

        # Clamp to valid range
        h_starts = h_starts.clamp(0, H - ps)
        w_starts = w_starts.clamp(0, W - ps)

        # Extract patches
        all_patches = []
        all_labels = []
        all_coords = []

        for b in range(B):
            batch_patches = []
            batch_labels = []
            batch_coords = []

            for k in range(k_safe):
                h, w = h_starts[b, k].item(), w_starts[b, k].item()
                patch = image[b, :, h:h+ps, w:w+ps]
                label_patch = labels[b, :, h:h+ps, w:w+ps]

                batch_patches.append(patch)
                batch_labels.append(label_patch)
                batch_coords.append([h, w])

            # Pad if fewer candidates
            while len(batch_patches) < K:
                batch_patches.append(torch.zeros(C, ps, ps, device=image.device))
                batch_labels.append(torch.zeros(1, ps, ps, device=image.device))
                batch_coords.append([0, 0])

            all_patches.append(torch.stack(batch_patches))
            all_labels.append(torch.stack(batch_labels))
            all_coords.append(torch.tensor(batch_coords, device=image.device))

        return (
            torch.stack(all_patches),  # [B, K, C, ps, ps]
            torch.stack(all_labels),   # [B, K, 1, ps, ps]
            torch.stack(all_coords),   # [B, K, 2]
        )

    def aggregate(
        self,
        patch_logits: torch.Tensor,
        coords: torch.Tensor,
        output_size: tuple[int, int],
        prev_pred: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Aggregate patch predictions back to a full mask.

        Args:
            patch_logits: [B, K, 1, ps, ps] - predictions for each patch
            coords: [B, K, 2] - patch coordinates
            output_size: (H, W) - output mask size
            prev_pred: [B, 1, H, W] - previous level prediction (optional, for combining)

        Returns:
            aggregated: [B, 1, H, W] - aggregated prediction at output_size
        """
        B = patch_logits.shape[0]
        K = patch_logits.shape[1]
        ps = self.patch_size
        H, W = output_size
        device = patch_logits.device

        # Initialize output and count tensors
        output = torch.zeros(B, 1, H, W, device=device)
        counts = torch.zeros(B, 1, H, W, device=device)

        # Place patches back
        for b in range(B):
            for k in range(K):
                h, w = coords[b, k].tolist()
                h, w = int(h), int(w)
                # Clamp to valid range
                h_end = min(h + ps, H)
                w_end = min(w + ps, W)
                patch_h = h_end - h
                patch_w = w_end - w

                output[b, :, h:h_end, w:w_end] += patch_logits[b, k, :, :patch_h, :patch_w]
                counts[b, :, h:h_end, w:w_end] += 1

        # Average overlapping regions
        counts = counts.clamp(min=1)
        aggregated = output / counts

        # Optionally combine with previous prediction
        if prev_pred is not None:
            # Simple average for now (can be learned)
            prev_resized = F.interpolate(prev_pred, size=(H, W), mode='bilinear', align_corners=False)
            aggregated = (aggregated + prev_resized) / 2

        return aggregated

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

                # Select K patches from this context
                patches, labels, coords = self.select_patches(ctx_img, ctx_mask, ctx_weight)
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

    def forward(
        self,
        image: torch.Tensor,
        labels: torch.Tensor = None,
        prev_pred: torch.Tensor = None,
        original_coords_scale: float = 1.0,
        context_in: torch.Tensor = None,
        context_out: torch.Tensor = None,
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

        Returns:
            Dict with: pred, patches, patch_labels, patch_logits, coords, context_*
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
        patches, patch_labels, coords = self.select_patches(image_ds, labels_ds, weights)

        # Scale coordinates for backbone
        coords_for_backbone = coords.float() * original_coords_scale

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

            # Concatenate target and context patches for joint processing
            all_patches = torch.cat([patches, context_patches], dim=1)
            all_coords = torch.cat([coords_for_backbone, context_coords_for_backbone], dim=1)

            # Process all patches through backbone
            all_logits = self.backbone(all_patches, coords=all_coords)

            # Split back: target predictions are first K
            K = patches.shape[1]
            patch_logits = all_logits[:, :K]  # [B, K, 1, ps, ps]
            # context_logits = all_logits[:, K:]  # Not used, but context influences target via attention

        else:
            # No context: process target patches only
            patch_logits = self.backbone(patches, coords=coords_for_backbone)

        # Aggregate to full prediction at this level's resolution
        pred = self.aggregate(
            patch_logits, coords,
            output_size=(self.resolution, self.resolution),
            # prev_pred=self.downsample(prev_pred) if prev_pred is not None else None,
        )

        return {
            'pred': pred,  # [B, 1, res, res]
            'patches': patches,  # [B, K, C, ps, ps]
            'patch_labels': patch_labels,  # [B, K, 1, ps, ps]
            'patch_logits': patch_logits,  # [B, K, 1, ps, ps]
            'coords': coords,  # [B, K, 2]
            'context_patches': context_patches,  # [B, K*k, C, ps, ps] or None
            'context_patch_labels': context_patch_labels,  # [B, K*k, 1, ps, ps] or None
            'context_coords': context_coords,  # [B, K*k, 2] or None
        }


class PatchICL(nn.Module):
    """
    Multi-resolution PatchICL model.

    Chains multiple PatchICL_Level modules, each operating at progressively
    finer resolution. Predictions flow from coarse to fine levels.
    """

    def __init__(self, config: dict, context_size: int = 0):
        """
        Args:
            config: Config dict with 'levels' list and 'backbone' params
                levels: List of dicts with {resolution, patch_size, num_patches}
                backbone: Backbone config (type, pretrained_path, etc.)
            context_size: Number of context examples (0 = no context) - TODO
        """
        super().__init__()
        self.context_size = context_size

        levels_cfg = config.get('levels', [
            {'resolution': 64, 'patch_size': 16, 'num_patches': 16},
            {'resolution': 128, 'patch_size': 16, 'num_patches': 32},
            {'resolution': 224, 'patch_size': 16, 'num_patches': 64},
        ])

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
            level = PatchICL_Level(
                resolution=lcfg['resolution'],
                patch_size=lcfg['patch_size'],
                num_patches=lcfg['num_patches'],
                backbone=backbones[i],
                level_idx=i,
                sampling_temperature=lcfg.get('sampling_temperature', 0.3),
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
        else:
            raise ValueError(f"Unknown backbone type: {backbone_type}")

    def forward(
        self,
        image: torch.Tensor,
        labels: torch.Tensor = None,
        context_in: torch.Tensor = None,
        context_out: torch.Tensor = None,
        mode: str = "train",  # Unused, kept for compatibility
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass through all levels.

        Args:
            image: [B, C, H, W] - input image
            labels: [B, 1, H, W] or [B, H, W] - GT mask
            context_in: [B, k, C, H, W] - context images
            context_out: [B, k, 1, H, W] - context masks
            mode: "train" or "test" (unused, kept for compatibility)

        Returns:
            Dict with predictions and intermediate outputs per level
        """
        del mode  # Unused
        _, _, H, W = image.shape
        self.original_size = (H, W)

        # Ensure labels have channel dim
        if labels is not None and labels.dim() == 3:
            labels = labels.unsqueeze(1)

        # Initialize prev_pred as None for first level
        prev_pred = None

        # Collect outputs from all levels
        level_outputs = []

        for level in self.levels:
            # Compute coordinate scale factor (for position encoding)
            coord_scale = H / level.resolution

            # Forward through level with context
            level_out = level(
                image=image,
                labels=labels,
                prev_pred=prev_pred,
                original_coords_scale=coord_scale,
                context_in=context_in,
                context_out=context_out,
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
            'context_coords': finest_level.get('context_coords'),
        }

    def compute_loss(
        self,
        outputs: dict[str, torch.Tensor],
        labels: torch.Tensor,
        criterion,
        level_weights: list[float] = None,
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

        for i, (level_out, weight) in enumerate(zip(outputs['level_outputs'], level_weights)):
            # Patch-level loss
            patch_logits = level_out['patch_logits']  # [B, K, 1, ps, ps]
            patch_labels = level_out['patch_labels']  # [B, K, 1, ps, ps]
            K = patch_logits.shape[1]

            patch_loss = criterion(
                patch_logits.reshape(B * K, -1),
                patch_labels.reshape(B * K, -1),
            )
            losses[f'level_{i}_patch_loss'] = patch_loss

            # Level prediction loss (vs downsampled GT)
            pred = level_out['pred']  # [B, 1, res, res]
            labels_ds = F.interpolate(
                labels.float(),
                size=pred.shape[-2:],
                mode='nearest',
            )
            level_loss = criterion(pred, labels_ds)
            losses[f'level_{i}_pred_loss'] = level_loss

            # Combined level loss
            level_total = patch_loss + level_loss
            losses[f'level_{i}_loss'] = level_total
            total_loss = total_loss + weight * level_total

        # Final prediction loss
        final_loss = criterion(outputs['final_pred'], labels)
        losses['final_loss'] = final_loss
        total_loss = total_loss + final_loss

        losses['total_loss'] = total_loss

        # Compatibility with GlobalLocalModel / train_utils
        # Map first level pred_loss to global_loss, last level patch_loss to local_loss
        losses['global_loss'] = losses.get('level_0_pred_loss', torch.tensor(0.0))
        losses['local_loss'] = losses.get(f'level_{self.num_levels - 1}_patch_loss', torch.tensor(0.0))
        losses['agg_loss'] = final_loss  # Aggregation loss = final prediction loss

        return losses
