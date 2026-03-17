"""CNN decoder for PatchICL backbone.

U-Net style decoder with skip connections, dynamically built for different feature sizes.
Pattern: upsample from 1x1 -> 2 -> 4 -> 8 -> (16 if needed)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import ResolutionConditionedNorm


class CNNDecoder(nn.Module):
    """U-Net style decoder with skip connections.

    Takes encoded features and skip connections from encoder,
    progressively upsamples with skip fusion to output resolution.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_classes: int = 1,
        patch_size: int = 16,
        feature_grid_size: int = 16,
        use_skip_connections: bool = True,
        predict_sampling_map: bool = False,
        detach_sampling_features: bool = False,
        scale_embed_dim: int | None = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.feature_grid_size = feature_grid_size
        self.use_skip_connections = use_skip_connections
        self.predict_sampling_map = predict_sampling_map
        self.detach_sampling_features = detach_sampling_features
        self.scale_embed_dim = scale_embed_dim or embed_dim

        D = embed_dim
        fuse_ch_in = D * 2 if use_skip_connections else D

        # Build decoder levels: 1x1 -> 2 -> 4 -> 8 -> (16)
        # We upsample by 2x at each level
        self._level_target_sizes = [2, 4, 8]
        if feature_grid_size == 16:
            self._level_target_sizes.append(16)

        self.levels = nn.ModuleList()
        for _ in self._level_target_sizes:
            self.levels.append(nn.ModuleDict({
                'up': nn.ConvTranspose2d(D, D, 2, stride=2),
                'fuse_conv': nn.Conv2d(fuse_ch_in, D, 3, padding=1),
                'fuse_norm': ResolutionConditionedNorm(D, self.scale_embed_dim),
            }))

        # Segmentation head
        self.seg_head = nn.Sequential(
            nn.Conv2d(D, D // 2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(D // 2, num_classes, 1),
        )

        # Sampling map head
        if predict_sampling_map:
            self.sampling_head = nn.Sequential(
                nn.Conv2d(D, D // 2, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(D // 2, 1, 1),
            )

    def forward(
        self,
        encoded: torch.Tensor,
        skips: dict[str, torch.Tensor],
        scale_embed: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Args:
            encoded: [B, K, D] - attention output
            skips: dict with skip tensors from encoder
            scale_embed: [scale_embed_dim] - resolution embedding

        Returns:
            seg_pred: [B, K, num_classes, patch_size, patch_size]
            sampling_map: [B, K, 1, patch_size, patch_size] or None
        """
        B, K, D = encoded.shape
        h = self.feature_grid_size

        # Default scale_embed
        if scale_embed is None:
            scale_embed = torch.zeros(self.scale_embed_dim, device=encoded.device)

        # Reshape encoded to [B*K, D, 1, 1]
        x = encoded.contiguous().view(B * K, D, 1, 1)

        # Make skips contiguous
        skips = {k: v.contiguous() for k, v in skips.items()}

        # Decode through levels
        for i, level in enumerate(self.levels):
            target_size = self._level_target_sizes[i]

            # Upsample
            x = level['up'](x)

            # Fuse with skip connection
            if self.use_skip_connections:
                skip_key = f'skip_{target_size}x{target_size}'
                if skip_key in skips:
                    skip = skips[skip_key].view(B * K, D, target_size, target_size)
                    x = torch.cat([x, skip], dim=1)

            # Conv + norm + activation
            x = level['fuse_conv'](x)
            x = F.gelu(level['fuse_norm'](x, scale_embed))

        # x is now [B*K, D, feature_grid_size, feature_grid_size]

        # Segmentation head
        seg_pred = self.seg_head(x)

        # Upsample to patch_size if needed
        if h != self.patch_size:
            seg_pred = F.interpolate(
                seg_pred, size=(self.patch_size, self.patch_size),
                mode='bilinear', align_corners=False
            )

        seg_pred = seg_pred.view(
            B, K, self.num_classes, self.patch_size, self.patch_size
        )

        # Sampling map head
        sampling_map = None
        if self.predict_sampling_map:
            map_features = x.detach() if self.detach_sampling_features else x
            map_logits = self.sampling_head(map_features)
            sampling_map = torch.sigmoid(map_logits)

            if h != self.patch_size:
                sampling_map = F.interpolate(
                    sampling_map, size=(self.patch_size, self.patch_size),
                    mode='bilinear', align_corners=False
                )

            sampling_map = sampling_map.view(B, K, 1, self.patch_size, self.patch_size)

        return seg_pred, sampling_map
