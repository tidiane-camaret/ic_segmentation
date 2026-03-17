"""CNN encoder for PatchICL backbone.

Loop-based implementation supporting both 8x8 and 16x16 feature grids.
Follows v2 pattern: first level preserves resolution, subsequent levels halve.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import ResolutionConditionedNorm


class CNNEncoder(nn.Module):
    """CNN encoder that processes feature patches through spatial convolutions.

    Takes [B, K, tokens, D] features and produces:
    - encoded: [B, K, embed_dim] - pooled representation for attention
    - skips: dict of intermediate features at each scale

    Pattern: first level preserves resolution, subsequent levels halve.
    For 8x8: 8->8->4->2
    For 16x16: 16->16->8->4->2
    """

    def __init__(
        self,
        input_dim: int = 1024,
        embed_dim: int = 128,
        start_size: int = 16,
        end_size: int = 2,
        scale_embed_dim: int | None = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.start_size = start_size
        self.end_size = end_size
        self.scale_embed_dim = scale_embed_dim or embed_dim

        # Project input features to working dimension
        self.input_proj = nn.Linear(input_dim, embed_dim)

        # Build encoder levels dynamically
        # First level: preserve resolution (stride=1)
        # Subsequent levels: halve resolution (stride=2) until end_size
        self.levels = nn.ModuleList()
        self._level_output_sizes = []

        size = start_size
        first_level = True
        while size >= end_size:
            if first_level:
                # First level preserves resolution
                stride = 1
                output_size = size
                first_level = False
            else:
                # Subsequent levels halve resolution
                stride = 2
                output_size = size // 2

            self.levels.append(nn.ModuleDict({
                'conv': nn.Conv2d(embed_dim, embed_dim, 3, stride=stride, padding=1),
                'norm': ResolutionConditionedNorm(embed_dim, self.scale_embed_dim),
            }))
            self._level_output_sizes.append(output_size)
            size = output_size

            if size == end_size:
                break

        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(
        self,
        features: torch.Tensor,
        scale_embed: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Args:
            features: [B, K, tokens, D_in] - patch features
            scale_embed: [scale_embed_dim] - resolution embedding

        Returns:
            encoded: [B, K, embed_dim] - pooled representation
            skips: dict with skip tensors keyed by size
        """
        B, K, NF, E = features.shape
        D = self.embed_dim
        h = w = self.start_size

        # Default scale_embed for backward compatibility
        if scale_embed is None:
            scale_embed = torch.zeros(self.scale_embed_dim, device=features.device)

        # Project and reshape to spatial: [B*K, D, h, w]
        x = self.input_proj(features.reshape(-1, E))
        x = x.view(B * K, NF, D).permute(0, 2, 1)
        x = x.view(B * K, D, h, w)

        # Encode with skip connections
        skips = {}
        for i, level in enumerate(self.levels):
            x = level['conv'](x)
            x = F.gelu(level['norm'](x, scale_embed))
            # Store skip at this level's output size
            output_size = self._level_output_sizes[i]
            skip_key = f'skip_{output_size}x{output_size}'
            skips[skip_key] = x.view(B, K, D, output_size, output_size)

        # Pool to [B, K, D]
        encoded = self.pool(x).view(B, K, D)

        return encoded, skips
