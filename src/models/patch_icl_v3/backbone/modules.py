"""Small utility modules for backbone.

Contains normalization layers, scale encoding, and mask encoding modules.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    """LayerNorm for 2D feature maps (SAM-style)."""

    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C, H, W]"""
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class ContinuousScaleEncoding(nn.Module):
    """Resolution-agnostic scale encoding using sinusoidal functions.

    Encodes continuous resolution values (e.g., 8, 16, 32, 64) into embeddings
    using log-spaced sinusoidal functions, similar to NeRF/Transformer PE.
    """

    def __init__(self, embed_dim: int, learnable_freqs: bool = True):
        super().__init__()
        self.embed_dim = embed_dim
        half_dim = embed_dim // 2
        freqs = torch.exp(
            torch.arange(half_dim).float() * -(math.log(10000.0) / half_dim)
        )
        if learnable_freqs:
            self.freqs = nn.Parameter(freqs)
        else:
            self.register_buffer('freqs', freqs)

    def forward(
        self, resolution: float, device: torch.device = None
    ) -> torch.Tensor:
        """Encode resolution to embedding vector.

        Args:
            resolution: Current level resolution (e.g., 8, 16, 32, 64)
            device: Target device for the output tensor

        Returns:
            [embed_dim] scale embedding
        """
        scale = math.log2(resolution)
        freqs = self.freqs if device is None else self.freqs.to(device)
        args = scale * freqs
        return torch.cat([torch.sin(args), torch.cos(args)])


class ResolutionConditionedNorm(nn.Module):
    """GroupNorm with FiLM conditioning from resolution embedding.

    Implements Feature-wise Linear Modulation:
        out = gamma(resolution) * GroupNorm(x) + beta(resolution)

    This allows normalization to adapt to any resolution.
    """

    def __init__(
        self, num_channels: int, scale_embed_dim: int, num_groups: int = 8
    ):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, num_channels, affine=False)

        self.gamma_proj = nn.Linear(scale_embed_dim, num_channels)
        self.beta_proj = nn.Linear(scale_embed_dim, num_channels)

        # Initialize to identity transform
        nn.init.ones_(self.gamma_proj.bias)
        nn.init.zeros_(self.gamma_proj.weight)
        nn.init.zeros_(self.beta_proj.weight)
        nn.init.zeros_(self.beta_proj.bias)

    def forward(
        self, x: torch.Tensor, scale_embed: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: [B*K, C, H, W] - features to normalize
            scale_embed: [scale_embed_dim] - continuous resolution embedding

        Returns:
            Normalized and modulated features [B*K, C, H, W]
        """
        x = self.norm(x)
        gamma = self.gamma_proj(scale_embed).view(1, -1, 1, 1)
        beta = self.beta_proj(scale_embed).view(1, -1, 1, 1)
        return gamma * x + beta


class MaskPriorEncoder(nn.Module):
    """SAM-style CNN to encode mask patches to embedding vectors.

    Architecture from SAM/MedSAM mask_downscaling:
    - Conv2d(1, 16, 2, stride=2) + LayerNorm2d + GELU -> h/2
    - Conv2d(16, 32, 2, stride=2) + LayerNorm2d + GELU -> h/4
    - Conv2d(32, embed_dim, 1) -> h/4
    - AdaptiveAvgPool2d(1) -> [B*K, embed_dim, 1, 1]
    """

    def __init__(self, embed_dim: int = 128, feature_grid_size: int = 16):
        super().__init__()
        self.embed_dim = embed_dim
        self.mask_downscaling = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=2, stride=2),
            LayerNorm2d(16),
            nn.GELU(),
            nn.Conv2d(16, 32, kernel_size=2, stride=2),
            LayerNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, embed_dim, kernel_size=1),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, mask_patches: torch.Tensor) -> torch.Tensor:
        """Encode mask patches to embeddings.

        Args:
            mask_patches: [B, K, 1, h, h] - mask patch values

        Returns:
            [B, K, embed_dim] - encoded mask embeddings
        """
        B, K = mask_patches.shape[:2]
        x = mask_patches.view(B * K, 1, *mask_patches.shape[3:])
        x = self.mask_downscaling(x)
        x = self.pool(x).view(B, K, -1)
        return x
