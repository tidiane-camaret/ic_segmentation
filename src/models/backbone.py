"""
Backbone modules for PatchICL.

Provides backbone classes that work with pre-computed DINOv3 features,
skipping the patch embedding step.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskEncoder(nn.Module):
    """Simple encoder that converts spatial masks to token embeddings."""

    def __init__(self, embed_dim: int = 1024):
        super().__init__()
        # Simple CNN: mask -> embedding
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, embed_dim, kernel_size=3, padding=1),
        )

    def forward(self, masks: torch.Tensor, token_grid_size: int) -> torch.Tensor:
        """
        Args:
            masks: [B, K, 1, ps, ps] - Spatial masks
            token_grid_size: Target token grid size (h=w)

        Returns:
            mask_tokens: [B, K, tokens, embed_dim]
        """
        B, K, C, H, W = masks.shape
        masks_flat = masks.view(B * K, C, H, W)

        # Encode and pool to token grid
        encoded = self.encoder(masks_flat)  # [B*K, embed_dim, H, W]
        encoded = F.adaptive_avg_pool2d(encoded, (token_grid_size, token_grid_size))

        # Reshape to tokens: [B, K, tokens, embed_dim]
        tokens = encoded.flatten(2).transpose(1, 2)  # [B*K, tokens, embed_dim]
        return tokens.view(B, K, token_grid_size * token_grid_size, -1)


class SegmentationHead(nn.Module):
    """Decoder head with CNN upsampling from spatial feature grid."""

    def __init__(
        self,
        embed_dim: int = 1024,
        num_classes: int = 1,
        patch_size: int = 16,
        feature_grid_size: int = 7,
    ):
        """
        Args:
            embed_dim: Input feature dimension
            num_classes: Number of output classes
            patch_size: Target output patch size
            feature_grid_size: Spatial size of input features (e.g., 7 for 7x7 grid)
        """
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.feature_grid_size = feature_grid_size

        # Calculate required upsampling factor
        # From feature_grid_size to patch_size
        upsample_factor = patch_size / feature_grid_size  # e.g., 112/7 = 16

        # Build decoder with appropriate upsampling
        # Each block does 2x upsample, so we need log2(upsample_factor) blocks
        num_upsample_blocks = max(1, int(math.log2(upsample_factor)))

        layers = []
        in_channels = embed_dim
        channel_schedule = [512, 256, 128, 64]  # Channel reduction schedule

        for i in range(num_upsample_blocks):
            out_channels = channel_schedule[min(i, len(channel_schedule) - 1)]
            layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ])
            in_channels = out_channels

        # Final conv to num_classes
        layers.append(nn.Conv2d(in_channels, num_classes, kernel_size=1))

        self.decoder = nn.Sequential(*layers)

    def forward(
        self,
        patch_features: torch.Tensor,
        target_size: int | None = None,
    ):
        """
        Args:
            patch_features: [B, K, D, h, w] - spatial feature grid per patch
            target_size: Target output size (H=W). If None, uses decoder output size.

        Returns:
            [B, K, num_classes, target_size, target_size]
        """
        B, K, D, h, w = patch_features.shape

        # Reshape to [B*K, D, h, w] for per-patch decoding
        x = patch_features.reshape(B * K, D, h, w)

        # Forward through decoder
        output = self.decoder(x)  # [B*K, num_classes, H_out, W_out]

        # Resize to target size if specified and different from output
        _, nc, H, W = output.shape
        if target_size is not None and (H != target_size or W != target_size):
            output = F.interpolate(
                output, size=(target_size, target_size), mode='bilinear', align_corners=False
            )
            H, W = target_size, target_size

        # Reshape to [B, K, num_classes, H, W]
        output = output.reshape(B, K, nc, H, W)

        return output


class PixelShuffleDecoder(nn.Module):
    """
    Lightweight decoder using PixelShuffle for efficient upsampling.

    Best for Perceiver backbone where tokens are already contextually rich.
    Uses sub-pixel convolution which is more parameter-efficient than
    transposed convolutions or bilinear + conv.

    Architecture: 7x7 → 14x14 → 28x28 → 56x56 → 112x112 (4 PixelShuffle stages)
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_classes: int = 1,
        patch_size: int = 112,
        feature_grid_size: int = 7,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.feature_grid_size = feature_grid_size

        upsample_factor = patch_size / feature_grid_size  # 16x
        num_stages = max(1, int(math.log2(upsample_factor)))  # 4 stages

        # Initial projection to prepare for PixelShuffle
        # Each PixelShuffle(2) needs 4x channels to produce 2x spatial
        self.input_proj = nn.Sequential(
            nn.Conv2d(embed_dim, hidden_dim, 3, padding=1),
            nn.GELU(),
        )

        # PixelShuffle stages: each does 2x upsampling
        self.upsample_stages = nn.ModuleList()
        in_ch = hidden_dim

        for i in range(num_stages):
            # Conv to expand channels for PixelShuffle
            # PixelShuffle(2) reduces channels by 4x while 2x spatial
            out_ch = max(hidden_dim // (2 ** i), 16)
            self.upsample_stages.append(nn.Sequential(
                nn.Conv2d(in_ch, out_ch * 4, 3, padding=1),  # 4x channels for shuffle
                nn.PixelShuffle(2),  # Reduces channels by 4, increases spatial by 2
                nn.GELU(),
            ))
            in_ch = out_ch

        # Final projection to num_classes
        self.output_proj = nn.Conv2d(in_ch, num_classes, 1)

    def forward(
        self,
        patch_features: torch.Tensor,
        target_size: int | None = None,
    ):
        """
        Args:
            patch_features: [B, K, D, h, w]
            target_size: Target output size

        Returns:
            [B, K, num_classes, target_size, target_size]
        """
        B, K, D, h, w = patch_features.shape
        x = patch_features.reshape(B * K, D, h, w)

        x = self.input_proj(x)

        for stage in self.upsample_stages:
            x = stage(x)

        output = self.output_proj(x)

        # Resize if needed
        _, nc, H, W = output.shape
        if target_size is not None and (H != target_size or W != target_size):
            output = F.interpolate(
                output, size=(target_size, target_size),
                mode='bilinear', align_corners=False
            )

        return output.reshape(B, K, nc, target_size or H, target_size or W)


class UNetDecoder(nn.Module):
    """
    U-Net style decoder with multi-scale skip connections.

    Best for CNN backbone where encoder produces intermediate features
    at multiple scales. Fuses features at each resolution level.

    Expects skip connections from encoder at: 7x7, 4x4, 2x2
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_classes: int = 1,
        patch_size: int = 112,
        feature_grid_size: int = 7,
        bottleneck_dim: int = 128,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.feature_grid_size = feature_grid_size
        self.embed_dim = embed_dim

        # Decoder blocks: upsample + skip fusion
        # 1x1 → 2x2
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(bottleneck_dim, embed_dim, 2, stride=2),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )
        self.fuse1 = nn.Sequential(
            nn.Conv2d(embed_dim * 2, embed_dim, 3, padding=1),  # Concat skip
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )

        # 2x2 → 4x4
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim, 2, stride=2),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )
        self.fuse2 = nn.Sequential(
            nn.Conv2d(embed_dim * 2, embed_dim, 3, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )

        # 4x4 → 7x7 (interpolate since not power of 2)
        self.up3 = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 3, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )
        self.fuse3 = nn.Sequential(
            nn.Conv2d(embed_dim * 2, embed_dim, 3, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )

        # 7x7 → 112x112 (final upsampling, 16x)
        self.final_upsample = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(embed_dim, num_classes * 16, 1),  # Channels for PixelShuffle(4)
            nn.PixelShuffle(4),  # 7x7 → 28x28
        )
        # 28x28 → 112x112
        self.final_conv = nn.Sequential(
            nn.Conv2d(num_classes, num_classes * 16, 3, padding=1),
            nn.PixelShuffle(4),  # 28x28 → 112x112... wait this gives 4x not enough
        )

        # Actually simpler: 7x7 → 112x112 via interpolate + refinement
        self.final_upsample = nn.Sequential(
            nn.Conv2d(embed_dim, 64, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, num_classes, 1),
        )

    def forward(
        self,
        bottleneck: torch.Tensor,
        skip_2x2: torch.Tensor,
        skip_4x4: torch.Tensor,
        skip_7x7: torch.Tensor,
        target_size: int | None = None,
    ):
        """
        Args:
            bottleneck: [B*K, D, 1, 1] - encoder bottleneck
            skip_2x2: [B*K, D, 2, 2] - encoder features at 2x2
            skip_4x4: [B*K, D, 4, 4] - encoder features at 4x4
            skip_7x7: [B*K, D, 7, 7] - encoder features at 7x7
            target_size: Target output size

        Returns:
            [B*K, num_classes, target_size, target_size]
        """
        # 1x1 → 2x2 + skip
        x = self.up1(bottleneck)  # [B*K, D, 2, 2]
        x = torch.cat([x, skip_2x2], dim=1)
        x = self.fuse1(x)

        # 2x2 → 4x4 + skip
        x = self.up2(x)  # [B*K, D, 4, 4]
        x = torch.cat([x, skip_4x4], dim=1)
        x = self.fuse2(x)

        # 4x4 → 7x7 + skip
        x = F.interpolate(x, size=(7, 7), mode='bilinear', align_corners=False)
        x = self.up3(x)
        x = torch.cat([x, skip_7x7], dim=1)
        x = self.fuse3(x)

        # 7x7 → target_size
        x = self.final_upsample(x)
        target = target_size or self.patch_size
        x = F.interpolate(x, size=(target, target), mode='bilinear', align_corners=False)

        return x

def build_rope_cache(seq_len: int, dim: int, base: float = 10000.0) -> torch.Tensor:
    """Precompute RoPE sin/cos cache."""
    theta = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    positions = torch.arange(seq_len).float()
    freqs = torch.einsum("i,j->ij", positions, theta)  # [seq_len, dim/2]
    return torch.stack([freqs.cos(), freqs.sin()], dim=-1)  # [seq_len, dim/2, 2]


def apply_rope(x: torch.Tensor, rope_cache: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embedding to input tensor."""
    # x: [B, seq_len, dim]
    seq_len = x.shape[1]
    dim = x.shape[2]

    # Ensure cache is on same device
    rope_cache = rope_cache[:seq_len].to(x.device)  # [seq_len, dim/2, 2]

    # Split into pairs for rotation
    x_reshape = x.view(*x.shape[:-1], dim // 2, 2)  # [B, seq_len, dim/2, 2]
    cos = rope_cache[..., 0].unsqueeze(0)  # [1, seq_len, dim/2]
    sin = rope_cache[..., 1].unsqueeze(0)  # [1, seq_len, dim/2]

    # Apply rotation: (x * cos) + (rotate(x) * sin)
    x0, x1 = x_reshape[..., 0], x_reshape[..., 1]
    out0 = x0 * cos - x1 * sin
    out1 = x0 * sin + x1 * cos

    return torch.stack([out0, out1], dim=-1).view(*x.shape)


def build_rope_cache_2d(max_pos: int, dim: int, base: float = 10000.0) -> torch.Tensor:
    """
    Precompute 2D RoPE sin/cos cache for spatial positions.

    Splits dim in half: first half for x-position, second half for y-position.

    Args:
        max_pos: Maximum position value in either dimension
        dim: Total embedding dimension (must be divisible by 4)
        base: Base for frequency computation

    Returns:
        rope_cache: [max_pos, dim/4, 2] - sin/cos for each position
    """
    assert dim % 4 == 0, f"dim must be divisible by 4 for 2D RoPE, got {dim}"
    half_dim = dim // 2  # Each spatial dimension gets half

    # Frequencies for half the dimension (since we split x and y)
    theta = 1.0 / (base ** (torch.arange(0, half_dim, 2).float() / half_dim))
    positions = torch.arange(max_pos).float()
    freqs = torch.einsum("i,j->ij", positions, theta)  # [max_pos, dim/4]
    return torch.stack([freqs.cos(), freqs.sin()], dim=-1)  # [max_pos, dim/4, 2]


def apply_rope_2d(
    x: torch.Tensor,
    coords: torch.Tensor,
    rope_cache: torch.Tensor,
    image_size: int,
) -> torch.Tensor:
    """
    Apply 2D rotary position embedding based on spatial coordinates.

    Splits embedding in half: first half rotated by x-coord, second half by y-coord.

    Args:
        x: [B, K, dim] - Input tokens
        coords: [B, K, 2] - Patch coordinates (y, x) in pixel space
        rope_cache: [max_pos, dim/4, 2] - Precomputed sin/cos cache
        image_size: Image size for normalizing coordinates

    Returns:
        x with 2D positional rotations applied: [B, K, dim]
    """
    B, K, D = x.shape
    device = x.device
    half_dim = D // 2
    quarter_dim = D // 4

    # Normalize coordinates to [0, max_pos) range
    # coords are (y, x) in pixel space, normalize by image_size
    max_pos = rope_cache.shape[0]
    coords_normalized = (coords.float() / image_size * (max_pos - 1)).clamp(0, max_pos - 1)

    # Get integer positions for indexing
    y_pos = coords_normalized[:, :, 0].long()  # [B, K]
    x_pos = coords_normalized[:, :, 1].long()  # [B, K]

    # Fetch sin/cos for each position
    rope_cache = rope_cache.to(device)

    # Index into cache: [B, K, dim/4, 2]
    y_rope = rope_cache[y_pos]  # [B, K, dim/4, 2]
    x_rope = rope_cache[x_pos]  # [B, K, dim/4, 2]

    # Split input into x-half and y-half
    x_part = x[:, :, :half_dim]  # [B, K, dim/2] - rotated by x-coord
    y_part = x[:, :, half_dim:]  # [B, K, dim/2] - rotated by y-coord

    # Reshape for rotation: [B, K, dim/4, 2]
    x_part = x_part.view(B, K, quarter_dim, 2)
    y_part = y_part.view(B, K, quarter_dim, 2)

    # Extract cos/sin
    x_cos, x_sin = x_rope[..., 0], x_rope[..., 1]  # [B, K, dim/4]
    y_cos, y_sin = y_rope[..., 0], y_rope[..., 1]  # [B, K, dim/4]

    # Apply rotation to x-part using x-coordinates
    x0, x1 = x_part[..., 0], x_part[..., 1]  # [B, K, dim/4]
    x_out0 = x0 * x_cos - x1 * x_sin
    x_out1 = x0 * x_sin + x1 * x_cos
    x_rotated = torch.stack([x_out0, x_out1], dim=-1).view(B, K, half_dim)

    # Apply rotation to y-part using y-coordinates
    y0, y1 = y_part[..., 0], y_part[..., 1]  # [B, K, dim/4]
    y_out0 = y0 * y_cos - y1 * y_sin
    y_out1 = y0 * y_sin + y1 * y_cos
    y_rotated = torch.stack([y_out0, y_out1], dim=-1).view(B, K, half_dim)

    # Concatenate back
    return torch.cat([x_rotated, y_rotated], dim=-1)


class CrossPatchAttentionBackbone(nn.Module):
    """
    Perceiver-style backbone that preserves spatial token structure.

    Instead of flattening 49 tokens into one large vector, uses latent tokens
    to gather/broadcast information while keeping spatial structure:
    1. Project DINO features: [B, K, 49, 1024] → [B, K, 49, D]
    2. Latents cross-attend to spatial tokens (gather per-patch info)
    3. Cross-patch attention on latents (target→context)
    4. Spatial tokens cross-attend to latents (broadcast)
    5. Segmentation head on spatial features
    """

    def __init__(
        self,
        embed_dim: int = 1024,
        embed_proj_dim: int = 128,
        nb_features_per_patch: int = 49,
        patch_size: int = 16,
        image_size: int = 224,
        num_classes: int = 1,
        dropout: float = 0.1,
        max_seq_len: int = 1024,
        use_rope_2d: bool = True,
        num_registers: int = 4,
        num_latents_per_patch: int = 4,
        num_heads: int = 8,
        decoder_hidden_dim: int = 64,
    ):
        """
        Args:
            embed_dim: Input DINO feature dimension (1024 for ViT-L)
            embed_proj_dim: Projected dimension for processing
            nb_features_per_patch: Spatial tokens per patch (49 = 7x7)
            patch_size: Output segmentation patch size
            image_size: Original image size (for position embeddings)
            num_classes: Number of segmentation classes
            num_latents_per_patch: Learnable latent tokens per patch for gathering info
            num_heads: Number of attention heads
            decoder_hidden_dim: Hidden dimension for PixelShuffle decoder
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.embed_proj_dim = embed_proj_dim
        self.nb_features_per_patch = nb_features_per_patch
        self.decoder_hidden_dim = decoder_hidden_dim
        self.patch_size = patch_size
        self.image_size = image_size
        self.num_classes = num_classes
        self.dropout = dropout
        self.use_rope_2d = use_rope_2d
        self.num_registers = num_registers
        self.num_latents = num_latents_per_patch

        # Feature grid size (sqrt of nb_features_per_patch)
        self.feature_grid_size = int(math.sqrt(nb_features_per_patch))
        assert self.feature_grid_size ** 2 == nb_features_per_patch, \
            f"nb_features_per_patch must be a perfect square, got {nb_features_per_patch}"

        # Project DINO features to working dimension
        self.embed_proj_in = nn.Linear(embed_dim, embed_proj_dim)

        # Learnable latent tokens (shared across patches, will be expanded)
        self.latent_tokens = nn.Parameter(
            torch.randn(1, 1, num_latents_per_patch, embed_proj_dim) * 0.02
        )

        # 2D position embedding for spatial tokens within patch (7x7 grid)
        self.spatial_pos_embed = nn.Parameter(
            torch.randn(1, 1, nb_features_per_patch, embed_proj_dim) * 0.02
        )

        # Target/context type embeddings (applied to latents)
        self.target_embed = nn.Parameter(torch.randn(1, 1, embed_proj_dim) * 0.02)
        self.context_embed = nn.Parameter(torch.randn(1, 1, embed_proj_dim) * 0.02)

        # Register tokens for global context (treated as context latents)
        if num_registers > 0:
            self.register_tokens = nn.Parameter(
                torch.randn(1, num_registers, embed_proj_dim) * 0.02
            )
        else:
            self.register_tokens = None

        # RoPE cache for patch-level positions
        if use_rope_2d:
            rope_cache = build_rope_cache_2d(max_seq_len, embed_proj_dim)
        else:
            rope_cache = build_rope_cache(max_seq_len, embed_proj_dim)
        self.register_buffer("rope_cache", rope_cache, persistent=False)

        # Stage 1: Latents gather from spatial tokens (per-patch cross-attention)
        self.gather_attn = nn.MultiheadAttention(
            embed_dim=embed_proj_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout,
        )
        self.gather_norm_q = nn.LayerNorm(embed_proj_dim)
        self.gather_norm_kv = nn.LayerNorm(embed_proj_dim)

        # Stage 2: Cross-patch attention on latents (target→context)
        self.cross_patch_attn = ContextTargetAttention(
            embed_dim=embed_proj_dim,
            num_heads=num_heads,
            append_zero_attn=True,
        )
        self.cross_patch_norm = nn.LayerNorm(embed_proj_dim)
        self.cross_patch_mlp = nn.Sequential(
            nn.Linear(embed_proj_dim, embed_proj_dim * 4),
            nn.GELU(),
            nn.Linear(embed_proj_dim * 4, embed_proj_dim),
        )
        self.cross_patch_mlp_norm = nn.LayerNorm(embed_proj_dim)

        # Stage 3: Spatial tokens gather from latents (broadcast)
        self.broadcast_attn = nn.MultiheadAttention(
            embed_dim=embed_proj_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout,
        )
        self.broadcast_norm_q = nn.LayerNorm(embed_proj_dim)
        self.broadcast_norm_kv = nn.LayerNorm(embed_proj_dim)

        # Final MLP on spatial tokens
        self.final_mlp = nn.Sequential(
            nn.Linear(embed_proj_dim, embed_proj_dim * 2),
            nn.GELU(),
            nn.Linear(embed_proj_dim * 2, embed_proj_dim),
        )
        self.final_norm = nn.LayerNorm(embed_proj_dim)

        # Segmentation head - lightweight PixelShuffle since tokens are already rich
        self.mask_proj_out = PixelShuffleDecoder(
            embed_dim=embed_proj_dim,
            num_classes=num_classes,
            patch_size=patch_size,
            feature_grid_size=self.feature_grid_size,
            hidden_dim=decoder_hidden_dim,
        )

    def forward(
        self,
        img_patches: torch.Tensor,
        coords: torch.Tensor = None,
        ctx_id_labels: torch.Tensor = None,
    ) -> dict[str, torch.Tensor]:
        """
        Perceiver-style forward: gather → cross-patch → broadcast.

        Args:
            img_patches: [B, K, 49, 1024] - DINO features per patch
            coords: [B, K, 2] - Patch coordinates (y, x)
            ctx_id_labels: [B, K] - 0 for target, >0 for context

        Returns:
            Dict with mask_patch_logit_preds [B, K, C, ps, ps]
        """
        B, K, NF, E = img_patches.shape
        device = img_patches.device
        D = self.embed_proj_dim

        # === Step 1: Project DINO features ===
        # [B, K, 49, 1024] → [B, K, 49, D]
        spatial_tokens = self.embed_proj_in(img_patches.view(-1, E)).view(B, K, NF, D)

        # Add spatial position embedding within each patch
        spatial_tokens = spatial_tokens + self.spatial_pos_embed

        # Determine context vs target
        if ctx_id_labels is not None:
            is_context = ctx_id_labels > 0  # [B, K]
        else:
            is_context = torch.ones(B, K, dtype=torch.bool, device=device)
            is_context[:, 0] = False

        # === Step 2: Initialize latents per patch ===
        # [1, 1, num_latents, D] → [B, K, num_latents, D]
        latents = self.latent_tokens.expand(B, K, -1, -1).clone()

        # Add type embedding to latents
        type_embed = torch.where(
            is_context.view(B, K, 1, 1).expand(-1, -1, self.num_latents, D),
            self.context_embed.view(1, 1, 1, D).expand(B, K, self.num_latents, -1),
            self.target_embed.view(1, 1, 1, D).expand(B, K, self.num_latents, -1),
        )
        latents = latents + type_embed

        # Apply 2D RoPE to latents based on patch coords
        if self.use_rope_2d and coords is not None:
            # Reshape for RoPE: [B, K, num_latents, D] → [B, K*num_latents, D]
            latents_flat = latents.view(B, K * self.num_latents, D)
            # Expand coords to match: [B, K, 2] → [B, K*num_latents, 2]
            coords_expanded = coords.unsqueeze(2).expand(-1, -1, self.num_latents, -1)
            coords_flat = coords_expanded.reshape(B, K * self.num_latents, 2)
            latents_flat = apply_rope_2d(latents_flat, coords_flat, self.rope_cache, self.image_size)
            latents = latents_flat.view(B, K, self.num_latents, D)

        # === Step 3: Gather - Latents cross-attend to spatial tokens ===
        # Process each patch: latents attend to spatial tokens
        # Reshape for batch processing: [B*K, num_latents, D] and [B*K, 49, D]
        latents_bk = latents.view(B * K, self.num_latents, D)
        spatial_bk = spatial_tokens.view(B * K, NF, D)

        # Cross-attention: latents (Q) attend to spatial (K, V)
        latents_gathered, _ = self.gather_attn(
            self.gather_norm_q(latents_bk),
            self.gather_norm_kv(spatial_bk),
            spatial_bk,
        )
        latents_bk = latents_bk + latents_gathered  # Residual

        # Reshape back: [B, K, num_latents, D]
        latents = latents_bk.view(B, K, self.num_latents, D)

        # === Step 4: Cross-patch attention on latents ===
        # Pool latents per patch for cross-patch attention: [B, K, D]
        latents_pooled = latents.mean(dim=2)  # [B, K, D]

        # Add registers (treated as context)
        if self.register_tokens is not None:
            registers = self.register_tokens.expand(B, -1, -1)  # [B, num_reg, D]
            registers = registers + self.context_embed.expand(B, self.num_registers, -1)
            latents_with_reg = torch.cat([registers, latents_pooled], dim=1)  # [B, num_reg+K, D]
            reg_mask = torch.ones(B, self.num_registers, dtype=torch.bool, device=device)
            is_context_with_reg = torch.cat([reg_mask, is_context], dim=1)
        else:
            latents_with_reg = latents_pooled
            is_context_with_reg = is_context

        # Context-target attention: target latents attend to context latents
        latents_cross = self.cross_patch_attn(
            self.cross_patch_norm(latents_with_reg),
            is_context_with_reg,
        )

        # Remove registers
        if self.register_tokens is not None:
            latents_cross = latents_cross[:, self.num_registers:]  # [B, K, D]

        # Residual + MLP
        latents_pooled = latents_pooled + latents_cross
        latents_pooled = latents_pooled + self.cross_patch_mlp(self.cross_patch_mlp_norm(latents_pooled))

        # Broadcast pooled info back to all latents: [B, K, D] → [B, K, num_latents, D]
        latents = latents + latents_pooled.unsqueeze(2)

        # === Step 5: Broadcast - Spatial tokens cross-attend to latents ===
        latents_bk = latents.view(B * K, self.num_latents, D)
        spatial_bk = spatial_tokens.view(B * K, NF, D)

        # Cross-attention: spatial (Q) attend to latents (K, V)
        spatial_gathered, _ = self.broadcast_attn(
            self.broadcast_norm_q(spatial_bk),
            self.broadcast_norm_kv(latents_bk),
            latents_bk,
        )
        spatial_bk = spatial_bk + spatial_gathered  # Residual

        # Final MLP
        spatial_bk = spatial_bk + self.final_mlp(self.final_norm(spatial_bk))

        # === Step 6: Segmentation head ===
        # Reshape to spatial: [B*K, 49, D] → [B, K, D, 7, 7]
        spatial_out = spatial_bk.view(B, K, NF, D)
        h = w = self.feature_grid_size
        spatial_out = spatial_out.permute(0, 1, 3, 2).view(B, K, D, h, w)

        # Decode to mask predictions
        mask_pred = self.mask_proj_out(spatial_out, target_size=self.patch_size)

        return {
            'mask_patch_logit_preds': mask_pred,
            'img_patches': spatial_tokens,  # Return projected features
        }


class CNNCrossPatchBackbone(nn.Module):
    """
    CNN-based backbone with U-Net style multi-scale skip connections.

    Architecture:
    1. Project DINO features to spatial grid: [B, K, 49, 1024] → [B*K, D, 7, 7]
    2. CNN encoder with intermediate feature extraction at 7x7, 4x4, 2x2
    3. Cross-patch attention on bottleneck features (target→context)
    4. U-Net decoder fusing skip connections at each scale
    5. Final upsampling to output resolution

    More efficient than attention for small grids, preserves fine details via skips.
    """

    def __init__(
        self,
        embed_dim: int = 1024,
        embed_proj_dim: int = 128,
        nb_features_per_patch: int = 49,
        patch_size: int = 112,
        image_size: int = 224,
        num_classes: int = 1,
        dropout: float = 0.1,
        max_seq_len: int = 1024,
        use_rope_2d: bool = True,
        num_registers: int = 4,
        bottleneck_dim: int = 128,
        num_heads: int = 8,
        decoder_hidden_dim: int = 64,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.embed_proj_dim = embed_proj_dim
        self.nb_features_per_patch = nb_features_per_patch
        self.patch_size = patch_size
        self.image_size = image_size
        self.num_classes = num_classes
        self.bottleneck_dim = bottleneck_dim
        self.use_rope_2d = use_rope_2d
        self.num_registers = num_registers
        self.decoder_hidden_dim = decoder_hidden_dim

        # Feature grid size (sqrt of nb_features_per_patch)
        self.feature_grid_size = int(math.sqrt(nb_features_per_patch))
        assert self.feature_grid_size ** 2 == nb_features_per_patch

        D = embed_proj_dim  # Shorthand

        # Project DINO features
        self.embed_proj_in = nn.Linear(embed_dim, D)

        # === Encoder (saves intermediate features) ===
        # Level 0: 7x7 → 7x7
        self.enc0 = nn.Sequential(
            nn.Conv2d(D, D, 3, padding=1),
            nn.BatchNorm2d(D),
            nn.GELU(),
        )
        # Level 1: 7x7 → 4x4
        self.enc1 = nn.Sequential(
            nn.Conv2d(D, D * 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(D * 2),
            nn.GELU(),
        )
        # Level 2: 4x4 → 2x2
        self.enc2 = nn.Sequential(
            nn.Conv2d(D * 2, bottleneck_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(bottleneck_dim),
            nn.GELU(),
        )
        # Bottleneck: 2x2 → 1x1
        self.bottleneck_pool = nn.AdaptiveAvgPool2d(1)

        # === Cross-patch attention on bottleneck ===
        self.target_embed = nn.Parameter(torch.randn(1, 1, bottleneck_dim) * 0.02)
        self.context_embed = nn.Parameter(torch.randn(1, 1, bottleneck_dim) * 0.02)

        if num_registers > 0:
            self.register_tokens = nn.Parameter(
                torch.randn(1, num_registers, bottleneck_dim) * 0.02
            )
        else:
            self.register_tokens = None

        if use_rope_2d:
            rope_cache = build_rope_cache_2d(max_seq_len, bottleneck_dim)
        else:
            rope_cache = build_rope_cache(max_seq_len, bottleneck_dim)
        self.register_buffer("rope_cache", rope_cache, persistent=False)

        self.cross_patch_attn = ContextTargetAttention(
            embed_dim=bottleneck_dim,
            num_heads=num_heads,
            append_zero_attn=True,
        )
        self.cross_patch_norm = nn.LayerNorm(bottleneck_dim)
        self.cross_patch_mlp = nn.Sequential(
            nn.Linear(bottleneck_dim, bottleneck_dim * 4),
            nn.GELU(),
            nn.Linear(bottleneck_dim * 4, bottleneck_dim),
        )
        self.cross_patch_mlp_norm = nn.LayerNorm(bottleneck_dim)

        # === U-Net Decoder with skip connections ===
        # 1x1 → 2x2, fuse with enc2 output
        self.dec2_up = nn.ConvTranspose2d(bottleneck_dim, bottleneck_dim, 2, stride=2)
        self.dec2_fuse = nn.Sequential(
            nn.Conv2d(bottleneck_dim * 2, D * 2, 3, padding=1),  # Concat skip
            nn.BatchNorm2d(D * 2),
            nn.GELU(),
        )

        # 2x2 → 4x4, fuse with enc1 output
        self.dec1_up = nn.ConvTranspose2d(D * 2, D * 2, 2, stride=2)
        self.dec1_fuse = nn.Sequential(
            nn.Conv2d(D * 4, D, 3, padding=1),  # Concat skip (D*2 + D*2)
            nn.BatchNorm2d(D),
            nn.GELU(),
        )

        # 4x4 → 7x7, fuse with enc0 output
        self.dec0_fuse = nn.Sequential(
            nn.Conv2d(D * 2, D, 3, padding=1),  # Concat skip (D + D)
            nn.BatchNorm2d(D),
            nn.GELU(),
        )

        # === Final mask decoder: 7x7 → patch_size ===
        self.mask_decoder = PixelShuffleDecoder(
            embed_dim=D,
            num_classes=num_classes,
            patch_size=patch_size,
            feature_grid_size=self.feature_grid_size,
            hidden_dim=decoder_hidden_dim,
        )

    def forward(
        self,
        img_patches: torch.Tensor,
        coords: torch.Tensor = None,
        ctx_id_labels: torch.Tensor = None,
    ) -> dict[str, torch.Tensor]:
        """
        Multi-scale CNN encoder → cross-patch attention → U-Net decoder.

        Args:
            img_patches: [B, K, 49, 1024] - DINO features per patch
            coords: [B, K, 2] - Patch coordinates (y, x)
            ctx_id_labels: [B, K] - 0 for target, >0 for context

        Returns:
            Dict with mask_patch_logit_preds [B, K, C, ps, ps]
        """
        B, K, NF, E = img_patches.shape
        device = img_patches.device
        D = self.embed_proj_dim
        h = w = self.feature_grid_size  # 7

        # === Step 1: Project and reshape to spatial ===
        # [B, K, 49, 1024] → [B*K, D, 7, 7]
        spatial = self.embed_proj_in(img_patches.view(-1, E))  # [B*K*49, D]
        spatial = spatial.view(B * K, NF, D).permute(0, 2, 1)  # [B*K, D, 49]
        spatial = spatial.view(B * K, D, h, w)  # [B*K, D, 7, 7]

        # === Step 2: Multi-scale CNN Encoder (save skip features) ===
        # Level 0: 7x7
        skip0 = self.enc0(spatial)  # [B*K, D, 7, 7]

        # Level 1: 4x4
        skip1 = self.enc1(skip0)  # [B*K, D*2, 4, 4]

        # Level 2: 2x2
        skip2 = self.enc2(skip1)  # [B*K, bottleneck_dim, 2, 2]

        # Bottleneck: 1x1
        bottleneck = self.bottleneck_pool(skip2)  # [B*K, bottleneck_dim, 1, 1]
        bottleneck = bottleneck.view(B, K, self.bottleneck_dim)  # [B, K, bottleneck_dim]

        # Determine context vs target
        if ctx_id_labels is not None:
            is_context = ctx_id_labels > 0
        else:
            is_context = torch.ones(B, K, dtype=torch.bool, device=device)
            is_context[:, 0] = False

        # === Step 3: Cross-patch attention on bottleneck ===
        type_embed = torch.where(
            is_context.unsqueeze(-1),
            self.context_embed.expand(B, K, -1),
            self.target_embed.expand(B, K, -1),
        )
        bottleneck = bottleneck + type_embed

        if self.use_rope_2d and coords is not None:
            bottleneck = apply_rope_2d(bottleneck, coords, self.rope_cache, self.image_size)

        # Add registers
        if self.register_tokens is not None:
            registers = self.register_tokens.expand(B, -1, -1)
            registers = registers + self.context_embed.expand(B, self.num_registers, -1)
            bottleneck_with_reg = torch.cat([registers, bottleneck], dim=1)
            reg_mask = torch.ones(B, self.num_registers, dtype=torch.bool, device=device)
            is_context_with_reg = torch.cat([reg_mask, is_context], dim=1)
        else:
            bottleneck_with_reg = bottleneck
            is_context_with_reg = is_context

        cross_out = self.cross_patch_attn(
            self.cross_patch_norm(bottleneck_with_reg),
            is_context_with_reg,
        )

        if self.register_tokens is not None:
            cross_out = cross_out[:, self.num_registers:]

        bottleneck = bottleneck + cross_out
        bottleneck = bottleneck + self.cross_patch_mlp(self.cross_patch_mlp_norm(bottleneck))

        # === Step 4: U-Net Decoder with skip connections ===
        # [B, K, bottleneck_dim] → [B*K, bottleneck_dim, 1, 1]
        x = bottleneck.view(B * K, self.bottleneck_dim, 1, 1)

        # 1x1 → 2x2, fuse with skip2
        x = self.dec2_up(x)  # [B*K, bottleneck_dim, 2, 2]
        x = torch.cat([x, skip2], dim=1)  # [B*K, bottleneck_dim*2, 2, 2]
        x = self.dec2_fuse(x)  # [B*K, D*2, 2, 2]

        # 2x2 → 4x4, fuse with skip1
        x = self.dec1_up(x)  # [B*K, D*2, 4, 4]
        x = torch.cat([x, skip1], dim=1)  # [B*K, D*4, 4, 4]
        x = self.dec1_fuse(x)  # [B*K, D, 4, 4]

        # 4x4 → 7x7, fuse with skip0
        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)
        x = torch.cat([x, skip0], dim=1)  # [B*K, D*2, 7, 7]
        x = self.dec0_fuse(x)  # [B*K, D, 7, 7]

        # === Step 5: Final mask decoder (PixelShuffle) ===
        x = x.view(B, K, D, h, w)
        mask_pred = self.mask_decoder(x, target_size=self.patch_size)

        return {
            'mask_patch_logit_preds': mask_pred,
            'img_patches': spatial.view(B, K, D, h, w),
        }


class ContextTargetAttention(nn.Module):
    """
    Attention where:
    - Context patches do self-attention among themselves
    - Target patches do cross-attention to context patches only
    """
    
    def __init__(self, embed_dim: int, num_heads: int, append_zero_attn: bool = False):
        super().__init__()
        self.mha_ctx = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, batch_first=True
        )
        self.mha_tgt = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, batch_first=True
        )
        
        if append_zero_attn:
            self.mha_ctx = AppendZeroAttn(self.mha_ctx)
            self.mha_tgt = AppendZeroAttn(self.mha_tgt)
    
    def forward(
        self,
        x: torch.Tensor,
        is_context: torch.Tensor,
        need_weights: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, K, D] - all patch tokens
            is_context: [B, K] - boolean mask, True for context patches
            
        Returns:
            [B, K, D] - updated tokens
        """
        B, K, D = x.shape
        device = x.device
        
        # Build index tensors for gathering/scattering
        ctx_mask = is_context  # [B, K]
        tgt_mask = ~is_context  # [B, K]
        
        # Count patches per type (assume uniform across batch for simplicity)
        n_ctx = ctx_mask[0].sum().item()
        n_tgt = tgt_mask[0].sum().item()
        
        # Gather context and target tokens separately
        ctx_indices = ctx_mask.nonzero(as_tuple=False)  # [total_ctx, 2]
        tgt_indices = tgt_mask.nonzero(as_tuple=False)  # [total_tgt, 2]
        
        # Reshape indices for batched gathering
        ctx_idx = ctx_mask.long().cumsum(dim=1) - 1  # [B, K]
        tgt_idx = tgt_mask.long().cumsum(dim=1) - 1  # [B, K]
        
        # Extract context tokens: [B, n_ctx, D]
        ctx_tokens = self._gather_masked(x, ctx_mask, n_ctx)
        
        # Extract target tokens: [B, n_tgt, D]
        tgt_tokens = self._gather_masked(x, tgt_mask, n_tgt)
        
        # Context self-attention
        ctx_out, _ = self.mha_ctx(
            ctx_tokens, ctx_tokens, ctx_tokens, need_weights=need_weights
        )
        
        # Target cross-attention to context
        tgt_out, _ = self.mha_tgt(
            tgt_tokens, ctx_tokens, ctx_tokens, need_weights=need_weights
        )
        
        # Scatter results back to original positions
        output = torch.zeros_like(x)
        output = self._scatter_masked(output, ctx_out, ctx_mask, n_ctx)
        output = self._scatter_masked(output, tgt_out, tgt_mask, n_tgt)
        
        return output
    
    def _gather_masked(
        self, x: torch.Tensor, mask: torch.Tensor, n: int
    ) -> torch.Tensor:
        """Gather tokens where mask is True into [B, n, D] tensor."""
        B, K, D = x.shape
        
        # Create gather indices [B, n]
        indices = torch.zeros(B, n, dtype=torch.long, device=x.device)
        for b in range(B):
            indices[b] = mask[b].nonzero(as_tuple=False).squeeze(-1)
        
        # Gather: [B, n, D]
        return torch.gather(x, 1, indices.unsqueeze(-1).expand(-1, -1, D))
    
    def _scatter_masked(
        self, output: torch.Tensor, values: torch.Tensor, mask: torch.Tensor, n: int
    ) -> torch.Tensor:
        """Scatter values back to positions where mask is True."""
        B, K, D = output.shape
        
        # Create scatter indices [B, n]
        indices = torch.zeros(B, n, dtype=torch.long, device=output.device)
        for b in range(B):
            indices[b] = mask[b].nonzero(as_tuple=False).squeeze(-1)
        
        # Scatter: [B, K, D]
        output.scatter_(1, indices.unsqueeze(-1).expand(-1, -1, D), values)
        return output


class TransformerLayerContextTarget(nn.Module):
    """Transformer layer using context/target attention pattern."""
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_dim: int,
        bias_mlp: bool = True,
        layernorm_affine: bool = True,
        append_zero_attn: bool = False,
    ):
        super().__init__()
        
        self.attention = ContextTargetAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            append_zero_attn=append_zero_attn,
        )
        
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim, bias=bias_mlp),
            nn.GELU(),
            nn.Linear(mlp_dim, embed_dim, bias=bias_mlp),
        )
        
        self.norm1 = nn.LayerNorm(embed_dim, elementwise_affine=layernorm_affine)
        self.norm2 = nn.LayerNorm(embed_dim, elementwise_affine=layernorm_affine)
    
    def forward(
        self,
        x: torch.Tensor,
        is_context: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, K, D] - patch tokens
            is_context: [B, K] - True for context patches
            
        Returns:
            [B, K, D] - updated tokens
        """
        # Pre-norm attention with residual
        x = x + self.attention(self.norm1(x), is_context)
        
        # Pre-norm MLP with residual
        x = x + self.mlp(self.norm2(x))
        
        return x


# =============================================================================
# IMPROVED MULTI-LAYER CROSS-PATCH ATTENTION ARCHITECTURE
# =============================================================================

class BidirectionalCrossPatchAttention(nn.Module):
    """
    Improved attention mechanism with three-phase information flow:
    
    Phase 1: Context self-attention (context patches refine each other)
    Phase 2: Target cross-attention to context (targets gather from context)
    Phase 3: Target self-attention (targets coordinate with each other)
    
    Uses a unified attention mask approach for efficiency and Flash Attention compatibility.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        use_flash: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.use_flash = use_flash and hasattr(F, 'scaled_dot_product_attention')
        
        # Unified Q/K/V projections (more efficient than separate MHAs)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        is_context: torch.Tensor,
        return_attn: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, K, D] - all patch tokens
            is_context: [B, K] - boolean mask, True for context patches
            
        Returns:
            [B, K, D] - updated tokens (and optionally attention weights)
        """
        B, K, D = x.shape
        H = self.num_heads
        
        # Project Q, K, V
        q = self.q_proj(x).view(B, K, H, -1).transpose(1, 2)  # [B, H, K, head_dim]
        k = self.k_proj(x).view(B, K, H, -1).transpose(1, 2)
        v = self.v_proj(x).view(B, K, H, -1).transpose(1, 2)
        
        # Build attention mask for bidirectional flow:
        # - Context can attend to: context only (self-attention)
        # - Target can attend to: context + target (cross + self)
        # This allows targets to coordinate while still gathering from context
        
        is_ctx = is_context.unsqueeze(1)  # [B, 1, K] - which keys are context
        is_tgt = (~is_context).unsqueeze(2)  # [B, K, 1] - which queries are target
        
        # Attention mask: True = CAN attend
        # Context queries: only attend to context keys
        # Target queries: attend to everything (context + other targets)
        ctx_to_ctx = is_ctx.transpose(1, 2) & is_ctx  # [B, K, K]: ctx queries → ctx keys
        tgt_to_all = is_tgt.expand(-1, -1, K)  # [B, K, K]: tgt queries → all keys
        
        attn_mask = ctx_to_ctx | tgt_to_all  # [B, K, K]
        
        if self.use_flash:
            # Flash attention expects: True = BLOCKED (opposite of our mask)
            # Use float mask for compatibility: 0 = attend, -inf = block
            float_mask = torch.zeros(B, K, K, device=x.device, dtype=x.dtype)
            float_mask.masked_fill_(~attn_mask, float('-inf'))
            
            # Expand for heads: [B, H, K, K]
            float_mask = float_mask.unsqueeze(1).expand(-1, H, -1, -1)
            
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=float_mask,
                dropout_p=self.dropout.p if self.training else 0.0,
            )
            attn_weights = None
        else:
            # Manual attention
            attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, H, K, K]
            attn = attn.masked_fill(~attn_mask.unsqueeze(1), float('-inf'))
            attn_weights = F.softmax(attn, dim=-1)
            attn_weights = self.dropout(attn_weights)
            out = attn_weights @ v
        
        # Reshape and project out
        out = out.transpose(1, 2).reshape(B, K, D)
        out = self.out_proj(out)
        
        if return_attn and attn_weights is not None:
            return out, attn_weights
        return out


class AttentionPooling(nn.Module):
    """
    Learnable attention-based pooling that preserves information better than mean pooling.
    
    Uses a learned query to attend over the input sequence and produce a fixed-size output.
    """
    
    def __init__(self, embed_dim: int, num_heads: int = 4, num_queries: int = 1):
        super().__init__()
        self.num_queries = num_queries
        self.query = nn.Parameter(torch.randn(1, num_queries, embed_dim) * 0.02)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, D] - input sequence
            
        Returns:
            [B, num_queries, D] - pooled representation
        """
        B = x.shape[0]
        q = self.query.expand(B, -1, -1)
        pooled, _ = self.attn(q, self.norm(x), x)
        return pooled


class GatedFusion(nn.Module):
    """
    Gated fusion for combining information from different sources.
    
    Instead of simple addition, learns to gate how much of each input to use.
    """
    
    def __init__(self, embed_dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid(),
        )
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Fuse x and y with learned gating."""
        g = self.gate(torch.cat([x, y], dim=-1))
        return g * x + (1 - g) * y


class MultiLayerCrossPatchBlock(nn.Module):
    """
    Single block of the multi-layer cross-patch architecture.
    
    Each block:
    1. Per-patch self-attention (spatial tokens within each patch)
    2. Attention pooling to compress spatial → latent
    3. Cross-patch attention (latents interact across patches)
    4. Gated broadcast back to spatial tokens
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        num_latents: int = 4,
        use_spatial_attn: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_latents = num_latents
        self.use_spatial_attn = use_spatial_attn
        mlp_dim = int(embed_dim * mlp_ratio)
        
        # Per-patch spatial self-attention (optional, can skip for efficiency)
        if use_spatial_attn:
            self.spatial_attn = nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                batch_first=True,
                dropout=dropout,
            )
            self.spatial_norm = nn.LayerNorm(embed_dim)
        
        # Attention pooling: spatial tokens → latent tokens
        self.pool = AttentionPooling(embed_dim, num_heads=num_heads // 2, num_queries=num_latents)
        
        # Cross-patch attention on latents
        self.cross_patch = BidirectionalCrossPatchAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.cross_norm = nn.LayerNorm(embed_dim)
        
        # MLP after cross-patch attention
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.mlp_norm = nn.LayerNorm(embed_dim)
        
        # Broadcast: latents → spatial (cross-attention)
        self.broadcast_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout,
        )
        self.broadcast_norm_q = nn.LayerNorm(embed_dim)
        self.broadcast_norm_kv = nn.LayerNorm(embed_dim)
        
        # Gated fusion for residual
        self.fusion = GatedFusion(embed_dim)
    
    def forward(
        self,
        spatial_tokens: torch.Tensor,
        is_context: torch.Tensor,
        prev_latents: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            spatial_tokens: [B, K, N, D] - spatial tokens per patch (N=49 for 7x7)
            is_context: [B, K] - context/target mask
            prev_latents: [B, K, num_latents, D] - latents from previous layer (optional)
            
        Returns:
            spatial_tokens: [B, K, N, D] - updated spatial tokens
            latents: [B, K, num_latents, D] - latents for next layer
        """
        B, K, N, D = spatial_tokens.shape
        
        # === Step 1: Per-patch spatial self-attention (optional) ===
        if self.use_spatial_attn:
            spatial_flat = spatial_tokens.reshape(B * K, N, D)
            spatial_attn_out, _ = self.spatial_attn(
                self.spatial_norm(spatial_flat),
                self.spatial_norm(spatial_flat),
                spatial_flat,
            )
            spatial_flat = spatial_flat + spatial_attn_out
            spatial_tokens = spatial_flat.reshape(B, K, N, D)
        
        # === Step 2: Attention pooling → latents ===
        # [B, K, N, D] → [B*K, N, D] → pool → [B*K, num_latents, D] → [B, K, num_latents, D]
        spatial_flat = spatial_tokens.reshape(B * K, N, D)
        latents = self.pool(spatial_flat).reshape(B, K, self.num_latents, D)
        
        # Fuse with previous layer's latents if available (skip connection across layers)
        if prev_latents is not None:
            latents = latents + prev_latents
        
        # === Step 3: Cross-patch attention on pooled latents ===
        # Flatten latents: [B, K, num_latents, D] → [B, K*num_latents, D]
        latents_flat = latents.reshape(B, K * self.num_latents, D)
        
        # Expand is_context to match: [B, K] → [B, K*num_latents]
        is_context_expanded = is_context.unsqueeze(-1).expand(-1, -1, self.num_latents).reshape(B, -1)
        
        # Cross-patch attention
        latents_cross = self.cross_patch(
            self.cross_norm(latents_flat),
            is_context_expanded,
        )
        latents_flat = latents_flat + latents_cross
        
        # MLP
        latents_flat = latents_flat + self.mlp(self.mlp_norm(latents_flat))
        
        # Reshape back: [B, K*num_latents, D] → [B, K, num_latents, D]
        latents = latents_flat.reshape(B, K, self.num_latents, D)
        
        # === Step 4: Broadcast latents → spatial tokens ===
        latents_bk = latents.reshape(B * K, self.num_latents, D)
        spatial_flat = spatial_tokens.reshape(B * K, N, D)
        
        broadcast_out, _ = self.broadcast_attn(
            self.broadcast_norm_q(spatial_flat),
            self.broadcast_norm_kv(latents_bk),
            latents_bk,
        )
        
        # Gated fusion instead of simple addition
        spatial_flat = self.fusion(spatial_flat, spatial_flat + broadcast_out)
        spatial_tokens = spatial_flat.reshape(B, K, N, D)
        
        return spatial_tokens, latents


class MultiLayerCrossPatchBackbone(nn.Module):
    """
    Improved backbone with multiple cross-patch attention layers.
    
    Key improvements over CrossPatchAttentionBackbone:
    1. Multiple layers of cross-patch attention for deeper reasoning
    2. Bidirectional attention: targets can coordinate with each other
    3. Attention pooling instead of mean pooling (preserves information)
    4. Gated fusion for residual connections
    5. Skip connections for latents across layers
    6. Flash Attention compatible
    
    Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │ Input: DINO features [B, K, 49, 1024]                       │
    │        ↓                                                     │
    │ Project: [B, K, 49, D]                                      │
    │        ↓                                                     │
    │ ┌─────────────────────────────────────────────────────────┐ │
    │ │  Layer 1                                                 │ │
    │ │  ├─ Spatial self-attn (per-patch)                       │ │
    │ │  ├─ Attention pool → latents                            │ │
    │ │  ├─ Cross-patch attn (latents, bidirectional)           │ │
    │ │  └─ Gated broadcast → spatial                           │ │
    │ └─────────────────────────────────────────────────────────┘ │
    │        ↓ (spatial + latents)                                │
    │ ┌─────────────────────────────────────────────────────────┐ │
    │ │  Layer 2 (latent skip connection from Layer 1)          │ │
    │ │  ...                                                     │ │
    │ └─────────────────────────────────────────────────────────┘ │
    │        ↓                                                     │
    │ ┌─────────────────────────────────────────────────────────┐ │
    │ │  Layer N                                                 │ │
    │ └─────────────────────────────────────────────────────────┘ │
    │        ↓                                                     │
    │ Decode: [B, K, 49, D] → [B, K, 1, 112, 112]                 │
    └─────────────────────────────────────────────────────────────┘
    """
    
    def __init__(
        self,
        embed_dim: int = 1024,
        embed_proj_dim: int = 128,
        nb_features_per_patch: int = 49,
        patch_size: int = 112,
        image_size: int = 224,
        num_classes: int = 1,
        dropout: float = 0.1,
        max_seq_len: int = 1024,
        use_rope_2d: bool = True,
        num_registers: int = 4,
        num_latents_per_patch: int = 4,
        num_heads: int = 8,
        num_layers: int = 3,
        decoder_hidden_dim: int = 64,
        use_spatial_attn: bool = True,
    ):
        """
        Args:
            embed_dim: Input DINO feature dimension (1024 for ViT-L)
            embed_proj_dim: Projected dimension for processing
            nb_features_per_patch: Spatial tokens per patch (49 = 7x7)
            patch_size: Output segmentation patch size
            image_size: Original image size
            num_classes: Number of segmentation classes
            dropout: Dropout rate
            num_latents_per_patch: Latent tokens per patch
            num_heads: Attention heads
            num_layers: Number of cross-patch attention layers
            decoder_hidden_dim: PixelShuffle decoder hidden dim
            use_spatial_attn: Whether to use per-patch spatial attention in blocks
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.embed_proj_dim = embed_proj_dim
        self.nb_features_per_patch = nb_features_per_patch
        self.patch_size = patch_size
        self.image_size = image_size
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.use_rope_2d = use_rope_2d
        self.num_registers = num_registers
        
        # Feature grid size
        self.feature_grid_size = int(math.sqrt(nb_features_per_patch))
        assert self.feature_grid_size ** 2 == nb_features_per_patch
        
        D = embed_proj_dim
        
        # === Input projection ===
        self.embed_proj_in = nn.Linear(embed_dim, D)
        
        # Spatial position embedding (7x7 grid within each patch)
        self.spatial_pos_embed = nn.Parameter(
            torch.randn(1, 1, nb_features_per_patch, D) * 0.02
        )
        
        # Type embeddings (applied per-patch before first layer)
        self.target_embed = nn.Parameter(torch.randn(1, 1, 1, D) * 0.02)
        self.context_embed = nn.Parameter(torch.randn(1, 1, 1, D) * 0.02)
        
        # Register tokens (global context, concatenated during cross-patch)
        if num_registers > 0:
            # Registers as "pseudo-patches" with their own latents
            self.register_tokens = nn.Parameter(
                torch.randn(1, num_registers, num_latents_per_patch, D) * 0.02
            )
        else:
            self.register_tokens = None
        
        # 2D RoPE for patch-level positions
        if use_rope_2d:
            rope_cache = build_rope_cache_2d(max_seq_len, D)
        else:
            rope_cache = build_rope_cache(max_seq_len, D)
        self.register_buffer("rope_cache", rope_cache, persistent=False)
        
        # === Multi-layer cross-patch blocks ===
        self.layers = nn.ModuleList([
            MultiLayerCrossPatchBlock(
                embed_dim=D,
                num_heads=num_heads,
                mlp_ratio=4.0,
                dropout=dropout,
                num_latents=num_latents_per_patch,
                use_spatial_attn=use_spatial_attn,
            )
            for _ in range(num_layers)
        ])
        
        # === Output ===
        self.final_norm = nn.LayerNorm(D)
        
        # Segmentation decoder
        self.mask_decoder = PixelShuffleDecoder(
            embed_dim=D,
            num_classes=num_classes,
            patch_size=patch_size,
            feature_grid_size=self.feature_grid_size,
            hidden_dim=decoder_hidden_dim,
        )
    
    def forward(
        self,
        img_patches: torch.Tensor,
        coords: torch.Tensor = None,
        ctx_id_labels: torch.Tensor = None,
    ) -> dict[str, torch.Tensor]:
        """
        Multi-layer cross-patch attention forward.
        
        Args:
            img_patches: [B, K, 49, 1024] - DINO features per patch
            coords: [B, K, 2] - Patch coordinates (y, x)
            ctx_id_labels: [B, K] - 0 for target, >0 for context
            
        Returns:
            Dict with mask_patch_logit_preds [B, K, C, ps, ps]
        """
        B, K, NF, E = img_patches.shape
        device = img_patches.device
        D = self.embed_proj_dim
        
        # === Project DINO features ===
        spatial_tokens = self.embed_proj_in(img_patches.reshape(-1, E)).reshape(B, K, NF, D)
        
        # Add spatial position embedding
        spatial_tokens = spatial_tokens + self.spatial_pos_embed
        
        # Determine context vs target
        if ctx_id_labels is not None:
            is_context = ctx_id_labels > 0  # [B, K]
        else:
            is_context = torch.ones(B, K, dtype=torch.bool, device=device)
            is_context[:, 0] = False
        
        # Add type embedding
        type_embed = torch.where(
            is_context.view(B, K, 1, 1).expand(-1, -1, NF, D),
            self.context_embed.expand(B, K, NF, D),
            self.target_embed.expand(B, K, NF, D),
        )
        spatial_tokens = spatial_tokens + type_embed
        
        # Apply 2D RoPE to spatial tokens based on patch coordinates
        if self.use_rope_2d and coords is not None:
            # Apply RoPE per spatial token (they all share the patch coord)
            for i in range(NF):
                spatial_tokens[:, :, i] = apply_rope_2d(
                    spatial_tokens[:, :, i],
                    coords,
                    self.rope_cache,
                    self.image_size,
                )
        
        # Add registers to the sequence (treated as context patches)
        if self.register_tokens is not None:
            reg = self.register_tokens.expand(B, -1, -1, -1)  # [B, num_reg, num_latents, D]
            # Create dummy spatial tokens for registers (they only participate via latents)
            reg_spatial = reg.mean(dim=2, keepdim=True).expand(-1, -1, NF, -1)  # [B, num_reg, NF, D]
            spatial_tokens = torch.cat([reg_spatial, spatial_tokens], dim=1)  # [B, num_reg+K, NF, D]
            
            reg_is_context = torch.ones(B, self.num_registers, dtype=torch.bool, device=device)
            is_context = torch.cat([reg_is_context, is_context], dim=1)  # [B, num_reg+K]
            K_total = K + self.num_registers
        else:
            K_total = K
        
        # === Multi-layer processing ===
        prev_latents = None
        for layer in self.layers:
            spatial_tokens, latents = layer(spatial_tokens, is_context, prev_latents)
            prev_latents = latents  # Pass latents to next layer as skip connection
        
        # Remove registers
        if self.register_tokens is not None:
            spatial_tokens = spatial_tokens[:, self.num_registers:]  # [B, K, NF, D]
        
        # === Final normalization ===
        spatial_tokens = self.final_norm(spatial_tokens)
        
        # === Decode to masks ===
        # Reshape: [B, K, NF, D] → [B, K, D, h, w]
        h = w = self.feature_grid_size
        spatial_out = spatial_tokens.permute(0, 1, 3, 2).reshape(B, K, D, h, w)
        
        mask_pred = self.mask_decoder(spatial_out, target_size=self.patch_size)
        
        return {
            'mask_patch_logit_preds': mask_pred,
            'img_patches': spatial_tokens,
        }


class AppendZeroAttn(nn.Module):
    def __init__(self, mha):
        super().__init__()
        assert mha.batch_first, (
            "AppendZeroAttn expects batch_first=True, otherwise rewrite the forward method"
        )
        self.mha = mha

    def forward(self, query, key, value, **kwargs):
        key_with_zero = torch.cat((key, torch.zeros_like(key[:, :1])), dim=1)
        value_with_zero = torch.cat((value, torch.zeros_like(value[:, :1])), dim=1)
        return self.mha(query, key_with_zero, value_with_zero, **kwargs)


class PrecomputedFeatureBackbone(nn.Module):
    """
    Backbone that uses pre-computed DINOv3 features directly.

    Skips patch embedding and processes features through a transformer
    with custom position embeddings, then decodes to segmentation masks.

    Supports:
    - Masked training where random patches are replaced with a learnable mask token
    - Mask conditioning: context mask embeddings are added to context features
    """

    def __init__(
        self,
        embed_dim: int = 1024,
        num_heads: int = 8,
        num_layers: int = 4,
        patch_size: int = 16,
        image_size: int = 224,
        num_classes: int = 1,
        dropout: float = 0.1,
        use_mask_token: bool = True,
        use_mask_conditioning: bool = True,
    ):
        """
        Args:
            embed_dim: Feature embedding dimension (1024 for DINOv3 ViT-L)
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            patch_size: Output patch size for segmentation
            image_size: Original image size (for position embeddings)
            num_classes: Number of segmentation classes
            dropout: Dropout rate
            use_mask_token: Whether to initialize a learnable mask token
            use_mask_conditioning: Whether to add mask embeddings to context features
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.image_size = image_size
        self.dino_patch_size = 16  # DINOv3's internal patch size
        self.use_mask_conditioning = use_mask_conditioning

        # Learnable mask token for masked training
        if use_mask_token:
            self.mask_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        else:
            self.mask_token = None

        # Mask encoder for conditioning context features
        if use_mask_conditioning:
            self.mask_encoder = MaskEncoder(embed_dim=embed_dim)
        else:
            self.mask_encoder = None

        # Tokens per input patch (depends on how features are extracted)
        # For a 16x16 patch from 224x224 image, we get 1 token
        # For larger patches, we get more tokens
        self.max_tokens_per_patch = (patch_size // self.dino_patch_size) ** 2

        # Position embeddings for tokens within a patch
        self.token_pos_embed = nn.Parameter(
            torch.randn(1, 196, embed_dim) * 0.02  # Max 14x14 = 196 tokens
        )

        # Position embeddings for patch locations in the image
        self.num_patch_positions = (image_size // patch_size) ** 2
        self.patch_pos_embed = nn.Parameter(
            torch.randn(1, max(1, self.num_patch_positions), embed_dim) * 0.02
        )

        # Learnable embeddings to distinguish target vs context patches
        self.target_embed = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.context_embed = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        # Transformer for cross-patch attention
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Segmentation head
        self.seg_head = SegmentationHead(
            embed_dim=embed_dim,
            num_classes=num_classes,
            patch_size=self.dino_patch_size,
        )

    def apply_mask(
        self,
        features: torch.Tensor,
        patch_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply masking to patch features.

        Args:
            features: [B, K, tokens_per_patch, embed_dim] - Patch features
            patch_mask: [B, K] - Boolean mask, True = mask this patch

        Returns:
            masked_features: [B, K, tokens_per_patch, embed_dim]
        """
        if self.mask_token is None:
            return features

        B, K, T, D = features.shape
        # Expand mask to match feature dimensions: [B, K, 1, 1]
        mask_expanded = patch_mask.view(B, K, 1, 1).expand(-1, -1, T, D)

        # Expand mask token: [B, K, T, D]
        mask_token_expanded = self.mask_token.expand(B, K, T, -1)

        # Replace masked patches with mask token
        masked_features = torch.where(mask_expanded, mask_token_expanded, features)
        return masked_features

    def forward(
        self,
        patches: torch.Tensor,
        coords: torch.Tensor = None,
        precomputed_features: torch.Tensor = None,
        patch_mask: torch.Tensor = None,
        actual_image_size: int | None = None,
        return_features: bool = False,
        context_masks: torch.Tensor = None,
        num_target_patches: int | None = None,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """
        Process patches using pre-computed features.

        Args:
            patches: [B, K, C, ps, ps] - Raw patches (used for shape info)
            coords: [B, K, 2] - Patch coordinates in original image space
            precomputed_features: [B, K, tokens_per_patch, embed_dim] - Pre-computed DINOv3 features
            patch_mask: [B, K] - Boolean mask, True = mask this patch (optional)
            actual_image_size: Actual image size that coords are relative to.
            return_features: If True, return dict with logits and features.
            context_masks: [B, K_ctx, 1, ps, ps] - GT masks for context patches (optional).
                When provided with num_target_patches, mask embeddings are added to
                context features (features after first num_target_patches).
            num_target_patches: Number of target patches (K_target). Context patches
                are features[K_target:]. Required when context_masks is provided.

        Returns:
            If return_features=False: patch_logits [B, K, 1, ps, ps]
            If return_features=True: dict with 'patch_logits' and 'features'
        """
        B, K, _, ps, _ = patches.shape

        if precomputed_features is None:
            raise ValueError("PrecomputedFeatureBackbone requires precomputed_features")

        # Apply masking if provided
        if patch_mask is not None:
            precomputed_features = self.apply_mask(precomputed_features, patch_mask)

        # precomputed_features: [B, K, tokens_per_patch, embed_dim]
        tokens_per_patch = precomputed_features.shape[2]
        h = w = int(math.sqrt(tokens_per_patch))

        # Add mask conditioning to context features if provided
        if (context_masks is not None and 
            num_target_patches is not None and 
            self.mask_encoder is not None):
            K_target = num_target_patches
            K_ctx = K - K_target

            if K_ctx > 0:
                # Encode context masks to embeddings
                # context_masks: [B, K_ctx, 1, ps, ps] -> [B, K_ctx, tokens, embed_dim]
                mask_embeddings = self.mask_encoder(context_masks, h)

                # Add mask embeddings to context features
                # precomputed_features: [B, K, tokens, embed_dim]
                # Context features are the last K_ctx patches
                precomputed_features = precomputed_features.clone()
                precomputed_features[:, K_target:] = (
                    precomputed_features[:, K_target:] + mask_embeddings
                )

        # Reshape to [B*K, tokens_per_patch, embed_dim]
        features = precomputed_features.reshape(B * K, tokens_per_patch, self.embed_dim)

        # Add token position embeddings (interpolate if needed)
        if tokens_per_patch <= self.token_pos_embed.shape[1]:
            features = features + self.token_pos_embed[:, :tokens_per_patch, :]
        else:
            # Interpolate position embeddings
            pos_embed = self.token_pos_embed.transpose(1, 2)  # [1, embed_dim, 196]
            pos_embed = F.interpolate(pos_embed, size=tokens_per_patch, mode='linear', align_corners=False)
            pos_embed = pos_embed.transpose(1, 2)  # [1, tokens_per_patch, embed_dim]
            features = features + pos_embed

        # Add patch position embeddings based on coordinates
        if coords is not None:
            # Scale coords from actual_image_size to backbone's image_size for position lookup
            img_size = actual_image_size if actual_image_size is not None else self.image_size
            coord_scale = self.image_size / img_size
            scaled_coords = coords.float() * coord_scale

            grid_size = max(1, self.image_size // self.patch_size)
            h_idx = (scaled_coords[:, :, 0] / self.patch_size).clamp(0, grid_size - 1)
            w_idx = (scaled_coords[:, :, 1] / self.patch_size).clamp(0, grid_size - 1)
            pos_idx = (h_idx * grid_size + w_idx).reshape(B * K)
            pos_idx = pos_idx.clamp(0, self.num_patch_positions - 1).long()

            patch_pos = self.patch_pos_embed[:, pos_idx, :].squeeze(0)  # [B*K, embed_dim]
            features = features + patch_pos.unsqueeze(1)  # Broadcast to all tokens

        # Add target/context embeddings to distinguish patch types
        # features: [B*K, tokens_per_patch, embed_dim]
        if num_target_patches is not None:
            K_target = num_target_patches
            K_ctx = K - K_target
            features = features.reshape(B, K, tokens_per_patch, self.embed_dim)
            
            # Add target embedding to first K_target patches
            features[:, :K_target] = features[:, :K_target] + self.target_embed.unsqueeze(2)
            
            # Add context embedding to remaining K_ctx patches
            if K_ctx > 0:
                features[:, K_target:] = features[:, K_target:] + self.context_embed.unsqueeze(2)
            
            features = features.reshape(B * K, tokens_per_patch, self.embed_dim)

        # Reshape for cross-patch attention: [B, K * tokens_per_patch, embed_dim]
        features = features.reshape(B, K * tokens_per_patch, self.embed_dim)

        # Apply transformer
        features = self.transformer(features)

        # Save features before segmentation head (reshape to per-patch format)
        features_before_head = features.reshape(B, K, tokens_per_patch, self.embed_dim)

        # Apply segmentation head with target size matching patch size
        seg_output = self.seg_head(features, h, w, num_patches=K, target_size=ps)

        if return_features:
            return {
                'patch_logits': seg_output,
                'features': features_before_head,  # [B, K, tokens, embed_dim]
            }
        return seg_output


# =============================================================================
# MODULAR ARCHITECTURE: Separate encoder, cross-attention, and decoder
# =============================================================================


class PatchEncoderBase(nn.Module):
    """Base class for patch encoders that convert DINO features to compact representations."""

    def __init__(self, embed_dim: int, output_dim: int, feature_grid_size: int = 7):
        super().__init__()
        self.embed_dim = embed_dim
        self.output_dim = output_dim
        self.feature_grid_size = feature_grid_size

    def forward(
        self, img_patches: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Args:
            img_patches: [B, K, 49, 1024] - DINO features per patch

        Returns:
            encoded: [B, K, output_dim] - Compact representation per patch
            skips: Dict of skip features for decoder (empty if not applicable)
        """
        raise NotImplementedError


class CNNPatchEncoder(PatchEncoderBase):
    """
    CNN encoder that compresses spatial features with multi-scale skip connections.

    Flow: [B, K, 49, 1024] → project → CNN layers → [B, K, output_dim]
    Also saves intermediate features at 7x7, 4x4, 2x2 for U-Net decoder.
    """

    def __init__(
        self,
        embed_dim: int = 1024,
        embed_proj_dim: int = 128,
        output_dim: int = 128,
        feature_grid_size: int = 7,
    ):
        super().__init__(embed_dim, output_dim, feature_grid_size)
        self.embed_proj_dim = embed_proj_dim
        D = embed_proj_dim

        # Project DINO features
        self.proj = nn.Linear(embed_dim, D)

        # Encoder levels (save intermediate for skips)
        self.enc0 = nn.Sequential(
            nn.Conv2d(D, D, 3, padding=1),
            nn.BatchNorm2d(D),
            nn.GELU(),
        )
        self.enc1 = nn.Sequential(
            nn.Conv2d(D, D * 2, 3, stride=2, padding=1),  # 7→4
            nn.BatchNorm2d(D * 2),
            nn.GELU(),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(D * 2, output_dim, 3, stride=2, padding=1),  # 4→2
            nn.BatchNorm2d(output_dim),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)  # 2→1

    def forward(
        self, img_patches: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        B, K, NF, E = img_patches.shape
        h = w = self.feature_grid_size
        D = self.embed_proj_dim

        # Project and reshape: [B, K, 49, 1024] → [B*K, D, 7, 7]
        x = self.proj(img_patches.view(-1, E))
        x = x.view(B * K, NF, D).permute(0, 2, 1).view(B * K, D, h, w)

        # Encode with skip connections
        skip0 = self.enc0(x)      # [B*K, D, 7, 7]
        skip1 = self.enc1(skip0)  # [B*K, D*2, 4, 4]
        skip2 = self.enc2(skip1)  # [B*K, output_dim, 2, 2]
        encoded = self.pool(skip2).view(B, K, self.output_dim)  # [B, K, output_dim]

        skips = {
            'skip0': skip0.view(B, K, D, h, w),
            'skip1': skip1.view(B, K, D * 2, 4, 4),
            'skip2': skip2.view(B, K, self.output_dim, 2, 2),
            'spatial_dim': D,
        }
        return encoded, skips


class PerceiverPatchEncoder(PatchEncoderBase):
    """
    Perceiver-style encoder using latent cross-attention to gather information.

    Flow: [B, K, 49, 1024] → project → latents cross-attend → pool → [B, K, output_dim]
    No skip connections (tokens are already contextually rich).
    """

    def __init__(
        self,
        embed_dim: int = 1024,
        embed_proj_dim: int = 128,
        output_dim: int = 128,
        feature_grid_size: int = 7,
        num_latents: int = 4,
        num_heads: int = 8,
    ):
        super().__init__(embed_dim, output_dim, feature_grid_size)
        self.embed_proj_dim = embed_proj_dim
        self.num_latents = num_latents
        nb_features = feature_grid_size ** 2

        # Project DINO features
        self.proj = nn.Linear(embed_dim, embed_proj_dim)

        # Spatial position embedding
        self.spatial_pos = nn.Parameter(torch.randn(1, 1, nb_features, embed_proj_dim) * 0.02)

        # Learnable latent tokens
        self.latents = nn.Parameter(torch.randn(1, 1, num_latents, embed_proj_dim) * 0.02)

        # Latents cross-attend to spatial tokens
        self.gather_attn = nn.MultiheadAttention(
            embed_dim=embed_proj_dim, num_heads=num_heads, batch_first=True
        )
        self.gather_norm_q = nn.LayerNorm(embed_proj_dim)
        self.gather_norm_kv = nn.LayerNorm(embed_proj_dim)

        # Project to output_dim if different
        self.out_proj = nn.Linear(embed_proj_dim, output_dim) if output_dim != embed_proj_dim else nn.Identity()

    def forward(
        self, img_patches: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        B, K, NF, E = img_patches.shape

        # Project: [B, K, 49, 1024] → [B, K, 49, D]
        spatial = self.proj(img_patches) + self.spatial_pos

        # Initialize latents: [B, K, num_latents, D]
        latents = self.latents.expand(B, K, -1, -1).clone()

        # Gather: latents cross-attend to spatial
        spatial_flat = spatial.view(B * K, NF, -1)
        latents_flat = latents.view(B * K, self.num_latents, -1)

        gathered, _ = self.gather_attn(
            self.gather_norm_q(latents_flat),
            self.gather_norm_kv(spatial_flat),
            spatial_flat,
        )
        latents_flat = latents_flat + gathered

        # Pool latents and project: [B, K, output_dim]
        encoded = latents_flat.mean(dim=1)  # [B*K, D]
        encoded = self.out_proj(encoded).view(B, K, self.output_dim)

        # Return spatial tokens for decoder (no multi-scale skips)
        skips = {
            'spatial': spatial,  # [B, K, 49, D]
            'latents': latents_flat.view(B, K, self.num_latents, -1),  # [B, K, num_latents, D]
        }
        return encoded, skips


class PatchDecoderBase(nn.Module):
    """Base class for patch decoders that convert compact representations to masks."""

    def __init__(self, input_dim: int, num_classes: int, patch_size: int, feature_grid_size: int = 7):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.feature_grid_size = feature_grid_size

    def forward(
        self, encoded: torch.Tensor, skips: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Args:
            encoded: [B, K, input_dim] - Encoded features after cross-attention
            skips: Dict of skip features from encoder

        Returns:
            masks: [B, K, num_classes, patch_size, patch_size]
        """
        raise NotImplementedError


class UNetPatchDecoder(PatchDecoderBase):
    """
    U-Net style decoder with multi-scale skip connections.

    Flow: [B, K, D] → expand → fuse skips at 2x2, 4x4, 7x7 → upsample → [B, K, C, H, W]
    """

    def __init__(
        self,
        input_dim: int = 128,
        num_classes: int = 1,
        patch_size: int = 112,
        feature_grid_size: int = 7,
        hidden_dim: int = 64,
    ):
        super().__init__(input_dim, num_classes, patch_size, feature_grid_size)
        D = input_dim  # Assume spatial_dim == input_dim for simplicity

        # Upsample and fuse: 1→2, fuse with skip2
        self.up2 = nn.ConvTranspose2d(input_dim, input_dim, 2, stride=2)
        self.fuse2 = nn.Sequential(
            nn.Conv2d(input_dim * 2, D * 2, 3, padding=1),
            nn.BatchNorm2d(D * 2),
            nn.GELU(),
        )

        # 2→4, fuse with skip1
        self.up1 = nn.ConvTranspose2d(D * 2, D * 2, 2, stride=2)
        self.fuse1 = nn.Sequential(
            nn.Conv2d(D * 4, D, 3, padding=1),
            nn.BatchNorm2d(D),
            nn.GELU(),
        )

        # 4→7, fuse with skip0
        self.fuse0 = nn.Sequential(
            nn.Conv2d(D * 2, D, 3, padding=1),
            nn.BatchNorm2d(D),
            nn.GELU(),
        )

        # Final upsampling: 7→patch_size
        self.final = PixelShuffleDecoder(
            embed_dim=D,
            num_classes=num_classes,
            patch_size=patch_size,
            feature_grid_size=feature_grid_size,
            hidden_dim=hidden_dim,
        )

    def forward(
        self, encoded: torch.Tensor, skips: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        B, K, D = encoded.shape
        h = self.feature_grid_size

        skip0 = skips['skip0'].view(B * K, -1, h, h)
        skip1 = skips['skip1'].view(B * K, -1, 4, 4)
        skip2 = skips['skip2'].view(B * K, -1, 2, 2)

        # Expand: [B, K, D] → [B*K, D, 1, 1]
        x = encoded.view(B * K, D, 1, 1)

        # 1→2, fuse skip2
        x = self.up2(x)
        x = torch.cat([x, skip2], dim=1)
        x = self.fuse2(x)

        # 2→4, fuse skip1
        x = self.up1(x)
        x = torch.cat([x, skip1], dim=1)
        x = self.fuse1(x)

        # 4→7, fuse skip0
        x = F.interpolate(x, size=(h, h), mode='bilinear', align_corners=False)
        x = torch.cat([x, skip0], dim=1)
        x = self.fuse0(x)

        # Final: 7→patch_size
        spatial_dim = skips.get('spatial_dim', D)
        x = x.view(B, K, spatial_dim, h, h)
        return self.final(x, target_size=self.patch_size)


class PerceiverPatchDecoder(PatchDecoderBase):
    """
    Perceiver-style decoder where spatial tokens cross-attend to encoded features.

    Flow: spatial tokens cross-attend to encoded → [B, K, 49, D] → upsample → [B, K, C, H, W]
    """

    def __init__(
        self,
        input_dim: int = 128,
        num_classes: int = 1,
        patch_size: int = 112,
        feature_grid_size: int = 7,
        hidden_dim: int = 64,
        num_heads: int = 8,
    ):
        super().__init__(input_dim, num_classes, patch_size, feature_grid_size)
        nb_features = feature_grid_size ** 2

        # Spatial tokens cross-attend to encoded (broadcast)
        self.broadcast_attn = nn.MultiheadAttention(
            embed_dim=input_dim, num_heads=num_heads, batch_first=True
        )
        self.broadcast_norm_q = nn.LayerNorm(input_dim)
        self.broadcast_norm_kv = nn.LayerNorm(input_dim)

        # Final MLP
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.GELU(),
            nn.Linear(input_dim * 2, input_dim),
        )
        self.mlp_norm = nn.LayerNorm(input_dim)

        # Upsample to output
        self.final = PixelShuffleDecoder(
            embed_dim=input_dim,
            num_classes=num_classes,
            patch_size=patch_size,
            feature_grid_size=feature_grid_size,
            hidden_dim=hidden_dim,
        )

    def forward(
        self, encoded: torch.Tensor, skips: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        B, K, D = encoded.shape
        h = self.feature_grid_size
        NF = h * h

        # Get spatial tokens and latents from skips
        spatial = skips.get('spatial')  # [B, K, 49, D]
        latents = skips.get('latents')  # [B, K, num_latents, D]

        if spatial is None:
            raise ValueError("PerceiverPatchDecoder requires 'spatial' in skips")

        # Broadcast: spatial cross-attend to encoded (expanded to match latent count)
        spatial_flat = spatial.view(B * K, NF, D)

        # Use latents + encoded as keys/values
        if latents is not None:
            # Add encoded info to latents
            latents_flat = latents.view(B * K, -1, D)
            encoded_expanded = encoded.view(B * K, 1, D).expand(-1, latents_flat.shape[1], -1)
            kv = latents_flat + encoded_expanded
        else:
            kv = encoded.view(B * K, 1, D)

        broadcasted, _ = self.broadcast_attn(
            self.broadcast_norm_q(spatial_flat),
            self.broadcast_norm_kv(kv),
            kv,
        )
        spatial_flat = spatial_flat + broadcasted
        spatial_flat = spatial_flat + self.mlp(self.mlp_norm(spatial_flat))

        # Reshape and upsample
        x = spatial_flat.view(B, K, NF, D)
        x = x.permute(0, 1, 3, 2).view(B, K, D, h, h)
        return self.final(x, target_size=self.patch_size)


class CrossPatchAttentionBlock(nn.Module):
    """
    Shared cross-patch attention mechanism.

    - Adds type embeddings (target vs context)
    - Applies 2D RoPE based on spatial coords
    - Context patches: self-attention
    - Target patches: cross-attention to context
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 8,
        num_registers: int = 4,
        use_rope_2d: bool = True,
        image_size: int = 224,
        max_seq_len: int = 1024,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_registers = num_registers
        self.use_rope_2d = use_rope_2d
        self.image_size = image_size

        # Type embeddings
        self.target_embed = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.context_embed = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        # Register tokens
        if num_registers > 0:
            self.register_tokens = nn.Parameter(
                torch.randn(1, num_registers, embed_dim) * 0.02
            )
        else:
            self.register_tokens = None

        # RoPE cache
        if use_rope_2d:
            rope_cache = build_rope_cache_2d(max_seq_len, embed_dim)
        else:
            rope_cache = build_rope_cache(max_seq_len, embed_dim)
        self.register_buffer("rope_cache", rope_cache, persistent=False)

        # Attention
        self.attn = ContextTargetAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            append_zero_attn=True,
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.mlp_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        coords: torch.Tensor,
        ctx_id_labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, K, D] - Encoded patch features
            coords: [B, K, 2] - Patch coordinates
            ctx_id_labels: [B, K] - 0=target, >0=context

        Returns:
            [B, K, D] - Features after cross-patch attention
        """
        B, K, D = x.shape
        device = x.device

        # Determine context vs target
        is_context = ctx_id_labels > 0 if ctx_id_labels is not None else torch.ones(B, K, dtype=torch.bool, device=device)
        if ctx_id_labels is None:
            is_context[:, 0] = False

        # Add type embeddings
        type_embed = torch.where(
            is_context.unsqueeze(-1),
            self.context_embed.expand(B, K, -1),
            self.target_embed.expand(B, K, -1),
        )
        x = x + type_embed

        # Apply RoPE
        if self.use_rope_2d and coords is not None:
            x = apply_rope_2d(x, coords, self.rope_cache, self.image_size)

        # Add registers
        if self.register_tokens is not None:
            registers = self.register_tokens.expand(B, -1, -1)
            registers = registers + self.context_embed.expand(B, self.num_registers, -1)
            x = torch.cat([registers, x], dim=1)
            reg_mask = torch.ones(B, self.num_registers, dtype=torch.bool, device=device)
            is_context = torch.cat([reg_mask, is_context], dim=1)

        # Cross-patch attention
        attn_out = self.attn(self.norm(x), is_context)

        # Remove registers
        if self.register_tokens is not None:
            attn_out = attn_out[:, self.num_registers:]
            x = x[:, self.num_registers:]

        # Residual + MLP
        x = x + attn_out
        x = x + self.mlp(self.mlp_norm(x))

        return x


class ModularBackbone(nn.Module):
    """
    Modular backbone composing encoder + cross-attention + decoder.

    Allows mixing different encoder/decoder types while keeping
    cross-patch attention mechanism the same.
    """

    def __init__(
        self,
        encoder: PatchEncoderBase,
        decoder: PatchDecoderBase,
        cross_attention: CrossPatchAttentionBlock,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.cross_attention = cross_attention

    def forward(
        self,
        img_patches: torch.Tensor,
        coords: torch.Tensor = None,
        ctx_id_labels: torch.Tensor = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            img_patches: [B, K, 49, 1024] - DINO features
            coords: [B, K, 2] - Patch coordinates
            ctx_id_labels: [B, K] - 0=target, >0=context

        Returns:
            Dict with mask_patch_logit_preds [B, K, C, H, W]
        """
        # Encode
        encoded, skips = self.encoder(img_patches)

        # Cross-patch attention
        encoded = self.cross_attention(encoded, coords, ctx_id_labels)

        # Decode
        masks = self.decoder(encoded, skips)

        return {
            'mask_patch_logit_preds': masks,
            'img_patches': img_patches,
        }


def build_modular_backbone(
    encoder_type: str = "cnn",
    decoder_type: str = "unet",
    embed_dim: int = 1024,
    embed_proj_dim: int = 128,
    bottleneck_dim: int = 128,
    num_classes: int = 1,
    patch_size: int = 112,
    image_size: int = 224,
    num_heads: int = 8,
    num_registers: int = 4,
    num_latents: int = 4,
    decoder_hidden_dim: int = 64,
    use_rope_2d: bool = True,
    feature_grid_size: int = 8,
) -> ModularBackbone:
    """Factory function to build modular backbone from config."""

    # Build encoder
    if encoder_type == "cnn":
        encoder = CNNPatchEncoder(
            embed_dim=embed_dim,
            embed_proj_dim=embed_proj_dim,
            output_dim=bottleneck_dim,
            feature_grid_size=feature_grid_size,
        )
    else:  # perceiver
        encoder = PerceiverPatchEncoder(
            embed_dim=embed_dim,
            embed_proj_dim=embed_proj_dim,
            output_dim=bottleneck_dim,
            feature_grid_size=feature_grid_size,
            num_latents=num_latents,
            num_heads=num_heads,
        )

    # Build decoder
    if decoder_type == "unet":
        decoder = UNetPatchDecoder(
            input_dim=bottleneck_dim,
            num_classes=num_classes,
            patch_size=patch_size,
            feature_grid_size=feature_grid_size,
            hidden_dim=decoder_hidden_dim,
        )
    else:  # perceiver
        decoder = PerceiverPatchDecoder(
            input_dim=bottleneck_dim,
            num_classes=num_classes,
            patch_size=patch_size,
            feature_grid_size=feature_grid_size,
            hidden_dim=decoder_hidden_dim,
            num_heads=num_heads,
        )

    # Build cross-attention (always the same)
    cross_attention = CrossPatchAttentionBlock(
        embed_dim=bottleneck_dim,
        num_heads=num_heads,
        num_registers=num_registers,
        use_rope_2d=use_rope_2d,
        image_size=image_size,
    )

    return ModularBackbone(encoder, decoder, cross_attention)

        