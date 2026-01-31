"""
Simplified backbone for PatchICL.

A clean, focused implementation replacing the complex backbone.py.
Uses CNNs for encoding/decoding and a modular attention mechanism.

Architecture:
    SimpleCNNEncoder → CrossPatchAttention → SimpleCNNDecoder
    [B,K,49,1024]        [B,K,D]              [B,K,D] + skips
         │                  │                      │
         ▼                  ▼                      ▼
    Linear(1024→D)      + type_embed          TransConv + skips
    Reshape [B*K,D,8,8] + 2D RoPE             Upsample
    Conv layers         + registers            → [B,K,C,ps,ps]
    Pool [B,K,D]        Masked attention

Config example:
    backbone:
      type: "simple"
      encoder:
        embed_dim: 1024        # DINO input dim
        embed_proj_dim: 128    # Working dimension
      cross_attention:
        num_heads: 8
        num_layers: 1          # Number of attention layers (default 1)
        num_registers: 4
        target_self_attention: false
        dropout: 0.0
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


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
    half_dim = dim // 2

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

    max_pos = rope_cache.shape[0]
    coords_normalized = (coords.float() / image_size * (max_pos - 1)).clamp(0, max_pos - 1)

    y_pos = coords_normalized[:, :, 0].long()
    x_pos = coords_normalized[:, :, 1].long()

    rope_cache = rope_cache.to(device)
    y_rope = rope_cache[y_pos]  # [B, K, dim/4, 2]
    x_rope = rope_cache[x_pos]

    x_part = x[:, :, :half_dim].view(B, K, quarter_dim, 2)
    y_part = x[:, :, half_dim:].view(B, K, quarter_dim, 2)

    x_cos, x_sin = x_rope[..., 0], x_rope[..., 1]
    y_cos, y_sin = y_rope[..., 0], y_rope[..., 1]

    x0, x1 = x_part[..., 0], x_part[..., 1]
    x_out0 = x0 * x_cos - x1 * x_sin
    x_out1 = x0 * x_sin + x1 * x_cos
    x_rotated = torch.stack([x_out0, x_out1], dim=-1).view(B, K, half_dim)

    y0, y1 = y_part[..., 0], y_part[..., 1]
    y_out0 = y0 * y_cos - y1 * y_sin
    y_out1 = y0 * y_sin + y1 * y_cos
    y_rotated = torch.stack([y_out0, y_out1], dim=-1).view(B, K, half_dim)

    return torch.cat([x_rotated, y_rotated], dim=-1)


class SimpleCNNEncoder(nn.Module):
    """
    CNN encoder that processes DINO features through spatial convolutions.

    Takes [B, K, 49, 1024] DINO features and produces:
    - encoded: [B, K, embed_dim] - pooled representation for attention
    - skips: dict of intermediate features at 8x8, 4x4, 2x2 scales
    """

    def __init__(
        self,
        input_dim: int = 1024,
        embed_dim: int = 128,
        feature_grid_size: int = 8,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.feature_grid_size = feature_grid_size

        D = embed_dim

        # Project DINO features to working dimension
        self.input_proj = nn.Linear(input_dim, D)

        # Encoder levels with skip connections
        # Level 0: 8x8 → 8x8 (preserve resolution)
        self.enc0 = nn.Sequential(
            nn.Conv2d(D, D, 3, padding=1),
            nn.BatchNorm2d(D),
            nn.GELU(),
        )

        # Level 1: 8x8 → 4x4 (stride-2 conv, 8→4 with padding)
        self.enc1 = nn.Sequential(
            nn.Conv2d(D, D, 3, stride=2, padding=1),
            nn.BatchNorm2d(D),
            nn.GELU(),
        )

        # Level 2: 4x4 → 2x2 (stride-2 conv)
        self.enc2 = nn.Sequential(
            nn.Conv2d(D, D, 3, stride=2, padding=1),
            nn.BatchNorm2d(D),
            nn.GELU(),
        )

        # Final pooling to [B*K, D, 1, 1]
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(
        self, features: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Args:
            features: [B, K, 49, 1024] - DINO patch features

        Returns:
            encoded: [B, K, embed_dim] - pooled representation
            skips: dict with 'skip_8x8', 'skip_4x4', 'skip_2x2' tensors
        """
        B, K, NF, E = features.shape
        D = self.embed_dim
        h = w = self.feature_grid_size

        # Project and reshape to spatial: [B*K, D, 8, 8]
        x = self.input_proj(features.view(-1, E))  # [B*K*49, D]
        x = x.view(B * K, NF, D).permute(0, 2, 1)  # [B*K, D, 49]
        x = x.view(B * K, D, h, w)  # [B*K, D, 8, 8]

        # Encode with skip connections
        skip_8x8 = self.enc0(x)  # [B*K, D, 8, 8]
        skip_4x4 = self.enc1(skip_8x8)  # [B*K, D, 4, 4]
        skip_2x2 = self.enc2(skip_4x4)  # [B*K, D, 2, 2]

        # Pool to [B, K, D]
        encoded = self.pool(skip_2x2).view(B, K, D)

        skips = {
            'skip_8x8': skip_8x8.view(B, K, D, h, w),
            'skip_4x4': skip_4x4.view(B, K, D, 4, 4),
            'skip_2x2': skip_2x2.view(B, K, D, 2, 2),
        }

        return encoded, skips


class AttentionBlock(nn.Module):
    """Single attention + MLP block with pre-norm and residuals."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        mlp_ratio: int = 4,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.norm1 = nn.LayerNorm(embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * mlp_ratio, embed_dim),
            nn.Dropout(dropout),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor,
        return_attn_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Args:
            x: [B, K, D] - tokens (including registers)
            attn_mask: [B, H, K, K] - float attention mask (-inf for blocked)
            return_attn_weights: If True, compute and return attention weights

        Returns:
            x: [B, K, D] - updated tokens
            attn_weights: [B, H, K, K] or None - attention weights if requested
        """
        B, K, D = x.shape
        H = self.num_heads

        # Pre-norm attention
        x_normed = self.norm1(x)
        q = self.q_proj(x_normed).view(B, K, H, -1).transpose(1, 2)
        k = self.k_proj(x_normed).view(B, K, H, -1).transpose(1, 2)
        v = self.v_proj(x_normed).view(B, K, H, -1).transpose(1, 2)

        attn_weights = None
        if return_attn_weights:
            # Manual attention computation to get weights
            scale = 1.0 / math.sqrt(q.shape[-1])
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            attn_scores = attn_scores + attn_mask
            attn_weights = F.softmax(attn_scores, dim=-1)
            if self.training and self.dropout.p > 0:
                attn_weights = self.dropout(attn_weights)
            out = torch.matmul(attn_weights, v)
        else:
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.dropout.p if self.training else 0.0,
            )

        out = out.transpose(1, 2).reshape(B, K, D)
        out = self.out_proj(out)
        x = x + out

        # Pre-norm MLP
        x = x + self.mlp(self.norm2(x))

        return x, attn_weights


class CrossPatchAttention(nn.Module):
    """
    Multi-layer attention module for cross-patch communication.

    Implements the context/target attention pattern:
    - Context patches: self-attention among themselves
    - Target patches: cross-attention to context patches (optionally + self-attention)

    Supports stacking multiple attention layers for deeper reasoning.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 1,
        num_registers: int = 4,
        image_size: int = 224,
        max_seq_len: int = 1024,
        target_self_attention: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_registers = num_registers
        self.image_size = image_size
        self.target_self_attention = target_self_attention

        # Type embeddings (applied once at input)
        self.target_embed = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.context_embed = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        # Register tokens for global context
        if num_registers > 0:
            self.register_tokens = nn.Parameter(
                torch.randn(1, num_registers, embed_dim) * 0.02
            )
        else:
            self.register_tokens = None

        # RoPE cache
        rope_cache = build_rope_cache_2d(max_seq_len, embed_dim)
        self.register_buffer("rope_cache", rope_cache, persistent=False)

        # Stack of attention blocks
        self.layers = nn.ModuleList([
            AttentionBlock(embed_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        # Final layer norm
        self.final_norm = nn.LayerNorm(embed_dim)

    def _build_attention_mask(
        self, is_context: torch.Tensor
    ) -> torch.Tensor:
        """
        Build attention mask for context/target pattern.

        Args:
            is_context: [B, K] - True for context/register tokens

        Returns:
            mask: [B, K, K] - True where attention is allowed
        """
        B, K = is_context.shape

        is_ctx = is_context.unsqueeze(1)  # [B, 1, K] - which keys are context
        is_tgt_q = (~is_context).unsqueeze(2)  # [B, K, 1] - which queries are target

        # Context queries → context keys only
        ctx_to_ctx = is_ctx.transpose(1, 2) & is_ctx  # [B, K, K]

        if self.target_self_attention:
            # Target queries → all keys (context + target)
            tgt_mask = is_tgt_q.expand(-1, -1, K)
        else:
            # Target queries → context keys only
            tgt_mask = is_tgt_q & is_ctx

        attn_mask = ctx_to_ctx | tgt_mask
        return attn_mask

    def forward(
        self,
        x: torch.Tensor,
        coords: torch.Tensor,
        is_context: torch.Tensor,
        return_attn_weights: bool = False,
    ) -> tuple[torch.Tensor, dict | None]:
        """
        Args:
            x: [B, K, D] - patch tokens
            coords: [B, K, 2] - patch coordinates (y, x)
            is_context: [B, K] - True for context patches
            return_attn_weights: If True, return attention weights and registers

        Returns:
            x: [B, K, D] - updated tokens
            extras: dict with 'attn_weights' and 'register_tokens' if requested, else None
        """
        B, K, D = x.shape
        device = x.device

        # Add type embeddings (once at input)
        type_embed = torch.where(
            is_context.unsqueeze(-1),
            self.context_embed.expand(B, K, -1),
            self.target_embed.expand(B, K, -1),
        )
        x = x + type_embed

        # Apply 2D RoPE (once at input)
        x = apply_rope_2d(x, coords, self.rope_cache, self.image_size)

        # Add registers
        if self.register_tokens is not None:
            registers = self.register_tokens.expand(B, -1, -1)
            registers = registers + self.context_embed.expand(B, self.num_registers, -1)
            x = torch.cat([registers, x], dim=1)  # [B, R+K, D]
            reg_mask = torch.ones(B, self.num_registers, dtype=torch.bool, device=device)
            is_context_with_reg = torch.cat([reg_mask, is_context], dim=1)
        else:
            is_context_with_reg = is_context

        # Build attention mask (shared across layers)
        K_total = x.shape[1]
        attn_mask = self._build_attention_mask(is_context_with_reg)
        float_mask = torch.zeros(B, K_total, K_total, device=device, dtype=x.dtype)
        float_mask.masked_fill_(~attn_mask, float('-inf'))
        float_mask = float_mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)

        # Run through attention layers
        all_attn_weights = []
        for layer in self.layers:
            x, attn_w = layer(x, float_mask, return_attn_weights=return_attn_weights)
            if attn_w is not None:
                all_attn_weights.append(attn_w)

        # Final norm
        x = self.final_norm(x)

        # Capture register tokens before removing
        register_tokens_out = None
        if return_attn_weights and self.register_tokens is not None:
            register_tokens_out = x[:, :self.num_registers].clone()

        # Remove registers
        if self.register_tokens is not None:
            x = x[:, self.num_registers:]

        extras = None
        if return_attn_weights:
            extras = {
                'attn_weights': all_attn_weights,  # List of [B, H, K, K] per layer
                'register_tokens': register_tokens_out,  # [B, R, D]
            }

        return x, extras


class SimpleCNNDecoder(nn.Module):
    """
    U-Net style decoder with skip connections.

    Takes encoded features and skip connections from encoder,
    progressively upsamples with skip fusion to output resolution.
    Uses bilinear interpolation for final upsampling to handle any patch_size.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_classes: int = 1,
        patch_size: int = 16,
        feature_grid_size: int = 8,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.feature_grid_size = feature_grid_size

        D = embed_dim

        # 1x1 → 2x2, fuse with skip_2x2
        self.up1 = nn.ConvTranspose2d(D, D, 2, stride=2)
        self.fuse1 = nn.Sequential(
            nn.Conv2d(D * 2, D, 3, padding=1),
            nn.BatchNorm2d(D),
            nn.GELU(),
        )

        # 2x2 → 4x4, fuse with skip_4x4
        self.up2 = nn.ConvTranspose2d(D, D, 2, stride=2)
        self.fuse2 = nn.Sequential(
            nn.Conv2d(D * 2, D, 3, padding=1),
            nn.BatchNorm2d(D),
            nn.GELU(),
        )

        # 4x4 → 8x8 (interpolate), fuse with skip_8x8
        self.fuse3 = nn.Sequential(
            nn.Conv2d(D * 2, D, 3, padding=1),
            nn.BatchNorm2d(D),
            nn.GELU(),
        )

        # Final conv and projection before upsampling
        self.final_conv = nn.Sequential(
            nn.Conv2d(D, D // 2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(D // 2, num_classes, 1),
        )

    def forward(
        self,
        encoded: torch.Tensor,
        skips: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            encoded: [B, K, D] - attention output
            skips: dict with 'skip_8x8', 'skip_4x4', 'skip_2x2' from encoder

        Returns:
            [B, K, num_classes, patch_size, patch_size]
        """
        B, K, D = encoded.shape
        h = self.feature_grid_size

        # Reshape encoded to [B*K, D, 1, 1] (use contiguous for after attention)
        x = encoded.contiguous().view(B * K, D, 1, 1)

        # Get skip connections reshaped to [B*K, D, H, W]
        skip_2x2 = skips['skip_2x2'].contiguous().view(B * K, D, 2, 2)
        skip_4x4 = skips['skip_4x4'].contiguous().view(B * K, D, 4, 4)
        skip_8x8 = skips['skip_8x8'].contiguous().view(B * K, D, h, h)

        # 1x1 → 2x2 + skip
        x = self.up1(x)  # [B*K, D, 2, 2]
        x = torch.cat([x, skip_2x2], dim=1)
        x = self.fuse1(x)  # [B*K, D, 2, 2]

        # 2x2 → 4x4 + skip
        x = self.up2(x)  # [B*K, D, 4, 4]
        x = torch.cat([x, skip_4x4], dim=1)
        x = self.fuse2(x)  # [B*K, D, 4, 4]

        # 4x4 → hxh + skip
        x = F.interpolate(x, size=(h, h), mode='bilinear', align_corners=False)
        x = torch.cat([x, skip_8x8], dim=1)
        x = self.fuse3(x)  # [B*K, D, 8, 8]

        # Final conv to num_classes
        x = self.final_conv(x)  # [B*K, num_classes, 8, 8]

        # Upsample to patch_size
        x = F.interpolate(
            x, size=(self.patch_size, self.patch_size),
            mode='bilinear', align_corners=False
        )

        return x.view(B, K, self.num_classes, self.patch_size, self.patch_size)


class SimpleBackbone(nn.Module):
    """
    Simplified backbone composing encoder → attention → decoder.

    Interface matches existing backbones:
        Input:
            img_patches: [B, K, 49, 1024] - DINO features
            coords: [B, K, 2] - patch coordinates (y, x)
            ctx_id_labels: [B, K] - 0=target, 1..k=context images

        Output:
            {'mask_patch_logit_preds': [B, K, num_classes, patch_size, patch_size]}
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 1,
        num_registers: int = 4,
        num_classes: int = 1,
        patch_size: int = 16,
        image_size: int = 224,
        input_dim: int = 1024,
        feature_grid_size: int = 8,
        target_self_attention: bool = False,
        dropout: float = 0.0,
        max_seq_len: int = 1024,
    ):
        """
        Args:
            embed_dim: Working dimension throughout the model
            num_heads: Number of attention heads
            num_layers: Number of attention layers (default 1)
            num_registers: Number of register tokens for global context
            num_classes: Number of output segmentation classes
            patch_size: Output patch size (typically 16)
            image_size: Full image size for position normalization
            input_dim: Input DINO feature dimension (1024 for ViT-L)
            feature_grid_size: DINO feature grid size (8 for 8x8 grid)
            target_self_attention: Allow targets to attend to each other
            dropout: Attention dropout rate
            max_seq_len: Maximum sequence length for RoPE cache
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.image_size = image_size

        self.encoder = SimpleCNNEncoder(
            input_dim=input_dim,
            embed_dim=embed_dim,
            feature_grid_size=feature_grid_size,
        )

        self.attention = CrossPatchAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            num_registers=num_registers,
            image_size=image_size,
            max_seq_len=max_seq_len,
            target_self_attention=target_self_attention,
            dropout=dropout,
        )

        self.decoder = SimpleCNNDecoder(
            embed_dim=embed_dim,
            num_classes=num_classes,
            patch_size=patch_size,
            feature_grid_size=feature_grid_size,
        )

    def forward(
        self,
        img_patches: torch.Tensor,
        coords: torch.Tensor = None,
        ctx_id_labels: torch.Tensor = None,
        return_attn_weights: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass: encode → attention → decode.

        Args:
            img_patches: [B, K, 49, 1024] - DINO features per patch
            coords: [B, K, 2] - Patch coordinates (y, x)
            ctx_id_labels: [B, K] - 0 for target, >0 for context
            return_attn_weights: If True, return attention weights and register tokens

        Returns:
            Dict with:
                'mask_patch_logit_preds': [B, K, C, ps, ps]
                'attn_weights': List of [B, H, K, K] per layer (if return_attn_weights)
                'register_tokens': [B, R, D] (if return_attn_weights)
        """
        B, K, NF, E = img_patches.shape
        device = img_patches.device

        # Default coords if not provided
        if coords is None:
            coords = torch.zeros(B, K, 2, device=device)

        # Determine context vs target
        if ctx_id_labels is not None:
            is_context = ctx_id_labels > 0
        else:
            is_context = torch.ones(B, K, dtype=torch.bool, device=device)
            is_context[:, 0] = False

        # Encode
        encoded, skips = self.encoder(img_patches)

        # Cross-patch attention
        attended, extras = self.attention(encoded, coords, is_context, return_attn_weights)

        # Decode
        mask_pred = self.decoder(attended, skips)

        result = {
            'mask_patch_logit_preds': mask_pred,
        }

        if extras is not None:
            result['attn_weights'] = extras['attn_weights']
            result['register_tokens'] = extras['register_tokens']

        return result
