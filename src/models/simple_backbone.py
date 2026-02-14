"""
Simplified backbone for PatchICL.

A clean, focused implementation replacing the complex backbone.py.
Uses CNNs for encoding/decoding and a modular attention mechanism.

Architecture:
    SimpleCNNEncoder → CrossPatchAttention → SimpleCNNDecoder
    [B,K,49,1024]        [B,K,D]              [B,K,D] + skips
         │                  │                      │
         ▼                  ▼                      ▼
    Linear(1024→D)      + registers           TransConv + skips
    Reshape [B*K,D,8,8] Per-layer:             Upsample
    Conv layers           + type_embed         → [B,K,C,ps,ps]
    Pool [B,K,D]          + RoPE on Q,K
                          Masked attention

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

    # Validate RoPE indices are within bounds
    max_y = y_pos.max().item()
    max_x = x_pos.max().item()
    if max_y >= max_pos or max_x >= max_pos:
        raise ValueError(
            f"apply_rope_2d: position indices out of bounds. "
            f"max_y={max_y}, max_x={max_x}, max_pos={max_pos}, "
            f"image_size={image_size}, coords range: [{coords.min().item()}, {coords.max().item()}]"
        )

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
    CNN encoder that processes feature patches through spatial convolutions.

    Takes [B, K, tokens, D] features (tokens = feature_grid_size^2) and produces:
    - encoded: [B, K, embed_dim] - pooled representation for attention
    - skips: dict of intermediate features at 16x16, 8x8, 4x4, 2x2 scales
    
    Supports feature_grid_size of 8 or 16.
    """

    def __init__(
        self,
        input_dim: int = 1024,
        embed_dim: int = 128,
        feature_grid_size: int = 16,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.feature_grid_size = feature_grid_size

        D = embed_dim

        # Project input features to working dimension
        self.input_proj = nn.Linear(input_dim, D)

        # Encoder levels with skip connections
        if feature_grid_size == 16:
            # Level 0: 16x16 → 16x16 (preserve resolution)
            self.enc0 = nn.Sequential(
                nn.Conv2d(D, D, 3, padding=1),
                nn.BatchNorm2d(D),
                nn.GELU(),
            )
            # Level 1: 16x16 → 8x8
            self.enc1 = nn.Sequential(
                nn.Conv2d(D, D, 3, stride=2, padding=1),
                nn.BatchNorm2d(D),
                nn.GELU(),
            )
            # Level 2: 8x8 → 4x4
            self.enc2 = nn.Sequential(
                nn.Conv2d(D, D, 3, stride=2, padding=1),
                nn.BatchNorm2d(D),
                nn.GELU(),
            )
            # Level 3: 4x4 → 2x2
            self.enc3 = nn.Sequential(
                nn.Conv2d(D, D, 3, stride=2, padding=1),
                nn.BatchNorm2d(D),
                nn.GELU(),
            )
        else:  # feature_grid_size == 8
            # Level 0: 8x8 → 8x8 (preserve resolution)
            self.enc0 = nn.Sequential(
                nn.Conv2d(D, D, 3, padding=1),
                nn.BatchNorm2d(D),
                nn.GELU(),
            )
            # Level 1: 8x8 → 4x4
            self.enc1 = nn.Sequential(
                nn.Conv2d(D, D, 3, stride=2, padding=1),
                nn.BatchNorm2d(D),
                nn.GELU(),
            )
            # Level 2: 4x4 → 2x2
            self.enc2 = nn.Sequential(
                nn.Conv2d(D, D, 3, stride=2, padding=1),
                nn.BatchNorm2d(D),
                nn.GELU(),
            )
            self.enc3 = None  # Not used for 8x8

        # Final pooling to [B*K, D, 1, 1]
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(
        self, features: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Args:
            features: [B, K, tokens, D_in] - patch features (tokens = h*w)

        Returns:
            encoded: [B, K, embed_dim] - pooled representation
            skips: dict with skip tensors at each scale
        """
        B, K, NF, E = features.shape
        D = self.embed_dim
        h = w = self.feature_grid_size
        
        """
        if NF != h * w:
            raise ValueError(
                f"Expected {h*w} tokens (feature_grid_size={h}), got {NF}. "
                f"Check that patch_feature_grid_size matches the actual feature extraction."
            )
        """
        # Project and reshape to spatial: [B*K, D, h, w]
        x = self.input_proj(features.reshape(-1, E))  # [B*K*NF, D]
        x = x.view(B * K, NF, D).permute(0, 2, 1)  # [B*K, D, NF]
        x = x.view(B * K, D, h, w)  # [B*K, D, h, w]

        if self.feature_grid_size == 16:
            # Encode with skip connections: 16→16→8→4→2
            skip_16x16 = self.enc0(x)      # [B*K, D, 16, 16]
            skip_8x8 = self.enc1(skip_16x16)  # [B*K, D, 8, 8]
            skip_4x4 = self.enc2(skip_8x8)    # [B*K, D, 4, 4]
            skip_2x2 = self.enc3(skip_4x4)    # [B*K, D, 2, 2]

            # Pool to [B, K, D]
            encoded = self.pool(skip_2x2).view(B, K, D)

            skips = {
                'skip_16x16': skip_16x16.view(B, K, D, 16, 16),
                'skip_8x8': skip_8x8.view(B, K, D, 8, 8),
                'skip_4x4': skip_4x4.view(B, K, D, 4, 4),
                'skip_2x2': skip_2x2.view(B, K, D, 2, 2),
            }
        else:  # 8x8
            # Encode with skip connections: 8→8→4→2
            skip_8x8 = self.enc0(x)        # [B*K, D, 8, 8]
            skip_4x4 = self.enc1(skip_8x8)    # [B*K, D, 4, 4]
            skip_2x2 = self.enc2(skip_4x4)    # [B*K, D, 2, 2]

            # Pool to [B, K, D]
            encoded = self.pool(skip_2x2).view(B, K, D)

            skips = {
                'skip_8x8': skip_8x8.view(B, K, D, 8, 8),
                'skip_4x4': skip_4x4.view(B, K, D, 4, 4),
                'skip_2x2': skip_2x2.view(B, K, D, 2, 2),
            }

        return encoded, skips


class AttentionBlock(nn.Module):
    """Single attention + MLP block with pre-norm, residuals, and per-layer embeddings."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        mlp_ratio: int = 4,
        append_zero_attn: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.embed_dim = embed_dim
        self.append_zero_attn = append_zero_attn

        # Per-layer type embeddings
        self.target_embed = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.context_embed = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

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
        is_context: torch.Tensor,
        coords: torch.Tensor | None = None,
        rope_cache: torch.Tensor | None = None,
        image_size: int = 224,
        num_registers: int = 0,
        return_attn_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Args:
            x: [B, K_total, D] - tokens (including registers)
            attn_mask: [B, H, K_total, K_total] - float attention mask (-inf for blocked)
            is_context: [B, K_total] - True for context/register tokens
            coords: [B, K, 2] - patch coordinates (y, x), without registers
            rope_cache: [max_pos, dim/4, 2] - precomputed RoPE sin/cos
            image_size: image size for RoPE coordinate normalization
            num_registers: number of register tokens prepended to sequence
            return_attn_weights: If True, compute and return attention weights

        Returns:
            x: [B, K_total, D] - updated tokens
            attn_weights: [B, H, K_total, K_total] or None
        """
        B, K_total, D = x.shape
        H = self.num_heads

        # Pre-norm
        x_normed = self.norm1(x)

        # Add per-layer type embeddings to normed input
        type_embed = torch.where(
            is_context.unsqueeze(-1),
            self.context_embed.expand(B, K_total, -1),
            self.target_embed.expand(B, K_total, -1),
        )
        x_normed = x_normed + type_embed

        # Project Q, K, V
        q = self.q_proj(x_normed)  # [B, K_total, D]
        k = self.k_proj(x_normed)  # [B, K_total, D]
        v = self.v_proj(x_normed)

        # Apply 2D RoPE to Q and K (non-register tokens only)
        if rope_cache is not None and coords is not None:
            if num_registers > 0:
                q_reg, q_patch = q[:, :num_registers], q[:, num_registers:]
                k_reg, k_patch = k[:, :num_registers], k[:, num_registers:]
                q_patch = apply_rope_2d(q_patch, coords, rope_cache, image_size)
                k_patch = apply_rope_2d(k_patch, coords, rope_cache, image_size)
                q = torch.cat([q_reg, q_patch], dim=1)
                k = torch.cat([k_reg, k_patch], dim=1)
            else:
                q = apply_rope_2d(q, coords, rope_cache, image_size)
                k = apply_rope_2d(k, coords, rope_cache, image_size)

        # Reshape for multi-head attention
        q = q.view(B, K_total, H, -1).transpose(1, 2)
        k = k.view(B, K_total, H, -1).transpose(1, 2)
        v = v.view(B, K_total, H, -1).transpose(1, 2)

        # Append zero token to keys/values — gives queries a "do nothing" option
        if self.append_zero_attn:
            zero_k = torch.zeros(B, H, 1, k.shape[-1], device=k.device, dtype=k.dtype)
            zero_v = torch.zeros(B, H, 1, v.shape[-1], device=v.device, dtype=v.dtype)
            k = torch.cat([k, zero_k], dim=2)  # [B, H, K_total+1, head_dim]
            v = torch.cat([v, zero_v], dim=2)
            # Extend mask: all queries can attend to the zero token (append 0.0 column)
            zero_col = torch.zeros(B, H, K_total, 1, device=attn_mask.device, dtype=attn_mask.dtype)
            attn_mask = torch.cat([attn_mask, zero_col], dim=-1)

        attn_weights = None
        if return_attn_weights:
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

        out = out.transpose(1, 2).reshape(B, K_total, D)
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
        append_zero_attn: bool = False,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_registers = num_registers
        self.image_size = image_size
        self.target_self_attention = target_self_attention
        self.gradient_checkpointing = gradient_checkpointing

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
            AttentionBlock(embed_dim, num_heads, dropout, append_zero_attn=append_zero_attn)
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

        # Add registers
        if self.register_tokens is not None:
            registers = self.register_tokens.expand(B, -1, -1)
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

        num_reg = self.num_registers if self.register_tokens is not None else 0

        # Run through attention layers (per-layer type embed + RoPE)
        all_attn_weights = []
        for layer in self.layers:
            if self.gradient_checkpointing and self.training and not return_attn_weights:
                # Use gradient checkpointing to save memory (recompute activations in backward)
                from torch.utils.checkpoint import checkpoint
                x, attn_w = checkpoint(
                    layer,
                    x, float_mask,
                    is_context_with_reg,
                    coords,
                    self.rope_cache,
                    self.image_size,
                    num_reg,
                    False,  # return_attn_weights must be False for checkpointing
                    use_reentrant=False,
                )
            else:
                x, attn_w = layer(
                    x, float_mask,
                    is_context=is_context_with_reg,
                    coords=coords,
                    rope_cache=self.rope_cache,
                    image_size=self.image_size,
                    num_registers=num_reg,
                    return_attn_weights=return_attn_weights,
                )
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
    
    Supports feature_grid_size of 8 or 16.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_classes: int = 1,
        patch_size: int = 16,
        feature_grid_size: int = 16,
        use_skip_connections: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.feature_grid_size = feature_grid_size
        self.use_skip_connections = use_skip_connections

        D = embed_dim
        fuse_ch_in = D * 2 if self.use_skip_connections else D

        # 1x1 → 2x2, fuse with skip_2x2
        self.up1 = nn.ConvTranspose2d(D, D, 2, stride=2)
        self.fuse1 = nn.Sequential(
            nn.Conv2d(fuse_ch_in, D, 3, padding=1),
            nn.BatchNorm2d(D),
            nn.GELU(),
        )

        # 2x2 → 4x4, fuse with skip_4x4
        self.up2 = nn.ConvTranspose2d(D, D, 2, stride=2)
        self.fuse2 = nn.Sequential(
            nn.Conv2d(fuse_ch_in, D, 3, padding=1),
            nn.BatchNorm2d(D),
            nn.GELU(),
        )

        # 4x4 → 8x8, fuse with skip_8x8
        self.up3 = nn.ConvTranspose2d(D, D, 2, stride=2)
        self.fuse3 = nn.Sequential(
            nn.Conv2d(fuse_ch_in, D, 3, padding=1),
            nn.BatchNorm2d(D),
            nn.GELU(),
        )

        if feature_grid_size == 16:
            # 8x8 → 16x16, fuse with skip_16x16
            self.up4 = nn.ConvTranspose2d(D, D, 2, stride=2)
            self.fuse4 = nn.Sequential(
                nn.Conv2d(fuse_ch_in, D, 3, padding=1),
                nn.BatchNorm2d(D),
                nn.GELU(),
            )
        else:
            self.up4 = None
            self.fuse4 = None

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
            skips: dict with skip tensors from encoder

        Returns:
            [B, K, num_classes, patch_size, patch_size]
        """
        B, K, D = encoded.shape
        h = self.feature_grid_size

        # Reshape encoded to [B*K, D, 1, 1] (use contiguous for after attention)
        x = encoded.contiguous().view(B * K, D, 1, 1)

        # 1x1 → 2x2 + skip
        x = self.up1(x)
        if self.use_skip_connections:
            skip_2x2 = skips['skip_2x2'].contiguous().view(B * K, D, 2, 2)
            x = torch.cat([x, skip_2x2], dim=1)
        x = self.fuse1(x)

        # 2x2 → 4x4 + skip
        x = self.up2(x)
        if self.use_skip_connections:
            skip_4x4 = skips['skip_4x4'].contiguous().view(B * K, D, 4, 4)
            x = torch.cat([x, skip_4x4], dim=1)
        x = self.fuse2(x)

        # 4x4 → 8x8 + skip
        x = self.up3(x)
        if self.use_skip_connections:
            skip_8x8 = skips['skip_8x8'].contiguous().view(B * K, D, 8, 8)
            x = torch.cat([x, skip_8x8], dim=1)
        x = self.fuse3(x)

        if self.feature_grid_size == 16:
            # 8x8 → 16x16 + skip
            x = self.up4(x)
            if self.use_skip_connections:
                skip_16x16 = skips['skip_16x16'].contiguous().view(B * K, D, 16, 16)
                x = torch.cat([x, skip_16x16], dim=1)
            x = self.fuse4(x)

        # Final conv to num_classes
        x = self.final_conv(x)  # [B*K, num_classes, h, h]

        # Upsample to patch_size if needed
        if h != self.patch_size:
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
        feature_grid_size: int = 16,
        target_self_attention: bool = False,
        dropout: float = 0.0,
        max_seq_len: int = 1024,
        decoder_use_skip_connections: bool = True,
        append_zero_attn: bool = False,
        max_levels: int = 4,
        gradient_checkpointing: bool = False,
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
            decoder_use_skip_connections: If True, use U-Net skips in decoder
            append_zero_attn: Append zero token to K/V for attention sink
            max_levels: Number of resolution levels for level embedding
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.image_size = image_size

        # Learnable level embedding (Deformable DETR style)
        self.level_embed = nn.Parameter(torch.randn(max_levels, embed_dim) * 0.02)

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
            gradient_checkpointing=gradient_checkpointing,
            dropout=dropout,
            append_zero_attn=append_zero_attn,
        )

        self.decoder = SimpleCNNDecoder(
            embed_dim=embed_dim,
            num_classes=num_classes,
            patch_size=patch_size,
            feature_grid_size=feature_grid_size,
            use_skip_connections=decoder_use_skip_connections,
        )

    def forward(
        self,
        img_patches: torch.Tensor,
        coords: torch.Tensor = None,
        ctx_id_labels: torch.Tensor = None,
        return_attn_weights: bool = False,
        level_idx: int = 0,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass: encode → attention → decode.

        Args:
            img_patches: [B, K, tokens, D] - features per patch
            coords: [B, K, 2] - Patch coordinates (y, x)
            ctx_id_labels: [B, K] - 0 for target, >0 for context
            return_attn_weights: If True, return attention weights and register tokens
            level_idx: Resolution level index for level embedding

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

        # Add level embedding (Deformable DETR style)
        encoded = encoded + self.level_embed[level_idx].view(1, 1, -1)

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
