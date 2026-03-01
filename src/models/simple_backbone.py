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
                          Full attention (Flash SDP)

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

Scale Encoding:
    Uses ContinuousScaleEncoding with log-sinusoidal functions instead of fixed
    level embeddings. Pass `resolution` (e.g., 8, 16, 32, 64) to forward() for
    resolution-agnostic multi-scale processing. Frequencies are learnable (SPE-style).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContinuousScaleEncoding(nn.Module):
    """Resolution-agnostic scale encoding using sinusoidal functions.

    Encodes continuous resolution values (e.g., 8, 16, 32, 64) into embeddings
    using log-spaced sinusoidal functions, similar to NeRF/Transformer PE.
    Supports learnable frequencies (SPE-style) for task-specific adaptation.
    """

    def __init__(self, embed_dim: int, learnable_freqs: bool = True):
        super().__init__()
        self.embed_dim = embed_dim
        half_dim = embed_dim // 2
        # Base frequencies (log-spaced like transformer PE)
        freqs = torch.exp(torch.arange(half_dim).float() * -(math.log(10000.0) / half_dim))
        if learnable_freqs:
            self.freqs = nn.Parameter(freqs)
        else:
            self.register_buffer('freqs', freqs)

    def forward(self, resolution: float, device: torch.device = None) -> torch.Tensor:
        """Encode resolution to embedding vector.

        Args:
            resolution: Current level resolution (e.g., 8, 16, 32, 64)
            device: Target device for the output tensor

        Returns:
            [embed_dim] scale embedding
        """
        # Log2 gives linear progression: 8→3, 16→4, 32→5, 64→6
        scale = math.log2(resolution)
        freqs = self.freqs if device is None else self.freqs.to(device)
        args = scale * freqs
        return torch.cat([torch.sin(args), torch.cos(args)])


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
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class MaskPriorEncoder(nn.Module):
    """SAM-style CNN to encode mask patches to embedding vectors.

    Architecture from SAM/MedSAM mask_downscaling:
    - Conv2d(1, 16, kernel=2, stride=2) + LayerNorm2d + GELU  → h/2
    - Conv2d(16, 32, kernel=2, stride=2) + LayerNorm2d + GELU → h/4
    - Conv2d(32, embed_dim, kernel=1)                         → h/4
    - AdaptiveAvgPool2d(1) → [B*K, embed_dim, 1, 1]
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
        x = self.mask_downscaling(x)  # [B*K, embed_dim, h/4, w/4]
        x = self.pool(x).view(B, K, -1)  # [B, K, embed_dim]
        return x


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
    half_dim = D // 2
    quarter_dim = D // 4

    max_pos = rope_cache.shape[0]
    coords_normalized = (coords.float() / image_size * (max_pos - 1)).clamp(0, max_pos - 1)

    y_pos = coords_normalized[:, :, 0].long()
    x_pos = coords_normalized[:, :, 1].long()

    y_rope = rope_cache[y_pos]  # [B, K, dim/4, 2]
    x_rope = rope_cache[x_pos]

    # Complex multiplication for RoPE rotation (cast to float32 for view_as_complex)
    orig_dtype = x.dtype
    x_part = x[:, :, :half_dim].float().reshape(B, K, quarter_dim, 2)
    y_part = x[:, :, half_dim:].float().reshape(B, K, quarter_dim, 2)
    x_rope = x_rope.float()
    y_rope = y_rope.float()

    x_complex = torch.view_as_complex(x_part.contiguous())
    x_rope_complex = torch.view_as_complex(x_rope.contiguous())
    x_rotated = torch.view_as_real(x_complex * x_rope_complex).reshape(B, K, half_dim)

    y_complex = torch.view_as_complex(y_part.contiguous())
    y_rope_complex = torch.view_as_complex(y_rope.contiguous())
    y_rotated = torch.view_as_real(y_complex * y_rope_complex).reshape(B, K, half_dim)

    return torch.cat([x_rotated, y_rotated], dim=-1).to(orig_dtype)


class ResolutionConditionedNorm(nn.Module):
    """GroupNorm with scale/shift predicted from continuous resolution embedding.

    Implements FiLM (Feature-wise Linear Modulation):
        out = γ(resolution) × GroupNorm(x) + β(resolution)

    This allows the normalization to adapt to any resolution, enabling:
    - Generalization to unseen resolutions at test time
    - Partial gradient isolation between resolutions (different γ/β paths)
    - Smooth interpolation in resolution space
    """

    def __init__(self, num_channels: int, scale_embed_dim: int, num_groups: int = 8):
        super().__init__()
        # GroupNorm without learnable affine params (we predict them)
        self.norm = nn.GroupNorm(num_groups, num_channels, affine=False)

        # Predict gamma/beta from resolution embedding
        self.gamma_proj = nn.Linear(scale_embed_dim, num_channels)
        self.beta_proj = nn.Linear(scale_embed_dim, num_channels)

        # Initialize to identity transform (gamma=1, beta=0)
        nn.init.ones_(self.gamma_proj.bias)
        nn.init.zeros_(self.gamma_proj.weight)
        nn.init.zeros_(self.beta_proj.weight)
        nn.init.zeros_(self.beta_proj.bias)

    def forward(self, x: torch.Tensor, scale_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B*K, C, H, W] - features to normalize
            scale_embed: [scale_embed_dim] - continuous resolution embedding

        Returns:
            Normalized and modulated features [B*K, C, H, W]
        """
        x = self.norm(x)

        # Predict per-channel scale and shift from resolution
        gamma = self.gamma_proj(scale_embed).view(1, -1, 1, 1)  # [1, C, 1, 1]
        beta = self.beta_proj(scale_embed).view(1, -1, 1, 1)

        return gamma * x + beta


class SimpleCNNEncoder(nn.Module):
    """
    CNN encoder that processes feature patches through spatial convolutions.

    Takes [B, K, tokens, D] features (tokens = feature_grid_size^2) and produces:
    - encoded: [B, K, embed_dim] - pooled representation for attention
    - skips: dict of intermediate features at 16x16, 8x8, 4x4, 2x2 scales

    Supports feature_grid_size of 8 or 16.
    Uses ResolutionConditionedNorm for resolution-agnostic multi-scale training.
    """

    def __init__(
        self,
        input_dim: int = 1024,
        embed_dim: int = 128,
        feature_grid_size: int = 16,
        scale_embed_dim: int | None = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.feature_grid_size = feature_grid_size
        # Default scale_embed_dim to embed_dim for backward compatibility
        self.scale_embed_dim = scale_embed_dim if scale_embed_dim is not None else embed_dim

        D = embed_dim

        # Project input features to working dimension
        self.input_proj = nn.Linear(input_dim, D)

        # Encoder levels with resolution-conditioned normalization
        if feature_grid_size == 16:
            # Level 0: 16x16 → 16x16 (preserve resolution)
            self.enc0_conv = nn.Conv2d(D, D, 3, padding=1)
            self.enc0_norm = ResolutionConditionedNorm(D, self.scale_embed_dim)
            # Level 1: 16x16 → 8x8
            self.enc1_conv = nn.Conv2d(D, D, 3, stride=2, padding=1)
            self.enc1_norm = ResolutionConditionedNorm(D, self.scale_embed_dim)
            # Level 2: 8x8 → 4x4
            self.enc2_conv = nn.Conv2d(D, D, 3, stride=2, padding=1)
            self.enc2_norm = ResolutionConditionedNorm(D, self.scale_embed_dim)
            # Level 3: 4x4 → 2x2
            self.enc3_conv = nn.Conv2d(D, D, 3, stride=2, padding=1)
            self.enc3_norm = ResolutionConditionedNorm(D, self.scale_embed_dim)
        else:  # feature_grid_size == 8
            # Level 0: 8x8 → 8x8 (preserve resolution)
            self.enc0_conv = nn.Conv2d(D, D, 3, padding=1)
            self.enc0_norm = ResolutionConditionedNorm(D, self.scale_embed_dim)
            # Level 1: 8x8 → 4x4
            self.enc1_conv = nn.Conv2d(D, D, 3, stride=2, padding=1)
            self.enc1_norm = ResolutionConditionedNorm(D, self.scale_embed_dim)
            # Level 2: 4x4 → 2x2
            self.enc2_conv = nn.Conv2d(D, D, 3, stride=2, padding=1)
            self.enc2_norm = ResolutionConditionedNorm(D, self.scale_embed_dim)
            self.enc3_conv = None
            self.enc3_norm = None

        # Final pooling to [B*K, D, 1, 1]
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(
        self, features: torch.Tensor, scale_embed: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Args:
            features: [B, K, tokens, D_in] - patch features (tokens = h*w)
            scale_embed: [scale_embed_dim] - continuous resolution embedding.
                If None, uses zeros (backward compatibility, not recommended).

        Returns:
            encoded: [B, K, embed_dim] - pooled representation
            skips: dict with skip tensors at each scale
        """
        B, K, NF, E = features.shape
        D = self.embed_dim
        h = w = self.feature_grid_size

        # Default scale_embed for backward compatibility
        if scale_embed is None:
            scale_embed = torch.zeros(self.scale_embed_dim, device=features.device)

        # Project and reshape to spatial: [B*K, D, h, w]
        x = self.input_proj(features.reshape(-1, E))  # [B*K*NF, D]
        x = x.view(B * K, NF, D).permute(0, 2, 1)  # [B*K, D, NF]
        x = x.view(B * K, D, h, w)  # [B*K, D, h, w]

        if self.feature_grid_size == 16:
            # Encode with skip connections: 16→16→8→4→2
            x = self.enc0_conv(x)
            skip_16x16 = F.gelu(self.enc0_norm(x, scale_embed))

            x = self.enc1_conv(skip_16x16)
            skip_8x8 = F.gelu(self.enc1_norm(x, scale_embed))

            x = self.enc2_conv(skip_8x8)
            skip_4x4 = F.gelu(self.enc2_norm(x, scale_embed))

            x = self.enc3_conv(skip_4x4)
            skip_2x2 = F.gelu(self.enc3_norm(x, scale_embed))

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
            x = self.enc0_conv(x)
            skip_8x8 = F.gelu(self.enc0_norm(x, scale_embed))

            x = self.enc1_conv(skip_8x8)
            skip_4x4 = F.gelu(self.enc1_norm(x, scale_embed))

            x = self.enc2_conv(skip_4x4)
            skip_2x2 = F.gelu(self.enc2_norm(x, scale_embed))

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

        attn_weights = None
        if return_attn_weights:
            scale = 1.0 / math.sqrt(q.shape[-1])
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            attn_weights = F.softmax(attn_scores, dim=-1)
            if self.training and self.dropout.p > 0:
                attn_weights = self.dropout(attn_weights)
            out = torch.matmul(attn_weights, v)
        else:
            out = F.scaled_dot_product_attention(
                q, k, v,
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
    Multi-layer full attention module for cross-patch communication.

    Uses unmasked bidirectional attention (enables Flash SDP backend).
    Per-layer type embeddings distinguish context vs target tokens.
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
        num_context_layers: int = 0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_registers = num_registers
        self.image_size = image_size
        self.gradient_checkpointing = gradient_checkpointing
        self.num_context_layers = num_context_layers

        # Register tokens for global context
        if num_registers > 0:
            self.register_tokens = nn.Parameter(
                torch.randn(1, num_registers, embed_dim) * 0.02
            )
        else:
            self.register_tokens = None

        # Context-first self-attention: context patches attend to each other
        # before the joint stage, so targets see pre-enriched context.
        # Both stages are fully bidirectional (Flash SDP compatible).
        if num_context_layers > 0:
            self.context_layers = nn.ModuleList([
                AttentionBlock(embed_dim, num_heads, dropout, append_zero_attn=append_zero_attn)
                for _ in range(num_context_layers)
            ])
            if num_registers > 0:
                self.context_registers = nn.Parameter(
                    torch.randn(1, num_registers, embed_dim) * 0.02
                )
            else:
                self.context_registers = None
        else:
            self.context_layers = None
            self.context_registers = None

        # RoPE cache
        rope_cache = build_rope_cache_2d(max_seq_len, embed_dim)
        self.register_buffer("rope_cache", rope_cache, persistent=False)

        # Stack of attention blocks (joint stage)
        self.layers = nn.ModuleList([
            AttentionBlock(embed_dim, num_heads, dropout, append_zero_attn=append_zero_attn)
            for _ in range(num_layers)
        ])

        # Final layer norm
        self.final_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        coords: torch.Tensor,
        is_context: torch.Tensor,
        return_attn_weights: bool = False,
        num_target_patches: int | None = None,
    ) -> tuple[torch.Tensor, dict | None]:
        """
        Args:
            x: [B, K, D] - patch tokens (target first, then context)
            coords: [B, K, 2] - patch coordinates (y, x)
            is_context: [B, K] - True for context patches
            return_attn_weights: If True, return attention weights and registers
            num_target_patches: Number of target patches (first N tokens).
                Required for context-first attention; if None, context stage is skipped.

        Returns:
            x: [B, K, D] - updated tokens
            extras: dict with 'attn_weights' and 'register_tokens' if requested, else None
        """
        B, K, D = x.shape
        device = x.device

        # Stage 1: Context-first self-attention
        # Context patches attend to each other before the joint stage,
        # so targets see pre-enriched context representations.
        if (self.context_layers is not None
                and num_target_patches is not None
                and num_target_patches < K):
            K_t = num_target_patches
            ctx_x = x[:, K_t:]           # [B, K_ctx, D]
            ctx_coords = coords[:, K_t:]  # [B, K_ctx, 2]

            # Add context-stage registers
            num_ctx_reg = 0
            if self.context_registers is not None:
                ctx_regs = self.context_registers.expand(B, -1, -1)
                ctx_x = torch.cat([ctx_regs, ctx_x], dim=1)
                num_ctx_reg = self.context_registers.shape[1]

            # All tokens in this stage are context (registers + context patches)
            ctx_is_context = torch.ones(
                B, ctx_x.shape[1], dtype=torch.bool, device=device
            )

            for layer in self.context_layers:
                if self.gradient_checkpointing and self.training:
                    from torch.utils.checkpoint import checkpoint
                    ctx_x, _ = checkpoint(
                        layer, ctx_x, ctx_is_context, ctx_coords,
                        self.rope_cache, self.image_size, num_ctx_reg, False,
                        use_reentrant=False,
                    )
                else:
                    ctx_x, _ = layer(
                        ctx_x, is_context=ctx_is_context, coords=ctx_coords,
                        rope_cache=self.rope_cache, image_size=self.image_size,
                        num_registers=num_ctx_reg, return_attn_weights=False,
                    )

            # Remove context registers, reassemble [target, enriched_context]
            if num_ctx_reg > 0:
                ctx_x = ctx_x[:, num_ctx_reg:]
            x = torch.cat([x[:, :K_t], ctx_x], dim=1)

        # Stage 2: Joint attention (all patches together)
        # Add registers
        if self.register_tokens is not None:
            registers = self.register_tokens.expand(B, -1, -1)
            x = torch.cat([registers, x], dim=1)  # [B, R+K, D]
            reg_mask = torch.ones(B, self.num_registers, dtype=torch.bool, device=device)
            is_context_with_reg = torch.cat([reg_mask, is_context], dim=1)
        else:
            is_context_with_reg = is_context

        num_reg = self.num_registers if self.register_tokens is not None else 0

        # Run through attention layers (per-layer type embed + RoPE)
        all_attn_weights = []
        for layer in self.layers:
            if self.gradient_checkpointing and self.training and not return_attn_weights:
                from torch.utils.checkpoint import checkpoint
                x, attn_w = checkpoint(
                    layer,
                    x,
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
                    x,
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

        # Always capture register tokens (needed for learned alpha)
        register_tokens_out = None
        if self.register_tokens is not None:
            register_tokens_out = x[:, :self.num_registers].clone()

        # Remove registers
        if self.register_tokens is not None:
            x = x[:, self.num_registers:]

        # Always return extras dict with register_tokens
        extras = {
            'attn_weights': all_attn_weights if return_attn_weights else None,
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
    Uses ResolutionConditionedNorm for resolution-agnostic multi-scale training.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_classes: int = 1,
        patch_size: int = 16,
        feature_grid_size: int = 16,
        use_skip_connections: bool = True,
        predict_confidence: bool = False,
        scale_embed_dim: int | None = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.feature_grid_size = feature_grid_size
        self.use_skip_connections = use_skip_connections
        self.predict_confidence = predict_confidence
        self.scale_embed_dim = scale_embed_dim if scale_embed_dim is not None else embed_dim

        D = embed_dim
        fuse_ch_in = D * 2 if self.use_skip_connections else D

        # 1x1 → 2x2, fuse with skip_2x2
        self.up1 = nn.ConvTranspose2d(D, D, 2, stride=2)
        self.fuse1_conv = nn.Conv2d(fuse_ch_in, D, 3, padding=1)
        self.fuse1_norm = ResolutionConditionedNorm(D, self.scale_embed_dim)

        # 2x2 → 4x4, fuse with skip_4x4
        self.up2 = nn.ConvTranspose2d(D, D, 2, stride=2)
        self.fuse2_conv = nn.Conv2d(fuse_ch_in, D, 3, padding=1)
        self.fuse2_norm = ResolutionConditionedNorm(D, self.scale_embed_dim)

        # 4x4 → 8x8, fuse with skip_8x8
        self.up3 = nn.ConvTranspose2d(D, D, 2, stride=2)
        self.fuse3_conv = nn.Conv2d(fuse_ch_in, D, 3, padding=1)
        self.fuse3_norm = ResolutionConditionedNorm(D, self.scale_embed_dim)

        if feature_grid_size == 16:
            # 8x8 → 16x16, fuse with skip_16x16
            self.up4 = nn.ConvTranspose2d(D, D, 2, stride=2)
            self.fuse4_conv = nn.Conv2d(fuse_ch_in, D, 3, padding=1)
            self.fuse4_norm = ResolutionConditionedNorm(D, self.scale_embed_dim)
        else:
            self.up4 = None
            self.fuse4_conv = None
            self.fuse4_norm = None

        # Segmentation head
        self.seg_head = nn.Sequential(
            nn.Conv2d(D, D // 2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(D // 2, num_classes, 1),
        )

        # Confidence head (predicts pixel-level confidence)
        if predict_confidence:
            self.conf_head = nn.Sequential(
                nn.Conv2d(D, D // 2, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(D // 2, 1, 1),  # Single channel confidence
            )

    def _decode_trunk(
        self,
        encoded: torch.Tensor,
        skips: dict[str, torch.Tensor],
        scale_embed: torch.Tensor,
    ) -> torch.Tensor:
        """Shared decoder trunk that produces features at feature_grid_size resolution.

        Args:
            encoded: [B*K, D, 1, 1] - attention output reshaped
            skips: dict with skip tensors from encoder
            scale_embed: [scale_embed_dim] - continuous resolution embedding

        Returns:
            [B*K, D, feature_grid_size, feature_grid_size] - decoded features
        """
        B_K, D = encoded.shape[0], encoded.shape[1]

        x = encoded

        # 1x1 → 2x2 + skip
        x = self.up1(x)
        if self.use_skip_connections:
            skip_2x2 = skips['skip_2x2'].view(B_K, D, 2, 2)
            x = torch.cat([x, skip_2x2], dim=1)
        x = self.fuse1_conv(x)
        x = F.gelu(self.fuse1_norm(x, scale_embed))

        # 2x2 → 4x4 + skip
        x = self.up2(x)
        if self.use_skip_connections:
            skip_4x4 = skips['skip_4x4'].view(B_K, D, 4, 4)
            x = torch.cat([x, skip_4x4], dim=1)
        x = self.fuse2_conv(x)
        x = F.gelu(self.fuse2_norm(x, scale_embed))

        # 4x4 → 8x8 + skip
        x = self.up3(x)
        if self.use_skip_connections:
            skip_8x8 = skips['skip_8x8'].view(B_K, D, 8, 8)
            x = torch.cat([x, skip_8x8], dim=1)
        x = self.fuse3_conv(x)
        x = F.gelu(self.fuse3_norm(x, scale_embed))

        if self.feature_grid_size == 16:
            # 8x8 → 16x16 + skip
            x = self.up4(x)
            if self.use_skip_connections:
                skip_16x16 = skips['skip_16x16'].view(B_K, D, 16, 16)
                x = torch.cat([x, skip_16x16], dim=1)
            x = self.fuse4_conv(x)
            x = F.gelu(self.fuse4_norm(x, scale_embed))

        return x  # [B*K, D, h, h]

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
            scale_embed: [scale_embed_dim] - continuous resolution embedding.
                If None, uses zeros (backward compatibility, not recommended).

        Returns:
            seg_pred: [B, K, num_classes, patch_size, patch_size]
            conf_pred: [B, K, 1, patch_size, patch_size] or None if predict_confidence=False
        """
        B, K, D = encoded.shape
        h = self.feature_grid_size

        # Default scale_embed for backward compatibility
        if scale_embed is None:
            scale_embed = torch.zeros(self.scale_embed_dim, device=encoded.device)

        # Reshape encoded to [B*K, D, 1, 1] (use contiguous for after attention)
        x = encoded.contiguous().view(B * K, D, 1, 1)

        # Make skips contiguous for the trunk
        skips_contiguous = {k: v.contiguous() for k, v in skips.items()}

        # Shared decoder trunk with resolution conditioning
        features = self._decode_trunk(x, skips_contiguous, scale_embed)  # [B*K, D, h, h]

        # Segmentation head
        seg_pred = self.seg_head(features)  # [B*K, num_classes, h, h]

        # Upsample segmentation to patch_size if needed
        if h != self.patch_size:
            seg_pred = F.interpolate(
                seg_pred, size=(self.patch_size, self.patch_size),
                mode='bilinear', align_corners=False
            )

        seg_pred = seg_pred.view(B, K, self.num_classes, self.patch_size, self.patch_size)

        # Confidence head
        conf_pred = None
        if self.predict_confidence:
            conf_logits = self.conf_head(features)  # [B*K, 1, h, h]
            conf_pred = torch.sigmoid(conf_logits)  # [0, 1] bounded

            # Upsample confidence to patch_size if needed
            if h != self.patch_size:
                conf_pred = F.interpolate(
                    conf_pred, size=(self.patch_size, self.patch_size),
                    mode='bilinear', align_corners=False
                )

            conf_pred = conf_pred.view(B, K, 1, self.patch_size, self.patch_size)

        return seg_pred, conf_pred


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
        max_levels: int = 4,  # Deprecated, kept for backward compatibility
        gradient_checkpointing: bool = False,
        use_mask_prior: bool = False,
        mask_fusion_type: str = "additive",
        predict_confidence: bool = False,
        **kwargs,
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
            target_self_attention: Allow targets to attend to each other (ignored, full attention)
            dropout: Attention dropout rate
            max_seq_len: Maximum sequence length for RoPE cache
            decoder_use_skip_connections: If True, use U-Net skips in decoder
            append_zero_attn: Append zero token to K/V for attention sink
            max_levels: Deprecated, no longer used. Scale encoding is now resolution-agnostic.
            use_mask_prior: If True, fuse mask prior from previous level into encoded features
            mask_fusion_type: How to fuse mask prior ("additive", "gated", or "concat")
            predict_confidence: If True, predict pixel-level confidence alongside segmentation
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.image_size = image_size
        self.use_mask_prior = use_mask_prior
        self.use_context_mask = kwargs.get('use_context_mask', False)
        self.mask_fusion_type = mask_fusion_type
        self.predict_confidence = predict_confidence

        # Continuous scale encoding (resolution-agnostic, replaces fixed level_embed)
        self.scale_encoder = ContinuousScaleEncoding(embed_dim, learnable_freqs=True)

        self.encoder = SimpleCNNEncoder(
            input_dim=input_dim,
            embed_dim=embed_dim,
            feature_grid_size=feature_grid_size,
            scale_embed_dim=embed_dim,  # Match scale_encoder output
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
            num_context_layers=kwargs.get('num_context_layers', 0),
        )

        self.decoder = SimpleCNNDecoder(
            embed_dim=embed_dim,
            num_classes=num_classes,
            patch_size=patch_size,
            feature_grid_size=feature_grid_size,
            use_skip_connections=decoder_use_skip_connections,
            predict_confidence=predict_confidence,
            scale_embed_dim=embed_dim,  # Match scale_encoder output
        )

        # Mask encoder and fusion (shared for target prior and context masks)
        if use_mask_prior or self.use_context_mask:
            self.mask_encoder = MaskPriorEncoder(embed_dim, feature_grid_size)
            if mask_fusion_type == "gated":
                self.mask_gate = nn.Parameter(torch.tensor(0.1))  # init small for target
                if self.use_context_mask:
                    self.context_mask_gate = nn.Parameter(torch.tensor(0.1))  # separate gate for context
            elif mask_fusion_type == "concat":
                self.fusion_proj = nn.Linear(embed_dim * 2, embed_dim)

    def forward(
        self,
        img_patches: torch.Tensor,
        coords: torch.Tensor = None,
        ctx_id_labels: torch.Tensor = None,
        return_attn_weights: bool = False,
        level_idx: int = 0,
        resolution: float | None = None,
        num_target_patches: int | None = None,
        mask_prior_patches: torch.Tensor | None = None,
        context_mask_patches: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass: encode → attention → decode.

        Args:
            img_patches: [B, K, tokens, D] - features per patch
            coords: [B, K, 2] - Patch coordinates (y, x)
            ctx_id_labels: [B, K] - 0 for target, >0 for context
            return_attn_weights: If True, return attention weights and register tokens
            level_idx: Deprecated, use resolution instead. Fallback level index.
            resolution: Current level resolution (e.g., 8, 16, 32, 64).
                If None, computed as patch_size * 2^level_idx.
            num_target_patches: Number of target patches (first N tokens).
                Enables context-first attention when num_context_layers > 0.
            mask_prior_patches: [B, K_target, 1, h, h] - mask prior patches from previous level
            context_mask_patches: [B, K_ctx, 1, h, h] - GT context mask patches

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

        # Compute continuous scale embedding (resolution-agnostic)
        if resolution is None:
            resolution = float(self.patch_size * (2 ** level_idx))
        scale_embed = self.scale_encoder(resolution, device=device)  # [embed_dim]

        # Encode with resolution conditioning
        encoded, skips = self.encoder(img_patches, scale_embed)

        # Also add scale embedding to encoded features (additive, as before)
        encoded = encoded + scale_embed.view(1, 1, -1)

        # Fuse mask prior into target patch embeddings (before attention)
        K_target = num_target_patches if num_target_patches is not None else K
        if self.use_mask_prior and mask_prior_patches is not None:
            mask_encoded = self.mask_encoder(mask_prior_patches)  # [B, K_target, embed_dim]

            if self.mask_fusion_type == "additive":
                # SAM-style additive fusion
                encoded[:, :K_target] = encoded[:, :K_target] + mask_encoded
            elif self.mask_fusion_type == "gated":
                # Gated additive (learnable scale, inspired by MAIS)
                gate = torch.sigmoid(self.mask_gate)
                encoded[:, :K_target] = encoded[:, :K_target] + gate * mask_encoded
            elif self.mask_fusion_type == "concat":
                # Concatenation + projection
                fused = torch.cat([encoded[:, :K_target], mask_encoded], dim=-1)
                encoded[:, :K_target] = self.fusion_proj(fused)

        # Fuse GT context masks into context patch embeddings (before attention)
        if self.use_context_mask and context_mask_patches is not None:
            ctx_mask_encoded = self.mask_encoder(context_mask_patches)  # [B, K_ctx, embed_dim]

            if self.mask_fusion_type == "additive":
                encoded[:, K_target:] = encoded[:, K_target:] + ctx_mask_encoded
            elif self.mask_fusion_type == "gated":
                # Use separate gate for context masks
                ctx_gate = torch.sigmoid(self.context_mask_gate)
                encoded[:, K_target:] = encoded[:, K_target:] + ctx_gate * ctx_mask_encoded
            elif self.mask_fusion_type == "concat":
                # Share fusion_proj for simplicity
                ctx_fused = torch.cat([encoded[:, K_target:], ctx_mask_encoded], dim=-1)
                encoded[:, K_target:] = self.fusion_proj(ctx_fused)

        # Cross-patch attention
        attended, extras = self.attention(
            encoded, coords, is_context, return_attn_weights,
            num_target_patches=num_target_patches,
        )

        # Decode with resolution conditioning
        seg_pred, conf_pred = self.decoder(attended, skips, scale_embed)

        result = {
            'mask_patch_logit_preds': seg_pred,
        }

        if conf_pred is not None:
            result['confidence_preds'] = conf_pred

        if extras is not None:
            result['attn_weights'] = extras['attn_weights']
            result['register_tokens'] = extras['register_tokens']

        return result
