"""Attention modules for PatchICL backbone.

Implements multi-head self-attention with RoPE and optional context-first attention.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rope import apply_rope_2d, build_rope_cache_2d


class TransformerBlock(nn.Module):
    """Single attention + MLP block with pre-norm and residuals."""

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

        # Attention
        self.norm1 = nn.LayerNorm(embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # MLP
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
            rope_cache: [max_pos, dim/4, 2] - precomputed RoPE
            image_size: Image size for RoPE normalization
            num_registers: Number of register tokens prepended
            return_attn_weights: Whether to return attention weights

        Returns:
            x: [B, K_total, D] - updated tokens
            attn_weights: [B, H, K_total, K_total] or None
        """
        B, K_total, D = x.shape
        H = self.num_heads

        # Pre-norm
        x_normed = self.norm1(x)

        # Add per-layer type embeddings
        type_embed = torch.where(
            is_context.unsqueeze(-1),
            self.context_embed.expand(B, K_total, -1),
            self.target_embed.expand(B, K_total, -1),
        )
        x_normed = x_normed + type_embed

        # Project Q, K, V
        q = self.q_proj(x_normed)
        k = self.k_proj(x_normed)
        v = self.v_proj(x_normed)

        # Apply 2D RoPE to non-register tokens
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

        # Append zero attention token
        if self.append_zero_attn:
            zero_k = torch.zeros(B, H, 1, k.shape[-1], device=k.device, dtype=k.dtype)
            zero_v = torch.zeros(B, H, 1, v.shape[-1], device=v.device, dtype=v.dtype)
            k = torch.cat([k, zero_k], dim=2)
            v = torch.cat([v, zero_v], dim=2)

        # Compute attention
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
    """Multi-layer attention for cross-patch communication.

    Uses bidirectional attention (Flash SDP compatible).
    Optional context-first stage for enriching context before joint attention.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 1,
        num_registers: int = 4,
        image_size: int = 224,
        max_seq_len: int = 1024,
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

        # Register tokens
        if num_registers > 0:
            self.register_tokens = nn.Parameter(
                torch.randn(1, num_registers, embed_dim) * 0.02
            )
        else:
            self.register_tokens = None

        # Context-first layers
        if num_context_layers > 0:
            self.context_layers = nn.ModuleList([
                TransformerBlock(embed_dim, num_heads, dropout, append_zero_attn=append_zero_attn)
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

        # Main attention layers
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, dropout, append_zero_attn=append_zero_attn)
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        coords: torch.Tensor,
        is_context: torch.Tensor,
        return_attn_weights: bool = False,
        num_target_patches: int | None = None,
        prev_register_tokens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict | None]:
        """
        Args:
            x: [B, K, D] - patch tokens (target first, then context)
            coords: [B, K, 2] - patch coordinates
            is_context: [B, K] - True for context patches
            return_attn_weights: Whether to return attention weights
            num_target_patches: Number of target patches for context-first
            prev_register_tokens: [B, R, D] - registers from previous level

        Returns:
            x: [B, K, D] - updated tokens
            extras: dict with attn_weights and register_tokens
        """
        B, K, D = x.shape
        device = x.device

        # Stage 1: Context-first attention
        if (self.context_layers is not None
                and num_target_patches is not None
                and num_target_patches < K):
            K_t = num_target_patches
            ctx_x = x[:, K_t:]
            ctx_coords = coords[:, K_t:]

            # Add context registers
            num_ctx_reg = 0
            if self.context_registers is not None:
                ctx_regs = self.context_registers.expand(B, -1, -1)
                ctx_x = torch.cat([ctx_regs, ctx_x], dim=1)
                num_ctx_reg = self.context_registers.shape[1]

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
                        ctx_x, ctx_is_context, ctx_coords,
                        self.rope_cache, self.image_size, num_ctx_reg, False,
                    )

            # Remove context registers
            if num_ctx_reg > 0:
                ctx_x = ctx_x[:, num_ctx_reg:]
            x = torch.cat([x[:, :K_t], ctx_x], dim=1)

        # Stage 2: Joint attention
        num_fresh_reg = self.num_registers if self.register_tokens is not None else 0
        if self.register_tokens is not None:
            registers = self.register_tokens.expand(B, -1, -1)
            x = torch.cat([registers, x], dim=1)
            reg_mask = torch.ones(B, num_fresh_reg, dtype=torch.bool, device=device)
            is_context_with_reg = torch.cat([reg_mask, is_context], dim=1)
        else:
            is_context_with_reg = is_context

        # Insert cascade tokens from previous level
        num_cascade = 0
        if prev_register_tokens is not None:
            num_cascade = prev_register_tokens.shape[1]
            cascade_mask = torch.ones(B, num_cascade, dtype=torch.bool, device=device)
            x = torch.cat([
                x[:, :num_fresh_reg],
                prev_register_tokens,
                x[:, num_fresh_reg:]
            ], dim=1)
            is_context_with_reg = torch.cat([
                is_context_with_reg[:, :num_fresh_reg],
                cascade_mask,
                is_context_with_reg[:, num_fresh_reg:],
            ], dim=1)

        num_reg_total = num_fresh_reg + num_cascade

        # Run through attention layers
        all_attn_weights = []
        for layer in self.layers:
            if self.gradient_checkpointing and self.training and not return_attn_weights:
                from torch.utils.checkpoint import checkpoint
                x, attn_w = checkpoint(
                    layer, x, is_context_with_reg, coords,
                    self.rope_cache, self.image_size, num_reg_total, False,
                    use_reentrant=False,
                )
            else:
                x, attn_w = layer(
                    x, is_context_with_reg, coords,
                    self.rope_cache, self.image_size, num_reg_total, return_attn_weights,
                )
            if attn_w is not None:
                all_attn_weights.append(attn_w)

        x = self.final_norm(x)

        # Extract register tokens
        register_tokens_out = None
        if self.register_tokens is not None:
            register_tokens_out = x[:, :num_fresh_reg].clone()

        # Remove all register-like tokens
        if num_reg_total > 0:
            x = x[:, num_reg_total:]

        extras = {
            'attn_weights': all_attn_weights if return_attn_weights else None,
            'register_tokens': register_tokens_out,
        }

        return x, extras
