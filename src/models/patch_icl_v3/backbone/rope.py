"""2D Rotary Position Embedding (RoPE) utilities.

Implements RoPE for 2D spatial positions by splitting dimensions
between x and y coordinates.
"""
from __future__ import annotations

import torch


def build_rope_cache_2d(
    max_pos: int, dim: int, base: float = 10000.0
) -> torch.Tensor:
    """Precompute 2D RoPE sin/cos cache for spatial positions.

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
    freqs = torch.einsum("i,j->ij", positions, theta)
    return torch.stack([freqs.cos(), freqs.sin()], dim=-1)


def apply_rope_2d(
    x: torch.Tensor,
    coords: torch.Tensor,
    rope_cache: torch.Tensor,
    image_size: int,
) -> torch.Tensor:
    """Apply 2D rotary position embedding based on spatial coordinates.

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
    coords_normalized = (
        coords.float() / image_size * (max_pos - 1)
    ).clamp(0, max_pos - 1)

    y_pos = coords_normalized[:, :, 0].long()
    x_pos = coords_normalized[:, :, 1].long()

    y_rope = rope_cache[y_pos]
    x_rope = rope_cache[x_pos]

    # Complex multiplication for RoPE rotation (cast to float32)
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
