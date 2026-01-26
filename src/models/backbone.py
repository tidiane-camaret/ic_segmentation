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
    Treats tgt/ctx patches as val/train examples.
    - val patches attend to train patches
    - patch-level input features:
        - img: pre-computed DINOv3 features
        - mask: random coloring in RGB space
    - embeddings: positional (RoPE), target vs context
    - token masking for img and mask inputs
    """

    def __init__(
        self,
        embed_dim: int = 1024,
        embed_proj_dim: int = 52,
        nb_features_per_patch: int = 49,
        patch_size: int = 16,
        image_size: int = 224,
        num_classes: int = 1,
        dropout: float = 0.1,
        max_seq_len: int = 1024,
        use_rope_2d: bool = True,
        num_registers: int = 4,
    ):
        """
        Args:
            embed_dim: Embedding dimension (matches DINO feature dim, default 1024)
            image_size: Original image size (for position embeddings)
            patch_size: Output patch size for segmentation
            num_classes: Number of segmentation classes
            dropout: Dropout rate
            max_seq_len: Maximum sequence length for RoPE cache
            use_rope_2d: If True, use 2D RoPE based on spatial coords. If False, use 1D sequential RoPE.
            num_registers: Number of register tokens (0 to disable). Registers act as global
                information aggregators and attention sinks, treated as context tokens.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.embed_proj_dim = embed_proj_dim
        self.nb_features_per_patch = nb_features_per_patch
        self.patch_size = patch_size
        self.image_size = image_size
        self.num_classes = num_classes
        self.dropout = dropout
        self.use_rope_2d = use_rope_2d
        self.num_registers = num_registers

        self.token_dim = nb_features_per_patch * embed_proj_dim

        # Target/context type embeddings
        self.target_embed = nn.Parameter(torch.randn(1, 1, self.token_dim) * 0.02)
        self.context_embed = nn.Parameter(torch.randn(1, 1, self.token_dim) * 0.02)

        # Learnable mask token for masked inputs (applied in embed_dim space)
        self.mask_token = nn.Parameter(torch.randn(1, 1, self.token_dim) * 0.02)

        # Register tokens (global tokens that act as attention sinks and info aggregators)
        # Treated as context tokens: participate in context self-attention,
        # visible to target cross-attention
        if num_registers > 0:
            self.register_tokens = nn.Parameter(
                torch.randn(1, num_registers, self.token_dim) * 0.02
            )
        else:
            self.register_tokens = None

        # RoPE cache (registered as buffer, not parameter)
        # 2D RoPE requires dim divisible by 4
        if use_rope_2d:
            # For 2D RoPE, max_seq_len represents max position in either dimension
            rope_cache = build_rope_cache_2d(max_seq_len, self.token_dim)
        else:
            rope_cache = build_rope_cache(max_seq_len, self.token_dim)
        self.register_buffer("rope_cache", rope_cache, persistent=False)

        # attention module
        self.context_target_attn = ContextTargetAttention(
            embed_dim=self.token_dim,
            num_heads=14,
            append_zero_attn=True,
        )

        # Feature grid size (sqrt of nb_features_per_patch)
        self.feature_grid_size = int(math.sqrt(nb_features_per_patch))
        assert self.feature_grid_size ** 2 == nb_features_per_patch, \
            f"nb_features_per_patch must be a perfect square, got {nb_features_per_patch}"

        self.mask_proj_out = SegmentationHead(
            embed_dim=embed_proj_dim,  # D dimension after projection
            num_classes=1,
            patch_size=patch_size,
            feature_grid_size=self.feature_grid_size,
        )

        self.embed_proj_in = nn.Linear(embed_dim , embed_proj_dim)

    def _compute_token_mask(
        self,
        is_target: torch.Tensor,
        mask_prob_tgt: float,
        mask_prob_ctx: float,
    ) -> torch.Tensor:
        """
        Compute which tokens should be masked based on target/context labels.

        Args:
            is_target: [B, K] - Boolean, True=target patch
            mask_prob_tgt: Probability of masking target tokens
            mask_prob_ctx: Probability of masking context tokens

        Returns:
            mask_applied: [B, K] - Boolean indicating which tokens to mask
        """
        B, K = is_target.shape
        device = is_target.device

        # Compute per-token mask probabilities
        mask_probs = torch.where(is_target, mask_prob_tgt, mask_prob_ctx)

        # Sample mask (only during training)
        if self.training:
            mask_applied = torch.rand(B, K, device=device) < mask_probs
        else:
            # During eval, use deterministic masking for targets only
            mask_applied = is_target & (mask_prob_tgt > 0.5)

        return mask_applied

    def _apply_token_masking(
        self,
        tokens: torch.Tensor,
        mask_applied: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply mask token to specified positions.

        Args:
            tokens: [B, K, embed_dim] - Input tokens (must match self.embed_dim)
            mask_applied: [B, K] - Boolean indicating which tokens to mask

        Returns:
            masked_tokens: [B, K, embed_dim]
        """
        B, K, D = tokens.shape

        # Apply mask token where masked
        mask_expanded = mask_applied.unsqueeze(-1).expand(-1, -1, D)
        masked_tokens = torch.where(mask_expanded, self.mask_token.expand(B, K, -1), tokens)

        return masked_tokens

    def _add_type_embeddings(
        self,
        tokens: torch.Tensor,
        is_target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Add target/context type embeddings.

        Args:
            tokens: [B, K, embed_dim]
            is_target: [B, K] - Boolean, True=target patch

        Returns:
            tokens with type embeddings added: [B, K, embed_dim]
        """
        B, K, D = tokens.shape

        # Create type embedding for each position
        type_embed = torch.where(
            is_target.unsqueeze(-1).expand(-1, -1, D),
            self.target_embed.expand(B, K, -1),
            self.context_embed.expand(B, K, -1),
        )

        return tokens + type_embed

    def forward(
        self,
        img_patches: torch.Tensor,
        coords: torch.Tensor = None,
        ctx_id_labels: torch.Tensor = None,
    ) -> dict[str, torch.Tensor]:
        """
        Process patches using only image features.

        Args:
            img_patches: [B, K, tokens_per_patch, embed_dim] - DINO features per patch
            coords: [B, K, 2] - Patch coordinates (optional)
            ctx_id_labels: [B, K] - 0 for target, >0 for context patches

        Returns:
            Dict with mask_patch_logit_preds [B, K, 1] and img_patches
        """

        #print("CrossPatchAttentionBackbone img_patches shape (B, K, T, D):", img_patches.shape)
        B, K, NF, E = img_patches.shape
        device = img_patches.device

        # Project embedding dimemsion
        img_patches = img_patches.view(B * K * NF, E)
        img_patches = self.embed_proj_in(img_patches)
        img_patches = img_patches.view(B, K, NF , -1)

        """
        # Average pool over tokens within each patch: [B, K, tokens, D] -> [B, K, D]
        if img_patches.dim() == 4:
            img_tokens = img_patches.mean(dim=2)  # [B, K, embed_dim]
        else:
            img_tokens = img_patches  # Already [B, K, embed_dim]
        """
        # Flatten features within each patch: [B, K, nb_features, D] -> [B, K, nb_features*D]
        img_tokens = img_patches.reshape(B, K, NF * img_patches.shape[-1])  # [B, K, nb_features*D]

        # Determine context vs target
        if ctx_id_labels is not None:
            is_context = ctx_id_labels > 0  # [B, K]
        else:
            is_context = torch.ones(B, K, dtype=torch.bool, device=device)
            is_context[:, 0] = False

        # Add type embeddings
        type_embed = torch.where(
            is_context.unsqueeze(-1),
            self.context_embed.expand(B, K, -1),
            self.target_embed.expand(B, K, -1),
        )
        combined = img_tokens + type_embed

        # Apply RoPE (2D if coords available and enabled, else 1D sequential)
        if self.use_rope_2d and coords is not None:
            combined = apply_rope_2d(combined, coords, self.rope_cache, self.image_size)
        else:
            combined = apply_rope(combined, self.rope_cache)

        # Prepend register tokens (treated as context)
        # Registers don't get RoPE (they have no spatial position)
        if self.register_tokens is not None:
            registers = self.register_tokens.expand(B, -1, -1)  # [B, num_reg, token_dim]
            # Add context embedding to registers
            registers = registers + self.context_embed.expand(B, self.num_registers, -1)
            combined = torch.cat([registers, combined], dim=1)  # [B, num_reg + K, token_dim]
            # Extend is_context mask: registers are context
            reg_mask = torch.ones(B, self.num_registers, dtype=torch.bool, device=device)
            is_context = torch.cat([reg_mask, is_context], dim=1)  # [B, num_reg + K]

        # Apply cross-patch attention
        output = self.context_target_attn(combined, is_context)

        # Remove register outputs (keep only patch outputs)
        if self.register_tokens is not None:
            output = output[:, self.num_registers:]  # [B, K, token_dim]

        # reshape [B, K, nb_features*D] -> [B, K, nb_features, D]
        D = self.embed_proj_dim
        output = output.view(B, K, NF, D)

        # Reshape to spatial dims for segmentation head: [B, K, NF, D] -> [B, K, D, h, w]
        h = w = self.feature_grid_size  # e.g., 7 for 49 features
        output = output.permute(0, 1, 3, 2)  # [B, K, D, NF]
        output = output.view(B, K, D, h, w)  # [B, K, D, h, w]

        # Project to output with explicit target size to match patch_size
        mask_pred = self.mask_proj_out(output, target_size=self.patch_size)

        return {
            'mask_patch_logit_preds': mask_pred,
            'img_patches': img_patches,
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



        