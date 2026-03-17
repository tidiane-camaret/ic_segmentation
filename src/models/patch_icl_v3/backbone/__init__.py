"""SimpleBackbone composing encoder, attention, and decoder.

Main backbone module for PatchICL v3.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .attention import CrossPatchAttention
from .decoder import CNNDecoder
from .encoder import CNNEncoder
from .modules import ContinuousScaleEncoding, MaskPriorEncoder


class SimpleBackbone(nn.Module):
    """Simplified backbone: encoder -> attention -> decoder.

    Interface matches existing backbones:
        Input:
            img_patches: [B, K, tokens, D] - DINO features
            coords: [B, K, 2] - patch coordinates (y, x)
            ctx_id_labels: [B, K] - 0=target, 1..k=context images

        Output:
            {'mask_patch_logit_preds': [B, K, C, patch_size, patch_size]}
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
        dropout: float = 0.0,
        max_seq_len: int = 1024,
        decoder_use_skip_connections: bool = True,
        append_zero_attn: bool = False,
        gradient_checkpointing: bool = False,
        use_mask_prior: bool = False,
        mask_fusion_type: str = "additive",
        predict_sampling_map: bool = False,
        detach_sampling_features: bool = False,
        num_context_layers: int = 0,
        use_context_mask: bool = False,
        **kwargs,  # Absorb unused config
    ):
        """
        Args:
            embed_dim: Working dimension throughout the model
            num_heads: Number of attention heads
            num_layers: Number of attention layers
            num_registers: Number of register tokens
            num_classes: Number of output classes
            patch_size: Output patch size
            image_size: Full image size for position normalization
            input_dim: Input DINO feature dimension
            feature_grid_size: DINO feature grid size (8 or 16)
            dropout: Attention dropout rate
            max_seq_len: Maximum sequence length for RoPE
            decoder_use_skip_connections: Use U-Net skips
            append_zero_attn: Append zero token to K/V
            gradient_checkpointing: Use gradient checkpointing
            use_mask_prior: Fuse mask prior from previous level
            mask_fusion_type: How to fuse mask prior
            predict_sampling_map: Predict pixel-level sampling map
            detach_sampling_features: Detach features for sampling head
            num_context_layers: Number of context-first attention layers
            use_context_mask: Fuse GT context masks
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.image_size = image_size
        self.use_mask_prior = use_mask_prior
        self.use_context_mask = use_context_mask
        self.mask_fusion_type = mask_fusion_type
        self.predict_sampling_map = predict_sampling_map

        # Scale encoding
        self.scale_encoder = ContinuousScaleEncoding(embed_dim, learnable_freqs=True)

        # Encoder
        self.encoder = CNNEncoder(
            input_dim=input_dim,
            embed_dim=embed_dim,
            start_size=feature_grid_size,
            end_size=2,
            scale_embed_dim=embed_dim,
        )

        # Attention
        self.attention = CrossPatchAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            num_registers=num_registers,
            image_size=image_size,
            max_seq_len=max_seq_len,
            dropout=dropout,
            append_zero_attn=append_zero_attn,
            gradient_checkpointing=gradient_checkpointing,
            num_context_layers=num_context_layers,
        )

        # Decoder
        self.decoder = CNNDecoder(
            embed_dim=embed_dim,
            num_classes=num_classes,
            patch_size=patch_size,
            feature_grid_size=feature_grid_size,
            use_skip_connections=decoder_use_skip_connections,
            predict_sampling_map=predict_sampling_map,
            detach_sampling_features=detach_sampling_features,
            scale_embed_dim=embed_dim,
        )

        # Mask encoder and fusion
        if use_mask_prior or use_context_mask:
            self.mask_encoder = MaskPriorEncoder(embed_dim, feature_grid_size)
            if mask_fusion_type == "gated":
                self.mask_gate = nn.Parameter(torch.tensor(0.1))
                if use_context_mask:
                    self.context_mask_gate = nn.Parameter(torch.tensor(0.1))
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
        prev_register_tokens: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass: encode -> attention -> decode.

        Args:
            img_patches: [B, K, tokens, D] - features per patch
            coords: [B, K, 2] - patch coordinates
            ctx_id_labels: [B, K] - 0 for target, >0 for context
            return_attn_weights: Return attention weights
            level_idx: Fallback level index (deprecated)
            resolution: Current level resolution
            num_target_patches: Number of target patches
            mask_prior_patches: [B, K_target, 1, h, h] - mask prior
            context_mask_patches: [B, K_ctx, 1, h, h] - GT context masks
            prev_register_tokens: [B, R, D] - registers from previous level

        Returns:
            Dict with mask_patch_logit_preds and optional extras
        """
        B, K, NF, E = img_patches.shape
        device = img_patches.device

        # Default coords
        if coords is None:
            coords = torch.zeros(B, K, 2, device=device)

        # Determine context vs target
        if ctx_id_labels is not None:
            is_context = ctx_id_labels > 0
        else:
            is_context = torch.ones(B, K, dtype=torch.bool, device=device)
            is_context[:, 0] = False

        # Compute scale embedding
        if resolution is None:
            resolution = float(self.patch_size * (2 ** level_idx))
        scale_embed = self.scale_encoder(resolution, device=device)

        # Encode with resolution conditioning
        encoded, skips = self.encoder(img_patches, scale_embed)

        # Add scale embedding to encoded features
        encoded = encoded + scale_embed.view(1, 1, -1)

        # Fuse mask prior into target patches
        K_target = num_target_patches if num_target_patches is not None else K
        if self.use_mask_prior and mask_prior_patches is not None:
            mask_encoded = self.mask_encoder(mask_prior_patches)

            if self.mask_fusion_type == "additive":
                encoded[:, :K_target] = encoded[:, :K_target] + mask_encoded
            elif self.mask_fusion_type == "gated":
                gate = torch.sigmoid(self.mask_gate)
                encoded[:, :K_target] = encoded[:, :K_target] + gate * mask_encoded
            elif self.mask_fusion_type == "concat":
                fused = torch.cat([encoded[:, :K_target], mask_encoded], dim=-1)
                encoded[:, :K_target] = self.fusion_proj(fused)

        # Fuse GT context masks
        if self.use_context_mask and context_mask_patches is not None:
            ctx_mask_encoded = self.mask_encoder(context_mask_patches)

            if self.mask_fusion_type == "additive":
                encoded[:, K_target:] = encoded[:, K_target:] + ctx_mask_encoded
            elif self.mask_fusion_type == "gated":
                ctx_gate = torch.sigmoid(self.context_mask_gate)
                encoded[:, K_target:] = encoded[:, K_target:] + ctx_gate * ctx_mask_encoded
            elif self.mask_fusion_type == "concat":
                ctx_fused = torch.cat([encoded[:, K_target:], ctx_mask_encoded], dim=-1)
                encoded[:, K_target:] = self.fusion_proj(ctx_fused)

        # Cross-patch attention
        attended, extras = self.attention(
            encoded, coords, is_context, return_attn_weights,
            num_target_patches=num_target_patches,
            prev_register_tokens=prev_register_tokens,
        )

        # Decode with resolution conditioning
        seg_pred, sampling_map = self.decoder(attended, skips, scale_embed)

        result = {
            'mask_patch_logit_preds': seg_pred,
        }

        if sampling_map is not None:
            result['sampling_map'] = sampling_map

        if extras is not None:
            result['attn_weights'] = extras['attn_weights']
            result['register_tokens'] = extras['register_tokens']

        return result


# Re-export submodules for convenience
__all__ = [
    'SimpleBackbone',
    'CNNEncoder',
    'CNNDecoder',
    'CrossPatchAttention',
    'TransformerBlock',
    'ContinuousScaleEncoding',
    'MaskPriorEncoder',
    'ResolutionConditionedNorm',
    'LayerNorm2d',
]

from .attention import TransformerBlock
from .modules import LayerNorm2d, ResolutionConditionedNorm
