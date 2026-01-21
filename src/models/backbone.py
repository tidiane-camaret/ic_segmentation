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
from transformers import AutoModel


class SegmentationHead(nn.Module):
    """Decoder head with CNN upsampling."""

    def __init__(
        self,
        embed_dim: int = 1024,
        num_classes: int = 1,
        patch_size: int = 16,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # Decoder with upsampling (4 x 2x upsamples = 16x total for patch_size=16)
        self.decoder = nn.Sequential(
            nn.Conv2d(embed_dim, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

            nn.Conv2d(64, num_classes, kernel_size=1),
        )

    def forward(
        self,
        patch_features: torch.Tensor,
        h: int,
        w: int,
        num_patches: int = 1,
        target_size: int | None = None,
    ):
        """
        Args:
            patch_features: [B, total_tokens, embed_dim] where total_tokens = K * tokens_per_patch
            h, w: spatial grid dimensions of tokens within a single patch
            num_patches: K, number of patches per example
            target_size: Target output size (H=W). If None, uses h*16.

        Returns:
            [B, K, num_classes, target_size, target_size]
        """
        B, _, C = patch_features.shape
        tokens_per_patch = h * w

        # Reshape to [B*K, tokens_per_patch, embed_dim] for per-patch decoding
        patch_features = patch_features.reshape(B * num_patches, tokens_per_patch, C)

        # Reshape to spatial grid and decode
        feature_map = patch_features.transpose(1, 2).reshape(B * num_patches, C, h, w)
        output = self.decoder(feature_map)

        # Resize to target size if specified and different from output
        _, nc, H, W = output.shape
        if target_size is not None and (H != target_size or W != target_size):
            output = F.interpolate(
                output, size=(target_size, target_size), mode='bilinear', align_corners=False
            )
            H, W = target_size, target_size

        # Reshape to [B, K, num_classes, H, W]
        output = output.reshape(B, num_patches, nc, H, W)

        return output


class PrecomputedFeatureBackbone(nn.Module):
    """
    Backbone that uses pre-computed DINOv3 features directly.

    Skips patch embedding and processes features through a transformer
    with custom position embeddings, then decodes to segmentation masks.

    Supports masked training where random patches are replaced with a
    learnable mask token before processing.
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
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.image_size = image_size
        self.dino_patch_size = 16  # DINOv3's internal patch size

        # Learnable mask token for masked training
        if use_mask_token:
            self.mask_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        else:
            self.mask_token = None

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
    ) -> torch.Tensor:
        """
        Process patches using pre-computed features.

        Args:
            patches: [B, K, C, ps, ps] - Raw patches (used for shape info, ignored if features provided)
            coords: [B, K, 2] - Patch coordinates in original image space
            precomputed_features: [B, K, tokens_per_patch, embed_dim] - Pre-computed DINOv3 features
            patch_mask: [B, K] - Boolean mask, True = mask this patch (optional)
            actual_image_size: Actual image size that coords are relative to. If None, uses self.image_size.

        Returns:
            patch_logits: [B, K, 1, ps, ps] - Segmentation logits for each patch
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

        # Reshape for cross-patch attention: [B, K * tokens_per_patch, embed_dim]
        features = features.reshape(B, K * tokens_per_patch, self.embed_dim)

        # Apply transformer
        features = self.transformer(features)

        # Apply segmentation head with target size matching patch size
        seg_output = self.seg_head(features, h, w, num_patches=K, target_size=ps)

        return seg_output


class PrecomputedDinoBackbone(nn.Module):
    """
    Backbone using pre-computed features with DINO's transformer layers.

    Loads DINO transformer weights but skips patch embedding,
    using pre-computed features directly.

    Supports masked training where random patches are replaced with a
    learnable mask token before processing.
    """

    def __init__(
        self,
        pretrained_path: str,
        patch_size: int = 16,
        image_size: int = 224,
        freeze_backbone: bool = True,
        num_classes: int = 1,
        use_mask_token: bool = True,
    ):
        """
        Args:
            pretrained_path: Path to pretrained DINO model
            patch_size: Size of input patches
            image_size: Original image size
            freeze_backbone: Whether to freeze DINO backbone weights
            num_classes: Number of segmentation classes
            use_mask_token: Whether to initialize a learnable mask token
        """
        super().__init__()
        self.patch_size = patch_size
        self.image_size = image_size
        self.dino_patch_size = 16

        # Load DINO backbone
        self.backbone = AutoModel.from_pretrained(pretrained_path)
        self.embed_dim = self.backbone.config.hidden_size

        # Learnable mask token for masked training
        if use_mask_token:
            self.mask_token = nn.Parameter(torch.randn(1, 1, self.embed_dim) * 0.02)
        else:
            self.mask_token = None
        self.num_heads = self.backbone.config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.rope_theta = self.backbone.config.rope_theta

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Compute RoPE inverse frequencies
        inv_freq = 1 / self.rope_theta ** torch.arange(0, 1, 4 / self.head_dim, dtype=torch.float32)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Segmentation head
        self.seg_head = SegmentationHead(
            embed_dim=self.embed_dim,
            num_classes=num_classes,
            patch_size=self.dino_patch_size,
        )

    def compute_rope_embeddings(
        self,
        coords: torch.Tensor,
        tokens_per_patch: int,
        device,
        dtype,
        actual_image_size: int | None = None,
    ):
        """Compute RoPE embeddings for tokens based on patch coordinates.

        Args:
            coords: [B, K, 2] patch coordinates in actual_image_size space
            tokens_per_patch: number of tokens per patch
            device: torch device
            dtype: torch dtype
            actual_image_size: the image size coords are relative to (if None, uses self.image_size)
        """
        B, K, _ = coords.shape
        tokens_h = tokens_w = int(math.sqrt(tokens_per_patch))

        inv_freq = self.inv_freq.to(device)

        # Token offset grid - scale offsets to actual image size
        img_size = actual_image_size if actual_image_size is not None else self.image_size
        offset_scale = img_size / self.image_size  # Scale from 224 space to actual space

        ti = torch.arange(tokens_h, device=device, dtype=torch.float32)
        tj = torch.arange(tokens_w, device=device, dtype=torch.float32)
        grid_i, grid_j = torch.meshgrid(ti, tj, indexing='ij')

        token_offsets_h = (grid_i + 0.5) * self.dino_patch_size * offset_scale
        token_offsets_w = (grid_j + 0.5) * self.dino_patch_size * offset_scale
        token_offsets = torch.stack([token_offsets_h, token_offsets_w], dim=-1)
        token_offsets = token_offsets.reshape(1, 1, tokens_per_patch, 2)

        coords_expanded = coords.unsqueeze(2).float()
        token_coords = coords_expanded + token_offsets
        token_coords = token_coords.reshape(B, K * tokens_per_patch, 2)

        # Normalize to [-1, +1] using actual image size
        token_coords = token_coords / img_size
        token_coords = 2.0 * token_coords - 1.0

        # Compute angles
        angles = 2 * math.pi * token_coords[:, :, :, None] * inv_freq[None, None, None, :]
        angles = angles.flatten(2, 3)
        angles = angles.tile(2)

        return torch.cos(angles).to(dtype), torch.sin(angles).to(dtype)

    def apply_rotary_pos_emb(self, q, k, cos, sin):
        """Apply rotary position embeddings."""
        def rotate_half(x):
            x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
            return torch.cat([-x2, x1], dim=-1)

        return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)

    def forward_with_custom_rope(self, hidden_states: torch.Tensor, cos, sin):
        """Forward through DINO layers with custom RoPE."""
        B, N, _ = hidden_states.shape

        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

        for layer in self.backbone.layer:
            residual = hidden_states
            hidden_states = layer.norm1(hidden_states)

            attn = layer.attention
            query = attn.q_proj(hidden_states)
            key = attn.k_proj(hidden_states)
            value = attn.v_proj(hidden_states)

            query = query.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
            key = key.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
            value = value.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

            query, key = self.apply_rotary_pos_emb(query, key, cos, sin)

            scale = self.head_dim ** -0.5
            attn_weights = torch.matmul(query, key.transpose(-2, -1)) * scale
            attn_weights = torch.softmax(attn_weights, dim=-1)
            attn_output = torch.matmul(attn_weights, value)

            attn_output = attn_output.transpose(1, 2).contiguous().view(B, N, self.embed_dim)
            attn_output = attn.o_proj(attn_output)

            hidden_states = residual + layer.drop_path(layer.layer_scale1(attn_output))

            residual = hidden_states
            hidden_states = layer.norm2(hidden_states)
            hidden_states = layer.mlp(hidden_states)
            hidden_states = residual + layer.drop_path(layer.layer_scale2(hidden_states))

        return self.backbone.norm(hidden_states)

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
    ) -> torch.Tensor:
        """
        Process patches using pre-computed features through DINO transformer.

        Args:
            patches: [B, K, C, ps, ps] - Raw patches (for shape info)
            coords: [B, K, 2] - Patch coordinates in actual image space
            precomputed_features: [B, K, tokens_per_patch, embed_dim] - Pre-computed features
            patch_mask: [B, K] - Boolean mask, True = mask this patch (optional)
            actual_image_size: Actual image size that coords are relative to. If None, uses self.image_size.

        Returns:
            patch_logits: [B, K, 1, ps, ps]
        """
        B, K, _, ps, _ = patches.shape
        device = patches.device
        dtype = patches.dtype

        if precomputed_features is None:
            raise ValueError("PrecomputedDinoBackbone requires precomputed_features")

        # Apply masking if provided
        if patch_mask is not None:
            precomputed_features = self.apply_mask(precomputed_features, patch_mask)

        tokens_per_patch = precomputed_features.shape[2]
        h = w = int(math.sqrt(tokens_per_patch))

        # Reshape features
        hidden_states = precomputed_features.reshape(B, K * tokens_per_patch, self.embed_dim)

        # Compute RoPE and process through transformer
        if coords is not None:
            cos, sin = self.compute_rope_embeddings(
                coords, tokens_per_patch, device, dtype, actual_image_size
            )
            hidden_states = self.forward_with_custom_rope(hidden_states, cos, sin)
        else:
            # Default grid coords
            grid_size = int(math.sqrt(K)) if K > 1 else 1
            default_coords = torch.zeros(B, K, 2, device=device, dtype=torch.float32)
            for k in range(K):
                default_coords[:, k, 0] = (k // grid_size) * ps
                default_coords[:, k, 1] = (k % grid_size) * ps
            cos, sin = self.compute_rope_embeddings(
                default_coords, tokens_per_patch, device, dtype, actual_image_size
            )
            hidden_states = self.forward_with_custom_rope(hidden_states, cos, sin)

        # Decode with target size matching input patch size
        seg_output = self.seg_head(hidden_states, h, w, num_patches=K, target_size=ps)

        return seg_output
