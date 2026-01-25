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

class CrossPatchAttentionBackbone(nn.Module):
    """
    treats tgt/ctx patches as val/train examples
    - val patches attend to train patches
    - patch-level input features :
        - img : pre-computed DINOv3 features TODO reduce size
        - mask : random coloring in RGB space 
    - embeddings :
        - positional
        - target vs context  
    - sink tokens
    - register tokens
    - token masking for img and mask inputs
    """

    def __init__(
        self,
        embed_dim: int = 1024,
        patch_size: int = 16,
        image_size: int = 224,
        num_classes: int = 1,
        dropout: float = 0.1,
        token_mask_tgt_prob: float = 1.0,
        token_mask_ctx_prob: float = 0.0,
    ):
        """
        Args:
            embed_dim: Feature embedding dimension (1024 for DINOv3 ViT-L)
            image_size: Original image size (for position embeddings)
            patch_size: Output patch size for segmentation
            num_classes: Number of segmentation classes
            dropout: Dropout rate

        """
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.image_size = image_size
        self.num_classes = num_classes
        self.dropout = dropout

        self.num_heads = 1
        self.num_layers = 1
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=self.num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
    def forward(
        self,
        img_patches: torch.Tensor,
        mask_patches: torch.Tensor,
        coords: torch.Tensor = None,
        ctx_id_labels: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Process patches using pre-computed features.

        Args:
            img_patches: [B, K, C_img, (p_dims)] - image patches
            mask_patches: [B, K, C_mask, (p_dims)] - mask patches
            coords: [B, K, D] - Patch coords in original image space 
            ctx_id_labels: [B, K] - 0=target, >0=context patch labels

        Returns:
            mask_patch_logit_preds: [B, K, C_mask, (p_dims)]
        """

        # apply token masking based on tgt/ctx labels

        # embed patches with positional and tgt/ctx embeddings

        # apply cross-patch attention 

        return {
                'mask_patch_logit_preds': mask_patches,
                'img_patches': img_patches
            }

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



        