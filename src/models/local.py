import math
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModel


class PatchEmbedding2D(nn.Module):
    """Embed 2D image patches into tokens."""

    def __init__(self, patch_size: int = 8, in_channels: int = 1, embed_dim: int = 256):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        # x: [B, C, H, W] -> [B, embed_dim, H//ps, W//ps]
        x = self.proj(x)
        # Flatten to [B, num_patches, embed_dim]
        x = x.flatten(2).transpose(1, 2)
        return x


class LocalTransformer(nn.Module):
    """Shallow transformer for processing local patches."""

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection to segmentation logits per token
        self.seg_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, x):
        # x: [B, num_tokens, embed_dim]
        x = self.transformer(x)
        logits = self.seg_head(x)  # [B, num_tokens, 1]
        return logits.squeeze(-1)


class SegmentationHead(nn.Module):
    """Decoder head with transformer for cross-patch attention, then CNN upsampling."""

    def __init__(
        self,
        embed_dim: int = 1024,
        num_classes: int = 1,
        patch_size: int = 16,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim

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

    def forward(self, patch_features, h, w, num_patches: int = 1):
        """
        Args:
            patch_features: [B, total_tokens, embed_dim] where total_tokens = K * tokens_per_patch
            h, w: spatial grid dimensions of tokens within a single patch
            num_patches: K, number of patches per example

        Returns:
            [B, K, num_classes, H, W] where H=h*patch_size, W=w*patch_size
        """
        B, _, C = patch_features.shape
        tokens_per_patch = h * w

        # Apply transformer for cross-patch attention (all tokens attend to each other)
        patch_features = self.transformer(patch_features)

        # Reshape to [B*K, tokens_per_patch, embed_dim] for per-patch decoding
        patch_features = patch_features.reshape(B * num_patches, tokens_per_patch, C)

        # Reshape to spatial grid and decode
        feature_map = patch_features.transpose(1, 2).reshape(B * num_patches, C, h, w)
        output = self.decoder(feature_map)

        # Reshape to [B, K, num_classes, H, W]
        _, nc, H, W = output.shape
        output = output.reshape(B, num_patches, nc, H, W)

        return output


class LocalDino(nn.Module):
    """
    Local branch using DINOv3 backbone for processing K patches with cross-patch attention.

    All K patches are processed together through the DINO transformer, allowing
    patches to attend to each other. Position embeddings are computed using RoPE
    (Rotary Position Embedding) based on each token's actual global coordinates
    in the original image.
    """

    def __init__(
        self,
        pretrained_path: str,
        patch_size: int = 32,
        image_size: int = 224,
        freeze_backbone: bool = True,
        num_classes: int = 1,
    ):
        """
        Args:
            pretrained_path: Path to pretrained DINO model
            patch_size: Size of input patches (from global branch selection)
            image_size: Original image size (for computing relative positions)
            freeze_backbone: Whether to freeze DINO backbone weights
            num_classes: Number of segmentation classes (1 for binary)
        """
        super().__init__()
        self.patch_size = patch_size
        self.image_size = image_size
        self.dino_patch_size = 16  # DINO's internal patch size

        # Load DINO backbone
        self.backbone = AutoModel.from_pretrained(pretrained_path)
        self.embed_dim = self.backbone.config.hidden_size  # 1024 for ViT-L
        self.num_heads = self.backbone.config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.rope_theta = self.backbone.config.rope_theta

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Number of DINO tokens per input patch
        self.tokens_per_patch = (patch_size // self.dino_patch_size) ** 2

        # Compute RoPE inverse frequencies (same as DINOv3)
        inv_freq = 1 / self.rope_theta ** torch.arange(0, 1, 4 / self.head_dim, dtype=torch.float32)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Segmentation head
        self.seg_head = SegmentationHead(
            embed_dim=self.embed_dim,
            num_classes=num_classes,
            patch_size=self.dino_patch_size,
        )

    def compute_rope_embeddings(self, coords, device, dtype):
        """
        Compute RoPE cos/sin embeddings for tokens based on their global coordinates.

        Args:
            coords: [B, K, 2] - (h, w) top-left coordinates of each patch in original image
            device: torch device
            dtype: torch dtype

        Returns:
            cos, sin: [B, K * tokens_per_patch, head_dim] - RoPE embeddings for all tokens
        """
        B, K, _ = coords.shape
        tokens_h = tokens_w = int(math.sqrt(self.tokens_per_patch))

        # Create token offset grid within a patch: [tokens_h, tokens_w, 2]
        # Each token's offset from patch top-left to token center
        ti = torch.arange(tokens_h, device=device, dtype=torch.float32)
        tj = torch.arange(tokens_w, device=device, dtype=torch.float32)
        grid_i, grid_j = torch.meshgrid(ti, tj, indexing='ij')

        # Token center offsets: (ti + 0.5) * dino_patch_size
        token_offsets_h = (grid_i + 0.5) * self.dino_patch_size  # [tokens_h, tokens_w]
        token_offsets_w = (grid_j + 0.5) * self.dino_patch_size
        token_offsets = torch.stack([token_offsets_h, token_offsets_w], dim=-1)  # [tokens_h, tokens_w, 2]
        token_offsets = token_offsets.reshape(1, 1, self.tokens_per_patch, 2)  # [1, 1, tpp, 2]

        # Expand coords: [B, K, 1, 2] + [1, 1, tpp, 2] -> [B, K, tpp, 2]
        coords_expanded = coords.unsqueeze(2)  # [B, K, 1, 2]
        token_coords = coords_expanded + token_offsets  # [B, K, tpp, 2]

        # Reshape to [B, K * tpp, 2]
        token_coords = token_coords.reshape(B, K * self.tokens_per_patch, 2)

        # Normalize to [-1, +1] range (same as DINOv3)
        token_coords = token_coords / self.image_size  # [0, 1]
        token_coords = 2.0 * token_coords - 1.0  # [-1, +1]

        # Compute RoPE angles: 2π * coords * inv_freq
        # token_coords: [B, num_tokens, 2]
        # inv_freq: [head_dim / 4]
        angles = 2 * math.pi * token_coords[:, :, :, None] * self.inv_freq[None, None, None, :]
        # angles: [B, num_tokens, 2, head_dim/4]

        angles = angles.flatten(2, 3)  # [B, num_tokens, head_dim/2]
        angles = angles.tile(2)  # [B, num_tokens, head_dim]

        cos = torch.cos(angles).to(dtype)
        sin = torch.sin(angles).to(dtype)

        return cos, sin

    def forward_with_custom_rope(self, hidden_states, cos, sin):
        """
        Forward pass through DINO transformer layers with custom RoPE embeddings.

        Args:
            hidden_states: [B, num_tokens, embed_dim] - embedded patch tokens
            cos, sin: [B, num_tokens, head_dim] - custom RoPE embeddings

        Returns:
            output: [B, num_tokens, embed_dim] - transformer output
        """
        # Reshape cos/sin for attention: [B, 1, num_tokens, head_dim]
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        position_embeddings = (cos, sin)

        # Process through transformer layers
        for layer in self.backbone.layer:
            hidden_states = layer(
                hidden_states,
                position_embeddings=position_embeddings,
            )[0]

        # Final layer norm
        hidden_states = self.backbone.norm(hidden_states)

        return hidden_states

    def forward(self, patches, coords=None):
        """
        Process K patches: DINO patch embedding -> SegmentationHead transformer -> decode.

        Args:
            patches: [B, K, C, ps, ps] - K patches per batch
            coords: [B, K, 2] - (h, w) coordinates (unused for now, reserved for future)

        Returns:
            patch_logits: [B, K, 1, ps, ps] - segmentation logits for each patch
        """
        B, K, C, ps, _ = patches.shape

        # Ensure patches are 3-channel for DINO (repeat grayscale)
        if C == 1:
            patches = patches.repeat(1, 1, 3, 1, 1)

        # Reshape to [B*K, 3, ps, ps] for patch embedding
        patches_flat = patches.reshape(B * K, 3, ps, ps)

        # Apply DINO patch embedding (Conv2d projection only)
        with torch.no_grad():
            hidden_states = self.backbone.embeddings.patch_embeddings(patches_flat)
            # hidden_states: [B*K, embed_dim, h, w] where h=w=ps/16
            hidden_states = hidden_states.flatten(2).transpose(1, 2)  # [B*K, tokens_per_patch, embed_dim]

        # Reshape to [B, K * tokens_per_patch, embed_dim] for cross-patch attention
        hidden_states = hidden_states.reshape(B, K * self.tokens_per_patch, self.embed_dim)

        # Compute spatial dimensions for seg head (per patch)
        h = w = ps // self.dino_patch_size

        # Apply segmentation head - has its own transformer for cross-patch attention
        # seg_head expects [B, K*tokens_per_patch, embed_dim], returns [B, K, num_classes, ps, ps]
        seg_output = self.seg_head(hidden_states, h, w, num_patches=K)

        return seg_output


class LocalDinoLight(nn.Module):
    """
    Lightweight local branch using only DINO's patch embeddings + custom transformer.

    Uses DINO's Conv2d patch embedding but not the full transformer,
    allowing for more flexibility and faster training.
    """

    def __init__(
        self,
        pretrained_path: Optional[str] = None,
        patch_size: int = 32,
        image_size: int = 224,
        embed_dim: int = 768,
        num_heads: int = 8,
        num_layers: int = 4,
        num_classes: int = 1,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.image_size = image_size
        self.dino_patch_size = 16
        self.embed_dim = embed_dim

        # Patch embedding (similar to DINO's)
        self.patch_embed = nn.Conv2d(
            3, embed_dim,
            kernel_size=self.dino_patch_size,
            stride=self.dino_patch_size
        )

        # Load pretrained patch embedding if available
        if pretrained_path is not None:
            try:
                backbone = AutoModel.from_pretrained(pretrained_path)
                # Copy patch embedding weights
                with torch.no_grad():
                    self.patch_embed.weight.copy_(backbone.embeddings.patch_embeddings.projection.weight)
                    self.patch_embed.bias.copy_(backbone.embeddings.patch_embeddings.projection.bias)
                del backbone
            except Exception as e:
                print(f"Could not load pretrained weights: {e}")

        # Number of tokens per patch
        self.tokens_per_patch = (patch_size // self.dino_patch_size) ** 2

        # Position embeddings for tokens within a patch
        self.token_pos_embed = nn.Parameter(
            torch.randn(1, self.tokens_per_patch, embed_dim) * 0.02
        )

        # Position embeddings for patch locations
        self.num_patch_positions = (image_size // patch_size) ** 2
        self.patch_pos_embed = nn.Parameter(
            torch.randn(1, self.num_patch_positions, embed_dim) * 0.02
        )

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
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

    def forward(self, patches, coords=None):
        """
        Args:
            patches: [B, K, C, ps, ps]
            coords: [B, K, 2] - patch coordinates in original image
        Returns:
            [B, K, num_classes, ps, ps]
        """
        B, K, C, ps, _ = patches.shape

        # Ensure 3 channels
        if C == 1:
            patches = patches.repeat(1, 1, 3, 1, 1)

        # Reshape and embed: [B*K, 3, ps, ps] -> [B*K, embed_dim, h, w]
        patches_flat = patches.reshape(B * K, 3, ps, ps)
        features = self.patch_embed(patches_flat)  # [B*K, embed_dim, h, w]

        h = w = ps // self.dino_patch_size
        features = features.flatten(2).transpose(1, 2)  # [B*K, tokens, embed_dim]

        # Add token position embeddings
        features = features + self.token_pos_embed

        # Add patch position embeddings
        if coords is not None:
            grid_size = self.image_size // self.patch_size
            h_idx = (coords[:, :, 0] // self.patch_size).clamp(0, grid_size - 1)
            w_idx = (coords[:, :, 1] // self.patch_size).clamp(0, grid_size - 1)
            pos_idx = (h_idx * grid_size + w_idx).reshape(B * K)
            pos_idx = pos_idx.clamp(0, self.num_patch_positions - 1)

            patch_pos = self.patch_pos_embed[:, pos_idx, :].squeeze(0)
            features = features + patch_pos.unsqueeze(1)

        # Reshape for cross-patch attention: [B, K*tokens, embed_dim]
        features = features.reshape(B, K * self.tokens_per_patch, -1)

        # Transformer
        features = self.transformer(features)

        # Reshape back and apply seg head
        features = features.reshape(B * K, self.tokens_per_patch, -1)
        seg_output = self.seg_head(features, h, w)  # [B*K, classes, ps, ps]

        return seg_output.reshape(B, K, -1, ps, ps)

