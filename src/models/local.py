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
    """Decoder head that upsamples patch features to segmentation mask."""

    def __init__(self, embed_dim: int = 1024, num_classes: int = 1, patch_size: int = 16):
        super().__init__()
        self.patch_size = patch_size

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

    def forward(self, patch_features, h, w):
        """
        Args:
            patch_features: [B, num_patches, embed_dim]
            h, w: spatial grid dimensions of patches
        Returns:
            [B, num_classes, H, W] where H=h*patch_size, W=w*patch_size
        """
        B, N, C = patch_features.shape
        feature_map = patch_features.transpose(1, 2).reshape(B, C, h, w)
        return self.decoder(feature_map)


class LocalDino(nn.Module):
    """
    Local branch using DINOv2/v3 backbone for processing K patches.

    Each patch is embedded using DINO's patch embedding, then positional
    embeddings are added based on the patch's location in the original image.
    All tokens from all K patches are processed together through the transformer.
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

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Number of DINO tokens per input patch
        self.tokens_per_patch = (patch_size // self.dino_patch_size) ** 2

        # Learnable position embeddings for patch locations in original image
        # Grid size for patch centers in original image
        self.num_patch_positions = (image_size // patch_size) ** 2
        self.patch_pos_embed = nn.Parameter(
            torch.randn(1, self.num_patch_positions, self.embed_dim) * 0.02
        )

        # Segmentation head
        self.seg_head = SegmentationHead(
            embed_dim=self.embed_dim,
            num_classes=num_classes,
            patch_size=self.dino_patch_size,
        )

    def get_patch_position_idx(self, coords, image_size):
        """Convert patch coordinates to position index in the grid."""
        grid_size = image_size // self.patch_size
        h_idx = coords[:, :, 0] // self.patch_size  # [B, K]
        w_idx = coords[:, :, 1] // self.patch_size  # [B, K]
        pos_idx = h_idx * grid_size + w_idx  # [B, K]
        return pos_idx

    def forward(self, patches, coords=None):
        """
        Process K patches through DINO backbone.

        Args:
            patches: [B, K, C, ps, ps] - K patches per batch
            coords: [B, K, 2] - (h, w) coordinates of each patch in original image

        Returns:
            patch_logits: [B, K, 1, ps, ps] - segmentation logits for each patch
        """
        B, K, C, ps, _ = patches.shape

        # Ensure patches are 3-channel for DINO (repeat grayscale)
        if C == 1:
            patches = patches.repeat(1, 1, 3, 1, 1)
            C = 3

        # Reshape to process all patches: [B*K, C, ps, ps]
        patches_flat = patches.reshape(B * K, C, ps, ps)

        # Resize patches to DINO expected size if needed
        # DINO expects multiples of patch_size (16), our patches should be ps x ps
        # For ps=32, we get 2x2=4 tokens per patch

        # Extract features through DINO backbone
        with torch.set_grad_enabled(not self.backbone.training or any(p.requires_grad for p in self.backbone.parameters())):
            outputs = self.backbone(patches_flat, output_hidden_states=True)
            # Get patch tokens (exclude CLS and register tokens)
            # DINOv2 has: 1 CLS + 4 registers + patch tokens
            features = outputs.last_hidden_state[:, 5:, :]  # [B*K, tokens_per_patch, embed_dim]

        # Add position embeddings based on patch location in original image
        if coords is not None:
            pos_idx = self.get_patch_position_idx(coords, self.image_size)  # [B, K]
            pos_idx_flat = pos_idx.reshape(B * K)  # [B*K]

            # Clamp indices to valid range
            pos_idx_flat = pos_idx_flat.clamp(0, self.num_patch_positions - 1)

            # Get position embeddings for each patch
            patch_pos = self.patch_pos_embed[:, pos_idx_flat, :]  # [1, B*K, embed_dim]
            patch_pos = patch_pos.squeeze(0)  # [B*K, embed_dim]

            # Add to all tokens of each patch
            features = features + patch_pos.unsqueeze(1)  # [B*K, tokens, embed_dim]

        # Compute spatial dimensions for seg head
        h = w = ps // self.dino_patch_size

        # Apply segmentation head per patch
        seg_output = self.seg_head(features, h, w)  # [B*K, num_classes, ps, ps]

        # Reshape back to [B, K, num_classes, ps, ps]
        seg_output = seg_output.reshape(B, K, -1, ps, ps)

        return seg_output


class LocalDinoLight(nn.Module):
    """
    Lightweight local branch using only DINO's patch embeddings + custom transformer.

    Uses DINO's Conv2d patch embedding but not the full transformer,
    allowing for more flexibility and faster training.
    """

    def __init__(
        self,
        pretrained_path: str = None,
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
        if pretrained_path:
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

