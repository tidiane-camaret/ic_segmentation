"""
MedSAM v1 Layer Feature Extractor for on-the-fly feature computation.

Extracts features from intermediate transformer layers of MedSAM v1 image encoder.
Unlike the standard MedSAM output (256-dim from neck), this extracts 768-dim features
directly from transformer blocks.

Features:
- Input: Images are resized to 1024x1024 (MedSAM's expected input)
- Output: [B, N, D] where N = 64*64 = 4096, D = 768 (from transformer blocks)
- Provides `extract_batch()` interface compatible with PatchICL

Usage:
    from src.models.medsam_extractor import MedSAMv1LayerExtractor

    extractor = MedSAMv1LayerExtractor(
        checkpoint_path="/path/to/medsam_vit_b.pth",
        layer_idx=11,
        device="cuda",
    )
    target_features, context_features = extractor.extract_batch(
        target_images, context_images
    )
"""
from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# Default checkpoint path
DEFAULT_MEDSAM_PATH = "/nfs/data/nii/data1/Analysis/camaret___in_context_segmentation/ANALYSIS_20251122/checkpoints/medsam_vit_b.pth"


class MedSAMProcessor:
    """Preprocessor for MedSAM v1 inputs."""

    def __init__(self, target_size: int = 1024):
        """
        Args:
            target_size: Target resolution (1024 for MedSAM v1)
        """
        self.target_size = target_size
        # MedSAM v1 normalization (same as SAM)
        self.pixel_mean = torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1)
        self.pixel_std = torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1)

    def __call__(
        self,
        images: torch.Tensor,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Process images for MedSAM v1 (vectorized for efficiency).

        Args:
            images: [B, 1, H, W] grayscale images, normalized to [0, 1]

        Returns:
            processed: [B, 3, target_size, target_size] normalized RGB tensor
        """
        if device is None:
            device = images.device

        if images.dim() == 3:
            images = images.unsqueeze(1)

        B = images.shape[0]

        # Flatten spatial dims for per-image quantile computation: [B, H*W]
        flat = images[:, 0].reshape(B, -1)

        # Vectorized percentile clipping
        lower = torch.quantile(flat, 0.005, dim=1, keepdim=True)  # [B, 1]
        upper = torch.quantile(flat, 0.995, dim=1, keepdim=True)  # [B, 1]

        # Reshape for broadcasting: [B, 1, 1, 1]
        lower = lower.view(B, 1, 1, 1)
        upper = upper.view(B, 1, 1, 1)

        # Clip values
        processed = torch.clamp(images, lower, upper)

        # Per-image min/max for rescaling
        img_min = processed.view(B, -1).min(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
        img_max = processed.view(B, -1).max(dim=1, keepdim=True)[0].view(B, 1, 1, 1)

        # Rescale to [0, 255] (SAM expects this range before normalization)
        denom = img_max - img_min
        denom = torch.where(denom > 0, denom, torch.ones_like(denom))
        processed = (processed - img_min) / denom * 255.0

        # Expand to RGB: [B, 1, H, W] -> [B, 3, H, W]
        processed = processed.expand(-1, 3, -1, -1).contiguous()

        # Resize to target_size
        processed = F.interpolate(
            processed,
            size=(self.target_size, self.target_size),
            mode="bilinear",
            align_corners=False,
        )

        # SAM normalization
        mean = self.pixel_mean.to(device)
        std = self.pixel_std.to(device)
        processed = (processed - mean) / std

        return processed


class MedSAMv1LayerExtractor(nn.Module):
    """
    Feature extractor using MedSAM v1 image encoder transformer layers.

    Extracts 768-dim features from intermediate transformer blocks (not the neck).
    This provides richer features compared to the 256-dim neck output.

    The native grid size depends on input resolution:
        - 1024×1024 input → 64×64 grid (4096 tokens)
        - 512×512 input → 32×32 grid (1024 tokens)  ← Recommended for 32×32 output
        - 256×256 input → 16×16 grid (256 tokens)

    SAM uses 16×16 patches, so native_grid = target_size // 16.

    If output_grid_size differs from native, pooling or interpolation is applied.
    For best quality, use target_size = output_grid_size * 16 to get native extraction.
    """

    PATCH_SIZE = 16  # SAM's ViT uses 16×16 patches

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        target_size: int = 1024,
        device: Union[str, torch.device] = "cuda",
        layer_idx: int = 11,
        freeze: bool = True,
        output_grid_size: Optional[int] = None,
    ):
        """
        Args:
            checkpoint_path: Path to MedSAM v1 checkpoint (medsam_vit_b.pth).
                            If None, uses default path.
            target_size: Input resolution for MedSAM. Must be divisible by 16.
                        Examples: 1024 (default), 512 (faster), 256 (fastest).
                        Native grid size = target_size // 16.
            device: Device for computation
            layer_idx: Which transformer block to extract features from (0-11).
                      Default 11 = last block.
            freeze: Whether to freeze encoder weights
            output_grid_size: Size of output feature grid. If None, uses native grid.
                             If different from native, pooling/interpolation is applied.
                             For best quality, set target_size = output_grid_size * 16.
        """
        super().__init__()

        # Validate target_size
        if target_size % self.PATCH_SIZE != 0:
            raise ValueError(
                f"target_size ({target_size}) must be divisible by patch size ({self.PATCH_SIZE})"
            )

        self.target_size = target_size
        self.device = torch.device(device) if isinstance(device, str) else device
        self.layer_idx = layer_idx

        # Compute native grid size from target_size
        self.native_grid_size = target_size // self.PATCH_SIZE
        
        # If output_grid_size not specified, use native
        if output_grid_size is None:
            output_grid_size = self.native_grid_size
        self.output_grid_size = output_grid_size

        self.processor = MedSAMProcessor(target_size=target_size)

        self._load_model(checkpoint_path)

        if freeze:
            for param in self.image_encoder.parameters():
                param.requires_grad = False
            self.image_encoder.eval()

        self._frozen = freeze

        # Setup pooling/interpolation if output differs from native
        self._setup_resize_layer()

    def _setup_resize_layer(self):
        """Setup pooling or interpolation layer if needed."""
        if self.output_grid_size == self.native_grid_size:
            self.resize = None
            print(f"  Feature grid: {self.native_grid_size}×{self.native_grid_size} (native, no resize)")
        elif self.output_grid_size < self.native_grid_size:
            self.resize = nn.AdaptiveAvgPool2d((self.output_grid_size, self.output_grid_size))
            print(f"  Pooling features: {self.native_grid_size}×{self.native_grid_size} → {self.output_grid_size}×{self.output_grid_size}")
        else:
            # Upsampling - use bilinear interpolation (stored as lambda, applied in forward)
            self.resize = "upsample"
            print(f"  WARNING: Upsampling features: {self.native_grid_size}×{self.native_grid_size} → {self.output_grid_size}×{self.output_grid_size}")
            print(f"           Consider using target_size={self.output_grid_size * self.PATCH_SIZE} for native extraction")

    def _load_model(self, checkpoint_path: Optional[str]):
        """Load MedSAM v1 image encoder."""
        try:
            from segment_anything import sam_model_registry
        except ImportError:
            raise ImportError(
                "segment-anything not installed. Run:\n"
                "  pip install git+https://github.com/bowang-lab/MedSAM.git"
            )

        # Use default path if not provided
        if checkpoint_path is None:
            checkpoint_path = DEFAULT_MEDSAM_PATH

        # Build SAM model
        print(f"Loading MedSAM v1 from {checkpoint_path}...")
        sam_model = sam_model_registry["vit_b"](checkpoint=checkpoint_path)
        self.image_encoder = sam_model.image_encoder
        self.image_encoder.to(self.device)

        # Feature dimensions: transformer blocks output 768-dim features
        self.feature_dim = 768

        print(f"MedSAM v1 loaded (12 blocks, extracting from block {self.layer_idx})")
        print(f"  Input size: {self.target_size}×{self.target_size}")
        print(f"  Native grid: {self.native_grid_size}×{self.native_grid_size} ({self.native_grid_size**2} tokens)")

    def train(self, mode: bool = True):
        """Keep encoder in eval mode if frozen."""
        super().train(mode)
        if self._frozen:
            self.image_encoder.eval()
        return self

    @torch.no_grad()
    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract features from a specific transformer block.

        Args:
            images: [B, 1, H, W] grayscale images, normalized to [0, 1]

        Returns:
            features: [B, N, D] where:
                - N = output_grid_size² (e.g., 32×32 = 1024)
                - D = 768
        """
        was_training = self.image_encoder.training
        self.image_encoder.eval()

        # Use input tensor's device (in case model was moved by accelerator.prepare)
        device = images.device

        # Ensure encoder is on the same device as input
        if next(self.image_encoder.parameters()).device != device:
            self.image_encoder.to(device)

        # Preprocess
        x = self.processor(images, device=device)
        x = x.to(device)

        # Patch embedding: [B, 3, target_size, target_size] -> [B, native_grid, native_grid, 768]
        x = self.image_encoder.patch_embed(x)

        # Add positional embedding (interpolate if grid size differs from trained 64x64)
        if self.image_encoder.pos_embed is not None:
            pos_embed = self.image_encoder.pos_embed  # [1, 64, 64, 768] for SAM
            if pos_embed.shape[1] != self.native_grid_size or pos_embed.shape[2] != self.native_grid_size:
                # Interpolate positional embeddings to match current grid size
                # [1, 64, 64, 768] -> [1, 768, 64, 64] -> [1, 768, native, native] -> [1, native, native, 768]
                pos_embed = pos_embed.permute(0, 3, 1, 2)
                pos_embed = F.interpolate(
                    pos_embed,
                    size=(self.native_grid_size, self.native_grid_size),
                    mode="bicubic",
                    align_corners=False,
                )
                pos_embed = pos_embed.permute(0, 2, 3, 1)
            x = x + pos_embed

        # Pass through transformer blocks up to layer_idx
        for i, blk in enumerate(self.image_encoder.blocks):
            x = blk(x)
            if i == self.layer_idx:
                break

        # x is [B, native_grid, native_grid, 768]
        B, H, W, D = x.shape

        # Resize to output_grid_size if needed
        if self.resize is not None:
            # [B, H, W, D] -> [B, D, H, W]
            x = x.permute(0, 3, 1, 2)
            
            if self.resize == "upsample":
                x = F.interpolate(
                    x, 
                    size=(self.output_grid_size, self.output_grid_size),
                    mode="bilinear",
                    align_corners=False,
                )
            else:
                x = self.resize(x)  # AdaptiveAvgPool2d
            
            # [B, D, out, out] -> [B, out, out, D]
            x = x.permute(0, 2, 3, 1)
            H, W = x.shape[1], x.shape[2]

        # Reshape to [B, N, D]
        features = x.reshape(B, H * W, D)

        if was_training and not self._frozen:
            self.image_encoder.train()

        return features

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.extract_features(images)

    def extract_batch(
        self,
        target_images: torch.Tensor,
        context_images: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Extract features for target and context images.

        This interface is compatible with PatchICL's feature_extractor.

        Args:
            target_images: [B, 1, H, W] target images
            context_images: [B, k, 1, H, W] context images (optional)

        Returns:
            target_features: [B, N, D] where N = output_grid_size^2, D = 768
            context_features: [B, k, N, D] or None
        """
        B = target_images.shape[0]

        if context_images is not None:
            k = context_images.shape[1]

            # Flatten context for batch processing
            ctx_flat = context_images.view(B * k, *context_images.shape[2:])

            # Concatenate all images for efficient extraction
            all_images = torch.cat([target_images, ctx_flat], dim=0)

            # Extract features for all images at once
            all_features = self.extract_features(all_images)

            # Split back
            target_features = all_features[:B]
            context_features = all_features[B:].view(B, k, *all_features.shape[1:])

            return target_features, context_features
        else:
            target_features = self.extract_features(target_images)
            return target_features, None

    def get_feature_info(self) -> dict:
        """Get information about the extracted features."""
        grid = self.output_grid_size
        native = self.native_grid_size
        return {
            "extractor": "medsam_v1_layer",
            "layer_idx": self.layer_idx,
            "feature_dim": self.feature_dim,
            "target_size": self.target_size,
            "native_grid_size": native,
            "output_grid_size": grid,
            "num_tokens": grid * grid,
            "resized": grid != native,
            "resize_type": "pooled" if grid < native else ("upsampled" if grid > native else "none"),
        }


def create_feature_extractor(
    extractor_type: str = "meddino",
    **kwargs,
) -> nn.Module:
    """
    Factory function to create feature extractors.

    Args:
        extractor_type: One of "meddino", "medsam_v1", "medsam_v1_layer"
        **kwargs: Arguments passed to the extractor constructor

    Returns:
        Feature extractor module with extract_batch() interface
    """
    if extractor_type == "meddino":
        from src.models.meddino_extractor import MedDINOFeatureExtractor
        return MedDINOFeatureExtractor(**kwargs)

    elif extractor_type in ("medsam_v1", "medsam_v1_layer"):
        return MedSAMv1LayerExtractor(**kwargs)

    elif extractor_type == "universeg":
        from src.models.universeg_extractor import UniverSegExtractor
        return UniverSegExtractor(**kwargs)

    else:
        raise ValueError(
            f"Unknown extractor type: {extractor_type}. "
            f"Supported: 'meddino', 'medsam_v1', 'medsam_v1_layer', 'universeg'"
        )
