"""RAD-DINO Feature Extractor for multi-modality medical imaging.

RAD-DINO is Microsoft's radiology-specific DINOv2 model trained on RadImageNet (1.35M images)
covering CT, MRI, ultrasound, and X-ray. This provides better cross-modality generalization
compared to domain-specific encoders.

Key features:
- Pretrained on diverse radiology data (11 anatomical regions)
- Standard ViT-B/14 architecture (768-dim features)
- Self-supervised training with DINOv2 framework
- Can be frozen or fine-tuned

Reference: https://huggingface.co/microsoft/rad-dino
"""
from __future__ import annotations

from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class RADDINOProcessor:
    """Preprocessor for RAD-DINO inputs."""

    def __init__(
        self,
        target_size: int = 224,
        interpolation: str = "bilinear",
    ):
        """
        Args:
            target_size: Target resolution for ViT (default 224 for ViT-B/14)
            interpolation: Interpolation mode for resizing
        """
        self.target_size = target_size
        self.interpolation = interpolation
        # ImageNet normalization (used by RAD-DINO/DINOv2)
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def __call__(
        self,
        images: torch.Tensor,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Process a batch of images for RAD-DINO.

        Args:
            images: [B, 1, H, W] or [B, H, W] grayscale images, normalized to [0, 1]
            device: Target device for output tensor

        Returns:
            processed: [B, 3, target_size, target_size] normalized RGB tensor
        """
        if device is None:
            device = images.device

        # Ensure 4D tensor
        if images.dim() == 3:
            images = images.unsqueeze(1)

        B = images.shape[0]

        # Flatten spatial dims for batched quantile: [B, H*W]
        flat = images[:, 0].reshape(B, -1)

        # Batched percentile clipping (0.5% - 99.5%) - only 2 quantile calls total
        lower = torch.quantile(flat, 0.005, dim=1, keepdim=True)  # [B, 1]
        upper = torch.quantile(flat, 0.995, dim=1, keepdim=True)  # [B, 1]

        # Clamp with per-image bounds
        processed = torch.clamp(flat, lower, upper)

        # Rescale to [0, 1] per image (batched min/max)
        img_min = processed.min(dim=1, keepdim=True)[0]
        img_max = processed.max(dim=1, keepdim=True)[0]
        denom = (img_max - img_min).clamp(min=1e-8)
        processed = (processed - img_min) / denom

        # Reshape back to spatial and expand to RGB
        H, W = images.shape[2], images.shape[3]
        processed = processed.view(B, 1, H, W).expand(-1, 3, -1, -1)

        # Resize to target size
        processed = F.interpolate(
            processed,
            size=(self.target_size, self.target_size),
            mode=self.interpolation,
            align_corners=False if self.interpolation != "nearest" else None,
        )

        # ImageNet normalization
        mean = self.mean.to(device)
        std = self.std.to(device)
        processed = (processed - mean) / std

        return processed


class RADDINOExtractor(nn.Module):
    """RAD-DINO feature extractor for multi-modality medical imaging.

    Wraps Microsoft's RAD-DINO (radiology-specific DINOv2) for feature extraction.
    Loads weights from HuggingFace and provides efficient batched extraction.

    RAD-DINO outputs 16x16 grid of 768-dim features for 224x224 input (ViT-B/14).
    """

    def __init__(
        self,
        model_name: str = "microsoft/rad-dino",
        target_size: int = 224,
        output_grid_size: Optional[int] = None,
        device: Union[str, torch.device] = "cuda",
        freeze: bool = True,
    ):
        """
        Args:
            model_name: HuggingFace model name (default: microsoft/rad-dino)
            target_size: Input resolution for ViT (default: 224)
            output_grid_size: Output spatial resolution. If None, uses native grid (16 for ViT-B/14).
                              If different from native, features are interpolated.
            device: Device for model and computation
            freeze: Whether to freeze model weights (default: True for pretrained use)
        """
        super().__init__()
        self.model_name = model_name
        self.target_size = target_size
        self.device = torch.device(device) if isinstance(device, str) else device

        # ViT-B/14: 224/14 = 16x16 grid
        self.native_grid_size = target_size // 14
        self.output_grid_size = output_grid_size or self.native_grid_size
        self.feature_dim = 768  # ViT-B dimension

        # Initialize processor
        self.processor = RADDINOProcessor(target_size=target_size)

        # Load model
        self._load_model(model_name)

        # Freeze weights if requested
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()

        self._frozen = freeze

        # Log info
        print(f"RADDINOExtractor: model={model_name}, native_grid={self.native_grid_size}, "
              f"output_grid={self.output_grid_size}, feature_dim={self.feature_dim}, "
              f"freeze={freeze}, params={sum(p.numel() for p in self.model.parameters())}")

    def _load_model(self, model_name: str):
        """Load RAD-DINO model from HuggingFace."""
        from transformers import AutoModel

        print(f"Loading RAD-DINO from HuggingFace: {model_name}...")
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        print("RAD-DINO loaded successfully")

    def train(self, mode: bool = True):
        """Override train to keep model in eval mode if frozen."""
        super().train(mode)
        if self._frozen:
            self.model.eval()
        return self

    def _interpolate_features(self, features: torch.Tensor) -> torch.Tensor:
        """Interpolate features to output_grid_size if needed.

        Args:
            features: [B, N, D] where N = native_grid_size^2

        Returns:
            features: [B, N', D] where N' = output_grid_size^2
        """
        if self.output_grid_size == self.native_grid_size:
            return features

        B, N, D = features.shape
        H = W = self.native_grid_size

        # Reshape to spatial: [B, H, W, D] -> [B, D, H, W]
        features = features.view(B, H, W, D).permute(0, 3, 1, 2)

        # Interpolate
        features = F.interpolate(
            features,
            size=(self.output_grid_size, self.output_grid_size),
            mode="bilinear",
            align_corners=False,
        )

        # Reshape back: [B, D, H', W'] -> [B, N', D]
        features = features.permute(0, 2, 3, 1).reshape(B, -1, D)

        return features

    @torch.no_grad()
    def extract_features(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        """Extract features from a batch of images.

        Args:
            images: [B, 1, H, W] or [B, H, W] grayscale images, normalized to [0, 1]

        Returns:
            features: [B, N, D] where N = output_grid_size^2, D = 768
        """
        was_training = self.model.training
        self.model.eval()

        # Preprocess images
        pixel_values = self.processor(images, device=self.device)
        pixel_values = pixel_values.to(self.device)

        # Forward through model
        outputs = self.model(pixel_values)

        # Get patch tokens (exclude CLS token)
        # RAD-DINO (DINOv2) outputs: last_hidden_state [B, 1+N, D] where N=16*16=256
        features = outputs.last_hidden_state[:, 1:, :]  # [B, N, D], skip CLS

        # Interpolate if needed
        features = self._interpolate_features(features)

        if was_training and not self._frozen:
            self.model.train()

        return features

    def forward(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass for feature extraction.

        Args:
            images: [B, 1, H, W] grayscale images normalized to [0, 1]

        Returns:
            features: [B, N, D] where N = output_grid_size^2, D = 768
        """
        return self.extract_features(images)

    def extract_batch(
        self,
        target_images: torch.Tensor,
        context_images: Optional[torch.Tensor] = None,
        context_masks: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Extract features for target and context images efficiently.

        Note: context_masks is IGNORED for pretrained encoders. Mask information
        is handled at the backbone/transformer level via use_context_mask.

        Args:
            target_images: [B, 1, H, W] target images
            context_images: [B, k, 1, H, W] context images (optional)
            context_masks: [B, k, 1, H, W] IGNORED (kept for API compatibility)

        Returns:
            target_features: [B, N, D]
            context_features: [B, k, N, D] or None
        """
        B = target_images.shape[0]

        if context_images is not None:
            # Batch all images together for efficient extraction
            k = context_images.shape[1]

            # Reshape context: [B, k, 1, H, W] -> [B*k, 1, H, W]
            ctx_flat = context_images.view(B * k, *context_images.shape[2:])

            # Concatenate target and context
            all_images = torch.cat([target_images, ctx_flat], dim=0)  # [B + B*k, 1, H, W]

            # Extract features for all images at once
            all_features = self.extract_features(all_images)  # [B + B*k, N, D]

            # Split back
            target_features = all_features[:B]  # [B, N, D]
            context_features = all_features[B:].view(B, k, *all_features.shape[1:])  # [B, k, N, D]

            return target_features, context_features
        else:
            target_features = self.extract_features(target_images)
            return target_features, None

    def get_feature_info(self) -> dict:
        """Get information about extracted features."""
        return {
            "model_name": self.model_name,
            "layer_indices": ["last"],
            "feature_dim": self.feature_dim,
            "native_grid_size": self.native_grid_size,
            "output_grid_size": self.output_grid_size,
            "num_tokens": self.output_grid_size ** 2,
            "frozen": self._frozen,
        }


def create_rad_dino_extractor(
    model_name: str = "microsoft/rad-dino",
    target_size: int = 224,
    output_grid_size: Optional[int] = None,
    device: Union[str, torch.device] = "cuda",
    freeze: bool = True,
) -> RADDINOExtractor:
    """Factory function to create a RAD-DINO feature extractor.

    Args:
        model_name: HuggingFace model name (default: microsoft/rad-dino)
        target_size: Input resolution for ViT (default: 224)
        output_grid_size: Output spatial resolution (default: native 16x16)
        device: Device for computation
        freeze: Whether to freeze model weights

    Returns:
        RADDINOExtractor instance
    """
    return RADDINOExtractor(
        model_name=model_name,
        target_size=target_size,
        output_grid_size=output_grid_size,
        device=device,
        freeze=freeze,
    )
