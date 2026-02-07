"""
UniverSeg Feature Extractor for on-the-fly feature computation.

Extracts per-image features from UniverSeg encoder blocks using forward hooks.
A dummy support (zeros) is passed so features are computed per-image.

Features:
- Input: Grayscale images resized to 128x128, normalized to [0, 1]
- Output: [B, N, D] where D = 64 (UniverSeg channel width)
- N depends on layer_idx:
    layer 0: 128x128 = 16384
    layer 1:  64x64  = 4096
    layer 2:  32x32  = 1024
    layer 3:  16x16  = 256  (bottleneck)

Usage:
    extractor = UniverSegExtractor(layer_idx=3, device="cuda")
    target_features, context_features = extractor.extract_batch(
        target_images, context_images
    )
"""
from __future__ import annotations

import sys
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# UniverSeg repo path
UNIVERSEG_REPO = "/work/dlclarge2/ndirt-SegFM3D/repos/UniverSeg"

# Grid sizes per encoder layer
LAYER_GRID_SIZES = {0: 128, 1: 64, 2: 32, 3: 16}


class UniverSegExtractor(nn.Module):
    """Extract per-image features from UniverSeg encoder blocks.

    Uses forward hooks to capture intermediate features. A single dummy
    support (zeros) is passed so the cross-conv sees no real context.

    Args:
        layer_idx: Encoder block to extract from (0-3). Default 3 (bottleneck).
        device: Device for computation.
        pretrained: Load pretrained weights.
        freeze: Freeze all model weights.
        output_grid_size: Resize features to this grid. None = native.
    """

    INPUT_SIZE = 128  # UniverSeg expects 128x128
    FEATURE_DIM = 64  # All encoder blocks output 64 channels

    def __init__(
        self,
        layer_idx: int = 3,
        device: Union[str, torch.device] = "cuda",
        pretrained: bool = True,
        freeze: bool = True,
        output_grid_size: Optional[int] = None,
    ):
        super().__init__()
        self.device = torch.device(device) if isinstance(device, str) else device
        self.layer_idx = layer_idx
        self.feature_dim = self.FEATURE_DIM

        # Native grid size for chosen layer
        self.native_grid_size = LAYER_GRID_SIZES[layer_idx]
        self.output_grid_size = output_grid_size or self.native_grid_size

        # Load UniverSeg model
        self._load_model(pretrained)

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()
        self._frozen = freeze

        # Resize layer if output differs from native
        if self.output_grid_size == self.native_grid_size:
            self.resize = None
        elif self.output_grid_size < self.native_grid_size:
            self.resize = nn.AdaptiveAvgPool2d(
                (self.output_grid_size, self.output_grid_size)
            )
        else:
            self.resize = "upsample"

        print(f"UniverSeg extractor: layer={layer_idx}, "
              f"native={self.native_grid_size}x{self.native_grid_size}, "
              f"output={self.output_grid_size}x{self.output_grid_size}")

    def _load_model(self, pretrained: bool):
        """Load UniverSeg model."""
        if UNIVERSEG_REPO not in sys.path:
            sys.path.insert(0, UNIVERSEG_REPO)
        from universeg import universeg
        self.model = universeg(pretrained=pretrained).to(self.device)
        print(f"UniverSeg loaded (pretrained={pretrained})")

    def train(self, mode: bool = True):
        """Keep model in eval mode if frozen."""
        super().train(mode)
        if self._frozen:
            self.model.eval()
        return self

    @torch.no_grad()
    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract features from encoder block via forward hook.

        Args:
            images: [B, 1, H, W] grayscale images, normalized to [0, 1]

        Returns:
            features: [B, N, D] where N = output_grid_size^2, D = 64
        """
        was_training = self.model.training
        self.model.eval()

        device = images.device
        if next(self.model.parameters()).device != device:
            self.model.to(device)

        B = images.shape[0]

        # Resize to 128x128
        if images.shape[-2:] != (self.INPUT_SIZE, self.INPUT_SIZE):
            images = F.interpolate(
                images,
                size=(self.INPUT_SIZE, self.INPUT_SIZE),
                mode="bilinear",
                align_corners=False,
            )

        # Dummy support: 1 support image of zeros with zero label
        support_images = torch.zeros(B, 1, 1, self.INPUT_SIZE, self.INPUT_SIZE,
                                     device=device)
        support_labels = torch.zeros(B, 1, 1, self.INPUT_SIZE, self.INPUT_SIZE,
                                     device=device)

        # Register hook to capture features at target layer
        captured = {}

        def hook_fn(module, input, output):
            target, support = output
            # target: [B, 1, C, H, W] -> squeeze group dim -> [B, C, H, W]
            captured["features"] = target[:, 0]

        handle = self.model.enc_blocks[self.layer_idx].register_forward_hook(hook_fn)

        # Forward pass (output is discarded, we only want the hooked features)
        self.model(images, support_images, support_labels)

        handle.remove()

        # features: [B, C, H, W]
        feat = captured["features"]

        # Resize if needed
        if self.resize is not None:
            if self.resize == "upsample":
                feat = F.interpolate(
                    feat,
                    size=(self.output_grid_size, self.output_grid_size),
                    mode="bilinear",
                    align_corners=False,
                )
            else:
                feat = self.resize(feat)

        # [B, C, H, W] -> [B, H*W, C] = [B, N, D]
        B, C, H, W = feat.shape
        features = feat.reshape(B, C, H * W).permute(0, 2, 1)

        if was_training and not self._frozen:
            self.model.train()

        return features

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.extract_features(images)

    def extract_batch(
        self,
        target_images: torch.Tensor,
        context_images: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Extract features for target and context images.

        Args:
            target_images: [B, 1, H, W]
            context_images: [B, k, 1, H, W] (optional)

        Returns:
            target_features: [B, N, D]
            context_features: [B, k, N, D] or None
        """
        B = target_images.shape[0]

        if context_images is not None:
            k = context_images.shape[1]
            ctx_flat = context_images.view(B * k, *context_images.shape[2:])
            all_images = torch.cat([target_images, ctx_flat], dim=0)
            all_features = self.extract_features(all_images)
            target_features = all_features[:B]
            context_features = all_features[B:].view(B, k, *all_features.shape[1:])
            return target_features, context_features
        else:
            return self.extract_features(target_images), None

    def get_feature_info(self) -> dict:
        """Get information about extracted features."""
        grid = self.output_grid_size
        native = self.native_grid_size
        return {
            "extractor": "universeg",
            "layer_idx": self.layer_idx,
            "feature_dim": self.feature_dim,
            "input_size": self.INPUT_SIZE,
            "native_grid_size": native,
            "output_grid_size": grid,
            "num_tokens": grid * grid,
            "resized": grid != native,
        }
