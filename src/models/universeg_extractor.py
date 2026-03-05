"""
UniverSeg Feature Extractor for on-the-fly feature computation.

Extracts per-image features from UniverSeg encoder blocks directly (encoder-only,
skipping decoder for efficiency).

For target images (no mask): dummy support (zeros) is used.
For context images (with mask): self-support trick — the image is passed as both
target and its own support with the GT mask, so the pretrained cross-conv fuses
mask information into the features without new parameters.

Features:
- Input: Grayscale images resized to 128x128, normalized to [0, 1]
- Single layer: [B, N, D] where D = 64
- Multi-layer:  [B, N, D*L] where L = number of layers, all resized to output_grid_size
- N depends on output_grid_size (or native size of chosen layer):
    layer 0: 128x128 = 16384
    layer 1:  64x64  = 4096
    layer 2:  32x32  = 1024
    layer 3:  16x16  = 256  (bottleneck)

Usage:
    # Single layer
    extractor = UniverSegExtractor(layer_idx=3, device="cuda")
    # All layers concatenated
    extractor = UniverSegExtractor(layer_idx="all", output_grid_size=8, device="cuda")
    # Specific layers
    extractor = UniverSegExtractor(layer_idx=[1, 2, 3], output_grid_size=8, device="cuda")

    target_features, context_features = extractor.extract_batch(
        target_images, context_images, context_masks
    )
"""
from __future__ import annotations

import sys
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# UniverSeg repo path
UNIVERSEG_REPO = "/work/dlclarge2/ndirt-SegFM3D/repos/UniverSeg"

# Grid sizes per encoder layer
LAYER_GRID_SIZES = {0: 128, 1: 64, 2: 32, 3: 16}
NUM_ENCODER_LAYERS = 4


def _parse_layer_idx(layer_idx) -> List[int]:
    """Parse layer_idx into a sorted list of layer indices."""
    if layer_idx == "all":
        return list(range(NUM_ENCODER_LAYERS))
    if isinstance(layer_idx, (list, tuple)):
        return sorted(int(i) for i in layer_idx)
    return [int(layer_idx)]


class UniverSegExtractor(nn.Module):
    """Extract per-image features from UniverSeg encoder blocks.

    Runs encoder blocks directly (skipping decoder for efficiency). A single
    dummy support (zeros) is passed so the cross-conv sees no real context.

    Args:
        layer_idx: Encoder block(s) to extract from. Options:
            - int (0-3): single layer (default 3 = bottleneck)
            - "all": all 4 layers, concatenated along feature dim
            - list of ints: specific layers, concatenated along feature dim
        device: Device for computation.
        pretrained: Load pretrained weights.
        freeze: Freeze all model weights.
        output_grid_size: Resize features to this grid. None = native (single-layer only).
            Required when using multiple layers since they have different native sizes.
        input_size: Input image size. Images are resized to this before feature extraction.
            Default 128 (UniverSeg native size).
        skip_preprocess: If True, skip percentile normalization (assumes dataloader
            already normalized images to [0, 1]). Saves ~7% compute. Default False.
    """

    DEFAULT_INPUT_SIZE = 128  # UniverSeg native input size
    FEATURE_DIM_PER_LAYER = 64  # Each encoder block outputs 64 channels

    def __init__(
        self,
        layer_idx: Union[int, str, List[int]] = 3,
        device: Union[str, torch.device] = "cuda",
        pretrained: bool = True,
        freeze: bool = True,
        output_grid_size: Optional[int] = None,
        input_size: int = 128,
        skip_preprocess: bool = True,
    ):
        super().__init__()
        self.device = torch.device(device) if isinstance(device, str) else device
        self.input_size = input_size
        self.skip_preprocess = skip_preprocess

        # Parse layer indices
        self.layer_indices = _parse_layer_idx(layer_idx)
        self.multi_layer = len(self.layer_indices) > 1
        self.feature_dim = self.FEATURE_DIM_PER_LAYER * len(self.layer_indices)

        # For backwards compat, expose single layer_idx when not multi-layer
        self.layer_idx = self.layer_indices[0] if not self.multi_layer else self.layer_indices

        # Determine output grid size
        if self.multi_layer:
            if output_grid_size is None:
                # Default: use the smallest native grid among selected layers
                output_grid_size = min(LAYER_GRID_SIZES[i] for i in self.layer_indices)
            self.native_grid_size = None  # No single native size
        else:
            self.native_grid_size = LAYER_GRID_SIZES[self.layer_indices[0]]
            if output_grid_size is None:
                output_grid_size = self.native_grid_size
        self.output_grid_size = output_grid_size

        # Load UniverSeg model
        self._load_model(pretrained)

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()
        self._frozen = freeze

        layers_str = "all" if layer_idx == "all" else self.layer_indices
        print(f"UniverSeg extractor: layers={layers_str}, "
              f"feature_dim={self.feature_dim}, "
              f"input={self.input_size}x{self.input_size}, "
              f"output={self.output_grid_size}x{self.output_grid_size}, "
              f"skip_preprocess={self.skip_preprocess}")

    def _load_model(self, pretrained: bool):
        """Load UniverSeg model."""
        if UNIVERSEG_REPO not in sys.path:
            sys.path.insert(0, UNIVERSEG_REPO)
        from universeg import universeg
        from universeg.nn import vmap
        self._vmap = vmap  # Store for use in encoder-only forward
        self.model = universeg(pretrained=pretrained).to(self.device)
        print(f"UniverSeg loaded (pretrained={pretrained})")

    def train(self, mode: bool = True):
        """Keep model in eval mode if frozen."""
        super().train(mode)
        if self._frozen:
            self.model.eval()
        return self

    def _resize_feat(self, feat: torch.Tensor) -> torch.Tensor:
        """Resize feature map to output_grid_size."""
        _, _, H, W = feat.shape
        if H == self.output_grid_size and W == self.output_grid_size:
            return feat
        if self.output_grid_size < H:
            return F.adaptive_avg_pool2d(feat, (self.output_grid_size, self.output_grid_size))
        return F.interpolate(
            feat, size=(self.output_grid_size, self.output_grid_size),
            mode="bilinear", align_corners=False,
        )

    def _encoder_only_forward(
        self,
        images: torch.Tensor,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
    ) -> dict[int, torch.Tensor]:
        """Run encoder blocks only, skipping decoder entirely.

        Args:
            images: [B, 1, H, W] preprocessed images
            support_images: [B, 1, 1, H, W] support images
            support_labels: [B, 1, 1, H, W] support labels

        Returns:
            Dict mapping layer_idx to features [B, 64, H, W]
        """
        # Reshape to UniverSeg expected format: [B, 1, 1, H, W]
        # images is [B, 1, H, W], need [B, 1, 1, H, W]
        target = images.unsqueeze(2)
        support = torch.cat([support_images, support_labels], dim=2)

        max_layer = max(self.layer_indices)
        captured = {}

        for i in range(max_layer + 1):
            target, support = self.model.enc_blocks[i](target, support)
            if i in self.layer_indices:
                captured[i] = target[:, 0]  # [B, C, H, W]
            if i < max_layer:
                target = self._vmap(self.model.downsample, target)
                support = self._vmap(self.model.downsample, support)

        return captured

    @torch.no_grad()
    def extract_features(
        self,
        images: torch.Tensor,
        masks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Extract features from encoder block(s) directly (encoder-only, no decoder).

        Args:
            images: [B, 1, H, W] grayscale images, normalized to [0, 1]
            masks: [B, 1, H, W] binary masks (optional). When provided, uses
                   self-support: the image is its own support with the mask,
                   so cross-conv fuses mask info into the features.

        Returns:
            features: [B, N, D] where N = output_grid_size^2,
                      D = 64 (single layer) or 64*L (multi-layer concat)
        """
        was_training = self.model.training
        self.model.eval()

        device = images.device
        if next(self.model.parameters()).device != device:
            self.model.to(device)

        B = images.shape[0]

        # Force float32 for numerical stability (UniverSeg not compatible with float16)
        images = images.float()

        if not self.skip_preprocess:
            # Batched percentile normalization to handle outliers (prevents NaN)
            # Skip if dataloader already normalized to [0, 1]
            flat = images[:, 0].reshape(B, -1)
            lower = torch.quantile(flat, 0.005, dim=1, keepdim=True)
            upper = torch.quantile(flat, 0.995, dim=1, keepdim=True)
            flat = torch.clamp(flat, lower, upper)
            img_min = flat.min(dim=1, keepdim=True)[0]
            img_max = flat.max(dim=1, keepdim=True)[0]
            flat = (flat - img_min) / (img_max - img_min).clamp(min=1e-8)
            images = flat.view(B, 1, images.shape[2], images.shape[3])

        # Resize to input_size if needed
        if images.shape[-2:] != (self.input_size, self.input_size):
            images = F.interpolate(
                images,
                size=(self.input_size, self.input_size),
                mode="bilinear",
                align_corners=False,
            )

        if masks is not None:
            # Self-support: image is its own support, with real mask
            masks = masks.float()
            if masks.shape[-2:] != (self.input_size, self.input_size):
                masks = F.interpolate(
                    masks,
                    size=(self.input_size, self.input_size),
                    mode="nearest",
                )
            support_images = images.unsqueeze(1)  # [B, 1, 1, H, W]
            support_labels = masks.unsqueeze(1)    # [B, 1, 1, H, W]
        else:
            # Dummy support: zeros (image-only features)
            support_images = torch.zeros(B, 1, 1, self.input_size, self.input_size,
                                         device=device, dtype=torch.float32)
            support_labels = torch.zeros(B, 1, 1, self.input_size, self.input_size,
                                         device=device, dtype=torch.float32)

        # Run encoder only (skip decoder for efficiency)
        captured = self._encoder_only_forward(images, support_images, support_labels)

        # Collect and resize features from each layer
        feat_list = []
        for lid in self.layer_indices:
            feat = self._resize_feat(captured[lid])  # [B, 64, G, G]
            feat_list.append(feat)

        # Concatenate along channel dim: [B, 64*L, G, G]
        feat = torch.cat(feat_list, dim=1)

        # [B, D, G, G] -> [B, G*G, D] = [B, N, D]
        B, D, G, _ = feat.shape
        features = feat.reshape(B, D, G * G).permute(0, 2, 1)

        # Replace NaN/Inf with zeros to prevent gradient explosion
        if torch.isnan(features).any() or torch.isinf(features).any():
            features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        if was_training and not self._frozen:
            self.model.train()

        return features

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.extract_features(images)

    def extract_batch(
        self,
        target_images: torch.Tensor,
        context_images: Optional[torch.Tensor] = None,
        context_masks: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Extract features for target and context images.

        Target images use dummy support (image-only features).
        Context images use self-support with their GT masks when provided,
        so context features are mask-conditioned via the pretrained cross-conv.

        Args:
            target_images: [B, 1, H, W]
            context_images: [B, k, 1, H, W] (optional)
            context_masks: [B, k, 1, H, W] binary masks for context (optional)

        Returns:
            target_features: [B, N, D]
            context_features: [B, k, N, D] or None
        """
        B = target_images.shape[0]

        if context_images is None:
            return self.extract_features(target_images), None

        k = context_images.shape[1]
        ctx_flat = context_images.view(B * k, *context_images.shape[2:])

        if context_masks is not None:
            # Separate paths: target with dummy support, context with self-support + mask
            target_features = self.extract_features(target_images)
            mask_flat = context_masks.view(B * k, *context_masks.shape[2:])
            context_features = self.extract_features(ctx_flat, masks=mask_flat)
            context_features = context_features.view(B, k, *context_features.shape[1:])
        else:
            # No masks: batch target + context together (faster)
            all_images = torch.cat([target_images, ctx_flat], dim=0)
            all_features = self.extract_features(all_images)
            target_features = all_features[:B]
            context_features = all_features[B:].view(B, k, *all_features.shape[1:])

        return target_features, context_features

    def get_feature_info(self) -> dict:
        """Get information about extracted features."""
        grid = self.output_grid_size
        return {
            "extractor": "universeg",
            "layer_indices": self.layer_indices,
            "layer_idx": self.layer_idx,
            "multi_layer": self.multi_layer,
            "feature_dim": self.feature_dim,
            "feature_dim_per_layer": self.FEATURE_DIM_PER_LAYER,
            "num_layers": len(self.layer_indices),
            "input_size": self.input_size,
            "native_grid_size": self.native_grid_size,
            "output_grid_size": grid,
            "num_tokens": grid * grid,
            "skip_preprocess": self.skip_preprocess,
        }
