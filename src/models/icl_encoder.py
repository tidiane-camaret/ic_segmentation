"""ICLEncoder — Trainable feature extractor for in-context segmentation.

Shared image encoder (4 ConvBlocks) + parallel mask encoder (4 ConvBlocks)
with additive fusion at each scale.

Target path:  image only → shared img_blocks → features at each level
Context path: image + mask → shared img_blocks + mask_blocks → additive fusion per level

Output: [B, N, D] with N=output_grid_size², D=64*num_layers
"""
from __future__ import annotations

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

INPUT_SIZE = 128
FEATURE_DIM_PER_LAYER = 64
NUM_BLOCKS = 4
LAYER_GRID_SIZES = {0: 128, 1: 64, 2: 32, 3: 16}


def _parse_layer_idx(layer_idx) -> List[int]:
    """Parse layer_idx into a sorted list of layer indices."""
    if layer_idx == "all":
        return list(range(NUM_BLOCKS))
    if isinstance(layer_idx, (list, tuple)):
        return sorted(int(i) for i in layer_idx)
    return [int(layer_idx)]


class ConvBlock(nn.Module):
    """Conv3x3 + BN + GELU + Conv3x3 + BN + GELU."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ICLEncoder(nn.Module):
    """Trainable feature extractor with shared image encoder and parallel mask encoder.

    Image and mask streams are pooled independently before the next level.
    Fused features (img + mask) are only used for output collection, not fed forward.
    """

    INPUT_SIZE = INPUT_SIZE
    FEATURE_DIM_PER_LAYER = FEATURE_DIM_PER_LAYER

    def __init__(
        self,
        layer_idx: Union[int, str, List[int]] = "all",
        output_grid_size: Optional[int] = None,
        in_channels: int = 1,
        mask_channels: int = 1,
        freeze: bool = False,
    ):
        super().__init__()
        self.layer_indices = _parse_layer_idx(layer_idx)
        self.multi_layer = len(self.layer_indices) > 1
        self.feature_dim = FEATURE_DIM_PER_LAYER * len(self.layer_indices)
        self.layer_idx = self.layer_indices[0] if not self.multi_layer else self.layer_indices

        # Output grid size
        if self.multi_layer:
            if output_grid_size is None:
                output_grid_size = min(LAYER_GRID_SIZES[i] for i in self.layer_indices)
            self.native_grid_size = None
        else:
            self.native_grid_size = LAYER_GRID_SIZES[self.layer_indices[0]]
            if output_grid_size is None:
                output_grid_size = self.native_grid_size
        self.output_grid_size = output_grid_size

        # Shared image encoder blocks
        img_dims = [in_channels] + [FEATURE_DIM_PER_LAYER] * NUM_BLOCKS
        self.img_blocks = nn.ModuleList([
            ConvBlock(img_dims[i], img_dims[i + 1]) for i in range(NUM_BLOCKS)
        ])

        # Parallel mask encoder blocks (context-only)
        msk_dims = [mask_channels] + [FEATURE_DIM_PER_LAYER] * NUM_BLOCKS
        self.msk_blocks = nn.ModuleList([
            ConvBlock(msk_dims[i], msk_dims[i + 1]) for i in range(NUM_BLOCKS)
        ])

        self.pool = nn.MaxPool2d(2)

        # Freeze if requested
        self._frozen = freeze
        if freeze:
            for param in self.parameters():
                param.requires_grad = False

        layers_str = "all" if layer_idx == "all" else self.layer_indices
        print(f"ICLEncoder: layers={layers_str}, feature_dim={self.feature_dim}, "
              f"output={self.output_grid_size}x{self.output_grid_size}, "
              f"freeze={freeze}, params={sum(p.numel() for p in self.parameters())}")

    def train(self, mode: bool = True):
        """Keep in eval mode if frozen."""
        if self._frozen and mode:
            return super().train(False)
        return super().train(mode)

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

    def extract_features(
        self,
        images: torch.Tensor,
        masks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Extract features from image (and optionally mask) through the encoder.

        Args:
            images: [B, 1, H, W] grayscale images, normalized to [0, 1]
            masks: [B, 1, H, W] binary masks (optional, context only)

        Returns:
            features: [B, N, D] where N=output_grid_size², D=64*num_layers
        """
        # Resize to INPUT_SIZE
        if images.shape[-2:] != (INPUT_SIZE, INPUT_SIZE):
            images = F.interpolate(images, size=(INPUT_SIZE, INPUT_SIZE),
                                   mode="bilinear", align_corners=False)
        if masks is not None and masks.shape[-2:] != (INPUT_SIZE, INPUT_SIZE):
            masks = F.interpolate(masks.float(), size=(INPUT_SIZE, INPUT_SIZE), mode="nearest")

        has_mask = masks is not None
        img_x = images
        msk_x = masks

        feat_list = []
        for i in range(NUM_BLOCKS):
            # Image stream
            img_x = self.img_blocks[i](img_x)

            # Mask stream (context only)
            if has_mask:
                msk_x = self.msk_blocks[i](msk_x)

            # Collect features at this level if requested
            if i in self.layer_indices:
                if has_mask:
                    fused = img_x + msk_x  # Additive fusion
                else:
                    fused = img_x
                feat_list.append(self._resize_feat(fused))

            # Downsample independently for next level
            if i < NUM_BLOCKS - 1:
                img_x = self.pool(img_x)
                if has_mask:
                    msk_x = self.pool(msk_x)

        # Concatenate along channel dim: [B, 64*L, G, G]
        feat = torch.cat(feat_list, dim=1)

        # [B, D, G, G] -> [B, G*G, D] = [B, N, D]
        B, D, G, _ = feat.shape
        return feat.reshape(B, D, G * G).permute(0, 2, 1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.extract_features(images)

    def extract_batch(
        self,
        target_images: torch.Tensor,
        context_images: Optional[torch.Tensor] = None,
        context_masks: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Extract features for target and context images.

        Batches all images through img_blocks in a single forward pass to reduce
        kernel launch overhead, then applies msk_blocks only to context.

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

        if context_masks is None:
            # No masks: batch target + context together through full encoder
            all_images = torch.cat([target_images, ctx_flat], dim=0)
            all_features = self.extract_features(all_images)
            target_features = all_features[:B]
            context_features = all_features[B:].view(B, k, *all_features.shape[1:])
            return target_features, context_features

        # With masks: run img_blocks on all images together, msk_blocks on context only
        mask_flat = context_masks.view(B * k, *context_masks.shape[2:])
        all_images = torch.cat([target_images, ctx_flat], dim=0)  # [B+B*k, 1, H, W]

        if all_images.shape[-2:] != (INPUT_SIZE, INPUT_SIZE):
            all_images = F.interpolate(all_images, size=(INPUT_SIZE, INPUT_SIZE),
                                       mode="bilinear", align_corners=False)
        if mask_flat.shape[-2:] != (INPUT_SIZE, INPUT_SIZE):
            mask_flat = F.interpolate(mask_flat.float(), size=(INPUT_SIZE, INPUT_SIZE),
                                      mode="nearest")

        img_x = all_images
        msk_x = mask_flat
        target_feats: List[torch.Tensor] = []
        context_feats: List[torch.Tensor] = []

        for i in range(NUM_BLOCKS):
            img_x = self.img_blocks[i](img_x)
            msk_x = self.msk_blocks[i](msk_x)

            if i in self.layer_indices:
                target_feats.append(self._resize_feat(img_x[:B]))
                context_feats.append(self._resize_feat(img_x[B:] + msk_x))

            if i < NUM_BLOCKS - 1:
                img_x = self.pool(img_x)
                msk_x = self.pool(msk_x)

        # Concatenate layers and reshape to [*, N, D]
        t_feat = torch.cat(target_feats, dim=1)
        Bt, Dt, Gt, _ = t_feat.shape
        target_features = t_feat.reshape(Bt, Dt, Gt * Gt).permute(0, 2, 1)

        c_feat = torch.cat(context_feats, dim=1)
        Bc, Dc, Gc, _ = c_feat.shape
        context_features = c_feat.reshape(Bc, Dc, Gc * Gc).permute(0, 2, 1)
        context_features = context_features.view(B, k, *context_features.shape[1:])

        return target_features, context_features

    def get_feature_info(self) -> dict:
        """Get information about extracted features."""
        grid = self.output_grid_size
        return {
            "extractor": "icl_encoder",
            "layer_indices": self.layer_indices,
            "layer_idx": self.layer_idx,
            "multi_layer": self.multi_layer,
            "feature_dim": self.feature_dim,
            "feature_dim_per_layer": FEATURE_DIM_PER_LAYER,
            "num_layers": len(self.layer_indices),
            "input_size": INPUT_SIZE,
            "native_grid_size": self.native_grid_size,
            "output_grid_size": grid,
            "num_tokens": grid * grid,
        }
