"""
NMSW-style Patch Aggregation Module.

Combines local patch predictions with global coarse prediction
using Gaussian weighting and optional learnable aggregation.
"""

from typing import List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.data.utils import compute_importance_map


class ConvBnReLU(nn.Module):
    """Convolution + BatchNorm + ReLU block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        repeat: int = 1,
        norm: Optional[str] = "batch",
        act: Optional[str] = "relu",
    ):
        super().__init__()
        layers = []
        for i in range(repeat):
            in_ch = in_channels if i == 0 else out_channels
            layers.append(nn.Conv3d(in_ch, out_channels, kernel_size, padding=padding))
            if norm == "batch":
                layers.append(nn.BatchNorm3d(out_channels))
            if act == "relu":
                layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class PatchAggregator(nn.Module):
    """
    Aggregate patch predictions into full volume.

    Uses Gaussian weighting for smooth blending and optionally
    includes a learnable aggregation module that combines
    local patch predictions with global context.

    Args:
        vol_size: Full volume spatial size (D, H, W)
        patch_size: Patch spatial size (Pd, Ph, Pw)
        down_size_rate: Downsampling rate for global branch
        num_classes: Number of output classes
        add_aggregation_module: Whether to use learnable aggregation
    """

    def __init__(
        self,
        vol_size: Tuple[int, int, int],
        patch_size: Tuple[int, int, int],
        down_size_rate: Tuple[int, int, int] = (2, 2, 2),
        num_classes: int = 1,
        add_aggregation_module: bool = False,
    ):
        super().__init__()

        self.vol_size = vol_size
        self.patch_size = patch_size
        self.down_size_rate = down_size_rate
        self.num_classes = num_classes
        self.add_aggregation_module = add_aggregation_module

        # Gaussian importance weighting for patches
        fixed_wt = compute_importance_map(
            self.patch_size,
            mode="gaussian",
            sigma_scale=0.125,
            dtype=torch.float,
        )[None, None]  # [1, 1, Pd, Ph, Pw]
        self.register_buffer("fixed_wt", fixed_wt)

        # Weight map for normalization (prevents division by zero)
        weight_map = torch.zeros(1, 1, *self.vol_size) + 1e-20
        self.register_buffer("weight_map", weight_map)

        # Learnable aggregation module (optional)
        if add_aggregation_module:
            self.aggregate = nn.Sequential(
                ConvBnReLU(num_classes * 2, num_classes * 2, 3, 1, repeat=2, norm=None),
                ConvBnReLU(num_classes * 2, num_classes, 3, 1, repeat=2, norm=None, act=None),
            )

    def forward(
        self,
        patch_logits: torch.Tensor,  # [K*B, C, Pd, Ph, Pw]
        global_logit: torch.Tensor,  # [B, C, Dg, Hg, Wg]
        slice_meta: List[Tuple[slice, slice, slice]],
        batch_size: int = 1,
    ) -> torch.Tensor:
        """
        Aggregate patch predictions with global prediction.

        Args:
            patch_logits: Patch-level predictions [K*B, C, Pd, Ph, Pw]
            global_logit: Global low-res prediction [B, C, Dg, Hg, Wg]
            slice_meta: List of slice tuples for each patch
            batch_size: Batch size

        Returns:
            final_logit: Aggregated full-resolution prediction [B, C, D, H, W]
        """
        device = patch_logits.device
        K = len(slice_meta) // batch_size

        # Compute weight map for current patch configuration
        weight_map = self._compute_weight_map(slice_meta, batch_size)

        # Initialize output
        final_logit = torch.zeros(
            batch_size,
            self.num_classes,
            *self.vol_size,
            device=device,
        )

        # Upsample global prediction to full resolution
        upsized_global_logit = F.interpolate(
            global_logit,
            size=self.vol_size,
            mode='trilinear',
            align_corners=False,
        )

        # Track which regions have been covered by patches
        mask = torch.zeros(batch_size, 1, *self.vol_size, device=device)

        # Paste patches with Gaussian weighting
        for i, patch_logit in enumerate(patch_logits):
            batch_idx = i // K
            slices = [slice(batch_idx, batch_idx + 1)] + list(slice_meta[i])

            if self.add_aggregation_module:
                # Combine patch with global context
                patch_with_context = torch.cat([
                    patch_logit.unsqueeze(0) if patch_logit.dim() == 4 else patch_logit,
                    upsized_global_logit[slices],
                ], dim=1)
                aggregated_patch = self.aggregate(patch_with_context)
            else:
                aggregated_patch = patch_logit.unsqueeze(0) if patch_logit.dim() == 4 else patch_logit

            # Paste with Gaussian weighting
            final_logit[slices] = final_logit[slices] + aggregated_patch * self.fixed_wt
            mask[slices] = 1

        # Normalize by accumulated weights
        final_logit = final_logit / weight_map

        # Fill uncovered regions with upsampled global prediction
        final_logit = final_logit + torch.where(mask == 1, 0, upsized_global_logit)

        return final_logit

    @torch.no_grad()
    def _compute_weight_map(
        self,
        slice_meta: List[Tuple[slice, slice, slice]],
        batch_size: int,
    ) -> torch.Tensor:
        """Compute accumulated Gaussian weights for normalization."""
        K = len(slice_meta) // batch_size

        weight_map = self.weight_map.repeat(batch_size, 1, 1, 1, 1)

        for i, s in enumerate(slice_meta):
            batch_idx = i // K
            slices = [slice(batch_idx, batch_idx + 1)] + list(s)
            weight_map[slices] = weight_map[slices] + self.fixed_wt

        return weight_map


class DynamicPatchAggregator(nn.Module):
    """
    Dynamic patch aggregator that handles variable volume sizes.

    Unlike PatchAggregator which requires fixed vol_size,
    this version adapts to the input volume size.
    """

    def __init__(
        self,
        patch_size: Tuple[int, int, int],
        down_size_rate: Tuple[int, int, int] = (2, 2, 2),
        num_classes: int = 1,
        add_aggregation_module: bool = False,
    ):
        super().__init__()

        self.patch_size = patch_size
        self.down_size_rate = down_size_rate
        self.num_classes = num_classes
        self.add_aggregation_module = add_aggregation_module

        # Gaussian importance weighting for patches
        fixed_wt = compute_importance_map(
            self.patch_size,
            mode="gaussian",
            sigma_scale=0.125,
            dtype=torch.float,
        )[None, None]  # [1, 1, Pd, Ph, Pw]
        self.register_buffer("fixed_wt", fixed_wt)

        # Learnable aggregation module (optional)
        if add_aggregation_module:
            self.aggregate = nn.Sequential(
                ConvBnReLU(num_classes * 2, num_classes * 2, 3, 1, repeat=2, norm=None),
                ConvBnReLU(num_classes * 2, num_classes, 3, 1, repeat=2, norm=None, act=None),
            )

    def forward(
        self,
        patch_logits: torch.Tensor,  # [K*B, C, Pd, Ph, Pw]
        global_logit: torch.Tensor,  # [B, C, Dg, Hg, Wg]
        slice_meta: List[Tuple[slice, slice, slice]],
        vol_size: Tuple[int, int, int],
        batch_size: int = 1,
    ) -> torch.Tensor:
        """
        Aggregate patch predictions with global prediction.

        Args:
            patch_logits: Patch-level predictions [K*B, C, Pd, Ph, Pw]
            global_logit: Global low-res prediction [B, C, Dg, Hg, Wg]
            slice_meta: List of slice tuples for each patch
            vol_size: Full volume size (D, H, W)
            batch_size: Batch size

        Returns:
            final_logit: Aggregated full-resolution prediction [B, C, D, H, W]
        """
        device = patch_logits.device
        K = len(slice_meta) // batch_size

        # Initialize output and weight map
        final_logit = torch.zeros(
            batch_size,
            self.num_classes,
            *vol_size,
            device=device,
        )
        weight_map = torch.zeros(batch_size, 1, *vol_size, device=device) + 1e-20

        # Upsample global prediction to full resolution
        upsized_global_logit = F.interpolate(
            global_logit,
            size=vol_size,
            mode='trilinear',
            align_corners=False,
        )

        # Track which regions have been covered by patches
        mask = torch.zeros(batch_size, 1, *vol_size, device=device)

        # Paste patches with Gaussian weighting
        for i, patch_logit in enumerate(patch_logits):
            batch_idx = i // K
            # slices for 5D tensor: [batch, channel, d, h, w]
            spatial_slices = list(slice_meta[i])
            slices = (slice(batch_idx, batch_idx + 1), slice(None)) + tuple(spatial_slices)
            slices_1ch = (slice(batch_idx, batch_idx + 1), slice(0, 1)) + tuple(spatial_slices)

            if self.add_aggregation_module:
                # Combine patch with global context
                patch_with_context = torch.cat([
                    patch_logit.unsqueeze(0) if patch_logit.dim() == 4 else patch_logit,
                    upsized_global_logit[slices],
                ], dim=1)
                aggregated_patch = self.aggregate(patch_with_context)
            else:
                aggregated_patch = patch_logit.unsqueeze(0) if patch_logit.dim() == 4 else patch_logit

            # Paste with Gaussian weighting
            final_logit[slices] = final_logit[slices] + aggregated_patch * self.fixed_wt
            weight_map[slices_1ch] = weight_map[slices_1ch] + self.fixed_wt
            mask[slices_1ch] = 1

        # Normalize by accumulated weights
        final_logit = final_logit / weight_map

        # Fill uncovered regions with upsampled global prediction
        final_logit = final_logit + torch.where(mask == 1, 0, upsized_global_logit)

        return final_logit
