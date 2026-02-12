"""
Simplified patch aggregation for PatchICL v2.

Only includes PatchAggregator (average) and GaussianAggregator - the two
strategies actually used by v1 and v2 experiment configs.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchAggregator(nn.Module):
    """Base patch aggregator using simple averaging."""

    def __init__(
        self,
        patch_size: int,
        combine_mode: str = "average",
        combine_weight: float = 0.5,
        fill_uncovered: str = "zero",
        min_coverage: float = 1e-6,
    ):
        """
        Args:
            patch_size: Size of patches (ps x ps)
            combine_mode: How to combine with previous prediction:
                - "average": Weighted average everywhere
                - "replace": Use only current prediction
                - "coverage": Use prev only where no patch coverage
            combine_weight: Weight for current prediction when combine_mode="average"
            fill_uncovered: "zero" fills uncovered with large negative logit, "prev" leaves as 0
            min_coverage: Minimum weight to consider a pixel covered
        """
        super().__init__()
        self.patch_size = patch_size
        self.combine_mode = combine_mode
        self.combine_weight = combine_weight
        self.fill_uncovered = fill_uncovered
        self.min_coverage = min_coverage

    def _compute_patch_weights(
        self,
        patch_logits: torch.Tensor,
        coords: torch.Tensor,
        output_size: tuple[int, int],
    ) -> torch.Tensor:
        """Compute weights for each patch pixel. Default: uniform (all 1s)."""
        return torch.ones_like(patch_logits)

    def _combine_with_prev(
        self,
        aggregated: torch.Tensor,
        prev_pred: torch.Tensor,
        counts: torch.Tensor,
    ) -> torch.Tensor:
        """Combine aggregated prediction with previous level prediction."""
        if self.combine_mode == "replace":
            return aggregated
        elif self.combine_mode == "coverage":
            covered = counts > self.min_coverage
            return torch.where(covered, aggregated, prev_pred)
        else:  # average
            return self.combine_weight * aggregated + (1 - self.combine_weight) * prev_pred

    def forward(
        self,
        patch_logits: torch.Tensor,
        coords: torch.Tensor,
        output_size: tuple[int, int],
        prev_pred: torch.Tensor = None,
    ) -> torch.Tensor:
        """Aggregate patch predictions back to a full mask (vectorized)."""
        B, K, C, ps_h, ps_w = patch_logits.shape
        ps = self.patch_size
        H, W = output_size
        device = patch_logits.device

        weights = self._compute_patch_weights(patch_logits, coords, output_size)

        # Build output position grid for all patches (no Python loops / .item() syncs)
        row_offsets = torch.arange(ps, device=device)
        col_offsets = torch.arange(ps, device=device)
        # [B, K, ps] + [ps] -> [B, K, ps]
        patch_rows = coords[:, :, 0:1].long() + row_offsets.view(1, 1, -1)
        patch_cols = coords[:, :, 1:2].long() + col_offsets.view(1, 1, -1)
        # Expand to 2D grid: [B, K, ps, ps]
        rows_2d = patch_rows.unsqueeze(-1).expand(B, K, ps, ps)
        cols_2d = patch_cols.unsqueeze(-2).expand(B, K, ps, ps)

        # Validity mask for boundary clipping
        valid = (rows_2d >= 0) & (rows_2d < H) & (cols_2d >= 0) & (cols_2d < W)
        valid_f = valid.unsqueeze(2).float()  # [B, K, 1, ps, ps]

        # Flat scatter indices (clamped; invalid positions are masked via valid_f)
        flat_idx = rows_2d.clamp(0, H - 1) * W + cols_2d.clamp(0, W - 1)  # [B, K, ps, ps]
        flat_idx = flat_idx.reshape(B, -1).unsqueeze(1).expand(B, C, K * ps * ps)

        # Prepare values with validity mask
        weighted = (patch_logits * weights * valid_f).permute(0, 2, 1, 3, 4).reshape(B, C, -1)
        w_masked = (weights * valid_f).permute(0, 2, 1, 3, 4).reshape(B, C, -1)

        output = torch.zeros(B, C, H * W, device=device)
        counts = torch.zeros(B, C, H * W, device=device)
        output.scatter_add_(2, flat_idx, weighted)
        counts.scatter_add_(2, flat_idx, w_masked)
        output = output.reshape(B, C, H, W)
        counts = counts.reshape(B, C, H, W)

        covered = counts > self.min_coverage
        counts_safe = counts.clamp(min=self.min_coverage)
        aggregated = output / counts_safe

        if prev_pred is not None:
            prev_resized = F.interpolate(prev_pred, size=(H, W), mode='bilinear', align_corners=False)
            aggregated = self._combine_with_prev(aggregated, prev_resized, counts)
        elif self.fill_uncovered == "zero":
            aggregated = torch.where(covered, aggregated, torch.full_like(aggregated, -10.0))

        return aggregated


class GaussianAggregator(PatchAggregator):
    """Aggregator with Gaussian weighting from patch centers."""

    def __init__(
        self,
        patch_size: int,
        sigma_ratio: float = 0.125,
        combine_mode: str = "coverage",
        combine_weight: float = 0.5,
        fill_uncovered: str = "zero",
        min_coverage: float = 1e-6,
    ):
        super().__init__(patch_size, combine_mode, combine_weight, fill_uncovered, min_coverage)
        self.sigma_ratio = sigma_ratio
        self._precompute_gaussian()

    def _precompute_gaussian(self):
        """Precompute 2D Gaussian kernel for patch weighting."""
        ps = self.patch_size
        sigma = ps * self.sigma_ratio
        x = torch.arange(ps, dtype=torch.float32) - (ps - 1) / 2
        y = torch.arange(ps, dtype=torch.float32) - (ps - 1) / 2
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        gaussian = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        gaussian = gaussian / gaussian.max()
        self.register_buffer('gaussian_kernel', gaussian.view(1, 1, ps, ps))

    def _compute_patch_weights(
        self,
        patch_logits: torch.Tensor,
        coords: torch.Tensor,
        output_size: tuple[int, int],
    ) -> torch.Tensor:
        """Return Gaussian weights for each patch."""
        B, K = patch_logits.shape[:2]
        return self.gaussian_kernel.expand(B, K, -1, -1, -1)


def create_aggregator(
    aggregator_type: str,
    patch_size: int,
    **kwargs,
) -> PatchAggregator:
    """Factory function to create aggregators from config."""
    combine_mode = kwargs.get('combine_mode', 'coverage')
    combine_weight = kwargs.get('combine_weight', 0.5)
    fill_uncovered = kwargs.get('fill_uncovered', 'zero')
    min_coverage = kwargs.get('min_coverage', 1e-6)

    if aggregator_type == "gaussian":
        return GaussianAggregator(
            patch_size=patch_size,
            sigma_ratio=kwargs.get('sigma_ratio', 0.125),
            combine_mode=combine_mode,
            combine_weight=combine_weight,
            fill_uncovered=fill_uncovered,
            min_coverage=min_coverage,
        )
    else:  # "average" or default
        return PatchAggregator(
            patch_size=patch_size,
            combine_mode=combine_mode,
            combine_weight=combine_weight,
            fill_uncovered=fill_uncovered,
            min_coverage=min_coverage,
        )
