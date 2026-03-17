"""Simplified patch aggregation for PatchICL v3.

Only PatchAggregator (average) and GaussianAggregator - the two strategies
actually used. Removed unused combine_mode and prev_pred logic.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class PatchAggregator(nn.Module):
    """Base patch aggregator using scatter-add averaging."""

    def __init__(
        self,
        patch_size: int,
        fill_uncovered: str = "zero",
        min_coverage: float = 1e-6,
        use_sampling_map: bool = False,
        sampling_map_mode: str = "multiply",
        detach_sampling_map: bool = False,
    ):
        """
        Args:
            patch_size: Size of patches (ps x ps)
            fill_uncovered: "zero" fills uncovered with large negative logit
            min_coverage: Minimum weight to consider a pixel covered
            use_sampling_map: If True, incorporate sampling_map into weights
            sampling_map_mode: How to use sampling_map ("multiply" or "replace")
            detach_sampling_map: If True, detach sampling_map before use
        """
        super().__init__()
        self.patch_size = patch_size
        self.fill_uncovered = fill_uncovered
        self.min_coverage = min_coverage
        self.use_sampling_map = use_sampling_map
        self.sampling_map_mode = sampling_map_mode
        self.detach_sampling_map = detach_sampling_map

    def _compute_patch_weights(
        self,
        patch_logits: torch.Tensor,
        coords: torch.Tensor,
        output_size: tuple[int, int],
    ) -> torch.Tensor:
        """Compute weights for each patch pixel. Default: uniform."""
        return torch.ones_like(patch_logits)

    def forward(
        self,
        patch_logits: torch.Tensor,
        coords: torch.Tensor,
        output_size: tuple[int, int],
        sampling_map: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Aggregate patch predictions to full mask.

        Args:
            patch_logits: [B, K, C, ps, ps] - patch prediction logits
            coords: [B, K, 2] - patch coordinates
            output_size: (H, W) output resolution
            sampling_map: [B, K, 1, ps, ps] - per-patch sampling maps

        Returns:
            If sampling_map is None: aggregated logits [B, C, H, W]
            If sampling_map provided: (aggregated, aggregated_sampling_map)
        """
        B, K, C, ps_h, ps_w = patch_logits.shape
        ps = self.patch_size
        H, W = output_size
        device = patch_logits.device

        # Compute base aggregation weights
        base_weights = self._compute_patch_weights(patch_logits, coords, output_size)
        aggregation_weights = base_weights

        # Modulate weights by sampling_map if enabled
        if self.use_sampling_map and sampling_map is not None:
            map_for_weights = sampling_map.detach() if self.detach_sampling_map else sampling_map
            if self.sampling_map_mode == "multiply":
                aggregation_weights = base_weights * map_for_weights
            elif self.sampling_map_mode == "replace":
                aggregation_weights = map_for_weights

        # Build output position grid
        row_offsets = torch.arange(ps, device=device)
        col_offsets = torch.arange(ps, device=device)
        patch_rows = coords[:, :, 0:1].long() + row_offsets.view(1, 1, -1)
        patch_cols = coords[:, :, 1:2].long() + col_offsets.view(1, 1, -1)
        rows_2d = patch_rows.unsqueeze(-1).expand(B, K, ps, ps)
        cols_2d = patch_cols.unsqueeze(-2).expand(B, K, ps, ps)

        # Validity mask
        valid = (rows_2d >= 0) & (rows_2d < H) & (cols_2d >= 0) & (cols_2d < W)
        valid_f = valid.unsqueeze(2).float()

        # Flat scatter indices
        flat_idx = rows_2d.clamp(0, H - 1) * W + cols_2d.clamp(0, W - 1)
        flat_idx = flat_idx.reshape(B, -1).unsqueeze(1).expand(B, C, K * ps * ps)

        # Prepare weighted values
        weighted = (patch_logits * aggregation_weights * valid_f).permute(0, 2, 1, 3, 4).reshape(B, C, -1)
        w_masked = (aggregation_weights * valid_f).permute(0, 2, 1, 3, 4).reshape(B, C, -1)

        # Scatter-add
        output = torch.zeros(B, C, H * W, device=device)
        counts = torch.zeros(B, C, H * W, device=device)
        output.scatter_add_(2, flat_idx, weighted)
        counts.scatter_add_(2, flat_idx, w_masked)
        output = output.reshape(B, C, H, W)
        counts = counts.reshape(B, C, H, W)

        # Average
        covered = counts > self.min_coverage
        counts_safe = counts.clamp(min=self.min_coverage)
        aggregated = output / counts_safe

        # Fill uncovered
        if self.fill_uncovered == "zero":
            aggregated = torch.where(covered, aggregated, torch.full_like(aggregated, -10.0))

        # Aggregate sampling_map if provided
        if sampling_map is not None:
            flat_idx_1ch = flat_idx[:, :1]
            base_w_masked = (base_weights * valid_f).permute(0, 2, 1, 3, 4).reshape(B, 1, -1)[:, :1]
            map_weighted = (sampling_map * base_weights * valid_f).permute(0, 2, 1, 3, 4).reshape(B, 1, -1)

            map_output = torch.zeros(B, 1, H * W, device=device)
            map_counts = torch.zeros(B, 1, H * W, device=device)
            map_output.scatter_add_(2, flat_idx_1ch, map_weighted)
            map_counts.scatter_add_(2, flat_idx_1ch, base_w_masked)
            map_output = map_output.reshape(B, 1, H, W)
            map_counts = map_counts.reshape(B, 1, H, W)

            map_counts_safe = map_counts.clamp(min=self.min_coverage)
            aggregated_sampling_map = map_output / map_counts_safe

            return aggregated, aggregated_sampling_map

        return aggregated


class GaussianAggregator(PatchAggregator):
    """Aggregator with Gaussian weighting from patch centers."""

    def __init__(
        self,
        patch_size: int,
        sigma_ratio: float = 0.125,
        fill_uncovered: str = "zero",
        min_coverage: float = 1e-6,
        use_sampling_map: bool = False,
        sampling_map_mode: str = "multiply",
        detach_sampling_map: bool = False,
    ):
        super().__init__(
            patch_size, fill_uncovered, min_coverage,
            use_sampling_map, sampling_map_mode, detach_sampling_map
        )
        self.sigma_ratio = sigma_ratio
        self._precompute_gaussian()

    def _precompute_gaussian(self):
        """Precompute 2D Gaussian kernel."""
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
        """Return Gaussian weights."""
        B, K = patch_logits.shape[:2]
        return self.gaussian_kernel.expand(B, K, -1, -1, -1)


def create_aggregator(
    aggregator_type: str,
    patch_size: int,
    **kwargs,
) -> PatchAggregator:
    """Factory function to create aggregators from config."""
    fill_uncovered = kwargs.get('fill_uncovered', 'zero')
    min_coverage = kwargs.get('min_coverage', 1e-6)
    use_sampling_map = kwargs.get('use_sampling_map', False)
    sampling_map_mode = kwargs.get('sampling_map_mode', 'multiply')
    detach_sampling_map = kwargs.get('detach_sampling_map', False)

    if aggregator_type == "gaussian":
        return GaussianAggregator(
            patch_size=patch_size,
            sigma_ratio=kwargs.get('sigma_ratio', 0.125),
            fill_uncovered=fill_uncovered,
            min_coverage=min_coverage,
            use_sampling_map=use_sampling_map,
            sampling_map_mode=sampling_map_mode,
            detach_sampling_map=detach_sampling_map,
        )
    else:  # "average" or default
        return PatchAggregator(
            patch_size=patch_size,
            fill_uncovered=fill_uncovered,
            min_coverage=min_coverage,
            use_sampling_map=use_sampling_map,
            sampling_map_mode=sampling_map_mode,
            detach_sampling_map=detach_sampling_map,
        )
