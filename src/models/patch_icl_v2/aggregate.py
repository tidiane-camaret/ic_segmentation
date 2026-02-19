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
        use_confidence: bool = False,
        confidence_mode: str = "multiply",
    ):
        """
        Args:
            patch_size: Size of patches (ps x ps)
            combine_mode: How to combine with previous prediction:
                - "average": Weighted average everywhere
                - "replace": Use only current prediction
                - "coverage": Use prev only where no patch coverage
                - "confidence": Use confidence for blending with previous
            combine_weight: Weight for current prediction when combine_mode="average"
            fill_uncovered: "zero" fills uncovered with large negative logit, "prev" leaves as 0
            min_coverage: Minimum weight to consider a pixel covered
            use_confidence: If True, incorporate confidence into aggregation weights
            confidence_mode: How to use confidence:
                - "multiply": Multiply base weights by confidence
                - "replace": Use confidence as the sole weighting
        """
        super().__init__()
        self.patch_size = patch_size
        self.combine_mode = combine_mode
        self.combine_weight = combine_weight
        self.fill_uncovered = fill_uncovered
        self.min_coverage = min_coverage
        self.use_confidence = use_confidence
        self.confidence_mode = confidence_mode

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
        confidence_map: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Combine aggregated prediction with previous level prediction.

        Args:
            aggregated: Current level aggregated prediction
            prev_pred: Upsampled prediction from previous level
            counts: Weight counts per pixel
            confidence_map: Aggregated confidence map (for combine_mode="confidence")
        """
        if self.combine_mode == "replace":
            return aggregated
        elif self.combine_mode == "coverage":
            covered = counts > self.min_coverage
            return torch.where(covered, aggregated, prev_pred)
        elif self.combine_mode == "confidence" and confidence_map is not None:
            # Blend using aggregated confidence
            # High confidence -> use current, low confidence -> use previous
            return confidence_map * aggregated + (1 - confidence_map) * prev_pred
        else:  # average
            return self.combine_weight * aggregated + (1 - self.combine_weight) * prev_pred

    def forward(
        self,
        patch_logits: torch.Tensor,
        coords: torch.Tensor,
        output_size: tuple[int, int],
        prev_pred: torch.Tensor = None,
        confidence: torch.Tensor | None = None,
        prev_conf: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Aggregate patch predictions back to a full mask (vectorized).

        Args:
            patch_logits: [B, K, C, ps, ps] - patch prediction logits
            coords: [B, K, 2] - patch coordinates
            output_size: (H, W) output resolution
            prev_pred: Previous level prediction for combination
            confidence: [B, K, 1, ps, ps] - patch confidence maps (optional)
            prev_conf: Previous level confidence map for combination

        Returns:
            If confidence is None: aggregated logits [B, C, H, W]
            If confidence is provided: (aggregated logits, aggregated confidence)
        """
        B, K, C, ps_h, ps_w = patch_logits.shape
        ps = self.patch_size
        H, W = output_size
        device = patch_logits.device

        # Compute base aggregation weights (before confidence modulation)
        base_weights = self._compute_patch_weights(patch_logits, coords, output_size)
        aggregation_weights = base_weights

        # Modulate aggregation weights by confidence if enabled
        if self.use_confidence and confidence is not None:
            if self.confidence_mode == "multiply":
                aggregation_weights = base_weights * confidence  # [B, K, 1, ps, ps]
            elif self.confidence_mode == "replace":
                aggregation_weights = confidence

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
        weighted = (patch_logits * aggregation_weights * valid_f).permute(0, 2, 1, 3, 4).reshape(B, C, -1)
        w_masked = (aggregation_weights * valid_f).permute(0, 2, 1, 3, 4).reshape(B, C, -1)

        output = torch.zeros(B, C, H * W, device=device)
        counts = torch.zeros(B, C, H * W, device=device)
        output.scatter_add_(2, flat_idx, weighted)
        counts.scatter_add_(2, flat_idx, w_masked)
        output = output.reshape(B, C, H, W)
        counts = counts.reshape(B, C, H, W)

        covered = counts > self.min_coverage
        counts_safe = counts.clamp(min=self.min_coverage)
        aggregated = output / counts_safe

        # Aggregate confidence if provided
        # Use base_weights (not confidence-modulated) to avoid biasing toward high confidence
        aggregated_conf = None
        if confidence is not None:
            flat_idx_1ch = flat_idx[:, :1]  # [B, 1, K*ps*ps]
            # Use base_weights for unbiased confidence averaging
            base_w_masked = (base_weights * valid_f).permute(0, 2, 1, 3, 4).reshape(B, 1, -1)[:, :1]
            conf_weighted = (confidence * base_weights * valid_f).permute(0, 2, 1, 3, 4).reshape(B, 1, -1)

            conf_output = torch.zeros(B, 1, H * W, device=device)
            conf_counts = torch.zeros(B, 1, H * W, device=device)
            conf_output.scatter_add_(2, flat_idx_1ch, conf_weighted)
            conf_counts.scatter_add_(2, flat_idx_1ch, base_w_masked)
            conf_output = conf_output.reshape(B, 1, H, W)
            conf_counts = conf_counts.reshape(B, 1, H, W)

            conf_counts_safe = conf_counts.clamp(min=self.min_coverage)
            aggregated_conf = conf_output / conf_counts_safe

        if prev_pred is not None:
            prev_resized = F.interpolate(prev_pred, size=(H, W), mode='bilinear', align_corners=False)
            # Use aggregated confidence for confidence-based blending
            conf_for_blend = None
            if self.combine_mode == "confidence" and aggregated_conf is not None:
                conf_for_blend = aggregated_conf
            aggregated = self._combine_with_prev(aggregated, prev_resized, counts, conf_for_blend)
        elif self.fill_uncovered == "zero":
            aggregated = torch.where(covered, aggregated, torch.full_like(aggregated, -10.0))

        if confidence is not None:
            return aggregated, aggregated_conf
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
        use_confidence: bool = False,
        confidence_mode: str = "multiply",
    ):
        super().__init__(
            patch_size, combine_mode, combine_weight, fill_uncovered, min_coverage,
            use_confidence, confidence_mode
        )
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
    use_confidence = kwargs.get('use_confidence', False)
    confidence_mode = kwargs.get('confidence_mode', 'multiply')

    if aggregator_type == "gaussian":
        return GaussianAggregator(
            patch_size=patch_size,
            sigma_ratio=kwargs.get('sigma_ratio', 0.125),
            combine_mode=combine_mode,
            combine_weight=combine_weight,
            fill_uncovered=fill_uncovered,
            min_coverage=min_coverage,
            use_confidence=use_confidence,
            confidence_mode=confidence_mode,
        )
    else:  # "average" or default
        return PatchAggregator(
            patch_size=patch_size,
            combine_mode=combine_mode,
            combine_weight=combine_weight,
            fill_uncovered=fill_uncovered,
            min_coverage=min_coverage,
            use_confidence=use_confidence,
            confidence_mode=confidence_mode,
        )
