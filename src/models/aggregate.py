"""
Patch Aggregation Module for PatchICL.

Provides modular aggregation strategies for combining patch predictions
back into full-resolution masks.

Inspired by NMSW (No More Sliding Window) aggregation approach.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchAggregator(nn.Module):
    """
    Base patch aggregator using simple averaging.

    Aggregates patch predictions back to a full mask by placing patches
    at their coordinates and averaging overlapping regions.

    Subclasses can override `_compute_weights` and `_combine_with_prev`
    to implement different aggregation strategies.
    """

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
                - "average": Weighted average everywhere (default)
                - "replace": Use only current prediction (ignore prev)
                - "coverage": Use prev only where no patch coverage (NMSW style)
            combine_weight: Fixed weight for current prediction when combine_mode="average"
                (weight for prev = 1 - combine_weight)
            fill_uncovered: How to fill regions with no patch coverage (when no prev_pred):
                - "zero": Fill with large negative logit so sigmoid ≈ 0 (default)
                - "prev": Leave as logit=0 (sigmoid = 0.5)
            min_coverage: Minimum weight threshold to consider a pixel "covered"
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
        """
        Compute weights for each patch pixel during aggregation.

        Default: uniform weights (all 1s).

        Override in subclasses for confidence-based or Gaussian weighting.

        Args:
            patch_logits: [B, K, 1, ps, ps] - patch predictions
            coords: [B, K, 2] - patch coordinates
            output_size: (H, W) - output mask size

        Returns:
            weights: [B, K, 1, ps, ps] - per-pixel weights
        """
        return torch.ones_like(patch_logits)

    def _combine_with_prev(
        self,
        aggregated: torch.Tensor,
        prev_pred: torch.Tensor,
        counts: torch.Tensor,
    ) -> torch.Tensor:
        """
        Combine aggregated prediction with previous level prediction.

        Args:
            aggregated: [B, 1, H, W] - current aggregated prediction
            prev_pred: [B, 1, H, W] - previous level prediction (resized)
            counts: [B, 1, H, W] - accumulated weights covering each pixel

        Returns:
            combined: [B, 1, H, W] - combined prediction
        """
        if self.combine_mode == "replace":
            return aggregated
        elif self.combine_mode == "coverage":
            # NMSW style: use prev only where no patch coverage
            # covered = counts > min_coverage
            covered = counts > self.min_coverage
            return torch.where(covered, aggregated, prev_pred)
        elif self.combine_mode == "average":
            return self.combine_weight * aggregated + (1 - self.combine_weight) * prev_pred
        else:
            # Default to simple average
            return (aggregated + prev_pred) / 2

    def forward(
        self,
        patch_logits: torch.Tensor,
        coords: torch.Tensor,
        output_size: tuple[int, int],
        prev_pred: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Aggregate patch predictions back to a full mask.

        Args:
            patch_logits: [B, K, 1, ps, ps] - predictions for each patch
            coords: [B, K, 2] - patch coordinates (top-left corner)
            output_size: (H, W) - output mask size
            prev_pred: [B, 1, H, W] - previous level prediction (optional)

        Returns:
            aggregated: [B, 1, H, W] - aggregated prediction at output_size
        """
        B = patch_logits.shape[0]
        K = patch_logits.shape[1]
        ps = self.patch_size
        H, W = output_size
        device = patch_logits.device

        # Compute per-pixel weights for patches
        weights = self._compute_patch_weights(patch_logits, coords, output_size)

        # Initialize output and count tensors
        output = torch.zeros(B, 1, H, W, device=device)
        counts = torch.zeros(B, 1, H, W, device=device)

        # Place patches back with weights
        for b in range(B):
            for k in range(K):
                h, w = coords[b, k].tolist()
                h, w = int(h), int(w)

                # Clamp to valid range
                h_end = min(h + ps, H)
                w_end = min(w + ps, W)
                patch_h = h_end - h
                patch_w = w_end - w

                # Add weighted prediction
                patch_weight = weights[b, k, :, :patch_h, :patch_w]
                output[b, :, h:h_end, w:w_end] += (
                    patch_logits[b, k, :, :patch_h, :patch_w] * patch_weight
                )
                counts[b, :, h:h_end, w:w_end] += patch_weight

        # Identify covered vs uncovered regions
        covered = counts > self.min_coverage

        # Average overlapping regions (only where covered)
        counts_safe = counts.clamp(min=self.min_coverage)
        aggregated = output / counts_safe

        # Handle uncovered regions
        # Note: aggregated contains LOGITS, so uncovered=0 would give sigmoid=0.5
        # We need to fill uncovered with negative logits for probability~0
        if prev_pred is not None:
            prev_resized = F.interpolate(
                prev_pred, size=(H, W), mode='bilinear', align_corners=False
            )
            aggregated = self._combine_with_prev(aggregated, prev_resized, counts)
        elif self.fill_uncovered == "zero":
            # Fill uncovered regions with large negative logit -> sigmoid ≈ 0
            # -10 gives sigmoid(-10) ≈ 4.5e-5 ≈ 0
            aggregated = torch.where(covered, aggregated, torch.full_like(aggregated, -10.0))
        # else fill_uncovered="prev" but no prev_pred - leave as is (logit=0 means prob=0.5)

        return aggregated


class GaussianAggregator(PatchAggregator):
    """
    Aggregator with Gaussian weighting from patch centers.

    Pixels near patch center contribute more to the final prediction,
    reducing boundary artifacts from overlapping patches.

    Default sigma_ratio=0.125 matches NMSW (No More Sliding Window) approach.
    """

    def __init__(
        self,
        patch_size: int,
        sigma_ratio: float = 0.125,
        combine_mode: str = "coverage",
        combine_weight: float = 0.5,
        fill_uncovered: str = "zero",
        min_coverage: float = 1e-6,
    ):
        """
        Args:
            patch_size: Size of patches
            sigma_ratio: Gaussian sigma as ratio of patch size (default: 0.125, NMSW value)
            combine_mode: How to combine with previous prediction
            combine_weight: Weight for current prediction
            fill_uncovered: How to fill uncovered regions (when no prev_pred)
            min_coverage: Minimum coverage threshold
        """
        super().__init__(patch_size, combine_mode, combine_weight, fill_uncovered, min_coverage)
        self.sigma_ratio = sigma_ratio

        # Precompute Gaussian kernel
        self._precompute_gaussian()

    def _precompute_gaussian(self):
        """Precompute 2D Gaussian kernel for patch weighting."""
        ps = self.patch_size
        sigma = ps * self.sigma_ratio

        # Create coordinate grid centered at patch center
        x = torch.arange(ps, dtype=torch.float32) - (ps - 1) / 2
        y = torch.arange(ps, dtype=torch.float32) - (ps - 1) / 2
        xx, yy = torch.meshgrid(x, y, indexing='ij')

        # Compute Gaussian
        gaussian = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        gaussian = gaussian / gaussian.max()  # Normalize to [0, 1]

        # Register as buffer (not a parameter, but moves with model)
        self.register_buffer('gaussian_kernel', gaussian.view(1, 1, ps, ps))

    def _compute_patch_weights(
        self,
        patch_logits: torch.Tensor,
        coords: torch.Tensor,
        output_size: tuple[int, int],
    ) -> torch.Tensor:
        """Return Gaussian weights for each patch."""
        B, K = patch_logits.shape[:2]
        # Expand kernel to match batch and num_patches
        return self.gaussian_kernel.expand(B, K, -1, -1, -1)


class ConfidenceAggregator(PatchAggregator):
    """
    Aggregator with confidence-based weighting.

    Patches with higher prediction confidence (further from 0.5)
    contribute more to the final prediction.
    """

    def __init__(
        self,
        patch_size: int,
        confidence_temperature: float = 2.0,
        combine_mode: str = "coverage",
        combine_weight: float = 0.5,
        fill_uncovered: str = "zero",
        min_coverage: float = 1e-6,
    ):
        """
        Args:
            patch_size: Size of patches
            confidence_temperature: Temperature for confidence scaling
                (higher = more uniform, lower = sharper)
            combine_mode: How to combine with previous prediction
            combine_weight: Weight for current prediction
            fill_uncovered: How to fill uncovered regions (when no prev_pred)
            min_coverage: Minimum coverage threshold
        """
        super().__init__(patch_size, combine_mode, combine_weight, fill_uncovered, min_coverage)
        self.confidence_temperature = confidence_temperature

    def _compute_patch_weights(
        self,
        patch_logits: torch.Tensor,
        coords: torch.Tensor,
        output_size: tuple[int, int],
    ) -> torch.Tensor:
        """Compute confidence-based weights from prediction probabilities."""
        # Convert logits to probabilities
        probs = torch.sigmoid(patch_logits)

        # Confidence = distance from 0.5 (scaled to [0, 1])
        # Confident predictions are near 0 or 1, uncertain are near 0.5
        confidence = (probs - 0.5).abs() * 2  # [0, 1] range

        # Apply temperature scaling
        weights = confidence ** (1 / self.confidence_temperature)

        # Ensure minimum weight to avoid division by zero
        weights = weights.clamp(min=0.1)

        return weights


class LearnedAggregator(PatchAggregator):
    """
    Aggregator with learned patch weighting.

    Uses a small CNN to predict per-pixel weights for each patch
    based on the patch content.
    """

    def __init__(
        self,
        patch_size: int,
        hidden_dim: int = 32,
        combine_mode: str = "coverage",
        combine_weight: float = 0.5,
        fill_uncovered: str = "zero",
        min_coverage: float = 1e-6,
    ):
        """
        Args:
            patch_size: Size of patches
            hidden_dim: Hidden dimension for weight predictor
            combine_mode: How to combine with previous prediction
            combine_weight: Weight for current prediction
            fill_uncovered: How to fill uncovered regions (when no prev_pred)
            min_coverage: Minimum coverage threshold
        """
        super().__init__(patch_size, combine_mode, combine_weight, fill_uncovered, min_coverage)

        # Small CNN to predict weights from patch logits
        self.weight_predictor = nn.Sequential(
            nn.Conv2d(1, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, kernel_size=1),
            nn.Sigmoid(),  # Output weights in [0, 1]
        )

    def _compute_patch_weights(
        self,
        patch_logits: torch.Tensor,
        coords: torch.Tensor,
        output_size: tuple[int, int],
    ) -> torch.Tensor:
        """Predict weights using learned CNN."""
        B, K, C, ps_h, ps_w = patch_logits.shape

        # Reshape for CNN: [B*K, 1, ps, ps]
        logits_flat = patch_logits.view(B * K, C, ps_h, ps_w)

        # Predict weights
        weights_flat = self.weight_predictor(logits_flat)

        # Reshape back: [B, K, 1, ps, ps]
        weights = weights_flat.view(B, K, 1, ps_h, ps_w)

        # Ensure minimum weight
        weights = weights.clamp(min=0.1)

        return weights


class LearnedCombineAggregator(PatchAggregator):
    """
    Aggregator with learned combination of current and previous predictions.

    Uses a small network to predict spatially-varying blend weights
    based on both predictions.
    """

    def __init__(
        self,
        patch_size: int,
        hidden_dim: int = 16,
    ):
        """
        Args:
            patch_size: Size of patches
            hidden_dim: Hidden dimension for blend weight predictor
        """
        super().__init__(patch_size, combine_mode="weighted")

        # Network to predict blend weights from both predictions
        self.blend_predictor = nn.Sequential(
            nn.Conv2d(2, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, kernel_size=1),
            nn.Sigmoid(),  # Output blend weight for current pred in [0, 1]
        )

    def _combine_with_prev(
        self,
        aggregated: torch.Tensor,
        prev_pred: torch.Tensor,
        counts: torch.Tensor,
    ) -> torch.Tensor:
        """Learn spatially-varying blend weights."""
        # Concatenate both predictions
        combined_input = torch.cat([aggregated, prev_pred], dim=1)  # [B, 2, H, W]

        # Predict blend weight for current prediction
        blend_weight = self.blend_predictor(combined_input)  # [B, 1, H, W]

        # Weighted combination
        return blend_weight * aggregated + (1 - blend_weight) * prev_pred


def create_aggregator(
    aggregator_type: str,
    patch_size: int,
    **kwargs,
) -> PatchAggregator:
    """
    Factory function to create aggregators from config.

    Args:
        aggregator_type: Type of aggregator:
            - "average": Simple averaging (default)
            - "gaussian": Gaussian weighting from patch centers (NMSW style)
            - "confidence": Confidence-based weighting
            - "learned": Learned per-pixel weights
            - "learned_combine": Learned combination with previous pred
        patch_size: Size of patches
        **kwargs: Additional arguments for specific aggregator types
            - combine_mode: "average", "replace", or "coverage" (NMSW style)
            - combine_weight: weight for current pred when combine_mode="average"
            - fill_uncovered: "prev" (NMSW style) or "zero"
            - min_coverage: threshold for coverage detection

    Returns:
        PatchAggregator instance
    """
    # Common kwargs
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
    elif aggregator_type == "confidence":
        return ConfidenceAggregator(
            patch_size=patch_size,
            confidence_temperature=kwargs.get('confidence_temperature', 2.0),
            combine_mode=combine_mode,
            combine_weight=combine_weight,
            fill_uncovered=fill_uncovered,
            min_coverage=min_coverage,
        )
    elif aggregator_type == "learned":
        return LearnedAggregator(
            patch_size=patch_size,
            hidden_dim=kwargs.get('hidden_dim', 32),
            combine_mode=combine_mode,
            combine_weight=combine_weight,
            fill_uncovered=fill_uncovered,
            min_coverage=min_coverage,
        )
    elif aggregator_type == "learned_combine":
        return LearnedCombineAggregator(
            patch_size=patch_size,
            hidden_dim=kwargs.get('hidden_dim', 16),
        )
    else:  # Default: "average"
        return PatchAggregator(
            patch_size=patch_size,
            combine_mode=combine_mode,
            combine_weight=combine_weight,
            fill_uncovered=fill_uncovered,
            min_coverage=min_coverage,
        )
