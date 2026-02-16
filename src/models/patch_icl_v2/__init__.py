"""Simplified PatchICL v2 implementation."""
from src.models.patch_icl_v2.patch_icl import PatchICL, extract_patch_features
from src.models.patch_icl_v2.sampling import (
    ContinuousSampler,
    PatchAugmenter,
    SlidingWindowSampler,
    create_sampler,
)
from src.models.patch_icl_v2.aggregate import (
    GaussianAggregator,
    PatchAggregator,
    create_aggregator,
)
from src.models.patch_icl_v2.metrics import (
    PRED_THRESHOLD,
    GT_AREA_THRESHOLD,
    compute_dice,
    compute_pixel_mae,
    compute_all_metrics,
    compute_per_sample_dice,
)

__all__ = [
    "PatchICL",
    "extract_patch_features",
    "ContinuousSampler",
    "PatchAugmenter",
    "SlidingWindowSampler",
    "create_sampler",
    "GaussianAggregator",
    "PatchAggregator",
    "create_aggregator",
    "PRED_THRESHOLD",
    "GT_AREA_THRESHOLD",
    "compute_dice",
    "compute_all_metrics",
    "compute_per_sample_dice",
]
