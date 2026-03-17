"""PatchICL v3 - Clean, simplified implementation.

Key improvements over v2:
- Single level combination mode (additive fusion only)
- Unified entropy/sampling weight computation
- Loop-based encoder/decoder (no code duplication for different grid sizes)
- Removed dead code (learned_alpha, sampling_map_blend, gt_border, etc.)
- Config via dataclasses for type safety

~57% reduction in total code lines while maintaining full functionality.
"""
from .model import PatchICL, PatchICLConfig
from .utils import (
    compute_binary_entropy,
    compute_confidence_map,
    compute_entropy_sampling_map,
    create_coverage_mask,
    downsample_mask,
    extract_mask_patches,
    extract_patch_features,
    mask_to_weights,
    resize_label,
)
from .sampling import (
    ContinuousSampler,
    PatchAugmenter,
    SlidingWindowSampler,
    compute_sampling_weights,
    create_sampler,
)
from .aggregate import (
    GaussianAggregator,
    PatchAggregator,
    create_aggregator,
)
from .level import LevelConfig, LevelOutput, process_level
from .metrics import (
    PRED_THRESHOLD,
    GT_AREA_THRESHOLD,
    compute_dice,
    compute_all_metrics,
    compute_per_sample_dice,
    compute_level_metrics,
    compute_hierarchical_metrics,
    compute_context_metrics,
    compute_uncertainty_metrics,
)
from .backbone import (
    SimpleBackbone,
    CNNEncoder,
    CNNDecoder,
    CrossPatchAttention,
    TransformerBlock,
    ContinuousScaleEncoding,
    MaskPriorEncoder,
    ResolutionConditionedNorm,
    LayerNorm2d,
)

__all__ = [
    # Main model
    "PatchICL",
    "PatchICLConfig",
    # Level processing
    "LevelConfig",
    "LevelOutput",
    "process_level",
    # Utilities
    "compute_binary_entropy",
    "compute_confidence_map",
    "compute_entropy_sampling_map",
    "create_coverage_mask",
    "downsample_mask",
    "extract_mask_patches",
    "extract_patch_features",
    "mask_to_weights",
    "resize_label",
    # Sampling
    "ContinuousSampler",
    "PatchAugmenter",
    "SlidingWindowSampler",
    "compute_sampling_weights",
    "create_sampler",
    # Aggregation
    "GaussianAggregator",
    "PatchAggregator",
    "create_aggregator",
    # Metrics
    "PRED_THRESHOLD",
    "GT_AREA_THRESHOLD",
    "compute_dice",
    "compute_all_metrics",
    "compute_per_sample_dice",
    "compute_level_metrics",
    "compute_hierarchical_metrics",
    "compute_context_metrics",
    "compute_uncertainty_metrics",
    # Backbone
    "SimpleBackbone",
    "CNNEncoder",
    "CNNDecoder",
    "CrossPatchAttention",
    "TransformerBlock",
    "ContinuousScaleEncoding",
    "MaskPriorEncoder",
    "ResolutionConditionedNorm",
    "LayerNorm2d",
]
