"""Legacy PatchICL implementation (preserved for reference)."""
from src.models.patch_icl_v1.patch_icl import PatchICL, PatchICL_Level, extract_patch_features
from src.models.patch_icl_v1.sampling import (
    ContinuousSampler,
    DeterministicTopKSampler,
    GumbelSoftmaxSampler,
    PatchAugmenter,
    PatchSampler,
    SlidingWindowSampler,
    UniformSampler,
)
from src.models.patch_icl_v1.aggregate import (
    ConfidenceAggregator,
    GaussianAggregator,
    LearnedAggregator,
    LearnedCombineAggregator,
    PatchAggregator,
    create_aggregator,
)

__all__ = [
    "PatchICL",
    "PatchICL_Level",
    "extract_patch_features",
    "ContinuousSampler",
    "DeterministicTopKSampler",
    "GumbelSoftmaxSampler",
    "PatchAugmenter",
    "PatchSampler",
    "SlidingWindowSampler",
    "UniformSampler",
    "ConfidenceAggregator",
    "GaussianAggregator",
    "LearnedAggregator",
    "LearnedCombineAggregator",
    "PatchAggregator",
    "create_aggregator",
]
