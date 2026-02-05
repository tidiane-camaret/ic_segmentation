"""Configuration for feature extraction experiments."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class ExperimentConfig:
    """Base configuration for feature extraction experiments."""

    # Paths
    root_dir: str = "/data/TotalSeg2D"
    stats_path: str = "/data/TotalSeg2D/totalseg_stats.pkl"
    checkpoint_path: Optional[str] = None
    meddino_path: str = "/software/notebooks/camaret/repos/MedDINOv3/checkpoints/MedDINOv3.pth"
    results_dir: str = "./results/feature_extraction"

    # Dataset settings
    labels: List[str] = field(default_factory=lambda: ["liver", "spleen", "kidney_left", "kidney_right", "aorta"])
    context_size: int = 3
    image_size: tuple = (256, 256)
    max_samples: int = 200  # Limit samples for faster experiments

    # Model settings
    device: str = "cuda"
    batch_size: int = 16

    # MedDINO settings
    meddino_target_size: int = 256
    meddino_layers: List[int] = field(default_factory=lambda: [2, 5, 8, 11])

    # MedSAM settings
    medsam_checkpoint: Optional[str] = None  # Will download from HuggingFace if None


@dataclass
class LayerComparisonConfig(ExperimentConfig):
    """Configuration for layer comparison experiment."""

    # Layers to compare (early -> late)
    layers_to_test: List[int] = field(default_factory=lambda: [2, 5, 8, 11])

    # Whether to recompute features or try to load from cache
    use_cached_features: bool = True


@dataclass
class MultiLayerFusionConfig(ExperimentConfig):
    """Configuration for multi-layer fusion experiment."""

    # Layers to fuse
    fusion_layers: List[int] = field(default_factory=lambda: [2, 5, 8, 11])

    # Fusion strategies to test
    fusion_strategies: List[str] = field(default_factory=lambda: ["average", "learned_weighted", "concat_proj"])

    # Training settings for learned fusion
    fusion_lr: float = 1e-4
    fusion_epochs: int = 10


@dataclass
class MedSAMConfig(ExperimentConfig):
    """Configuration for MedSAM feature extraction experiment."""

    # MedSAM v1 settings
    medsam_input_size: int = 1024  # MedSAM expects 1024x1024

    # How to adapt 64x64 MedSAM features to 14x14 grid
    # Options: "downsample", "patch_average", "learned_proj"
    feature_adaptation: str = "downsample"


# Default experiment configurations
LAYER_COMPARISON_CONFIG = LayerComparisonConfig()
MULTILAYER_FUSION_CONFIG = MultiLayerFusionConfig()
MEDSAM_CONFIG = MedSAMConfig()


def get_config(experiment_type: str = "layer_comparison", **overrides) -> ExperimentConfig:
    """Get experiment configuration with optional overrides.

    Args:
        experiment_type: One of "layer_comparison", "multilayer_fusion", "medsam"
        **overrides: Override any config parameter

    Returns:
        Configured experiment config
    """
    configs = {
        "layer_comparison": LayerComparisonConfig,
        "multilayer_fusion": MultiLayerFusionConfig,
        "medsam": MedSAMConfig,
    }

    if experiment_type not in configs:
        raise ValueError(f"Unknown experiment type: {experiment_type}. "
                        f"Available: {list(configs.keys())}")

    config = configs[experiment_type]()

    # Apply overrides
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            raise ValueError(f"Unknown config parameter: {key}")

    return config
