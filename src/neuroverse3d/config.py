"""
Configuration utilities for Neuroverse3D training.

This module provides configuration structures for dataset setup,
model parameters, and training hyperparameters.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class DatasetConfig:
    """Configuration for a single dataset."""
    name: str
    path: str
    input_modalities: List[int] = field(default_factory=lambda: [0])  # Which modalities to use
    output_modalities: List[int] = field(default_factory=lambda: [0])  # Which labels to use
    foreground_classes: str = "random"  # "random" or list of class indices
    sample_rate: float = 1.0  # Fraction of dataset to use


@dataclass
class MultiDatasetConfig:
    """Configuration for multiple datasets."""
    datasets: List[DatasetConfig] = field(default_factory=list)
    train_val_split: float = 0.8  # Fraction for training


def create_dataset_config(
    data_dir: str,
    datasets: Optional[List[str]] = None,
    sample_rate: float = 0.25,
) -> MultiDatasetConfig:
    """
    Create a dataset configuration.

    Args:
        data_dir: Base directory containing all datasets
        datasets: List of dataset names (folder names in data_dir)
        sample_rate: Sampling rate for each dataset

    Returns:
        MultiDatasetConfig object
    """
    if datasets is None:
        # Default to looking for nnUNet-formatted datasets in data_dir
        import os
        datasets = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

    config = MultiDatasetConfig()

    for dataset_name in datasets:
        dataset_config = DatasetConfig(
            name=dataset_name,
            path=f"{data_dir}/{dataset_name}",
            sample_rate=sample_rate,
        )
        config.datasets.append(dataset_config)

    return config


def get_default_training_config(stage: int = 1) -> Dict[str, Any]:
    """
    Get default training configuration for a given stage.

    Args:
        stage: Training stage (1 or 2)

    Returns:
        Dictionary of training hyperparameters
    """
    if stage == 1:
        return {
            'learning_rate': 1e-5,
            'epochs': 50,
            'context_size': 3,
            'batch_size': 1,
            'gradient_clip_val': 1.0,
            'warmup_epochs': 5,
        }
    elif stage == 2:
        return {
            'learning_rate': 2e-6,
            'epochs': 100,
            'context_size_min': 2,
            'context_size_max': 9,
            'batch_size': 1,
            'gradient_clip_val': 1.0,
            'warmup_epochs': 0,  # No warmup for fine-tuning
        }
    else:
        raise ValueError(f"Invalid stage: {stage}. Must be 1 or 2.")


def get_model_config() -> Dict[str, Any]:
    """
    Get default model configuration.

    Returns:
        Dictionary of model hyperparameters
    """
    return {
        'input_size': (128, 128, 128),
        'in_channels': 1,
        'hidden_dim': 768,
        'num_heads': 12,
        'num_layers': 12,
        'patch_size': (16, 16, 16),
        'dropout': 0.1,
    }
