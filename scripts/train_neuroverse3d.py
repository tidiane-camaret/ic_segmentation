"""
Neuroverse3D Training Pipeline

This script provides a training pipeline for the Neuroverse3D model,
adapted for the ic_segmentation project structure.

Usage:
    # Stage 1 training (fixed context size)
    python scripts/train_neuroverse3d.py --stage 1 --context-size 3 --epochs 50 --lr 0.00001

    # Stage 2 training (variable context size)
    python scripts/train_neuroverse3d.py --stage 2 --checkpoint /path/to/stage1.ckpt --epochs 100 --lr 0.000002
"""

import argparse
import os
import sys
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))


class Neuroverse3DTrainingPipeline:
    """
    Training pipeline for Neuroverse3D model with two-stage training approach.

    Stage 1: Train with fixed context size (e.g., 3 examples)
    Stage 2: Fine-tune with variable context size (2-9 examples)
    """

    def __init__(self, config):
        """
        Initialize training pipeline.

        Args:
            config: Configuration object or dictionary with training parameters
        """
        self.config = config
        self.setup_environment()

    def setup_environment(self):
        """Configure environment variables for distributed training."""
        # Disable InfiniBand for compatibility
        os.environ['NCCL_IB_DISABLE'] = '1'
        # Use socket for NCCL communication
        os.environ['NCCL_SOCKET_IFNAME'] = 'lo'
        os.environ['NCCL_P2P_DISABLE'] = '1'

    def prepare_data(self):
        """
        Prepare datasets for training and validation.

        Note: This requires implementing a custom dataset loader based on your data format.
        The dataset should follow nnUNet naming conventions:
        - imagesTr/: Training images (*.nii.gz)
        - labelsTr/: Training labels (*.nii.gz)
        """
        from src.neuroverse3d.dataloader import MetaDataset_Multi_Extended
        from src.neuroverse3d.config import create_dataset_config

        # Create dataset configuration
        dataset_config = create_dataset_config(
            data_dir=self.config.data_dir,
            datasets=self.config.datasets,
            sample_rate=self.config.sample_rate,
        )

        # Create training dataset
        train_dataset = MetaDataset_Multi_Extended(
            mode='train',
            config=dataset_config,
            group_size=self.config.context_size + 1,  # context + query
            random_context=self.config.stage == 2,  # Stage 2 uses random context
            min_context=2 if self.config.stage == 2 else self.config.context_size,
            max_context=9 if self.config.stage == 2 else self.config.context_size,
        )

        # Create validation dataset
        val_dataset = MetaDataset_Multi_Extended(
            mode='val',
            config=dataset_config,
            group_size=self.config.context_size + 1,
            random_context=False,  # Validation uses fixed context
        )

        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True,
            drop_last=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )

        return train_loader, val_loader

    def create_model(self):
        """
        Create or load Neuroverse3D model.

        Returns:
            LightningModule: Neuroverse3D model
        """
        from src.neuroverse3d.lightning_model import LightningModel

        if self.config.checkpoint:
            # Load from checkpoint
            print(f"Loading model from checkpoint: {self.config.checkpoint}")
            model = LightningModel.load_from_checkpoint(
                self.config.checkpoint,
                learning_rate=self.config.lr,
                strict=False,  # Allow loading with different parameters
            )
        else:
            # Create new model
            print("Creating new Neuroverse3D model")
            model = LightningModel(
                learning_rate=self.config.lr,
                # Add other model parameters here
            )

        return model

    def create_callbacks(self):
        """Create training callbacks."""
        callbacks = []

        # Model checkpoint callback
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.config.checkpoint_dir,
            filename=f'neuroverse3d-stage{self.config.stage}-{{epoch:02d}}-{{val_loss:.4f}}',
            monitor='val_loss',
            mode='min',
            save_top_k=3,
            save_last=True,
        )
        callbacks.append(checkpoint_callback)

        # Learning rate monitor
        lr_monitor = LearningRateMonitor(logging_interval='step')
        callbacks.append(lr_monitor)

        return callbacks

    def create_logger(self):
        """Create W&B logger for experiment tracking."""
        if self.config.use_wandb:
            logger = WandbLogger(
                project=self.config.project_name,
                name=f'neuroverse3d_stage{self.config.stage}',
                log_model=True,
                config=vars(self.config),
            )
            return logger
        return None

    def train(self):
        """Execute training pipeline."""
        print(f"\n{'='*60}")
        print(f"Starting Neuroverse3D Training - Stage {self.config.stage}")
        print(f"{'='*60}\n")

        # Prepare data
        print("Preparing datasets...")
        train_loader, val_loader = self.prepare_data()
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")

        # Create model
        print("\nInitializing model...")
        model = self.create_model()

        # Setup callbacks and logger
        callbacks = self.create_callbacks()
        logger = self.create_logger()

        # Create trainer
        print("\nConfiguring trainer...")
        trainer = pl.Trainer(
            max_epochs=self.config.epochs,
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            devices=self.config.num_gpus,
            strategy='ddp' if self.config.num_gpus > 1 else 'auto',
            callbacks=callbacks,
            logger=logger,
            precision='16-mixed' if self.config.mixed_precision else 32,
            gradient_clip_val=self.config.gradient_clip_val,
            log_every_n_steps=self.config.log_interval,
            val_check_interval=self.config.val_check_interval,
        )

        # Start training
        print("\nStarting training...\n")
        trainer.fit(model, train_loader, val_loader)

        print(f"\n{'='*60}")
        print("Training completed!")
        print(f"Best checkpoint: {trainer.checkpoint_callback.best_model_path}")
        print(f"{'='*60}\n")

        return trainer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Neuroverse3D Training Pipeline'
    )

    # Stage configuration
    parser.add_argument(
        '--stage',
        type=int,
        choices=[1, 2],
        default=1,
        help='Training stage: 1 (fixed context) or 2 (variable context)'
    )

    # Data configuration
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/neuroverse3d',
        help='Directory containing training data in nnUNet format'
    )
    parser.add_argument(
        '--datasets',
        type=str,
        nargs='+',
        default=None,
        help='List of dataset names to use for training'
    )
    parser.add_argument(
        '--sample-rate',
        type=float,
        default=0.25,
        help='Sampling rate for training data'
    )

    # Training configuration
    parser.add_argument(
        '--context-size',
        type=int,
        default=3,
        help='Number of context examples (Stage 1 only)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1,
        help='Batch size for training'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.00001,
        help='Learning rate (Stage 1: 0.00001, Stage 2: 0.000002)'
    )

    # Model configuration
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to checkpoint to resume from (required for Stage 2)'
    )
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default='checkpoints/neuroverse3d',
        help='Directory to save checkpoints'
    )

    # Hardware configuration
    parser.add_argument(
        '--num-gpus',
        type=int,
        default=1,
        help='Number of GPUs to use'
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=4,
        help='Number of data loading workers'
    )
    parser.add_argument(
        '--mixed-precision',
        action='store_true',
        help='Use mixed precision training'
    )

    # Training parameters
    parser.add_argument(
        '--gradient-clip-val',
        type=float,
        default=1.0,
        help='Gradient clipping value'
    )
    parser.add_argument(
        '--log-interval',
        type=int,
        default=10,
        help='Logging interval (steps)'
    )
    parser.add_argument(
        '--val-check-interval',
        type=float,
        default=1.0,
        help='Validation check interval (epochs or fraction)'
    )

    # Experiment tracking
    parser.add_argument(
        '--project-name',
        type=str,
        default='ic_segmentation',
        help='W&B project name'
    )
    parser.add_argument(
        '--use-wandb',
        action='store_true',
        default=True,
        help='Use Weights & Biases for logging'
    )
    parser.add_argument(
        '--no-wandb',
        action='store_true',
        help='Disable Weights & Biases logging'
    )

    args = parser.parse_args()

    # Handle wandb flag
    if args.no_wandb:
        args.use_wandb = False

    # Validate Stage 2 requirements
    if args.stage == 2 and args.checkpoint is None:
        parser.error("Stage 2 training requires --checkpoint argument")

    # Set recommended learning rates if not specified
    if args.lr == 0.00001 and args.stage == 2:
        print("Automatically setting learning rate to 0.000002 for Stage 2")
        args.lr = 0.000002

    return args


def main():
    """Main training entry point."""
    # Parse arguments
    args = parse_args()

    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Print configuration
    print("\nTraining Configuration:")
    print(f"{'='*60}")
    for arg, value in vars(args).items():
        print(f"{arg:25s}: {value}")
    print(f"{'='*60}\n")

    # Create and run training pipeline
    pipeline = Neuroverse3DTrainingPipeline(args)
    trainer = pipeline.train()

    return trainer


if __name__ == '__main__':
    main()
