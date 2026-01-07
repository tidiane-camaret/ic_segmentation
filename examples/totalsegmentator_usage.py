"""
Example: Using TotalSegmentator DataLoader with Medverse Model

This script demonstrates how to:
1. Load TotalSegmentator data with the custom dataloader
2. Run inference with the Medverse model
3. Evaluate results
"""

import torch
import sys
from pathlib import Path

# Add medverse to path
sys.path.append(str(Path(__file__).parent.parent))

from medverse.data.totalseg_dataloader import get_dataloader
from medverse.lightning_model import LightningModel
from argparse import Namespace


def demo_dataloader():
    """Demonstrate basic dataloader usage."""
    print("=" * 80)
    print("DEMO 1: Basic DataLoader Usage")
    print("=" * 80)

    # Define organs to segment
    # Each sample will focus on ONE organ at a time
    organ_list = [
        "liver",
        "kidney_left",
        "kidney_right",
        "spleen",
        "pancreas",
    ]

    # Optional: Get empty segmentations from scan
    # from medverse.data.totalseg_utils import scan_dataset
    # stats = scan_dataset("/path/to/TotalSegmentator/dataset")
    # empty_segs = stats['empty_segmentations']

    # Create dataloader
    dataloader = get_dataloader(
        root_dir="/path/to/TotalSegmentator/dataset",
        organ_list=organ_list,
        empty_segmentations=None,  # Provide to exclude cases with zero voxels
        context_size=3,  # Use 3 example pairs
        batch_size=1,
        image_size=(128, 128, 128),
        spacing=(1.5, 1.5, 1.5),
        num_workers=4,
        mode='train',
        shuffle=True,
        num_samples=10,  # Limit to 10 samples for demo
    )

    # Iterate through batches
    for batch_idx, batch in enumerate(dataloader):
        print(f"\nBatch {batch_idx}:")
        print(f"  Target input shape: {batch['target_in'].shape}")  # [1, 1, 128, 128, 128]
        print(f"  Context input shape: {batch['context_in'].shape}")  # [1, 3, 1, 128, 128, 128]
        print(f"  Context output shape: {batch['context_out'].shape}")  # [1, 3, 1, 128, 128, 128]
        print(f"  Target output shape: {batch['target_out'].shape}")  # [1, 1, 128, 128, 128]
        print(f"  Target case: {batch['target_case_ids'][0]}")
        print(f"  Context cases: {batch['context_case_ids'][0]}")

        if batch_idx >= 2:  # Show only first 3 batches
            break


def demo_inference():
    """Demonstrate inference with Medverse model."""
    print("\n" + "=" * 80)
    print("DEMO 2: Inference with Medverse Model")
    print("=" * 80)

    # Load pretrained model
    checkpoint_path = "/path/to/Medverse.ckpt"

    if Path(checkpoint_path).exists():
        model = LightningModel.load_from_checkpoint(checkpoint_path)
        model.eval()
        model.cuda()
        print("✓ Model loaded successfully")
    else:
        print("⚠ Checkpoint not found, creating model from scratch")
        # Create model with default hyperparameters
        hparams = Namespace(
            nb_inner_channels=[32, 64, 128, 256],
            nb_conv_layers_per_stage=2,
            data_slice_only=False,  # 3D mode
        )
        model = LightningModel(hparams)
        model.eval()
        model.cuda()

    # Create dataloader
    organ_list = ["liver"]
    dataloader = get_dataloader(
        root_dir="/path/to/TotalSegmentator/dataset",
        organ_list=organ_list,
        context_size=5,
        batch_size=1,
        image_size=(128, 128, 128),
        num_workers=2,
        mode='val',
        shuffle=False,
        num_samples=5,
    )

    print(f"\nRunning inference on {len(dataloader)} samples...")

    # Run inference
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # Move to GPU
            target_in = batch['target_in'].cuda()
            context_in = batch['context_in'].cuda()
            context_out = batch['context_out'].cuda()

            print(f"\n--- Sample {batch_idx} ---")
            print(f"Case: {batch['target_case_ids'][0]}")

            # Run autoregressive inference
            prediction = model.autoregressive_inference(
                target_in=target_in,
                context_in=context_in,
                context_out=context_out,
                level=2,  # Use 2 resolution levels for speed
                forward_l_arg=3,  # Process 3 contexts at a time
                sw_roi_size=(128, 128, 128),
                sw_overlap=0.25,
                sw_batch_size_val=1,
            )

            print(f"  Input shape: {target_in.shape}")
            print(f"  Prediction shape: {prediction.shape}")
            print(f"  Prediction range: [{prediction.min():.3f}, {prediction.max():.3f}]")

            # Convert to binary mask
            prediction_binary = (prediction > 0.5).float()
            print(f"  Predicted voxels: {prediction_binary.sum().item()}")

            # Calculate Dice score if target available
            target_out = batch['target_out'].cuda()
            intersection = (prediction_binary * target_out).sum()
            dice = (2.0 * intersection) / (prediction_binary.sum() + target_out.sum() + 1e-8)
            print(f"  Dice score: {dice.item():.4f}")


def demo_training_setup():
    """Demonstrate how to set up training loop."""
    print("\n" + "=" * 80)
    print("DEMO 3: Training Setup")
    print("=" * 80)

    # Define organs
    organ_list = ["liver", "kidney_left", "kidney_right"]

    # Create train and validation dataloaders
    train_loader = get_dataloader(
        root_dir="/path/to/TotalSegmentator/dataset",
        organ_list=organ_list,
        context_size=5,
        batch_size=1,
        image_size=(128, 128, 128),
        num_workers=4,
        mode='train',
        shuffle=True,
        random_context=True,  # Randomly sample contexts
    )

    val_loader = get_dataloader(
        root_dir="/path/to/TotalSegmentator/dataset",
        organ_list=organ_list,
        context_size=5,
        batch_size=1,
        image_size=(128, 128, 128),
        num_workers=4,
        mode='val',
        shuffle=False,
        random_context=False,  # Use fixed contexts for validation
    )

    print(f"Train dataset size: {len(train_loader.dataset)}")
    print(f"Val dataset size: {len(val_loader.dataset)}")

    # Create model
    hparams = Namespace(
        nb_inner_channels=[32, 64, 128, 256],
        nb_conv_layers_per_stage=2,
        data_slice_only=False,
    )
    model = LightningModel(hparams)

    print("\n✓ Training setup complete!")
    print("\nTo train with PyTorch Lightning:")
    print("""
    import pytorch_lightning as pl

    trainer = pl.Trainer(
        max_epochs=100,
        accelerator='gpu',
        devices=1,
        precision=16,  # Mixed precision training
    )

    trainer.fit(model, train_loader, val_loader)
    """)


def demo_custom_organs():
    """Demonstrate selecting specific organs."""
    print("\n" + "=" * 80)
    print("DEMO 4: Custom Organ Selection")
    print("=" * 80)

    # Example: Segment only cardiac structures
    cardiac_organs = [
        "heart",
        "atrial_appendage_left",
        "pulmonary_artery",
        "superior_vena_cava",
    ]

    # Example: Segment abdominal organs
    abdominal_organs = [
        "liver",
        "spleen",
        "pancreas",
        "kidney_left",
        "kidney_right",
        "adrenal_gland_left",
        "adrenal_gland_right",
        "stomach",
    ]

    # Example: Segment skeletal structures
    skeletal_organs = [
        "rib_left_1",
        "rib_left_2",
        "rib_right_1",
        "rib_right_2",
        "vertebrae_L1",
        "vertebrae_L2",
    ]

    organ_groups = {
        "Cardiac": cardiac_organs,
        "Abdominal": abdominal_organs,
        "Skeletal": skeletal_organs,
    }

    for group_name, organs in organ_groups.items():
        print(f"\n{group_name} organs ({len(organs)}):")
        for organ in organs:
            print(f"  - {organ}")


if __name__ == "__main__":
    # Run all demos
    try:
        demo_dataloader()
    except Exception as e:
        print(f"⚠ Demo 1 failed: {e}")
        print("  (This is expected if dataset path is not set)")

    try:
        demo_inference()
    except Exception as e:
        print(f"⚠ Demo 2 failed: {e}")
        print("  (This is expected if dataset/checkpoint path is not set)")

    try:
        demo_training_setup()
    except Exception as e:
        print(f"⚠ Demo 3 failed: {e}")

    demo_custom_organs()

    print("\n" + "=" * 80)
    print("All demos completed!")
    print("=" * 80)
