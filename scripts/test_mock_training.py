"""
Simplified test script for Neuroverse3D pipeline without requiring the full model.

This tests:
- Data loading
- Configuration
- Batch generation

Without requiring PyTorch Lightning or the actual Neuroverse3D model.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import numpy as np
from src.neuroverse3d.config import create_dataset_config, get_default_training_config
from src.neuroverse3d.dataloader import Neuroverse3DDataset

def test_dataset_loading(data_dir):
    """Test loading the dataset."""
    print("="*60)
    print("Testing Dataset Loading")
    print("="*60)

    # Create train dataset
    print("\n1. Creating training dataset...")
    train_dataset = Neuroverse3DDataset(
        data_dir=data_dir,
        mode='train',
        train_val_split=0.8,
        target_size=(64, 64, 64),
        normalize=True,
        augment=True,
    )
    print(f"   ✓ Train dataset created: {len(train_dataset)} samples")

    # Create val dataset
    print("\n2. Creating validation dataset...")
    val_dataset = Neuroverse3DDataset(
        data_dir=data_dir,
        mode='val',
        train_val_split=0.8,
        target_size=(64, 64, 64),
        normalize=True,
        augment=False,
    )
    print(f"   ✓ Val dataset created: {len(val_dataset)} samples")

    # Load a sample
    print("\n3. Loading a sample...")
    sample = train_dataset[0]
    print(f"   ✓ Sample loaded:")
    print(f"     - Image shape: {sample['image'].shape}")
    print(f"     - Label shape: {sample['label'].shape}")
    print(f"     - Case ID: {sample['case_id']}")
    print(f"     - Image dtype: {sample['image'].dtype}")
    print(f"     - Label dtype: {sample['label'].dtype}")

    # Check image statistics
    img_data = sample['image'].numpy()
    label_data = sample['label'].numpy()
    print(f"\n4. Image statistics:")
    print(f"   - Min: {img_data.min():.3f}")
    print(f"   - Max: {img_data.max():.3f}")
    print(f"   - Mean: {img_data.mean():.3f}")
    print(f"   - Std: {img_data.std():.3f}")
    print(f"\n5. Label statistics:")
    print(f"   - Unique values: {np.unique(label_data)}")
    print(f"   - Foreground ratio: {(label_data > 0).mean():.3f}")

    return train_dataset, val_dataset


def test_config():
    """Test configuration utilities."""
    print("\n" + "="*60)
    print("Testing Configuration")
    print("="*60)

    # Test training configs
    print("\n1. Stage 1 config:")
    config1 = get_default_training_config(stage=1)
    for key, value in config1.items():
        print(f"   - {key}: {value}")

    print("\n2. Stage 2 config:")
    config2 = get_default_training_config(stage=2)
    for key, value in config2.items():
        print(f"   - {key}: {value}")


def test_meta_dataset(data_dir):
    """Test meta-dataset for in-context learning."""
    print("\n" + "="*60)
    print("Testing Meta-Dataset")
    print("="*60)

    from src.neuroverse3d.dataloader import MetaDataset_Multi_Extended

    # Create dataset config
    config = create_dataset_config(
        data_dir=data_dir,
        datasets=None,  # Auto-detect
        sample_rate=1.0,
    )

    print(f"\n1. Created config for {len(config.datasets)} dataset(s)")

    # Create meta-dataset
    print("\n2. Creating meta-dataset (fixed context)...")
    meta_dataset = MetaDataset_Multi_Extended(
        mode='train',
        config=config,
        group_size=4,  # 3 context + 1 query
        random_context=False,
    )
    print(f"   ✓ Meta-dataset created: {len(meta_dataset)} samples")

    # Get a batch
    print("\n3. Loading a meta-batch...")
    batch = meta_dataset[0]
    print(f"   ✓ Batch loaded:")
    print(f"     - target_in shape: {batch['target_in'].shape}")
    print(f"     - target_out shape: {batch['target_out'].shape}")
    print(f"     - context_in shape: {batch['context_in'].shape}")
    print(f"     - context_out shape: {batch['context_out'].shape}")

    # Create meta-dataset with random context
    print("\n4. Creating meta-dataset (random context)...")
    meta_dataset_random = MetaDataset_Multi_Extended(
        mode='train',
        config=config,
        group_size=4,
        random_context=True,
        min_context=2,
        max_context=4,
    )
    print(f"   ✓ Random context meta-dataset created")

    return meta_dataset


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("Neuroverse3D Pipeline Test Suite")
    print("="*60)

    data_dir = "data/mock_neuroverse3d"

    # Check if data exists
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"\nERROR: Data directory not found: {data_dir}")
        print("Please run create_mock_data.py and prepare_neuroverse3d_data.py first")
        return

    try:
        # Test 1: Configuration
        test_config()

        # Test 2: Basic dataset loading
        train_ds, val_ds = test_dataset_loading(data_dir)

        # Test 3: Meta-dataset for in-context learning
        meta_ds = test_meta_dataset(data_dir)

        print("\n" + "="*60)
        print("✅ All tests passed!")
        print("="*60)
        print("\nThe pipeline is ready for training.")
        print("Next steps:")
        print("1. Install PyTorch and PyTorch Lightning:")
        print("   pip install torch pytorch-lightning")
        print("2. Clone Neuroverse3D repository:")
        print("   git clone https://github.com/jiesihu/Neuroverse3D")
        print("3. Run training:")
        print("   python scripts/train_neuroverse3d.py --data-dir data/mock_neuroverse3d")

    except Exception as e:
        print(f"\n❌ Test failed with error:")
        print(f"{type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
