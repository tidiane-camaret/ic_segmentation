"""Quick test for SynthMorph dataloader."""

import sys
sys.path.insert(0, "/software/notebooks/camaret/ic_segmentation")

import numpy as np
import torch


def test_synthmorph_utils():
    """Test core generation functions."""
    from src.dataloaders.synthmorph_utils import (
        generate_base_label_map,
        random_deformation_field,
        apply_deformation,
        synthesize_image,
        generate_subject,
    )

    shape = (128, 128)
    num_labels = 16
    rng = np.random.default_rng(42)

    # Test base label map
    label_map = generate_base_label_map(shape, num_labels, rng=rng)
    assert label_map.shape == shape
    assert label_map.dtype == np.int16
    assert label_map.min() >= 0
    assert label_map.max() < num_labels
    print(f"Base label map: shape={label_map.shape}, unique={len(np.unique(label_map))}")

    # Test deformation field
    flow = random_deformation_field(shape, rng=rng)
    assert flow.shape == (2, *shape)
    print(f"Deformation field: shape={flow.shape}, range=[{flow.min():.2f}, {flow.max():.2f}]")

    # Test apply deformation
    warped = apply_deformation(label_map, flow, order=0)
    assert warped.shape == shape
    assert warped.dtype == label_map.dtype
    print(f"Warped label: shape={warped.shape}, unique={len(np.unique(warped))}")

    # Test image synthesis
    image = synthesize_image(label_map, num_labels, rng=rng)
    assert image.shape == shape
    assert image.dtype == np.float32
    assert 0 <= image.min() <= image.max() <= 1
    print(f"Synthesized image: shape={image.shape}, range=[{image.min():.2f}, {image.max():.2f}]")

    # Test generate_subject
    img, lbl = generate_subject(label_map, num_labels, rng=rng)
    assert img.shape == shape
    assert lbl.shape == shape
    print(f"Subject: img shape={img.shape}, label unique={len(np.unique(lbl))}")

    print("\n[PASS] synthmorph_utils tests passed!")


def test_synthmorph_dataset():
    """Test SynthMorphDataset."""
    from src.dataloaders.synthmorph_dataloader import SynthMorphDataset

    dataset = SynthMorphDataset(
        num_tasks=10,
        num_labels=8,
        context_size=3,
        image_size=(128, 128),
        epoch_length=100,
        max_cache_size=10,
    )

    assert len(dataset) == 100

    # Get a sample
    sample = dataset[0]

    # Check keys
    assert "image" in sample
    assert "label" in sample
    assert "context_in" in sample
    assert "context_out" in sample
    assert "target_case_id" in sample
    assert "context_case_ids" in sample
    assert "label_id" in sample

    # Check shapes
    assert sample["image"].shape == (1, 128, 128)
    assert sample["label"].shape == (1, 128, 128)
    assert sample["context_in"].shape == (3, 1, 128, 128)
    assert sample["context_out"].shape == (3, 1, 128, 128)

    # Check types
    assert sample["image"].dtype == torch.float32
    assert sample["label"].dtype == torch.float32

    # Check case IDs
    assert "synth_t" in sample["target_case_id"]
    assert len(sample["context_case_ids"]) == 3

    print(f"\nSample keys: {list(sample.keys())}")
    print(f"Image shape: {sample['image'].shape}")
    print(f"Label shape: {sample['label'].shape}")
    print(f"Context in shape: {sample['context_in'].shape}")
    print(f"Target case ID: {sample['target_case_id']}")
    print(f"Label ID: {sample['label_id']}")

    print("\n[PASS] SynthMorphDataset tests passed!")


def test_dataloader():
    """Test DataLoader creation and batching."""
    from src.dataloaders.synthmorph_dataloader import get_synthmorph_dataloader

    loader = get_synthmorph_dataloader(
        num_tasks=10,
        num_labels=8,
        context_size=3,
        batch_size=4,
        image_size=(128, 128),
        epoch_length=20,
        num_workers=0,
    )

    batch = next(iter(loader))

    assert batch["image"].shape == (4, 1, 128, 128)
    assert batch["label"].shape == (4, 1, 128, 128)
    assert batch["context_in"].shape == (4, 3, 1, 128, 128)
    assert batch["context_out"].shape == (4, 3, 1, 128, 128)
    assert len(batch["target_case_ids"]) == 4
    assert len(batch["label_ids"]) == 4

    print(f"\nBatch image shape: {batch['image'].shape}")
    print(f"Batch context_in shape: {batch['context_in'].shape}")

    print("\n[PASS] DataLoader tests passed!")


def test_timing():
    """Measure generation speed."""
    import time
    from src.dataloaders.synthmorph_dataloader import SynthMorphDataset

    dataset = SynthMorphDataset(
        num_tasks=100,
        num_labels=16,
        context_size=3,
        image_size=(256, 256),
        epoch_length=50,
        max_cache_size=100,
    )

    # Warmup (fills cache)
    for i in range(10):
        _ = dataset[i]

    # Timed run
    start = time.time()
    n_samples = 20
    for i in range(n_samples):
        _ = dataset[i]
    elapsed = time.time() - start

    print(f"\nTiming (256x256, k=3): {elapsed/n_samples*1000:.1f} ms/sample")
    print(f"Throughput: {n_samples/elapsed:.1f} samples/sec")


if __name__ == "__main__":
    test_synthmorph_utils()
    test_synthmorph_dataset()
    test_dataloader()
    test_timing()
    print("\n" + "="*50)
    print("All tests passed!")
