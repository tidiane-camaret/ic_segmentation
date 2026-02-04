#!/usr/bin/env python
"""Test script for foreground-based context patch sampling."""

import torch
import numpy as np


def compute_foreground_patch_centers(context_out, roi_size, overlap, min_foreground_ratio=0.01):
    """Test version of _compute_foreground_patch_centers."""
    C, D, H, W = context_out.shape
    D_roi, H_roi, W_roi = roi_size

    # Calculate stride from overlap
    stride_d = max(1, int(D_roi * (1 - overlap)))
    stride_h = max(1, int(H_roi * (1 - overlap)))
    stride_w = max(1, int(W_roi * (1 - overlap)))

    foreground_centers = []
    threshold = min_foreground_ratio * (D_roi * H_roi * W_roi)

    # Iterate over all possible patch centers
    for d in range(0, D - D_roi + 1, stride_d):
        for h in range(0, H - H_roi + 1, stride_h):
            for w in range(0, W - W_roi + 1, stride_w):
                # Extract patch
                patch = context_out[:, d:d+D_roi, h:h+H_roi, w:w+W_roi]

                # Check if patch has enough foreground
                if patch.sum() >= threshold:
                    # Store patch top-left corner
                    foreground_centers.append((d, h, w))

    # If no foreground patches found, fall back to all possible centers
    if len(foreground_centers) == 0:
        for d in range(0, D - D_roi + 1, stride_d):
            for h in range(0, H - H_roi + 1, stride_h):
                for w in range(0, W - W_roi + 1, stride_w):
                    foreground_centers.append((d, h, w))

    return foreground_centers


def test_foreground_detection():
    """Test that foreground patches are correctly identified."""
    print("=" * 60)
    print("Test 1: Foreground patch detection")
    print("=" * 60)

    # Create a simple binary mask with a small object
    context_out = torch.zeros(1, 64, 64, 64)
    # Add a 20x20x20 object in the center
    context_out[:, 22:42, 22:42, 22:42] = 1.0

    roi_size = (32, 32, 32)
    overlap = 0.25

    centers = compute_foreground_patch_centers(context_out, roi_size, overlap)

    print(f"Image size: {context_out.shape}")
    print(f"ROI size: {roi_size}")
    print(f"Foreground voxels: {context_out.sum().item()}")
    print(f"Found {len(centers)} patches with foreground")
    print(f"Sample centers: {centers[:5]}")

    # Verify that at least some patches contain the object
    assert len(centers) > 0, "Should find at least one foreground patch"
    print("✓ Test passed: Foreground patches detected\n")


def test_empty_mask_fallback():
    """Test fallback behavior when mask is empty."""
    print("=" * 60)
    print("Test 2: Empty mask fallback")
    print("=" * 60)

    # Create empty mask
    context_out = torch.zeros(1, 64, 64, 64)

    roi_size = (32, 32, 32)
    overlap = 0.25

    centers = compute_foreground_patch_centers(context_out, roi_size, overlap)

    print(f"Empty mask - Found {len(centers)} patches (fallback to all)")
    assert len(centers) > 0, "Should fall back to all patches when mask is empty"
    print("✓ Test passed: Fallback works correctly\n")


def test_patch_sampling():
    """Test that sampled patches contain the context."""
    print("=" * 60)
    print("Test 3: Context patch sampling")
    print("=" * 60)

    # Create context with multiple examples
    L, C, D, H, W = 3, 1, 64, 64, 64
    roi_size = (32, 32, 32)

    context_in = torch.randn(L, C, D, H, W)
    context_out = torch.zeros(L, C, D, H, W)

    # Add foreground in different locations for each context
    context_out[0, :, 10:30, 10:30, 10:30] = 1.0  # Top-left
    context_out[1, :, 30:50, 30:50, 30:50] = 1.0  # Center
    context_out[2, :, 40:60, 40:60, 40:60] = 1.0  # Bottom-right

    # Compute foreground centers for each
    foreground_centers = []
    for l in range(L):
        centers = compute_foreground_patch_centers(
            context_out[l],
            roi_size,
            overlap=0.25
        )
        foreground_centers.append(centers)

    print(f"Context 0: {len(foreground_centers[0])} foreground patches")
    print(f"Context 1: {len(foreground_centers[1])} foreground patches")
    print(f"Context 2: {len(foreground_centers[2])} foreground patches")

    # Sample a patch from each
    import random
    for l in range(L):
        center = random.choice(foreground_centers[l])
        d, h, w = center
        D_roi, H_roi, W_roi = roi_size

        patch_out = context_out[l:l+1, :, d:d+D_roi, h:h+H_roi, w:w+W_roi]
        foreground_in_patch = patch_out.sum().item()

        print(f"  Context {l} sampled patch at {center}: {foreground_in_patch:.0f} foreground voxels")
        assert foreground_in_patch > 0, f"Sampled patch should contain foreground for context {l}"

    print("✓ Test passed: Patches sampled from foreground locations\n")


def test_multiple_sampling():
    """Test that we can sample different patches each time."""
    print("=" * 60)
    print("Test 4: Multiple random samples")
    print("=" * 60)

    context_out = torch.zeros(1, 128, 128, 128)
    # Add large foreground region
    context_out[:, 20:100, 20:100, 20:100] = 1.0

    roi_size = (32, 32, 32)
    centers = compute_foreground_patch_centers(context_out, roi_size, overlap=0.25)

    print(f"Available foreground patches: {len(centers)}")

    # Sample multiple times and check diversity
    import random
    samples = [random.choice(centers) for _ in range(10)]
    unique_samples = len(set(samples))

    print(f"Sampled 10 patches, {unique_samples} unique")
    print(f"Sample centers: {samples[:5]}")

    # With many available patches, we should get some diversity
    if len(centers) >= 10:
        assert unique_samples > 1, "Should get some diversity when many patches available"

    print("✓ Test passed: Random sampling works\n")


if __name__ == "__main__":
    print("\nTesting foreground-based context patch sampling\n")

    test_foreground_detection()
    test_empty_mask_fallback()
    test_patch_sampling()
    test_multiple_sampling()

    print("=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
