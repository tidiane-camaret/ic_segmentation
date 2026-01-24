#!/usr/bin/env python3
"""
Extract DINOv3 features for TotalSeg2D images.

Reads all *_img.npy (or *_img.nii.gz) files from TotalSeg2D directory,
computes DINOv3 patch features, and saves them to *_img_dinov3.npz.

Usage:
    python scripts/extract_dinov3_features.py paths=dlclarge
    python scripts/extract_dinov3_features.py paths=dlclarge batch_size=64 skip_existing=true
"""

from pathlib import Path
from queue import Queue
import threading

import hydra
from omegaconf import DictConfig
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel


def find_all_images(data_dir: Path) -> list[Path]:
    """Find all image files in TotalSeg2D directory. Prefers .npy over .nii.gz."""
    # First try .npy (fast format)
    npy_files = sorted(data_dir.glob("**/*_slice_img.npy"))
    if npy_files:
        return npy_files

    # Fall back to .nii.gz (legacy format)
    return sorted(data_dir.glob("**/*_img.nii.gz"))


def load_image(img_path: Path) -> np.ndarray:
    """Load an image and return as numpy array."""
    if img_path.suffix == ".npy":
        return np.load(img_path).astype(np.float32)
    else:
        # Legacy .nii.gz format
        import nibabel as nib
        img = nib.load(img_path)
        return img.get_fdata().astype(np.float32)


def preprocess_batch(images: list[np.ndarray], processor) -> torch.Tensor:
    """
    Preprocess a batch of grayscale images for DINOv3.

    Converts grayscale to RGB (repeat 3 channels) and applies processor.
    """
    rgb_images = []
    for img in images:
        # Normalize to [0, 255] range
        img_min, img_max = img.min(), img.max()
        if img_max > img_min:
            img_norm = (img - img_min) / (img_max - img_min) * 255
        else:
            img_norm = np.zeros_like(img)
        img_norm = img_norm.astype(np.uint8)

        # Convert grayscale to RGB by repeating channels
        img_rgb = np.stack([img_norm, img_norm, img_norm], axis=-1)
        rgb_images.append(img_rgb)

    # Use processor to resize to 224x224 and normalize
    inputs = processor(images=rgb_images, return_tensors="pt")
    return inputs["pixel_values"]


class FeatureExtractor:
    """Extract DINOv3 features using batched inference."""

    def __init__(self, model_path: str, device: str = "cuda"):
        self.device = device
        print(f"Loading DINOv3 model from {model_path}...")
        self.processor = AutoImageProcessor.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path).to(device)
        self.model.eval()
        print("Model loaded.")

    @torch.no_grad()
    def extract_features(self, pixel_values: torch.Tensor) -> np.ndarray:
        """
        Extract patch features from preprocessed images.

        Args:
            pixel_values: [B, 3, 224, 224] preprocessed images

        Returns:
            features: [B, 196, 1024] patch features (14x14 grid)
        """
        pixel_values = pixel_values.to(self.device)
        outputs = self.model(pixel_values)

        # Get patch tokens (skip CLS + 4 register tokens)
        # last_hidden_state: [B, 201, 1024] -> [B, 196, 1024]
        patch_features = outputs.last_hidden_state[:, 5:, :]

        return patch_features.cpu().numpy()


def process_batch(
    batch_paths: list[Path],
    extractor: FeatureExtractor,
    pbar: tqdm,
) -> list[tuple[Path, np.ndarray]]:
    """Process a batch of images and return features."""
    images = []
    valid_paths = []
    for path in batch_paths:
        try:
            img = load_image(path)
            images.append(img)
            valid_paths.append(path)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            pbar.update(1)

    if not images:
        return []

    # Preprocess and extract features
    pixel_values = preprocess_batch(images, extractor.processor)
    features = extractor.extract_features(pixel_values)

    results = list(zip(valid_paths, features))
    pbar.update(len(results))
    return results


def get_output_path(img_path: Path) -> Path:
    """Get output path for features file."""
    if img_path.suffix == ".npy":
        # *_slice_img.npy -> *_slice_img_dinov3.npz
        return img_path.parent / (img_path.stem + "_dinov3.npz")
    else:
        # *_img.nii.gz -> *_img_dinov3.npz
        return img_path.parent / (img_path.name.replace("_img.nii.gz", "_img_dinov3.npz"))


def save_features(path: Path, features: np.ndarray):
    """Save features to .npz file."""
    output_path = get_output_path(path)
    np.savez_compressed(output_path, features=features)


def save_worker(save_queue: Queue, stop_event: threading.Event):
    """Worker thread to save features asynchronously."""
    while not stop_event.is_set() or not save_queue.empty():
        try:
            item = save_queue.get(timeout=0.1)
            path, features = item
            save_features(path, features)
            save_queue.task_done()
        except Exception:
            continue


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    """Extract DINOv3 features for TotalSeg2D images."""
    # Get config values with defaults
    batch_size = cfg.get("batch_size", 64)
    num_workers = cfg.get("num_save_workers", 20)
    skip_existing = cfg.get("skip_existing", True)

    data_dir = Path(cfg.paths.totalseg2d)
    model_path = cfg.paths.ckpts.dino_vit

    print(f"Data directory: {data_dir}")
    print(f"Model path: {model_path}")
    print(f"Batch size: {batch_size}")
    print(f"Num save workers: {num_workers}")
    print(f"Skip existing: {skip_existing}")

    # Find all images
    print("Finding all images...")
    all_images = find_all_images(data_dir)
    print(f"Found {len(all_images)} images")

    if not all_images:
        print("No images found.")
        return

    # Show format being used
    sample_ext = all_images[0].suffix
    print(f"Image format: {sample_ext}")

    # Filter out images that already have features
    if skip_existing:
        images_to_process = []
        for img_path in all_images:
            output_path = get_output_path(img_path)
            if not output_path.exists():
                images_to_process.append(img_path)
        print(f"Skipping {len(all_images) - len(images_to_process)} existing, processing {len(images_to_process)}")
    else:
        images_to_process = all_images

    if not images_to_process:
        print("No images to process.")
        return

    # Initialize extractor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    extractor = FeatureExtractor(model_path, device)

    # Create save queue and workers
    save_queue = Queue(maxsize=batch_size * 4)
    stop_event = threading.Event()

    save_threads = []
    for _ in range(num_workers):
        t = threading.Thread(target=save_worker, args=(save_queue, stop_event))
        t.start()
        save_threads.append(t)

    # Process in batches
    pbar = tqdm(total=len(images_to_process), desc="Extracting features")

    try:
        for i in range(0, len(images_to_process), batch_size):
            batch_paths = images_to_process[i:i + batch_size]
            results = process_batch(batch_paths, extractor, pbar)

            # Queue results for saving
            for path, features in results:
                save_queue.put((path, features))

    finally:
        # Wait for save queue to empty
        pbar.close()
        print("Waiting for save workers to finish...")
        save_queue.join()
        stop_event.set()
        for t in save_threads:
            t.join()

    print("Done!")


if __name__ == "__main__":
    main()
