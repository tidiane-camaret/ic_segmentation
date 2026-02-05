"""
MedSAM v1 Feature Extractor

Uses the original MedSAM (v1) image encoder as an alternative feature extractor.
MedSAM v1 uses a ViT-B backbone pre-trained on 1.5M medical image-mask pairs.

Features:
- Input: 1024x1024 images
- Output: [B, 256, 64, 64] features (64x64 spatial grid, 256 channels)
- Need to adapt to 14x14 grid for compatibility with MedDINO-based models

Installation:
    pip install git+https://github.com/bowang-lab/MedSAM.git

Usage:
    python Experiments/feature_extraction/medsam_extractor.py \
        --checkpoint /path/to/model.pt \
        --medsam-checkpoint /path/to/medsam_vit_b.pth \
        --context-size 3
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.dataloaders.totalseg2d_dataloader import get_dataloader
from src.losses import build_loss_fn
from src.models.patch_icl import PatchICL


class MedSAMv1Processor:
    """Preprocessor for MedSAM v1 inputs."""

    def __init__(self, target_size: int = 1024):
        """
        Args:
            target_size: Target resolution (1024 for MedSAM v1)
        """
        self.target_size = target_size
        # MedSAM v1 normalization
        self.pixel_mean = torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1)
        self.pixel_std = torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1)

    def __call__(self, images: torch.Tensor, device: torch.device = None) -> torch.Tensor:
        """Process images for MedSAM v1.

        Args:
            images: [B, 1, H, W] grayscale images, normalized to [0, 1]

        Returns:
            processed: [B, 3, 1024, 1024] normalized RGB tensor
        """
        if device is None:
            device = images.device

        if images.dim() == 3:
            images = images.unsqueeze(1)

        B = images.shape[0]
        processed = []

        for i in range(B):
            img = images[i, 0]

            # Percentile clipping
            lower = torch.quantile(img, 0.005)
            upper = torch.quantile(img, 0.995)
            img = torch.clamp(img, lower, upper)

            # Rescale to [0, 255]
            img_min, img_max = img.min(), img.max()
            if img_max > img_min:
                img = (img - img_min) / (img_max - img_min) * 255.0
            else:
                img = torch.zeros_like(img)

            processed.append(img)

        # Stack and expand to RGB
        processed = torch.stack(processed, dim=0).unsqueeze(1)
        processed = processed.expand(-1, 3, -1, -1).clone()

        # Resize to 1024x1024
        processed = F.interpolate(
            processed,
            size=(self.target_size, self.target_size),
            mode="bilinear",
            align_corners=False,
        )

        # Normalize
        mean = self.pixel_mean.to(device)
        std = self.pixel_std.to(device)
        processed = (processed - mean) / std

        return processed


class MedSAMv1FeatureExtractor(nn.Module):
    """
    Feature extractor using MedSAM v1 image encoder.

    Extracts features from MedSAM v1 ViT-B encoder and optionally adapts them
    to be compatible with MedDINO-style features (14x14 grid, 768 dim).
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        target_size: int = 1024,
        device: Union[str, torch.device] = "cuda",
        freeze: bool = True,
        adapt_features: bool = True,
        output_dim: int = 768,
    ):
        """
        Args:
            checkpoint_path: Path to MedSAM v1 checkpoint. If None, downloads.
            target_size: Input resolution (1024 for MedSAM v1)
            device: Device for computation
            freeze: Whether to freeze encoder weights
            adapt_features: If True, adapt 64x64x256 to 14x14x768 for MedDINO compatibility
            output_dim: Output feature dimension when adapting
        """
        super().__init__()
        self.target_size = target_size
        self.device = torch.device(device) if isinstance(device, str) else device
        self.adapt_features = adapt_features
        self.output_dim = output_dim

        self.processor = MedSAMv1Processor(target_size=target_size)

        self._load_model(checkpoint_path)

        if freeze:
            for param in self.image_encoder.parameters():
                param.requires_grad = False
            self.image_encoder.eval()

        self._frozen = freeze

        # Feature adaptation layer (64x64x256 -> 14x14x768)
        if adapt_features:
            self.adapter = nn.Sequential(
                # Pool spatial dimensions: 64x64 -> 14x14
                nn.AdaptiveAvgPool2d((14, 14)),
                # Project channels: 256 -> 768
                nn.Conv2d(256, output_dim, kernel_size=1),
            )
            self.adapter.to(self.device)

    def _load_model(self, checkpoint_path: Optional[str]):
        """Load MedSAM v1 image encoder."""
        try:
            from segment_anything import sam_model_registry
        except ImportError:
            raise ImportError(
                "MedSAM not installed. Run: pip install git+https://github.com/bowang-lab/MedSAM.git"
            )

        # Download checkpoint if needed
        if checkpoint_path is None:
            from huggingface_hub import hf_hub_download
            print("Downloading MedSAM v1 checkpoint from HuggingFace...")
            checkpoint_path = hf_hub_download(
                repo_id="wanglab/MedSAM",
                filename="medsam_vit_b.pth",
            )
            print(f"Downloaded to {checkpoint_path}")

        # Build SAM model
        print(f"Loading MedSAM v1 from {checkpoint_path}...")
        sam_model = sam_model_registry["vit_b"](checkpoint=checkpoint_path)
        self.image_encoder = sam_model.image_encoder
        self.image_encoder.to(self.device)
        print("MedSAM v1 image encoder loaded successfully")

        # Feature dimensions: ViT-B outputs [B, 256, 64, 64]
        self.feature_dim = 256
        self.feature_grid_size = 64

    def train(self, mode: bool = True):
        """Keep encoder in eval mode if frozen."""
        super().train(mode)
        if self._frozen:
            self.image_encoder.eval()
        return self

    @torch.no_grad()
    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract features from images.

        Args:
            images: [B, 1, H, W] grayscale images, normalized to [0, 1]

        Returns:
            features: [B, N, D] where N = 196 (14x14) or 4096 (64x64), D = 768 or 256
        """
        was_training = self.image_encoder.training
        self.image_encoder.eval()

        # Preprocess
        pixel_values = self.processor(images, device=self.device)
        pixel_values = pixel_values.to(self.device)

        # Extract features: [B, 256, 64, 64]
        features = self.image_encoder(pixel_values)

        if self.adapt_features:
            # Adapt to MedDINO-compatible format
            # [B, 256, 64, 64] -> [B, 768, 14, 14] -> [B, 196, 768]
            features = self.adapter(features)
            B, C, H, W = features.shape
            features = features.permute(0, 2, 3, 1).reshape(B, H * W, C)
        else:
            # Keep original format: [B, 256, 64, 64] -> [B, 4096, 256]
            B, C, H, W = features.shape
            features = features.permute(0, 2, 3, 1).reshape(B, H * W, C)

        if was_training and not self._frozen:
            self.image_encoder.train()

        return features

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.extract_features(images)

    def extract_batch(
        self,
        target_images: torch.Tensor,
        context_images: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Extract features for target and context images.

        Args:
            target_images: [B, 1, H, W]
            context_images: [B, k, 1, H, W] (optional)

        Returns:
            target_features: [B, N, D]
            context_features: [B, k, N, D] or None
        """
        B = target_images.shape[0]

        if context_images is not None:
            k = context_images.shape[1]
            ctx_flat = context_images.view(B * k, *context_images.shape[2:])
            all_images = torch.cat([target_images, ctx_flat], dim=0)
            all_features = self.extract_features(all_images)
            target_features = all_features[:B]
            context_features = all_features[B:].view(B, k, *all_features.shape[1:])
            return target_features, context_features
        else:
            target_features = self.extract_features(target_images)
            return target_features, None


def compute_dice(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute Dice score."""
    pred_binary = (torch.sigmoid(pred) > 0.5).float()
    target_binary = (target > 0).float()
    spatial_dims = tuple(range(2, pred_binary.dim()))
    intersection = (pred_binary * target_binary).sum(dim=spatial_dims)
    union = pred_binary.sum(dim=spatial_dims) + target_binary.sum(dim=spatial_dims)
    return (2 * intersection + 1e-6) / (union + 1e-6)


def evaluate_medsam_features(
    model: PatchICL,
    feature_extractor: MedSAMv1FeatureExtractor,
    dataloader,
    device: str,
) -> Dict[str, float]:
    """Evaluate model using MedSAM features."""
    model.eval()

    total_dice = 0.0
    total_local_dice = 0.0
    total_samples = 0
    per_label_dice = {}

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating MedSAM features"):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            context_in = batch.get("context_in")
            context_out = batch.get("context_out")
            if context_in is not None:
                context_in = context_in.to(device)
            if context_out is not None:
                context_out = context_out.to(device)

            if labels.dim() == 3:
                labels = labels.unsqueeze(1)

            # Extract MedSAM features on-the-fly
            target_features, context_features = feature_extractor.extract_batch(
                images, context_in
            )

            outputs = model(
                images,
                labels=labels,
                context_in=context_in,
                context_out=context_out,
                target_features=target_features,
                context_features=context_features,
                mode="test",
            )

            # Metrics
            final_dice = compute_dice(outputs["final_pred"], labels)
            total_dice += final_dice.sum().item()

            patch_logits = outputs["patch_logits"]
            patch_labels = outputs["patch_labels"]
            patch_dice = compute_dice(
                patch_logits.flatten(0, 1),
                patch_labels.flatten(0, 1)
            )
            total_local_dice += patch_dice.sum().item()

            # Per-label
            batch_labels = batch.get("label_ids", [])
            for i, label_id in enumerate(batch_labels):
                if label_id not in per_label_dice:
                    per_label_dice[label_id] = []
                per_label_dice[label_id].append(final_dice[i].item())

            total_samples += images.shape[0]

    avg_dice = total_dice / total_samples
    avg_local = total_local_dice / (total_samples * patch_logits.shape[1])
    per_label_avg = {
        label: sum(scores) / len(scores)
        for label, scores in per_label_dice.items()
    }

    return {
        "extractor": "medsam_v1",
        "final_dice": avg_dice,
        "local_dice": avg_local,
        "per_label_dice": per_label_avg,
        "num_samples": total_samples,
    }


def run_medsam_comparison(
    checkpoint_path: str,
    medsam_checkpoint: Optional[str] = None,
    context_size: int = 3,
    root_dir: str = "/data/TotalSeg2D",
    stats_path: str = "/data/TotalSeg2D/totalseg_stats.pkl",
    labels: List[str] = ["liver", "spleen", "kidney_left", "kidney_right", "aorta"],
    max_samples: Optional[int] = None,
    batch_size: int = 8,  # Smaller due to 1024x1024 inputs
    device: str = "cuda",
    output_path: Optional[str] = None,
) -> Dict:
    """Run MedSAM feature comparison experiment."""
    print("Running MedSAM v1 feature extraction experiment")
    print(f"  Context size: {context_size}")
    print(f"  Labels: {labels}")

    # Load PatchICL model
    print(f"\nLoading model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint.get("config", {})
    patch_icl_cfg = config.get("model", {}).get("patch_icl", {})

    model = PatchICL(patch_icl_cfg, context_size=context_size)

    # Filter out feature_extractor keys (may be saved if model was trained with on-the-fly extraction)
    state_dict = checkpoint["model_state_dict"]
    filtered_state_dict = {
        k: v for k, v in state_dict.items()
        if not k.startswith("feature_extractor.")
    }
    model.load_state_dict(filtered_state_dict, strict=False)
    model.to(device)

    patch_criterion = build_loss_fn("dice", None)
    aggreg_criterion = build_loss_fn("dice", None)
    model.set_loss_functions(patch_criterion, aggreg_criterion)

    # Create MedSAM feature extractor
    print("\nInitializing MedSAM v1 feature extractor...")
    feature_extractor = MedSAMv1FeatureExtractor(
        checkpoint_path=medsam_checkpoint,
        device=device,
        freeze=True,
        adapt_features=True,  # Adapt to MedDINO format
        output_dim=768,
    )

    # Create dataloader (without precomputed features)
    dataloader = get_dataloader(
        root_dir=root_dir,
        stats_path=stats_path,
        label_id_list=labels,
        context_size=context_size,
        batch_size=batch_size,
        image_size=(256, 256),  # Will be resized to 1024x1024 in extractor
        num_workers=4,
        split="val",
        shuffle=False,
        load_dinov3_features=False,  # Don't load precomputed features
        max_ds_len=max_samples,
    )

    # Evaluate
    print("\nEvaluating MedSAM features...")
    results = evaluate_medsam_features(model, feature_extractor, dataloader, device)

    print(f"\nResults:")
    print(f"  Final Dice: {results['final_dice']:.4f}")
    print(f"  Local Dice: {results['local_dice']:.4f}")
    print(f"  Per-label:")
    for label, dice in sorted(results['per_label_dice'].items()):
        print(f"    {label}: {dice:.4f}")

    # Save results
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="MedSAM feature extraction experiment")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to PatchICL model checkpoint")
    parser.add_argument("--medsam-checkpoint", type=str, default=None,
                       help="Path to MedSAM v1 checkpoint (downloads if not provided)")
    parser.add_argument("--context-size", type=int, default=3)
    parser.add_argument("--root-dir", type=str, default="/data/TotalSeg2D")
    parser.add_argument("--stats-path", type=str, default="/data/TotalSeg2D/totalseg_stats.pkl")
    parser.add_argument("--labels", type=str, nargs="+",
                       default=["liver", "spleen", "kidney_left", "kidney_right", "aorta"])
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default="./results/medsam_features.json")

    args = parser.parse_args()

    run_medsam_comparison(
        checkpoint_path=args.checkpoint,
        medsam_checkpoint=args.medsam_checkpoint,
        context_size=args.context_size,
        root_dir=args.root_dir,
        stats_path=args.stats_path,
        labels=args.labels,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        device=args.device,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
