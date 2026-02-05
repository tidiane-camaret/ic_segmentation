"""
Simple Segmentation Evaluation for MedSAM Features

Evaluates feature quality by training a simple segmentation head on top of
frozen features from different MedSAM layers.

Features are extracted on-the-fly using the MedSAM model.

Architecture:
    Image -> MedSAM (frozen) -> Features -> Simple Head -> Mask [B, 1, H, W]

MedSAM outputs 256-dim features at 64x64 spatial resolution from its neck,
or we can extract 768-dim features from intermediate transformer blocks.

Usage:
    python Experiments/feature_extraction/simple_seg_eval_medsam.py \
        --layers 2 5 8 11 \
        --head-type linear \
        --epochs 10
"""
import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.dataloaders.totalseg2d_dataloader import TotalSeg2DDataset, collate_fn


# Default paths
DEFAULT_MEDSAM_PATH = "/nfs/data/nii/data1/Analysis/camaret___in_context_segmentation/ANALYSIS_20251122/checkpoints/medsam_vit_b.pth"
DEFAULT_ROOT_DIR = "/nfs/data/nii/data1/Analysis/camaret___in_context_segmentation/ANALYSIS_20251122/data/TotalSeg2D"
DEFAULT_STATS_PATH = "/nfs/data/nii/data1/Analysis/camaret___in_context_segmentation/ANALYSIS_20251122/data/TotalSeg/totalseg_stats.pkl"


class MedSAMFeatureExtractor(nn.Module):
    """Extract features from MedSAM image encoder at different layers."""

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda",
        layer_idx: int = 11,
        freeze: bool = True,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.layer_idx = layer_idx

        # Load MedSAM
        from segment_anything import sam_model_registry
        print(f"Loading MedSAM from {checkpoint_path}...")
        self.sam = sam_model_registry["vit_b"](checkpoint=checkpoint_path)
        self.image_encoder = self.sam.image_encoder
        self.image_encoder.to(self.device)

        if freeze:
            for param in self.image_encoder.parameters():
                param.requires_grad = False
            self.image_encoder.eval()

        self._frozen = freeze

        # SAM uses these normalization values
        self.pixel_mean = torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1)
        self.pixel_std = torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1)

        print(f"MedSAM loaded (12 blocks, extracting from block {layer_idx})")

    def preprocess(self, images: torch.Tensor) -> torch.Tensor:
        """Preprocess images for MedSAM (expects 1024x1024 input)."""
        # images: [B, 1, H, W] grayscale normalized to [0, 1]
        B = images.shape[0]

        processed = []
        for i in range(B):
            img = images[i, 0]  # [H, W]

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

        processed = torch.stack(processed, dim=0).unsqueeze(1)  # [B, 1, H, W]
        processed = processed.expand(-1, 3, -1, -1).clone()  # [B, 3, H, W]

        # Resize to 1024x1024
        processed = F.interpolate(processed, size=(1024, 1024), mode="bilinear", align_corners=False)

        # SAM normalization
        mean = self.pixel_mean.to(processed.device)
        std = self.pixel_std.to(processed.device)
        processed = (processed - mean) / std

        return processed

    @torch.no_grad()
    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract features from a specific transformer block.

        Args:
            images: [B, 1, H, W] grayscale images

        Returns:
            features: [B, N, D] where N = 64*64 = 4096, D = 768
        """
        self.image_encoder.eval()

        # Preprocess
        x = self.preprocess(images)
        x = x.to(self.device)

        # Patch embedding
        x = self.image_encoder.patch_embed(x)  # [B, 64, 64, 768]

        # Add positional embedding
        if self.image_encoder.pos_embed is not None:
            x = x + self.image_encoder.pos_embed

        # Pass through transformer blocks up to layer_idx
        for i, blk in enumerate(self.image_encoder.blocks):
            x = blk(x)
            if i == self.layer_idx:
                break

        # x is [B, 64, 64, 768], reshape to [B, N, D]
        B, H, W, D = x.shape
        features = x.view(B, H * W, D)

        return features

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.extract_features(images)


class SimpleSegHead(nn.Module):
    """Simple segmentation head for feature evaluation."""

    def __init__(
        self,
        in_dim: int = 768,
        hidden_dim: int = 256,
        out_size: int = 256,
        num_patches: int = 4096,
        head_type: str = "linear",
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_size = out_size
        self.num_patches = num_patches
        self.head_type = head_type

        # Compute grid size
        self.feature_grid = int(math.ceil(math.sqrt(num_patches)))
        self.padded_size = self.feature_grid ** 2
        self.pad_amount = self.padded_size - num_patches

        if head_type == "linear":
            self.head = nn.Linear(in_dim, 1)
        elif head_type == "conv":
            self.head = nn.Conv2d(in_dim, 1, kernel_size=1)
        elif head_type == "mlp":
            self.head = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )
        else:
            raise ValueError(f"Unknown head type: {head_type}")

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        B, N, D = features.shape

        # Pad to square if needed
        if self.pad_amount > 0:
            padding = torch.zeros(B, self.pad_amount, D, device=features.device, dtype=features.dtype)
            features = torch.cat([features, padding], dim=1)

        if self.head_type == "conv":
            x = features.permute(0, 2, 1).view(B, D, self.feature_grid, self.feature_grid)
            x = self.head(x)
        else:
            x = self.head(features)
            x = x.permute(0, 2, 1).view(B, 1, self.feature_grid, self.feature_grid)

        x = F.interpolate(x, size=(self.out_size, self.out_size), mode="bilinear", align_corners=False)
        return x


def compute_dice(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_binary = (torch.sigmoid(pred) > 0.5).float()
    target_binary = (target > 0).float()
    intersection = (pred_binary * target_binary).sum(dim=(1, 2, 3))
    union = pred_binary.sum(dim=(1, 2, 3)) + target_binary.sum(dim=(1, 2, 3))
    return (2 * intersection + 1e-6) / (union + 1e-6)


def dice_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_prob = torch.sigmoid(pred)
    target_binary = (target > 0).float()
    intersection = (pred_prob * target_binary).sum(dim=(1, 2, 3))
    union = pred_prob.sum(dim=(1, 2, 3)) + target_binary.sum(dim=(1, 2, 3))
    dice = (2 * intersection + 1e-6) / (union + 1e-6)
    return 1 - dice.mean()


def train_and_eval_layer(
    layer_idx: int,
    feature_extractor: MedSAMFeatureExtractor,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str,
    head_type: str = "linear",
    epochs: int = 10,
    lr: float = 1e-3,
) -> Dict:
    """Train and evaluate a simple segmentation head on features from one layer."""

    # Update feature extractor layer
    feature_extractor.layer_idx = layer_idx

    # Determine feature shape from first batch
    with torch.no_grad():
        sample_batch = next(iter(train_loader))
        sample_img = sample_batch["image"][:1].to(device)
        sample_features = feature_extractor.extract_features(sample_img)
        num_patches = sample_features.shape[1]
        feature_dim = sample_features.shape[2]
        feature_grid = int(math.sqrt(num_patches))
        print(f"  Feature shape: {sample_features.shape}, grid: {feature_grid}x{feature_grid}")

    # Create segmentation head
    head = SimpleSegHead(
        in_dim=feature_dim,
        hidden_dim=256,
        out_size=256,
        num_patches=num_patches,
        head_type=head_type,
    ).to(device)

    optimizer = torch.optim.Adam(head.parameters(), lr=lr)

    best_dice = 0.0
    train_losses = []

    for epoch in range(epochs):
        head.train()
        epoch_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Layer {layer_idx} - Epoch {epoch+1}/{epochs}", leave=False):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            if labels.dim() == 3:
                labels = labels.unsqueeze(1)

            with torch.no_grad():
                features = feature_extractor.extract_features(images)

            optimizer.zero_grad()
            logits = head(features)
            loss = dice_loss(logits, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)

        # Validation
        head.eval()
        val_dice = 0.0
        val_samples = 0

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                labels = batch["label"].to(device)

                if labels.dim() == 3:
                    labels = labels.unsqueeze(1)

                features = feature_extractor.extract_features(images)
                logits = head(features)
                dice = compute_dice(logits, labels)
                val_dice += dice.sum().item()
                val_samples += labels.shape[0]

        avg_dice = val_dice / val_samples

        if avg_dice > best_dice:
            best_dice = avg_dice

        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            print(f"  Epoch {epoch+1}: Loss={avg_loss:.4f}, Val Dice={avg_dice:.4f}")

    # Final evaluation
    head.eval()
    total_dice = 0.0
    total_samples = 0
    per_label_dice = {}

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            if labels.dim() == 3:
                labels = labels.unsqueeze(1)

            features = feature_extractor.extract_features(images)
            logits = head(features)
            dice = compute_dice(logits, labels)

            batch_labels = batch.get("label_ids", [])
            for i, label_id in enumerate(batch_labels):
                if label_id not in per_label_dice:
                    per_label_dice[label_id] = []
                per_label_dice[label_id].append(dice[i].item())

            total_dice += dice.sum().item()
            total_samples += labels.shape[0]

    final_dice = total_dice / total_samples
    per_label_avg = {
        label: sum(scores) / len(scores)
        for label, scores in per_label_dice.items()
    }

    return {
        "layer": layer_idx,
        "final_dice": final_dice,
        "best_dice": best_dice,
        "per_label_dice": per_label_avg,
        "train_losses": train_losses,
    }


def run_medsam_eval(
    layers: List[int] = [2, 5, 8, 11],
    head_type: str = "linear",
    epochs: int = 10,
    lr: float = 1e-3,
    root_dir: str = DEFAULT_ROOT_DIR,
    stats_path: str = DEFAULT_STATS_PATH,
    medsam_path: str = DEFAULT_MEDSAM_PATH,
    labels: List[str] = ["liver", "spleen", "kidney_left", "kidney_right", "aorta"],
    max_samples: Optional[int] = None,
    batch_size: int = 16,
    device: str = "cuda",
    output_path: Optional[str] = None,
) -> Dict:
    """Run simple segmentation evaluation across MedSAM layers."""

    print("=" * 60)
    print("SIMPLE SEGMENTATION EVALUATION - MedSAM")
    print("=" * 60)
    print(f"Layers: {layers}")
    print(f"Head type: {head_type}")
    print(f"Epochs: {epochs}")
    print(f"Labels: {labels}")
    print(f"MedSAM path: {medsam_path}")
    print("=" * 60)

    # Load MedSAM feature extractor
    print("\nLoading MedSAM feature extractor...")
    feature_extractor = MedSAMFeatureExtractor(
        checkpoint_path=medsam_path,
        device=device,
        layer_idx=11,
        freeze=True,
    )

    # Create dataloaders
    print("\nLoading datasets...")
    train_dataset = TotalSeg2DDataset(
        root_dir=root_dir,
        stats_path=stats_path,
        label_id_list=labels,
        context_size=0,
        image_size=(256, 256),
        split="train",
        load_dinov3_features=False,
        max_ds_len=max_samples,
    )

    val_dataset = TotalSeg2DDataset(
        root_dir=root_dir,
        stats_path=stats_path,
        label_id_list=labels,
        context_size=0,
        image_size=(256, 256),
        split="val",
        load_dinov3_features=False,
        max_ds_len=max_samples,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, collate_fn=collate_fn, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, collate_fn=collate_fn, pin_memory=True
    )

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    results = {}

    for layer_idx in layers:
        print(f"\n--- Layer {layer_idx} ---")

        layer_results = train_and_eval_layer(
            layer_idx=layer_idx,
            feature_extractor=feature_extractor,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            head_type=head_type,
            epochs=epochs,
            lr=lr,
        )

        results[f"layer_{layer_idx}"] = layer_results
        print(f"  Final Dice: {layer_results['final_dice']:.4f}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY - MedSAM")
    print("=" * 60)
    print(f"{'Layer':<10} {'Final Dice':<15} {'Best Dice':<15}")
    print("-" * 40)

    for layer_idx in layers:
        r = results[f"layer_{layer_idx}"]
        print(f"{layer_idx:<10} {r['final_dice']:<15.4f} {r['best_dice']:<15.4f}")

    best_layer = max(layers, key=lambda l: results[f"layer_{l}"]["final_dice"])
    print(f"\nBest layer: {best_layer} (Dice: {results[f'layer_{best_layer}']['final_dice']:.4f})")

    # Save results
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="MedSAM simple segmentation evaluation")
    parser.add_argument("--layers", type=int, nargs="+", default=[2, 5, 8, 11],
                       help="MedSAM layers to compare (0-11)")
    parser.add_argument("--head-type", type=str, default="linear",
                       choices=["linear", "conv", "mlp"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--root-dir", type=str, default=DEFAULT_ROOT_DIR)
    parser.add_argument("--stats-path", type=str, default=DEFAULT_STATS_PATH)
    parser.add_argument("--medsam-path", type=str, default=DEFAULT_MEDSAM_PATH)
    parser.add_argument("--labels", type=str, nargs="+",
                       default=["liver", "spleen", "kidney_left", "kidney_right", "aorta"])
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default="./results/feature_extraction/simple_seg_eval_medsam.json")

    args = parser.parse_args()

    run_medsam_eval(
        layers=args.layers,
        head_type=args.head_type,
        epochs=args.epochs,
        lr=args.lr,
        root_dir=args.root_dir,
        stats_path=args.stats_path,
        medsam_path=args.medsam_path,
        labels=args.labels,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        device=args.device,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
