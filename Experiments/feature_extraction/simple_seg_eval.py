"""
Simple Segmentation Evaluation for Feature Comparison

Evaluates feature quality by training a simple segmentation head on top of
frozen features from different MedDINO layers.

Features are extracted on-the-fly using the MedDINO model (no precomputed features).

Architecture:
    Image -> MedDINO (frozen) -> Patches [B, 256, 768] -> Simple Head -> Mask [B, 1, H, W]

Simple Head options:
    - linear: Single linear layer + reshape + upsample
    - conv: 1x1 conv + upsample
    - mlp: 2-layer MLP + reshape + upsample

Usage:
    python Experiments/feature_extraction/simple_seg_eval.py \
        --layers 2 5 8 11 \
        --head-type linear \
        --epochs 10
"""
import argparse
import json
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
from src.models.meddino_extractor import MedDINOFeatureExtractor


# Default MedDINO checkpoint path
DEFAULT_MEDDINO_PATH = "/nfs/data/nii/data1/Analysis/camaret___in_context_segmentation/ANALYSIS_20251122/checkpoints/med_dino_v3model.pth"


class SimpleSegHead(nn.Module):
    """Simple segmentation head for feature evaluation."""

    def __init__(
        self,
        in_dim: int = 768,
        hidden_dim: int = 256,
        out_size: int = 256,
        num_patches: int = 256,
        head_type: str = "linear",
    ):
        """
        Args:
            in_dim: Input feature dimension (768 for MedDINO)
            hidden_dim: Hidden dimension for MLP head
            out_size: Output spatial size
            num_patches: Number of patch tokens (will be padded to nearest square if needed)
            head_type: "linear", "conv", or "mlp"
        """
        super().__init__()
        self.in_dim = in_dim
        self.out_size = out_size
        self.num_patches = num_patches
        self.head_type = head_type

        # Compute grid size (pad to nearest square)
        import math
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
        """
        Args:
            features: [B, N, D] where N = num_patches (patch tokens only, no CLS/registers)

        Returns:
            logits: [B, 1, out_size, out_size]
        """
        B, N, D = features.shape

        # Pad to square if needed
        if self.pad_amount > 0:
            padding = torch.zeros(B, self.pad_amount, D, device=features.device, dtype=features.dtype)
            features = torch.cat([features, padding], dim=1)

        if self.head_type == "conv":
            # Reshape to spatial: [B, D, H, W]
            x = features.permute(0, 2, 1).view(B, D, self.feature_grid, self.feature_grid)
            x = self.head(x)  # [B, 1, H, W]
        else:
            # Linear or MLP: process each token
            x = self.head(features)  # [B, N, 1]
            x = x.permute(0, 2, 1).view(B, 1, self.feature_grid, self.feature_grid)

        # Upsample to output size
        x = F.interpolate(x, size=(self.out_size, self.out_size), mode="bilinear", align_corners=False)

        return x


def compute_dice(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute Dice score."""
    pred_binary = (torch.sigmoid(pred) > 0.5).float()
    target_binary = (target > 0).float()

    intersection = (pred_binary * target_binary).sum(dim=(1, 2, 3))
    union = pred_binary.sum(dim=(1, 2, 3)) + target_binary.sum(dim=(1, 2, 3))

    return (2 * intersection + 1e-6) / (union + 1e-6)


def dice_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Dice loss."""
    pred_prob = torch.sigmoid(pred)
    target_binary = (target > 0).float()

    intersection = (pred_prob * target_binary).sum(dim=(1, 2, 3))
    union = pred_prob.sum(dim=(1, 2, 3)) + target_binary.sum(dim=(1, 2, 3))

    dice = (2 * intersection + 1e-6) / (union + 1e-6)
    return 1 - dice.mean()


def train_and_eval_layer(
    layer_idx: int,
    feature_extractor: MedDINOFeatureExtractor,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str,
    head_type: str = "linear",
    epochs: int = 10,
    lr: float = 1e-3,
) -> Dict:
    """Train and evaluate a simple segmentation head on features from one layer."""

    # Update feature extractor to use this layer
    feature_extractor.layer_idx = layer_idx

    # Determine feature grid size from first batch
    with torch.no_grad():
        sample_batch = next(iter(train_loader))
        sample_img = sample_batch["image"][:1].to(device)
        sample_features = feature_extractor.extract_features(sample_img)
        # Features: [B, total_tokens, D] where total_tokens = 1 CLS + 4 registers + patches
        num_patches = sample_features.shape[1] - 5  # Subtract CLS and registers
        import math
        feature_grid = int(math.ceil(math.sqrt(num_patches)))
        print(f"  Feature shape: {sample_features.shape}, patches: {num_patches}, grid: {feature_grid}x{feature_grid}")

    # Create segmentation head
    head = SimpleSegHead(
        in_dim=768,
        hidden_dim=256,
        out_size=256,
        num_patches=num_patches,
        head_type=head_type,
    ).to(device)

    optimizer = torch.optim.Adam(head.parameters(), lr=lr)

    # Training
    best_dice = 0.0
    train_losses = []

    for epoch in range(epochs):
        head.train()
        epoch_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Layer {layer_idx} - Epoch {epoch+1}/{epochs}", leave=False):
            images = batch["image"].to(device)  # [B, 1, H, W]
            labels = batch["label"].to(device)  # [B, 1, H, W]

            if labels.dim() == 3:
                labels = labels.unsqueeze(1)

            # Extract features on-the-fly
            with torch.no_grad():
                features = feature_extractor.extract_features(images)
                # Extract only patch tokens (skip CLS at 0 and registers at 1:5)
                patch_features = features[:, 5:, :]  # [B, num_patches, D]

            optimizer.zero_grad()
            logits = head(patch_features)
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
                patch_features = features[:, 5:, :]

                logits = head(patch_features)
                dice = compute_dice(logits, labels)
                val_dice += dice.sum().item()
                val_samples += labels.shape[0]

        avg_dice = val_dice / val_samples

        if avg_dice > best_dice:
            best_dice = avg_dice

        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            print(f"  Epoch {epoch+1}: Loss={avg_loss:.4f}, Val Dice={avg_dice:.4f}")

    # Final evaluation with per-label metrics
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
            patch_features = features[:, 5:, :]

            logits = head(patch_features)
            dice = compute_dice(logits, labels)

            # Per-label tracking
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


def run_simple_seg_eval(
    layers: List[int] = [2, 5, 8, 11],
    head_type: str = "linear",
    epochs: int = 10,
    lr: float = 1e-3,
    root_dir: str = "/data/TotalSeg2D",
    stats_path: str = "/data/TotalSeg2D/totalseg_stats.pkl",
    meddino_path: str = DEFAULT_MEDDINO_PATH,
    labels: List[str] = ["liver", "spleen", "kidney_left", "kidney_right", "aorta"],
    max_samples: Optional[int] = None,
    batch_size: int = 16,
    device: str = "cuda",
    output_path: Optional[str] = None,
) -> Dict:
    """Run simple segmentation evaluation across layers."""

    print("=" * 60)
    print("SIMPLE SEGMENTATION EVALUATION (On-the-fly Features)")
    print("=" * 60)
    print(f"Layers: {layers}")
    print(f"Head type: {head_type}")
    print(f"Epochs: {epochs}")
    print(f"Labels: {labels}")
    print(f"MedDINO path: {meddino_path}")
    print("=" * 60)

    # Load MedDINO feature extractor once (will change layer_idx per experiment)
    print("\nLoading MedDINO feature extractor...")
    feature_extractor = MedDINOFeatureExtractor(
        model_path=meddino_path,
        target_size=256,
        device=device,
        layer_idx=11,  # Will be changed per layer
        freeze=True,
    )

    # Create dataloaders (without loading precomputed features)
    print("\nLoading datasets...")
    train_dataset = TotalSeg2DDataset(
        root_dir=root_dir,
        stats_path=stats_path,
        label_id_list=labels,
        context_size=0,  # No context needed for simple eval
        image_size=(256, 256),
        split="train",
        load_dinov3_features=False,  # Extract on-the-fly
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

        # Train and evaluate
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
    print("SUMMARY")
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
    parser = argparse.ArgumentParser(description="Simple segmentation evaluation")
    parser.add_argument("--layers", type=int, nargs="+", default=[2, 5, 8, 11],
                       help="MedDINO layers to compare")
    parser.add_argument("--head-type", type=str, default="linear",
                       choices=["linear", "conv", "mlp"],
                       help="Segmentation head type")
    parser.add_argument("--epochs", type=int, default=10,
                       help="Training epochs per layer")
    parser.add_argument("--lr", type=float, default=1e-3,
                       help="Learning rate")
    parser.add_argument("--root-dir", type=str, default="/nfs/data/nii/data1/Analysis/camaret___in_context_segmentation/ANALYSIS_20251122/data/TotalSeg2D")
    parser.add_argument("--stats-path", type=str, default="/nfs/data/nii/data1/Analysis/camaret___in_context_segmentation/ANALYSIS_20251122/data/TotalSeg/totalseg_stats.pkl")
    parser.add_argument("--meddino-path", type=str, default=DEFAULT_MEDDINO_PATH,
                       help="Path to MedDINO checkpoint")
    parser.add_argument("--labels", type=str, nargs="+",
                       default=["liver", "spleen", "kidney_left", "kidney_right", "aorta"])
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default="./results/feature_extraction/simple_seg_eval.json")

    args = parser.parse_args()

    run_simple_seg_eval(
        layers=args.layers,
        head_type=args.head_type,
        epochs=args.epochs,
        lr=args.lr,
        root_dir=args.root_dir,
        stats_path=args.stats_path,
        meddino_path=args.meddino_path,
        labels=args.labels,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        device=args.device,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
