"""
Multi-Layer Fusion Experiment

Compare different strategies for combining features from multiple MedDINO layers:
- Average: Simple averaging across layers [2, 5, 8, 11]
- Learned weighted: Learnable weights for each layer
- Concat + projection: Concatenate + linear projection

Usage:
    python Experiments/feature_extraction/multilayer_fusion.py \
        --checkpoint /path/to/checkpoint.pt \
        --strategies average learned_weighted concat_proj \
        --context-size 3
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
from src.losses import build_loss_fn
from src.models.patch_icl import PatchICL


class FusionModule(nn.Module):
    """Module for fusing features from multiple layers."""

    def __init__(
        self,
        num_layers: int = 4,
        embed_dim: int = 768,
        fusion: str = "average",
    ):
        super().__init__()
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.fusion = fusion

        if fusion == "learned_weighted":
            self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)
        elif fusion == "concat_proj":
            self.proj = nn.Linear(embed_dim * num_layers, embed_dim)

    def forward(self, layer_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Fuse features from multiple layers.

        Args:
            layer_features: Dict mapping layer names to [B, N, D] features

        Returns:
            fused: [B, N, D] fused features
        """
        # Stack features: [num_layers, B, N, D]
        stacked = torch.stack(list(layer_features.values()), dim=0)

        if self.fusion == "average":
            return stacked.mean(dim=0)
        elif self.fusion == "learned_weighted":
            weights = F.softmax(self.layer_weights, dim=0).view(-1, 1, 1, 1)
            return (stacked * weights).sum(dim=0)
        elif self.fusion == "concat_proj":
            B, N, D = stacked.shape[1:]
            concat = stacked.permute(1, 2, 0, 3).reshape(B, N, -1)
            return self.proj(concat)
        else:
            raise ValueError(f"Unknown fusion: {self.fusion}")

    def get_weights(self) -> Optional[torch.Tensor]:
        """Get fusion weights (for learned_weighted)."""
        if self.fusion == "learned_weighted":
            return F.softmax(self.layer_weights, dim=0).detach()
        return None


def compute_dice(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute Dice score."""
    pred_binary = (torch.sigmoid(pred) > 0.5).float()
    target_binary = (target > 0).float()
    spatial_dims = tuple(range(2, pred_binary.dim()))
    intersection = (pred_binary * target_binary).sum(dim=spatial_dims)
    union = pred_binary.sum(dim=spatial_dims) + target_binary.sum(dim=spatial_dims)
    return (2 * intersection + 1e-6) / (union + 1e-6)


def create_multilayer_dataloader(
    root_dir: str,
    stats_path: str,
    labels: List[str],
    layers: List[int],
    context_size: int,
    batch_size: int,
    max_samples: Optional[int] = None,
    split: str = "val",
) -> DataLoader:
    """Create dataloader that loads features from multiple layers."""
    dataset = TotalSeg2DDataset(
        root_dir=root_dir,
        stats_path=stats_path,
        label_id_list=labels,
        context_size=context_size,
        image_size=(256, 256),
        split=split,
        load_dinov3_features=True,
        feature_layers=layers,  # Load multiple layers
        max_ds_len=max_samples,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
    )


def multilayer_collate_fn(batch: List[Dict]) -> Dict:
    """Custom collate for multi-layer features."""
    result = collate_fn(batch)

    # Handle multi-layer features
    if "target_features" in batch[0] and isinstance(batch[0]["target_features"], dict):
        # Stack features per layer
        layers = list(batch[0]["target_features"].keys())
        target_features = {}
        for layer in layers:
            target_features[layer] = torch.stack(
                [item["target_features"][layer] for item in batch]
            )
        result["target_features"] = target_features

        if "context_features" in batch[0] and batch[0]["context_features"] is not None:
            context_features = {}
            for layer in layers:
                context_features[layer] = torch.stack(
                    [item["context_features"][layer] for item in batch]
                )
            result["context_features"] = context_features

    return result


def evaluate_fusion_strategy(
    model: PatchICL,
    fusion_module: FusionModule,
    dataloader: DataLoader,
    device: str,
    strategy: str,
) -> Dict[str, float]:
    """Evaluate a fusion strategy."""
    model.eval()
    fusion_module.eval()

    total_dice = 0.0
    total_local_dice = 0.0
    total_samples = 0
    per_label_dice = {}

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating {strategy}"):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            context_in = batch.get("context_in")
            context_out = batch.get("context_out")
            if context_in is not None:
                context_in = context_in.to(device)
            if context_out is not None:
                context_out = context_out.to(device)

            # Get multi-layer features
            target_features_dict = batch.get("target_features")
            context_features_dict = batch.get("context_features")

            if target_features_dict is not None:
                # Move to device and fuse
                target_features_dict = {
                    k: v.to(device) for k, v in target_features_dict.items()
                }
                target_features = fusion_module(target_features_dict)
            else:
                target_features = None

            if context_features_dict is not None:
                # Fuse context features per image
                B, k = context_features_dict[list(context_features_dict.keys())[0]].shape[:2]
                context_features_list = []
                for ctx_idx in range(k):
                    ctx_dict = {
                        layer: feats[:, ctx_idx].to(device)
                        for layer, feats in context_features_dict.items()
                    }
                    fused = fusion_module(ctx_dict)
                    context_features_list.append(fused)
                context_features = torch.stack(context_features_list, dim=1)
            else:
                context_features = None

            if labels.dim() == 3:
                labels = labels.unsqueeze(1)

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

    # Get learned weights if applicable
    weights = fusion_module.get_weights()
    weights_dict = None
    if weights is not None:
        weights_dict = {f"layer_{i}": w.item() for i, w in enumerate(weights)}

    return {
        "strategy": strategy,
        "final_dice": avg_dice,
        "local_dice": avg_local,
        "per_label_dice": per_label_avg,
        "num_samples": total_samples,
        "learned_weights": weights_dict,
    }


def train_fusion_module(
    model: PatchICL,
    fusion_module: FusionModule,
    train_loader: DataLoader,
    device: str,
    epochs: int = 10,
    lr: float = 1e-4,
) -> None:
    """Train learned fusion weights while keeping model frozen."""
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    optimizer = torch.optim.Adam(fusion_module.parameters(), lr=lr)
    criterion = build_loss_fn("dice", None)

    for epoch in range(epochs):
        fusion_module.train()
        total_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Training fusion epoch {epoch+1}"):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            context_in = batch.get("context_in")
            context_out = batch.get("context_out")
            if context_in is not None:
                context_in = context_in.to(device)
            if context_out is not None:
                context_out = context_out.to(device)

            # Fuse features
            target_features_dict = batch.get("target_features")
            if target_features_dict is not None:
                target_features_dict = {
                    k: v.to(device) for k, v in target_features_dict.items()
                }
                target_features = fusion_module(target_features_dict)
            else:
                target_features = None

            context_features_dict = batch.get("context_features")
            if context_features_dict is not None:
                B, k = context_features_dict[list(context_features_dict.keys())[0]].shape[:2]
                context_features_list = []
                for ctx_idx in range(k):
                    ctx_dict = {
                        layer: feats[:, ctx_idx].to(device)
                        for layer, feats in context_features_dict.items()
                    }
                    fused = fusion_module(ctx_dict)
                    context_features_list.append(fused)
                context_features = torch.stack(context_features_list, dim=1)
            else:
                context_features = None

            if labels.dim() == 3:
                labels = labels.unsqueeze(1)

            optimizer.zero_grad()

            with torch.no_grad():
                outputs = model(
                    images,
                    labels=labels,
                    context_in=context_in,
                    context_out=context_out,
                    target_features=target_features,
                    context_features=context_features,
                    mode="train",
                )

            # Compute loss on final prediction
            loss = criterion(outputs["final_pred"], labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        weights = fusion_module.get_weights()
        if weights is not None:
            print(f"  Epoch {epoch+1}: Loss={avg_loss:.4f}, Weights={weights.tolist()}")
        else:
            print(f"  Epoch {epoch+1}: Loss={avg_loss:.4f}")


def run_multilayer_fusion(
    checkpoint_path: str,
    strategies: List[str] = ["average", "learned_weighted", "concat_proj"],
    layers: List[int] = [2, 5, 8, 11],
    context_size: int = 3,
    root_dir: str = "/data/TotalSeg2D",
    stats_path: str = "/data/TotalSeg2D/totalseg_stats.pkl",
    labels: List[str] = ["liver", "spleen", "kidney_left", "kidney_right", "aorta"],
    max_samples: Optional[int] = None,
    batch_size: int = 16,
    device: str = "cuda",
    fusion_epochs: int = 10,
    fusion_lr: float = 1e-4,
    output_path: Optional[str] = None,
) -> Dict:
    """Run multi-layer fusion experiment."""
    print(f"Running multi-layer fusion experiment")
    print(f"  Layers: {layers}")
    print(f"  Strategies: {strategies}")
    print(f"  Context size: {context_size}")

    # Load model
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

    # Create dataloaders
    val_loader = create_multilayer_dataloader(
        root_dir, stats_path, labels, layers, context_size, batch_size,
        max_samples, split="val"
    )

    results = {}

    for strategy in strategies:
        print(f"\n--- Testing {strategy} fusion ---")

        fusion_module = FusionModule(
            num_layers=len(layers),
            embed_dim=768,
            fusion=strategy,
        ).to(device)

        # Train learned strategies
        if strategy in ["learned_weighted", "concat_proj"]:
            print(f"  Training fusion module for {fusion_epochs} epochs...")
            train_loader = create_multilayer_dataloader(
                root_dir, stats_path, labels, layers, context_size, batch_size,
                max_samples, split="train"
            )
            train_fusion_module(
                model, fusion_module, train_loader, device,
                epochs=fusion_epochs, lr=fusion_lr
            )

        # Evaluate
        strategy_results = evaluate_fusion_strategy(
            model, fusion_module, val_loader, device, strategy
        )
        results[strategy] = strategy_results

        print(f"  Final Dice: {strategy_results['final_dice']:.4f}")
        print(f"  Local Dice: {strategy_results['local_dice']:.4f}")
        if strategy_results.get("learned_weights"):
            print(f"  Learned weights: {strategy_results['learned_weights']}")

    # Summary
    print("\n" + "="*50)
    print("SUMMARY - Multi-Layer Fusion")
    print("="*50)
    print(f"{'Strategy':<20} {'Final Dice':<15} {'Local Dice':<15}")
    print("-"*50)
    for strategy in strategies:
        r = results[strategy]
        print(f"{strategy:<20} {r['final_dice']:<15.4f} {r['local_dice']:<15.4f}")

    best_strategy = max(strategies, key=lambda s: results[s]["final_dice"])
    print(f"\nBest strategy: {best_strategy} (Dice: {results[best_strategy]['final_dice']:.4f})")

    # Save results
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Multi-layer fusion experiment")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--strategies", type=str, nargs="+",
                       default=["average", "learned_weighted", "concat_proj"])
    parser.add_argument("--layers", type=int, nargs="+", default=[2, 5, 8, 11])
    parser.add_argument("--context-size", type=int, default=3)
    parser.add_argument("--root-dir", type=str, default="/data/TotalSeg2D")
    parser.add_argument("--stats-path", type=str, default="/data/TotalSeg2D/totalseg_stats.pkl")
    parser.add_argument("--labels", type=str, nargs="+",
                       default=["liver", "spleen", "kidney_left", "kidney_right", "aorta"])
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--fusion-epochs", type=int, default=10)
    parser.add_argument("--fusion-lr", type=float, default=1e-4)
    parser.add_argument("--output", type=str, default="./results/multilayer_fusion.json")

    args = parser.parse_args()

    run_multilayer_fusion(
        checkpoint_path=args.checkpoint,
        strategies=args.strategies,
        layers=args.layers,
        context_size=args.context_size,
        root_dir=args.root_dir,
        stats_path=args.stats_path,
        labels=args.labels,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        device=args.device,
        fusion_epochs=args.fusion_epochs,
        fusion_lr=args.fusion_lr,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
