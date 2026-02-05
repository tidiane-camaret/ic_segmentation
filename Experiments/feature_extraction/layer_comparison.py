"""
Layer Comparison Experiment

Compare MedDINO features from different transformer layers:
- Layer 2: Early - edges, textures
- Layer 5: Mid - patterns, local structures
- Layer 8: Late-mid - object parts
- Layer 11: Final - semantic features (current default)

Usage:
    python Experiments/feature_extraction/layer_comparison.py \
        --checkpoint /path/to/checkpoint.pt \
        --layers 2 5 8 11 \
        --context-size 3
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.dataloaders.totalseg2d_dataloader import get_dataloader
from src.losses import build_loss_fn
from src.models.patch_icl import PatchICL


def compute_dice(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute Dice score between prediction and target."""
    pred_binary = (torch.sigmoid(pred) > 0.5).float()
    target_binary = (target > 0).float()

    # Flatten spatial dimensions
    spatial_dims = tuple(range(2, pred_binary.dim()))
    intersection = (pred_binary * target_binary).sum(dim=spatial_dims)
    union = pred_binary.sum(dim=spatial_dims) + target_binary.sum(dim=spatial_dims)

    dice = (2 * intersection + 1e-6) / (union + 1e-6)
    return dice


def evaluate_layer(
    model: PatchICL,
    dataloader,
    device: str,
    layer_idx: int,
) -> Dict[str, float]:
    """Evaluate model performance using features from a specific layer.

    Args:
        model: PatchICL model
        dataloader: DataLoader with features from specified layer
        device: Device for computation
        layer_idx: Layer index being evaluated (for logging)

    Returns:
        Dict with evaluation metrics
    """
    model.eval()

    total_local_dice = 0.0
    total_final_dice = 0.0
    total_samples = 0
    per_label_dice = {}

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating layer {layer_idx}"):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            context_in = batch.get("context_in")
            context_out = batch.get("context_out")
            if context_in is not None:
                context_in = context_in.to(device)
            if context_out is not None:
                context_out = context_out.to(device)

            target_features = batch.get("target_features")
            context_features = batch.get("context_features")
            if target_features is not None:
                target_features = target_features.to(device)
            if context_features is not None:
                context_features = context_features.to(device)

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

            # Compute metrics
            final_dice = compute_dice(outputs["final_pred"], labels)
            total_final_dice += final_dice.sum().item()

            # Patch-level dice
            patch_logits = outputs["patch_logits"]
            patch_labels = outputs["patch_labels"]
            patch_dice = compute_dice(
                patch_logits.flatten(0, 1),
                patch_labels.flatten(0, 1)
            )
            total_local_dice += patch_dice.sum().item()

            # Per-label tracking
            batch_labels = batch.get("label_ids", [])
            for i, label_id in enumerate(batch_labels):
                if label_id not in per_label_dice:
                    per_label_dice[label_id] = []
                per_label_dice[label_id].append(final_dice[i].item())

            total_samples += images.shape[0]

    # Compute averages
    avg_final_dice = total_final_dice / total_samples
    avg_local_dice = total_local_dice / (total_samples * patch_logits.shape[1])

    per_label_avg = {
        label: sum(scores) / len(scores)
        for label, scores in per_label_dice.items()
    }

    return {
        "layer": layer_idx,
        "final_dice": avg_final_dice,
        "local_dice": avg_local_dice,
        "per_label_dice": per_label_avg,
        "num_samples": total_samples,
    }


def run_layer_comparison(
    checkpoint_path: str,
    layers: List[int] = [2, 5, 8, 11],
    context_size: int = 3,
    root_dir: str = "/data/TotalSeg2D",
    stats_path: str = "/data/TotalSeg2D/totalseg_stats.pkl",
    labels: List[str] = ["liver", "spleen", "kidney_left", "kidney_right", "aorta"],
    max_samples: Optional[int] = None,
    batch_size: int = 16,
    device: str = "cuda",
    output_path: Optional[str] = None,
) -> Dict:
    """Run layer comparison experiment.

    Args:
        checkpoint_path: Path to model checkpoint
        layers: List of MedDINO layer indices to test
        context_size: Number of context examples
        root_dir: Path to TotalSeg2D directory
        stats_path: Path to stats file
        labels: Label IDs to evaluate
        max_samples: Maximum samples (None for all)
        batch_size: Batch size for evaluation
        device: Device for computation
        output_path: Path to save results JSON

    Returns:
        Dict with results for each layer
    """
    print(f"Running layer comparison experiment")
    print(f"  Layers: {layers}")
    print(f"  Context size: {context_size}")
    print(f"  Labels: {labels}")

    # Load model
    print(f"\nLoading model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Reconstruct model config from checkpoint
    config = checkpoint.get("config", {})
    patch_icl_cfg = config.get("model", {}).get("patch_icl", {})

    # Filter out feature_extractor keys (may be saved if model was trained with on-the-fly extraction)
    state_dict = checkpoint["model_state_dict"]
    filtered_state_dict = {
        k: v for k, v in state_dict.items()
        if not k.startswith("feature_extractor.")
    }

    # Infer num_registers from state_dict if there's a mismatch
    for key, value in filtered_state_dict.items():
        if "register_tokens" in key:
            actual_num_registers = value.shape[1]
            if patch_icl_cfg.get("backbone", {}).get("num_registers") != actual_num_registers:
                print(f"  Adjusting num_registers: {patch_icl_cfg.get('backbone', {}).get('num_registers')} -> {actual_num_registers}")
                if "backbone" not in patch_icl_cfg:
                    patch_icl_cfg["backbone"] = {}
                patch_icl_cfg["backbone"]["num_registers"] = actual_num_registers
            break

    model = PatchICL(patch_icl_cfg, context_size=context_size)
    model.load_state_dict(filtered_state_dict, strict=False)
    model.to(device)
    model.eval()

    # Set loss functions
    patch_criterion = build_loss_fn("dice", None)
    aggreg_criterion = build_loss_fn("dice", None)
    model.set_loss_functions(patch_criterion, aggreg_criterion)

    results = {}

    for layer_idx in layers:
        print(f"\n--- Testing Layer {layer_idx} ---")

        # Create dataloader with features from this layer
        dataloader = get_dataloader(
            root_dir=root_dir,
            stats_path=stats_path,
            label_id_list=labels,
            context_size=context_size,
            batch_size=batch_size,
            image_size=(256, 256),
            num_workers=4,
            split="val",
            shuffle=False,
            load_dinov3_features=True,
            feature_layer_idx=layer_idx,
            max_ds_len=max_samples,
        )

        # Evaluate
        layer_results = evaluate_layer(model, dataloader, device, layer_idx)
        results[f"layer_{layer_idx}"] = layer_results

        print(f"  Final Dice: {layer_results['final_dice']:.4f}")
        print(f"  Local Dice: {layer_results['local_dice']:.4f}")
        print(f"  Per-label:")
        for label, dice in sorted(layer_results['per_label_dice'].items()):
            print(f"    {label}: {dice:.4f}")

    # Summary comparison
    print("\n" + "="*50)
    print("SUMMARY - Layer Comparison")
    print("="*50)
    print(f"{'Layer':<10} {'Final Dice':<15} {'Local Dice':<15}")
    print("-"*40)
    for layer_idx in layers:
        r = results[f"layer_{layer_idx}"]
        print(f"{layer_idx:<10} {r['final_dice']:<15.4f} {r['local_dice']:<15.4f}")

    # Find best layer
    best_layer = max(layers, key=lambda l: results[f"layer_{l}"]["final_dice"])
    print(f"\nBest layer: {best_layer} (Final Dice: {results[f'layer_{best_layer}']['final_dice']:.4f})")

    # Save results
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Layer comparison experiment")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--layers", type=int, nargs="+", default=[2, 5, 8, 11],
                       help="MedDINO layers to compare")
    parser.add_argument("--context-size", type=int, default=3,
                       help="Number of context examples")
    parser.add_argument("--root-dir", type=str, default="/data/TotalSeg2D",
                       help="Path to TotalSeg2D directory")
    parser.add_argument("--stats-path", type=str, default="/data/TotalSeg2D/totalseg_stats.pkl",
                       help="Path to stats file")
    parser.add_argument("--labels", type=str, nargs="+",
                       default=["liver", "spleen", "kidney_left", "kidney_right", "aorta"],
                       help="Label IDs to evaluate")
    parser.add_argument("--max-samples", type=int, default=None,
                       help="Maximum samples per layer (None for all)")
    parser.add_argument("--batch-size", type=int, default=16,
                       help="Batch size")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device (cuda/cpu)")
    parser.add_argument("--output", type=str, default="./results/layer_comparison.json",
                       help="Path to save results JSON")

    args = parser.parse_args()

    run_layer_comparison(
        checkpoint_path=args.checkpoint,
        layers=args.layers,
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
