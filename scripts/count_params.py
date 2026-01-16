"""
Count parameters for each module in TokenNMSW model.

Usage:
    python scripts/count_params.py
    python scripts/count_params.py --pos-encoding relative
    python scripts/count_params.py --num-layers 12
"""

import argparse
import sys
from pathlib import Path

import torch

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.token_nmsw import TokenNMSW


def count_parameters(model, detailed=True):
    """Count and display parameters for each module."""
    print("\n" + "=" * 70)
    print("Model Architecture & Parameters")
    print("=" * 70)
    
    if detailed:
        total_params = 0
        for name, module in model.named_children():
            num_params = sum(p.numel() for p in module.parameters())
            trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            total_params += trainable_params
            
            if num_params != trainable_params:
                print(f"{name:25s}: {trainable_params:>12,} trainable / {num_params:>12,} total")
            else:
                print(f"{name:25s}: {num_params:>12,} params")
        
        print("=" * 70)
        print(f"{'Total Trainable':25s}: {total_params:>12,} params")
        print("=" * 70)
    else:
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total:,}")
        print(f"Trainable parameters: {trainable:,}")
        print("=" * 70)
    
    return total_params if detailed else trainable


def main():
    parser = argparse.ArgumentParser(description="Count TokenNMSW parameters")
    parser.add_argument("--pos-encoding", type=str, default=None,
                        choices=["sinusoidal", "learnable", "relative"],
                        help="Positional encoding type")
    parser.add_argument("--num-layers", type=int, default=None,
                        help="Number of transformer layers")
    parser.add_argument("--num-patches", type=int, default=None,
                        help="Number of patches to select")
    parser.add_argument("--embed-dim", type=int, default=None,
                        help="Embedding dimension")
    parser.add_argument("--patch-size", type=int, default=None,
                        help="Patch size")
    parser.add_argument("--simple", action="store_true",
                        help="Show only total count")
    args = parser.parse_args()

    # Load config
    config = load_config()
    train_cfg = config.get("train_totalseg", {})
    token_cfg = train_cfg.get("token_nmsw", {})

    # Apply CLI overrides
    if args.pos_encoding:
        token_cfg["pos_encoding"] = args.pos_encoding
    if args.num_layers:
        token_cfg["num_layers"] = args.num_layers
    if args.num_patches:
        token_cfg["num_patches"] = args.num_patches
    if args.embed_dim:
        token_cfg["embed_dim"] = args.embed_dim
    if args.patch_size:
        token_cfg["patch_size"] = args.patch_size

    # Build model config
    model_params = {
        "in_channels": 1,
        "num_classes": 1,
        "patch_size": token_cfg.get("patch_size", 8),
        "num_patches": token_cfg.get("num_patches", 100),
        "num_random_patches": token_cfg.get("num_random_patches", 0),
        "global_base_channels": token_cfg.get("global_base_channels", 16),
        "down_size_rate": tuple(token_cfg.get("down_size_rate", [2, 2, 2])),
        "embed_dim": token_cfg.get("embed_dim", 256),
        "num_heads": token_cfg.get("num_heads", 8),
        "num_layers": token_cfg.get("num_layers", 8),
        "mlp_ratio": token_cfg.get("mlp_ratio", 4.0),
        "dropout": token_cfg.get("dropout", 0.1),
        "pos_encoding": token_cfg.get("pos_encoding", "learnable"),
        "tau": token_cfg.get("starting_tau", 2/3),
        "global_loss_weight": token_cfg.get("global_loss_weight", 1.0),
        "local_loss_weight": token_cfg.get("local_loss_weight", 1.0),
        "agg_loss_weight": token_cfg.get("agg_loss_weight", 1.0),
        "entropy_multiplier": token_cfg.get("entropy_multiplier", 1e-5),
    }

    print("\nModel Configuration:")
    print(f"  Patch size: {model_params['patch_size']}³")
    print(f"  Num patches: {model_params['num_patches']}")
    print(f"  Embedding dim: {model_params['embed_dim']}")
    print(f"  Transformer layers: {model_params['num_layers']}")
    print(f"  Attention heads: {model_params['num_heads']}")
    print(f"  Positional encoding: {model_params['pos_encoding']}")

    # Build model
    model = TokenNMSW(**model_params)

    # Count parameters
    count_parameters(model, detailed=not args.simple)


if __name__ == "__main__":
    main()
