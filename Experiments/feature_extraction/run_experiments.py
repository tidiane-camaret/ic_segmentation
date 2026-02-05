"""
Main Runner Script for Feature Extraction Experiments

Runs all experiments in sequence and generates comparison reports.

Usage:
    # Run all experiments
    python Experiments/feature_extraction/run_experiments.py \
        --checkpoint /path/to/checkpoint.pt \
        --output-dir ./results/feature_extraction

    # Run specific experiments
    python Experiments/feature_extraction/run_experiments.py \
        --checkpoint /path/to/checkpoint.pt \
        --experiments layer_comparison multilayer_fusion
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Experiments.feature_extraction.layer_comparison import run_layer_comparison
from Experiments.feature_extraction.multilayer_fusion import run_multilayer_fusion


def run_all_experiments(
    checkpoint_path: str,
    output_dir: str = "./results/feature_extraction",
    experiments: Optional[List[str]] = None,
    context_size: int = 3,
    root_dir: str = "/data/TotalSeg2D",
    stats_path: str = "/data/TotalSeg2D/totalseg_stats.pkl",
    labels: List[str] = ["liver", "spleen", "kidney_left", "kidney_right", "aorta"],
    max_samples: Optional[int] = None,
    batch_size: int = 16,
    device: str = "cuda",
    medsam_checkpoint: Optional[str] = None,
) -> Dict:
    """Run all feature extraction experiments.

    Args:
        checkpoint_path: Path to PatchICL model checkpoint
        output_dir: Directory to save results
        experiments: List of experiments to run (None for all)
        context_size: Number of context examples
        root_dir: Path to TotalSeg2D
        stats_path: Path to stats file
        labels: Label IDs to evaluate
        max_samples: Maximum samples per experiment
        batch_size: Batch size
        device: Device (cuda/cpu)
        medsam_checkpoint: Path to MedSAM checkpoint (optional)

    Returns:
        Dict with all experiment results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_results = {
        "timestamp": timestamp,
        "checkpoint": checkpoint_path,
        "context_size": context_size,
        "labels": labels,
        "experiments": {},
    }

    available_experiments = ["layer_comparison", "multilayer_fusion", "medsam"]
    if experiments is None:
        experiments = ["layer_comparison", "multilayer_fusion"]  # MedSAM optional

    print("=" * 60)
    print("FEATURE EXTRACTION EXPERIMENTS")
    print("=" * 60)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output directory: {output_dir}")
    print(f"Experiments: {experiments}")
    print(f"Context size: {context_size}")
    print(f"Labels: {labels}")
    print("=" * 60)

    # Run experiments
    for exp_name in experiments:
        if exp_name not in available_experiments:
            print(f"\nWarning: Unknown experiment '{exp_name}', skipping")
            continue

        print(f"\n{'='*60}")
        print(f"RUNNING: {exp_name}")
        print(f"{'='*60}")

        try:
            if exp_name == "layer_comparison":
                results = run_layer_comparison(
                    checkpoint_path=checkpoint_path,
                    layers=[2, 5, 8, 11],
                    context_size=context_size,
                    root_dir=root_dir,
                    stats_path=stats_path,
                    labels=labels,
                    max_samples=max_samples,
                    batch_size=batch_size,
                    device=device,
                    output_path=output_dir / "layer_comparison.json",
                )
                all_results["experiments"]["layer_comparison"] = results

            elif exp_name == "multilayer_fusion":
                results = run_multilayer_fusion(
                    checkpoint_path=checkpoint_path,
                    strategies=["average", "learned_weighted", "concat_proj"],
                    layers=[2, 5, 8, 11],
                    context_size=context_size,
                    root_dir=root_dir,
                    stats_path=stats_path,
                    labels=labels,
                    max_samples=max_samples,
                    batch_size=batch_size,
                    device=device,
                    output_path=output_dir / "multilayer_fusion.json",
                )
                all_results["experiments"]["multilayer_fusion"] = results

            elif exp_name == "medsam":
                # Import here to avoid dependency issues if MedSAM not installed
                try:
                    from Experiments.feature_extraction.medsam_extractor import run_medsam_comparison
                    results = run_medsam_comparison(
                        checkpoint_path=checkpoint_path,
                        medsam_checkpoint=medsam_checkpoint,
                        context_size=context_size,
                        root_dir=root_dir,
                        stats_path=stats_path,
                        labels=labels,
                        max_samples=max_samples,
                        batch_size=batch_size // 2,  # Smaller batch for MedSAM
                        device=device,
                        output_path=output_dir / "medsam_features.json",
                    )
                    all_results["experiments"]["medsam"] = results
                except ImportError as e:
                    print(f"\nSkipping MedSAM experiment: {e}")
                    print("Install MedSAM: pip install git+https://github.com/bowang-lab/MedSAM.git")
                    continue

        except Exception as e:
            print(f"\nError in {exp_name}: {e}")
            import traceback
            traceback.print_exc()
            all_results["experiments"][exp_name] = {"error": str(e)}
            continue

    # Generate summary report
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)

    # Layer comparison summary
    if "layer_comparison" in all_results["experiments"]:
        lc = all_results["experiments"]["layer_comparison"]
        if "error" not in lc:
            print("\nLayer Comparison:")
            print(f"  {'Layer':<10} {'Final Dice':<15}")
            print(f"  {'-'*25}")
            for layer in [2, 5, 8, 11]:
                key = f"layer_{layer}"
                if key in lc:
                    print(f"  {layer:<10} {lc[key]['final_dice']:<15.4f}")

    # Multi-layer fusion summary
    if "multilayer_fusion" in all_results["experiments"]:
        mf = all_results["experiments"]["multilayer_fusion"]
        if "error" not in mf:
            print("\nMulti-Layer Fusion:")
            print(f"  {'Strategy':<20} {'Final Dice':<15}")
            print(f"  {'-'*35}")
            for strategy in ["average", "learned_weighted", "concat_proj"]:
                if strategy in mf:
                    print(f"  {strategy:<20} {mf[strategy]['final_dice']:<15.4f}")

    # MedSAM summary
    if "medsam" in all_results["experiments"]:
        ms = all_results["experiments"]["medsam"]
        if "error" not in ms:
            print(f"\nMedSAM v1: Final Dice = {ms['final_dice']:.4f}")

    # Save combined results
    combined_path = output_dir / f"all_results_{timestamp}.json"
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nCombined results saved to {combined_path}")

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Run feature extraction experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run all experiments (except MedSAM)
    python run_experiments.py --checkpoint model.pt

    # Run specific experiments
    python run_experiments.py --checkpoint model.pt --experiments layer_comparison

    # Include MedSAM experiment
    python run_experiments.py --checkpoint model.pt --experiments layer_comparison medsam
        """
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to PatchICL model checkpoint")
    parser.add_argument("--output-dir", type=str, default="./results/feature_extraction",
                       help="Directory to save results")
    parser.add_argument("--experiments", type=str, nargs="+", default=None,
                       help="Experiments to run: layer_comparison, multilayer_fusion, medsam")
    parser.add_argument("--context-size", type=int, default=3)
    parser.add_argument("--root-dir", type=str, default="/data/TotalSeg2D")
    parser.add_argument("--stats-path", type=str, default="/data/TotalSeg2D/totalseg_stats.pkl")
    parser.add_argument("--labels", type=str, nargs="+",
                       default=["liver", "spleen", "kidney_left", "kidney_right", "aorta"])
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--medsam-checkpoint", type=str, default=None,
                       help="Path to MedSAM v1 checkpoint (for medsam experiment)")

    args = parser.parse_args()

    run_all_experiments(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        experiments=args.experiments,
        context_size=args.context_size,
        root_dir=args.root_dir,
        stats_path=args.stats_path,
        labels=args.labels,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        device=args.device,
        medsam_checkpoint=args.medsam_checkpoint,
    )


if __name__ == "__main__":
    main()
