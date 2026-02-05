"""
Analysis and Visualization for Feature Extraction Experiments

Generates comparison plots and tables from experiment results.

Usage:
    python Experiments/feature_extraction/analysis.py \
        --results-dir ./results/feature_extraction \
        --output-dir ./results/feature_extraction/plots
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# Optional plotting imports
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available, plotting disabled")

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


def load_results(results_dir: str) -> Dict:
    """Load all experiment results from directory."""
    results_dir = Path(results_dir)
    results = {}

    # Load individual experiment results
    for json_file in results_dir.glob("*.json"):
        if json_file.name.startswith("all_results"):
            # Skip combined results, load individual ones
            continue
        exp_name = json_file.stem
        with open(json_file) as f:
            results[exp_name] = json.load(f)

    # Also try to load combined results
    combined_files = list(results_dir.glob("all_results_*.json"))
    if combined_files:
        latest = max(combined_files, key=lambda p: p.stat().st_mtime)
        with open(latest) as f:
            results["combined"] = json.load(f)

    return results


def plot_layer_comparison(
    results: Dict,
    output_path: Optional[str] = None,
    figsize: tuple = (10, 6),
) -> None:
    """Plot layer comparison results."""
    if not HAS_MATPLOTLIB:
        print("Matplotlib not available, skipping plot")
        return

    layers = []
    final_dice = []
    local_dice = []

    for key in sorted(results.keys()):
        if key.startswith("layer_"):
            layer_idx = int(key.split("_")[1])
            layers.append(layer_idx)
            final_dice.append(results[key]["final_dice"])
            local_dice.append(results[key]["local_dice"])

    x = np.arange(len(layers))
    width = 0.35

    fig, ax = plt.subplots(figsize=figsize)
    bars1 = ax.bar(x - width/2, final_dice, width, label="Final Dice", color="#2ecc71")
    bars2 = ax.bar(x + width/2, local_dice, width, label="Local Dice", color="#3498db")

    ax.set_xlabel("MedDINO Layer")
    ax.set_ylabel("Dice Score")
    ax.set_title("Feature Quality by MedDINO Layer")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Layer {l}" for l in layers])
    ax.legend()
    ax.set_ylim(0, 1)

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved layer comparison plot to {output_path}")
    else:
        plt.show()

    plt.close()


def plot_fusion_comparison(
    results: Dict,
    output_path: Optional[str] = None,
    figsize: tuple = (10, 6),
) -> None:
    """Plot multi-layer fusion comparison results."""
    if not HAS_MATPLOTLIB:
        print("Matplotlib not available, skipping plot")
        return

    strategies = []
    final_dice = []
    local_dice = []

    for strategy in ["average", "learned_weighted", "concat_proj"]:
        if strategy in results:
            strategies.append(strategy.replace("_", "\n"))
            final_dice.append(results[strategy]["final_dice"])
            local_dice.append(results[strategy]["local_dice"])

    x = np.arange(len(strategies))
    width = 0.35

    fig, ax = plt.subplots(figsize=figsize)
    bars1 = ax.bar(x - width/2, final_dice, width, label="Final Dice", color="#e74c3c")
    bars2 = ax.bar(x + width/2, local_dice, width, label="Local Dice", color="#9b59b6")

    ax.set_xlabel("Fusion Strategy")
    ax.set_ylabel("Dice Score")
    ax.set_title("Multi-Layer Feature Fusion Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(strategies)
    ax.legend()
    ax.set_ylim(0, 1)

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved fusion comparison plot to {output_path}")
    else:
        plt.show()

    plt.close()


def plot_learned_weights(
    results: Dict,
    layers: List[int] = [2, 5, 8, 11],
    output_path: Optional[str] = None,
    figsize: tuple = (8, 5),
) -> None:
    """Plot learned layer weights from fusion experiment."""
    if not HAS_MATPLOTLIB:
        print("Matplotlib not available, skipping plot")
        return

    if "learned_weighted" not in results:
        print("No learned_weighted results found")
        return

    weights_dict = results["learned_weighted"].get("learned_weights")
    if not weights_dict:
        print("No learned weights found in results")
        return

    weights = [weights_dict.get(f"layer_{i}", 0) for i in range(len(layers))]

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(range(len(layers)), weights, color="#f39c12")

    ax.set_xlabel("MedDINO Layer")
    ax.set_ylabel("Learned Weight")
    ax.set_title("Learned Layer Weights for Multi-Layer Fusion")
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels([f"Layer {l}" for l in layers])
    ax.set_ylim(0, max(weights) * 1.2 if weights else 1)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=10)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved learned weights plot to {output_path}")
    else:
        plt.show()

    plt.close()


def plot_per_label_comparison(
    results: Dict,
    experiment_name: str,
    output_path: Optional[str] = None,
    figsize: tuple = (12, 6),
) -> None:
    """Plot per-label Dice scores for an experiment."""
    if not HAS_MATPLOTLIB:
        print("Matplotlib not available, skipping plot")
        return

    # Collect all labels and configurations
    all_labels = set()
    configs = []

    for key, value in results.items():
        if isinstance(value, dict) and "per_label_dice" in value:
            configs.append(key)
            all_labels.update(value["per_label_dice"].keys())

    if not configs or not all_labels:
        print(f"No per-label data found for {experiment_name}")
        return

    labels = sorted(all_labels)
    x = np.arange(len(labels))
    width = 0.8 / len(configs)

    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.tab10(np.linspace(0, 1, len(configs)))

    for i, config in enumerate(configs):
        dice_scores = [results[config]["per_label_dice"].get(label, 0) for label in labels]
        offset = (i - len(configs)/2 + 0.5) * width
        ax.bar(x + offset, dice_scores, width, label=config, color=colors[i])

    ax.set_xlabel("Label")
    ax.set_ylabel("Dice Score")
    ax.set_title(f"Per-Label Dice Scores: {experiment_name}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend(loc="upper right")
    ax.set_ylim(0, 1)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved per-label plot to {output_path}")
    else:
        plt.show()

    plt.close()


def generate_summary_table(results: Dict) -> str:
    """Generate a markdown summary table of all experiments."""
    lines = []
    lines.append("# Feature Extraction Experiment Results\n")

    # Layer comparison
    if "layer_comparison" in results:
        lines.append("## Layer Comparison\n")
        lines.append("| Layer | Final Dice | Local Dice |")
        lines.append("|-------|------------|------------|")
        lc = results["layer_comparison"]
        for layer in [2, 5, 8, 11]:
            key = f"layer_{layer}"
            if key in lc:
                fd = lc[key]["final_dice"]
                ld = lc[key]["local_dice"]
                lines.append(f"| {layer} | {fd:.4f} | {ld:.4f} |")
        lines.append("")

    # Multi-layer fusion
    if "multilayer_fusion" in results:
        lines.append("## Multi-Layer Fusion\n")
        lines.append("| Strategy | Final Dice | Local Dice |")
        lines.append("|----------|------------|------------|")
        mf = results["multilayer_fusion"]
        for strategy in ["average", "learned_weighted", "concat_proj"]:
            if strategy in mf:
                fd = mf[strategy]["final_dice"]
                ld = mf[strategy]["local_dice"]
                lines.append(f"| {strategy} | {fd:.4f} | {ld:.4f} |")
        lines.append("")

    # MedSAM
    if "medsam_features" in results:
        lines.append("## MedSAM v1 Features\n")
        ms = results["medsam_features"]
        lines.append(f"- Final Dice: {ms['final_dice']:.4f}")
        lines.append(f"- Local Dice: {ms['local_dice']:.4f}")
        lines.append("")

    return "\n".join(lines)


def analyze_results(
    results_dir: str,
    output_dir: Optional[str] = None,
) -> None:
    """Load results and generate all analysis outputs."""
    results = load_results(results_dir)

    if output_dir is None:
        output_dir = Path(results_dir) / "plots"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loaded results: {list(results.keys())}")

    # Generate plots
    if "layer_comparison" in results:
        plot_layer_comparison(
            results["layer_comparison"],
            output_path=output_dir / "layer_comparison.png"
        )
        plot_per_label_comparison(
            results["layer_comparison"],
            "Layer Comparison",
            output_path=output_dir / "layer_comparison_per_label.png"
        )

    if "multilayer_fusion" in results:
        plot_fusion_comparison(
            results["multilayer_fusion"],
            output_path=output_dir / "fusion_comparison.png"
        )
        plot_learned_weights(
            results["multilayer_fusion"],
            output_path=output_dir / "learned_weights.png"
        )
        plot_per_label_comparison(
            results["multilayer_fusion"],
            "Multi-Layer Fusion",
            output_path=output_dir / "fusion_per_label.png"
        )

    # Generate summary table
    summary = generate_summary_table(results)
    summary_path = output_dir / "summary.md"
    with open(summary_path, "w") as f:
        f.write(summary)
    print(f"Saved summary to {summary_path}")

    # Print summary to console
    print("\n" + summary)


def main():
    parser = argparse.ArgumentParser(description="Analyze feature extraction experiments")
    parser.add_argument("--results-dir", type=str, default="./results/feature_extraction",
                       help="Directory containing experiment results")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Directory to save plots (default: results_dir/plots)")

    args = parser.parse_args()
    analyze_results(args.results_dir, args.output_dir)


if __name__ == "__main__":
    main()
