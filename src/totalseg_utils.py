"""
Utility functions for TotalSegmentator dataset exploration and management.
"""

from pathlib import Path
from typing import List, Dict, Set, Tuple
import json
import nibabel as nib
import numpy as np
from tqdm import tqdm


def get_available_organs(
    root_dir: str,
    case_id: str = "s0000",
    check_non_zero: bool = True
) -> List[str]:
    """
    Get list of all available organ segmentations in a case.

    Args:
        root_dir: Root directory of TotalSegmentator dataset
        case_id: Case ID to check (default: s0000)
        check_non_zero: If True, only return organs with non-zero voxels

    Returns:
        List of organ names
    """
    root = Path(root_dir)
    case_folder = root / case_id
    seg_folder = case_folder / "segmentations"

    if not seg_folder.exists():
        raise ValueError(f"Segmentation folder not found: {seg_folder}")

    organ_files = sorted(seg_folder.glob("*.nii.gz"))
    organ_names = []

    for organ_file in organ_files:
        organ_name = organ_file.stem.replace(".nii", "")

        if check_non_zero:
            # Load and check if has non-zero voxels
            try:
                seg_data = nib.load(str(organ_file)).get_fdata()
                if np.any(seg_data > 0):
                    organ_names.append(organ_name)
            except Exception as e:
                print(f"Warning: Could not load {organ_file}: {e}")
        else:
            organ_names.append(organ_name)

    return organ_names


def scan_dataset(
    root_dir: str,
    max_cases: int = None,
    check_non_zero: bool = True,
    verbose: bool = True
) -> Dict[str, any]:
    """
    Scan the dataset and collect statistics, checking for non-zero segmentations.

    Args:
        root_dir: Root directory of TotalSegmentator dataset
        max_cases: Maximum number of cases to scan (None = all)
        check_non_zero: If True, count non-zero voxels for each organ
        verbose: If True, show progress

    Returns:
        Dictionary with dataset statistics including voxel counts
    """
    root = Path(root_dir)
    case_folders = sorted([f for f in root.iterdir() if f.is_dir() and f.name.startswith('s')])

    if max_cases is not None:
        case_folders = case_folders[:max_cases]

    stats = {
        "total_cases": len(case_folders),
        "valid_cases": 0,
        "missing_ct": 0,
        "missing_seg_folder": 0,
        "organs": {},  # organ_name -> {'count': int, 'total_voxels': int, 'cases': [case_ids]}
        "case_ids": [],
        "empty_segmentations": {},  # organ_name -> [case_ids with zero voxels]
    }

    if verbose:
        print(f"Scanning {len(case_folders)} cases...")
        iterator = tqdm(case_folders, desc="Scanning cases")
    else:
        iterator = case_folders

    for case_folder in iterator:
        case_id = case_folder.name

        # Check CT
        ct_path = case_folder / "ct.nii.gz"
        if not ct_path.exists():
            stats["missing_ct"] += 1
            continue

        # Check segmentation folder
        seg_folder = case_folder / "segmentations"
        if not seg_folder.exists():
            stats["missing_seg_folder"] += 1
            continue

        # Process each organ
        organ_files = list(seg_folder.glob("*.nii.gz"))
        case_has_valid_seg = False

        for organ_file in organ_files:
            organ_name = organ_file.stem.replace(".nii", "")

            # Initialize organ stats if needed
            if organ_name not in stats["organs"]:
                stats["organs"][organ_name] = {
                    'count': 0,
                    'total_voxels': 0,
                    'cases': []
                }
            if organ_name not in stats["empty_segmentations"]:
                stats["empty_segmentations"][organ_name] = []

            # Load and check segmentation
            try:
                seg_data = nib.load(str(organ_file)).get_fdata()
                non_zero_voxels = int(np.sum(seg_data > 0))

                if check_non_zero and non_zero_voxels > 0:
                    stats["organs"][organ_name]['count'] += 1
                    stats["organs"][organ_name]['total_voxels'] += non_zero_voxels
                    stats["organs"][organ_name]['cases'].append(case_id)
                    case_has_valid_seg = True
                elif check_non_zero and non_zero_voxels == 0:
                    # Track empty segmentations
                    stats["empty_segmentations"][organ_name].append(case_id)
                elif not check_non_zero:
                    # Count all files regardless of content
                    stats["organs"][organ_name]['count'] += 1
                    stats["organs"][organ_name]['total_voxels'] += non_zero_voxels
                    stats["organs"][organ_name]['cases'].append(case_id)
                    case_has_valid_seg = True

            except Exception as e:
                if verbose:
                    print(f"\nWarning: Could not load {organ_file}: {e}")

        if case_has_valid_seg:
            stats["valid_cases"] += 1
            stats["case_ids"].append(case_id)

    # Remove organs with zero occurrences
    stats["organs"] = {k: v for k, v in stats["organs"].items() if v['count'] > 0}

    # Remove empty segmentations entries for organs that don't exist
    stats["empty_segmentations"] = {
        k: v for k, v in stats["empty_segmentations"].items()
        if len(v) > 0
    }

    return stats


def print_dataset_stats(stats: Dict, show_empty: bool = False):
    """
    Print dataset statistics in a readable format.

    Args:
        stats: Statistics dictionary from scan_dataset
        show_empty: If True, also show empty segmentation statistics
    """
    print("\n" + "=" * 80)
    print("TotalSegmentator Dataset Statistics")
    print("=" * 80)

    print(f"\nTotal cases scanned: {stats['total_cases']}")
    print(f"Valid cases (with CT + non-empty segmentations): {stats['valid_cases']}")
    print(f"Cases missing CT: {stats['missing_ct']}")
    print(f"Cases missing segmentation folder: {stats['missing_seg_folder']}")

    print(f"\nTotal unique organs (with non-zero voxels): {len(stats['organs'])}")

    # Sort organs by frequency
    sorted_organs = sorted(
        stats['organs'].items(),
        key=lambda x: x[1]['count'],
        reverse=True
    )

    print("\nOrgans sorted by frequency:")
    print(f"{'Organ Name':<40} {'Cases':>8} {'%':>6} {'Avg Voxels':>15}")
    print("-" * 80)

    for organ_name, organ_stats in sorted_organs:
        count = organ_stats['count']
        percentage = (count / stats['valid_cases']) * 100
        avg_voxels = organ_stats['total_voxels'] / count if count > 0 else 0

        print(f"{organ_name:<40} {count:>8} {percentage:>5.1f}% {avg_voxels:>15,.0f}")

    if show_empty and stats['empty_segmentations']:
        print("\n" + "=" * 80)
        print("Empty Segmentations (files exist but all zeros)")
        print("=" * 80)

        sorted_empty = sorted(
            stats['empty_segmentations'].items(),
            key=lambda x: len(x[1]),
            reverse=True
        )

        print(f"\n{'Organ Name':<40} {'Empty Cases':>12}")
        print("-" * 80)

        for organ_name, empty_cases in sorted_empty:
            print(f"{organ_name:<40} {len(empty_cases):>12}")


def get_organs_by_min_cases(
    stats: Dict,
    min_cases: int = 100,
    min_avg_voxels: int = 0
) -> List[str]:
    """
    Get list of organs that appear in at least min_cases with minimum voxel count.

    Args:
        stats: Statistics dictionary from scan_dataset
        min_cases: Minimum number of cases required
        min_avg_voxels: Minimum average voxel count required

    Returns:
        List of organ names meeting criteria
    """
    valid_organs = []

    for organ_name, organ_stats in stats['organs'].items():
        count = organ_stats['count']
        avg_voxels = organ_stats['total_voxels'] / count if count > 0 else 0

        if count >= min_cases and avg_voxels >= min_avg_voxels:
            valid_organs.append(organ_name)

    return sorted(valid_organs)


def get_common_organ_groups() -> Dict[str, List[str]]:
    """
    Get predefined groups of commonly segmented organs.

    Returns:
        Dictionary mapping group names to lists of organ names
    """
    return {
        "Major Abdominal": [
            "liver",
            "spleen",
            "pancreas",
            "kidney_left",
            "kidney_right",
            "stomach",
            "gallbladder",
        ],
        "Cardiovascular": [
            "heart",
            "aorta",
            "pulmonary_artery",
            "inferior_vena_cava",
            "superior_vena_cava",
        ],
        "Urinary": [
            "kidney_left",
            "kidney_right",
            "urinary_bladder",
        ],
        "Digestive": [
            "liver",
            "stomach",
            "small_bowel",
            "colon",
            "esophagus",
            "duodenum",
        ],
        "Endocrine": [
            "thyroid_gland",
            "adrenal_gland_left",
            "adrenal_gland_right",
        ],
        "Thoracic": [
            "lung_upper_lobe_left",
            "lung_upper_lobe_right",
            "lung_middle_lobe_right",
            "lung_lower_lobe_left",
            "lung_lower_lobe_right",
            "trachea",
        ],
        "Skeletal_Spine": [
            "vertebrae_L1",
            "vertebrae_L2",
            "vertebrae_L3",
            "vertebrae_L4",
            "vertebrae_L5",
            "vertebrae_T12",
        ],
        "Brain": [
            "brain",
        ],
        "Muscles": [
            "autochthon_left",
            "autochthon_right",
            "gluteus_maximus_left",
            "gluteus_maximus_right",
            "iliopsoas_left",
            "iliopsoas_right",
        ],
    }


def filter_organs_by_availability(
    organ_list: List[str],
    stats: Dict,
    min_cases: int = 100,
    min_avg_voxels: int = 100
) -> Tuple[List[str], List[str]]:
    """
    Filter organ list to only include organs present in sufficient cases.

    Args:
        organ_list: List of organ names to filter
        stats: Statistics dictionary from scan_dataset
        min_cases: Minimum number of cases that must have the organ
        min_avg_voxels: Minimum average voxel count

    Returns:
        Tuple of (valid_organs, filtered_out_organs)
    """
    valid_organs = []
    filtered_out = []

    for organ in organ_list:
        if organ in stats["organs"]:
            organ_stats = stats["organs"][organ]
            count = organ_stats['count']
            avg_voxels = organ_stats['total_voxels'] / count if count > 0 else 0

            if count >= min_cases and avg_voxels >= min_avg_voxels:
                valid_organs.append(organ)
            else:
                filtered_out.append(organ)
        else:
            filtered_out.append(organ)

    return valid_organs, filtered_out


def create_train_val_split(
    root_dir: str,
    train_ratio: float = 0.8,
    output_dir: str = None,
    seed: int = 42,
    organ_list: List[str] = None,
    min_cases_per_organ: int = 1
) -> Dict[str, List[str]]:
    """
    Create train/validation split of case IDs, ensuring organs appear in both sets.

    Args:
        root_dir: Root directory of dataset
        train_ratio: Ratio of training data (0.0 to 1.0)
        output_dir: If provided, save split to JSON files
        seed: Random seed for reproducibility
        organ_list: If provided, only include cases with these organs
        min_cases_per_organ: Ensure each organ appears at least this many times in each split

    Returns:
        Dictionary with 'train' and 'val' lists of case IDs
    """
    import random

    print("Scanning dataset for split creation...")
    stats = scan_dataset(root_dir)
    case_ids = stats["case_ids"]

    # Filter by organ list if provided
    if organ_list is not None:
        filtered_case_ids = []
        for case_id in case_ids:
            case_organs = []
            for organ_name, organ_stats in stats["organs"].items():
                if case_id in organ_stats['cases']:
                    case_organs.append(organ_name)

            # Include case if it has all required organs
            if all(organ in case_organs for organ in organ_list):
                filtered_case_ids.append(case_id)

        case_ids = filtered_case_ids
        print(f"Filtered to {len(case_ids)} cases containing all required organs")

    random.seed(seed)
    random.shuffle(case_ids)

    split_idx = int(len(case_ids) * train_ratio)
    train_ids = case_ids[:split_idx]
    val_ids = case_ids[split_idx:]

    split = {
        "train": train_ids,
        "val": val_ids,
    }

    print(f"\nCreated split:")
    print(f"  Training: {len(train_ids)} cases")
    print(f"  Validation: {len(val_ids)} cases")

    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        with open(output_path / "train_split.json", "w") as f:
            json.dump(train_ids, f, indent=2)

        with open(output_path / "val_split.json", "w") as f:
            json.dump(val_ids, f, indent=2)

        print(f"\nSplit saved to: {output_path}")

    return split


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TotalSegmentator Dataset Utilities")
    parser.add_argument("--root_dir", type=str, required=True, help="Root directory of dataset")
    parser.add_argument("--action", type=str, required=True,
                        choices=["scan", "list_organs", "show_groups", "create_split", "filter_group"],
                        help="Action to perform")
    parser.add_argument("--max_cases", type=int, default=None, help="Max cases to scan")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for splits")
    parser.add_argument("--show_empty", action="store_true", help="Show empty segmentations")
    parser.add_argument("--min_cases", type=int, default=100, help="Minimum cases for filtering")
    parser.add_argument("--min_voxels", type=int, default=100, help="Minimum avg voxels for filtering")
    parser.add_argument("--group", type=str, default=None, help="Organ group name for filtering")

    args = parser.parse_args()

    if args.action == "scan":
        stats = scan_dataset(args.root_dir, max_cases=args.max_cases, check_non_zero=True)
        print_dataset_stats(stats, show_empty=args.show_empty)

        # Save to JSON
        output_path = Path(args.root_dir) / "dataset_stats.json"

        # Convert for JSON serialization (remove numpy types)
        stats_serializable = {
            k: (int(v) if isinstance(v, (np.integer, np.int64)) else v)
            for k, v in stats.items()
            if k != 'organs'  # Exclude detailed organ info for now
        }

        with open(output_path, "w") as f:
            json.dump(stats_serializable, f, indent=2)
        print(f"\nStatistics saved to: {output_path}")

        # Print recommended organs
        print("\n" + "=" * 80)
        print("Recommended organs (≥100 cases, ≥1000 avg voxels):")
        print("=" * 80)
        recommended = get_organs_by_min_cases(stats, min_cases=100, min_avg_voxels=1000)
        for organ in recommended:
            organ_stats = stats['organs'][organ]
            print(f"  {organ:<40} ({organ_stats['count']} cases, "
                  f"{organ_stats['total_voxels']/organ_stats['count']:,.0f} avg voxels)")

    elif args.action == "list_organs":
        try:
            organs = get_available_organs(args.root_dir, check_non_zero=True)
            print(f"\nAvailable organs with non-zero voxels ({len(organs)}):")
            for idx, organ in enumerate(organs, 1):
                print(f"{idx:3d}. {organ}")
        except Exception as e:
            print(f"Error: {e}")

    elif args.action == "show_groups":
        groups = get_common_organ_groups()
        print("\nPredefined Organ Groups:")
        print("=" * 80)
        for group_name, organs in groups.items():
            print(f"\n{group_name} ({len(organs)} organs):")
            for organ in organs:
                print(f"  - {organ}")

    elif args.action == "filter_group":
        if args.group is None:
            print("Error: --group required for filter_group action")
        else:
            stats = scan_dataset(args.root_dir, max_cases=args.max_cases)
            groups = get_common_organ_groups()

            if args.group not in groups:
                print(f"Error: Group '{args.group}' not found")
                print(f"Available groups: {', '.join(groups.keys())}")
            else:
                organ_list = groups[args.group]
                valid, filtered = filter_organs_by_availability(
                    organ_list, stats,
                    min_cases=args.min_cases,
                    min_avg_voxels=args.min_voxels
                )

                print(f"\nFiltering '{args.group}' group:")
                print(f"  Min cases: {args.min_cases}")
                print(f"  Min avg voxels: {args.min_voxels}")
                print(f"\n✓ Valid organs ({len(valid)}):")
                for organ in valid:
                    organ_stats = stats['organs'][organ]
                    print(f"    {organ:<30} ({organ_stats['count']} cases, "
                          f"{organ_stats['total_voxels']/organ_stats['count']:,.0f} avg voxels)")

                if filtered:
                    print(f"\n✗ Filtered out ({len(filtered)}):")
                    for organ in filtered:
                        if organ in stats['organs']:
                            organ_stats = stats['organs'][organ]
                            print(f"    {organ:<30} ({organ_stats['count']} cases, "
                                  f"{organ_stats['total_voxels']/organ_stats['count']:,.0f} avg voxels)")
                        else:
                            print(f"    {organ:<30} (not found in dataset)")

    elif args.action == "create_split":
        split = create_train_val_split(
            args.root_dir,
            train_ratio=0.8,
            output_dir=args.output_dir or args.root_dir,
        )
