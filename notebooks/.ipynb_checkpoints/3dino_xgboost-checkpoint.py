"""
Fixed XGBoost approach for MHSA-based segmentation
Key fixes:
1. Per-case feature normalization
2. Training on multiple cases
3. Proper train/test split (different cases)
"""

import numpy as np
import nibabel as nib
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.metrics import balanced_accuracy_score
import subprocess
import os
import sys


def generate_mhsa_features(
    image_path,
    output_dir,
    config_file='dinov2/configs/train/vit3d_highres.yaml',
    pretrained_weights='/home/dpxuser/.cache/huggingface/hub/models--AICONSlab--3DINO-ViT/snapshots/8a00a2bb14becb3fbe955064837322a3217f7ac0/3dino_vit_weights.pth',
    vis_type='mhsa',
    input_type='sliding_window',
    overlap=0.75,
    interpolation='trilinear',
    repo_path='../../repos/3DINO'
):
    """
    Generate MHSA features for an image using the vis_pca.py script

    Args:
        image_path: Path to input NIfTI image (.nii.gz)
        output_dir: Directory where MHSA features will be saved
        config_file: Path to model config file (relative to repo_path)
        pretrained_weights: Path to pretrained model weights
        vis_type: Type of visualization ('mhsa' or 'pca')
        input_type: Input processing type ('sliding_window', 'full_image', or 'resize')
        overlap: Overlap ratio for sliding window (0.0 to 0.9)
        interpolation: Interpolation mode ('nearest' or 'trilinear')
        repo_path: Path to 3DINO repository

    Returns:
        output_dir: Path to directory containing generated features
    """

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Build the command
    cmd = [
        sys.executable,  # Use current Python interpreter
        'dinov2/eval/vis_pca.py',
        '--config-file', config_file,
        '--output-dir', output_dir,
        '--pretrained-weights', pretrained_weights,
        '--image-path', image_path,
        '--vis-type', vis_type,
        '--input-type', input_type,
        '--overlap', str(overlap),
        '--interpolation', interpolation,
    ]

    # Set environment and working directory
    env = os.environ.copy()
    env['PYTHONPATH'] = '.'

    print(f"Generating {vis_type} features for: {image_path}")
    print(f"Output directory: {output_dir}")
    print(f"Command: PYTHONPATH=. {' '.join(cmd)}")

    # Run the command
    try:
        result = subprocess.run(
            cmd,
            env=env,
            cwd=repo_path,
            check=True,
            capture_output=True,
            text=True
        )
        print("Feature generation completed successfully!")
        print(result.stdout)
        return output_dir
    except subprocess.CalledProcessError as e:
        print(f"Error generating features: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        raise


def generate_features_for_cases(
    case_names,
    results_dir,
    repo_path='../../repos/3DINO',
    overlap=0.75,
    interpolation='trilinear',
    **kwargs
):
    """
    Generate MHSA features for multiple cases

    Args:
        case_names: List of case names (e.g., ['MR_BraTS-T2f_bratsgli_0006', ...])
        results_dir: Base results directory containing _img.nii.gz files
        repo_path: Path to 3DINO repository
        overlap: Overlap ratio for sliding window
        interpolation: Interpolation mode
        **kwargs: Additional arguments to pass to generate_mhsa_features

    Returns:
        dict: Mapping of case_name -> output_dir
    """

    output_dirs = {}

    for case_name in case_names:
        print(f"\n{'='*60}")
        print(f"Processing case: {case_name}")
        print(f"{'='*60}")

        # Paths
        image_path = os.path.join(results_dir, f"{case_name}_img.nii.gz")
        output_dir = os.path.join(results_dir, f"3DINO_features/{case_name}/mhsa")

        # Check if image exists
        if not os.path.exists(image_path):
            print(f"WARNING: Image not found: {image_path}")
            print(f"Skipping case {case_name}")
            continue

        # Check if features already exist
        if os.path.exists(output_dir) and len([f for f in os.listdir(output_dir) if f.startswith('nifti_')]) > 0:
            print(f"Features already exist in {output_dir}")
            print("Skipping feature generation. Delete directory to regenerate.")
            output_dirs[case_name] = output_dir
            continue

        # Generate features
        try:
            output_dirs[case_name] = generate_mhsa_features(
                image_path=image_path,
                output_dir=output_dir,
                repo_path=repo_path,
                overlap=overlap,
                interpolation=interpolation,
                **kwargs
            )
        except Exception as e:
            print(f"Failed to generate features for {case_name}: {e}")
            continue

    print(f"\n{'='*60}")
    print(f"Feature generation complete for {len(output_dirs)}/{len(case_names)} cases")
    print(f"{'='*60}")

    return output_dirs


def load_mhsa_features(case_name, results_dir):
    """Load all MHSA head features for a case"""
    import os
    mhsa_files = sorted([
        os.path.join(results_dir, f"3DINO_features/{case_name}/mhsa", f)
        for f in os.listdir(os.path.join(results_dir, f"3DINO_features/{case_name}/mhsa"))
        if f.startswith("nifti_") and f.endswith(".nii.gz")
    ])

    mhsa_vols = []
    for f in mhsa_files:
        vol = nib.load(f).get_fdata()
        mhsa_vols.append(vol)

    return np.stack(mhsa_vols, axis=3)  # H, W, D, num_heads


def prepare_case_features(mhsa_vols, gt, normalize=True):
    """
    Prepare features from a single case with proper normalization

    Args:
        mhsa_vols: MHSA features (H, W, D, num_heads)
        gt: Ground truth (H, W, D)
        normalize: Whether to apply per-case standardization

    Returns:
        X: Features (num_voxels, num_heads)
        y: Labels (num_voxels,)
        scaler: Fitted scaler (or None)
    """
    X = mhsa_vols.reshape(-1, mhsa_vols.shape[3])
    y = gt.flatten()

    scaler = None
    if normalize:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    return X, y, scaler


# ============================================================
# CORRECT APPROACH: Train on multiple cases, test on others
# ============================================================

def train_multi_case(train_cases, results_dir):
    """Train XGBoost on multiple cases with proper normalization"""

    all_X = []
    all_y = []

    for case_name in train_cases:
        print(f"Loading training case: {case_name}")

        # Load MHSA features
        mhsa_vols = load_mhsa_features(case_name, results_dir)

        # Load ground truth
        gt = nib.load(f"{results_dir}/{case_name}_gt.nii.gz").get_fdata()

        # Prepare features with per-case normalization
        X_case, y_case, _ = prepare_case_features(mhsa_vols, gt, normalize=True)

        all_X.append(X_case)
        all_y.append(y_case)

    # Combine all training data
    X_train = np.vstack(all_X)
    y_train = np.concatenate(all_y)

    # Calculate class weights
    unique, counts = np.unique(y_train, return_counts=True)
    total = len(y_train)
    class_weights = {cls: total / (len(unique) * count) for cls, count in zip(unique, counts)}
    sample_weights = np.array([class_weights[label] for label in y_train])

    # Train XGBoost
    print(f"Training XGBoost on {len(train_cases)} cases...")
    model = xgb.XGBClassifier(
        objective='multi:softmax',
        num_class=len(unique),
        max_depth=6,
        learning_rate=0.1,
        n_estimators=100
    )
    model.fit(X_train, y_train, sample_weight=sample_weights)

    return model


def evaluate_case(model, test_case, results_dir):
    """Evaluate model on a test case with proper normalization"""

    print(f"Evaluating on test case: {test_case}")

    # Load MHSA features
    mhsa_vols = load_mhsa_features(test_case, results_dir)

    # Load ground truth
    gt = nib.load(f"{results_dir}/{test_case}_gt.nii.gz").get_fdata()

    # Prepare features with per-case normalization
    X_test, y_test, _ = prepare_case_features(mhsa_vols, gt, normalize=True)

    # Predict
    y_pred = model.predict(X_test)

    # Evaluate
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    accuracy = np.mean(y_pred == y_test)

    return balanced_acc, accuracy


# ============================================================
# USAGE EXAMPLE
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate MHSA features and train XGBoost segmentation model")
    parser.add_argument("--mode", type=str, choices=["generate", "train", "both"], default="both",
                        help="Mode: 'generate' features only, 'train' model only, or 'both'")
    parser.add_argument("--results-dir", type=str,
                        default="/nfs/data/nii/data1/Analysis/camaret___in_context_segmentation/ANALYSIS_20251122/results",
                        help="Base results directory")
    parser.add_argument("--repo-path", type=str, default="../../repos/3DINO",
                        help="Path to 3DINO repository")
    parser.add_argument("--overlap", type=float, default=0.75,
                        help="Overlap ratio for sliding window (0.0-0.9)")
    parser.add_argument("--interpolation", type=str, default="trilinear",
                        choices=["nearest", "trilinear"],
                        help="Interpolation mode for upsampling")

    args = parser.parse_args()

    results_dir = args.results_dir
    repo_path = args.repo_path

    # Define training and test cases
    train_cases = ['MR_BraTS-T2f_bratsgli_0703', 
                   'MR_BraTS-T2w_bratsgli_0321', 
                   'MR_BraTS-T2w_bratsgli_0771', 
                   'MR_BraTS-T2f_bratsgli_0070', 
                   'MR_BraTS-T2f_bratsgli_0546', 
                   'MR_BraTS-T2f_bratsgli_1033', 
                   'MR_BraTS-T2f_bratsgli_1262', 
                   'MR_BraTS-T2f_bratsgli_0107', 

        # Add more cases...
    ]

    test_cases = [
                   'MR_BraTS-T2f_bratsgli_1096', 
                   'MR_BraTS-T2f_bratsgli_1172'
        # Add more test cases...
    ]

    # Step 1: Generate MHSA features
    if args.mode in ["generate", "both"]:
        print("\n" + "="*60)
        print("STEP 1: Generating MHSA features")
        print("="*60)

        all_cases = train_cases + test_cases
        generate_features_for_cases(
            case_names=all_cases,
            results_dir=results_dir,
            repo_path=repo_path,
            overlap=args.overlap,
            interpolation=args.interpolation
        )

    # Step 2: Train and evaluate model
    if args.mode in ["train", "both"]:
        print("\n" + "="*60)
        print("STEP 2: Training XGBoost model")
        print("="*60)

        # Train on multiple cases
        model = train_multi_case(train_cases, results_dir)

        # Evaluate on test cases
        print("\n" + "="*60)
        print("STEP 3: Evaluating on test cases")
        print("="*60)

        for test_case in test_cases:
            balanced_acc, accuracy = evaluate_case(model, test_case, results_dir)
            print(f"{test_case}: Balanced Acc = {balanced_acc:.3f}, Accuracy = {accuracy:.3f}")


# ============================================================
# QUICK FIX: If you only have one training case
# ============================================================

def quick_fix_single_case(train_case, test_case, results_dir):
    """
    Quick fix when you only have one training case.
    This will still perform poorly, but better than 0.2.
    """

    # Training
    print(f"Training on single case: {train_case}")
    mhsa_train = load_mhsa_features(train_case, results_dir)
    gt_train = nib.load(f"{results_dir}/{train_case}_gt.nii.gz").get_fdata()
    X_train, y_train, _ = prepare_case_features(mhsa_train, gt_train, normalize=True)

    # Calculate class weights
    unique, counts = np.unique(y_train, return_counts=True)
    total = len(y_train)
    class_weights = {cls: total / (len(unique) * count) for cls, count in zip(unique, counts)}
    sample_weights = np.array([class_weights[label] for label in y_train])

    # Train
    model = xgb.XGBClassifier(objective='multi:softmax', num_class=len(unique))
    model.fit(X_train, y_train, sample_weight=sample_weights)

    # Testing with normalization
    print(f"Testing on: {test_case}")
    mhsa_test = load_mhsa_features(test_case, results_dir)
    gt_test = nib.load(f"{results_dir}/{test_case}_gt.nii.gz").get_fdata()
    X_test, y_test, _ = prepare_case_features(mhsa_test, gt_test, normalize=True)

    # Predict
    y_pred = model.predict(X_test)

    # Evaluate
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    accuracy = np.mean(y_pred == y_test)

    print(f"Balanced Accuracy: {balanced_acc:.3f}")
    print(f"Accuracy: {accuracy:.3f}")

    return model, balanced_acc, accuracy


# ============================================================
# USAGE EXAMPLES
# ============================================================

"""
# Example 1: Generate features for a single case (from command line)
python fixed_xgboost_approach.py --mode generate --overlap 0.75 --interpolation trilinear

# Example 2: Train model only (assuming features already generated)
python fixed_xgboost_approach.py --mode train

# Example 3: Generate features and train model
python fixed_xgboost_approach.py --mode both --overlap 0.75

# Example 4: Use in a Jupyter notebook
from fixed_xgboost_approach import *

# Generate MHSA features for a single case
generate_mhsa_features(
    image_path='/path/to/MR_BraTS-T2f_bratsgli_0006_img.nii.gz',
    output_dir='/path/to/output/3DINO_features/MR_BraTS-T2f_bratsgli_0006/mhsa',
    overlap=0.75,
    interpolation='trilinear',
    repo_path='../../repos/3DINO'
)

# Generate features for multiple cases
results_dir = "/nfs/data/nii/data1/Analysis/camaret___in_context_segmentation/ANALYSIS_20251122/results"
cases = ["MR_BraTS-T2f_bratsgli_0006", "MR_BraTS-T2f_bratsgli_0007", "MR_BraTS-T2f_bratsgli_0009"]

generate_features_for_cases(
    case_names=cases,
    results_dir=results_dir,
    repo_path='../../repos/3DINO',
    overlap=0.75,
    interpolation='trilinear'
)

# Train XGBoost model
train_cases = ["MR_BraTS-T2f_bratsgli_0006", "MR_BraTS-T2f_bratsgli_0007"]
model = train_multi_case(train_cases, results_dir)

# Evaluate on test case
test_case = "MR_BraTS-T2f_bratsgli_0009"
balanced_acc, accuracy = evaluate_case(model, test_case, results_dir)
print(f"Balanced Accuracy: {balanced_acc:.3f}")

# Quick fix if you only have one training case
model, balanced_acc, accuracy = quick_fix_single_case(
    train_case="MR_BraTS-T2f_bratsgli_0006",
    test_case="MR_BraTS-T2f_bratsgli_0009",
    results_dir=results_dir
)
"""
