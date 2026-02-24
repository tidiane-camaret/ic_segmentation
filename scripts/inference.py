"""
Simple inference script for PatchICL segmentation.

Usage:
    python scripts/inference.py \
        --checkpoint path/to/checkpoint.pt \
        --target_image path/to/target.nii.gz \
        --context_images path/to/ctx1.nii.gz path/to/ctx2.nii.gz \
        --context_masks path/to/ctx1_mask.nii.gz path/to/ctx2_mask.nii.gz \
        --output path/to/output_mask.nii.gz

    # With config file override:
    python scripts/inference.py \
        --checkpoint path/to/checkpoint.pt \
        --config configs/experiment/83_unified_augmentation.yaml \
        ...
"""
import argparse
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.losses import build_loss_fn


# ============================================================================
# Image Loading and Preprocessing
# ============================================================================

def load_image(filepath: str) -> tuple:
    """Load an image (NIfTI or numpy) and return as numpy array."""
    path = Path(filepath)

    if path.suffix == '.npy':
        img = np.load(path).astype(np.float32)
        affine = np.eye(4)
        return img, affine
    else:
        nii = nib.load(str(path))
        return nii.get_fdata().astype(np.float32), nii.affine


def save_image(data: np.ndarray, affine: np.ndarray, path: str):
    """Save numpy array as NIfTI image."""
    nii = nib.Nifti1Image(data, affine)
    nib.save(nii, path)


def normalize_image(img: np.ndarray, a_min: float = -200, a_max: float = 300) -> np.ndarray:
    """Normalize CT image to [0, 1] range."""
    img = np.clip(img, a_min, a_max)
    img = (img - a_min) / (a_max - a_min)
    return img


def preprocess_2d(
    img: np.ndarray,
    mask: np.ndarray = None,
    image_size: tuple = (128, 128),
) -> tuple:
    """Preprocess 2D image and optional mask."""
    if img.ndim > 2:
        img = img.squeeze()

    img = normalize_image(img)
    img_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float()
    img_tensor = F.interpolate(img_tensor, size=image_size, mode='bilinear', align_corners=False)

    if mask is not None:
        if mask.ndim > 2:
            mask = mask.squeeze()
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float()
        mask_tensor = F.interpolate(mask_tensor, size=image_size, mode='nearest')
        return img_tensor, mask_tensor

    return img_tensor, None


# ============================================================================
# Feature Extractor Creation (from eval.py pattern)
# ============================================================================

def create_feature_extractor(config: dict, device: str = "cuda"):
    """Create feature extractor from config (matches eval.py logic)."""
    fe_cfg = config.get('model', {}).get('patch_icl', {}).get('feature_extractor', {})
    extractor_type = fe_cfg.get('type', config.get('feature_extractor_type', 'icl_encoder')).lower()

    if extractor_type == 'icl_encoder':
        from src.models.icl_encoder import ICLEncoder
        return ICLEncoder(
            layer_idx=fe_cfg.get('layer_idx', 'all'),
            output_grid_size=fe_cfg.get('output_grid_size', 64),
            freeze=fe_cfg.get('freeze', False),
        )

    elif extractor_type in ['meddino', 'meddinov3', 'meddino_v3']:
        from src.models.meddino_extractor import create_meddino_extractor
        return create_meddino_extractor(
            model_path=fe_cfg.get('model_path'),
            target_size=fe_cfg.get('target_size', 256),
            device=device,
            layer_idx=fe_cfg.get('layer_idx', 11),
            freeze=fe_cfg.get('freeze', True),
        )

    elif extractor_type in ['medsam_v1', 'medsam_v1_layer']:
        from src.models.medsam_extractor import MedSAMv1LayerExtractor
        return MedSAMv1LayerExtractor(
            checkpoint_path=fe_cfg.get('checkpoint_path'),
            target_size=fe_cfg.get('target_size', 1024),
            device=device,
            layer_idx=fe_cfg.get('layer_idx', 11),
            freeze=fe_cfg.get('freeze', True),
            output_grid_size=fe_cfg.get('output_grid_size'),
        )

    elif extractor_type == 'universeg':
        from src.models.universeg_extractor import UniverSegExtractor
        return UniverSegExtractor(
            layer_idx=fe_cfg.get('layer_idx', 3),
            device=device,
            pretrained=fe_cfg.get('pretrained', True),
            freeze=fe_cfg.get('freeze', True),
            output_grid_size=fe_cfg.get('output_grid_size'),
            input_size=fe_cfg.get('input_size', 128),
        )

    else:
        raise ValueError(f"Unknown feature_extractor_type: {extractor_type}")


def load_config(config_path: str) -> dict:
    """Load config from yaml file using OmegaConf."""
    cfg = OmegaConf.load(config_path)
    return OmegaConf.to_container(cfg, resolve=True)


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep merge two dicts, with override taking precedence."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_model(checkpoint_path: str, device: torch.device, config_override: dict = None) -> tuple:
    """Load PatchICL model from checkpoint with feature extractor.

    Args:
        checkpoint_path: Path to model checkpoint
        device: torch device
        config_override: Optional config dict to override checkpoint config
    """
    from src.models.patch_icl_v2.patch_icl import PatchICL

    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get('config', {})

    # Merge with config override if provided
    if config_override:
        # Deep merge: override takes precedence
        config = _deep_merge(config, config_override)
        print("Applied config override from file")

    # Get model config
    patch_icl_cfg = config.get('model', {}).get('patch_icl', {})
    context_size = config.get('context_size', 1)

    # Set num_mask_channels
    random_coloring_nb = config.get('random_coloring_nb', 0)
    patch_icl_cfg['num_mask_channels'] = 3 if random_coloring_nb > 0 else 1

    # Create feature extractor if on_the_fly mode
    feature_extractor = None
    feature_mode = config.get('feature_mode', 'precomputed')
    if feature_mode == 'on_the_fly':
        print("Creating feature extractor for on-the-fly mode...")
        feature_extractor = create_feature_extractor(config, str(device))
        if hasattr(feature_extractor, 'get_feature_info'):
            info = feature_extractor.get_feature_info()
            print(f"  Feature extractor: {info.get('extractor', 'unknown')}, dim={info.get('feature_dim')}")

    # Build model
    model = PatchICL(patch_icl_cfg, context_size=context_size, feature_extractor=feature_extractor)

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)

    # Set loss functions (required for forward pass)
    loss_cfg = patch_icl_cfg.get('loss', {})
    patch_loss_cfg = loss_cfg.get('patch_loss', {'type': 'dice', 'args': None})
    aggreg_loss_cfg = loss_cfg.get('aggreg_loss', {'type': 'dice', 'args': None})
    patch_criterion = build_loss_fn(patch_loss_cfg['type'], patch_loss_cfg.get('args'))
    aggreg_criterion = build_loss_fn(aggreg_loss_cfg['type'], aggreg_loss_cfg.get('args'))
    model.set_loss_functions(patch_criterion, aggreg_criterion)

    model.to(device)
    model.eval()

    print(f"Loaded model from {checkpoint_path}")
    print(f"  Epoch: {checkpoint.get('epoch', '?')}, Best Dice: {checkpoint.get('best_dice', '?'):.4f}")

    return model, config


@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    target_image: torch.Tensor,
    context_images: torch.Tensor,
    context_masks: torch.Tensor,
    device: torch.device,
    prior_mask: torch.Tensor = None,
) -> torch.Tensor:
    """Run inference on a single target image with context examples."""
    target_image = target_image.to(device)
    context_images = context_images.to(device)
    context_masks = context_masks.to(device)

    if prior_mask is not None:
        prior_mask = prior_mask.to(device)

    # Forward pass (feature extraction happens inside model if on_the_fly)
    outputs = model(
        image=target_image,
        labels=prior_mask,
        context_in=context_images,
        context_out=context_masks,
        mode="test",
    )

    pred = outputs['final_pred']
    pred = torch.sigmoid(pred)

    return pred.squeeze().cpu()


def main():
    parser = argparse.ArgumentParser(description="PatchICL Inference Script")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default=None, help="Path to config file (overrides checkpoint config)")
    parser.add_argument("--target_image", type=str, required=True, help="Path to target image (2D NIfTI or .npy)")
    parser.add_argument("--target_mask", type=str, default=None, help="Path to target GT mask (for oracle sampling)")
    parser.add_argument("--context_images", type=str, nargs="+", required=True, help="Paths to context images")
    parser.add_argument("--context_masks", type=str, nargs="+", required=True, help="Paths to context masks")
    parser.add_argument("--output", type=str, required=True, help="Path to save output mask")
    parser.add_argument("--image_size", type=int, nargs=2, default=None, help="Image size (default: from config or 128x128)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for binary mask")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    if len(args.context_images) != len(args.context_masks):
        raise ValueError("Number of context images must match number of context masks")

    device = torch.device(args.device)

    # Load config override if provided
    config_override = None
    if args.config:
        print(f"Loading config from {args.config}")
        config_override = load_config(args.config)

    # Load model (includes feature extractor if on_the_fly)
    model, config = load_model(args.checkpoint, device, config_override=config_override)

    # Determine image size: CLI arg > config > default
    if args.image_size:
        image_size = tuple(args.image_size)
    else:
        cfg_size = config.get('preprocessing', {}).get('image_size', [128, 128])
        image_size = tuple(cfg_size[:2])  # Take first 2 dims (H, W)
    print(f"Using image size: {image_size}")

    # Load and preprocess target image
    target_np, target_affine = load_image(args.target_image)
    original_shape = target_np.shape[:2]
    target_tensor, _ = preprocess_2d(target_np, image_size=image_size)

    # Load target mask if provided (for oracle sampling)
    target_mask_tensor = None
    if args.target_mask:
        target_mask_np, _ = load_image(args.target_mask)
        _, target_mask_tensor = preprocess_2d(target_np, target_mask_np, image_size=image_size)
        print(f"Loaded target mask for oracle sampling: {args.target_mask}")

    # Load and preprocess context examples
    context_imgs = []
    context_masks = []

    for ctx_img_path, ctx_mask_path in zip(args.context_images, args.context_masks):
        ctx_img_np, _ = load_image(ctx_img_path)
        ctx_mask_np, _ = load_image(ctx_mask_path)

        ctx_img_tensor, ctx_mask_tensor = preprocess_2d(ctx_img_np, ctx_mask_np, image_size=image_size)
        context_imgs.append(ctx_img_tensor)
        context_masks.append(ctx_mask_tensor)

    # Stack context: [1, K, 1, H, W]
    context_imgs = torch.cat(context_imgs, dim=0).unsqueeze(0)
    context_masks = torch.cat(context_masks, dim=0).unsqueeze(0)

    print(f"Target shape: {target_tensor.shape}")
    print(f"Context images shape: {context_imgs.shape}")
    print(f"Context masks shape: {context_masks.shape}")

    # Run inference
    pred_probs = run_inference(
        model, target_tensor, context_imgs, context_masks, device,
        prior_mask=target_mask_tensor,
    )

    # Resize prediction back to original size
    pred_probs_resized = F.interpolate(
        pred_probs.unsqueeze(0).unsqueeze(0),
        size=original_shape,
        mode='bilinear',
        align_corners=False
    ).squeeze()

    # Apply threshold
    pred_mask = (pred_probs_resized > args.threshold).float().numpy()

    # Save output
    save_image(pred_mask, target_affine, args.output)
    print(f"Saved prediction to {args.output}")
    print(f"  Prediction stats: min={pred_probs.min():.3f}, max={pred_probs.max():.3f}, mean={pred_probs.mean():.3f}")
    print(f"  Mask coverage: {pred_mask.mean() * 100:.2f}%")


if __name__ == "__main__":
    main()
