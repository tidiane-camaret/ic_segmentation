"""
Simple inference script for PatchICL segmentation.

Usage:
    python scripts/inference.py \
        --checkpoint path/to/checkpoint.pt \
        --target_image path/to/target.nii.gz \
        --context_images path/to/ctx1.nii.gz path/to/ctx2.nii.gz \
        --context_masks path/to/ctx1_mask.nii.gz path/to/ctx2_mask.nii.gz \
        --output path/to/output_mask.nii.gz \
        --meddino_path path/to/med_dino_v3model.pth
"""
import argparse
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ============================================================================
# MedDINOv3 Feature Extractor (adapted from extract_meddinov3_features.py)
# ============================================================================

class MedDINOProcessor:
    """Preprocessor for MedDINOv3 feature extraction."""
    
    def __init__(self, target_size: int = 896, interpolation: str = "bilinear"):
        self.target_size = target_size
        self.interpolation = interpolation
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __call__(self, images: list) -> torch.Tensor:
        if isinstance(images, np.ndarray):
            images = [images]

        processed_batch = []
        for img in images:
            # 1. Percentile Clipping (0.5% - 99.5%)
            lower, upper = np.percentile(img, [0.5, 99.5])
            img = np.clip(img, lower, upper)

            # 2. Rescale to [0, 1]
            img_min, img_max = img.min(), img.max()
            if img_max > img_min:
                img = (img - img_min) / (img_max - img_min)
            else:
                img = np.zeros_like(img)

            # 3. Grayscale to RGB & ImageNet Normalization
            img_rgb = np.stack([img] * 3, axis=-1)
            img_rgb = (img_rgb - self.mean) / self.std

            # 4. To Tensor & Resize
            img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float()
            img_tensor = F.interpolate(
                img_tensor.unsqueeze(0),
                size=(self.target_size, self.target_size),
                mode=self.interpolation,
                align_corners=False
            ).squeeze(0)
            
            processed_batch.append(img_tensor)

        return torch.stack(processed_batch)


class MedDINOFeatureExtractor:
    """MedDINOv3 feature extractor for inference."""
    
    def __init__(self, model_path: str, target_size: int = 896, device: str = "cuda"):
        self.device = device
        self.processor = MedDINOProcessor(target_size=target_size)
        
        # Import MedDINOv3 model
        meddino_path = Path("/software/notebooks/camaret/repos/MedDINOv3/nnUNet/nnunetv2/training/nnUNetTrainer/dinov3/")
        if meddino_path.exists():
            sys.path.insert(0, str(meddino_path))
        from dinov3.models.vision_transformer import vit_base

        # Initialize architecture
        self.model = vit_base(
            drop_path_rate=0.2, 
            layerscale_init=1.0e-05, 
            n_storage_tokens=4, 
            qkv_bias=False, 
            mask_k_bias=True
        )

        # Load MedDINOv3 weights
        print(f"Loading MedDINOv3 weights from {model_path}...")
        chkpt = torch.load(model_path, weights_only=False, map_location='cpu')
        state_dict = chkpt['teacher']
        state_dict = {
            k.replace('backbone.', ''): v
            for k, v in state_dict.items()
            if 'ibot' not in k and 'dino_head' not in k
        }
        self.model.load_state_dict(state_dict)
        self.model.to(device).eval()
        print("MedDINOv3 model loaded successfully")

    @torch.no_grad()
    def extract(self, imgs: list) -> torch.Tensor:
        """
        Extract features from images.
        
        Args:
            imgs: List of numpy arrays [H, W] (grayscale images)
            
        Returns:
            features: [B, N, D] full sequence (CLS + Registers + Patches)
                      to match precomputed features format from dataloader
                      Shape: [B, 256, 768] for 256x256 input with vit_base
        """
        pixel_values = self.processor(imgs).to(self.device)
        
        # Get final layer features (layer 11) - same as extract_meddinov3_features.py
        intermediate_layers = self.model.get_intermediate_layers(
            pixel_values, 
            n=[11], 
            reshape=False
        )
        
        feat = intermediate_layers[0]  # [B, Tokens, Dim]
        # Token decomposition: CLS (0), Registers (1:5), Patches (5:)
        cls_tokens = feat[:, 0:1, :]      # [B, 1, D]
        reg_tokens = feat[:, 1:5, :]      # [B, 4, D]
        patch_tokens = feat[:, 5:, :]     # [B, N, D] where N=251 for 256x256 input
        
        # Concatenate in same order as dataloader: CLS + Registers + Patches
        # This matches _load_features in totalseg2d_dataloader.py
        full_sequence = torch.cat([cls_tokens, reg_tokens, patch_tokens], dim=1)
        
        return full_sequence


# ============================================================================
# Image Loading and Preprocessing
# ============================================================================

def load_image(path: str) -> tuple:
    """Load an image (NIfTI or numpy) and return as numpy array.
    
    Supports:
    - .nii.gz / .nii files (NIfTI format)
    - .npy files (numpy format, same as used in eval)
    """
    path = Path(path)
    
    if path.suffix == '.npy' or str(path).endswith('.npy'):
        # Load numpy array directly (same format as eval dataloader uses)
        img = np.load(path).astype(np.float32)
        affine = np.eye(4)  # Identity affine for numpy files
        return img, affine
    else:
        # Load NIfTI format
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
    image_size: tuple = (512, 512),
) -> tuple:
    """Preprocess 2D image and optional mask."""
    # Normalize image
    img = normalize_image(img)
    
    # Convert to tensor and resize
    img_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float()  # [1, 1, H, W]
    img_tensor = F.interpolate(img_tensor, size=image_size, mode='bilinear', align_corners=False)
    
    if mask is not None:
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float()  # [1, 1, H, W]
        mask_tensor = F.interpolate(mask_tensor, size=image_size, mode='nearest')
        return img_tensor, mask_tensor
    
    return img_tensor, None


def load_model(checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    """Load PatchICL model from checkpoint."""
    from src.models.patch_icl import PatchICL
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get('config', {})
    
    # Get model config
    patch_icl_cfg = config.get('model', {}).get('patch_icl', {})
    context_size = config.get('context_size', 1)
    
    # Set num_mask_channels
    random_coloring_nb = config.get('random_coloring_nb', 0)
    patch_icl_cfg['num_mask_channels'] = 3 if random_coloring_nb > 0 else 1
    
    # Build model
    model = PatchICL(patch_icl_cfg, context_size=context_size)
    model.load_state_dict(checkpoint['model_state_dict'])
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
    target_features: torch.Tensor = None,
    context_features: torch.Tensor = None,
) -> torch.Tensor:
    """
    Run inference on a single target image with context examples.
    
    Args:
        model: PatchICL model
        target_image: [1, 1, H, W] target image tensor
        context_images: [1, K, 1, H, W] context images
        context_masks: [1, K, 1, H, W] context masks
        device: torch device
        target_features: [1, N, D] precomputed target features (optional)
        context_features: [1, K, N, D] precomputed context features (optional)
        
    Returns:
        pred_mask: [H, W] predicted mask (probabilities)
    """
    target_image = target_image.to(device)
    context_images = context_images.to(device)
    context_masks = context_masks.to(device)
    
    if target_features is not None:
        target_features = target_features.to(device)
    if context_features is not None:
        context_features = context_features.to(device)
    
    # Forward pass
    outputs = model(
        image=target_image,
        labels=None,
        context_in=context_images,
        context_out=context_masks,
        target_features=target_features,
        context_features=context_features,
        mode="eval",
    )
    
    # Get prediction
    pred = outputs['final_pred']  # [1, 1, H, W]
    pred = torch.sigmoid(pred)  # Convert logits to probabilities
    
    return pred.squeeze().cpu()


def main():
    parser = argparse.ArgumentParser(description="PatchICL Inference Script")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--target_image", type=str, required=True, help="Path to target image (2D NIfTI)")
    parser.add_argument("--context_images", type=str, nargs="+", required=True, help="Paths to context images")
    parser.add_argument("--context_masks", type=str, nargs="+", required=True, help="Paths to context masks")
    parser.add_argument("--output", type=str, required=True, help="Path to save output mask")
    parser.add_argument("--meddino_path", type=str, required=True, help="Path to MedDINOv3 model weights")
    parser.add_argument("--image_size", type=int, nargs=2, default=[512, 512], help="Image size for inference")
    parser.add_argument("--feature_size", type=int, default=256, help="Image size for MedDINOv3 feature extraction (must match training config)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for binary mask")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    
    # Validate inputs
    if len(args.context_images) != len(args.context_masks):
        raise ValueError("Number of context images must match number of context masks")
    
    device = torch.device(args.device)
    image_size = tuple(args.image_size)
    
    # Load model
    model, config = load_model(args.checkpoint, device)
    
    # Load MedDINOv3 feature extractor
    print("Initializing MedDINOv3 feature extractor...")
    feature_extractor = MedDINOFeatureExtractor(
        model_path=args.meddino_path,
        target_size=args.feature_size,
        device=args.device,
    )
    
    # Load and preprocess target image
    target_np, target_affine = load_image(args.target_image)
    original_shape = target_np.shape[:2]
    target_tensor, _ = preprocess_2d(target_np, image_size=image_size)
    
    # Extract target features
    print("Extracting target features...")
    target_features = feature_extractor.extract([target_np])  # [1, N, D]
    
    # Load and preprocess context examples
    context_imgs = []
    context_masks = []
    context_np_list = []
    
    for ctx_img_path, ctx_mask_path in zip(args.context_images, args.context_masks):
        ctx_img_np, _ = load_image(ctx_img_path)
        ctx_mask_np, _ = load_image(ctx_mask_path)
        
        ctx_img_tensor, ctx_mask_tensor = preprocess_2d(ctx_img_np, ctx_mask_np, image_size=image_size)
        context_imgs.append(ctx_img_tensor)
        context_masks.append(ctx_mask_tensor)
        context_np_list.append(ctx_img_np)
    
    # Extract context features
    print("Extracting context features...")
    context_features = feature_extractor.extract(context_np_list)  # [K, N, D]
    context_features = context_features.unsqueeze(0)  # [1, K, N, D]
    
    # Stack context: [1, K, 1, H, W]
    context_imgs = torch.cat(context_imgs, dim=0).unsqueeze(0)  # [1, K, 1, H, W]
    context_masks = torch.cat(context_masks, dim=0).unsqueeze(0)  # [1, K, 1, H, W]
    
    print(f"Target shape: {target_tensor.shape}")
    print(f"Context images shape: {context_imgs.shape}")
    print(f"Context masks shape: {context_masks.shape}")
    print(f"Target features shape: {target_features.shape}")
    print(f"Context features shape: {context_features.shape}")
    
    # Run inference
    pred_probs = run_inference(
        model, target_tensor, context_imgs, context_masks, device,
        target_features=target_features,
        context_features=context_features,
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
