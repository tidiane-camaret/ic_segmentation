"""
MedSAM2 Feature Extractor for on-the-fly feature computation.

Extracts per-image features from MedSAM2's Hiera encoder + FPN neck.
Unlike UniverSeg, this is a standard vision encoder: image in, features out.
No support/target/cross-conv concept - both target and context images go through
the same pure encoder path.

Features:
- Input: Grayscale images resized to 512x512, replicated to 3 channels
- FPN outputs all projected to 256 channels
- With scalp=1 (default MedSAM2 config), 3 FPN levels are returned:
    level 0: 128x128 (stride 4 from 512 input) - highest resolution
    level 1: 64x64   (stride 8)
    level 2: 32x32   (stride 16) - lowest resolution

Usage:
    # Single level (default: highest resolution 128x128)
    extractor = MedSAM2Extractor(layer_idx=0, device="cuda")
    # All levels concatenated
    extractor = MedSAM2Extractor(layer_idx="all", output_grid_size=32, device="cuda")

    target_features, context_features = extractor.extract_batch(
        target_images, context_images, context_masks
    )
    # Note: context_masks are ignored - MedSAM2 encoder doesn't use them
"""
from __future__ import annotations

import os
import sys
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# MedSAM2 repo path - try multiple possible locations
_MEDSAM2_REPO_CANDIDATES = [
    "/software/notebooks/camaret/repos/MedSAM2",
    "/home/dpxuser/repos/MedSAM2",  # Jailed user path
    os.path.expanduser("~/repos/MedSAM2"),
]

def _find_medsam2_repo():
    """Find MedSAM2 repo in possible locations."""
    for path in _MEDSAM2_REPO_CANDIDATES:
        if os.path.isdir(path) and os.path.exists(os.path.join(path, "sam2")):
            return path
    raise RuntimeError(
        f"MedSAM2 repo not found in any of: {_MEDSAM2_REPO_CANDIDATES}. "
        "Clone it from https://github.com/bowang-lab/MedSAM2"
    )

MEDSAM2_REPO = _find_medsam2_repo()
MEDSAM2_CHECKPOINT = os.path.join(MEDSAM2_REPO, "checkpoints", "MedSAM2_latest.pt")
MEDSAM2_CONFIG = "configs/sam2.1_hiera_t512.yaml"

# Grid sizes per FPN level at 512 input (after scalp=1 removes lowest-res 16x16 level)
# Level ordering: 0 = highest res (128x128), 2 = lowest res (32x32)
LAYER_GRID_SIZES = {0: 128, 1: 64, 2: 32}
NUM_FPN_LEVELS = 3  # After scalp=1
FPN_DIM = 256  # All FPN outputs are projected to this dim


def _parse_layer_idx(layer_idx) -> List[int]:
    """Parse layer_idx into a sorted list of layer indices."""
    if layer_idx == "all":
        return list(range(NUM_FPN_LEVELS))
    if isinstance(layer_idx, (list, tuple)):
        return sorted(int(i) for i in layer_idx)
    return [int(layer_idx)]


class MedSAM2Extractor(nn.Module):
    """Extract per-image features from MedSAM2's Hiera encoder + FPN.

    This is a standard vision encoder - no cross-conv or support images.
    Both target and context images use the same encoder path.

    Args:
        layer_idx: FPN level(s) to extract from. Options:
            - int (0-2): single level (0=128x128, 1=64x64, 2=32x32)
            - "all": all 3 levels, concatenated along feature dim
            - list of ints: specific levels, concatenated
        device: Device for computation.
        checkpoint_path: Path to MedSAM2 checkpoint. Uses default if None.
        freeze: Freeze all model weights.
        output_grid_size: Resize features to this grid. None = native (single-layer only).
            Required when using multiple layers since they have different native sizes.
        input_size: Input image size. Default 512 (MedSAM2 native size).
    """

    DEFAULT_INPUT_SIZE = 512  # MedSAM2 native input size
    FEATURE_DIM_PER_LAYER = FPN_DIM  # 256 for all FPN levels

    def __init__(
        self,
        layer_idx: Union[int, str, List[int]] = 0,  # Default to highest res (128x128)
        device: Union[str, torch.device] = "cuda",
        checkpoint_path: Optional[str] = None,
        freeze: bool = True,
        output_grid_size: Optional[int] = None,
        input_size: int = 512,
        compile_model: bool = False,
    ):
        super().__init__()
        self.device = torch.device(device) if isinstance(device, str) else device
        self.input_size = input_size
        self.checkpoint_path = checkpoint_path or MEDSAM2_CHECKPOINT

        # Parse layer indices
        self.layer_indices = _parse_layer_idx(layer_idx)
        self.multi_layer = len(self.layer_indices) > 1
        self.feature_dim = self.FEATURE_DIM_PER_LAYER * len(self.layer_indices)

        # For backwards compat, expose single layer_idx when not multi-layer
        self.layer_idx = self.layer_indices[0] if not self.multi_layer else self.layer_indices

        # Determine output grid size
        if self.multi_layer:
            if output_grid_size is None:
                # Default: use the smallest native grid among selected layers
                output_grid_size = min(LAYER_GRID_SIZES[i] for i in self.layer_indices)
            self.native_grid_size = None
        else:
            self.native_grid_size = LAYER_GRID_SIZES[self.layer_indices[0]]
            if output_grid_size is None:
                output_grid_size = self.native_grid_size
        self.output_grid_size = output_grid_size
        self._compile_model = compile_model
        # Load MedSAM2 model
        self._load_model()

        if freeze:
            for param in self.image_encoder.parameters():
                param.requires_grad = False
            self.image_encoder.eval()
        self._frozen = freeze

        layers_str = "all" if layer_idx == "all" else self.layer_indices
        print(f"MedSAM2 extractor: layers={layers_str}, "
              f"feature_dim={self.feature_dim}, "
              f"input={self.input_size}x{self.input_size}, "
              f"output={self.output_grid_size}x{self.output_grid_size}")

    def _load_model(self):
        """Load MedSAM2 model and extract image encoder."""
        if MEDSAM2_REPO not in sys.path:
            sys.path.insert(0, MEDSAM2_REPO)

        # Handle hydra config conflicts - sam2 uses hydra internally
        from hydra import initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra

        # Clear any existing hydra state
        gh = GlobalHydra.instance()
        if gh.is_initialized():
            gh.clear()

        # Use absolute path to sam2 configs directory
        sam2_config_dir = os.path.join(MEDSAM2_REPO, "sam2")

        try:
            # Initialize hydra with sam2 config directory
            initialize_config_dir(config_dir=sam2_config_dir, version_base="1.2")

            from sam2.build_sam import build_sam2

            # Build full SAM2 model
            model = build_sam2(
                config_file=MEDSAM2_CONFIG,
                ckpt_path=self.checkpoint_path,
                device=self.device,
                mode="eval",
            )
        finally:
            # Clear sam2's hydra state
            if gh.is_initialized():
                gh.clear()

        # Extract only the image encoder (we don't need decoder/memory modules)
        self.image_encoder = model.image_encoder

        # Clean up the rest of the model
        del model
        torch.cuda.empty_cache()
        if self._compile_model and hasattr(torch, "compile"):
                    print("Compiling MedSAM2 image encoder via PyTorch 2.0...")
                    self.image_encoder = torch.compile(
                        self.image_encoder,
                        mode="max-autotune", # Or "reduce-overhead" for faster initial compilation
                        dynamic=True,        # CRITICAL: prevents recompiling when batch size changes
                        fullgraph=False      # Allows graceful fallback if a Hiera operation blocks compilation
                    )
        print(f"MedSAM2 image encoder loaded from {self.checkpoint_path}")

    def train(self, mode: bool = True):
        """Keep model in eval mode if frozen."""
        super().train(mode)
        if self._frozen:
            self.image_encoder.eval()
        return self

    def _resize_feat(self, feat: torch.Tensor) -> torch.Tensor:
        """Resize feature map to output_grid_size."""
        _, _, H, W = feat.shape
        if H == self.output_grid_size and W == self.output_grid_size:
            return feat
        if self.output_grid_size < H:
            return F.adaptive_avg_pool2d(feat, (self.output_grid_size, self.output_grid_size))
        return F.interpolate(
            feat, size=(self.output_grid_size, self.output_grid_size),
            mode="bilinear", align_corners=False,
        )

    @torch.no_grad()
    def extract_features(
        self,
        images: torch.Tensor,
        masks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Extract features from MedSAM2 image encoder.

        Args:
            images: [B, 1, H, W] grayscale images, normalized to [0, 1]
            masks: Ignored - MedSAM2 encoder doesn't use masks.
                   Kept for API compatibility with UniverSegExtractor.

        Returns:
            features: [B, N, D] where N = output_grid_size^2,
                      D = 256 (single layer) or 256*L (multi-layer concat)
        """
        was_training = self.image_encoder.training
        self.image_encoder.eval()

        device = images.device
        if next(self.image_encoder.parameters()).device != device:
            self.image_encoder.to(device)

        B = images.shape[0]

        # Resize to input_size if needed
        if images.shape[-2:] != (self.input_size, self.input_size):
            images = F.interpolate(
                images,
                size=(self.input_size, self.input_size),
                mode="bilinear",
                align_corners=False,
            )

        # Replicate grayscale to 3 channels (Hiera expects RGB)
        if images.shape[1] == 1:
            images = images.expand(-1, 3, -1, -1)

        # Run image encoder
        # Returns: {"vision_features", "vision_pos_enc", "backbone_fpn"}
        encoder_output = self.image_encoder(images)
        backbone_fpn = encoder_output["backbone_fpn"]

        # backbone_fpn is a list of 3 tensors (after scalp=1):
        # [0]: 128x128, 256ch (highest res)
        # [1]: 64x64, 256ch
        # [2]: 32x32, 256ch (lowest res)

        # Collect and resize features from each requested level
        feat_list = []
        for lid in self.layer_indices:
            feat = self._resize_feat(backbone_fpn[lid])  # [B, 256, G, G]
            feat_list.append(feat)

        # Concatenate along channel dim: [B, 256*L, G, G]
        feat = torch.cat(feat_list, dim=1)

        # [B, D, G, G] -> [B, G*G, D] = [B, N, D]
        B, D, G, _ = feat.shape
        features = feat.reshape(B, D, G * G).permute(0, 2, 1)

        if was_training and not self._frozen:
            self.image_encoder.train()

        return features

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.extract_features(images)

    def extract_batch(
        self,
        target_images: torch.Tensor,
        context_images: Optional[torch.Tensor] = None,
        context_masks: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Extract features for target and context images.

        Unlike UniverSeg, MedSAM2 doesn't have mask-conditioned features.
        Both target and context go through the same encoder path.
        context_masks are ignored (kept for API compatibility).

        Args:
            target_images: [B, 1, H, W]
            context_images: [B, k, 1, H, W] (optional)
            context_masks: [B, k, 1, H, W] - IGNORED (kept for API compat)

        Returns:
            target_features: [B, N, D]
            context_features: [B, k, N, D] or None
        """
        B = target_images.shape[0]

        if context_images is None:
            return self.extract_features(target_images), None

        k = context_images.shape[1]
        ctx_flat = context_images.view(B * k, *context_images.shape[2:])

        # Batch target + context together (faster, no separate paths needed)
        all_images = torch.cat([target_images, ctx_flat], dim=0)
        all_features = self.extract_features(all_images)

        target_features = all_features[:B]
        context_features = all_features[B:].view(B, k, *all_features.shape[1:])

        return target_features, context_features

    def get_feature_info(self) -> dict:
        """Get information about extracted features."""
        grid = self.output_grid_size
        return {
            "extractor": "medsam2",
            "layer_indices": self.layer_indices,
            "layer_idx": self.layer_idx,
            "multi_layer": self.multi_layer,
            "feature_dim": self.feature_dim,
            "feature_dim_per_layer": self.FEATURE_DIM_PER_LAYER,
            "num_layers": len(self.layer_indices),
            "input_size": self.input_size,
            "native_grid_size": self.native_grid_size,
            "output_grid_size": grid,
            "num_tokens": grid * grid,
        }
    
    @torch.no_grad()
    def warmup(self, batch_size: int = 1):
        """Forces PyTorch to compile the graph before real data arrives."""
        if not getattr(self, "_compile_model", False):
            return
            
        print("Warming up compiled model (this may take a minute)...")
        self.image_encoder.eval()
        
        # Create a dummy tensor of the exact size the model expects
        dummy_input = torch.randn(
            batch_size, 3, self.input_size, self.input_size, 
            device=self.device, 
            dtype=torch.float32 # Or bfloat16 if using AMP
        )
        
        # Trigger compilation
        _ = self.image_encoder(dummy_input)
        torch.cuda.synchronize()
        print("Warmup complete. Model is fully compiled.")
