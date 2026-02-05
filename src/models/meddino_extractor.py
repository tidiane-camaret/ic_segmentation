"""
MedDINOv3 Feature Extractor for on-the-fly feature computation.

Provides efficient batched feature extraction from MedDINOv3 ViT-base model
for use during training/inference when precomputed features are not available.
"""
from __future__ import annotations

import sys
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MedDINOProcessor:
    """Preprocessor for MedDINO inputs."""

    def __init__(
        self,
        target_size: int = 256,
        interpolation: str = "bilinear",
    ):
        """
        Args:
            target_size: Target resolution for feature extraction
            interpolation: Interpolation mode for resizing
        """
        self.target_size = target_size
        self.interpolation = interpolation
        # ImageNet normalization values
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def __call__(
        self,
        images: torch.Tensor,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Process a batch of images for MedDINO.

        Args:
            images: [B, 1, H, W] or [B, H, W] grayscale images, normalized to [0, 1]
            device: Target device for output tensor

        Returns:
            processed: [B, 3, target_size, target_size] normalized RGB tensor
        """
        if device is None:
            device = images.device

        # Ensure 4D tensor
        if images.dim() == 3:
            images = images.unsqueeze(1)

        B = images.shape[0]

        # Percentile clipping and rescaling (per image)
        processed = []
        for i in range(B):
            img = images[i, 0]  # [H, W]

            # Percentile clipping (0.5% - 99.5%)
            lower = torch.quantile(img, 0.005)
            upper = torch.quantile(img, 0.995)
            img = torch.clamp(img, lower, upper)

            # Rescale to [0, 1]
            img_min, img_max = img.min(), img.max()
            if img_max > img_min:
                img = (img - img_min) / (img_max - img_min)
            else:
                img = torch.zeros_like(img)

            processed.append(img)

        # Stack and expand to RGB
        processed = torch.stack(processed, dim=0).unsqueeze(1)  # [B, 1, H, W]
        processed = processed.expand(-1, 3, -1, -1)  # [B, 3, H, W]

        # Resize to target size
        processed = F.interpolate(
            processed,
            size=(self.target_size, self.target_size),
            mode=self.interpolation,
            align_corners=False if self.interpolation != "nearest" else None,
        )

        # ImageNet normalization
        mean = self.mean.to(device)
        std = self.std.to(device)
        processed = (processed - mean) / std

        return processed


class MedDINOFeatureExtractor(nn.Module):
    """
    On-the-fly MedDINOv3 feature extractor.

    Wraps MedDINOv3 ViT-base model for efficient batched feature extraction.
    Features are computed on-the-fly during training, eliminating the need
    for precomputed feature files.
    """

    def __init__(
        self,
        model_path: str,
        target_size: int = 256,
        device: Union[str, torch.device] = "cuda",
        layer_idx: int = 11,
        freeze: bool = True,
    ):
        """
        Args:
            model_path: Path to MedDINOv3 checkpoint
            target_size: Input resolution for feature extraction
            device: Device for model and computation
            layer_idx: Which transformer layer to extract features from (default: 11)
            freeze: Whether to freeze model weights (default: True)
        """
        super().__init__()
        self.target_size = target_size
        self.layer_idx = layer_idx
        self.device = torch.device(device) if isinstance(device, str) else device

        # Initialize processor
        self.processor = MedDINOProcessor(target_size=target_size)

        # Load MedDINOv3 model
        self._load_model(model_path)

        # Freeze weights if requested
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()

        self._frozen = freeze

    def _load_model(self, model_path: str):
        """Load MedDINOv3 ViT-base model."""
        # Add MedDINOv3 to path
        meddino_path = "/software/notebooks/camaret/repos/MedDINOv3/nnUNet/nnunetv2/training/nnUNetTrainer/dinov3/"
        if meddino_path not in sys.path:
            sys.path.insert(0, meddino_path)

        from dinov3.models.vision_transformer import vit_base

        # Initialize architecture
        self.model = vit_base(
            drop_path_rate=0.2,
            layerscale_init=1.0e-05,
            n_storage_tokens=4,
            qkv_bias=False,
            mask_k_bias=True,
        )

        # Load weights
        print(f"Loading MedDINOv3 weights from {model_path}...")
        checkpoint = torch.load(model_path, weights_only=False, map_location="cpu")
        state_dict = checkpoint["teacher"]

        # Clean up state dict keys
        state_dict = {
            k.replace("backbone.", ""): v
            for k, v in state_dict.items()
            if "ibot" not in k and "dino_head" not in k
        }

        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        print(f"MedDINOv3 loaded successfully (layer {self.layer_idx})")

    def train(self, mode: bool = True):
        """Override train to keep model in eval mode if frozen."""
        super().train(mode)
        if self._frozen:
            self.model.eval()
        return self

    @torch.no_grad()
    def extract_features(
        self,
        images: torch.Tensor,
        return_all_layers: bool = False,
    ) -> torch.Tensor:
        """
        Extract features from a batch of images.

        Args:
            images: [B, 1, H, W] or [B, H, W] grayscale images, normalized to [0, 1]
            return_all_layers: If True, return features from layers [2, 5, 8, 11]

        Returns:
            If return_all_layers=False:
                features: [B, N, D] where N = 1 (CLS) + 4 (registers) + num_patches
                          and D = 768 for ViT-base
            If return_all_layers=True:
                dict with layer_{idx}_* keys for each layer
        """
        was_training = self.model.training
        self.model.eval()

        # Preprocess images
        pixel_values = self.processor(images, device=self.device)
        pixel_values = pixel_values.to(self.device)

        # Extract intermediate layers
        if return_all_layers:
            layer_indices = [2, 5, 8, 11]
        else:
            layer_indices = [self.layer_idx]

        intermediate_layers = self.model.get_intermediate_layers(
            pixel_values,
            n=layer_indices,
            reshape=False,
        )

        if return_all_layers:
            results = {}
            for i, idx in enumerate(layer_indices):
                feat = intermediate_layers[i]  # [B, Tokens, D]
                # Token decomposition: CLS (0), Registers (1:5), Patches (5:)
                results[f"layer_{idx}_cls"] = feat[:, 0:1, :]  # [B, 1, D]
                results[f"layer_{idx}_registers"] = feat[:, 1:5, :]  # [B, 4, D]
                results[f"layer_{idx}_patches"] = feat[:, 5:, :]  # [B, num_patches, D]

            if was_training and not self._frozen:
                self.model.train()
            return results
        else:
            # Return concatenated features from target layer
            feat = intermediate_layers[0]  # [B, Tokens, D]
            # Concatenate CLS + registers + patches
            # feat is already [B, total_tokens, D] where total_tokens = 1 + 4 + num_patches

            if was_training and not self._frozen:
                self.model.train()
            return feat

    def forward(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for feature extraction.

        Args:
            images: [B, 1, H, W] grayscale images normalized to [0, 1]

        Returns:
            features: [B, N, D] where N = total tokens, D = 768
        """
        return self.extract_features(images, return_all_layers=False)

    def extract_batch(
        self,
        target_images: torch.Tensor,
        context_images: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Extract features for target and context images efficiently.

        Args:
            target_images: [B, 1, H, W] target images
            context_images: [B, k, 1, H, W] context images (optional)

        Returns:
            target_features: [B, N, D]
            context_features: [B, k, N, D] or None
        """
        B = target_images.shape[0]

        if context_images is not None:
            # Batch all images together for efficient extraction
            k = context_images.shape[1]

            # Reshape context: [B, k, 1, H, W] -> [B*k, 1, H, W]
            ctx_flat = context_images.view(B * k, *context_images.shape[2:])

            # Concatenate target and context
            all_images = torch.cat([target_images, ctx_flat], dim=0)  # [B + B*k, 1, H, W]

            # Extract features for all images at once
            all_features = self.extract_features(all_images)  # [B + B*k, N, D]

            # Split back
            target_features = all_features[:B]  # [B, N, D]
            context_features = all_features[B:].view(B, k, *all_features.shape[1:])  # [B, k, N, D]

            return target_features, context_features
        else:
            target_features = self.extract_features(target_images)
            return target_features, None


class MultiLayerFeatureExtractor(nn.Module):
    """
    Extract and fuse features from multiple MedDINO layers.

    Supports different fusion strategies:
    - "average": Simple averaging across layers
    - "learned_weighted": Learned weighted combination (trainable)
    - "concat_proj": Concatenate + project back to original dimension
    """

    def __init__(
        self,
        base_extractor: MedDINOFeatureExtractor,
        layers: list[int] = [2, 5, 8, 11],
        fusion: str = "average",
        embed_dim: int = 768,
    ):
        """
        Args:
            base_extractor: MedDINOFeatureExtractor instance
            layers: List of layer indices to extract (default: [2, 5, 8, 11])
            fusion: Fusion strategy - "average", "learned_weighted", "concat_proj"
            embed_dim: Embedding dimension (768 for ViT-base)
        """
        super().__init__()
        self.base_extractor = base_extractor
        self.layers = layers
        self.fusion = fusion
        self.embed_dim = embed_dim
        self.num_layers = len(layers)

        # Initialize fusion parameters based on strategy
        if fusion == "learned_weighted":
            # Learnable weights for each layer, initialized to uniform
            self.layer_weights = nn.Parameter(torch.ones(self.num_layers) / self.num_layers)
        elif fusion == "concat_proj":
            # Linear projection from concatenated features back to embed_dim
            self.proj = nn.Linear(embed_dim * self.num_layers, embed_dim)
        # "average" requires no additional parameters

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Extract and fuse features from multiple layers."""
        return self.extract_features(images)

    @torch.no_grad()
    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract and fuse features from multiple MedDINO layers.

        Args:
            images: [B, 1, H, W] or [B, H, W] grayscale images

        Returns:
            fused_features: [B, N, D] where N = total tokens, D = embed_dim
        """
        # Get features from all specified layers
        layer_features = self.base_extractor.extract_features(images, return_all_layers=True)

        # Collect patch features from each layer
        # Each layer has CLS + registers + patches, we focus on patches
        all_layer_patches = []
        for layer_idx in self.layers:
            patches = layer_features[f"layer_{layer_idx}_patches"]  # [B, num_patches, D]
            all_layer_patches.append(patches)

        # Stack: [num_layers, B, N, D]
        stacked = torch.stack(all_layer_patches, dim=0)

        # Apply fusion strategy
        if self.fusion == "average":
            fused = stacked.mean(dim=0)  # [B, N, D]
        elif self.fusion == "learned_weighted":
            # Normalize weights with softmax for stable training
            weights = F.softmax(self.layer_weights, dim=0)
            # Weighted sum: [num_layers, B, N, D] * [num_layers, 1, 1, 1] -> sum
            weights = weights.view(-1, 1, 1, 1)
            fused = (stacked * weights).sum(dim=0)  # [B, N, D]
        elif self.fusion == "concat_proj":
            # Concatenate along feature dimension
            B, N, D = stacked.shape[1:]
            concat = stacked.permute(1, 2, 0, 3).reshape(B, N, -1)  # [B, N, D*num_layers]
            fused = self.proj(concat)  # [B, N, D]
        else:
            raise ValueError(f"Unknown fusion strategy: {self.fusion}")

        return fused

    def extract_batch(
        self,
        target_images: torch.Tensor,
        context_images: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Extract fused features for target and context images.

        Args:
            target_images: [B, 1, H, W] target images
            context_images: [B, k, 1, H, W] context images (optional)

        Returns:
            target_features: [B, N, D]
            context_features: [B, k, N, D] or None
        """
        B = target_images.shape[0]

        if context_images is not None:
            k = context_images.shape[1]
            ctx_flat = context_images.view(B * k, *context_images.shape[2:])
            all_images = torch.cat([target_images, ctx_flat], dim=0)
            all_features = self.extract_features(all_images)
            target_features = all_features[:B]
            context_features = all_features[B:].view(B, k, *all_features.shape[1:])
            return target_features, context_features
        else:
            target_features = self.extract_features(target_images)
            return target_features, None

    def get_layer_weights(self) -> Optional[torch.Tensor]:
        """Get current layer weights (for learned_weighted fusion)."""
        if self.fusion == "learned_weighted":
            return F.softmax(self.layer_weights, dim=0).detach()
        return None


def create_multilayer_extractor(
    model_path: str,
    layers: list[int] = [2, 5, 8, 11],
    fusion: str = "average",
    target_size: int = 256,
    device: Union[str, torch.device] = "cuda",
    freeze_base: bool = True,
) -> MultiLayerFeatureExtractor:
    """
    Factory function to create a multi-layer MedDINO feature extractor.

    Args:
        model_path: Path to MedDINOv3 checkpoint
        layers: List of layer indices to extract (default: [2, 5, 8, 11])
        fusion: Fusion strategy - "average", "learned_weighted", "concat_proj"
        target_size: Input resolution for feature extraction
        device: Device for computation
        freeze_base: Whether to freeze base extractor weights

    Returns:
        MultiLayerFeatureExtractor instance
    """
    # Create base extractor (layer_idx doesn't matter since we use return_all_layers)
    base_extractor = MedDINOFeatureExtractor(
        model_path=model_path,
        target_size=target_size,
        device=device,
        layer_idx=11,  # Default, won't be used
        freeze=freeze_base,
    )

    extractor = MultiLayerFeatureExtractor(
        base_extractor=base_extractor,
        layers=layers,
        fusion=fusion,
        embed_dim=768,  # ViT-base
    )
    extractor.to(device)

    return extractor


def create_meddino_extractor(
    model_path: str,
    target_size: int = 256,
    device: Union[str, torch.device] = "cuda",
    layer_idx: int = 11,
    freeze: bool = True,
) -> MedDINOFeatureExtractor:
    """
    Factory function to create a MedDINO feature extractor.

    Args:
        model_path: Path to MedDINOv3 checkpoint
        target_size: Input resolution (should match feature_extraction_resolution in config)
        device: Device for computation
        layer_idx: Transformer layer to extract (default: 11, final layer)
        freeze: Whether to freeze model weights

    Returns:
        MedDINOFeatureExtractor instance
    """
    return MedDINOFeatureExtractor(
        model_path=model_path,
        target_size=target_size,
        device=device,
        layer_idx=layer_idx,
        freeze=freeze,
    )


class MedSAM2Processor:
    """Preprocessor for MedSAM2 inputs."""

    def __init__(
        self,
        target_size: int = 1024,
        interpolation: str = "bilinear",
    ):
        """
        Args:
            target_size: Target resolution for feature extraction (default 1024 for SAM2)
            interpolation: Interpolation mode for resizing
        """
        self.target_size = target_size
        self.interpolation = interpolation
        # SAM2 uses pixel_mean and pixel_std for normalization
        self.pixel_mean = torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1)
        self.pixel_std = torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1)

    def __call__(
        self,
        images: torch.Tensor,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Process a batch of images for MedSAM2.

        Args:
            images: [B, 1, H, W] or [B, H, W] grayscale images, normalized to [0, 1]
            device: Target device for output tensor

        Returns:
            processed: [B, 3, target_size, target_size] normalized RGB tensor
        """
        if device is None:
            device = images.device

        # Ensure 4D tensor
        if images.dim() == 3:
            images = images.unsqueeze(1)

        B = images.shape[0]

        # Percentile clipping and rescaling (per image) to [0, 255] for SAM2
        processed = []
        for i in range(B):
            img = images[i, 0]  # [H, W]

            # Percentile clipping (0.5% - 99.5%)
            lower = torch.quantile(img, 0.005)
            upper = torch.quantile(img, 0.995)
            img = torch.clamp(img, lower, upper)

            # Rescale to [0, 255] (SAM2 expects uint8-like values before normalization)
            img_min, img_max = img.min(), img.max()
            if img_max > img_min:
                img = (img - img_min) / (img_max - img_min) * 255.0
            else:
                img = torch.zeros_like(img)

            processed.append(img)

        # Stack and expand to RGB
        processed = torch.stack(processed, dim=0).unsqueeze(1)  # [B, 1, H, W]
        processed = processed.expand(-1, 3, -1, -1).clone()  # [B, 3, H, W]

        # Resize to target size
        processed = F.interpolate(
            processed,
            size=(self.target_size, self.target_size),
            mode=self.interpolation,
            align_corners=False if self.interpolation != "nearest" else None,
        )

        # SAM2 normalization: (x - mean) / std
        mean = self.pixel_mean.to(device)
        std = self.pixel_std.to(device)
        processed = (processed - mean) / std

        return processed


class MedSAM2FeatureExtractor(nn.Module):
    """
    On-the-fly MedSAM2 feature extractor.

    Wraps MedSAM2 (SAM2.1-based) model for efficient batched feature extraction.
    Features are computed on-the-fly during training, eliminating the need
    for precomputed feature files.

    MedSAM2 uses a Hiera backbone with FPN neck, outputting 256-dim features.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        config_name: str = "sam2.1_hiera_t.yaml",
        target_size: int = 1024,
        device: Union[str, torch.device] = "cuda",
        freeze: bool = True,
        use_fpn_features: bool = True,
    ):
        """
        Args:
            model_path: Path to MedSAM2 checkpoint (if None, downloads from HuggingFace)
            config_name: SAM2 config file name (default: "sam2.1_hiera_t.yaml" for MedSAM2)
            target_size: Input resolution for feature extraction (default: 1024)
            device: Device for model and computation
            freeze: Whether to freeze model weights (default: True)
            use_fpn_features: If True, use FPN output. If False, use raw backbone features.
        """
        super().__init__()
        self.target_size = target_size
        self.device = torch.device(device) if isinstance(device, str) else device
        self.use_fpn_features = use_fpn_features
        self.config_name = config_name

        # Initialize processor
        self.processor = MedSAM2Processor(target_size=target_size)

        # Load model
        self._load_model(model_path, config_name)

        # Freeze weights if requested
        if freeze:
            for param in self.image_encoder.parameters():
                param.requires_grad = False
            self.image_encoder.eval()

        self._frozen = freeze

    def _load_model(self, model_path: Optional[str], config_name: str):
        """Load MedSAM2 model."""
        # Add SAM2 to path
        sam2_path = "/software/notebooks/camaret/repos/nnInteractive_fork/nnInteractive/supervoxel/src/sam2"
        if sam2_path not in sys.path:
            sys.path.insert(0, sam2_path)

        # Download from HuggingFace if no path provided
        if model_path is None:
            from huggingface_hub import hf_hub_download
            print("Downloading MedSAM2 from HuggingFace...")
            model_path = hf_hub_download(
                repo_id="wanglab/MedSAM2",
                filename="MedSAM2_latest.pt",
            )
            print(f"Downloaded to {model_path}")

        # Build image encoder directly (avoid Hydra instantiate issues)
        print(f"Building SAM2 image encoder ({config_name})...")
        self.image_encoder = self._build_image_encoder(config_name)

        # Load MedSAM2 weights (only image encoder part)
        print(f"Loading MedSAM2 weights from {model_path}...")
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)

        # Handle different checkpoint formats
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint

        # Filter to only image_encoder weights
        encoder_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("image_encoder."):
                new_key = k.replace("image_encoder.", "")
                encoder_state_dict[new_key] = v

        if encoder_state_dict:
            missing, unexpected = self.image_encoder.load_state_dict(encoder_state_dict, strict=False)
            if missing:
                print(f"  Missing keys: {len(missing)}")
            if unexpected:
                print(f"  Unexpected keys: {len(unexpected)}")
        else:
            print("Warning: No image_encoder weights found in checkpoint")

        self.image_encoder.to(self.device)
        print(f"MedSAM2 image encoder loaded successfully")

    def _build_image_encoder(self, config_name: str):
        """Build SAM2 image encoder from config name."""
        # Mock iopath if not installed (only needed for checkpoint loading, not inference)
        try:
            import iopath
        except ImportError:
            import types
            iopath_mock = types.ModuleType("iopath")
            iopath_mock.common = types.ModuleType("iopath.common")
            iopath_mock.common.file_io = types.ModuleType("iopath.common.file_io")
            iopath_mock.common.file_io.g_pathmgr = None
            sys.modules["iopath"] = iopath_mock
            sys.modules["iopath.common"] = iopath_mock.common
            sys.modules["iopath.common.file_io"] = iopath_mock.common.file_io

        # Import SAM2 components directly
        from sam2.modeling.backbones.hieradet import Hiera
        from sam2.modeling.backbones.image_encoder import FpnNeck, ImageEncoder
        from sam2.modeling.position_encoding import PositionEmbeddingSine

        # Model configurations for different sizes
        configs = {
            "sam2.1_hiera_t.yaml": {
                "embed_dim": 96,
                "num_heads": 1,
                "stages": [1, 2, 7, 2],
                "global_att_blocks": [5, 7, 9],
                "backbone_channel_list": [768, 384, 192, 96],
            },
            "sam2.1_hiera_s.yaml": {
                "embed_dim": 96,
                "num_heads": 1,
                "stages": [1, 2, 11, 2],
                "global_att_blocks": [7, 10, 13],
                "backbone_channel_list": [768, 384, 192, 96],
            },
            "sam2.1_hiera_b+.yaml": {
                "embed_dim": 112,
                "num_heads": 2,
                "stages": [2, 3, 16, 3],
                "global_att_blocks": [12, 16, 20],
                "backbone_channel_list": [896, 448, 224, 112],
            },
            "sam2.1_hiera_l.yaml": {
                "embed_dim": 144,
                "num_heads": 2,
                "stages": [2, 6, 36, 4],
                "global_att_blocks": [23, 33, 43],
                "backbone_channel_list": [1152, 576, 288, 144],
            },
        }

        if config_name not in configs:
            raise ValueError(f"Unknown config: {config_name}. Available: {list(configs.keys())}")

        cfg = configs[config_name]
        self.feature_dim = 256  # FPN d_model is always 256

        # Build Hiera backbone (trunk)
        trunk = Hiera(
            embed_dim=cfg["embed_dim"],
            num_heads=cfg["num_heads"],
            stages=cfg["stages"],
            global_att_blocks=cfg["global_att_blocks"],
            window_pos_embed_bkg_spatial_size=[7, 7],
        )

        # Build position encoding
        position_encoding = PositionEmbeddingSine(
            num_pos_feats=256,
            normalize=True,
            scale=None,
            temperature=10000,
        )

        # Build FPN neck
        neck = FpnNeck(
            position_encoding=position_encoding,
            d_model=256,
            backbone_channel_list=cfg["backbone_channel_list"],
            fpn_top_down_levels=[2, 3],
            fpn_interp_model="nearest",
        )

        # Build full image encoder
        image_encoder = ImageEncoder(
            trunk=trunk,
            neck=neck,
            scalp=1,
        )

        return image_encoder

    def train(self, mode: bool = True):
        """Override train to keep model in eval mode if frozen."""
        super().train(mode)
        if self._frozen:
            self.image_encoder.eval()
        return self

    @torch.no_grad()
    def extract_features(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract features from a batch of images.

        Args:
            images: [B, 1, H, W] or [B, H, W] grayscale images, normalized to [0, 1]

        Returns:
            features: [B, N, D] where N = H*W/256 (spatial tokens), D = 256 (FPN output dim)
        """
        was_training = self.image_encoder.training
        self.image_encoder.eval()

        # Preprocess images
        pixel_values = self.processor(images, device=self.device)
        pixel_values = pixel_values.to(self.device)

        # Forward through image encoder
        encoder_output = self.image_encoder(pixel_values)

        # Get vision features: [B, D, H/16, W/16]
        vision_features = encoder_output["vision_features"]

        # Reshape to [B, N, D] format (flatten spatial dimensions)
        B, D, H, W = vision_features.shape
        features = vision_features.permute(0, 2, 3, 1)  # [B, H, W, D]
        features = features.reshape(B, H * W, D)  # [B, N, D]

        if was_training and not self._frozen:
            self.image_encoder.train()

        return features

    def forward(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for feature extraction.

        Args:
            images: [B, 1, H, W] grayscale images normalized to [0, 1]

        Returns:
            features: [B, N, D] where N = spatial tokens, D = 256
        """
        return self.extract_features(images)

    def extract_batch(
        self,
        target_images: torch.Tensor,
        context_images: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Extract features for target and context images efficiently.

        Args:
            target_images: [B, 1, H, W] target images
            context_images: [B, k, 1, H, W] context images (optional)

        Returns:
            target_features: [B, N, D]
            context_features: [B, k, N, D] or None
        """
        B = target_images.shape[0]

        if context_images is not None:
            # Batch all images together for efficient extraction
            k = context_images.shape[1]

            # Reshape context: [B, k, 1, H, W] -> [B*k, 1, H, W]
            ctx_flat = context_images.view(B * k, *context_images.shape[2:])

            # Concatenate target and context
            all_images = torch.cat([target_images, ctx_flat], dim=0)  # [B + B*k, 1, H, W]

            # Extract features for all images at once
            all_features = self.extract_features(all_images)  # [B + B*k, N, D]

            # Split back
            target_features = all_features[:B]  # [B, N, D]
            context_features = all_features[B:].view(B, k, *all_features.shape[1:])  # [B, k, N, D]

            return target_features, context_features
        else:
            target_features = self.extract_features(target_images)
            return target_features, None


def create_medsam2_extractor(
    model_path: Optional[str] = None,
    config_name: str = "sam2.1_hiera_t.yaml",
    target_size: int = 1024,
    device: Union[str, torch.device] = "cuda",
    freeze: bool = True,
) -> MedSAM2FeatureExtractor:
    """
    Factory function to create a MedSAM2 feature extractor.

    Args:
        model_path: Path to MedSAM2 checkpoint. If None, downloads from HuggingFace.
        config_name: SAM2 config file (default: "sam2.1_hiera_t.yaml" for MedSAM2 tiny model)
        target_size: Input resolution (default: 1024 for SAM2)
        device: Device for computation
        freeze: Whether to freeze model weights

    Returns:
        MedSAM2FeatureExtractor instance
    """
    return MedSAM2FeatureExtractor(
        model_path=model_path,
        config_name=config_name,
        target_size=target_size,
        device=device,
        freeze=freeze,
    )


def create_feature_extractor(
    extractor_type: str,
    device: Union[str, torch.device] = "cuda",
    **kwargs,
) -> nn.Module:
    """
    Factory function to create a feature extractor based on type.

    Args:
        extractor_type: "meddino" or "medsam2"
        device: Device for computation
        **kwargs: Additional arguments passed to the specific extractor

    Returns:
        Feature extractor instance (MedDINOFeatureExtractor or MedSAM2FeatureExtractor)
    """
    if extractor_type.lower() in ["meddino", "meddinov3", "meddino_v3"]:
        return create_meddino_extractor(device=device, **kwargs)
    elif extractor_type.lower() in ["medsam2", "medsam", "sam2"]:
        return create_medsam2_extractor(device=device, **kwargs)
    else:
        raise ValueError(f"Unknown extractor type: {extractor_type}. Choose 'meddino' or 'medsam2'.")
