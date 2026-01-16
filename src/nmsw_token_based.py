"""
Token-Based NMSW SegFormer3D Model.

Replaces the local SegFormer3D branch with a Transformer-based pipeline
that processes small 8³ patches as tokens with cross-patch attention.

Architecture:
1. Global Branch: Small SegFormer3D on downsampled input (unchanged)
2. Objectness Scoring: Predict which patches are important (unchanged)
3. Patch Sampling: Select top-k patches using Gumbel-softmax (unchanged)
4. Token-Based Local Branch:
   - Patch Tokenizer: 3D CNN that converts 8³ patches to embeddings
   - 3D Positional Encoding: Learnable embeddings from normalized coords
   - Cross-Patch Transformer: Patches attend to each other
   - Patch Decoder: Project tokens back to 8³ masks
5. Aggregation: Combine global + local predictions (unchanged)
"""

import sys
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Add SegFormer3D to path
sys.path.insert(0, "/software/notebooks/camaret/repos/SegFormer3D")
from architectures.segformer3d import SegFormer3D

from .nmsw_aggregation import DynamicPatchAggregator
from .nmsw_sampling import PatchExtractor, PatchSampler, RandomPatchSampler


def build_small_segformer3d(config: Dict, scale: float = 0.5) -> SegFormer3D:
    """Build a smaller SegFormer3D for the global branch.

    Args:
        config: Model configuration
        scale: Scale factor for embed_dims (0.5 = half the channels)

    Returns:
        Smaller SegFormer3D model
    """
    model_params = config["model_parameters"]

    # Scale down embedding dimensions
    embed_dims = [max(8, int(d * scale)) for d in model_params["embed_dims"]]
    decoder_head_embedding_dim = max(32, int(model_params["decoder_head_embedding_dim"] * scale))

    model = SegFormer3D(
        in_channels=model_params["in_channels"],
        sr_ratios=model_params["sr_ratios"],
        embed_dims=embed_dims,
        patch_kernel_size=model_params["patch_kernel_size"],
        patch_stride=model_params["patch_stride"],
        patch_padding=model_params["patch_padding"],
        mlp_ratios=model_params["mlp_ratios"],
        num_heads=[max(1, h // 2) for h in model_params["num_heads"]],
        depths=[max(1, d // 2) for d in model_params["depths"]],
        decoder_head_embedding_dim=decoder_head_embedding_dim,
        num_classes=model_params["num_classes"],
        decoder_dropout=model_params["decoder_dropout"],
    )
    return model


class PatchTokenizer(nn.Module):
    """
    3D CNN that converts small patches (8³) into feature vectors.

    Uses a series of 3D convolutions to encode each patch into
    a compact embedding vector for transformer processing.

    Args:
        in_channels: Number of input channels (typically 1 for medical images)
        embed_dim: Output embedding dimension
        patch_size: Expected patch size (default 8³)
    """

    def __init__(
        self,
        in_channels: int = 1,
        embed_dim: int = 256,
        patch_size: Tuple[int, int, int] = (8, 8, 8),
    ):
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.patch_size = patch_size

        # 3-layer 3D CNN: progressively reduce spatial dimensions
        # Input: [B*K, C, 8, 8, 8]
        # After conv1: [B*K, 64, 4, 4, 4]
        # After conv2: [B*K, 128, 2, 2, 2]
        # After conv3: [B*K, 256, 1, 1, 1]
        self.encoder = nn.Sequential(
            # Layer 1: 8³ -> 4³
            nn.Conv3d(in_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.GELU(),

            # Layer 2: 4³ -> 2³
            nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.GELU(),

            # Layer 3: 2³ -> 1³
            nn.Conv3d(128, embed_dim, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm3d(embed_dim),
            nn.GELU(),
        )

        # Final projection to ensure exact embed_dim
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Tokenize patches into embeddings.

        Args:
            x: Patches [B*K, C, 8, 8, 8]

        Returns:
            tokens: Patch embeddings [B*K, embed_dim]
        """
        # Encode patches
        features = self.encoder(x)  # [B*K, embed_dim, 1, 1, 1]

        # Flatten spatial dimensions
        features = features.view(features.shape[0], -1)  # [B*K, embed_dim]

        # Final projection
        tokens = self.proj(features)  # [B*K, embed_dim]

        return tokens


class LearnablePositionalEncoding3D(nn.Module):
    """
    Learnable 3D positional encoding from normalized patch coordinates.

    Takes patch coordinates and generates position embeddings using
    a small MLP. Coordinates are normalized to [0, 1] range for
    robustness across different volume sizes.

    Args:
        embed_dim: Embedding dimension
        hidden_dim: Hidden layer dimension (default: embed_dim // 2)
    """

    def __init__(
        self,
        embed_dim: int = 256,
        hidden_dim: Optional[int] = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        hidden_dim = hidden_dim or embed_dim // 2

        # MLP to convert 3D coordinates to embeddings
        # Input: normalized (x, y, z) coordinates [0, 1]
        self.pos_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(
        self,
        slice_meta: List[Tuple[slice, slice, slice]],
        vol_size: Tuple[int, int, int],
        batch_size: int,
        num_patches: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Generate positional embeddings from patch coordinates.

        Args:
            slice_meta: List of slice tuples for each patch (length K*B)
            vol_size: Volume dimensions (D, H, W) for normalization
            batch_size: Batch size B
            num_patches: Number of patches per sample K
            device: Device for output tensor

        Returns:
            pos_embed: Position embeddings [B, K, embed_dim]
        """
        K = num_patches
        B = batch_size

        # Extract normalized center coordinates from slice_meta
        coords = []
        for slice_tuple in slice_meta:
            # Get center of patch in each dimension
            centers = []
            for i, (sl, vol_dim) in enumerate(zip(slice_tuple, vol_size)):
                center = (sl.start + sl.stop) / 2.0
                # Normalize to [0, 1]
                normalized = center / vol_dim
                centers.append(normalized)
            coords.append(centers)

        # Convert to tensor [K*B, 3]
        coords_tensor = torch.tensor(coords, dtype=torch.float32, device=device)

        # Generate embeddings [K*B, embed_dim]
        pos_embed = self.pos_encoder(coords_tensor)

        # Reshape to [B, K, embed_dim]
        # slice_meta is ordered as: [k0_b0, k0_b1, ..., k0_bB-1, k1_b0, ...]
        # We need to reshape carefully
        pos_embed = pos_embed.view(K, B, self.embed_dim).permute(1, 0, 2)

        return pos_embed


class CrossPatchTransformer(nn.Module):
    """
    Transformer encoder for cross-patch attention.

    Allows patches within the same volume to attend to each other,
    enabling spatial reasoning across the selected patches.

    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
        mlp_ratio: MLP hidden dimension ratio
        dropout: Dropout rate
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # Pre-normalization layer
        self.norm_pre = nn.LayerNorm(embed_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation='gelu',
            batch_first=True,  # Input: [B, K, embed_dim]
            norm_first=True,   # Pre-LN for better training stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # Post-normalization layer
        self.norm_post = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply cross-patch transformer attention.

        Args:
            x: Patch tokens [B, K, embed_dim]

        Returns:
            out: Transformed tokens [B, K, embed_dim]
        """
        x = self.norm_pre(x)
        x = self.transformer(x)
        x = self.norm_post(x)
        return x


class PatchDecoder(nn.Module):
    """
    Decode transformed tokens back to patch masks.

    Projects the transformer output tokens back to 8³ spatial
    masks using transposed convolutions.

    Args:
        embed_dim: Input embedding dimension
        num_classes: Number of output segmentation classes
        patch_size: Output patch size (default 8³)
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_classes: int = 1,
        patch_size: Tuple[int, int, int] = (8, 8, 8),
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.patch_size = patch_size

        # Project embedding to initial feature map size
        # Start from 1³ and upsample to 8³
        self.proj = nn.Linear(embed_dim, 256 * 1 * 1 * 1)

        # Transposed convolution decoder
        # 1³ -> 2³ -> 4³ -> 8³
        self.decoder = nn.Sequential(
            # Layer 1: 1³ -> 2³
            nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm3d(128),
            nn.GELU(),

            # Layer 2: 2³ -> 4³
            nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm3d(64),
            nn.GELU(),

            # Layer 3: 4³ -> 8³
            nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm3d(32),
            nn.GELU(),

            # Final projection to num_classes
            nn.Conv3d(32, num_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decode tokens to patch masks.

        Args:
            x: Transformed tokens [B*K, embed_dim]

        Returns:
            masks: Patch masks [B*K, num_classes, 8, 8, 8]
        """
        # Project to initial spatial features
        features = self.proj(x)  # [B*K, 256]
        features = features.view(-1, 256, 1, 1, 1)  # [B*K, 256, 1, 1, 1]

        # Decode to spatial mask
        masks = self.decoder(features)  # [B*K, num_classes, 8, 8, 8]

        return masks


class PatchTransformerBranch(nn.Module):
    """
    Token-based local branch for NMSW segmentation.

    Replaces the SegFormer3D local backbone with a transformer-based
    architecture that processes 8³ patches as tokens with cross-attention.

    Pipeline:
    1. Tokenize: 3D CNN encodes each patch to embedding
    2. Position: Add learnable positional embeddings from coordinates
    3. Transform: Cross-patch attention via transformer encoder
    4. Decode: Project tokens back to patch masks

    Args:
        in_channels: Input image channels
        embed_dim: Transformer embedding dimension
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
        mlp_ratio: MLP hidden dimension ratio
        num_classes: Number of segmentation classes
        patch_size: Input patch size (should be 8³)
        dropout: Dropout rate
    """

    def __init__(
        self,
        in_channels: int = 1,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        mlp_ratio: float = 4.0,
        num_classes: int = 1,
        patch_size: Tuple[int, int, int] = (8, 8, 8),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.patch_size = patch_size

        # Patch tokenizer: 3D CNN
        self.tokenizer = PatchTokenizer(
            in_channels=in_channels,
            embed_dim=embed_dim,
            patch_size=patch_size,
        )

        # 3D positional encoding
        self.pos_encoding = LearnablePositionalEncoding3D(
            embed_dim=embed_dim,
        )

        # Cross-patch transformer
        self.transformer = CrossPatchTransformer(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )

        # Patch decoder
        self.decoder = PatchDecoder(
            embed_dim=embed_dim,
            num_classes=num_classes,
            patch_size=patch_size,
        )

    def forward(
        self,
        patches: torch.Tensor,  # [B*K, C, 8, 8, 8]
        slice_meta: List[Tuple[slice, slice, slice]],
        vol_size: Tuple[int, int, int],
        batch_size: int,
    ) -> torch.Tensor:
        """
        Process patches through the token-based pipeline.

        Args:
            patches: Input patches [B*K, C, 8, 8, 8]
            slice_meta: Patch coordinate information (length B*K)
            vol_size: Original volume size for position normalization
            batch_size: Batch size B

        Returns:
            patch_logits: Decoded patch masks [B*K, num_classes, 8, 8, 8]
        """
        BK = patches.shape[0]
        K = BK // batch_size
        B = batch_size
        device = patches.device

        # 1. Tokenize patches: [B*K, C, 8, 8, 8] -> [B*K, embed_dim]
        tokens = self.tokenizer(patches)

        # 2. Reshape for transformer: [B*K, embed_dim] -> [B, K, embed_dim]
        tokens = tokens.view(K, B, self.embed_dim).permute(1, 0, 2)  # [B, K, embed_dim]

        # 3. Add positional encoding
        pos_embed = self.pos_encoding(
            slice_meta=slice_meta,
            vol_size=vol_size,
            batch_size=B,
            num_patches=K,
            device=device,
        )
        tokens = tokens + pos_embed

        # 4. Apply cross-patch transformer: [B, K, embed_dim] -> [B, K, embed_dim]
        tokens = self.transformer(tokens)

        # 5. Reshape back for decoding: [B, K, embed_dim] -> [B*K, embed_dim]
        tokens = tokens.permute(1, 0, 2).reshape(BK, self.embed_dim)

        # 6. Decode to patch masks: [B*K, embed_dim] -> [B*K, num_classes, 8, 8, 8]
        patch_logits = self.decoder(tokens)

        return patch_logits


class NMSWTokenSegFormer3D(nn.Module):
    """
    Token-based NMSW SegFormer3D with Transformer local branch.

    This model replaces the SegFormer3D local backbone with a
    transformer-based architecture that processes small 8³ patches
    as tokens with cross-patch attention.

    Changes from NMSWSegFormer3D:
    - Uses PatchTransformerBranch instead of SegFormer3D for local processing
    - Default patch_size changed from 128³ to 8³
    - Default num_train_patches increased to 64
    - Default num_inference_patches increased to 128

    Args:
        config: Full model configuration dict
        nmsw_config: NMSW-specific configuration
    """

    def __init__(
        self,
        config: Dict,
        nmsw_config: Optional[Dict] = None,
    ):
        super().__init__()

        # Default NMSW config with token-based defaults
        if nmsw_config is None:
            nmsw_config = {}

        # Updated defaults for token-based architecture
        self.down_size_rate = tuple(nmsw_config.get("down_size_rate", [2, 2, 2]))
        self.patch_size = tuple(nmsw_config.get("patch_size", [8, 8, 8]))  # Changed from 128³
        self.overlap = nmsw_config.get("overlap", 0.5)
        self.num_train_patches = nmsw_config.get("num_train_patches", 64)  # Increased
        self.num_train_random_patches = nmsw_config.get("num_train_random_patches", 8)
        self.num_inference_patches = nmsw_config.get("num_inference_patches", 128)  # Increased
        self.tau = nmsw_config.get("starting_tau", 2/3)
        self.global_model_scale = nmsw_config.get("global_model_scale", 0.5)
        self.add_aggregation_module = nmsw_config.get("add_aggregation_module", False)

        # Token transformer config
        self.transformer_embed_dim = nmsw_config.get("transformer_embed_dim", 256)
        self.transformer_num_heads = nmsw_config.get("transformer_num_heads", 8)
        self.transformer_num_layers = nmsw_config.get("transformer_num_layers", 6)
        self.transformer_mlp_ratio = nmsw_config.get("transformer_mlp_ratio", 4.0)
        self.transformer_dropout = nmsw_config.get("transformer_dropout", 0.1)

        # Loss weights
        self.global_loss_weight = nmsw_config.get("global_loss_weight", 1.0)
        self.local_loss_weight = nmsw_config.get("local_loss_weight", 1.0)
        self.agg_loss_weight = nmsw_config.get("agg_loss_weight", 1.0)
        self.entropy_multiplier = nmsw_config.get("entropy_multiplier", 1e-5)

        self.num_classes = config["model_parameters"]["num_classes"]
        self.in_channels = config["model_parameters"]["in_channels"]

        # Global branch: Small SegFormer3D on downsampled input (unchanged)
        self.global_backbone = build_small_segformer3d(config, scale=self.global_model_scale)

        # Downsampler for global branch
        self.downsampler = nn.AvgPool3d(
            kernel_size=self.down_size_rate,
            stride=self.down_size_rate,
        )

        # NEW: Token-based local branch (replaces SegFormer3D)
        self.local_backbone = PatchTransformerBranch(
            in_channels=self.in_channels,
            embed_dim=self.transformer_embed_dim,
            num_heads=self.transformer_num_heads,
            num_layers=self.transformer_num_layers,
            mlp_ratio=self.transformer_mlp_ratio,
            num_classes=self.num_classes,
            patch_size=self.patch_size,
            dropout=self.transformer_dropout,
        )

        # Objectness scoring: Convert global features to patch-level scores
        self.objectness_head = nn.Sequential(
            nn.Conv3d(self.num_classes, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 1, kernel_size=1),
        )

        # Patch sampler with Gumbel top-k
        self.patch_sampler = PatchSampler(
            patch_size=self.patch_size,
            overlap=self.overlap,
            tau=self.tau,
        )

        # Random patch sampler for exploration
        self.random_sampler = RandomPatchSampler(
            patch_size=self.patch_size,
            overlap=self.overlap,
        )

        # Patch extractor for computing objectness grid
        self.patch_extractor = PatchExtractor(
            patch_size=self.patch_size,
            overlap=self.overlap,
        )

        # Aggregator
        self.aggregator = DynamicPatchAggregator(
            patch_size=self.patch_size,
            down_size_rate=self.down_size_rate,
            num_classes=self.num_classes,
            add_aggregation_module=self.add_aggregation_module,
        )

    @property
    def current_tau(self):
        return self.patch_sampler.tau

    @current_tau.setter
    def current_tau(self, value):
        self.patch_sampler.tau = value

    def forward(
        self,
        x: torch.Tensor,  # [B, C, D, H, W]
        labels: Optional[torch.Tensor] = None,  # [B, C, D, H, W]
        mode: str = "train",
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with token-based NMSW patch processing.

        Args:
            x: Input volume [B, C, D, H, W]
            labels: Ground truth labels (for extracting label patches during training)
            mode: 'train' or 'test'

        Returns:
            dict with:
                'final_logit': Final aggregated prediction [B, num_classes, D, H, W]
                'global_logit': Global coarse prediction [B, num_classes, Dg, Hg, Wg]
                'patch_logits': Local patch predictions [K*B, num_classes, 8, 8, 8]
                'sample_probs': Sampling probabilities for entropy regularization
                'label_patches': Label patches for local loss (if labels provided)
        """
        B, C, D, H, W = x.shape
        vol_size = (D, H, W)

        # 1. Global branch: Downsample and get coarse prediction
        x_down = self.downsampler(x)
        global_logit = self.global_backbone(x_down)  # [B, num_classes, Dg, Hg, Wg]

        # 2. Compute objectness scores from global prediction
        global_pred_for_obj = F.interpolate(
            global_logit,
            scale_factor=tuple(d // 2 for d in self.down_size_rate),
            mode='trilinear',
            align_corners=False,
        )
        objectness_logits = self.objectness_head(global_pred_for_obj)

        # Compute patch grid shape for current volume
        _, grid_shape = self.patch_extractor.get_patch_grid(vol_size)

        # Resize objectness to match patch grid
        objectness_logits = F.interpolate(
            objectness_logits,
            size=grid_shape,
            mode='trilinear',
            align_corners=False,
        )

        # 3. Sample patches
        num_patches = self.num_train_patches if mode == "train" else self.num_inference_patches

        sampled = self.patch_sampler(
            volume=x,
            objectness_logits=objectness_logits,
            k=num_patches,
            mode=mode,
            labels=labels,
        )

        selected_patches = sampled['patches']  # [K*B, C, 8, 8, 8]
        slice_meta = sampled['slice_meta']
        sample_probs = sampled['sample_probs']
        label_patches = sampled.get('label_patches')

        # 4. Add random patches during training for exploration
        if mode == "train" and self.num_train_random_patches > 0:
            random_sampled = self.random_sampler(
                volume=x,
                k=self.num_train_random_patches,
                labels=labels,
            )
            selected_patches = torch.cat([selected_patches, random_sampled['patches']], dim=0)
            slice_meta = slice_meta + random_sampled['slice_meta']
            if label_patches is not None and random_sampled['label_patches'] is not None:
                label_patches = torch.cat([label_patches, random_sampled['label_patches']], dim=0)

        # 5. Process patches through token-based local backbone
        # The new local_backbone expects slice_meta for positional encoding
        patch_logits = self.local_backbone(
            patches=selected_patches,  # [K*B, C, 8, 8, 8]
            slice_meta=slice_meta,
            vol_size=vol_size,
            batch_size=B,
        )  # [K*B, num_classes, 8, 8, 8]

        # 6. Aggregate predictions
        final_logit = self.aggregator(
            patch_logits=patch_logits,
            global_logit=global_logit,
            slice_meta=slice_meta,
            vol_size=vol_size,
            batch_size=B,
        )

        return {
            'final_logit': final_logit,
            'global_logit': global_logit,
            'patch_logits': patch_logits,
            'sample_probs': sample_probs,
            'label_patches': label_patches,
            'slice_meta': slice_meta,
        }

    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        criterion: nn.Module,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute multi-component NMSW loss.

        Args:
            outputs: Forward pass outputs
            labels: Ground truth labels [B, C, D, H, W]
            criterion: Loss function (e.g., DiceLoss)

        Returns:
            dict with:
                'total_loss': Combined loss
                'global_loss': Loss on global prediction
                'local_loss': Loss on patch predictions
                'agg_loss': Loss on aggregated prediction
                'entropy': Sampling entropy for regularization
        """
        # 1. Global loss (on downsampled labels)
        labels_down = F.interpolate(
            labels.float(),
            size=outputs['global_logit'].shape[2:],
            mode='nearest',
        )
        global_loss = criterion(outputs['global_logit'], labels_down)

        # 2. Local loss (on patch labels)
        if outputs['label_patches'] is not None:
            local_loss = criterion(outputs['patch_logits'], outputs['label_patches'])
        else:
            local_loss = torch.tensor(0.0, device=labels.device)

        # 3. Aggregation loss
        agg_loss = criterion(outputs['final_logit'], labels)

        # 4. Entropy regularization (encourage diverse sampling)
        sample_probs = outputs['sample_probs']  # [K, B, N]
        entropy = self._compute_entropy(sample_probs)

        # Combine losses
        total_loss = (
            self.agg_loss_weight * agg_loss +
            self.global_loss_weight * global_loss +
            self.local_loss_weight * local_loss -
            self.entropy_multiplier * entropy
        )

        return {
            'total_loss': total_loss,
            'global_loss': global_loss,
            'local_loss': local_loss,
            'agg_loss': agg_loss,
            'entropy': entropy,
        }

    def _compute_entropy(self, probs: torch.Tensor) -> torch.Tensor:
        """Compute entropy of sampling distribution for regularization."""
        eps = 1e-20
        entropy = -(probs + eps) * torch.log(probs + eps)
        return entropy.sum(dim=-1).mean()

    def get_param_groups(self, base_lr: float) -> List[Dict]:
        """Get parameter groups with different learning rates."""
        return [
            {'params': self.global_backbone.parameters(), 'lr': base_lr * 0.1},
            {'params': self.objectness_head.parameters(), 'lr': base_lr},
            {'params': self.local_backbone.parameters(), 'lr': base_lr},
            {'params': self.aggregator.parameters(), 'lr': base_lr},
        ]


class TokenTauScheduler:
    """
    Temperature scheduler for Gumbel-softmax in token-based model.

    Decays temperature from starting_tau to final_tau over
    a portion of training to gradually make selection harder.
    """

    def __init__(
        self,
        model: NMSWTokenSegFormer3D,
        starting_tau: float = 2/3,
        final_tau: float = 2/3,
        decay_epochs: int = 100,
        total_epochs: int = 200,
    ):
        self.model = model
        self.starting_tau = starting_tau
        self.final_tau = final_tau
        self.decay_epochs = decay_epochs
        self.total_epochs = total_epochs

        if decay_epochs > 0 and starting_tau != final_tau:
            self.decay_rate = (starting_tau - final_tau) / decay_epochs
        else:
            self.decay_rate = 0

        self.model.current_tau = starting_tau

    def step(self, epoch: int):
        """Update tau based on current epoch."""
        if epoch < self.decay_epochs:
            new_tau = self.starting_tau - self.decay_rate * epoch
        else:
            new_tau = self.final_tau

        self.model.current_tau = max(new_tau, self.final_tau)

    def get_tau(self) -> float:
        return self.model.current_tau
