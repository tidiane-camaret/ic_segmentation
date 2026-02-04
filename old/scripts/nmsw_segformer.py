"""
NMSW-Enhanced SegFormer3D Model.

Combines global coarse prediction with local patch refinement
using differentiable top-k patch sampling.

Architecture:
1. Global Branch: Small SegFormer3D on downsampled input
2. Objectness Scoring: Predict which patches are important
3. Patch Sampling: Select top-k patches using Gumbel-softmax
4. Local Branch: Full SegFormer3D on selected patches
5. Aggregation: Combine global + local predictions
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Add SegFormer3D to path
sys.path.insert(0, "/software/notebooks/camaret/repos/SegFormer3D")
from architectures.segformer3d import SegFormer3D, build_segformer3d_model

from .nmsw_sampling import PatchSampler, RandomPatchSampler, PatchExtractor
from .nmsw_aggregation import DynamicPatchAggregator


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


class NMSWSegFormer3D(nn.Module):
    """
    SegFormer3D with NMSW-style patch sampling.

    Instead of processing all patches during sliding window inference,
    this model:
    1. Generates a coarse prediction using a lightweight global branch
    2. Scores patches by foreground probability (objectness)
    3. Selects top-k patches using differentiable Gumbel-softmax
    4. Processes selected patches with full SegFormer3D
    5. Aggregates results for final prediction

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

        # Default NMSW config
        if nmsw_config is None:
            nmsw_config = {}

        self.down_size_rate = tuple(nmsw_config.get("down_size_rate", [2, 2, 2]))
        self.patch_size = tuple(nmsw_config.get("patch_size", [128, 128, 128]))
        self.overlap = nmsw_config.get("overlap", 0.5)
        self.num_train_patches = nmsw_config.get("num_train_patches", 4)
        self.num_train_random_patches = nmsw_config.get("num_train_random_patches", 1)
        self.num_inference_patches = nmsw_config.get("num_inference_patches", 8)
        self.tau = nmsw_config.get("starting_tau", 2/3)
        self.global_model_scale = nmsw_config.get("global_model_scale", 0.5)
        self.add_aggregation_module = nmsw_config.get("add_aggregation_module", False)

        # Loss weights
        self.global_loss_weight = nmsw_config.get("global_loss_weight", 1.0)
        self.local_loss_weight = nmsw_config.get("local_loss_weight", 1.0)
        self.agg_loss_weight = nmsw_config.get("agg_loss_weight", 1.0)
        self.entropy_multiplier = nmsw_config.get("entropy_multiplier", 1e-5)

        self.num_classes = config["model_parameters"]["num_classes"]

        # Global branch: Small SegFormer3D on downsampled input
        self.global_backbone = build_small_segformer3d(config, scale=self.global_model_scale)

        # Downsampler for global branch
        self.downsampler = nn.AvgPool3d(
            kernel_size=self.down_size_rate,
            stride=self.down_size_rate,
        )

        # Local branch: Full SegFormer3D for patch processing
        self.local_backbone = build_segformer3d_model(config)

        # Objectness scoring: Convert global features to patch-level scores
        # Input: global prediction, Output: objectness logits
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
        Forward pass with NMSW patch sampling.

        Args:
            x: Input volume [B, C, D, H, W]
            labels: Ground truth labels (for extracting label patches during training)
            mode: 'train' or 'test'

        Returns:
            dict with:
                'final_logit': Final aggregated prediction [B, num_classes, D, H, W]
                'global_logit': Global coarse prediction [B, num_classes, Dg, Hg, Wg]
                'patch_logits': Local patch predictions [K*B, num_classes, Pd, Ph, Pw]
                'sample_probs': Sampling probabilities for entropy regularization
                'label_patches': Label patches for local loss (if labels provided)
        """
        B, C, D, H, W = x.shape
        vol_size = (D, H, W)

        # 1. Global branch: Downsample and get coarse prediction
        x_down = self.downsampler(x)
        global_logit = self.global_backbone(x_down)  # [B, num_classes, Dg, Hg, Wg]

        # 2. Compute objectness scores from global prediction
        # Upsample global prediction slightly for objectness scoring
        global_pred_for_obj = F.interpolate(
            global_logit,
            scale_factor=tuple(d // 2 for d in self.down_size_rate),
            mode='trilinear',
            align_corners=False,
        )
        objectness_logits = self.objectness_head(global_pred_for_obj)  # [B, 1, No, No, No]

        # Compute patch grid shape for current volume
        _, grid_shape = self.patch_extractor.get_patch_grid(vol_size)

        # Resize objectness to match patch grid
        objectness_logits = F.interpolate(
            objectness_logits,
            size=grid_shape,
            mode='trilinear',
            align_corners=False,
        )  # [B, 1, Nd, Nh, Nw]

        # 3. Sample patches
        num_patches = self.num_train_patches if mode == "train" else self.num_inference_patches

        sampled = self.patch_sampler(
            volume=x,
            objectness_logits=objectness_logits,
            k=num_patches,
            mode=mode,
            labels=labels,
        )

        selected_patches = sampled['patches']  # [K*B, C, Pd, Ph, Pw]
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

        # 5. Process patches through local backbone
        patch_logits = self.local_backbone(selected_patches)  # [K*B, num_classes, Pd, Ph, Pw]

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
        B = labels.shape[0]
        vol_size = labels.shape[2:]

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
        # probs: [K, B, N]
        # Add small epsilon to avoid log(0)
        eps = 1e-20
        entropy = -(probs + eps) * torch.log(probs + eps)
        # Sum over patches, mean over K and B
        return entropy.sum(dim=-1).mean()

    def get_param_groups(self, base_lr: float) -> List[Dict]:
        """Get parameter groups with different learning rates.

        Global branch uses lower learning rate since it trains faster.
        """
        return [
            {'params': self.global_backbone.parameters(), 'lr': base_lr * 0.1},
            {'params': self.objectness_head.parameters(), 'lr': base_lr},
            {'params': self.local_backbone.parameters(), 'lr': base_lr},
            {'params': self.aggregator.parameters(), 'lr': base_lr},
        ]


class TauScheduler:
    """
    Temperature scheduler for Gumbel-softmax.

    Decays temperature from starting_tau to final_tau over
    a portion of training to gradually make selection harder.
    """

    def __init__(
        self,
        model: NMSWSegFormer3D,
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

        # Compute decay rate
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
