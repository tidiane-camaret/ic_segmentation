"""
NMSW-style Patch Sampling Module.

Implements differentiable top-k patch selection using Gumbel-softmax
for SegFormer3D training and inference.

Reference: https://arxiv.org/pdf/1611.01144 (Gumbel-Softmax)
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.data.utils import compute_importance_map


class GumbelTopK(nn.Module):
    """Differentiable top-k selection using Gumbel-softmax.

    Uses the Gumbel-softmax trick to enable gradient flow through
    discrete patch selection during training.

    Args:
        tau: Temperature parameter controlling softness of selection.
             Lower tau -> harder selection, higher tau -> softer selection.
    """

    epsilon = np.finfo(np.float32).tiny
    epsilon_test = 1e-200

    def __init__(self, tau: float = 2/3):
        super().__init__()
        self.tau = tau
        self.register_buffer("epsilon_tensor", torch.tensor([self.epsilon]))
        self.register_buffer("epsilon_tensor_test", torch.tensor([self.epsilon_test]))

    def forward(
        self,
        logits: torch.Tensor,  # [B, N] where N is number of patches
        k: int,
        mode: str = "train",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select top-k patches using differentiable Gumbel-softmax.

        Args:
            logits: Patch scores of shape [B, N]
            k: Number of patches to select
            mode: 'train' or 'test'. Train uses Gumbel noise, test is deterministic.

        Returns:
            one_hots: Hard one-hot selections with straight-through gradient [K, B, N]
            soft_hots: Soft probability distributions [K, B, N]
        """
        device = logits.device
        self.cur_device = device

        # Flatten logits if needed
        flatten_logits = logits.flatten(1)
        log_p = flatten_logits

        # Sample soft top-k iteratively
        soft_one_hots = []
        onehot_approx = torch.zeros_like(log_p, device=device)

        # Add Gumbel noise for stochastic selection during training
        if mode == "train":
            noisy_logits = log_p + self._get_gumbel_noise(log_p.shape)
        else:
            noisy_logits = log_p

        for i in range(k):
            if i == 0:
                mask = torch.max(
                    1.0 - onehot_approx,
                    self.epsilon_tensor if mode == "train" else self.epsilon_tensor_test,
                )
            else:
                # Create hard one-hot from previous soft selection
                hard_one_hot = torch.zeros_like(onehot_approx, device=device)
                _, ind = torch.topk(soft_one_hots[i - 1], 1, dim=1)
                hard_one_hot.scatter_(1, ind, 1)
                mask = torch.max(
                    1.0 - hard_one_hot,
                    self.epsilon_tensor if mode == "train" else self.epsilon_tensor_test,
                )

            # Mask out previously selected patches
            noisy_logits = noisy_logits + torch.log(mask)

            # Softmax with temperature
            onehot_approx = F.softmax(noisy_logits / self.tau, dim=1)
            soft_one_hots.append(onehot_approx)

        # Straight-through estimator for hard selection with soft gradients
        st_one_hots = []
        for i in range(k):
            hard_one_hot = torch.zeros_like(soft_one_hots[i], device=device)
            _, ind = torch.topk(soft_one_hots[i], 1, dim=1)
            hard_one_hot.scatter_(1, ind, 1)
            # Straight-through: hard forward, soft backward
            st_one_hots.append(
                (hard_one_hot - soft_one_hots[i]).detach() + soft_one_hots[i]
            )

        return (
            torch.stack(st_one_hots, dim=0),
            torch.stack(soft_one_hots, dim=0),
        )

    def _get_gumbel_noise(self, shape: Tuple) -> torch.Tensor:
        """Generate Gumbel noise for stochastic sampling."""
        return -torch.log(
            -torch.log(torch.rand(shape, device=self.cur_device) + 1e-20) + 1e-20
        )


class PatchExtractor(nn.Module):
    """Extract overlapping patches from a volume.

    Uses MONAI-style sliding window approach to extract patches
    with configurable overlap.
    """

    def __init__(
        self,
        patch_size: Tuple[int, int, int],
        overlap: float = 0.5,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.overlap = overlap

    def get_patch_grid(
        self,
        volume_shape: Tuple[int, int, int],
    ) -> Tuple[List[Tuple[slice, slice, slice]], Tuple[int, int, int]]:
        """
        Compute grid of patch slices for a given volume shape.

        Args:
            volume_shape: (D, H, W) spatial dimensions

        Returns:
            slices: List of (slice_d, slice_h, slice_w) for each patch
            grid_shape: (num_d, num_h, num_w) number of patches per dimension
        """
        slices = []
        grid_shape = []

        for dim_size, patch_dim in zip(volume_shape, self.patch_size):
            step = int(patch_dim * (1 - self.overlap))
            step = max(step, 1)  # Ensure at least 1 step

            starts = list(range(0, dim_size - patch_dim + 1, step))
            if not starts or starts[-1] + patch_dim < dim_size:
                starts.append(max(0, dim_size - patch_dim))

            grid_shape.append(len(starts))

        grid_shape = tuple(grid_shape)

        # Generate all patch slices
        d_steps = int(volume_shape[0] - self.patch_size[0] + 1)
        h_steps = int(volume_shape[1] - self.patch_size[1] + 1)
        w_steps = int(volume_shape[2] - self.patch_size[2] + 1)

        d_step = max(1, int(self.patch_size[0] * (1 - self.overlap)))
        h_step = max(1, int(self.patch_size[1] * (1 - self.overlap)))
        w_step = max(1, int(self.patch_size[2] * (1 - self.overlap)))

        for d in range(0, max(1, d_steps), d_step):
            d = min(d, volume_shape[0] - self.patch_size[0])
            for h in range(0, max(1, h_steps), h_step):
                h = min(h, volume_shape[1] - self.patch_size[1])
                for w in range(0, max(1, w_steps), w_step):
                    w = min(w, volume_shape[2] - self.patch_size[2])
                    slices.append((
                        slice(d, d + self.patch_size[0]),
                        slice(h, h + self.patch_size[1]),
                        slice(w, w + self.patch_size[2]),
                    ))

        return slices, grid_shape

    def extract_patches(
        self,
        volume: torch.Tensor,  # [B, C, D, H, W]
    ) -> Tuple[torch.Tensor, List[Tuple[slice, slice, slice]]]:
        """
        Extract all patches from volume.

        Args:
            volume: Input volume [B, C, D, H, W]

        Returns:
            patches: All patches [B*N, C, Pd, Ph, Pw] where N is number of patches
            slice_meta: List of slice tuples for each patch
        """
        B, C, D, H, W = volume.shape
        slices, grid_shape = self.get_patch_grid((D, H, W))

        patches = []
        for s in slices:
            patch = volume[:, :, s[0], s[1], s[2]]  # [B, C, Pd, Ph, Pw]
            patches.append(patch)

        # Stack: [N, B, C, Pd, Ph, Pw] -> [B*N, C, Pd, Ph, Pw]
        patches = torch.stack(patches, dim=1)  # [B, N, C, Pd, Ph, Pw]
        patches = patches.view(-1, C, *self.patch_size)  # [B*N, C, Pd, Ph, Pw]

        return patches, slices


class PatchSampler(nn.Module):
    """
    Sample top-k patches based on objectness scores.

    Combines patch extraction with Gumbel top-k selection to
    enable differentiable patch sampling during training.
    """

    def __init__(
        self,
        patch_size: Tuple[int, int, int],
        overlap: float = 0.5,
        tau: float = 2/3,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.overlap = overlap

        self.patch_extractor = PatchExtractor(patch_size, overlap)
        self.gumbel_topk = GumbelTopK(tau=tau)

        self.register_buffer(
            "background_epsilon",
            torch.tensor([self.gumbel_topk.epsilon])
        )

    @property
    def tau(self):
        return self.gumbel_topk.tau

    @tau.setter
    def tau(self, value):
        self.gumbel_topk.tau = value

    def forward(
        self,
        volume: torch.Tensor,  # [B, C, D, H, W]
        objectness_logits: torch.Tensor,  # [B, 1, Nd, Nh, Nw] patch-level scores
        k: int,
        mode: str = "train",
        background_mask: Optional[torch.Tensor] = None,  # [B, 1, Nd, Nh, Nw]
        labels: Optional[torch.Tensor] = None,  # [B, C, D, H, W] for extracting label patches
    ) -> Dict[str, torch.Tensor]:
        """
        Sample top-k patches based on objectness scores.

        Args:
            volume: Input volume [B, C, D, H, W]
            objectness_logits: Patch-level scores [B, 1, Nd, Nh, Nw]
            k: Number of patches to select
            mode: 'train' or 'test'
            background_mask: Optional mask to prevent sampling from background
            labels: Optional labels to extract corresponding label patches

        Returns:
            dict with:
                'patches': Selected patches [K*B, C, Pd, Ph, Pw]
                'label_patches': Selected label patches (if labels provided)
                'slice_meta': Slice indices for selected patches
                'sample_probs': Sampling probabilities for entropy regularization
        """
        B, C, D, H, W = volume.shape
        device = volume.device

        # Extract all patches
        all_patches, slice_meta = self.patch_extractor.extract_patches(volume)
        N = len(slice_meta)  # Total number of patches

        if labels is not None:
            all_label_patches, _ = self.patch_extractor.extract_patches(labels)
        else:
            all_label_patches = None

        # Apply background mask
        if background_mask is not None:
            log_mask = torch.log(
                torch.max(1.0 - background_mask, self.background_epsilon)
            )
            masked_logits = objectness_logits + log_mask
        else:
            masked_logits = objectness_logits

        # Flatten for Gumbel top-k - objectness_logits should have N elements per batch
        masked_logits_flat = masked_logits.flatten(1)  # [B, M] where M is objectness grid size

        # If objectness grid size doesn't match patch count, resize
        M = masked_logits_flat.shape[1]
        if M != N:
            # Interpolate to match patch count
            masked_logits_flat = F.interpolate(
                masked_logits_flat.unsqueeze(1),  # [B, 1, M]
                size=N,
                mode='linear',
                align_corners=False,
            ).squeeze(1)  # [B, N]

        # Get one-hot selections
        one_hots, soft_hots = self.gumbel_topk(masked_logits_flat, k=k, mode=mode)
        # one_hots: [K, B, N], soft_hots: [K, B, N]

        # Get selected patch indices
        patch_indices = one_hots.detach().argmax(dim=-1)  # [K, B]

        # Reshape all patches for batch processing
        all_patches_batched = all_patches.view(B, N, C, *self.patch_size)  # [B, N, C, Pd, Ph, Pw]

        # Select patches using index-based approach (simpler and more reliable)
        selected_patches_list = []
        selected_label_patches_list = []
        selected_slices = []

        for ki in range(k):
            for bi in range(B):
                idx = patch_indices[ki, bi].item()
                selected_patches_list.append(all_patches_batched[bi, idx])  # [C, Pd, Ph, Pw]
                selected_slices.append(slice_meta[idx])
                if all_label_patches is not None:
                    all_label_patches_batched = all_label_patches.view(B, N, -1, *self.patch_size)
                    selected_label_patches_list.append(all_label_patches_batched[bi, idx])

        selected_patches = torch.stack(selected_patches_list, dim=0)  # [K*B, C, Pd, Ph, Pw]

        if all_label_patches is not None:
            selected_label_patches = torch.stack(selected_label_patches_list, dim=0)
        else:
            selected_label_patches = None

        return {
            'patches': selected_patches,  # [K*B, C, Pd, Ph, Pw]
            'label_patches': selected_label_patches,  # [K*B, LC, Pd, Ph, Pw] or None
            'slice_meta': selected_slices,
            'sample_probs': soft_hots,  # For entropy regularization
            'patch_indices': patch_indices,  # [K, B]
            'num_patches': N,
        }


class RandomPatchSampler(nn.Module):
    """
    Random patch sampler for exploration during training.

    Samples patches uniformly at random to encourage exploration
    of the full volume during training.
    """

    def __init__(
        self,
        patch_size: Tuple[int, int, int],
        overlap: float = 0.5,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.patch_extractor = PatchExtractor(patch_size, overlap)

    def forward(
        self,
        volume: torch.Tensor,  # [B, C, D, H, W]
        k: int,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Sample k random patches.

        Args:
            volume: Input volume [B, C, D, H, W]
            k: Number of patches to sample
            labels: Optional labels

        Returns:
            dict with patches, label_patches, slice_meta
        """
        B, C, D, H, W = volume.shape

        # Extract all patches
        all_patches, slice_meta = self.patch_extractor.extract_patches(volume)
        N = len(slice_meta)

        if labels is not None:
            all_label_patches, _ = self.patch_extractor.extract_patches(labels)
        else:
            all_label_patches = None

        # Random selection
        rand_indices = torch.randint(0, N, (k, B), device=volume.device)

        # Gather patches
        all_patches_batched = all_patches.view(B, N, C, *self.patch_size)
        selected_patches = []
        selected_label_patches = [] if labels is not None else None
        selected_slices = []

        for ki in range(k):
            for bi in range(B):
                idx = rand_indices[ki, bi].item()
                selected_patches.append(all_patches_batched[bi, idx])
                if labels is not None:
                    all_label_patches_batched = all_label_patches.view(B, N, -1, *self.patch_size)
                    selected_label_patches.append(all_label_patches_batched[bi, idx])
                selected_slices.append(slice_meta[idx])

        selected_patches = torch.stack(selected_patches, dim=0)
        if selected_label_patches is not None:
            selected_label_patches = torch.stack(selected_label_patches, dim=0)

        return {
            'patches': selected_patches,
            'label_patches': selected_label_patches,
            'slice_meta': selected_slices,
            'patch_indices': rand_indices,
        }
