"""
Patch sampling strategies for PatchICL.

Provides modular patch selection mechanisms that can be swapped or extended.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchSampler(nn.Module):
    """
    Samples K patches from an image based on a weight map.

    Uses temperature-scaled softmax + multinomial sampling.
    Non-differentiable through the sampling operation.
    """

    def __init__(
        self,
        patch_size: int,
        num_patches: int,
        temperature: float = 0.3,
        exploration_noise: float = 0.5,
        stride_divisor: int = 4,
    ):
        """
        Args:
            patch_size: Size of sampled patches (ps x ps)
            num_patches: Number of patches to sample (K)
            temperature: Softmax temperature (lower = sharper distribution)
            exploration_noise: Noise magnitude during training (0 = no noise)
            stride_divisor: patch_size // stride_divisor = stride between candidates
        """
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.temperature = temperature
        self.exploration_noise = exploration_noise
        self.stride_divisor = stride_divisor

    def compute_scores(
        self,
        weights: torch.Tensor,
        grid_h: int,
        grid_w: int,
    ) -> torch.Tensor:
        """
        Compute patch scores from weight map.

        Args:
            weights: [B, 1, H, W] - weight map for sampling
            grid_h: Number of candidate rows
            grid_w: Number of candidate columns

        Returns:
            scores: [B, grid_h * grid_w] - flat scores per candidate
        """
        ps = self.patch_size
        stride = max(1, ps // self.stride_divisor)

        # Avg pool to get per-patch scores
        scores_map = F.avg_pool2d(weights, kernel_size=ps, stride=stride, padding=0)

        # Resize to expected grid if needed
        if scores_map.shape[-2:] != (grid_h, grid_w):
            scores_map = F.interpolate(
                scores_map, size=(grid_h, grid_w), mode='bilinear', align_corners=False
            )

        # Flatten: [B, N_candidates]
        return scores_map.flatten(2).squeeze(1)

    def sample_indices(
        self,
        scores: torch.Tensor,
        k: int,
    ) -> torch.Tensor:
        """
        Sample k patch indices from scores using multinomial sampling.

        Args:
            scores: [B, N] - raw scores per candidate
            k: Number of patches to sample

        Returns:
            indices: [B, k] - selected candidate indices
        """
        # Normalize scores to [0, 1]
        score_min = scores.min(dim=1, keepdim=True)[0]
        score_max = scores.max(dim=1, keepdim=True)[0]
        score_range = (score_max - score_min).clamp(min=1e-6)
        normalized = (scores - score_min) / score_range

        # Apply temperature scaling
        scaled = normalized / self.temperature

        # Add exploration noise during training
        if self.training and self.exploration_noise > 0:
            noise = torch.rand_like(scaled) * self.exploration_noise
            scaled = scaled + noise

        # Convert to probabilities
        probs = F.softmax(scaled, dim=1)

        # Sample without replacement
        k_safe = min(k, scores.shape[1])
        indices = torch.multinomial(probs, k_safe, replacement=False)

        return indices

    def indices_to_coords(
        self,
        indices: torch.Tensor,
        grid_w: int,
        stride: int,
        max_h: int,
        max_w: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Convert flat indices to (h, w) coordinates.

        Args:
            indices: [B, k] - selected indices
            grid_w: Width of candidate grid
            stride: Stride between candidates
            max_h: Maximum valid h coordinate (H - patch_size)
            max_w: Maximum valid w coordinate (W - patch_size)

        Returns:
            h_coords: [B, k] - row coordinates
            w_coords: [B, k] - column coordinates
        """
        row_indices = indices // grid_w
        col_indices = indices % grid_w

        h_coords = (row_indices * stride).clamp(0, max_h)
        w_coords = (col_indices * stride).clamp(0, max_w)

        return h_coords, w_coords

    def extract_patches(
        self,
        image: torch.Tensor,
        labels: torch.Tensor,
        h_coords: torch.Tensor,
        w_coords: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract patches at given coordinates.

        Args:
            image: [B, C, H, W] - source image
            labels: [B, 1, H, W] - source labels
            h_coords: [B, k] - row coordinates
            w_coords: [B, k] - column coordinates

        Returns:
            patches: [B, K, C, ps, ps]
            patch_labels: [B, K, 1, ps, ps]
            coords: [B, K, 2] - (h, w) coordinate pairs
        """
        B, C, _, _ = image.shape
        ps = self.patch_size
        K = self.num_patches
        k = h_coords.shape[1]
        device = image.device

        all_patches = []
        all_labels = []
        all_coords = []

        for b in range(B):
            batch_patches = []
            batch_labels = []
            batch_coords = []

            for i in range(k):
                h = int(h_coords[b, i].item())
                w = int(w_coords[b, i].item())

                patch = image[b, :, h:h+ps, w:w+ps]
                label_patch = labels[b, :, h:h+ps, w:w+ps]

                batch_patches.append(patch)
                batch_labels.append(label_patch)
                batch_coords.append([h, w])

            # Pad if fewer patches than requested
            while len(batch_patches) < K:
                batch_patches.append(torch.zeros(C, ps, ps, device=device))
                batch_labels.append(torch.zeros(1, ps, ps, device=device))
                batch_coords.append([0, 0])

            all_patches.append(torch.stack(batch_patches))
            all_labels.append(torch.stack(batch_labels))
            all_coords.append(torch.tensor(batch_coords, device=device))

        return (
            torch.stack(all_patches),   # [B, K, C, ps, ps]
            torch.stack(all_labels),    # [B, K, 1, ps, ps]
            torch.stack(all_coords),    # [B, K, 2]
        )

    def forward(
        self,
        image: torch.Tensor,
        labels: torch.Tensor,
        weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample K patches from image based on weight map.

        Args:
            image: [B, C, H, W] - source image
            labels: [B, 1, H, W] - ground truth mask
            weights: [B, 1, H, W] - weight map for sampling

        Returns:
            patches: [B, K, C, ps, ps]
            patch_labels: [B, K, 1, ps, ps]
            coords: [B, K, 2] - (h, w) coordinates
        """
        _, _, H, W = image.shape
        ps = self.patch_size
        K = self.num_patches
        stride = max(1, ps // self.stride_divisor)

        # Compute grid dimensions
        grid_h = max(1, (H - ps) // stride + 1)
        grid_w = max(1, (W - ps) // stride + 1)

        # Compute scores
        scores = self.compute_scores(weights, grid_h, grid_w)

        # Sample indices
        indices = self.sample_indices(scores, K)

        # Convert to coordinates
        h_coords, w_coords = self.indices_to_coords(
            indices, grid_w, stride,
            max_h=H - ps, max_w=W - ps,
        )

        # Extract patches
        patches, patch_labels, coords = self.extract_patches(
            image, labels, h_coords, w_coords,
        )

        return patches, patch_labels, coords


class UniformSampler(PatchSampler):
    """
    Samples patches uniformly at random (ignores weights).

    Useful for first level when oracle is disabled.
    """

    def compute_scores(
        self,
        weights: torch.Tensor,
        grid_h: int,
        grid_w: int,
    ) -> torch.Tensor:
        """Return uniform scores (ignore weight map)."""
        B = weights.shape[0]
        device = weights.device
        num_candidates = grid_h * grid_w
        return torch.ones(B, num_candidates, device=device)


class DeterministicTopKSampler(PatchSampler):
    """
    Selects top-K patches deterministically (no stochasticity).

    Useful for inference when reproducibility is needed.
    """

    def sample_indices(
        self,
        scores: torch.Tensor,
        k: int,
    ) -> torch.Tensor:
        """Select top-k indices by score (deterministic)."""
        k_safe = min(k, scores.shape[1])
        _, indices = torch.topk(scores, k_safe, dim=1)
        return indices


class GumbelSoftmaxSampler(PatchSampler):
    """
    Differentiable patch sampling using Gumbel-Softmax.

    Uses the Gumbel-Softmax reparameterization trick to allow gradients
    to flow through the discrete patch selection process.

    For K patches, samples sequentially with masking to prevent re-selection.
    Uses straight-through estimator: hard selection in forward, soft gradients in backward.
    """

    def __init__(
        self,
        patch_size: int,
        num_patches: int,
        temperature: float = 0.3,
        tau: float = 1.0,
        tau_min: float = 0.1,
        hard: bool = True,
        stride_divisor: int = 4,
    ):
        """
        Args:
            patch_size: Size of sampled patches (ps x ps)
            num_patches: Number of patches to sample (K)
            temperature: Score scaling temperature
            tau: Gumbel-Softmax temperature (higher = softer, lower = harder)
            tau_min: Minimum tau for annealing
            hard: If True, use straight-through estimator (hard forward, soft backward)
            stride_divisor: patch_size // stride_divisor = stride between candidates
        """
        super().__init__(
            patch_size=patch_size,
            num_patches=num_patches,
            temperature=temperature,
            exploration_noise=0.0,  # Gumbel noise replaces exploration noise
            stride_divisor=stride_divisor,
        )
        self.tau = tau
        self.tau_min = tau_min
        self.hard = hard

    def set_tau(self, tau: float):
        """Update Gumbel temperature (for annealing during training)."""
        self.tau = max(tau, self.tau_min)

    def _gumbel_softmax_sample(
        self,
        logits: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Single Gumbel-Softmax sample with masking.

        Args:
            logits: [B, N] - unnormalized log probabilities
            mask: [B, N] - 1 for already selected, 0 for available

        Returns:
            sample: [B, N] - soft or hard one-hot vector
        """
        # Mask out already selected indices
        masked_logits = logits - 1e9 * mask

        # Sample Gumbel noise
        u = torch.rand_like(masked_logits).clamp(1e-8, 1 - 1e-8)
        gumbel_noise = -torch.log(-torch.log(u))

        # Gumbel-Softmax
        y_soft = F.softmax((masked_logits + gumbel_noise) / self.tau, dim=1)

        if self.hard:
            # Straight-through estimator: hard forward, soft backward
            idx = y_soft.argmax(dim=1, keepdim=True)
            y_hard = torch.zeros_like(y_soft).scatter_(1, idx, 1.0)
            # Gradient flows through y_soft
            return y_hard - y_soft.detach() + y_soft
        else:
            return y_soft

    def sample_soft_indices(
        self,
        scores: torch.Tensor,
        k: int,
    ) -> torch.Tensor:
        """
        Sample K patches using Gumbel-Softmax (differentiable).

        Args:
            scores: [B, N] - raw scores per candidate
            k: Number of patches to sample

        Returns:
            soft_indices: [B, K, N] - soft one-hot selection weights
        """
        B, N = scores.shape
        device = scores.device
        k_safe = min(k, N)

        # Normalize scores to logits
        score_min = scores.min(dim=1, keepdim=True)[0]
        score_max = scores.max(dim=1, keepdim=True)[0]
        score_range = (score_max - score_min).clamp(min=1e-6)
        normalized = (scores - score_min) / score_range
        logits = normalized / self.temperature

        # Sample K patches sequentially with masking
        samples = []
        mask = torch.zeros(B, N, device=device)

        for _ in range(k_safe):
            sample = self._gumbel_softmax_sample(logits, mask)
            samples.append(sample)
            # Update mask (detach to not backprop through mask updates)
            mask = mask + sample.detach()

        return torch.stack(samples, dim=1)  # [B, K, N]

    def extract_patches_soft(
        self,
        image: torch.Tensor,
        labels: torch.Tensor,
        soft_indices: torch.Tensor,
        grid_h: int,  # noqa: ARG002 - kept for API clarity
        grid_w: int,
        stride: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract patches using soft attention weights (differentiable).

        Args:
            image: [B, C, H, W] - source image
            labels: [B, 1, H, W] - source labels
            soft_indices: [B, K, N] - soft one-hot selection weights
            grid_h, grid_w: Grid dimensions
            stride: Stride between candidates

        Returns:
            patches: [B, K, C, ps, ps]
            patch_labels: [B, K, 1, ps, ps]
            coords: [B, K, 2] - (h, w) coordinates (from hard selection)
        """
        del grid_h  # Unused, N is derived from soft_indices
        B, C, H, W = image.shape
        K = soft_indices.shape[1]
        N = soft_indices.shape[2]
        ps = self.patch_size
        device = image.device

        # Extract ALL candidate patches first
        # Using unfold for efficient patch extraction
        # image: [B, C, H, W] -> patches: [B, C, grid_h, grid_w, ps, ps]
        patches_unfolded = image.unfold(2, ps, stride).unfold(3, ps, stride)
        # Reshape to [B, C, N, ps, ps] where N = grid_h * grid_w
        all_patches = patches_unfolded.reshape(B, C, -1, ps, ps)

        # Same for labels
        labels_unfolded = labels.unfold(2, ps, stride).unfold(3, ps, stride)
        all_labels = labels_unfolded.reshape(B, 1, -1, ps, ps)

        # Handle case where unfold gives fewer patches than expected
        actual_N = all_patches.shape[2]
        if actual_N < N:
            # Pad with zeros
            pad_patches = torch.zeros(B, C, N - actual_N, ps, ps, device=device)
            all_patches = torch.cat([all_patches, pad_patches], dim=2)
            pad_labels = torch.zeros(B, 1, N - actual_N, ps, ps, device=device)
            all_labels = torch.cat([all_labels, pad_labels], dim=2)
        elif actual_N > N:
            all_patches = all_patches[:, :, :N]
            all_labels = all_labels[:, :, :N]

        # Apply soft attention: [B, K, N] @ [B, N, C*ps*ps] -> [B, K, C*ps*ps]
        # Reshape patches for matmul
        all_patches_flat = all_patches.permute(0, 2, 1, 3, 4).reshape(B, N, C * ps * ps)
        all_labels_flat = all_labels.permute(0, 2, 1, 3, 4).reshape(B, N, ps * ps)

        # Weighted combination (differentiable)
        selected_patches_flat = torch.bmm(soft_indices, all_patches_flat)  # [B, K, C*ps*ps]
        selected_labels_flat = torch.bmm(soft_indices, all_labels_flat)    # [B, K, ps*ps]

        # Reshape back
        patches = selected_patches_flat.reshape(B, K, C, ps, ps)
        patch_labels = selected_labels_flat.reshape(B, K, 1, ps, ps)

        # Get hard coordinates for position encoding (non-differentiable, for reference)
        hard_indices = soft_indices.argmax(dim=2)  # [B, K]
        row_indices = hard_indices // grid_w
        col_indices = hard_indices % grid_w
        h_coords = (row_indices * stride).clamp(0, H - ps)
        w_coords = (col_indices * stride).clamp(0, W - ps)
        coords = torch.stack([h_coords, w_coords], dim=2)  # [B, K, 2]

        return patches, patch_labels, coords

    def forward(
        self,
        image: torch.Tensor,
        labels: torch.Tensor,
        weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample K patches using differentiable Gumbel-Softmax.

        Args:
            image: [B, C, H, W] - source image
            labels: [B, 1, H, W] - ground truth mask
            weights: [B, 1, H, W] - weight map for sampling

        Returns:
            patches: [B, K, C, ps, ps]
            patch_labels: [B, K, 1, ps, ps]
            coords: [B, K, 2] - (h, w) coordinates
        """
        _, _, H, W = image.shape
        ps = self.patch_size
        K = self.num_patches
        stride = max(1, ps // self.stride_divisor)

        # Compute grid dimensions
        grid_h = max(1, (H - ps) // stride + 1)
        grid_w = max(1, (W - ps) // stride + 1)

        # Compute scores
        scores = self.compute_scores(weights, grid_h, grid_w)

        # Sample soft indices (differentiable)
        soft_indices = self.sample_soft_indices(scores, K)  # [B, K, N]

        # Extract patches using soft attention
        patches, patch_labels, coords = self.extract_patches_soft(
            image, labels, soft_indices, grid_h, grid_w, stride,
        )

        return patches, patch_labels, coords
