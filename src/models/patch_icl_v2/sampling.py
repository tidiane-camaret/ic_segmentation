"""
Simplified patch sampling for PatchICL v2.

Only includes ContinuousSampler and SlidingWindowSampler - the two strategies
actually used by v1 and v2 experiment configs.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchAugmenter(nn.Module):
    """Augments patches with rotation, flipping, and scaling."""

    def __init__(
        self,
        rotation: str = "none",  # "none", "90", "continuous"
        rotation_range: float = 0.5,
        flip_horizontal: bool = False,
        flip_vertical: bool = False,
        scale_range: tuple[float, float] | None = None,
    ):
        super().__init__()
        self.rotation = rotation
        self.rotation_range = rotation_range
        self.flip_horizontal = flip_horizontal
        self.flip_vertical = flip_vertical
        self.scale_range = scale_range

    def _rotate_90(self, patches: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """Rotate patches by k * 90 degrees. patches: [B, K, C, H, W], k: [B, K]"""
        B, K, _, _, _ = patches.shape
        result = patches.clone()
        for b in range(B):
            for i in range(K):
                ki = k[b, i].item()
                if ki > 0:
                    result[b, i] = torch.rot90(patches[b, i], int(ki), dims=(1, 2))
        return result

    def _rotate_features_90(self, features: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """Rotate feature patches by k * 90 degrees. features: [B, K, tokens, D]"""
        B, K, T, D = features.shape
        h = w = int(T ** 0.5)
        assert h * w == T, f"tokens must form a square grid, got {T}"
        spatial = features.view(B, K, h, w, D).permute(0, 1, 4, 2, 3)
        result = spatial.clone()
        for b in range(B):
            for i in range(K):
                ki = k[b, i].item()
                if ki > 0:
                    result[b, i] = torch.rot90(spatial[b, i], int(ki), dims=(1, 2))
        return result.permute(0, 1, 3, 4, 2).reshape(B, K, T, D)

    def _rotate_continuous(self, patches: torch.Tensor, angles: torch.Tensor, mode: str = "bilinear") -> torch.Tensor:
        """Rotate patches by arbitrary angles. patches: [B, K, C, H, W], angles: [B, K]"""
        B, K, C, H, W = patches.shape
        device = patches.device
        patches_flat = patches.view(B * K, C, H, W)
        angles_flat = angles.view(B * K)
        cos_a = torch.cos(angles_flat)
        sin_a = torch.sin(angles_flat)
        theta = torch.zeros(B * K, 2, 3, device=device)
        theta[:, 0, 0] = cos_a
        theta[:, 0, 1] = -sin_a
        theta[:, 1, 0] = sin_a
        theta[:, 1, 1] = cos_a
        grid = F.affine_grid(theta, patches_flat.shape, align_corners=False)
        rotated_flat = F.grid_sample(patches_flat, grid, mode=mode, padding_mode="zeros", align_corners=False)
        return rotated_flat.view(B, K, C, H, W)

    def _rotate_features_continuous(self, features: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
        """Rotate feature patches by arbitrary angles. features: [B, K, tokens, D]"""
        B, K, T, D = features.shape
        h = w = int(T ** 0.5)
        assert h * w == T
        device = features.device
        spatial = features.view(B, K, h, w, D).permute(0, 1, 4, 2, 3)
        spatial_flat = spatial.view(B * K, D, h, w)
        angles_flat = angles.view(B * K)
        cos_a = torch.cos(angles_flat)
        sin_a = torch.sin(angles_flat)
        theta = torch.zeros(B * K, 2, 3, device=device)
        theta[:, 0, 0] = cos_a
        theta[:, 0, 1] = -sin_a
        theta[:, 1, 0] = sin_a
        theta[:, 1, 1] = cos_a
        grid = F.affine_grid(theta, spatial_flat.shape, align_corners=False)
        rotated_flat = F.grid_sample(spatial_flat, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        return rotated_flat.view(B, K, D, h, w).permute(0, 1, 3, 4, 2).reshape(B, K, T, D)

    def _flip(self, patches: torch.Tensor, flip_h: torch.Tensor, flip_v: torch.Tensor) -> torch.Tensor:
        """Flip patches. patches: [B, K, C, H, W]"""
        B, K, _, _, _ = patches.shape
        result = patches.clone()
        for b in range(B):
            for i in range(K):
                p = result[b, i]
                if flip_h[b, i]:
                    p = torch.flip(p, dims=[2])
                if flip_v[b, i]:
                    p = torch.flip(p, dims=[1])
                result[b, i] = p
        return result

    def _flip_features(self, features: torch.Tensor, flip_h: torch.Tensor, flip_v: torch.Tensor) -> torch.Tensor:
        """Flip feature patches. features: [B, K, tokens, D]"""
        B, K, T, D = features.shape
        h = w = int(T ** 0.5)
        assert h * w == T
        spatial = features.view(B, K, h, w, D)
        result = spatial.clone()
        for b in range(B):
            for i in range(K):
                f = result[b, i]
                if flip_h[b, i]:
                    f = torch.flip(f, dims=[1])
                if flip_v[b, i]:
                    f = torch.flip(f, dims=[0])
                result[b, i] = f
        return result.reshape(B, K, T, D)

    def forward(
        self,
        patches: torch.Tensor,
        patch_labels: torch.Tensor,
        patch_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, dict]:
        """Apply augmentation. Returns (aug_patches, aug_labels, aug_features, aug_params)."""
        B, K = patches.shape[:2]
        device = patches.device
        aug_params = {'rotation_k': None, 'rotation_angles': None, 'flip_h': None, 'flip_v': None}

        if not self.training:
            return patches, patch_labels, patch_features, aug_params

        aug_patches, aug_labels, aug_features = patches, patch_labels, patch_features

        # Rotation
        if self.rotation == "90":
            k = torch.randint(0, 4, (B, K), device=device)
            aug_params['rotation_k'] = k
            aug_patches = self._rotate_90(aug_patches, k)
            aug_labels = self._rotate_90(aug_labels, k)
            if aug_features is not None:
                aug_features = self._rotate_features_90(aug_features, k)
        elif self.rotation == "continuous":
            angles = torch.empty(B, K, device=device).uniform_(-self.rotation_range, self.rotation_range)
            aug_params['rotation_angles'] = angles
            aug_patches = self._rotate_continuous(aug_patches, angles, mode="bilinear")
            aug_labels = self._rotate_continuous(aug_labels, angles, mode="nearest")
            if aug_features is not None:
                aug_features = self._rotate_features_continuous(aug_features, angles)

        # Flipping
        if self.flip_horizontal or self.flip_vertical:
            flip_h = torch.rand(B, K, device=device) > 0.5 if self.flip_horizontal else torch.zeros(B, K, dtype=torch.bool, device=device)
            flip_v = torch.rand(B, K, device=device) > 0.5 if self.flip_vertical else torch.zeros(B, K, dtype=torch.bool, device=device)
            aug_params['flip_h'] = flip_h
            aug_params['flip_v'] = flip_v
            aug_patches = self._flip(aug_patches, flip_h, flip_v)
            aug_labels = self._flip(aug_labels, flip_h, flip_v)
            if aug_features is not None:
                aug_features = self._flip_features(aug_features, flip_h, flip_v)

        return aug_patches, aug_labels, aug_features, aug_params

    def augment_features_only(self, features: torch.Tensor, aug_params: dict) -> torch.Tensor:
        """Apply augmentation to features using pre-determined aug_params."""
        if all(v is None for v in aug_params.values()):
            return features
        aug_features = features
        if aug_params.get('rotation_k') is not None:
            aug_features = self._rotate_features_90(aug_features, aug_params['rotation_k'])
        elif aug_params.get('rotation_angles') is not None:
            aug_features = self._rotate_features_continuous(aug_features, aug_params['rotation_angles'])
        if aug_params.get('flip_h') is not None or aug_params.get('flip_v') is not None:
            flip_h = aug_params.get('flip_h') or torch.zeros_like(aug_params.get('flip_v'))
            flip_v = aug_params.get('flip_v') or torch.zeros_like(aug_params.get('flip_h'))
            aug_features = self._flip_features(aug_features, flip_h, flip_v)
        return aug_features

    def inverse(self, predictions: torch.Tensor, aug_params: dict) -> torch.Tensor:
        """Apply inverse augmentation to predictions before aggregation."""
        if all(v is None for v in aug_params.values()):
            return predictions
        inv_pred = predictions
        # Inverse flipping (flip is its own inverse)
        if aug_params.get('flip_h') is not None or aug_params.get('flip_v') is not None:
            flip_h = aug_params.get('flip_h') or torch.zeros_like(aug_params.get('flip_v'))
            flip_v = aug_params.get('flip_v') or torch.zeros_like(aug_params.get('flip_h'))
            inv_pred = self._flip(inv_pred, flip_h, flip_v)
        # Inverse rotation
        if aug_params.get('rotation_k') is not None:
            inv_k = (4 - aug_params['rotation_k']) % 4
            inv_pred = self._rotate_90(inv_pred, inv_k)
        elif aug_params.get('rotation_angles') is not None:
            inv_pred = self._rotate_continuous(inv_pred, -aug_params['rotation_angles'], mode="bilinear")
        return inv_pred


class ContinuousSampler(nn.Module):
    """Samples patches at any pixel coordinate (not restricted to a grid)."""

    def __init__(
        self,
        patch_size: int,
        num_patches: int,
        temperature: float = 1.0,
        augmenter: PatchAugmenter | None = None,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.temperature = temperature
        self.augmenter = augmenter

    def forward(
        self,
        image: torch.Tensor,
        labels: torch.Tensor,
        weights: torch.Tensor,
        patch_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, dict]:
        """Sample K patches at any pixel coordinate."""
        B, C, H, W = image.shape
        C_mask = labels.shape[1]
        ps = self.patch_size
        K = self.num_patches
        device = image.device

        valid_h = H - ps + 1
        valid_w = W - ps + 1
        if valid_h <= 0 or valid_w <= 0:
            raise ValueError(f"Image size ({H}, {W}) too small for patch size {ps}")

        # Pool weights over patch area
        pooled_weights = F.max_pool2d(weights, kernel_size=ps, stride=1, padding=0)
        flat_weights = pooled_weights.reshape(B, -1) / self.temperature
        probs = F.softmax(flat_weights, dim=1)

        # Sample K indices without replacement
        indices = torch.multinomial(probs, K, replacement=False)
        h_coords = indices // valid_w
        w_coords = indices % valid_w
        coords = torch.stack([h_coords, w_coords], dim=2)

        # Extract patches
        patches = torch.zeros(B, K, C, ps, ps, device=device)
        patch_labels = torch.zeros(B, K, C_mask, ps, ps, device=device)
        for b in range(B):
            for k in range(K):
                h, w = h_coords[b, k].item(), w_coords[b, k].item()
                patches[b, k] = image[b, :, h:h+ps, w:w+ps]
                patch_labels[b, k] = labels[b, :, h:h+ps, w:w+ps]

        # Apply augmentation
        aug_params = {}
        aug_features = patch_features
        if self.augmenter is not None:
            patches, patch_labels, aug_features, aug_params = self.augmenter(patches, patch_labels, patch_features)

        return patches, patch_labels, coords, aug_features, aug_params


class SlidingWindowSampler(nn.Module):
    """Extracts patches in a regular grid pattern (sliding window)."""

    def __init__(
        self,
        patch_size: int,
        stride: int | None = None,
        augmenter: PatchAugmenter | None = None,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride if stride is not None else patch_size
        self.augmenter = augmenter

    def forward(
        self,
        image: torch.Tensor,
        labels: torch.Tensor,
        weights: torch.Tensor,  # Ignored for sliding window
        patch_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, dict]:
        """Extract all patches in a sliding window pattern."""
        B, C, H, W = image.shape
        C_mask = labels.shape[1]
        ps = self.patch_size
        stride = self.stride

        grid_h = max(1, (H - ps) // stride + 1)
        grid_w = max(1, (W - ps) // stride + 1)

        # Extract patches using unfold
        patches_unfolded = image.unfold(2, ps, stride).unfold(3, ps, stride)
        patches = patches_unfolded.reshape(B, C, -1, ps, ps).permute(0, 2, 1, 3, 4)

        labels_unfolded = labels.unfold(2, ps, stride).unfold(3, ps, stride)
        patch_labels = labels_unfolded.reshape(B, C_mask, -1, ps, ps).permute(0, 2, 1, 3, 4)

        # Generate coordinates
        device = image.device
        h_indices = torch.arange(grid_h, device=device) * stride
        w_indices = torch.arange(grid_w, device=device) * stride
        h_grid, w_grid = torch.meshgrid(h_indices, w_indices, indexing='ij')
        coords_single = torch.stack([h_grid.flatten(), w_grid.flatten()], dim=1)
        coords = coords_single.unsqueeze(0).expand(B, -1, -1)

        # Apply augmentation
        aug_params = {}
        aug_features = patch_features
        if self.augmenter is not None:
            patches, patch_labels, aug_features, aug_params = self.augmenter(patches, patch_labels, patch_features)

        return patches, patch_labels, coords, aug_features, aug_params


def create_sampler(
    sampler_type: str,
    patch_size: int,
    num_patches: int = 16,
    temperature: float = 1.0,
    stride: int | None = None,
    augmenter: PatchAugmenter | None = None,
) -> nn.Module:
    """Factory function to create samplers from config."""
    if sampler_type == "continuous":
        return ContinuousSampler(
            patch_size=patch_size,
            num_patches=num_patches,
            temperature=temperature,
            augmenter=augmenter,
        )
    elif sampler_type == "sliding_window":
        return SlidingWindowSampler(
            patch_size=patch_size,
            stride=stride,
            augmenter=augmenter,
        )
    else:
        raise ValueError(f"Unknown sampler type: {sampler_type}. Use 'continuous' or 'sliding_window'.")
