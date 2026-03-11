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
            flip_h = aug_params['flip_h'] if aug_params.get('flip_h') is not None else torch.zeros_like(aug_params['flip_v'])
            flip_v = aug_params['flip_v'] if aug_params.get('flip_v') is not None else torch.zeros_like(aug_params['flip_h'])
            aug_features = self._flip_features(aug_features, flip_h, flip_v)
        return aug_features

    def inverse(self, predictions: torch.Tensor, aug_params: dict) -> torch.Tensor:
        """Apply inverse augmentation to predictions before aggregation."""
        if all(v is None for v in aug_params.values()):
            return predictions
        inv_pred = predictions
        # Inverse flipping (flip is its own inverse)
        if aug_params.get('flip_h') is not None or aug_params.get('flip_v') is not None:
            flip_h = aug_params['flip_h'] if aug_params.get('flip_h') is not None else torch.zeros_like(aug_params['flip_v'])
            flip_v = aug_params['flip_v'] if aug_params.get('flip_v') is not None else torch.zeros_like(aug_params['flip_h'])
            inv_pred = self._flip(inv_pred, flip_h, flip_v)
        # Inverse rotation
        if aug_params.get('rotation_k') is not None:
            inv_k = (4 - aug_params['rotation_k']) % 4
            inv_pred = self._rotate_90(inv_pred, inv_k)
        elif aug_params.get('rotation_angles') is not None:
            inv_pred = self._rotate_continuous(inv_pred, -aug_params['rotation_angles'], mode="bilinear")
        return inv_pred


class ContinuousSampler(nn.Module):
    """Samples patches at any pixel coordinate whose center is inside the image."""

    def __init__(
        self,
        patch_size: int,
        num_patches: int,
        num_patches_val: int | None = None,
        temperature: float = 1.0,
        stride: int | None = None,
        augmenter: PatchAugmenter | None = None,
        pad_before: int | None = None,
        pad_after: int | None = None,
        extract_patches: bool = False,
        spread_sigma: float = 0.0,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.num_patches_val = num_patches_val if num_patches_val is not None else num_patches
        self.temperature = temperature
        self.stride = stride if stride is not None else 1
        self.augmenter = augmenter
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.extract_patches = extract_patches
        self.spread_sigma = spread_sigma

        # Precompute Gaussian kernel if spreading is enabled
        if spread_sigma > 0:
            self.register_buffer('_gaussian_kernel', self._make_gaussian_kernel(spread_sigma))
        else:
            self._gaussian_kernel = None

    def _make_gaussian_kernel(self, sigma: float) -> torch.Tensor:
        """Create a 2D Gaussian kernel for blurring."""
        k = int(4 * sigma + 1) | 1  # odd kernel size, ~4 sigma coverage
        x = torch.arange(k, dtype=torch.float32) - k // 2
        gauss_1d = torch.exp(-x ** 2 / (2 * sigma ** 2))
        gauss_2d = gauss_1d[:, None] * gauss_1d[None, :]
        gauss_2d = gauss_2d / gauss_2d.sum()
        return gauss_2d.view(1, 1, k, k)

    def _gaussian_blur(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Gaussian blur to spread high-weight regions."""
        if self._gaussian_kernel is None:
            return x
        k = self._gaussian_kernel.shape[-1]
        pad = k // 2
        return F.conv2d(x, self._gaussian_kernel.to(x.dtype), padding=pad)

    def forward(
        self,
        image: torch.Tensor,
        labels: torch.Tensor,
        weights: torch.Tensor,
        patch_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, dict, torch.Tensor]:
        """Sample K patches whose center is inside the image (may extend beyond borders)."""
        B, C, H, W = image.shape
        C_mask = labels.shape[1]
        ps = self.patch_size
        K = self.num_patches if self.training else self.num_patches_val
        device = image.device

        pad_before = self.pad_before if self.pad_before is not None else ps // 4
        pad_after = self.pad_after if self.pad_after is not None else ps - pad_before - 1

        # Pad inputs so all center-inside patches can be fully extracted
        image_pad = F.pad(image, (pad_before, pad_after, pad_before, pad_after), mode='constant', value=0)
        labels_pad = F.pad(labels, (pad_before, pad_after, pad_before, pad_after), mode='constant', value=0)
        weights_pad = F.pad(weights, (pad_before, pad_after, pad_before, pad_after), mode='constant', value=0)

        # Validity map: 1 for original image pixels, 0 for padding
        validity_map = torch.zeros(1, 1, H + ps - 1, W + ps - 1, device=device)
        validity_map[:, :, pad_before:pad_before + H, pad_before:pad_before + W] = 1.0

        # In padded space, valid_h = H, valid_w = W (one position per center pixel)
        H_pad, W_pad = image_pad.shape[2], image_pad.shape[3]
        valid_h = H_pad - ps + 1  # = H
        valid_w = W_pad - ps + 1  # = W
        if valid_h <= 0 or valid_w <= 0:
            raise ValueError(f"Image size ({H}, {W}) too small for patch size {ps}")

        # Pool weights over patch area (on padded weights)
        pooled_weights = F.max_pool2d(weights_pad, kernel_size=ps, stride=1, padding=0)
        # Min-max normalize to [0, 1] per sample for discriminative sampling
        flat = pooled_weights.flatten(1)
        lo = flat.min(dim=1, keepdim=True).values
        hi = flat.max(dim=1, keepdim=True).values
        # Use larger epsilon for fp16 numerical stability
        eps = 1e-4 if pooled_weights.dtype == torch.float16 else 1e-6
        pooled_weights = (pooled_weights - lo.view(B, 1, 1, 1)) / (hi.view(B, 1, 1, 1) - lo.view(B, 1, 1, 1) + eps)

        # Gaussian blur to spread high-weight regions (creates larger clusters around positive zones)
        if self.spread_sigma > 0:
            pooled_weights = self._gaussian_blur(pooled_weights)

        # Apply stride to space out possible sampling positions
        if self.stride > 1:
            # Subsample pooled_weights at stride intervals
            pooled_weights_strided = pooled_weights[:, :, ::self.stride, ::self.stride]
            flat_weights = pooled_weights_strided.reshape(B, -1) / self.temperature

            # Gumbel-Top-K sampling from strided positions
            # Use fp32 for numerical stability in nested log (critical for fp16 training)
            u = torch.rand_like(flat_weights, dtype=torch.float32).clamp(1e-6, 1 - 1e-6)
            gumbel = (-torch.log(-torch.log(u))).to(flat_weights.dtype)
            scores = flat_weights + gumbel
            _, indices = torch.topk(scores, K, dim=1)
            
            # Map indices back to original coordinate space
            strided_w = pooled_weights_strided.shape[3]
            h_coords_strided = indices // strided_w
            w_coords_strided = indices % strided_w
            h_coords_pad = h_coords_strided * self.stride
            w_coords_pad = w_coords_strided * self.stride
        else:
            # Original sampling without stride
            flat_weights = pooled_weights.reshape(B, -1) / self.temperature

            # Gumbel-Top-K: equivalent to multinomial without replacement, but ~10x faster
            # Adding Gumbel noise to log-probs and taking top-k gives same distribution
            # Use fp32 for numerical stability in nested log (critical for fp16 training)
            u = torch.rand_like(flat_weights, dtype=torch.float32).clamp(1e-6, 1 - 1e-6)
            gumbel = (-torch.log(-torch.log(u))).to(flat_weights.dtype)
            scores = flat_weights + gumbel
            _, indices = torch.topk(scores, K, dim=1)
            h_coords_pad = indices // valid_w
            w_coords_pad = indices % valid_w

        # Vectorized patch extraction using gather (no Python loops / .item() syncs)
        row_offsets = torch.arange(ps, device=device)
        col_offsets = torch.arange(ps, device=device)
        rows_2d = (h_coords_pad.unsqueeze(-1) + row_offsets).unsqueeze(-1).expand(B, K, ps, ps)
        cols_2d = (w_coords_pad.unsqueeze(-1) + col_offsets).unsqueeze(-2).expand(B, K, ps, ps)
        flat_idx = (rows_2d * W_pad + cols_2d).reshape(B, K * ps * ps)

        # Only extract image patches if needed (labels/validity always needed)
        if self.extract_patches:
            img_flat = image_pad.reshape(B, C, -1)
            patches = img_flat.gather(2, flat_idx.unsqueeze(1).expand(B, C, -1))
            patches = patches.reshape(B, C, K, ps, ps).permute(0, 2, 1, 3, 4)
        else:
            patches = None

        lbl_flat = labels_pad.reshape(B, C_mask, -1)
        patch_labels = lbl_flat.gather(2, flat_idx.unsqueeze(1).expand(B, C_mask, -1))
        patch_labels = patch_labels.reshape(B, C_mask, K, ps, ps).permute(0, 2, 1, 3, 4)

        val_flat = validity_map.reshape(1, 1, -1).expand(B, -1, -1)
        patch_validity = val_flat.gather(2, flat_idx.unsqueeze(1))
        patch_validity = patch_validity.reshape(B, 1, K, ps, ps).permute(0, 2, 1, 3, 4)

        # Convert to original-space coordinates (can be negative)
        h_coords = h_coords_pad - pad_before
        w_coords = w_coords_pad - pad_before
        coords = torch.stack([h_coords, w_coords], dim=2)

        # Apply augmentation (validity is augmented alongside labels)
        aug_params = {}
        aug_features = patch_features
        if self.augmenter is not None and patches is not None:
            combined = torch.cat([patch_labels, patch_validity], dim=2)
            patches, combined, aug_features, aug_params = self.augmenter(patches, combined, patch_features)
            patch_labels = combined[:, :, :C_mask]
            patch_validity = combined[:, :, C_mask:]

        return patches, patch_labels, coords, aug_features, aug_params, patch_validity, K


class SlidingWindowSampler(nn.Module):
    """Extracts patches in a regular grid pattern, including border patches whose center is inside."""

    def __init__(
        self,
        patch_size: int,
        num_patches: int = 16,
        num_patches_val: int | None = None,
        stride: int | None = None,
        augmenter: PatchAugmenter | None = None,
        pad_before: int | None = None,
        pad_after: int | None = None,
        extract_patches: bool = False,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.num_patches_val = num_patches_val if num_patches_val is not None else num_patches
        self.stride = stride  # None means auto-compute from num_patches
        self.augmenter = augmenter
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.extract_patches = extract_patches

    def forward(
        self,
        image: torch.Tensor,
        labels: torch.Tensor,
        weights: torch.Tensor,  # Ignored for sliding window
        patch_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, dict, torch.Tensor]:
        """Extract all patches in a sliding window pattern (patches may extend beyond borders)."""
        B, C, H, W = image.shape
        C_mask = labels.shape[1]
        ps = self.patch_size
        device = image.device

        pad_before = self.pad_before if self.pad_before is not None else ps // 4
        pad_after = self.pad_after if self.pad_after is not None else ps - pad_before - 1

        # Pad inputs
        image_pad = F.pad(image, (pad_before, pad_after, pad_before, pad_after), mode='constant', value=0)
        labels_pad = F.pad(labels, (pad_before, pad_after, pad_before, pad_after), mode='constant', value=0)

        # Validity map
        H_pad, W_pad = image_pad.shape[2], image_pad.shape[3]
        validity_map = torch.zeros(1, 1, H_pad, W_pad, device=device)
        validity_map[:, :, pad_before:pad_before + H, pad_before:pad_before + W] = 1.0

        # Compute stride: use explicit stride or auto-compute from num_patches
        if self.stride is not None:
            stride = self.stride
        else:
            K = self.num_patches if self.training else self.num_patches_val
            grid_side = max(1, round(K ** 0.5))
            if grid_side > 1 and H > 1:
                stride = max(1, (H - 1) // (grid_side - 1))
            else:
                stride = 1

        grid_h = max(1, (H_pad - ps) // stride + 1)
        grid_w = max(1, (W_pad - ps) // stride + 1)
        K = grid_h * grid_w

        # Extract patches using unfold on padded tensors
        if self.extract_patches:
            patches_unfolded = image_pad.unfold(2, ps, stride).unfold(3, ps, stride)
            patches = patches_unfolded.reshape(B, C, -1, ps, ps).permute(0, 2, 1, 3, 4)
        else:
            patches = None

        labels_unfolded = labels_pad.unfold(2, ps, stride).unfold(3, ps, stride)
        patch_labels = labels_unfolded.reshape(B, C_mask, -1, ps, ps).permute(0, 2, 1, 3, 4)

        validity_expanded = validity_map.expand(B, -1, -1, -1)
        validity_unfolded = validity_expanded.unfold(2, ps, stride).unfold(3, ps, stride)
        patch_validity = validity_unfolded.reshape(B, 1, -1, ps, ps).permute(0, 2, 1, 3, 4)

        # Generate coordinates in original space (can be negative)
        h_indices = torch.arange(grid_h, device=device) * stride - pad_before
        w_indices = torch.arange(grid_w, device=device) * stride - pad_before
        h_grid, w_grid = torch.meshgrid(h_indices, w_indices, indexing='ij')
        coords_single = torch.stack([h_grid.flatten(), w_grid.flatten()], dim=1)
        coords = coords_single.unsqueeze(0).expand(B, -1, -1)

        # Apply augmentation (validity is augmented alongside labels)
        aug_params = {}
        aug_features = patch_features
        if self.augmenter is not None and patches is not None:
            combined = torch.cat([patch_labels, patch_validity], dim=2)
            patches, combined, aug_features, aug_params = self.augmenter(patches, combined, patch_features)
            patch_labels = combined[:, :, :C_mask]
            patch_validity = combined[:, :, C_mask:]

        return patches, patch_labels, coords, aug_features, aug_params, patch_validity, K


def create_sampler(
    sampler_type: str,
    patch_size: int,
    num_patches: int = 16,
    num_patches_val: int | None = None,
    temperature: float = 1.0,
    stride: int | None = None,
    augmenter: PatchAugmenter | None = None,
    pad_before: int | None = None,
    pad_after: int | None = None,
    extract_patches: bool = False,
    spread_sigma: float = 0.0,
) -> nn.Module:
    """Factory function to create samplers from config."""
    if sampler_type == "continuous":
        return ContinuousSampler(
            patch_size=patch_size,
            num_patches=num_patches,
            num_patches_val=num_patches_val,
            temperature=temperature,
            stride=stride,
            augmenter=augmenter,
            pad_before=pad_before,
            pad_after=pad_after,
            extract_patches=extract_patches,
            spread_sigma=spread_sigma,
        )
    elif sampler_type == "sliding_window":
        return SlidingWindowSampler(
            patch_size=patch_size,
            num_patches=num_patches,
            num_patches_val=num_patches_val,
            stride=stride,
            augmenter=augmenter,
            pad_before=pad_before,
            pad_after=pad_after,
            extract_patches=extract_patches,
        )
    else:
        raise ValueError(f"Unknown sampler type: {sampler_type}. Use 'continuous' or 'sliding_window'.")
