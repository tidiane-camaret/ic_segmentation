"""Patch sampling for PatchICL v3.

Unified sampling weight computation and ContinuousSampler/SlidingWindowSampler.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import compute_binary_entropy


def compute_sampling_weights(
    mode: str,
    gt_mask: torch.Tensor | None = None,
    pred_logits: torch.Tensor | None = None,
    prev_pred: torch.Tensor | None = None,
    sampling_map: torch.Tensor | None = None,
) -> torch.Tensor:
    """Unified sampling weight computation.

    Args:
        mode: Sampling strategy. GT-based modes require gt_mask:
            - "gt_foreground": Sample from GT mask foreground
            - "gt_foreground_entropy_balanced": 50% foreground + 50% entropy
            - "gt_previous_pred_error": Sample where prev pred differs from GT
            Prediction-based modes require pred_logits:
            - "predicted_entropy": Entropy of prediction
            - "predicted_foreground_entropy_balanced": 50% pred + 50% entropy
            - "sampling_map": Use 1 - sampling_map
        gt_mask: [B, 1, H, W] - Ground truth mask (for GT-based modes)
        pred_logits: [B, C, H, W] - Previous prediction logits
        prev_pred: [B, 1, H, W] - Previous prediction for error-based mode
        sampling_map: [B, 1, H, W] - Sampling map from previous level

    Returns:
        sampling_weights: [B, 1, H, W]
    """
    # GT-based modes
    if mode == "gt_foreground":
        return gt_mask

    if mode == "gt_foreground_entropy_balanced":
        soft_mask = gt_mask.float().clamp(1e-6, 1 - 1e-6)
        entropy = compute_binary_entropy(soft_mask)
        return 0.5 * soft_mask + 0.5 * entropy

    if mode == "gt_previous_pred_error":
        if prev_pred is None:
            return gt_mask
        soft_mask = gt_mask.float().clamp(1e-6, 1 - 1e-6)
        pred_prob = torch.sigmoid(prev_pred.float())
        return torch.abs(pred_prob - soft_mask)

    # Prediction-based modes
    if mode == "predicted_entropy":
        pred_prob = torch.sigmoid(pred_logits.float())
        entropy = compute_binary_entropy(pred_prob)
        if entropy.shape[1] > 1:
            entropy = entropy.max(dim=1, keepdim=True)[0]
        return entropy

    if mode == "predicted_foreground_entropy_balanced":
        pred_prob = torch.sigmoid(pred_logits.float()).clamp(1e-6, 1 - 1e-6)
        entropy = compute_binary_entropy(pred_prob)
        if pred_prob.shape[1] > 1:
            pred_prob = pred_prob.max(dim=1, keepdim=True)[0]
            entropy = entropy.max(dim=1, keepdim=True)[0]
        return 0.5 * pred_prob + 0.5 * entropy

    if mode == "sampling_map":
        if sampling_map is not None:
            return 1.0 - sampling_map
        return torch.sigmoid(pred_logits)

    # Fallback
    if gt_mask is not None:
        return gt_mask
    return torch.ones_like(pred_logits[:, :1])


class PatchAugmenter(nn.Module):
    """Augments patches with rotation and flipping."""

    def __init__(
        self,
        rotation: str = "none",
        rotation_range: float = 0.5,
        flip_horizontal: bool = False,
        flip_vertical: bool = False,
    ):
        super().__init__()
        self.rotation = rotation
        self.rotation_range = rotation_range
        self.flip_horizontal = flip_horizontal
        self.flip_vertical = flip_vertical

    def _rotate_90(self, patches: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """Rotate patches by k * 90 degrees. patches: [B, K, C, H, W], k: [B, K]"""
        B, K = patches.shape[:2]
        result = patches.clone()
        for b in range(B):
            for i in range(K):
                ki = k[b, i].item()
                if ki > 0:
                    result[b, i] = torch.rot90(patches[b, i], int(ki), dims=(1, 2))
        return result

    def _rotate_features_90(self, features: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """Rotate features by k * 90 degrees. features: [B, K, tokens, D]"""
        B, K, T, D = features.shape
        h = w = int(T ** 0.5)
        spatial = features.view(B, K, h, w, D).permute(0, 1, 4, 2, 3)
        result = spatial.clone()
        for b in range(B):
            for i in range(K):
                ki = k[b, i].item()
                if ki > 0:
                    result[b, i] = torch.rot90(spatial[b, i], int(ki), dims=(1, 2))
        return result.permute(0, 1, 3, 4, 2).reshape(B, K, T, D)

    def _rotate_continuous(self, patches: torch.Tensor, angles: torch.Tensor, mode: str = "bilinear") -> torch.Tensor:
        """Rotate patches by arbitrary angles."""
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
        """Rotate features by arbitrary angles."""
        B, K, T, D = features.shape
        h = w = int(T ** 0.5)
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
        """Flip patches."""
        B, K = patches.shape[:2]
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
        """Flip features."""
        B, K, T, D = features.shape
        h = w = int(T ** 0.5)
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
        """Apply augmentation to features using pre-determined params."""
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
        """Apply inverse augmentation to predictions."""
        if all(v is None for v in aug_params.values()):
            return predictions
        inv_pred = predictions
        # Inverse flipping
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
    """Samples patches at continuous coordinates with Gumbel-TopK."""

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
        differentiable: bool = False,
    ):
        """
        Args:
            differentiable: If True, use Gumbel-Softmax for gradient flow.
        """
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.num_patches_val = num_patches_val or num_patches
        self.temperature = temperature
        self.stride = stride or 1
        self.augmenter = augmenter
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.extract_patches = extract_patches
        self.spread_sigma = spread_sigma
        self.differentiable = differentiable

        if spread_sigma > 0:
            self.register_buffer('_gaussian_kernel', self._make_gaussian_kernel(spread_sigma))
        else:
            self._gaussian_kernel = None

    def _make_gaussian_kernel(self, sigma: float) -> torch.Tensor:
        """Create 2D Gaussian kernel for blurring."""
        k = int(4 * sigma + 1) | 1
        x = torch.arange(k, dtype=torch.float32) - k // 2
        gauss_1d = torch.exp(-x ** 2 / (2 * sigma ** 2))
        gauss_2d = gauss_1d[:, None] * gauss_1d[None, :]
        gauss_2d = gauss_2d / gauss_2d.sum()
        return gauss_2d.view(1, 1, k, k)

    def _gaussian_blur(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Gaussian blur."""
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
    ) -> tuple[
        torch.Tensor | None,  # patches
        torch.Tensor,  # patch_labels
        torch.Tensor,  # coords
        torch.Tensor | None,  # aug_features
        dict,  # aug_params
        torch.Tensor,  # patch_validity
        int,  # K
        torch.Tensor | None,  # selection_probs
    ]:
        """Sample K patches. Returns (patches, labels, coords, aug_features, aug_params, validity, K, selection_probs)."""
        B, C, H, W = image.shape
        C_mask = labels.shape[1]
        ps = self.patch_size
        K = self.num_patches if self.training else self.num_patches_val
        device = image.device

        pad_before = self.pad_before if self.pad_before is not None else ps // 4
        pad_after = self.pad_after if self.pad_after is not None else ps - pad_before - 1

        # Pad inputs
        image_pad = F.pad(image, (pad_before, pad_after, pad_before, pad_after), mode='constant', value=0)
        labels_pad = F.pad(labels, (pad_before, pad_after, pad_before, pad_after), mode='constant', value=0)
        weights_pad = F.pad(weights, (pad_before, pad_after, pad_before, pad_after), mode='constant', value=0)

        # Validity map
        validity_map = torch.zeros(1, 1, H + ps - 1, W + ps - 1, device=device)
        validity_map[:, :, pad_before:pad_before + H, pad_before:pad_before + W] = 1.0

        H_pad, W_pad = image_pad.shape[2], image_pad.shape[3]

        # Pool weights
        pooled_weights = F.max_pool2d(weights_pad, kernel_size=ps, stride=1, padding=0)
        flat = pooled_weights.flatten(1)
        lo = flat.min(dim=1, keepdim=True).values
        hi = flat.max(dim=1, keepdim=True).values
        eps = 1e-4 if pooled_weights.dtype == torch.float16 else 1e-6
        pooled_weights = (pooled_weights - lo.view(B, 1, 1, 1)) / (hi.view(B, 1, 1, 1) - lo.view(B, 1, 1, 1) + eps)

        if self.spread_sigma > 0:
            pooled_weights = self._gaussian_blur(pooled_weights)

        selection_probs = None

        # Gumbel-TopK sampling
        if self.stride > 1:
            pooled_weights_strided = pooled_weights[:, :, ::self.stride, ::self.stride]
            flat_weights = pooled_weights_strided.reshape(B, -1) / self.temperature
            u = torch.rand_like(flat_weights, dtype=torch.float32).clamp(1e-6, 1 - 1e-6)
            gumbel = (-torch.log(-torch.log(u))).to(flat_weights.dtype)
            scores = flat_weights + gumbel
            _, indices = torch.topk(scores, K, dim=1)

            if self.differentiable and self.training:
                soft_probs = F.softmax(scores, dim=1)
                selection_probs = soft_probs.gather(1, indices)
                selection_probs = selection_probs / (selection_probs.sum(dim=1, keepdim=True) + eps)

            strided_w = pooled_weights_strided.shape[3]
            h_coords_strided = indices // strided_w
            w_coords_strided = indices % strided_w
            h_coords_pad = h_coords_strided * self.stride
            w_coords_pad = w_coords_strided * self.stride
        else:
            valid_h = H_pad - ps + 1
            valid_w = W_pad - ps + 1
            flat_weights = pooled_weights.reshape(B, -1) / self.temperature
            u = torch.rand_like(flat_weights, dtype=torch.float32).clamp(1e-6, 1 - 1e-6)
            gumbel = (-torch.log(-torch.log(u))).to(flat_weights.dtype)
            scores = flat_weights + gumbel
            _, indices = torch.topk(scores, K, dim=1)

            if self.differentiable and self.training:
                soft_probs = F.softmax(scores, dim=1)
                selection_probs = soft_probs.gather(1, indices)
                selection_probs = selection_probs / (selection_probs.sum(dim=1, keepdim=True) + eps)

            h_coords_pad = indices // valid_w
            w_coords_pad = indices % valid_w

        # Extract patches via gather
        row_offsets = torch.arange(ps, device=device)
        col_offsets = torch.arange(ps, device=device)
        rows_2d = (h_coords_pad.unsqueeze(-1) + row_offsets).unsqueeze(-1).expand(B, K, ps, ps)
        cols_2d = (w_coords_pad.unsqueeze(-1) + col_offsets).unsqueeze(-2).expand(B, K, ps, ps)
        flat_idx = (rows_2d * W_pad + cols_2d).reshape(B, K * ps * ps)

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

        # Convert to original-space coordinates
        h_coords = h_coords_pad - pad_before
        w_coords = w_coords_pad - pad_before
        coords = torch.stack([h_coords, w_coords], dim=2)

        # Apply augmentation
        aug_params = {}
        aug_features = patch_features
        if self.augmenter is not None and patches is not None:
            combined = torch.cat([patch_labels, patch_validity], dim=2)
            patches, combined, aug_features, aug_params = self.augmenter(patches, combined, patch_features)
            patch_labels = combined[:, :, :C_mask]
            patch_validity = combined[:, :, C_mask:]

        return patches, patch_labels, coords, aug_features, aug_params, patch_validity, K, selection_probs


class SlidingWindowSampler(nn.Module):
    """Extracts patches in a regular grid pattern."""

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
        self.num_patches_val = num_patches_val or num_patches
        self.stride = stride
        self.augmenter = augmenter
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.extract_patches = extract_patches

    def forward(
        self,
        image: torch.Tensor,
        labels: torch.Tensor,
        weights: torch.Tensor,
        patch_features: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        dict,
        torch.Tensor,
        int,
        torch.Tensor | None,
    ]:
        """Extract patches in sliding window pattern."""
        B, C, H, W = image.shape
        C_mask = labels.shape[1]
        ps = self.patch_size
        device = image.device

        pad_before = self.pad_before if self.pad_before is not None else ps // 4
        pad_after = self.pad_after if self.pad_after is not None else ps - pad_before - 1

        image_pad = F.pad(image, (pad_before, pad_after, pad_before, pad_after), mode='constant', value=0)
        labels_pad = F.pad(labels, (pad_before, pad_after, pad_before, pad_after), mode='constant', value=0)

        H_pad, W_pad = image_pad.shape[2], image_pad.shape[3]
        validity_map = torch.zeros(1, 1, H_pad, W_pad, device=device)
        validity_map[:, :, pad_before:pad_before + H, pad_before:pad_before + W] = 1.0

        # Compute stride
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

        # Extract with unfold
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

        # Generate coordinates
        h_indices = torch.arange(grid_h, device=device) * stride - pad_before
        w_indices = torch.arange(grid_w, device=device) * stride - pad_before
        h_grid, w_grid = torch.meshgrid(h_indices, w_indices, indexing='ij')
        coords_single = torch.stack([h_grid.flatten(), w_grid.flatten()], dim=1)
        coords = coords_single.unsqueeze(0).expand(B, -1, -1)

        # Apply augmentation
        aug_params = {}
        aug_features = patch_features
        if self.augmenter is not None and patches is not None:
            combined = torch.cat([patch_labels, patch_validity], dim=2)
            patches, combined, aug_features, aug_params = self.augmenter(patches, combined, patch_features)
            patch_labels = combined[:, :, :C_mask]
            patch_validity = combined[:, :, C_mask:]

        return patches, patch_labels, coords, aug_features, aug_params, patch_validity, K, None


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
    differentiable: bool = False,
) -> nn.Module:
    """Factory function to create samplers."""
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
            differentiable=differentiable,
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
        raise ValueError(f"Unknown sampler type: {sampler_type}")
