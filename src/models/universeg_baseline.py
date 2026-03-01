"""
UniverSeg baseline wrapper for evaluation.

Wraps the UniverSeg model to match the interface expected by eval.py.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

class UniverSegBaseline(nn.Module):
    """
    Wrapper for UniverSeg model that matches the PatchICL evaluation interface.

    UniverSeg expects:
        - target_image: (B, 1, H, W)
        - support_images: (B, S, 1, H, W)
        - support_labels: (B, S, 1, H, W)

    And outputs: (B, 1, H, W) predictions
    """

    def __init__(self, pretrained: bool = True, input_size: int = 128, freeze: bool = False):
        super().__init__()
        sys.path.append("/work/dlclarge2/ndirt-SegFM3D/repos/UniVerseg")  # Add path to import universeg
        from universeg import universeg
        self.model = universeg(pretrained=pretrained)
        self.input_size = input_size  # Configurable eval resolution (trained on 128)

        # Optionally freeze the pretrained model (for evaluation only)
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

        # Placeholder loss function (can be set via set_loss_functions)
        self.aggreg_criterion = None
        self.patch_criterion = None

    def set_loss_functions(self, patch_criterion: nn.Module, aggreg_criterion: nn.Module):
        """Set the loss functions for compatibility with train_utils."""
        self.patch_criterion = patch_criterion
        self.aggreg_criterion = aggreg_criterion

    def compute_loss(
        self,
        outputs: dict[str, torch.Tensor],
        labels: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute loss for training. Matches PatchICL interface."""
        if self.aggreg_criterion is None:
            raise RuntimeError("Loss functions not set. Call set_loss_functions() first.")

        if labels.dim() == 3:
            labels = labels.unsqueeze(1)

        final_logit = outputs["final_logit"]

        # Compute main segmentation loss
        seg_loss = self.aggreg_criterion(final_logit, labels.float())

        return {
            "total_loss": seg_loss,
        }

    def _resize_to_model(self, x: torch.Tensor) -> torch.Tensor:
        """Resize tensor to model input size (128x128)."""
        if x.shape[-2:] != (self.input_size, self.input_size):
            return F.interpolate(x, size=(self.input_size, self.input_size), 
                               mode='bilinear', align_corners=False)
        return x

    def _resize_mask_to_model(self, x: torch.Tensor) -> torch.Tensor:
        """Resize mask to model input size (128x128) using nearest neighbor."""
        if x.shape[-2:] != (self.input_size, self.input_size):
            return F.interpolate(x, size=(self.input_size, self.input_size), mode='nearest')
        return x

    def _minmax_normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Min-max normalize tensor to [0, 1] range per sample."""
        # x: (B, C, H, W) or (B*k, C, H, W)
        # Normalize each sample independently
        B = x.shape[0]
        x_flat = x.view(B, -1)
        x_min = x_flat.min(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
        x_max = x_flat.max(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
        # Avoid division by zero
        return (x - x_min) / (x_max - x_min + 1e-8)

    def forward(
        self,
        image: torch.Tensor,
        labels: torch.Tensor = None,
        context_in: torch.Tensor = None,
        context_out: torch.Tensor = None,
        target_features: torch.Tensor = None,
        context_features: torch.Tensor = None,
        mode: str = "train",
        return_attn_weights: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass matching PatchICL interface.
        
        Args:
            image: Target image (B, C, H, W)
            labels: Target labels (B, 1, H, W) - not used during inference
            context_in: Support images (B, k, C, H, W)
            context_out: Support labels (B, k, 1, H, W)
            target_features: Pre-computed features (ignored)
            context_features: Pre-computed features (ignored)
            mode: 'train' or 'test'
            return_attn_weights: Whether to return attention weights (ignored)
            
        Returns:
            Dict with 'final_logit' and other keys for compatibility
        """
        B, C, H, W = image.shape
        device = image.device
        
        if labels is not None and labels.dim() == 3:
            labels = labels.unsqueeze(1)
        
        # UniverSeg expects single-channel inputs
        if C > 1:
            # Convert to grayscale if RGB
            target_image = image.mean(dim=1, keepdim=True)
        else:
            target_image = image
            
        # Resize target to 128x128 and apply min-max normalization to [0, 1]
        target_resized = self._resize_to_model(target_image)
        target_resized = self._minmax_normalize(target_resized)
        
        # Prepare support set from context
        if context_in is not None and context_out is not None:
            # context_in: (B, k, C, H, W) -> support_images: (B, S, 1, H, W)
            # context_out: (B, k, 1, H, W) -> support_labels: (B, S, 1, H, W)
            B_ctx, k, C_ctx, H_ctx, W_ctx = context_in.shape
            
            # Convert to grayscale if needed
            if C_ctx > 1:
                support_images = context_in.mean(dim=2, keepdim=True)  # (B, k, 1, H, W)
            else:
                support_images = context_in
                
            support_labels = context_out
            
            # Resize support images and labels to 128x128
            support_images_flat = support_images.view(B_ctx * k, 1, H_ctx, W_ctx)
            support_images_resized = self._resize_to_model(support_images_flat)
            # Apply min-max normalization to [0, 1] for support images
            support_images_resized = self._minmax_normalize(support_images_resized)
            support_images_resized = support_images_resized.view(B_ctx, k, 1, self.input_size, self.input_size)
            
            support_labels_flat = support_labels.view(B_ctx * k, 1, H_ctx, W_ctx)
            support_labels_resized = self._resize_mask_to_model(support_labels_flat)
            support_labels_resized = support_labels_resized.view(B_ctx, k, 1, self.input_size, self.input_size)
        else:
            # No context provided - create dummy support (will give poor results)
            support_images_resized = target_resized.unsqueeze(1)  # (B, 1, 1, 256, 256)
            support_labels_resized = torch.zeros(B, 1, 1, self.input_size, self.input_size, device=device)
        
        # Run UniverSeg model
        with torch.set_grad_enabled(mode == "train"):
            prediction = self.model(
                target_resized,           # (B, 1, H, W)
                support_images_resized,   # (B, S, 1, H, W)
                support_labels_resized,   # (B, S, 1, H, W)
            )  # -> (B, 1, H, W)
        
        # Resize prediction back to original size
        if (H, W) != (self.input_size, self.input_size):
            final_logit = F.interpolate(prediction, size=(H, W), mode='bilinear', align_corners=False)
        else:
            final_logit = prediction

        # Compute entropy-based confidence from logits (no learned params)
        p = torch.sigmoid(final_logit).clamp(1e-6, 1 - 1e-6)
        entropy = -(p * p.log() + (1 - p) * (1 - p).log())
        final_conf = 1.0 - entropy / math.log(2)  # [0, 1]

        # Create dummy outputs for compatibility with validate()
        # patch_logits and patch_coords are needed by validate()
        dummy_patch_size = 32
        num_patches = 1
        dummy_patches = target_image[:, :, :dummy_patch_size, :dummy_patch_size].unsqueeze(1)
        dummy_patch_logits = final_logit[:, :, :dummy_patch_size, :dummy_patch_size].unsqueeze(1)
        if labels is not None:
            dummy_patch_labels = labels[:, :, :dummy_patch_size, :dummy_patch_size].unsqueeze(1)
        else:
            dummy_patch_labels = torch.zeros_like(dummy_patch_logits)
        dummy_coords = torch.zeros(B, num_patches, 2, device=device)

        return {
            'final_pred': final_logit,
            'final_logit': final_logit,
            'final_conf': final_conf,
            'coarse_pred': final_logit,
            'level_outputs': [],  # No level outputs for this simple model
            'patches': dummy_patches,
            'patch_labels': dummy_patch_labels,
            'patch_logits': dummy_patch_logits,
            'patch_coords': dummy_coords,
            'context_patches': None,
            'context_patch_labels': None,
            'context_patch_logits': None,
            'context_coords': None,
            'target_mask': None,
            'context_mask': None,
            'attn_weights': None,
            'register_tokens': None,
        }
