"""Loss functions for segmentation and regression tasks."""
from __future__ import annotations

import torch
import torch.nn as nn
from typing import Dict, Optional

class SoftDiceLoss(nn.Module):
    """Pure PyTorch implementation of Soft Dice Loss."""
    def __init__(self, p: int = 1, smooth: float = 1e-5) -> None:
        super().__init__()
        self.p = p
        self.smooth = smooth

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Standardize shapes to (Batch, Flattened_Pixels)
        # Assuming predictions are LOGITS
        probs = torch.sigmoid(predictions)
        
        # Flatten
        probs = probs.view(probs.shape[0], -1)
        targets = targets.view(targets.shape[0], -1)

        intersection = (probs * targets).sum(dim=1)
        
        if self.p == 2:
            denominator = (probs.pow(2) + targets.pow(2)).sum(dim=1)
        else:
            denominator = (probs + targets).sum(dim=1)

        dice_score = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        return 1.0 - dice_score.mean()


class SoftDiceBCELoss(nn.Module):
    """Hybrid Loss: Soft Dice + Binary Cross Entropy.
    Provides the global overlap optimization of Dice and the local pixel 
    stability of BCE.
    """
    def __init__(self, dice_weight: float = 1.0, bce_weight: float = 1.0, smooth: float = 1e-5) -> None:
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice = SoftDiceLoss(smooth=smooth)
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets_float = targets.float()
        # Clamp logits to prevent NaN from extreme aggregator values
        predictions = predictions.clamp(-10.0, 10.0)

        dice_loss = self.dice(predictions, targets_float)
        bce_loss = self.bce(predictions, targets_float)

        return (self.dice_weight * dice_loss) + (self.bce_weight * bce_loss)

class SoftDiceFocalLoss(nn.Module):
    """Hybrid Loss: Soft Dice + Focal Loss.
    Focal loss down-weights easy background pixels, addressing foreground/background
    imbalance better than BCE for small structures.
    """
    def __init__(
        self,
        dice_weight: float = 1.0,
        focal_weight: float = 1.0,
        smooth: float = 1e-5,
        alpha: float = 0.25,
        gamma: float = 2.0,
    ) -> None:
        super().__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.dice = SoftDiceLoss(smooth=smooth)
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets_float = targets.float()
        predictions = predictions.clamp(-10.0, 10.0)

        dice_loss = self.dice(predictions, targets_float)

        # Focal loss (inline to avoid extra sigmoid call)
        probs = torch.sigmoid(predictions)
        ce = nn.functional.binary_cross_entropy_with_logits(
            predictions, targets_float, reduction="none"
        )
        p_t = probs * targets_float + (1 - probs) * (1 - targets_float)
        alpha_t = self.alpha * targets_float + (1 - self.alpha) * (1 - targets_float)
        focal_loss = (alpha_t * (1 - p_t) ** self.gamma * ce).mean()

        return (self.dice_weight * dice_loss) + (self.focal_weight * focal_loss)


class CrossEntropyLoss(nn.Module):
    """Cross-entropy loss wrapper for semantic segmentation."""

    def __init__(self) -> None:
        super().__init__()
        self._loss = nn.CrossEntropyLoss(reduction="mean")

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute cross-entropy loss."""
        return self._loss(predictions, targets)


class BinaryCrossEntropyWithLogits(nn.Module):
    """Binary cross-entropy with logits for binary segmentation tasks."""

    def __init__(self) -> None:
        super().__init__()
        self._loss = nn.BCEWithLogitsLoss(reduction="mean")

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute BCE with logits loss."""
        return self._loss(predictions, targets)


class DiceLoss(nn.Module):
    """Dice loss for segmentation.

    Computes: 1 - (2 * intersection + smooth) / (sum_pred + sum_target + smooth)
    """

    def __init__(
        self,
        smooth: float = 1e-5,
        apply_sigmoid: bool = True,
        squared: bool = False,
    ) -> None:
        """
        Args:
            smooth: Smoothing factor to avoid division by zero
            apply_sigmoid: If True, apply sigmoid to predictions
            squared: If True, square the denominator terms (Dice^2)
        """
        super().__init__()
        self.smooth = smooth
        self.apply_sigmoid = apply_sigmoid
        self.squared = squared

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute Dice loss.

        Args:
            predictions: Model predictions - supports various shapes:
                - (B, N) - flattened predictions
                - (B, C, ...) - spatial predictions with channels
            targets: Ground truth labels (same shape as predictions)

        Returns:
            Scalar loss tensor (1 - Dice score)
        """
        if self.apply_sigmoid:
            predictions = torch.sigmoid(predictions)

        # Handle different input shapes
        if predictions.dim() == 2:
            # Flattened input [B, N] - compute dice per sample
            intersection = (predictions * targets).sum(dim=1)
            if self.squared:
                pred_sum = (predictions ** 2).sum(dim=1)
                target_sum = (targets ** 2).sum(dim=1)
            else:
                pred_sum = predictions.sum(dim=1)
                target_sum = targets.sum(dim=1)
            dice = (2.0 * intersection + self.smooth) / (pred_sum + target_sum + self.smooth)
        else:
            # Spatial input [B, C, ...] - flatten spatial dims, compute per channel
            predictions = predictions.flatten(2)  # [B, C, N]
            targets = targets.flatten(2)  # [B, C, N]

            intersection = (predictions * targets).sum(dim=2)
            if self.squared:
                pred_sum = (predictions ** 2).sum(dim=2)
                target_sum = (targets ** 2).sum(dim=2)
            else:
                pred_sum = predictions.sum(dim=2)
                target_sum = targets.sum(dim=2)
            dice = (2.0 * intersection + self.smooth) / (pred_sum + target_sum + self.smooth)

        # Average over batch (and channels if present)
        return 1.0 - dice.mean()


class DiceBCELoss(nn.Module):
    """Combined Dice and Binary Cross-Entropy loss."""

    def __init__(
        self,
        dice_weight: float = 0.5,
        bce_weight: float = 0.5,
        smooth: float = 1e-5,
    ) -> None:
        """
        Args:
            dice_weight: Weight for Dice loss component
            bce_weight: Weight for BCE loss component
            smooth: Smoothing factor for Dice loss
        """
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice_loss = DiceLoss(smooth=smooth, apply_sigmoid=True)
        self.bce_loss = nn.BCEWithLogitsLoss(reduction="mean")

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute combined Dice-BCE loss."""
        dice = self.dice_loss(predictions, targets)
        bce = self.bce_loss(predictions, targets)
        return self.dice_weight * dice + self.bce_weight * bce


class SmoothL1Loss(nn.Module):
    """Pixel-wise Smooth L1 (Huber) loss for regression tasks.

    Useful for RGB mask prediction where we want to regress color values.
    Less sensitive to outliers than MSE.
    """

    def __init__(self, beta: float = 1.0, apply_sigmoid: bool = True) -> None:
        """
        Args:
            beta: Threshold at which to change from L2 to L1 loss
            apply_sigmoid: If True, apply sigmoid to predictions before loss
        """
        super().__init__()
        self.beta = beta
        self.apply_sigmoid = apply_sigmoid
        self._loss = nn.SmoothL1Loss(reduction="mean", beta=beta)

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute pixel-wise Smooth L1 loss."""
        if self.apply_sigmoid:
            predictions = torch.sigmoid(predictions)
        return self._loss(predictions, targets)


class MSELoss(nn.Module):
    """Pixel-wise Mean Squared Error loss for regression tasks."""

    def __init__(self, apply_sigmoid: bool = True) -> None:
        """
        Args:
            apply_sigmoid: If True, apply sigmoid to predictions before loss
        """
        super().__init__()
        self.apply_sigmoid = apply_sigmoid
        self._loss = nn.MSELoss(reduction="mean")

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute pixel-wise MSE loss."""
        if self.apply_sigmoid:
            predictions = torch.sigmoid(predictions)
        return self._loss(predictions, targets)


class L1Loss(nn.Module):
    """Pixel-wise L1 (Mean Absolute Error) loss for regression tasks."""

    def __init__(self, apply_sigmoid: bool = True) -> None:
        """
        Args:
            apply_sigmoid: If True, apply sigmoid to predictions before loss
        """
        super().__init__()
        self.apply_sigmoid = apply_sigmoid
        self._loss = nn.L1Loss(reduction="mean")

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute pixel-wise L1 loss."""
        if self.apply_sigmoid:
            predictions = torch.sigmoid(predictions)
        return self._loss(predictions, targets)


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance.

    Reduces loss for well-classified examples, focusing on hard examples.
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        apply_sigmoid: bool = True,
    ) -> None:
        """
        Args:
            alpha: Weighting factor for positive class
            gamma: Focusing parameter (higher = more focus on hard examples)
            apply_sigmoid: If True, apply sigmoid to predictions
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.apply_sigmoid = apply_sigmoid

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss."""
        if self.apply_sigmoid:
            probs = torch.sigmoid(predictions)
        else:
            probs = predictions

        # Binary focal loss
        ce_loss = nn.functional.binary_cross_entropy_with_logits(
            predictions, targets, reduction="none"
        )
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma

        return (focal_weight * ce_loss).mean()


def build_loss_fn(loss_type: str, loss_args: Optional[Dict] = None) -> nn.Module:
    """Factory function to build loss functions.

    Args:
        loss_type: Type of loss function. Supported types:
            - "softdice": SoftDiceLoss,        # Pure PyTorch version
            - "softdiceBCE": SoftDiceBCELoss,  # Combined version
            - 'crossentropy': Cross-entropy loss
            - 'binarycrossentropy': Binary cross-entropy with logits
            - 'dice': Dice loss
            - 'diceBCE': Combined Dice + BCE loss
            - 'smoothl1': Smooth L1 (Huber) loss for regression
            - 'mse': Mean Squared Error loss for regression
            - 'l1': L1 (Mean Absolute Error) loss for regression
            - 'focal': Focal loss for class imbalance
        loss_args: Additional arguments for loss function. Examples:
            - dice: {'smooth': 1e-5, 'apply_sigmoid': True, 'squared': False}
            - diceBCE: {'dice_weight': 0.5, 'bce_weight': 0.5}
            - smoothl1: {'beta': 1.0, 'apply_sigmoid': True}
            - mse/l1: {'apply_sigmoid': True}
            - focal: {'alpha': 0.25, 'gamma': 2.0}

    Returns:
        Instantiated loss module

    Raises:
        ValueError: If loss_type is not supported
    """
    loss_registry = {
        "softdice": SoftDiceLoss,
        "softdiceBCE": SoftDiceBCELoss,
        "softdiceFocal": SoftDiceFocalLoss,
        "crossentropy": CrossEntropyLoss,
        "binarycrossentropy": BinaryCrossEntropyWithLogits,
        "dice": DiceLoss,
        "diceBCE": DiceBCELoss,
        "diceCE": DiceBCELoss,  # Alias for compatibility
        "smoothl1": SmoothL1Loss,
        "mse": MSELoss,
        "l1": L1Loss,
        "focal": FocalLoss,
    }

    if loss_type not in loss_registry:
        raise ValueError(
            f"Unsupported loss type: {loss_type}. "
            f"Supported types: {list(loss_registry.keys())}"
        )

    if loss_args is not None:
        return loss_registry[loss_type](**loss_args)
    return loss_registry[loss_type]()
