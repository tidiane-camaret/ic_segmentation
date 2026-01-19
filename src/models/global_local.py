"""
Global-Local Architecture for 2D Medical Image Segmentation

Architecture:
1. Global Branch: Coarse prediction (uses GT mask for now)
2. Patch Selection: Selects K patches based on global prediction
3. Local Branch: Shallow transformer processes patches
4. Aggregation: (not implemented yet)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.local import (
    LocalDino,
    LocalDinoLight,
    LocalTransformer,
    PatchEmbedding2D,
)


class GlobalLocalModel(nn.Module):
    """
    Global-Local 2D segmentation model.

    Global branch: Uses GT mask (oracle) to produce coarse prediction.
    Patch selection: Selects K patches based on coarse prediction.
    Local branch: Shallow transformer refines selected patches.
    """

    def __init__(self, config: dict):
        """
        Args:
            config: Config dict with 'global' and 'local' sections
        """
        super().__init__()

        global_cfg = config.get("global", {})
        local_cfg = config.get("local", {})

        # Global branch params
        self.patch_size = global_cfg.get("patch_size", 16)
        self.num_patches = global_cfg.get("num_train_patches", 16)
        self.coarse_scale = global_cfg.get("down_size_rate", 4)

        # Local branch params
        embed_dim = local_cfg.get("embed_dim", 256)
        num_heads = local_cfg.get("num_heads", 4)
        num_layers = local_cfg.get("num_layers", 2)
        in_channels = local_cfg.get("in_channels", 1)

        # Patch embedding (internal tokenization within each selected patch)
        internal_patch_size = min(4, self.patch_size)
        self.internal_patch_size = internal_patch_size
        self.patch_embed = PatchEmbedding2D(
            patch_size=internal_patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )
        self.local_transformer = LocalDino(
            pretrained_path="/nfs/data/nii/data1/Analysis/camaret___in_context_segmentation/ANALYSIS_20251122/checkpoints/models--facebook--dinov3-vitl16-pretrain-lvd1689m/snapshots/ea8dc2863c51be0a264bab82070e3e8836b02d51"
        )

        # Position embeddings
        num_tokens = max(1, (self.patch_size // internal_patch_size) ** 2)
        self.num_tokens = num_tokens
        self.pos_embed = nn.Parameter(torch.randn(1, num_tokens, embed_dim) * 0.02)

        # Decoder to upsample token logits back to patch size
        self.decoder = nn.ConvTranspose2d(
            1, 1,
            kernel_size=internal_patch_size,
            stride=internal_patch_size
        )

    def global_branch(self, gt_mask: torch.Tensor) -> torch.Tensor:
        """Global branch: downsample GT mask (oracle)."""
        # Ensure 4D: [B, C, H, W]
        if gt_mask.dim() == 3:
            gt_mask = gt_mask.unsqueeze(1)
        coarse = F.avg_pool2d(gt_mask.float(), self.coarse_scale)
        return coarse

    def select_patches(
        self,
        image: torch.Tensor,
        labels: torch.Tensor,
        coarse_pred: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Efficiently select K patches based on coarse prediction scores.
        Uses vectorized pooling and interpolation to avoid python loops.
        """
        B, C, H, W = image.shape
        ps = self.patch_size
        K = self.num_patches
        
        # Stride for dense sampling
        stride = max(1, ps // 8)

        # 1. GENERATE SCORE MAP (Vectorized)
        # ---------------------------------------------------------
        # Instead of looping, we use AvgPool to calculate regional scores.
        # We map the patch size to the coarse feature map scale.
        coarse_kernel = max(1, ps // self.coarse_scale)
        
        # Slice channel 0 (foreground) and keep dim: [B, 1, H_c, W_c]
        fg_pred = coarse_pred[:, 0:1, :, :] 

        # Compute "patch scores" on the coarse grid using pooling
        # stride=1 on coarse map ensures we get a score for every coarse location
        coarse_scores = F.avg_pool2d(fg_pred, kernel_size=coarse_kernel, stride=1)

        # Calculate the target dimensions of our dense sampling grid
        grid_h = (H - ps) // stride + 1
        grid_w = (W - ps) // stride + 1

        # Interpolate scores to match the dense grid size
        # This approximates the score for every possible stride location
        scores_map = F.interpolate(
            coarse_scores, 
            size=(grid_h, grid_w), 
            mode='bilinear', 
            align_corners=False
        )
        
        # Flatten spatial dimensions: [B, 1, GridH, GridW] -> [B, N_candidates]
        flat_scores = scores_map.flatten(2).squeeze(1)

        # 2. SELECT INDICES
        # ---------------------------------------------------------
        if self.training:
            # Add noise to randomize selection among high-score patches
            # (Satisfies "high score but not necessarily the top ones")
            noise = torch.rand_like(flat_scores) * 0.1
            flat_scores = flat_scores + noise

        # Select Top K indices per batch element
        # This is much faster than sorting a list in Python
        num_candidates = flat_scores.shape[1]
        k_safe = min(K, num_candidates)
        
        top_indices = torch.multinomial(F.softmax(flat_scores, dim=1), k_safe)

        # 3. EXTRACT PATCHES
        # ---------------------------------------------------------
        # Convert flattened indices back to (h, w) coordinates in image space
        row_indices = top_indices // grid_w
        col_indices = top_indices % grid_w
        
        h_starts = row_indices * stride
        w_starts = col_indices * stride

        # We collect patches in lists. Because K is small (e.g. 16-64), 
        # a loop here is negligible compared to the scoring loop we removed.
        all_patches = []
        all_labels = []
        all_coords = []

        for b in range(B):
            batch_patches = []
            batch_labels = []
            batch_coords = []
            
            for k in range(k_safe):
                h, w = h_starts[b, k].item(), w_starts[b, k].item()
                
                # Extract patch
                patch = image[b, :, h : h + ps, w : w + ps]
                label_patch = labels[b, :, h : h + ps, w : w + ps]
                
                batch_patches.append(patch)
                batch_labels.append(label_patch)
                batch_coords.append([h, w])
            
            # Padding if fewer than K candidates (rare, but good for safety)
            while len(batch_patches) < K:
                batch_patches.append(torch.zeros(C, ps, ps, device=image.device))
                batch_labels.append(torch.zeros(1, ps, ps, device=image.device))
                batch_coords.append([0, 0])

            all_patches.append(torch.stack(batch_patches))
            all_labels.append(torch.stack(batch_labels))
            all_coords.append(torch.tensor(batch_coords, device=image.device))

        return (
            torch.stack(all_patches),   # [B, K, C, ps, ps]
            torch.stack(all_labels),    # [B, K, 1, ps, ps]
            torch.stack(all_coords)     # [B, K, 2]
        )

    def local_branch(self, patches: torch.Tensor, coords: torch.Tensor = None) -> torch.Tensor:
        """Process patches through local model (LocalDino or LocalTransformer)."""
        # LocalDino/LocalDinoLight expect [B, K, C, ps, ps] and coords
        # They return [B, K, num_classes, ps, ps]
        if hasattr(self.local_transformer, 'backbone') or hasattr(self.local_transformer, 'patch_embed'):
            # LocalDino or LocalDinoLight
            return self.local_transformer(patches, coords)
        else:
            # Fallback for LocalTransformer (token-based)
            B, K, C, ps, _ = patches.shape
            ips = self.internal_patch_size
            grid_size = ps // ips
            tokens_per_patch = self.num_tokens

            patches_flat = patches.reshape(B * K, C, ps, ps)
            all_tokens = self.patch_embed(patches_flat)
            all_tokens = all_tokens.reshape(B, K * tokens_per_patch, -1)

            pos_embed_tiled = self.pos_embed.repeat(1, K, 1)
            all_tokens = all_tokens + pos_embed_tiled

            all_logits = self.local_transformer(all_tokens)

            all_logits = all_logits.view(B, K, tokens_per_patch)
            all_logits = all_logits.view(B * K, 1, grid_size, grid_size)
            all_logits = self.decoder(all_logits)
            all_logits = all_logits.view(B, K, 1, ps, ps)

            return all_logits

    def forward(
        self,
        image: torch.Tensor,
        labels: torch.Tensor = None,
        mode: str = "train",
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            image: Input image [B, C, H, W]
            labels: Ground truth mask [B, 1, H, W] or [B, H, W]
            mode: "train" or "test"
        """
        # Ensure labels have channel dim
        if labels is not None and labels.dim() == 3:
            labels = labels.unsqueeze(1)

        # Use labels for global branch (oracle), or zeros for inference
        if labels is not None:
            coarse_pred = self.global_branch(labels)
        else:
            # For inference without GT, use uniform sampling (placeholder)
            coarse_pred = torch.ones(
                image.shape[0], 1,
                image.shape[2] // self.coarse_scale,
                image.shape[3] // self.coarse_scale,
                device=image.device
            ) * 0.5

        # Use labels for patch selection too (needed for local loss)
        if labels is None:
            labels = torch.zeros(
                image.shape[0], 1, image.shape[2], image.shape[3],
                device=image.device
            )

        patches, patch_labels, coords = self.select_patches(image, labels, coarse_pred)
        patch_logits = self.local_branch(patches, coords)

        # Create full output volume (for final_logit)
        B, C, H, W = image.shape
        final_logit = torch.zeros(B, 1, H, W, device=image.device)
        counts = torch.zeros(B, 1, H, W, device=image.device)
        ps = self.patch_size

        for b in range(B):
            for k in range(coords.shape[1]):
                h, w = coords[b, k].tolist()
                final_logit[b, :, h:h+ps, w:w+ps] += patch_logits[b, k]
                counts[b, :, h:h+ps, w:w+ps] += 1

        counts = counts.clamp(min=1)
        final_logit = final_logit / counts

        return {
            "coarse_pred": coarse_pred,
            "patches": patches,
            "patch_labels": patch_labels,
            "patch_coords": coords,
            "patch_logits": patch_logits,
            "final_logit": final_logit,
        }

    def compute_loss(
        self,
        outputs: dict[str, torch.Tensor],
        labels: torch.Tensor,
        criterion,
    ) -> dict[str, torch.Tensor]:
        """Compute losses for training."""
        # Ensure labels have channel dim
        if labels.dim() == 3:
            labels = labels.unsqueeze(1)

        # Global loss: coarse prediction vs downsampled GT
        coarse_gt = F.avg_pool2d(labels.float(), self.coarse_scale)
        global_loss = criterion(outputs["coarse_pred"], coarse_gt)

        # Local loss: patch predictions vs patch labels
        patch_logits = outputs["patch_logits"]  # [B, K, 1, ps, ps]
        patch_labels = outputs["patch_labels"]  # [B, K, 1, ps, ps]
        B, K = patch_logits.shape[:2]
        local_loss = criterion(
            patch_logits.view(B * K, -1),
            patch_labels.view(B * K, -1)
        )

        # Aggregation loss (placeholder - just final vs GT)
        agg_loss = criterion(outputs["final_logit"], labels)

        total_loss = global_loss + local_loss + agg_loss

        return {
            "total_loss": total_loss,
            "global_loss": global_loss,
            "local_loss": local_loss,
            "agg_loss": agg_loss,
        }
