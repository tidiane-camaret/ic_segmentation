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


class PatchEmbedding2D(nn.Module):
    """Embed 2D image patches into tokens."""

    def __init__(self, patch_size: int = 8, in_channels: int = 1, embed_dim: int = 256):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        # x: [B, C, H, W] -> [B, embed_dim, H//ps, W//ps]
        x = self.proj(x)
        # Flatten to [B, num_patches, embed_dim]
        x = x.flatten(2).transpose(1, 2)
        return x


class LocalTransformer(nn.Module):
    """Shallow transformer for processing local patches."""

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection to segmentation logits per token
        self.seg_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, x):
        # x: [B, num_tokens, embed_dim]
        x = self.transformer(x)
        logits = self.seg_head(x)  # [B, num_tokens, 1]
        return logits.squeeze(-1)


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
        self.local_transformer = LocalTransformer(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
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
        """Select K patches based on coarse prediction foreground scores."""
        B, C, H, W = image.shape
        ps = self.patch_size
        K = self.num_patches

        num_h, num_w = H // ps, W // ps

        all_patches = []
        all_labels = []
        all_coords = []

        for b in range(B):
            scores = []
            coords = []

            for hi in range(num_h):
                for wi in range(num_w):
                    # Map to coarse prediction coords
                    ch = hi * ps // self.coarse_scale
                    cw = wi * ps // self.coarse_scale
                    ch_end = min((hi + 1) * ps // self.coarse_scale, coarse_pred.shape[2])
                    cw_end = min((wi + 1) * ps // self.coarse_scale, coarse_pred.shape[3])

                    region = coarse_pred[b, 0, ch:ch_end, cw:cw_end]
                    score = region.mean().item() if region.numel() > 0 else 0

                    scores.append(score)
                    coords.append((hi * ps, wi * ps))

            scores = torch.tensor(scores, device=image.device)
            if self.training:
                scores = scores + torch.rand_like(scores) * 0.1

            _, indices = torch.topk(scores, min(K, len(scores)))

            batch_patches = []
            batch_labels = []
            batch_coords = []
            for idx in indices:
                h, w = coords[idx]
                patch = image[b, :, h:h+ps, w:w+ps]
                label_patch = labels[b, :, h:h+ps, w:w+ps]
                batch_patches.append(patch)
                batch_labels.append(label_patch)
                batch_coords.append([h, w])

            while len(batch_patches) < K:
                batch_patches.append(torch.zeros(C, ps, ps, device=image.device))
                batch_labels.append(torch.zeros(1, ps, ps, device=image.device))
                batch_coords.append([0, 0])

            all_patches.append(torch.stack(batch_patches))
            all_labels.append(torch.stack(batch_labels))
            all_coords.append(torch.tensor(batch_coords, device=image.device))

        patches = torch.stack(all_patches)      # [B, K, C, ps, ps]
        patch_labels = torch.stack(all_labels)  # [B, K, 1, ps, ps]
        coords = torch.stack(all_coords)        # [B, K, 2]

        return patches, patch_labels, coords

    def local_branch(self, patches: torch.Tensor) -> torch.Tensor:
        """Process all patches as tokens in a single transformer forward pass."""
        B, K, C, ps, _ = patches.shape
        ips = self.internal_patch_size
        grid_size = ps // ips
        tokens_per_patch = self.num_tokens

        # Embed all patches: [B, K, C, ps, ps] -> [B*K, C, ps, ps] -> [B*K, tokens_per_patch, embed_dim]
        patches_flat = patches.reshape(B * K, C, ps, ps)
        all_tokens = self.patch_embed(patches_flat)  # [B*K, tokens_per_patch, embed_dim]

        # Reshape to [B, K*tokens_per_patch, embed_dim] for cross-patch attention
        all_tokens = all_tokens.reshape(B, K * tokens_per_patch, -1)

        # Add position embeddings (tiled for each patch)
        pos_embed_tiled = self.pos_embed.repeat(1, K, 1)  # [1, K*tokens_per_patch, embed_dim]
        all_tokens = all_tokens + pos_embed_tiled

        # Single transformer forward pass with cross-patch attention
        all_logits = self.local_transformer(all_tokens)  # [B, K*tokens_per_patch]

        # Reshape back to per-patch logits and upsample
        all_logits = all_logits.view(B, K, tokens_per_patch)  # [B, K, tokens_per_patch]
        all_logits = all_logits.view(B * K, 1, grid_size, grid_size)
        all_logits = self.decoder(all_logits)  # [B*K, 1, ps, ps]
        all_logits = all_logits.view(B, K, 1, ps, ps)

        return all_logits  # [B, K, 1, ps, ps]

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
        patch_logits = self.local_branch(patches)

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
