"""
Token-Based NMSW Architecture.

Instead of processing patches independently, this architecture treats
selected patches as tokens that can attend to each other via a transformer.

Architecture:
1. Global Branch: Coarse prediction + objectness scoring
2. Patch Selection: Gumbel top-k selects ~100 patches (8³ each)
3. Patch Tokenizer: CNN encodes each patch to an embedding
4. Cross-Patch Transformer: Patches attend to each other
5. Patch Decoder: Decode tokens back to segmentation masks
6. Aggregation: Reassemble into full volume
"""

import math
from typing import Dict, List, Optional, Tuple, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from .nmsw_sampling import GumbelTopK, PatchExtractor


# =============================================================================
# Positional Encodings (3 options)
# =============================================================================

class SinusoidalPositionalEncoding3D(nn.Module):
    """3D sinusoidal positional encoding (non-learnable).

    Extends the 1D sinusoidal encoding from "Attention Is All You Need"
    to 3D by concatenating separate encodings for each dimension.
    """

    def __init__(self, embed_dim: int, max_positions: int = 1000):
        super().__init__()
        self.embed_dim = embed_dim
        # Each dimension gets embed_dim // 3 features
        self.dim_per_axis = embed_dim // 3

        # Precompute frequency bands
        freq = torch.exp(
            torch.arange(0, self.dim_per_axis, 2).float() *
            (-math.log(10000.0) / self.dim_per_axis)
        )
        self.register_buffer('freq', freq)

    def forward(
        self,
        positions: torch.Tensor,  # [B, N, 3] - (d, h, w) coordinates
    ) -> torch.Tensor:
        """
        Args:
            positions: 3D coordinates [B, N, 3]

        Returns:
            Positional encodings [B, N, embed_dim]
        """
        B, N, _ = positions.shape
        device = positions.device

        encodings = []
        for axis in range(3):
            pos = positions[:, :, axis:axis+1]  # [B, N, 1]
            # Compute sin/cos for this axis
            angles = pos * self.freq.to(device)  # [B, N, dim_per_axis//2]
            sin_enc = torch.sin(angles)
            cos_enc = torch.cos(angles)
            enc = torch.cat([sin_enc, cos_enc], dim=-1)  # [B, N, dim_per_axis]
            encodings.append(enc)

        # Concatenate all axes
        pos_encoding = torch.cat(encodings, dim=-1)  # [B, N, 3*dim_per_axis]

        # Pad or truncate to embed_dim
        if pos_encoding.shape[-1] < self.embed_dim:
            padding = torch.zeros(B, N, self.embed_dim - pos_encoding.shape[-1], device=device)
            pos_encoding = torch.cat([pos_encoding, padding], dim=-1)
        elif pos_encoding.shape[-1] > self.embed_dim:
            pos_encoding = pos_encoding[:, :, :self.embed_dim]

        return pos_encoding


class LearnablePositionalEncoding3D(nn.Module):
    """Learnable positional encoding for 3D patch positions.

    Learns separate embeddings for each dimension and combines them.
    """

    def __init__(
        self,
        embed_dim: int,
        max_positions_per_dim: int = 64,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_pos = max_positions_per_dim

        # Learnable embeddings for each dimension
        self.d_embed = nn.Embedding(max_positions_per_dim, embed_dim // 3)
        self.h_embed = nn.Embedding(max_positions_per_dim, embed_dim // 3)
        self.w_embed = nn.Embedding(max_positions_per_dim, embed_dim // 3)

        # Projection to combine
        self.proj = nn.Linear((embed_dim // 3) * 3, embed_dim)

    def forward(
        self,
        positions: torch.Tensor,  # [B, N, 3] - (d, h, w) indices
    ) -> torch.Tensor:
        """
        Args:
            positions: 3D indices [B, N, 3] (integer indices, not coordinates)

        Returns:
            Positional encodings [B, N, embed_dim]
        """
        # Clamp to valid range
        positions = positions.long().clamp(0, self.max_pos - 1)

        d_enc = self.d_embed(positions[:, :, 0])  # [B, N, embed_dim//3]
        h_enc = self.h_embed(positions[:, :, 1])
        w_enc = self.w_embed(positions[:, :, 2])

        combined = torch.cat([d_enc, h_enc, w_enc], dim=-1)  # [B, N, embed_dim]
        return self.proj(combined)


class RelativePositionalEncoding3D(nn.Module):
    """Relative positional encoding based on patch-to-patch distances.

    Computes relative positions between all pairs of patches and
    adds them as bias to attention scores.
    """

    def __init__(
        self,
        num_heads: int,
        max_relative_distance: int = 32,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.max_dist = max_relative_distance

        # Learnable bias for each relative distance (per head)
        # Range: [-max_dist, max_dist] for each dimension
        table_size = 2 * max_relative_distance + 1
        self.rel_bias_d = nn.Embedding(table_size, num_heads)
        self.rel_bias_h = nn.Embedding(table_size, num_heads)
        self.rel_bias_w = nn.Embedding(table_size, num_heads)

    def forward(
        self,
        positions: torch.Tensor,  # [B, N, 3]
    ) -> torch.Tensor:
        """
        Args:
            positions: 3D indices [B, N, 3]

        Returns:
            Relative position bias [B, num_heads, N, N]
        """
        B, N, _ = positions.shape

        # Compute pairwise relative positions
        # positions[:, :, None, :] - positions[:, None, :, :] -> [B, N, N, 3]
        rel_pos = positions[:, :, None, :] - positions[:, None, :, :]  # [B, N, N, 3]

        # Clamp to valid range and shift to positive indices
        rel_pos = rel_pos.long().clamp(-self.max_dist, self.max_dist) + self.max_dist

        # Get bias for each dimension
        bias_d = self.rel_bias_d(rel_pos[:, :, :, 0])  # [B, N, N, num_heads]
        bias_h = self.rel_bias_h(rel_pos[:, :, :, 1])
        bias_w = self.rel_bias_w(rel_pos[:, :, :, 2])

        # Sum biases and transpose to [B, num_heads, N, N]
        total_bias = (bias_d + bias_h + bias_w).permute(0, 3, 1, 2)

        return total_bias


# =============================================================================
# Patch Tokenizer (CNN-based)
# =============================================================================

class PatchTokenizer(nn.Module):
    """CNN-based patch tokenizer.

    Converts 8³ patches into embedding vectors using a small 3D CNN.
    """

    def __init__(
        self,
        in_channels: int = 1,
        embed_dim: int = 256,
        patch_size: int = 8,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # Small 3D CNN encoder
        self.encoder = nn.Sequential(
            # 8 -> 4
            nn.Conv3d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.GELU(),

            # 4 -> 2
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.GELU(),

            # 2 -> 1
            nn.Conv3d(64, 128, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm3d(128),
            nn.GELU(),
        )

        # Final projection to embed_dim
        self.proj = nn.Linear(128, embed_dim)

    def forward(
        self,
        patches: torch.Tensor,  # [B, N, C, P, P, P]
    ) -> torch.Tensor:
        """
        Args:
            patches: Input patches [B, N, C, P, P, P]

        Returns:
            Token embeddings [B, N, embed_dim]
        """
        B, N, C, P, P, P = patches.shape

        # Reshape for batch processing
        x = patches.view(B * N, C, P, P, P)

        # Encode
        x = self.encoder(x)  # [B*N, 128, 1, 1, 1]
        x = x.view(B * N, -1)  # [B*N, 128]

        # Project
        x = self.proj(x)  # [B*N, embed_dim]

        return x.view(B, N, -1)  # [B, N, embed_dim]


# =============================================================================
# Cross-Patch Transformer
# =============================================================================

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with optional relative position bias."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        use_relative_pos: bool = False,
        max_relative_distance: int = 32,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

        self.use_relative_pos = use_relative_pos
        if use_relative_pos:
            self.rel_pos = RelativePositionalEncoding3D(num_heads, max_relative_distance)

    def forward(
        self,
        x: torch.Tensor,  # [B, N, embed_dim]
        positions: Optional[torch.Tensor] = None,  # [B, N, 3] for relative pos
    ) -> torch.Tensor:
        B, N, C = x.shape

        # Compute Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, heads, N, N]

        # Add relative position bias if enabled
        if self.use_relative_pos and positions is not None:
            rel_bias = self.rel_pos(positions)  # [B, heads, N, N]
            attn = attn + rel_bias

        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)

        return x


class TransformerBlock(nn.Module):
    """Transformer block with pre-norm architecture."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        use_relative_pos: bool = False,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(
            embed_dim, num_heads, dropout, use_relative_pos
        )
        self.norm2 = nn.LayerNorm(embed_dim)

        mlp_hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), positions)
        x = x + self.mlp(self.norm2(x))
        return x


class CrossPatchTransformer(nn.Module):
    """Full transformer for cross-patch attention.

    Processes patch tokens through multiple transformer layers,
    allowing patches to attend to each other.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        pos_encoding: Literal["sinusoidal", "learnable", "relative"] = "learnable",
        max_positions: int = 64,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.pos_encoding_type = pos_encoding

        # Positional encoding
        use_relative = pos_encoding == "relative"
        if pos_encoding == "sinusoidal":
            self.pos_encoder = SinusoidalPositionalEncoding3D(embed_dim, max_positions)
        elif pos_encoding == "learnable":
            self.pos_encoder = LearnablePositionalEncoding3D(embed_dim, max_positions)
        else:  # relative - handled in attention
            self.pos_encoder = None

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim, num_heads, mlp_ratio, dropout, use_relative
            )
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x: torch.Tensor,  # [B, N, embed_dim]
        positions: torch.Tensor,  # [B, N, 3]
    ) -> torch.Tensor:
        """
        Args:
            x: Patch token embeddings [B, N, embed_dim]
            positions: 3D positions of patches [B, N, 3]

        Returns:
            Transformed tokens [B, N, embed_dim]
        """
        # Add positional encoding (if not relative)
        if self.pos_encoder is not None:
            x = x + self.pos_encoder(positions)

        # Pass positions for relative encoding
        pos_for_attn = positions if self.pos_encoding_type == "relative" else None

        # Transformer blocks
        for block in self.blocks:
            x = block(x, pos_for_attn)

        return self.norm(x)


# =============================================================================
# Patch Decoder
# =============================================================================

class PatchDecoder(nn.Module):
    """Decode token embeddings back to patch segmentation masks.

    Uses a small 3D CNN decoder to upsample from embedding to patch size.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_classes: int = 1,
        patch_size: int = 8,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_classes = num_classes

        # Project embedding to spatial feature
        self.proj = nn.Linear(embed_dim, 128)

        # 3D CNN decoder
        self.decoder = nn.Sequential(
            # 1 -> 2
            nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm3d(64),
            nn.GELU(),

            # 2 -> 4
            nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm3d(32),
            nn.GELU(),

            # 4 -> 8
            nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2),
            nn.BatchNorm3d(16),
            nn.GELU(),

            # Final conv
            nn.Conv3d(16, num_classes, kernel_size=3, padding=1),
        )

    def forward(
        self,
        tokens: torch.Tensor,  # [B, N, embed_dim]
    ) -> torch.Tensor:
        """
        Args:
            tokens: Patch token embeddings [B, N, embed_dim]

        Returns:
            Patch segmentation masks [B, N, num_classes, P, P, P]
        """
        B, N, D = tokens.shape

        # Project to spatial feature
        x = self.proj(tokens)  # [B, N, 128]
        x = x.view(B * N, 128, 1, 1, 1)

        # Decode to patch size
        x = self.decoder(x)  # [B*N, num_classes, P, P, P]

        return x.view(B, N, self.num_classes, self.patch_size, self.patch_size, self.patch_size)


# =============================================================================
# Global Branch (lightweight segmentation for patch selection)
# =============================================================================

class GlobalBranch(nn.Module):
    """Lightweight encoder-decoder for coarse prediction and objectness scoring.

    Uses a simple UNet-like architecture instead of full SegFormer3D
    for efficiency.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 1,
        base_channels: int = 16,
    ):
        super().__init__()

        # Encoder
        self.enc1 = self._conv_block(in_channels, base_channels)
        self.enc2 = self._conv_block(base_channels, base_channels * 2)
        self.enc3 = self._conv_block(base_channels * 2, base_channels * 4)

        self.pool = nn.MaxPool3d(2)

        # Bottleneck
        self.bottleneck = self._conv_block(base_channels * 4, base_channels * 8)

        # Decoder
        self.up3 = nn.ConvTranspose3d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec3 = self._conv_block(base_channels * 8, base_channels * 4)

        self.up2 = nn.ConvTranspose3d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = self._conv_block(base_channels * 4, base_channels * 2)

        self.up1 = nn.ConvTranspose3d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = self._conv_block(base_channels * 2, base_channels)

        # Output heads
        self.seg_head = nn.Conv3d(base_channels, num_classes, kernel_size=1)
        self.obj_head = nn.Conv3d(base_channels, 1, kernel_size=1)  # Objectness scores

    def _conv_block(self, in_ch: int, out_ch: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.GELU(),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.GELU(),
        )

    def forward(
        self,
        x: torch.Tensor,  # [B, C, D, H, W]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input volume [B, C, D, H, W]

        Returns:
            seg_logits: Segmentation logits [B, num_classes, D, H, W]
            obj_logits: Objectness logits [B, 1, D, H, W]
        """
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        # Bottleneck
        b = self.bottleneck(self.pool(e3))

        # Decoder
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        # Output
        seg_logits = self.seg_head(d1)
        obj_logits = self.obj_head(d1)

        return seg_logits, obj_logits


# =============================================================================
# Main Model: TokenNMSW
# =============================================================================

class TokenNMSW(nn.Module):
    """Token-based NMSW model with cross-patch attention.

    Architecture:
    1. Global branch produces coarse prediction + objectness scores
    2. Gumbel top-k selects ~100 important patches
    3. CNN tokenizer converts patches to embeddings
    4. Transformer enables cross-patch attention
    5. Decoder predicts per-patch segmentation
    6. Aggregation combines patches into full volume
    """

    def __init__(
        self,
        # Volume parameters
        in_channels: int = 1,
        num_classes: int = 1,

        # Patch parameters
        patch_size: int = 8,
        num_patches: int = 100,
        num_random_patches: int = 0,

        # Global branch
        global_base_channels: int = 16,
        down_size_rate: Tuple[int, int, int] = (2, 2, 2),

        # Transformer parameters
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,

        # Positional encoding: "sinusoidal", "learnable", "relative"
        pos_encoding: str = "learnable",

        # Gumbel-softmax
        tau: float = 2/3,

        # Loss weights
        global_loss_weight: float = 1.0,
        local_loss_weight: float = 1.0,
        agg_loss_weight: float = 1.0,
        entropy_multiplier: float = 1e-5,
    ):
        super().__init__()

        self.patch_size = patch_size
        self.num_patches = num_patches
        self.num_random_patches = num_random_patches
        self.num_classes = num_classes
        self.down_size_rate = down_size_rate

        # Loss weights
        self.global_loss_weight = global_loss_weight
        self.local_loss_weight = local_loss_weight
        self.agg_loss_weight = agg_loss_weight
        self.entropy_multiplier = entropy_multiplier

        # Downsampler for global branch
        self.downsampler = nn.AvgPool3d(kernel_size=down_size_rate, stride=down_size_rate)

        # Global branch
        self.global_branch = GlobalBranch(in_channels, num_classes, global_base_channels)

        # Patch extraction
        self.patch_extractor = PatchExtractor(
            patch_size=(patch_size, patch_size, patch_size),
            overlap=0.0,  # No overlap for small patches
        )

        # Gumbel top-k
        self.gumbel_topk = GumbelTopK(tau=tau)

        # Patch tokenizer
        self.tokenizer = PatchTokenizer(in_channels, embed_dim, patch_size)

        # Cross-patch transformer
        self.transformer = CrossPatchTransformer(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            pos_encoding=pos_encoding,
        )

        # Patch decoder
        self.decoder = PatchDecoder(embed_dim, num_classes, patch_size)

    @property
    def tau(self):
        return self.gumbel_topk.tau

    @tau.setter
    def tau(self, value):
        self.gumbel_topk.tau = value

    def _extract_patch_positions(
        self,
        slice_meta: List[Tuple[slice, slice, slice]],
        device: torch.device,
    ) -> torch.Tensor:
        """Convert slice metadata to patch center positions."""
        positions = []
        for slices in slice_meta:
            # Get center of each slice
            d = (slices[0].start + slices[0].stop) // 2
            h = (slices[1].start + slices[1].stop) // 2
            w = (slices[2].start + slices[2].stop) // 2
            positions.append([d, h, w])
        return torch.tensor(positions, device=device, dtype=torch.float32)

    def _aggregate_patches(
        self,
        patch_masks: torch.Tensor,  # [B, N, num_classes, P, P, P]
        slice_meta: List[Tuple[slice, slice, slice]],
        vol_size: Tuple[int, int, int],
        global_logit: torch.Tensor,  # [B, num_classes, Dg, Hg, Wg]
        batch_size: int,
    ) -> torch.Tensor:
        """Aggregate patch predictions into full volume."""
        device = patch_masks.device
        B, N, C, P, P, P = patch_masks.shape

        # Initialize output and count
        output = torch.zeros(B, C, *vol_size, device=device)
        count = torch.zeros(B, 1, *vol_size, device=device)

        # Upsample global prediction
        global_upsampled = F.interpolate(
            global_logit, size=vol_size, mode='trilinear', align_corners=False
        )

        # Place each patch
        for b in range(B):
            for i in range(N):
                slices = slice_meta[i]
                output[b, :, slices[0], slices[1], slices[2]] += patch_masks[b, i]
                count[b, :, slices[0], slices[1], slices[2]] += 1

        # Average overlapping regions
        count = count.clamp(min=1)
        output = output / count

        # Fill uncovered regions with global prediction
        mask = (count > 0).float()
        output = output * mask + global_upsampled * (1 - mask)

        return output

    def forward(
        self,
        x: torch.Tensor,  # [B, C, D, H, W]
        labels: Optional[torch.Tensor] = None,
        mode: str = "train",
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input volume [B, C, D, H, W]
            labels: Ground truth labels (optional)
            mode: "train" or "test"

        Returns:
            Dict with predictions and intermediate outputs
        """
        B, C, D, H, W = x.shape
        vol_size = (D, H, W)
        device = x.device

        # 1. Global branch on downsampled input
        x_down = self.downsampler(x)
        global_logit, obj_logit = self.global_branch(x_down)

        # 2. Extract all patches from full-res input
        all_patches, all_slice_meta = self.patch_extractor.extract_patches(x)
        N_total = len(all_slice_meta)
        all_patches = all_patches.view(B, N_total, C, self.patch_size, self.patch_size, self.patch_size)

        # Also extract label patches if provided
        if labels is not None:
            all_label_patches, _ = self.patch_extractor.extract_patches(labels)
            LC = labels.shape[1]
            all_label_patches = all_label_patches.view(B, N_total, LC, self.patch_size, self.patch_size, self.patch_size)
        else:
            all_label_patches = None

        # 3. Compute objectness scores and select patches
        # Upsample objectness to match patch grid
        obj_upsampled = F.interpolate(obj_logit, size=vol_size, mode='trilinear', align_corners=False)

        # Pool objectness to patch-level scores
        patch_scores = []
        for slices in all_slice_meta:
            score = obj_upsampled[:, :, slices[0], slices[1], slices[2]].mean(dim=(2, 3, 4))
            patch_scores.append(score)
        patch_scores = torch.stack(patch_scores, dim=2).squeeze(1)  # [B, N_total]

        # Gumbel top-k selection
        k = self.num_patches
        one_hots, soft_hots = self.gumbel_topk(patch_scores, k=k, mode=mode)
        patch_indices = one_hots.detach().argmax(dim=-1)  # [K, B]

        # Gather selected patches
        selected_patches = []
        selected_label_patches = [] if labels is not None else None
        selected_slice_meta = []

        for ki in range(k):
            for bi in range(B):
                idx = patch_indices[ki, bi].item()
                selected_patches.append(all_patches[bi, idx])
                selected_slice_meta.append(all_slice_meta[idx])
                if all_label_patches is not None:
                    selected_label_patches.append(all_label_patches[bi, idx])

        # Reshape to [B, K, ...]
        selected_patches = torch.stack(selected_patches).view(B, k, C, self.patch_size, self.patch_size, self.patch_size)
        if selected_label_patches is not None:
            selected_label_patches = torch.stack(selected_label_patches).view(B, k, LC, self.patch_size, self.patch_size, self.patch_size)

        # 4. Get patch positions for positional encoding
        positions = self._extract_patch_positions(selected_slice_meta[:k], device)  # [K, 3]
        positions = positions.unsqueeze(0).expand(B, -1, -1)  # [B, K, 3]

        # 5. Tokenize patches
        tokens = self.tokenizer(selected_patches)  # [B, K, embed_dim]

        # 6. Cross-patch transformer
        tokens = self.transformer(tokens, positions)  # [B, K, embed_dim]

        # 7. Decode to patch masks
        patch_masks = self.decoder(tokens)  # [B, K, num_classes, P, P, P]

        # 8. Aggregate into full volume
        # Reorganize slice_meta for aggregation (per batch)
        batch_slice_meta = selected_slice_meta[:k]  # Same for all batches in this impl
        final_logit = self._aggregate_patches(
            patch_masks, batch_slice_meta, vol_size, global_logit, B
        )

        return {
            'final_logit': final_logit,
            'global_logit': global_logit,
            'patch_masks': patch_masks,
            'label_patches': selected_label_patches,
            'sample_probs': soft_hots,
            'slice_meta': selected_slice_meta,
            'positions': positions,
        }

    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        criterion: nn.Module,
    ) -> Dict[str, torch.Tensor]:
        """Compute multi-component loss."""
        B = labels.shape[0]

        # 1. Global loss
        labels_down = F.interpolate(
            labels.float(),
            size=outputs['global_logit'].shape[2:],
            mode='nearest',
        )
        global_loss = criterion(outputs['global_logit'], labels_down)

        # 2. Local loss (on selected patches)
        if outputs['label_patches'] is not None:
            # Reshape for loss computation
            patch_masks = outputs['patch_masks']  # [B, K, C, P, P, P]
            label_patches = outputs['label_patches']  # [B, K, C, P, P, P]

            B, K, C, P, _, _ = patch_masks.shape
            patch_masks_flat = patch_masks.view(B * K, C, P, P, P)
            label_patches_flat = label_patches.view(B * K, -1, P, P, P)

            local_loss = criterion(patch_masks_flat, label_patches_flat)
        else:
            local_loss = torch.tensor(0.0, device=labels.device)

        # 3. Aggregation loss
        agg_loss = criterion(outputs['final_logit'], labels)

        # 4. Entropy regularization
        sample_probs = outputs['sample_probs']
        eps = 1e-20
        entropy = -(sample_probs + eps) * torch.log(sample_probs + eps)
        entropy = entropy.sum(dim=-1).mean()

        # Total loss
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
