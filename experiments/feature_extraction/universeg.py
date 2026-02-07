"""Explore per-image feature extraction from UniverSeg encoder."""
import sys
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, "/work/dlclarge2/ndirt-SegFM3D/repos/UniverSeg")

from universeg import universeg

# ── Load pretrained model ──
device = "cuda" if torch.cuda.is_available() else "cpu"
model = universeg(pretrained=True).to(device).eval()

print("=== UniverSeg Architecture ===")
print(f"Encoder blocks: {len(model.enc_blocks)}")
print(f"Decoder blocks: {len(model.dec_blocks)}")

# ── Dummy input (UniverSeg expects 128x128, [0,1] normalized) ──
B, S, H, W = 2, 3, 128, 128
target_image = torch.rand(B, 1, H, W, device=device)
support_images = torch.rand(B, S, 1, H, W, device=device)
support_labels = torch.zeros(B, S, 1, H, W, device=device)

# ── Hook encoder blocks to capture per-image (target) features ──
import einops as E

encoder_features = {}

def make_hook(name):
    def hook(module, input, output):
        target, support = output
        # target: [B, 1, C, H, W] → squeeze the group dim
        encoder_features[name] = target[:, 0].detach().cpu()  # [B, C, H, W]
    return hook

hooks = []
for i, block in enumerate(model.enc_blocks):
    hooks.append(block.register_forward_hook(make_hook(f"enc_{i}")))

# Also hook decoder blocks
decoder_features = {}
def make_dec_hook(name):
    def hook(module, input, output):
        target, support = output
        decoder_features[name] = target[:, 0].detach().cpu()
    return hook

for i, block in enumerate(model.dec_blocks):
    hooks.append(block.register_forward_hook(make_dec_hook(f"dec_{i}")))

# ── Forward pass ──
with torch.no_grad():
    output = model(target_image, support_images, support_labels)

for h in hooks:
    h.remove()

# ── Print feature shapes ──
print(f"\nInput:  target {tuple(target_image.shape)}, support {tuple(support_images.shape)}")
print(f"Output: {tuple(output.shape)}")

print("\n=== Encoder features (per-image) ===")
for name, feat in encoder_features.items():
    print(f"  {name}: {tuple(feat.shape)}")

print("\n=== Decoder features (per-image) ===")
for name, feat in decoder_features.items():
    print(f"  {name}: {tuple(feat.shape)}")

# ── Visualize features from each encoder level ──
ncols = len(encoder_features) + len(decoder_features)
fig, axes = plt.subplots(2, ncols, figsize=(3 * ncols, 6))

all_feats = list(encoder_features.items()) + list(decoder_features.items())
for col, (name, feat) in enumerate(all_feats):
    # feat: [B, C, H, W] — show first sample, mean over channels
    mean_feat = feat[0].mean(dim=0).numpy()  # [H, W]
    axes[0, col].imshow(mean_feat, cmap="viridis")
    axes[0, col].set_title(f"{name}\n{tuple(feat.shape[1:])}")
    axes[0, col].axis("off")

    # Show std over channels
    std_feat = feat[0].std(dim=0).numpy()
    axes[1, col].imshow(std_feat, cmap="magma")
    axes[1, col].set_title(f"std")
    axes[1, col].axis("off")

axes[0, 0].set_ylabel("mean(C)")
axes[1, 0].set_ylabel("std(C)")
plt.suptitle("UniverSeg per-image features (random input)")
plt.tight_layout()
save_path = Path(__file__).parent / "universeg_features.png"
plt.savefig(save_path, dpi=150)
print(f"\nSaved to {save_path}")
