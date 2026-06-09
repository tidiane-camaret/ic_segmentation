"""
UniverSeg scaling study: FLOPs and latency vs image size and context size.

Theory:
- FLOPs ∝ S   (CrossConv2d computes 1×S pairwise convolutions; linear in support size)
- FLOPs ∝ H²  (fully-convolutional U-Net; each level's cost sums to ~4/3 * H²)
"""

import os
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from torch.profiler import ProfilerActivity, profile

# This file is named universeg.py, so we must remove its own directory from
# sys.path before importing the real universeg package to avoid a circular import.
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir in sys.path:
    sys.path.remove(_script_dir)
sys.path.insert(0, "/home/dpxuser/repos/UniverSeg")

from universeg.model import UniverSeg


# ── helpers ──────────────────────────────────────────────────────────────────

def make_inputs(B, S, H, device):
    target     = torch.randn(B, 1, H, H, device=device)
    supp_imgs  = torch.randn(B, S, 1, H, H, device=device)
    supp_lbls  = torch.zeros(B, S, 1, H, H, device=device)
    return target, supp_imgs, supp_lbls


def count_flops(model, target, supp_imgs, supp_lbls):
    """One forward pass through torch.profiler; return total FLOPs."""
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 with_flops=True, record_shapes=False) as prof:
        with torch.no_grad():
            model(target, supp_imgs, supp_lbls)
    torch.cuda.synchronize()
    return sum(e.flops for e in prof.key_averages())


def measure(model, target, supp_imgs, supp_lbls, n_warmup=5, n_runs=20):
    """Return (GFLOPs, mean_ms, std_ms) for a single forward pass."""
    # warmup
    for _ in range(n_warmup):
        with torch.no_grad():
            model(target, supp_imgs, supp_lbls)
    torch.cuda.synchronize()

    flops = count_flops(model, target, supp_imgs, supp_lbls)

    # peak VRAM during one forward pass (excludes model weights)
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    with torch.no_grad():
        model(target, supp_imgs, supp_lbls)
    torch.cuda.synchronize()
    vram_mb = (torch.cuda.max_memory_allocated() - baseline) / 1024 ** 2

    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(n_runs):
        start.record()
        with torch.no_grad():
            model(target, supp_imgs, supp_lbls)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))  # ms

    return flops / 1e9, float(np.mean(times)), float(np.std(times)), float(vram_mb)


# ── study ────────────────────────────────────────────────────────────────────

def run_study(model, device):
    results = {}

    # ── 1. vary context size S ────────────────────────────────────────────────
    H_fixed = 128
    context_sizes = [1, 2, 4, 8, 16, 32, 64]

    print(f"\n=== Context-size scaling  (image {H_fixed}×{H_fixed}) ===")
    print(f"{'S':>5}  {'GFLOPs':>9}  {'time (ms)':>11}  {'± std':>7}  {'VRAM (MB)':>11}")

    cs_gflops, cs_time, cs_std, cs_vram = [], [], [], []
    for S in context_sizes:
        target, si, sl = make_inputs(1, S, H_fixed, device)
        gf, t, s, v = measure(model, target, si, sl)
        cs_gflops.append(gf); cs_time.append(t); cs_std.append(s); cs_vram.append(v)
        print(f"{S:>5}  {gf:>9.2f}  {t:>11.1f}  {s:>7.2f}  {v:>11.1f}")

    results["context"] = dict(S=context_sizes, gflops=cs_gflops,
                               time=cs_time, std=cs_std, vram=cs_vram, H=H_fixed)

    # ── 2. vary image size H ──────────────────────────────────────────────────
    S_fixed = 3
    image_sizes = [64, 128, 256, 512]

    print(f"\n=== Image-size scaling  (S={S_fixed}) ===")
    print(f"{'H':>5}  {'GFLOPs':>9}  {'time (ms)':>11}  {'± std':>7}  {'VRAM (MB)':>11}")

    im_gflops, im_time, im_std, im_vram = [], [], [], []
    for H in image_sizes:
        target, si, sl = make_inputs(1, S_fixed, H, device)
        gf, t, s, v = measure(model, target, si, sl)
        im_gflops.append(gf); im_time.append(t); im_std.append(s); im_vram.append(v)
        print(f"{H:>5}  {gf:>9.2f}  {t:>11.1f}  {s:>7.2f}  {v:>11.1f}")

    results["image"] = dict(H=image_sizes, gflops=im_gflops,
                             time=im_time, std=im_std, vram=im_vram, S=S_fixed)

    return results


# ── plotting ─────────────────────────────────────────────────────────────────

def plot_results(results, save_dir="results/presentation"):
    import os; os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    fig.suptitle("UniverSeg scaling study", fontsize=13, fontweight="bold")

    # ── context size ──
    ctx = results["context"]
    S   = np.array(ctx["S"])
    H   = ctx["H"]

    ax = axes[0, 0]
    ax.plot(S, ctx["gflops"], "o-", color="steelblue")
    slope = ctx["gflops"][0] / S[0]
    ax.plot(S, slope * S, "--", color="gray", label="O(S) ref", lw=1)
    ax.set_xlabel("Context size S")
    ax.set_ylabel("GFLOPs")
    ax.set_title(f"FLOPs vs context size  (H={H})")
    ax.legend(fontsize=8)
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())

    ax = axes[0, 1]
    ax.errorbar(S, ctx["time"], yerr=ctx["std"], fmt="o-", color="coral", capsize=3)
    ax.plot(S, [ctx["time"][0] / S[0] * s for s in S], "--", color="gray",
            label="O(S) ref", lw=1)
    ax.set_xlabel("Context size S")
    ax.set_ylabel("Latency (ms)")
    ax.set_title(f"Time vs context size  (H={H})")
    ax.legend(fontsize=8)
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())

    ax = axes[0, 2]
    ax.plot(S, ctx["vram"], "o-", color="mediumseagreen")
    slope_v = ctx["vram"][0] / S[0]
    ax.plot(S, slope_v * S, "--", color="gray", label="O(S) ref", lw=1)
    ax.set_xlabel("Context size S")
    ax.set_ylabel("Peak VRAM (MB)")
    ax.set_title(f"VRAM vs context size  (H={H})")
    ax.legend(fontsize=8)
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())

    # ── image size ──
    img = results["image"]
    Hs  = np.array(img["H"])
    Sf  = img["S"]

    ax = axes[1, 0]
    ax.plot(Hs, img["gflops"], "o-", color="steelblue")
    scale = img["gflops"][0] / Hs[0] ** 2
    ax.plot(Hs, scale * Hs ** 2, "--", color="gray", label="O(H²) ref", lw=1)
    ax.set_xlabel("Image size H")
    ax.set_ylabel("GFLOPs")
    ax.set_title(f"FLOPs vs image size  (S={Sf})")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.errorbar(Hs, img["time"], yerr=img["std"], fmt="o-", color="coral", capsize=3)
    scale_t = img["time"][0] / Hs[0] ** 2
    ax.plot(Hs, scale_t * Hs ** 2, "--", color="gray", label="O(H²) ref", lw=1)
    ax.set_xlabel("Image size H")
    ax.set_ylabel("Latency (ms)")
    ax.set_title(f"Time vs image size  (S={Sf})")
    ax.legend(fontsize=8)

    ax = axes[1, 2]
    ax.plot(Hs, img["vram"], "o-", color="mediumseagreen")
    scale_v = img["vram"][0] / Hs[0] ** 2
    ax.plot(Hs, scale_v * Hs ** 2, "--", color="gray", label="O(H²) ref", lw=1)
    ax.set_xlabel("Image size H")
    ax.set_ylabel("Peak VRAM (MB)")
    ax.set_title(f"VRAM vs image size  (S={Sf})")
    ax.legend(fontsize=8)

    plt.tight_layout()
    path = f"{save_dir}/universeg_scaling.pdf"
    plt.savefig(path, bbox_inches="tight")
    print(f"\nFigure saved → {path}")
    plt.show()


# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = UniverSeg(encoder_blocks=[64, 64, 64, 64]).to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params / 1e6:.1f}M")

    results = run_study(model, device)
    plot_results(results)
