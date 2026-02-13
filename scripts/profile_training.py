"""Profile training batch to identify bottlenecks.

Usage:
    python scripts/profile_training.py +experiment=60_2_levels
"""

import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import hydra
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.profiler import ProfilerActivity, profile, record_function

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.losses import build_loss_fn
from src.train_utils import seed_everything


class Timer:
    """Simple timer for measuring execution time."""

    def __init__(self):
        self.times = defaultdict(list)
        self.cuda_times = defaultdict(list)

    @contextmanager
    def track(self, name: str):
        torch.cuda.synchronize()
        start = time.perf_counter()
        start_cuda = torch.cuda.Event(enable_timing=True)
        end_cuda = torch.cuda.Event(enable_timing=True)
        start_cuda.record()
        try:
            yield
        finally:
            end_cuda.record()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            cuda_elapsed = start_cuda.elapsed_time(end_cuda)
            self.times[name].append(elapsed * 1000)  # ms
            self.cuda_times[name].append(cuda_elapsed)

    def report(self):
        print("\n" + "=" * 80)
        print("TIMING REPORT (CPU wall time / CUDA kernel time)")
        print("=" * 80)

        total_cpu = sum(sum(v) for v in self.times.values())
        total_cuda = sum(sum(v) for v in self.cuda_times.values())

        # Sort by total time
        sorted_names = sorted(
            self.times.keys(), key=lambda n: sum(self.times[n]), reverse=True
        )

        for name in sorted_names:
            cpu_times = self.times[name]
            cuda_times = self.cuda_times[name]
            avg_cpu = sum(cpu_times) / len(cpu_times)
            avg_cuda = sum(cuda_times) / len(cuda_times)
            total_cpu_name = sum(cpu_times)
            pct = total_cpu_name / total_cpu * 100 if total_cpu > 0 else 0
            print(
                f"{name:40s}: {avg_cpu:8.2f}ms CPU / {avg_cuda:8.2f}ms CUDA ({pct:5.1f}%)"
            )

        print("-" * 80)
        print(f"{'TOTAL':40s}: {total_cpu:8.2f}ms CPU / {total_cuda:8.2f}ms CUDA")


def profile_dataloader(train_loader, device, num_batches=5):
    """Profile data loading time."""
    print("\n" + "=" * 80)
    print("DATALOADER PROFILING")
    print("=" * 80)

    timer = Timer()

    for idx, batch in enumerate(train_loader):
        if idx >= num_batches:
            break

        with timer.track("total_batch_load"):
            with timer.track("get_batch_from_iterator"):
                pass  # Already got batch

            with timer.track("to_device_images"):
                images = batch["image"].to(device)

            with timer.track("to_device_labels"):
                labels = batch["label"].to(device)

            with timer.track("to_device_context"):
                context_in = batch.get("context_in")
                context_out = batch.get("context_out")
                if context_in is not None:
                    context_in = context_in.to(device)
                if context_out is not None:
                    context_out = context_out.to(device)

    timer.report()
    return timer


def profile_model_forward(model, batch, device, timer, unwrapped_model):
    """Profile model forward pass in detail."""
    images = batch["image"].to(device)
    labels = batch["label"].to(device)
    context_in = batch.get("context_in")
    context_out = batch.get("context_out")
    if context_in is not None:
        context_in = context_in.to(device)
    if context_out is not None:
        context_out = context_out.to(device)

    if labels.dim() == 3:
        labels = labels.unsqueeze(1)

    # Manual profiling of forward pass components
    B, _, H, W = images.shape

    # Feature extraction
    with timer.track("feature_extraction"):
        if unwrapped_model.feature_extractor is not None:
            target_features, context_features = unwrapped_model._extract_features(
                images, context_in, context_out
            )
        else:
            target_features, context_features = None, None

    # Full forward with tracking
    with timer.track("full_forward"):
        outputs = model(
            images,
            labels=labels,
            context_in=context_in,
            context_out=context_out,
            target_features=target_features,
            context_features=context_features,
            mode="train",
        )

    with timer.track("compute_loss"):
        losses = unwrapped_model.compute_loss(outputs, labels)
        loss = losses["total_loss"]

    return outputs, loss


def profile_forward_detailed(model, batch, device, unwrapped_model):
    """Detailed profiling of forward pass internals."""
    print("\n" + "=" * 80)
    print("DETAILED FORWARD PASS PROFILING")
    print("=" * 80)

    timer = Timer()

    images = batch["image"].to(device)
    labels = batch["label"].to(device)
    context_in = batch.get("context_in")
    context_out = batch.get("context_out")
    if context_in is not None:
        context_in = context_in.to(device)
    if context_out is not None:
        context_out = context_out.to(device)

    if labels.dim() == 3:
        labels = labels.unsqueeze(1)

    B, _, H, W = images.shape

    # 1. Feature extraction breakdown
    with timer.track("1.feature_extraction_total"):
        if unwrapped_model.feature_extractor is not None:
            fe = unwrapped_model.feature_extractor

            # Prepare inputs
            with timer.track("1a.prepare_inputs"):
                k = context_in.shape[1] if context_in is not None else 0
                if context_in is not None:
                    ctx_flat = context_in.view(B * k, *context_in.shape[2:])
                    mask_flat = (
                        context_out.view(B * k, *context_out.shape[2:])
                        if context_out is not None
                        else None
                    )
                    all_images = torch.cat([images, ctx_flat], dim=0)
                else:
                    all_images = images
                    mask_flat = None

            with timer.track("1b.resize_inputs"):
                if all_images.shape[-2:] != (128, 128):
                    all_images = F.interpolate(
                        all_images,
                        size=(128, 128),
                        mode="bilinear",
                        align_corners=False,
                    )
                if mask_flat is not None and mask_flat.shape[-2:] != (128, 128):
                    mask_flat = F.interpolate(
                        mask_flat.float(), size=(128, 128), mode="nearest"
                    )

            with timer.track("1c.encoder_forward"):
                # Run through conv blocks
                img_x = all_images
                msk_x = mask_flat
                for i in range(4):
                    with timer.track(f"1c_{i}_img_block"):
                        img_x = fe.img_blocks[i](img_x)
                    if msk_x is not None:
                        with timer.track(f"1c_{i}_msk_block"):
                            msk_x = fe.msk_blocks[i](msk_x)
                    if i < 3:
                        img_x = fe.pool(img_x)
                        if msk_x is not None:
                            msk_x = fe.pool(msk_x)

            # Get proper features
            target_features, context_features = fe.extract_batch(
                images, context_in, context_out
            )
        else:
            target_features, context_features = None, None

    # 2. Per-level forward
    for level_idx in range(unwrapped_model.num_levels):
        level_cfg = unwrapped_model.levels[level_idx]
        resolution = level_cfg["resolution"]
        sampler = unwrapped_model.samplers[level_idx]
        aggregator = unwrapped_model.aggregators[level_idx]

        with timer.track(f"2.level_{level_idx}_total"):
            # Downsample
            with timer.track(f"2a.level_{level_idx}_downsample"):
                image_ds = unwrapped_model._downsample(images, resolution)
                labels_ds = unwrapped_model._downsample_mask(labels, resolution)
                weights = unwrapped_model._mask_to_weights(labels_ds)

            # Target sampling
            with timer.track(f"2b.level_{level_idx}_target_sampling"):
                patches, patch_labels, coords, _, aug_params, validity = sampler(
                    image_ds, labels_ds, weights, None
                )

            # Extract target patch features
            with timer.track(f"2c.level_{level_idx}_extract_target_features"):
                if target_features is not None:
                    from src.models.patch_icl_v2.patch_icl import extract_patch_features

                    target_patch_features = extract_patch_features(
                        features=target_features,
                        coords=coords,
                        patch_size=level_cfg["patch_size"],
                        level_resolution=resolution,
                        feature_grid_size=unwrapped_model.feature_grid_size,
                        target_patch_grid_size=unwrapped_model.patch_feature_grid_size,
                    )

            # Context sampling
            if context_in is not None:
                k = context_in.shape[1]
                context_in_flat = context_in.view(B * k, *context_in.shape[2:])
                context_out_flat = context_out.view(B * k, *context_out.shape[2:])

                with timer.track(f"2d.level_{level_idx}_context_downsample"):
                    context_in_ds = unwrapped_model._downsample(
                        context_in_flat, resolution
                    ).view(B, k, -1, resolution, resolution)
                    context_out_ds = unwrapped_model._downsample_mask(
                        context_out_flat, resolution
                    ).view(B, k, context_out.shape[2], resolution, resolution)
                    context_weights = unwrapped_model._mask_to_weights(
                        context_out_ds.view(B * k, *context_out_ds.shape[2:])
                    ).view(B, k, 1, resolution, resolution)

                with timer.track(f"2e.level_{level_idx}_context_sampling"):
                    ctx_patches, ctx_labels, ctx_coords, _, ctx_validity = (
                        unwrapped_model._select_context_patches(
                            context_in_ds, context_out_ds, context_weights, sampler
                        )
                    )

                with timer.track(f"2f.level_{level_idx}_extract_context_features"):
                    if context_features is not None:
                        K_ctx = ctx_patches.shape[1]
                        K_per_ctx = K_ctx // k
                        ctx_features_list = []
                        for ctx_idx in range(k):
                            ctx_feats = context_features[:, ctx_idx]
                            ctx_coords_slice = ctx_coords[
                                :, ctx_idx * K_per_ctx : (ctx_idx + 1) * K_per_ctx
                            ]
                            extracted = extract_patch_features(
                                features=ctx_feats,
                                coords=ctx_coords_slice,
                                patch_size=level_cfg["patch_size"],
                                level_resolution=resolution,
                                feature_grid_size=unwrapped_model.feature_grid_size,
                                target_patch_grid_size=unwrapped_model.patch_feature_grid_size,
                            )
                            ctx_features_list.append(extracted)
                        context_patch_features = torch.cat(ctx_features_list, dim=1)

                # Backbone (encoder + attention + decoder)
                with timer.track(f"2g.level_{level_idx}_backbone_total"):
                    K = target_patch_features.shape[1]
                    K_ctx = context_patch_features.shape[1]
                    coord_scale = H / resolution

                    img_patches = torch.cat(
                        [target_patch_features, context_patch_features], dim=1
                    )
                    all_coords = torch.cat(
                        [
                            coords.float() * coord_scale,
                            ctx_coords.float() * coord_scale,
                        ],
                        dim=1,
                    )
                    ctx_id_labels = torch.zeros(
                        B, K + K_ctx, dtype=torch.long, device=device
                    )
                    K_per_ctx = K_ctx // k
                    for ctx_idx in range(k):
                        start = K + ctx_idx * K_per_ctx
                        end = K + (ctx_idx + 1) * K_per_ctx
                        ctx_id_labels[:, start:end] = ctx_idx + 1

                    with timer.track(f"2g1.level_{level_idx}_cnn_encoder"):
                        encoded, skips = unwrapped_model.backbone.encoder(img_patches)
                        encoded = encoded + unwrapped_model.backbone.level_embed[
                            level_idx
                        ].view(1, 1, -1)

                    with timer.track(f"2g2.level_{level_idx}_attention"):
                        is_context = ctx_id_labels > 0
                        attended, _ = unwrapped_model.backbone.attention(
                            encoded, all_coords, is_context, return_attn_weights=False
                        )

                    with timer.track(f"2g3.level_{level_idx}_cnn_decoder"):
                        mask_pred = unwrapped_model.backbone.decoder(attended, skips)

                # Aggregation
                with timer.track(f"2h.level_{level_idx}_aggregation"):
                    patch_logits = mask_pred[:, :K]
                    pred = aggregator(patch_logits, coords, (resolution, resolution))

    timer.report()
    return timer


def profile_backward(model, batch, device, unwrapped_model):
    """Profile backward pass."""
    print("\n" + "=" * 80)
    print("BACKWARD PASS PROFILING")
    print("=" * 80)

    timer = Timer()

    images = batch["image"].to(device)
    labels = batch["label"].to(device)
    context_in = batch.get("context_in")
    context_out = batch.get("context_out")
    if context_in is not None:
        context_in = context_in.to(device)
    if context_out is not None:
        context_out = context_out.to(device)

    if labels.dim() == 3:
        labels = labels.unsqueeze(1)

    # Forward
    with timer.track("forward"):
        outputs = model(
            images,
            labels=labels,
            context_in=context_in,
            context_out=context_out,
            mode="train",
        )
        losses = unwrapped_model.compute_loss(outputs, labels)
        loss = losses["total_loss"]

    # Backward
    with timer.track("backward"):
        loss.backward()

    timer.report()
    return timer


def run_pytorch_profiler(model, train_loader, device, num_batches=3):
    """Run PyTorch profiler for detailed CUDA analysis."""
    print("\n" + "=" * 80)
    print("PYTORCH PROFILER (CUDA kernels)")
    print("=" * 80)

    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]

    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        for idx, batch in enumerate(train_loader):
            if idx >= num_batches:
                break

            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            context_in = batch.get("context_in")
            context_out = batch.get("context_out")
            if context_in is not None:
                context_in = context_in.to(device)
            if context_out is not None:
                context_out = context_out.to(device)

            if labels.dim() == 3:
                labels = labels.unsqueeze(1)

            with record_function("forward"):
                outputs = model(
                    images,
                    labels=labels,
                    context_in=context_in,
                    context_out=context_out,
                    mode="train",
                )

            with record_function("compute_loss"):
                from accelerate import Accelerator

                accelerator = Accelerator()
                unwrapped = accelerator.unwrap_model(model)
                losses = unwrapped.compute_loss(outputs, labels)
                loss = losses["total_loss"]

            with record_function("backward"):
                loss.backward()

            model.zero_grad()

    # Print summary
    print("\n--- Top 30 CUDA operations by total time ---")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))

    print("\n--- Top 30 operations by CPU time ---")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=30))

    # Export chrome trace
    trace_path = "/tmp/profile_trace.json"
    prof.export_chrome_trace(trace_path)
    print(f"\nChrome trace exported to {trace_path}")
    print("Open chrome://tracing and load this file for visualization")

    return prof


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    """Main profiling function."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    seed_everything(cfg.training.seed)

    # Image augmentation config
    img_aug_cfg = cfg.get("image_augmentation", {})
    use_image_augmentation = img_aug_cfg.get("enabled", False)
    augment_config = (
        OmegaConf.to_container(img_aug_cfg, resolve=True)
        if use_image_augmentation
        else None
    )

    # Create dataloader
    dataset_type = cfg.get("dataset", "totalseg2d")
    feature_mode = cfg.get("feature_mode", "precomputed")

    if dataset_type == "totalseg2d":
        from src.dataloaders.totalseg2d_dataloader import (
            get_dataloader as get_totalseg2d_dataloader,
        )

        train_labels = (
            cfg.train_label_ids
            if isinstance(cfg.train_label_ids, str)
            else list(cfg.train_label_ids)
        )

        max_ds_len_cfg = cfg.get("max_ds_len")
        if isinstance(max_ds_len_cfg, dict) or OmegaConf.is_dict(max_ds_len_cfg):
            max_ds_len_train = max_ds_len_cfg.get("train")
        else:
            max_ds_len_train = max_ds_len_cfg

        carve_mix_cfg = cfg.get("carve_mix", {})
        use_carve_mix = carve_mix_cfg.get("enabled", False)
        carve_mix_config = (
            OmegaConf.to_container(carve_mix_cfg, resolve=True)
            if use_carve_mix
            else None
        )

        adv_aug_cfg = cfg.get("advanced_augmentation", {})
        use_adv_aug = adv_aug_cfg.get("enabled", False)
        adv_aug_config = (
            OmegaConf.to_container(adv_aug_cfg, resolve=True) if use_adv_aug else None
        )

        # Limit dataset for profiling
        max_ds_len_train = min(max_ds_len_train or 1000, 100)

        train_loader = get_totalseg2d_dataloader(
            root_dir=cfg.paths.totalseg2d,
            stats_path=cfg.paths.totalseg_stats,
            label_id_list=train_labels,
            context_size=cfg.context_size,
            batch_size=cfg.train_batch_size,
            image_size=tuple(cfg.preprocessing.image_size[:2]),
            crop_to_bbox=cfg.preprocessing.crop_to_bbox,
            bbox_padding=cfg.preprocessing.bbox_padding,
            num_workers=cfg.training.get("num_workers", 4),
            split="train",
            shuffle=True,
            max_ds_len=max_ds_len_train,
            random_coloring_nb=cfg.get("random_coloring_nb", 0),
            augment=use_image_augmentation,
            augment_config=augment_config,
            carve_mix=use_carve_mix,
            carve_mix_config=carve_mix_config,
            advanced_augmentation=use_adv_aug,
            advanced_augmentation_config=adv_aug_config,
            max_labels=cfg.get("max_labels", None),
        )
    else:
        raise ValueError(
            f"Profiling only supports totalseg2d dataset, got {dataset_type}"
        )

    print(
        f"Dataset size: {len(train_loader.dataset)}, batch size: {cfg.train_batch_size}"
    )

    # Create model
    from src.models.patch_icl_v2 import PatchICL

    patch_icl_cfg = OmegaConf.to_container(cfg.model.patch_icl, resolve=True)
    random_coloring_nb = cfg.get("random_coloring_nb", 0)
    patch_icl_cfg["num_mask_channels"] = 3 if random_coloring_nb > 0 else 1

    # Feature extractor
    feature_extractor = None
    if feature_mode == "on_the_fly":
        fe_cfg = patch_icl_cfg.get("feature_extractor", None)
        extractor_type = fe_cfg.get("type", "meddino").lower() if fe_cfg else "meddino"

        if extractor_type == "icl_encoder":
            from src.models.icl_encoder import ICLEncoder

            feature_extractor = ICLEncoder(
                layer_idx=fe_cfg.get("layer_idx", "all") if fe_cfg else "all",
                output_grid_size=fe_cfg.get("output_grid_size") if fe_cfg else None,
                freeze=fe_cfg.get("freeze", False) if fe_cfg else False,
            )

    model = PatchICL(
        patch_icl_cfg,
        context_size=cfg.get("context_size", 0),
        feature_extractor=feature_extractor,
    )

    # Loss functions
    loss_cfg = patch_icl_cfg.get("loss", {})
    patch_loss_cfg = loss_cfg.get("patch_loss", {"type": "dice", "args": None})
    aggreg_loss_cfg = loss_cfg.get("aggreg_loss", {"type": "dice", "args": None})
    patch_criterion = build_loss_fn(patch_loss_cfg["type"], patch_loss_cfg.get("args"))
    aggreg_criterion = build_loss_fn(
        aggreg_loss_cfg["type"], aggreg_loss_cfg.get("args")
    )
    model.set_loss_functions(patch_criterion, aggreg_criterion)

    model = model.to(device)
    model.train()

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")

    # Get a batch for profiling
    batch = next(iter(train_loader))

    print("\n" + "=" * 80)
    print("BATCH INFO")
    print("=" * 80)
    print(f"Images shape: {batch['image'].shape}")
    print(f"Labels shape: {batch['label'].shape}")
    if batch.get("context_in") is not None:
        print(f"Context in shape: {batch['context_in'].shape}")
        print(f"Context out shape: {batch['context_out'].shape}")

    # Warmup
    print("\nWarming up...")
    for _ in range(3):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        context_in = batch.get("context_in")
        context_out = batch.get("context_out")
        if context_in is not None:
            context_in = context_in.to(device)
        if context_out is not None:
            context_out = context_out.to(device)
        if labels.dim() == 3:
            labels = labels.unsqueeze(1)
        outputs = model(
            images,
            labels=labels,
            context_in=context_in,
            context_out=context_out,
            mode="train",
        )
        loss = model.compute_loss(outputs, labels)["total_loss"]
        loss.backward()
        model.zero_grad()
    torch.cuda.synchronize()

    # 1. Profile data loading
    print("\n\n" + "#" * 80)
    print("# PROFILING DATA LOADING")
    print("#" * 80)
    profile_dataloader(train_loader, device, num_batches=10)

    # 2. Detailed forward profiling
    print("\n\n" + "#" * 80)
    print("# PROFILING FORWARD PASS (DETAILED)")
    print("#" * 80)
    profile_forward_detailed(model, batch, device, model)

    # 3. Profile backward
    print("\n\n" + "#" * 80)
    print("# PROFILING BACKWARD PASS")
    print("#" * 80)
    profile_backward(model, batch, device, model)

    # 4. Full training iteration timing
    print("\n\n" + "#" * 80)
    print("# FULL TRAINING ITERATION TIMING")
    print("#" * 80)
    timer = Timer()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for idx, batch in enumerate(train_loader):
        if idx >= 10:
            break

        with timer.track("total_iteration"):
            with timer.track("data_to_device"):
                images = batch["image"].to(device)
                labels = batch["label"].to(device)
                context_in = batch.get("context_in")
                context_out = batch.get("context_out")
                if context_in is not None:
                    context_in = context_in.to(device)
                if context_out is not None:
                    context_out = context_out.to(device)
                if labels.dim() == 3:
                    labels = labels.unsqueeze(1)

            with timer.track("zero_grad"):
                optimizer.zero_grad()

            with timer.track("forward"):
                outputs = model(
                    images,
                    labels=labels,
                    context_in=context_in,
                    context_out=context_out,
                    mode="train",
                )

            with timer.track("loss"):
                losses = model.compute_loss(outputs, labels)
                loss = losses["total_loss"]

            with timer.track("backward"):
                loss.backward()

            with timer.track("optimizer_step"):
                optimizer.step()

    timer.report()

    # 5. PyTorch profiler for CUDA kernel analysis
    print("\n\n" + "#" * 80)
    print("# PYTORCH CUDA PROFILER")
    print("#" * 80)
    model.zero_grad()
    run_pytorch_profiler(model, train_loader, device, num_batches=3)

    print("\n\n" + "=" * 80)
    print("PROFILING COMPLETE")
    print("=" * 80)
    print(
        """
ANALYSIS TIPS:
1. If data_to_device time is high → CPU-GPU transfer bottleneck, consider pinned memory
2. If dataloader time is high → Data augmentation or disk I/O bottleneck
3. If feature_extraction time is high → Consider freezing encoder or using precomputed features
4. If attention time is high → Reduce num_patches or use more efficient attention
5. If backward time >> forward time → Memory pressure, consider gradient checkpointing
6. Check chrome trace for detailed kernel-level analysis
"""
    )


if __name__ == "__main__":
    from contextlib import redirect_stdout

    output_path = "profiler_output.txt"
    with open(output_path, "w") as f, redirect_stdout(f):
        main()
