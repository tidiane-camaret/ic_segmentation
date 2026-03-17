"""Simple profiling script for training batch analysis.

Usage:
    python scripts/profile_training_simple.py experiment=60_2_levels
"""
import sys
from pathlib import Path
import time
from collections import defaultdict
from contextlib import contextmanager

import hydra
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

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

    def report(self, title="TIMING REPORT"):
        print("\n" + "="*90)
        print(f"{title} (CPU wall time / CUDA kernel time)")
        print("="*90)

        total_cpu = sum(sum(v) for v in self.times.values())
        total_cuda = sum(sum(v) for v in self.cuda_times.values())

        # Sort by total time
        sorted_names = sorted(
            self.times.keys(),
            key=lambda n: sum(self.times[n]),
            reverse=True
        )

        for name in sorted_names:
            cpu_times = self.times[name]
            cuda_times = self.cuda_times[name]
            avg_cpu = sum(cpu_times) / len(cpu_times)
            avg_cuda = sum(cuda_times) / len(cuda_times)
            total_cpu_name = sum(cpu_times)
            pct = total_cpu_name / total_cpu * 100 if total_cpu > 0 else 0
            print(f"{name:50s}: {avg_cpu:8.2f}ms CPU / {avg_cuda:8.2f}ms CUDA ({pct:5.1f}%)")

        print("-"*90)
        print(f"{'TOTAL':50s}: {total_cpu:8.2f}ms CPU / {total_cuda:8.2f}ms CUDA")
        return total_cpu, total_cuda


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    """Main profiling function."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")

    seed_everything(cfg.training.seed)

    # Image augmentation config
    img_aug_cfg = cfg.get("image_augmentation", {})
    use_image_augmentation = img_aug_cfg.get("enabled", False)
    augment_config = OmegaConf.to_container(img_aug_cfg, resolve=True) if use_image_augmentation else None

    # Create dataloader with small dataset
    from src.dataloaders.totalseg2d_dataloader import get_dataloader as get_totalseg2d_dataloader

    train_labels = cfg.train_label_ids if isinstance(cfg.train_label_ids, str) else list(cfg.train_label_ids)

    max_ds_len_cfg = cfg.get("max_ds_len")
    if isinstance(max_ds_len_cfg, dict) or OmegaConf.is_dict(max_ds_len_cfg):
        max_ds_len_train = max_ds_len_cfg.get("train")
    else:
        max_ds_len_train = max_ds_len_cfg

    carve_mix_cfg = cfg.get("carve_mix", {})
    use_carve_mix = carve_mix_cfg.get("enabled", False)
    carve_mix_config = OmegaConf.to_container(carve_mix_cfg, resolve=True) if use_carve_mix else None

    adv_aug_cfg = cfg.get("advanced_augmentation", {})
    use_adv_aug = adv_aug_cfg.get("enabled", False)
    adv_aug_config = OmegaConf.to_container(adv_aug_cfg, resolve=True) if use_adv_aug else None

    # Limit dataset for profiling
    max_ds_len_train = min(max_ds_len_train or 1000, 200)

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

    print(f"\nDataset size: {len(train_loader.dataset)}, batch size: {cfg.train_batch_size}")
    print(f"Context size: {cfg.context_size}")
    print(f"Augmentation: {use_image_augmentation}, CarveMix: {use_carve_mix}, Advanced: {use_adv_aug}")

    # Create model
    patch_icl_cfg = OmegaConf.to_container(cfg.model.patch_icl, resolve=True)
    model_version = patch_icl_cfg.get("model_version", "v2")
    if model_version == "v3":
        from src.models.patch_icl_v3 import PatchICL
        print("Using PatchICL v3")
    else:
        from src.models.patch_icl_v2 import PatchICL
        print("Using PatchICL v2")
    random_coloring_nb = cfg.get("random_coloring_nb", 0)
    patch_icl_cfg["num_mask_channels"] = 3 if random_coloring_nb > 0 else 1

    # Feature extractor
    feature_mode = cfg.get("feature_mode", "precomputed")
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

    model = PatchICL(patch_icl_cfg, context_size=cfg.get("context_size", 0), feature_extractor=feature_extractor)

    # Loss functions
    loss_cfg = patch_icl_cfg.get("loss", {})
    patch_loss_cfg = loss_cfg.get("patch_loss", {"type": "dice", "args": None})
    aggreg_loss_cfg = loss_cfg.get("aggreg_loss", {"type": "dice", "args": None})
    patch_criterion = build_loss_fn(patch_loss_cfg["type"], patch_loss_cfg.get("args"))
    aggreg_criterion = build_loss_fn(aggreg_loss_cfg["type"], aggreg_loss_cfg.get("args"))
    model.set_loss_functions(patch_criterion, aggreg_criterion)

    model = model.to(device)
    model.train()

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")
    print(f"Levels: {len(patch_icl_cfg['levels'])}")
    for i, level in enumerate(patch_icl_cfg['levels']):
        print(f"  Level {i}: resolution={level['resolution']}, patch_size={level['patch_size']}, "
              f"num_patches={level['num_patches']}, method={level.get('sampling_method', 'continuous')}")

    # Get a batch for profiling
    data_iter = iter(train_loader)
    batch = next(data_iter)

    print("\n" + "="*90)
    print("BATCH INFO")
    print("="*90)
    print(f"Images shape: {batch['image'].shape}")
    print(f"Labels shape: {batch['label'].shape}")
    if batch.get('context_in') is not None:
        print(f"Context in shape: {batch['context_in'].shape}")
        print(f"Context out shape: {batch['context_out'].shape}")

    # Warmup
    print("\nWarming up (5 iterations)...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    for _ in range(5):
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
        optimizer.zero_grad()
        outputs = model(images, labels=labels, context_in=context_in, context_out=context_out, mode="train")
        loss = model.compute_loss(outputs, labels)["total_loss"]
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

    # =========================================================================
    # 1. DATALOADER PROFILING
    # =========================================================================
    print("\n\n" + "#"*90)
    print("# SECTION 1: DATALOADER PROFILING")
    print("#"*90)

    timer = Timer()
    num_batches = 20

    # Measure time to iterate
    data_times = []
    for idx in range(num_batches):
        torch.cuda.synchronize()
        start = time.perf_counter()
        batch = next(data_iter)
        data_times.append((time.perf_counter() - start) * 1000)

    avg_data_time = sum(data_times) / len(data_times)
    print(f"Average data loading time: {avg_data_time:.2f}ms per batch")
    print(f"Data loading breakdown: min={min(data_times):.2f}ms, max={max(data_times):.2f}ms")

    # =========================================================================
    # 2. DETAILED FORWARD PASS PROFILING
    # =========================================================================
    print("\n\n" + "#"*90)
    print("# SECTION 2: DETAILED FORWARD PASS (single batch)")
    print("#"*90)

    timer = Timer()
    batch = next(data_iter)

    # Prepare inputs
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
    k = context_in.shape[1] if context_in is not None else 0

    # Feature extraction breakdown
    fe = model.feature_extractor
    if fe is not None:
        with timer.track("FE: prepare_inputs"):
            if context_in is not None:
                ctx_flat = context_in.view(B * k, *context_in.shape[2:])
                mask_flat = context_out.view(B * k, *context_out.shape[2:]) if context_out is not None else None
                all_images = torch.cat([images, ctx_flat], dim=0)
            else:
                all_images = images
                mask_flat = None

        with timer.track("FE: resize_to_128"):
            if all_images.shape[-2:] != (128, 128):
                all_images = F.interpolate(all_images, size=(128, 128), mode="bilinear", align_corners=False)
            if mask_flat is not None and mask_flat.shape[-2:] != (128, 128):
                mask_flat = F.interpolate(mask_flat.float(), size=(128, 128), mode="nearest")

        # Full feature extraction
        with timer.track("FE: img_blocks_all"):
            img_x = all_images
            for i in range(4):
                img_x = fe.img_blocks[i](img_x)
                if i < 3:
                    img_x = fe.pool(img_x)

        with timer.track("FE: msk_blocks_all"):
            if mask_flat is not None:
                msk_x = mask_flat
                for i in range(4):
                    msk_x = fe.msk_blocks[i](msk_x)
                    if i < 3:
                        msk_x = fe.pool(msk_x)

        # Get features properly
        with timer.track("FE: extract_batch_total"):
            target_features, context_features = fe.extract_batch(images, context_in, context_out)
    else:
        target_features, context_features = None, None

    # Per-level forward breakdown
    for level_idx in range(model.num_levels):
        level_cfg = model.levels[level_idx]
        resolution = level_cfg['resolution']
        sampler = model.samplers[level_idx]
        aggregator = model.aggregators[level_idx]

        with timer.track(f"L{level_idx}: downsample"):
            image_ds = model._downsample(images, resolution)
            labels_ds = model._downsample_mask(labels, resolution)
            weights = model._mask_to_weights(labels_ds)

        with timer.track(f"L{level_idx}: target_sampling"):
            patches, patch_labels, coords, _, aug_params, validity = sampler(
                image_ds, labels_ds, weights, None
            )

        with timer.track(f"L{level_idx}: extract_target_patch_features"):
            if target_features is not None:
                from src.models.patch_icl_v2.patch_icl import extract_patch_features
                target_patch_features = extract_patch_features(
                    features=target_features,
                    coords=coords,
                    patch_size=level_cfg['patch_size'],
                    level_resolution=resolution,
                    feature_grid_size=model.feature_grid_size,
                    target_patch_grid_size=model.patch_feature_grid_size,
                )

        if context_in is not None:
            context_in_flat = context_in.view(B * k, *context_in.shape[2:])
            context_out_flat = context_out.view(B * k, *context_out.shape[2:])

            with timer.track(f"L{level_idx}: context_downsample"):
                context_in_ds = model._downsample(context_in_flat, resolution).view(B, k, -1, resolution, resolution)
                context_out_ds = model._downsample_mask(context_out_flat, resolution).view(B, k, context_out.shape[2], resolution, resolution)
                context_weights = model._mask_to_weights(
                    context_out_ds.view(B * k, *context_out_ds.shape[2:])
                ).view(B, k, 1, resolution, resolution)

            with timer.track(f"L{level_idx}: context_sampling"):
                ctx_patches, ctx_labels, ctx_coords, _, ctx_validity = model._select_context_patches(
                    context_in_ds, context_out_ds, context_weights, sampler
                )

            with timer.track(f"L{level_idx}: extract_context_patch_features"):
                if context_features is not None:
                    K_ctx = ctx_patches.shape[1]
                    K_per_ctx = K_ctx // k
                    ctx_features_list = []
                    for ctx_idx in range(k):
                        ctx_feats = context_features[:, ctx_idx]
                        ctx_coords_slice = ctx_coords[:, ctx_idx * K_per_ctx:(ctx_idx + 1) * K_per_ctx]
                        extracted = extract_patch_features(
                            features=ctx_feats,
                            coords=ctx_coords_slice,
                            patch_size=level_cfg['patch_size'],
                            level_resolution=resolution,
                            feature_grid_size=model.feature_grid_size,
                            target_patch_grid_size=model.patch_feature_grid_size,
                        )
                        ctx_features_list.append(extracted)
                    context_patch_features = torch.cat(ctx_features_list, dim=1)

            # Backbone
            K = target_patch_features.shape[1]
            K_ctx = context_patch_features.shape[1]
            coord_scale = H / resolution
            img_patches = torch.cat([target_patch_features, context_patch_features], dim=1)
            all_coords = torch.cat([coords.float() * coord_scale, ctx_coords.float() * coord_scale], dim=1)
            ctx_id_labels = torch.zeros(B, K + K_ctx, dtype=torch.long, device=device)
            K_per_ctx = K_ctx // k
            for ctx_idx in range(k):
                start = K + ctx_idx * K_per_ctx
                end = K + (ctx_idx + 1) * K_per_ctx
                ctx_id_labels[:, start:end] = ctx_idx + 1

            with timer.track(f"L{level_idx}: CNN_encoder"):
                encoded, skips = model.backbone.encoder(img_patches)
                encoded = encoded + model.backbone.level_embed[level_idx].view(1, 1, -1)

            with timer.track(f"L{level_idx}: attention"):
                is_context = ctx_id_labels > 0
                attended, _ = model.backbone.attention(
                    encoded, all_coords, is_context, return_attn_weights=False
                )

            with timer.track(f"L{level_idx}: CNN_decoder"):
                mask_pred = model.backbone.decoder(attended, skips)

            with timer.track(f"L{level_idx}: aggregation"):
                patch_logits = mask_pred[:, :K]
                pred = aggregator(patch_logits, coords, (resolution, resolution))

    timer.report("FORWARD PASS BREAKDOWN")

    # =========================================================================
    # 3. FULL TRAINING ITERATION PROFILING
    # =========================================================================
    print("\n\n" + "#"*90)
    print("# SECTION 3: FULL TRAINING ITERATIONS (20 batches)")
    print("#"*90)

    timer = Timer()
    data_iter = iter(train_loader)

    for idx in range(20):
        with timer.track("total_iteration"):
            with timer.track("data_loading"):
                batch = next(data_iter)

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

            with timer.track("forward_total"):
                outputs = model(
                    images, labels=labels,
                    context_in=context_in, context_out=context_out,
                    mode="train"
                )

            with timer.track("compute_loss"):
                losses = model.compute_loss(outputs, labels)
                loss = losses["total_loss"]

            with timer.track("backward"):
                loss.backward()

            with timer.track("optimizer_step"):
                optimizer.step()

    timer.report("FULL TRAINING ITERATION")

    # =========================================================================
    # SUMMARY AND RECOMMENDATIONS
    # =========================================================================
    print("\n\n" + "="*90)
    print("PROFILING SUMMARY")
    print("="*90)

    # Calculate key percentages
    times = timer.times
    total_time = sum(sum(times['total_iteration']))
    data_time = sum(times['data_loading']) + sum(times['data_to_device'])
    forward_time = sum(times['forward_total'])
    backward_time = sum(times['backward'])
    loss_time = sum(times['compute_loss'])

    print(f"""
KEY FINDINGS:
  Total iteration time: {total_time/20:.1f}ms average
  Data loading + transfer: {data_time/20:.1f}ms ({data_time/total_time*100:.1f}%)
  Forward pass: {forward_time/20:.1f}ms ({forward_time/total_time*100:.1f}%)
  Backward pass: {backward_time/20:.1f}ms ({backward_time/total_time*100:.1f}%)
  Loss computation: {loss_time/20:.1f}ms ({loss_time/total_time*100:.1f}%)

GPU MEMORY:
  Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB
  Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB
  Max allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB

RECOMMENDATIONS:
""")

    # Analysis and recommendations
    data_pct = data_time / total_time * 100

    if data_pct > 30:
        print("""  [HIGH DATA LOADING TIME]
  - Data loading is {:.1f}% of iteration time
  - Consider: increase num_workers, use pin_memory=True, preload data to RAM
  - Current dataloader uses in-memory cache, which should be fast
  - Check if augmentation is the bottleneck""".format(data_pct))

    if forward_time > backward_time * 1.5:
        print("""  [FORWARD HEAVY]
  - Forward pass is significantly longer than backward
  - This is unusual - check for unnecessary computations in forward
  - Consider gradient checkpointing to trade compute for memory""")

    if backward_time > forward_time * 2:
        print("""  [BACKWARD HEAVY]
  - Backward pass is much longer than forward ({:.1f}x)
  - This indicates memory pressure or complex gradient computation
  - Consider: mixed precision (AMP), gradient accumulation, smaller batch""".format(
      backward_time/forward_time))

    # Check for specific bottlenecks
    print("\n  [COMPONENT ANALYSIS]")
    if 'FE: img_blocks_all' in times:
        fe_time = sum(times.get('FE: img_blocks_all', [0])) + sum(times.get('FE: msk_blocks_all', [0]))
        print(f"  - Feature extraction (ICLEncoder): {fe_time:.1f}ms per iter")

    attention_time = 0
    for key in times:
        if 'attention' in key.lower():
            attention_time += sum(times[key])
    if attention_time > 0:
        print(f"  - Attention (all levels): {attention_time:.1f}ms total")

    sampling_time = 0
    for key in times:
        if 'sampling' in key.lower():
            sampling_time += sum(times[key])
    if sampling_time > 0:
        print(f"  - Sampling (all levels): {sampling_time:.1f}ms total")

    print("\n" + "="*90)
    print("PROFILING COMPLETE")
    print("="*90)


if __name__ == "__main__":
    main()
