"""Benchmark forward and backward pass timing."""
import sys
from pathlib import Path
import torch
import time
import hydra
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@hydra.main(config_path="../configs", config_name="train", version_base=None)
def main(cfg: DictConfig):
    device = torch.device('cuda')

    # Feature extractor for on-the-fly mode
    feature_mode = cfg.get("feature_mode", "precomputed")
    patch_icl_cfg = OmegaConf.to_container(cfg.model.patch_icl, resolve=True)
    random_coloring_nb = cfg.get("random_coloring_nb", 0)
    patch_icl_cfg["num_mask_channels"] = 3 if random_coloring_nb > 0 else 1

    feature_extractor = None
    if feature_mode == "on_the_fly":
        fe_cfg = patch_icl_cfg.get("feature_extractor", None)
        extractor_type = fe_cfg.get("type", "meddino").lower() if fe_cfg else "meddino"

        if extractor_type == "icl_encoder":
            from src.models.icl_encoder import ICLEncoder
            print("Initializing ICLEncoder for on-the-fly feature extraction...")
            feature_extractor = ICLEncoder(
                layer_idx=fe_cfg.get("layer_idx", "all") if fe_cfg else "all",
                output_grid_size=fe_cfg.get("output_grid_size") if fe_cfg else None,
                freeze=fe_cfg.get("freeze", False) if fe_cfg else False,
            )
            info = feature_extractor.get_feature_info()
            print(f"ICLEncoder: layers={info['layer_indices']}, dim={info['feature_dim']}, grid={info['output_grid_size']}")
    else:
        print("Feature mode: precomputed")

    # Create model
    from src.models.patch_icl_v2 import PatchICL
    model = PatchICL(patch_icl_cfg, context_size=cfg.get("context_size", 0), feature_extractor=feature_extractor)
    model = model.to(device)
    if feature_extractor:
        feature_extractor = feature_extractor.to(device)

    # Create dataloader (totalseg2d)
    from src.dataloaders.totalseg2d_dataloader import get_dataloader

    # Handle train_label_ids correctly - can be "train"/"val" string or list of labels
    train_label_ids = cfg.get("train_label_ids", "train")
    train_labels = train_label_ids if isinstance(train_label_ids, str) else list(train_label_ids)

    img_aug_cfg = cfg.get("image_augmentation", {})
    use_image_augmentation = img_aug_cfg.get("enabled", False)
    augment_config = OmegaConf.to_container(img_aug_cfg, resolve=True) if use_image_augmentation else None

    train_loader = get_dataloader(
        root_dir=cfg.paths.totalseg2d,
        stats_path=cfg.paths.totalseg_stats,
        label_id_list=train_labels,
        context_size=cfg.context_size,
        batch_size=4,  # Use batch_size 4 for benchmark
        image_size=tuple(cfg.preprocessing.image_size[:2]),
        crop_to_bbox=cfg.preprocessing.crop_to_bbox,
        bbox_padding=cfg.preprocessing.bbox_padding,
        num_workers=4,
        split="train",
        shuffle=True,
        max_ds_len=100,  # Small dataset for benchmarking
        augment=use_image_augmentation,
        augment_config=augment_config,
        max_labels=cfg.get("max_labels", None),
    )

    # Get a batch
    batch = next(iter(train_loader))

    # Move to device - keys are: image, label, context_in, context_out
    target_images = batch['image'].to(device)
    target_masks = batch['label'].to(device)
    context_images = batch['context_in'].to(device)
    context_masks = batch['context_out'].to(device)

    # No precomputed features for on_the_fly mode
    target_features = None
    context_features = None

    all_params = list(model.parameters())
    if feature_extractor:
        all_params += list(feature_extractor.parameters())
    optimizer = torch.optim.AdamW(all_params, lr=1e-4)

    print(f'Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M')
    print(f'Batch size: {target_images.shape[0]}')
    print()

    # Warmup
    print('Warmup...')
    for _ in range(5):
        optimizer.zero_grad()
        outputs = model(
            image=target_images,
            labels=target_masks,
            context_in=context_images,
            context_out=context_masks,
        )
        loss = outputs['final_pred'].mean()
        loss.backward()
        optimizer.step()

    # Benchmark
    print('Benchmarking (20 iterations)...')
    torch.cuda.synchronize()
    n_iters = 20
    forward_times = []
    backward_times = []
    total_times = []

    for _ in range(n_iters):
        optimizer.zero_grad()

        torch.cuda.synchronize()
        t0 = time.time()

        outputs = model(
            image=target_images,
            labels=target_masks,
            context_in=context_images,
            context_out=context_masks,
        )
        loss = outputs['final_pred'].mean()

        torch.cuda.synchronize()
        t1 = time.time()

        loss.backward()

        torch.cuda.synchronize()
        t2 = time.time()

        optimizer.step()

        torch.cuda.synchronize()
        t3 = time.time()

        forward_times.append((t1 - t0) * 1000)
        backward_times.append((t2 - t1) * 1000)
        total_times.append((t3 - t0) * 1000)

    avg_forward = sum(forward_times) / len(forward_times)
    avg_backward = sum(backward_times) / len(backward_times)
    avg_total = sum(total_times) / len(total_times)

    print()
    print('=' * 60)
    print('RESULTS WITH GRID_SAMPLE OPTIMIZATION')
    print('=' * 60)
    print(f'Forward:  {avg_forward:.1f}ms')
    print(f'Backward: {avg_backward:.1f}ms')
    print(f'Total:    {avg_total:.1f}ms')
    print(f'Backward/Forward ratio: {avg_backward / avg_forward:.2f}x')
    print()
    print('Previous benchmarks (from PROFILING_RESULTS.md):')
    print('  Forward:  63ms (with Gumbel-TopK sampling)')
    print('  Backward: 262ms')
    print('  Ratio: 4.2x')


if __name__ == "__main__":
    main()
