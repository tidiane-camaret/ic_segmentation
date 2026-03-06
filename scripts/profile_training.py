"""Profile PatchICL training with Accelerate's built-in profiler.

Runs a few training batches with full profiling: CPU/CUDA time, memory,
FLOPs, and chrome trace export. Produces a detailed summary and trace file.

Usage (multi-GPU with accelerate):
    uv run accelerate launch \
        --multi_gpu --num_processes=2 --mixed_precision=fp16 \
        scripts/profile_training.py \
        +max_labels=10 experiment=70_attention cluster=dlclarge

Usage (single GPU):
    uv run python scripts/profile_training.py \
        +max_labels=10 experiment=70_attention cluster=dlclarge
"""

import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import hydra
import torch
import torch.nn.functional as F
from accelerate import Accelerator, DistributedDataParallelKwargs, ProfileKwargs
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.losses import build_loss_fn
from src.train_utils import seed_everything


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class Timer:
    """Wall-clock + CUDA-event timer for manual annotations."""

    def __init__(self):
        self.times = defaultdict(list)
        self.cuda_times = defaultdict(list)

    @contextmanager
    def track(self, name: str):
        torch.cuda.synchronize()
        start = time.perf_counter()
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        start_evt.record()
        try:
            yield
        finally:
            end_evt.record()
            torch.cuda.synchronize()
            self.times[name].append((time.perf_counter() - start) * 1000)
            self.cuda_times[name].append(start_evt.elapsed_time(end_evt))

    def report(self, title="TIMING REPORT"):
        total_cpu = sum(sum(v) for v in self.times.values())
        total_cuda = sum(sum(v) for v in self.cuda_times.values())
        sorted_names = sorted(
            self.times.keys(), key=lambda n: sum(self.times[n]), reverse=True
        )
        print(f"\n{'=' * 90}")
        print(f"  {title} (CPU wall / CUDA kernel)")
        print(f"{'=' * 90}")
        for name in sorted_names:
            avg_cpu = sum(self.times[name]) / len(self.times[name])
            avg_cuda = sum(self.cuda_times[name]) / len(self.cuda_times[name])
            pct = sum(self.times[name]) / total_cpu * 100 if total_cpu > 0 else 0
            print(f"{name:50s} {avg_cpu:8.1f}ms CPU  {avg_cuda:8.1f}ms CUDA  ({pct:5.1f}%)")
        print(f"{'-' * 90}")
        print(f"{'TOTAL':50s} {total_cpu:8.1f}ms CPU  {total_cuda:8.1f}ms CUDA")


def _sep(title, char="=", width=100):
    print(f"\n{char * width}\n  {title}\n{char * width}")


def _print_gpu_info():
    if not torch.cuda.is_available():
        return
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {p.name}  |  VRAM {p.total_memory / 1024**3:.1f} GB  |  "
              f"SMs {p.multi_processor_count}  |  CC {p.major}.{p.minor}")


def _print_vram(label=""):
    if not torch.cuda.is_available():
        return
    alloc = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    peak = torch.cuda.max_memory_allocated() / 1024**3
    print(f"  VRAM {label}: alloc={alloc:.2f} GB  reserved={reserved:.2f} GB  peak={peak:.2f} GB")


# ---------------------------------------------------------------------------
# Build data + model (mirrors train.py)
# ---------------------------------------------------------------------------

def _get_image_size(cfg) -> tuple[int, int]:
    """Get image size as tuple, handling both scalar and list formats."""
    img_size = cfg.preprocessing.image_size
    if isinstance(img_size, (list, tuple)):
        return tuple(img_size[:2])
    return (img_size, img_size)


def build_dataloader(cfg):
    """Build train dataloader from config."""
    # Unified augmentation config (takes precedence if present)
    unified_aug_cfg = cfg.get("augmentation", {})
    use_unified_augmentation = unified_aug_cfg.get("enabled", False)
    augmentation_config = (
        OmegaConf.to_container(unified_aug_cfg, resolve=True)
        if use_unified_augmentation
        else None
    )

    # Legacy image augmentation config (only used if unified is not enabled)
    img_aug_cfg = cfg.get("image_augmentation", {})
    use_image_augmentation = img_aug_cfg.get("enabled", False) and not use_unified_augmentation
    augment_config = (
        OmegaConf.to_container(img_aug_cfg, resolve=True)
        if use_image_augmentation
        else None
    )

    # TotalSeg2D dataloader - select based on dataloader_type config
    dataloader_type = cfg.get("dataloader_type", "fast")  # "fast" or "shared"
    if dataloader_type == "shared":
        from src.dataloaders.totalseg2d_shared_dataloader import (
            get_dataloader as get_totalseg2d_dataloader,
        )
    else:
        from src.dataloaders.totalseg2d_dataloader_fast import (
            get_dataloader as get_totalseg2d_dataloader,
        )

    train_labels = (
        cfg.train_label_ids
        if isinstance(cfg.train_label_ids, str)
        else list(cfg.train_label_ids)
    )
    train_split_cfg = cfg.get("train_split", "train")
    train_split = list(train_split_cfg) if OmegaConf.is_list(train_split_cfg) else train_split_cfg

    # Support separate max_ds_len for train/val, with fallback to single value
    max_ds_len_cfg = cfg.get("max_ds_len")
    if isinstance(max_ds_len_cfg, dict) or OmegaConf.is_dict(max_ds_len_cfg):
        max_ds_len_train = max_ds_len_cfg.get("train")
    else:
        max_ds_len_train = max_ds_len_cfg

    # Support separate max_cases for train/val (limits unique cases, not samples)
    max_cases_cfg = cfg.get("max_cases")
    if isinstance(max_cases_cfg, dict) or OmegaConf.is_dict(max_cases_cfg):
        max_cases_train = max_cases_cfg.get("train")
    else:
        max_cases_train = max_cases_cfg

    # CarveMix config (only for training)
    carve_mix_cfg = cfg.get("carve_mix", {})
    use_carve_mix = carve_mix_cfg.get("enabled", False)
    carve_mix_config = (
        OmegaConf.to_container(carve_mix_cfg, resolve=True) if use_carve_mix else None
    )

    # Legacy advanced augmentation config (only for training, if unified not used)
    adv_aug_cfg = cfg.get("advanced_augmentation", {})
    use_adv_aug = adv_aug_cfg.get("enabled", False) and not use_unified_augmentation
    adv_aug_config = (
        OmegaConf.to_container(adv_aug_cfg, resolve=True) if use_adv_aug else None
    )

    # Coverage filtering config (unified for fast and shared dataloaders)
    same_case_context = cfg.get("same_case_context", False)
    min_coverage = cfg.get("min_coverage", 100)
    min_coverage_ratio = cfg.get("min_coverage_ratio", 0.1)

    # Slice subsampling config (shared dataloader only)
    max_slices_per_group = cfg.get("max_slices_per_group", None)
    slice_selection = cfg.get("slice_selection", "all")

    # Resolve paths based on dataloader type
    if dataloader_type == "shared":
        # Use shared format paths: {base_dataset}_2d_shared/
        data_dir = Path(cfg.paths.DATA_DIR)
        base_dataset = cfg.get("base_dataset", "totalseg")
        shared_dir = data_dir / f"{base_dataset}_2d_shared"
        root_dir = str(shared_dir)
        stats_path = str(shared_dir / "stats.pkl")
    else:
        root_dir = cfg.paths.dataset
        stats_path = cfg.paths.dataset_stats

    # Build train dataloader kwargs based on type
    train_kwargs = dict(
        root_dir=root_dir,
        stats_path=stats_path,
        label_id_list=train_labels,
        context_size=cfg.context_size,
        batch_size=cfg.train_batch_size,
        image_size=_get_image_size(cfg),
        num_workers=cfg.training.get("num_workers", 4),
        split=train_split,
        shuffle=True,
        max_labels=cfg.get("max_labels", None),
        max_cases=max_cases_train,
    )
    if dataloader_type == "shared":
        # Shared dataloader params
        train_kwargs.update(
            crop_to_bbox=cfg.preprocessing.crop_to_bbox,
            bbox_padding=cfg.preprocessing.bbox_padding,
            min_coverage=min_coverage,
            min_coverage_ratio=min_coverage_ratio,
            same_case_context=same_case_context,
            max_ds_len=max_ds_len_train,
            class_balanced=cfg.get("class_balanced", False),
            augmentation_config=augmentation_config,
            max_slices_per_group=max_slices_per_group,
            slice_selection=slice_selection,
        )
    else:
        # Fast dataloader params
        train_kwargs.update(
            crop_to_bbox=cfg.preprocessing.crop_to_bbox,
            bbox_padding=cfg.preprocessing.bbox_padding,
            max_ds_len=max_ds_len_train,
            random_coloring_nb=cfg.get("random_coloring_nb", 0),
            augment=use_image_augmentation,
            augment_config=augment_config,
            carve_mix=use_carve_mix,
            carve_mix_config=carve_mix_config,
            advanced_augmentation=use_adv_aug,
            advanced_augmentation_config=adv_aug_config,
            augmentation_config=augmentation_config,
            class_balanced=cfg.get("class_balanced", False),
            min_coverage=min_coverage,
            min_coverage_ratio=min_coverage_ratio,
        )
    return get_totalseg2d_dataloader(**train_kwargs)


def build_model(cfg, device):
    """Build model from config (PatchICL or UniverSeg)."""
    method = cfg.get("method", "patch_icl")

    if method == "universeg":
        # UniverSeg baseline model
        from src.models.universeg_baseline import UniverSegBaseline

        universeg_cfg = cfg.model.get("universeg", {})
        model = UniverSegBaseline(
            pretrained=universeg_cfg.get("pretrained", True),
            input_size=universeg_cfg.get("input_size", 128),
            freeze=universeg_cfg.get("freeze", False),
        )
        patch_icl_cfg = {}  # Empty config for loss setup

        # Simple loss config for UniverSeg
        loss_cfg = cfg.get("loss", {})
        patch_loss_cfg = loss_cfg.get("patch_loss", {"type": "dice", "args": None})
        aggreg_loss_cfg = loss_cfg.get("aggreg_loss", {"type": "dice", "args": None})
    else:
        # Default: PatchICL v2
        from src.models.patch_icl_v2 import PatchICL

        patch_icl_cfg = OmegaConf.to_container(cfg.model.patch_icl, resolve=True)
        random_coloring_nb = cfg.get("random_coloring_nb", 0)
        patch_icl_cfg["num_mask_channels"] = 3 if random_coloring_nb > 0 else 1

        feature_mode = cfg.get("feature_mode", "precomputed")
        feature_extractor = None
        if feature_mode == "on_the_fly":
            fe_cfg = patch_icl_cfg.get("feature_extractor", None)
            extractor_type = (
                fe_cfg.get("type", "meddino").lower() if fe_cfg
                else cfg.get("feature_extractor_type", "meddino").lower()
            )

            if extractor_type in ["meddino", "meddinov3", "meddino_v3"]:
                from src.models.meddino_extractor import create_meddino_extractor
                if fe_cfg and fe_cfg.get("type") in ["meddino", "meddinov3", "meddino_v3"]:
                    feature_extractor = create_meddino_extractor(
                        model_path=fe_cfg.get("model_path", cfg.paths.ckpts.meddino_vit),
                        target_size=fe_cfg.get("target_size", 256),
                        device=device,
                        layer_idx=fe_cfg.get("layer_idx", 11),
                        freeze=fe_cfg.get("freeze", True),
                    )
                else:
                    feature_extractor = create_meddino_extractor(
                        model_path=cfg.paths.ckpts.meddino_vit,
                        target_size=cfg.get("feature_extraction_resolution", 256),
                        device=device,
                        layer_idx=cfg.get("meddino_layer_idx", 11),
                        freeze=True,
                    )
            elif extractor_type in ["medsam_v1", "medsam_v1_layer"]:
                from src.models.medsam_extractor import MedSAMv1LayerExtractor

                target_size = fe_cfg.get("target_size", 1024) if fe_cfg else 1024
                output_grid = fe_cfg.get("output_grid_size") if fe_cfg else None
                feature_extractor = MedSAMv1LayerExtractor(
                    checkpoint_path=fe_cfg.get("checkpoint_path") if fe_cfg else None,
                    target_size=target_size,
                    device=device,
                    layer_idx=fe_cfg.get("layer_idx", 11) if fe_cfg else 11,
                    freeze=fe_cfg.get("freeze", True) if fe_cfg else True,
                    output_grid_size=output_grid,
                )
            elif extractor_type == "universeg":
                from src.models.universeg_extractor import UniverSegExtractor

                feature_extractor = UniverSegExtractor(
                    layer_idx=fe_cfg.get("layer_idx", 3) if fe_cfg else 3,
                    device=device,
                    pretrained=fe_cfg.get("pretrained", True) if fe_cfg else True,
                    freeze=fe_cfg.get("freeze", True) if fe_cfg else True,
                    output_grid_size=fe_cfg.get("output_grid_size") if fe_cfg else None,
                    input_size=fe_cfg.get("input_size", 128) if fe_cfg else 128,
                    skip_preprocess=fe_cfg.get("skip_preprocess", True) if fe_cfg else True,
                )
            elif extractor_type == "icl_encoder":
                from src.models.icl_encoder import ICLEncoder
                feature_extractor = ICLEncoder(
                    layer_idx=fe_cfg.get("layer_idx", "all") if fe_cfg else "all",
                    output_grid_size=fe_cfg.get("output_grid_size") if fe_cfg else None,
                    freeze=fe_cfg.get("freeze", False) if fe_cfg else False,
                )
            elif extractor_type == "rad_dino":
                from src.models.rad_dino_extractor import RADDINOExtractor

                feature_extractor = RADDINOExtractor(
                    model_name=fe_cfg.get("model_name", "microsoft/rad-dino") if fe_cfg else "microsoft/rad-dino",
                    target_size=fe_cfg.get("target_size", 224) if fe_cfg else 224,
                    output_grid_size=fe_cfg.get("output_grid_size") if fe_cfg else None,
                    device=device,
                    freeze=fe_cfg.get("freeze", True) if fe_cfg else True,
                )
            else:
                raise ValueError(f"Unknown feature_extractor_type: {extractor_type}")

        model = PatchICL(
            patch_icl_cfg,
            context_size=cfg.get("context_size", 0),
            feature_extractor=feature_extractor,
        )

        loss_cfg = patch_icl_cfg.get("loss", {})
        patch_loss_cfg = loss_cfg.get("patch_loss", {"type": "dice", "args": None})
        aggreg_loss_cfg = loss_cfg.get("aggreg_loss", {"type": "dice", "args": None})

    model.set_loss_functions(
        build_loss_fn(patch_loss_cfg["type"], patch_loss_cfg.get("args")),
        build_loss_fn(aggreg_loss_cfg["type"], aggreg_loss_cfg.get("args")),
    )
    return model, patch_icl_cfg


# ---------------------------------------------------------------------------
# Profiling phases
# ---------------------------------------------------------------------------

def phase1_manual_timing(model, train_loader, optimizer, device, accelerator,
                         grad_accumulate_steps, n_batches=10):
    """Phase 1: Wall-clock timing of each training stage."""
    unwrapped = accelerator.unwrap_model(model)
    timer = Timer()
    model.train()

    for idx, batch in enumerate(train_loader):
        if idx >= n_batches:
            break

        with timer.track("TOTAL_ITERATION"):
            with timer.track("01_data_to_device"):
                images = batch["image"].to(device)
                labels = batch["label"].to(device)
                context_in = batch.get("context_in")
                context_out = batch.get("context_out")
                if context_in is not None:
                    context_in = context_in.to(device)
                if context_out is not None:
                    context_out = context_out.to(device)
                target_features = batch.get("target_features")
                context_features = batch.get("context_features")
                if target_features is not None:
                    target_features = target_features.to(device)
                if context_features is not None:
                    context_features = context_features.to(device)
                if labels.dim() == 3:
                    labels = labels.unsqueeze(1)

            with timer.track("02_zero_grad"):
                if idx % grad_accumulate_steps == 0:
                    optimizer.zero_grad()

            with timer.track("03_forward"):
                outputs = model(
                    images, labels=labels,
                    context_in=context_in, context_out=context_out,
                    target_features=target_features,
                    context_features=context_features,
                    mode="train",
                )

            with timer.track("04_compute_loss"):
                losses = unwrapped.compute_loss(outputs, labels)
                loss = losses["total_loss"]

            with timer.track("05_backward"):
                scaled_loss = loss / grad_accumulate_steps
                accelerator.backward(scaled_loss)

            if (idx + 1) % grad_accumulate_steps == 0:
                with timer.track("06_clip_grad"):
                    accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
                with timer.track("07_optimizer_step"):
                    optimizer.step()

        del outputs, losses

    timer.report("Phase 1 — Per-Stage Wall-Clock Timing")
    return timer


def phase2_pytorch_profiler(model, train_loader, optimizer, device,
                            accelerator, grad_accumulate_steps, trace_dir,
                            n_batches=3):
    """Phase 2: Raw PyTorch profiler on unwrapped model with reduced batch.

    Uses a smaller sub-batch to leave headroom for profiler's CUDA allocations.
    Only runs on main process to avoid DDP sync issues.
    """
    from torch.profiler import profile, record_function, ProfilerActivity

    unwrapped = accelerator.unwrap_model(model)
    is_main = accelerator.is_main_process

    if not is_main:
        # Non-main ranks just idle; no DDP sync needed since we use unwrapped
        accelerator.wait_for_everyone()
        return None

    # Free DDP memory — profile unwrapped model directly
    torch.cuda.empty_cache()
    unwrapped.train()

    # Grab a batch and slice to half size to leave room for profiler
    batch = next(iter(train_loader))
    half_bs = max(1, batch["image"].shape[0] // 4)  # Use 1/4 batch

    images = batch["image"][:half_bs].to(device)
    labels = batch["label"][:half_bs].to(device)
    context_in = batch.get("context_in")
    context_out = batch.get("context_out")
    if context_in is not None:
        context_in = context_in[:half_bs].to(device)
    if context_out is not None:
        context_out = context_out[:half_bs].to(device)
    if labels.dim() == 3:
        labels = labels.unsqueeze(1)

    print(f"  Profiling with sub-batch size {half_bs} (from {batch['image'].shape[0]})")

    # Warmup run (no profiling)
    with torch.no_grad():
        out = unwrapped(images, labels=labels, context_in=context_in,
                        context_out=context_out, mode="train")
        del out
    torch.cuda.empty_cache()

    trace_path = str(Path(trace_dir) / "trace.json")
    Path(trace_dir).mkdir(parents=True, exist_ok=True)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_flops=True,
        with_stack=False,
    ) as prof:
        for i in range(n_batches):
            optimizer.zero_grad()

            with record_function("FORWARD_PASS"):
                outputs = unwrapped(
                    images, labels=labels,
                    context_in=context_in, context_out=context_out,
                    mode="train",
                )

            with record_function("COMPUTE_LOSS"):
                losses = unwrapped.compute_loss(outputs, labels)
                loss = losses["total_loss"]

            with record_function("BACKWARD_PASS"):
                loss.backward()

            with record_function("OPTIMIZER_STEP"):
                torch.nn.utils.clip_grad_norm_(unwrapped.parameters(), max_norm=1.0)
                optimizer.step()

            del outputs, losses
            torch.cuda.empty_cache()

    prof.export_chrome_trace(trace_path)
    print(f"  Chrome trace exported to: {trace_path}")

    accelerator.wait_for_everyone()
    return prof


def print_profiler_results(prof):
    """Print comprehensive tables from PyTorch profiler."""

    _sep("CUDA TIME — Top 30 Operations")
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=30))

    _sep("CPU TIME — Top 30 Operations")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=30))

    # Memory table (only if profile_memory was enabled)
    try:
        _sep("CUDA MEMORY — Top 30 Operations (by self_cuda_memory_usage)")
        print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=30))
    except Exception:
        pass  # profile_memory was disabled

    _sep("FLOPs — Top 30 Operations")
    print(prof.key_averages().table(sort_by="flops", row_limit=30))

    _sep("CUDA TIME grouped by Input Shape — Top 30")
    print(prof.key_averages(group_by_input_shape=True).table(
        sort_by="self_cuda_time_total", row_limit=30))

    # Custom annotated regions
    _sep("Custom Annotated Regions Summary")
    region_names = {"FORWARD_PASS", "COMPUTE_LOSS", "BACKWARD_PASS", "OPTIMIZER_STEP"}
    for evt in prof.key_averages():
        if evt.key in region_names:
            # Use getattr for CUDA attrs since they may not exist on all systems
            cuda_time = getattr(evt, "cuda_time", 0) or 0
            cuda_mem = getattr(evt, "cuda_memory_usage", 0) or 0
            cpu_mem = getattr(evt, "cpu_memory_usage", 0) or 0
            print(
                f"  {evt.key:20s}  "
                f"CPU: {evt.cpu_time_total / 1e3:>10.1f} ms  "
                f"CUDA: {cuda_time / 1e3:>10.1f} ms  "
                f"Calls: {evt.count:>3d}  "
                f"CPU Mem: {cpu_mem / 1024**2:>8.1f} MB  "
                f"CUDA Mem: {cuda_mem / 1024**2:>8.1f} MB"
            )

    # Total FLOPs
    total_flops = sum(evt.flops for evt in prof.key_averages() if evt.flops)
    if total_flops > 0:
        print(f"\n  Total FLOPs (profiled batches): {total_flops:,.0f}")
        print(f"  Total GFLOPs: {total_flops / 1e9:.2f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    """Profile training with Accelerate profiler."""

    # --- Accelerator setup ---
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    mixed_precision = cfg.training.get("mixed_precision", None)

    trace_dir = Path(cfg.paths.RESULTS_DIR) / "profiling"
    profile_kwargs = ProfileKwargs(
        activities=["cpu", "cuda"],
        record_shapes=True,
        profile_memory=False,  # Disable to avoid OOM on small VRAM GPUs
        with_flops=True,
        with_stack=False,
        schedule_option={
            "skip_first": 0,
            "wait": 1,       # 1 warmup (no recording)
            "warmup": 1,     # 1 profiler warmup
            "active": 3,     # 3 active profiling batches (reduced for memory)
            "repeat": 1,
        },
        output_trace_dir=str(trace_dir),
    )

    accelerator = Accelerator(
        kwargs_handlers=[ddp_kwargs, profile_kwargs],
        mixed_precision=mixed_precision,
    )
    device = accelerator.device
    is_main = accelerator.is_main_process
    seed_everything(cfg.training.seed)

    if is_main:
        _sep("PatchICL Training Profiler")
        print(f"  Device: {device}  |  Processes: {accelerator.num_processes}  |  "
              f"Mixed precision: {mixed_precision}")
        _print_gpu_info()

    # --- Data + model ---
    train_loader = build_dataloader(cfg)
    model, patch_icl_cfg = build_model(cfg, device)

    if is_main:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        param_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2
        buf_mb = sum(b.numel() * b.element_size() for b in model.buffers()) / 1024**2
        print(f"\n  Trainable params: {trainable:,}  |  Total: {total:,}")
        print(f"  Param memory: {param_mb:.1f} MB  |  Buffer memory: {buf_mb:.1f} MB")

    # Optimizer
    opt_args = cfg.optimizer.optimizer_args
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=opt_args.lr, weight_decay=opt_args.weight_decay
    )

    # Prepare for distributed
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
    grad_accumulate_steps = cfg.training.get("grad_accumulate_steps", 1)

    if is_main and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        _print_vram("after model load + prepare")

    # --- Warmup (also prints batch shape info from first batch) ---
    if is_main:
        _sep("Warmup (2 batches)", "-")
    model.train()
    unwrapped = accelerator.unwrap_model(model)
    printed_shapes = False
    for idx, batch in enumerate(train_loader):
        if idx >= 2:
            break

        # Print batch shapes from first batch (all ranks participate in iteration)
        if not printed_shapes and is_main:
            _sep("Batch Shape Info")
            for k, v in batch.items():
                if hasattr(v, "shape"):
                    print(f"  {k:25s} {list(v.shape)}  dtype={v.dtype}")
                elif isinstance(v, (list, tuple)) and len(v) > 0:
                    print(f"  {k:25s} len={len(v)}  type={type(v[0]).__name__}")
            printed_shapes = True

        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        context_in = batch.get("context_in")
        context_out = batch.get("context_out")
        if context_in is not None:
            context_in = context_in.to(device)
        if context_out is not None:
            context_out = context_out.to(device)
        target_features = batch.get("target_features")
        context_features = batch.get("context_features")
        if target_features is not None:
            target_features = target_features.to(device)
        if context_features is not None:
            context_features = context_features.to(device)
        if labels.dim() == 3:
            labels = labels.unsqueeze(1)
        outputs = model(
            images, labels=labels,
            context_in=context_in, context_out=context_out,
            target_features=target_features,
            context_features=context_features,
            mode="train",
        )
        loss = unwrapped.compute_loss(outputs, labels)["total_loss"]
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
    torch.cuda.synchronize()

    if is_main:
        _print_vram("after warmup (peak includes forward+backward)")

    # ===================================================================
    # Phase 1: Manual wall-clock timing
    # ===================================================================
    if is_main:
        _sep("PHASE 1 — Wall-Clock Timing (10 batches)", "#")
    phase1_manual_timing(
        model, train_loader, optimizer, device, accelerator,
        grad_accumulate_steps, n_batches=10,
    )

    # ===================================================================
    # Phase 2: Accelerate profiler (CPU/CUDA time, memory, FLOPs)
    # ===================================================================
    if is_main:
        _sep("PHASE 2 — PyTorch Profiler (3 iterations, reduced batch, main rank only)", "#")
        torch.cuda.reset_peak_memory_stats()

    prof = phase2_pytorch_profiler(
        model, train_loader, optimizer, device, accelerator,
        grad_accumulate_steps, trace_dir=str(trace_dir), n_batches=3,
    )

    if is_main and prof is not None:
        torch.cuda.synchronize()
        print_profiler_results(prof)

        # Peak VRAM summary
        _sep("Peak VRAM Summary")
        _print_vram("final")
        peak = torch.cuda.max_memory_allocated() / 1024**3
        print(f"  Peak allocated (during profiled run): {peak:.2f} GB")

        # Trace location
        if prof is not None:
            _sep("Chrome Trace Export")
            print(f"  Traces saved to: {trace_dir}")
            print("  Open at: chrome://tracing  or  https://ui.perfetto.dev")

        _sep("Profiling Complete")
        print("""
OPTIMIZATION CHECKLIST:
  1. Data loading slow?    → Increase num_workers, use pinned memory, or prefetch
  2. Feature extraction?   → Freeze encoder, reduce resolution, precompute features
  3. Attention bottleneck? → Reduce num_patches/num_layers, use Flash Attention
  4. Backward >> Forward?  → Gradient checkpointing, reduce batch size
  5. Memory pressure?      → Mixed precision (fp16/bf16), gradient accumulation
  6. Optimizer step slow?  → Fused AdamW (torch.optim.AdamW(fused=True))
""")


if __name__ == "__main__":
    main()
