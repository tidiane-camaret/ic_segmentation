"""Shared model building logic for eval and train scripts."""

import torch
from omegaconf import DictConfig, OmegaConf


def build_model(
    cfg: DictConfig,
    device: torch.device,
    context_size: int = None,
    verbose: bool = True,
):
    """Build a model from hydra config, dispatching on cfg.method.

    Supports: "patch_icl", "universeg"
    """
    method = cfg.get("method", "patch_icl")

    if method == "patch_icl":
        return build_patch_icl_model(cfg, device, context_size=context_size, verbose=verbose)
    elif method == "universeg":
        return build_universeg_model(cfg, verbose=verbose)
    else:
        raise ValueError(f"Unknown method: {method}")


def build_universeg_model(cfg: DictConfig, verbose: bool = True):
    """Build a UniverSeg baseline model from config."""
    from src.models.universeg_baseline import UniverSegBaseline
    from src.losses import build_loss_fn

    universeg_cfg = cfg.model.get("universeg", {}) if cfg.get("model") else {}
    model = UniverSegBaseline(
        pretrained=universeg_cfg.get("pretrained", True),
        input_size=universeg_cfg.get("input_size", 128),
        freeze=universeg_cfg.get("freeze", False),  # Match train.py
    )

    # Read loss config (match train.py behavior)
    loss_cfg = cfg.get("loss", {})
    patch_loss_cfg = loss_cfg.get("patch_loss", {"type": "dice", "args": None})
    aggreg_loss_cfg = loss_cfg.get("aggreg_loss", {"type": "dice", "args": None})
    patch_criterion = build_loss_fn(patch_loss_cfg.get("type", "dice"), patch_loss_cfg.get("args"))
    aggreg_criterion = build_loss_fn(aggreg_loss_cfg.get("type", "dice"), aggreg_loss_cfg.get("args"))
    model.set_loss_functions(patch_criterion, aggreg_criterion)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if verbose:
        print(f"Using UniverSeg baseline model (input_size={model.input_size}, "
              f"pretrained={universeg_cfg.get('pretrained', True)}, "
              f"freeze={universeg_cfg.get('freeze', False)})")
        print(f"Model parameters: {num_params:,}")

    return model


def build_patch_icl_model(
    cfg: DictConfig,
    device: torch.device,
    context_size: int = None,
    verbose: bool = True,
):
    """Build a PatchICL model with feature extractor from hydra config.

    Args:
        cfg: Hydra config (must include model.patch_icl and feature_mode)
        device: Target device for feature extractor
        context_size: Override context_size (default: cfg.context_size)
        verbose: Print build info

    Returns:
        model: PatchICL model (not yet on device)
    """
    from src.losses import build_loss_fn

    # Select model version (v2 default for backward compat)
    model_version = cfg.model.patch_icl.get("model_version", "v2")
    if model_version == "v3":
        from src.models.patch_icl_v3 import PatchICL
        if verbose:
            print("Using PatchICL v3")
    else:
        from src.models.patch_icl_v2 import PatchICL
        if verbose:
            print("Using PatchICL v2")

    patch_icl_cfg = OmegaConf.to_container(cfg.model.patch_icl, resolve=True)
    random_coloring_nb = cfg.get("random_coloring_nb", 0)
    patch_icl_cfg["num_mask_channels"] = 3 if random_coloring_nb > 0 else 1
    if verbose:
        print(f"Mask channels: {patch_icl_cfg['num_mask_channels']}")

    # Build feature extractor
    feature_extractor = None
    feature_mode = cfg.get("feature_mode", "precomputed")
    if feature_mode == "on_the_fly":
        feature_extractor = _build_feature_extractor(cfg, patch_icl_cfg, device, verbose)
    elif verbose:
        print("Feature mode: precomputed")

    # Build model
    ctx_size = context_size if context_size is not None else cfg.get("context_size", 0)
    model = PatchICL(patch_icl_cfg, context_size=ctx_size, feature_extractor=feature_extractor)

    # Set loss functions
    loss_cfg = patch_icl_cfg.get("loss", {})
    patch_loss_cfg = loss_cfg.get("patch_loss", {"type": "dice", "args": None})
    aggreg_loss_cfg = loss_cfg.get("aggreg_loss", {"type": "dice", "args": None})
    patch_criterion = build_loss_fn(patch_loss_cfg["type"], patch_loss_cfg.get("args"))
    aggreg_criterion = build_loss_fn(aggreg_loss_cfg["type"], aggreg_loss_cfg.get("args"))
    model.set_loss_functions(patch_criterion, aggreg_criterion)
    if verbose:
        print(f"Loss functions: patch={patch_loss_cfg['type']}, aggreg={aggreg_loss_cfg['type']}")

    # Load checkpoint
    ckpt_path = cfg.get("checkpoint", None)
    if ckpt_path:
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        ckpt_state_dict = checkpoint["model_state_dict"]

        # Filter out gaussian kernel buffers (shape depends on spread_sigma, not learned)
        gaussian_keys = [k for k in ckpt_state_dict.keys() if '_gaussian_kernel' in k]
        for k in gaussian_keys:
            del ckpt_state_dict[k]
        if gaussian_keys and verbose:
            print(f"Filtered {len(gaussian_keys)} gaussian kernel buffers from checkpoint")

        missing, unexpected = model.load_state_dict(ckpt_state_dict, strict=False)
        if verbose:
            epoch = checkpoint.get("epoch", "?")
            dice = checkpoint.get("best_dice", float("nan"))
            print(f"Loaded checkpoint from {ckpt_path} (epoch {epoch}, dice {dice:.4f})")
            if missing:
                fe_missing = [k for k in missing if k.startswith("feature_extractor.")]
                other_missing = [k for k in missing if not k.startswith("feature_extractor.")]
                if other_missing:
                    print(f"Note: {len(other_missing)} keys missing from checkpoint: {other_missing[:5]}...")
    elif verbose:
        print("No checkpoint loaded, using default weights")

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if verbose:
        print(f"Model parameters: {num_params:,}")

    return model


def _build_feature_extractor(cfg, patch_icl_cfg, device, verbose=True):
    """Build feature extractor from config."""
    fe_cfg = patch_icl_cfg.get("feature_extractor", None)
    if fe_cfg is not None:
        extractor_type = fe_cfg.get("type", "meddino").lower()
    else:
        extractor_type = cfg.get("feature_extractor_type", "meddino").lower()

    if extractor_type in ["meddino", "meddinov3", "meddino_v3"]:
        from src.models.meddino_extractor import create_meddino_extractor
        if fe_cfg is not None:
            fe = create_meddino_extractor(
                model_path=fe_cfg.get("model_path", cfg.paths.ckpts.meddino_vit),
                target_size=fe_cfg.get("target_size", 256),
                device=device,
                layer_idx=fe_cfg.get("layer_idx", 11),
                freeze=fe_cfg.get("freeze", True),
            )
        else:
            fe = create_meddino_extractor(
                model_path=cfg.paths.ckpts.meddino_vit,
                target_size=cfg.get("feature_extraction_resolution", 256),
                device=device,
                layer_idx=cfg.get("meddino_layer_idx", 11),
                freeze=True,
            )
        if verbose:
            print(f"Feature mode: on_the_fly (MedDINO)")

    elif extractor_type in ["medsam_v1", "medsam_v1_layer"]:
        from src.models.medsam_extractor import MedSAMv1LayerExtractor
        target_size = fe_cfg.get("target_size", 1024) if fe_cfg else 1024
        output_grid = fe_cfg.get("output_grid_size") if fe_cfg else None
        fe = MedSAMv1LayerExtractor(
            checkpoint_path=fe_cfg.get("checkpoint_path") if fe_cfg else None,
            target_size=target_size,
            device=device,
            layer_idx=fe_cfg.get("layer_idx", 11) if fe_cfg else 11,
            freeze=fe_cfg.get("freeze", True) if fe_cfg else True,
            output_grid_size=output_grid,
        )
        if verbose:
            info = fe.get_feature_info()
            print(f"Feature mode: on_the_fly (MedSAM v1 layer {info['layer_idx']}, "
                  f"grid={info['output_grid_size']})")

    elif extractor_type == "universeg":
        from src.models.universeg_extractor import UniverSegExtractor
        fe = UniverSegExtractor(
            layer_idx=fe_cfg.get("layer_idx", 3) if fe_cfg else 3,
            device=device,
            pretrained=fe_cfg.get("pretrained", True) if fe_cfg else True,
            freeze=fe_cfg.get("freeze", True) if fe_cfg else True,
            output_grid_size=fe_cfg.get("output_grid_size") if fe_cfg else None,
            input_size=fe_cfg.get("input_size", 128) if fe_cfg else 128,
            skip_preprocess=fe_cfg.get("skip_preprocess", True) if fe_cfg else True,
        )
        if verbose:
            info = fe.get_feature_info()
            print(f"Feature mode: on_the_fly (UniverSeg layers={info['layer_indices']}, "
                  f"dim={info['feature_dim']}, input={info['input_size']}x{info['input_size']}, "
                  f"grid={info['output_grid_size']}, skip_preprocess={info['skip_preprocess']})")

    elif extractor_type == "icl_encoder":
        from src.models.icl_encoder import ICLEncoder
        fe = ICLEncoder(
            layer_idx=fe_cfg.get("layer_idx", "all") if fe_cfg else "all",
            output_grid_size=fe_cfg.get("output_grid_size") if fe_cfg else None,
            freeze=fe_cfg.get("freeze", False) if fe_cfg else False,
        )
        if verbose:
            info = fe.get_feature_info()
            print(f"Feature mode: on_the_fly (ICLEncoder layers={info['layer_indices']}, "
                  f"dim={info['feature_dim']}, grid={info['output_grid_size']})")

    elif extractor_type == "rad_dino":
        from src.models.rad_dino_extractor import RADDINOExtractor
        fe = RADDINOExtractor(
            model_name=fe_cfg.get("model_name", "microsoft/rad-dino") if fe_cfg else "microsoft/rad-dino",
            target_size=fe_cfg.get("target_size", 224) if fe_cfg else 224,
            output_grid_size=fe_cfg.get("output_grid_size") if fe_cfg else None,
            device=device,
            freeze=fe_cfg.get("freeze", True) if fe_cfg else True,
        )
        if verbose:
            info = fe.get_feature_info()
            print(f"Feature mode: on_the_fly (RAD-DINO model={info['model_name']}, "
                  f"dim={info['feature_dim']}, grid={info['output_grid_size']}, frozen={info['frozen']})")

    elif extractor_type == "medsam2":
        from src.models.medsam2_extractor import MedSAM2Extractor
        fe = MedSAM2Extractor(
            layer_idx=fe_cfg.get("layer_idx", 0) if fe_cfg else 0,
            device=device,
            checkpoint_path=fe_cfg.get("checkpoint_path") if fe_cfg else None,
            freeze=fe_cfg.get("freeze", True) if fe_cfg else True,
            output_grid_size=fe_cfg.get("output_grid_size") if fe_cfg else None,
            input_size=fe_cfg.get("input_size", 512) if fe_cfg else 512,
            compile_model=fe_cfg.get("compile_model", False) if fe_cfg else False,
        )
        if verbose:
            info = fe.get_feature_info()
            print(f"Feature mode: on_the_fly (MedSAM2 layers={info['layer_indices']}, "
                  f"dim={info['feature_dim']}, input={info['input_size']}x{info['input_size']}, "
                  f"grid={info['output_grid_size']})")

    else:
        raise ValueError(f"Unknown feature_extractor_type: {extractor_type}")

    return fe
