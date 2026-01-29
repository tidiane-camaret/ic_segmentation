#!/usr/bin/env python3
import sys
import torch
import numpy as np
import threading
from pathlib import Path
from queue import Queue
from typing import List, Union

import torch.nn.functional as F
import hydra
from omegaconf import DictConfig
from tqdm import tqdm

# --- PROCESSOR ---
class MedDINOProcessor:
    def __init__(self, target_size: int = 896, interpolation: str = "bilinear"):
        self.target_size = target_size
        self.interpolation = interpolation
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __call__(self, images: Union[np.ndarray, List[np.ndarray]]) -> torch.Tensor:
        if isinstance(images, np.ndarray):
            images = [images]

        processed_batch = []
        for img in images:
            # 1. Percentile Clipping (0.5% - 99.5%)
            lower, upper = np.percentile(img, [0.5, 99.5])
            img = np.clip(img, lower, upper)

            # 2. Rescale to [0, 1]
            img_min, img_max = img.min(), img.max()
            if img_max > img_min:
                img = (img - img_min) / (img_max - img_min)
            else:
                img = np.zeros_like(img)

            # 3. Grayscale to RGB & ImageNet Normalization
            img_rgb = np.stack([img] * 3, axis=-1)
            img_rgb = (img_rgb - self.mean) / self.std

            # 4. To Tensor & Resize
            img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float()
            img_tensor = F.interpolate(
                img_tensor.unsqueeze(0),
                size=(self.target_size, self.target_size),
                mode=self.interpolation,
                align_corners=False
            ).squeeze(0)
            
            processed_batch.append(img_tensor)

        return torch.stack(processed_batch)

# --- EXTRACTOR ---
class MedDINOFeatureExtractor:
    def __init__(self, model_path: str, target_size: int = 896, device: str = "cuda"):
        self.device = device
        self.processor = MedDINOProcessor(target_size=target_size)
        
        # Adjust path to your specific repository structure
        sys.path.insert(0, "/software/notebooks/camaret/repos/MedDINOv3/nnUNet/nnunetv2/training/nnUNetTrainer/dinov3/")
        from dinov3.models.vision_transformer import vit_base
        
        # Initialize architecture
        self.model = vit_base(
            drop_path_rate=0.2, 
            layerscale_init=1.0e-05, 
            n_storage_tokens=4, 
            qkv_bias=False, 
            mask_k_bias=True
        )

        # Load MedDINOv3 weights
        print(f"Loading weights from {model_path}...")
        chkpt = torch.load(model_path, weights_only=False, map_location='cpu')
        state_dict = chkpt['teacher']
        state_dict = {
            k.replace('backbone.', ''): v
            for k, v in state_dict.items()
            if 'ibot' not in k and 'dino_head' not in k
        }
        self.model.load_state_dict(state_dict)
        self.model.to(device).eval()

    @torch.no_grad()
    def extract(self, imgs: list[np.ndarray]) -> dict:
        pixel_values = self.processor(imgs).to(self.device)
        
        # Layers 2, 5, 8, 11 as per MedDINOv3 paper
        layer_indices = [2, 5, 8, 11]
        
        # Retrieve intermediate layers
        intermediate_layers = self.model.get_intermediate_layers(
            pixel_values, 
            n=layer_indices, 
            reshape=False
        )

        results = {}
        for i, idx in enumerate(layer_indices):
            feat = intermediate_layers[i] # [B, Tokens, Dim]
            
            # Token decomposition: CLS (0), Registers (1:5), Patches (5:)
            results[f"layer_{idx}_cls"] = feat[:, 0, :].cpu().numpy()
            results[f"layer_{idx}_registers"] = feat[:, 1:5, :].cpu().numpy()
            results[f"layer_{idx}_patches"] = feat[:, 5:, :].cpu().numpy()

        return results

# --- STORAGE ---
def save_worker(save_queue, stop_event):
    while not stop_event.is_set() or not save_queue.empty():
        try:
            path, feats = save_queue.get(timeout=0.5)
            out_path = path.parent / (path.stem + "_meddino.npz")
            np.savez_compressed(out_path, **feats)
            save_queue.task_done()
        except:
            continue

# --- MAIN ---
@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig):
    data_dir = Path(cfg.paths.totalseg2d)
    model_path = cfg.paths.ckpts.meddino_vit
    batch_size = cfg.get("batch_size", 512)
    target_res = cfg.get("feature_extraction_resolution", 256)

    extractor = MedDINOFeatureExtractor(model_path, target_size=target_res)
    train_label_ids = ['colon', 'costal_cartilages', 'duodenum', 'iliopsoas_right', 'inferior_vena_cava', 'kidney_left', 'kidney_right', 'lung_middle_lobe_right', 'pancreas', 'scapula_left', 'small_bowel', 'spleen', 'sternum', 'stomach', 'trachea', 'vertebrae_T10', 'vertebrae_T11', 'vertebrae_T12', 'vertebrae_T3', 'vertebrae_T4', 'vertebrae_T5', 'vertebrae_T9']
    val_label_ids = ['aorta', 'autochthon_left', 'autochthon_right', 'esophagus', 'heart', 'iliopsoas_left', 'liver', 'lung_lower_lobe_left', 'lung_lower_lobe_right', 'lung_upper_lobe_left', 'lung_upper_lobe_right', 'scapula_right', 'spinal_cord', 'vertebrae_L1', 'vertebrae_T7', 'vertebrae_T8']
    # 1. Find all potential images
    all_imgs = sorted(data_dir.glob("**/*_slice_img.npy"))
    all_imgs = [p for p in all_imgs if any(label in str(p) for label in train_label_ids + val_label_ids)]
    
    # 2. Filter: Check for existing .npz files to skip redundant work
    to_do = []
    for p in all_imgs:
        output_path = p.parent / (p.stem + "_meddino.npz")
        if not output_path.exists():
            to_do.append(p)
            
    print(f"Total images found: {len(all_imgs)}")
    print(f"Images already processed (skipping): {len(all_imgs) - len(to_do)}")
    print(f"Images remaining to process: {len(to_do)}")

    if not to_do:
        print("Done! No new images to extract.")
        return

    # Save Queue Setup
    q = Queue(maxsize=100)
    stop = threading.Event()
    threads = [threading.Thread(target=save_worker, args=(q, stop)) for _ in range(4)]
    for t in threads: t.start()

    pbar = tqdm(total=len(to_do), desc="MedDINOv3 Extraction")
    for i in range(0, len(to_do), batch_size):
        batch_paths = to_do[i : i + batch_size]
        imgs = [np.load(p).astype(np.float32) for p in batch_paths]
        
        batch_feats = extractor.extract(imgs)

        for j, p in enumerate(batch_paths):
            sample_feats = {key: val[j] for key, val in batch_feats.items()}
            q.put((p, sample_feats))
        
        pbar.update(len(batch_paths))

    q.join()
    stop.set()
    for t in threads: t.join()
    print("Feature extraction complete.")

if __name__ == "__main__":
    main()