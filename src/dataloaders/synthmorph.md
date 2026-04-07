# Reproducing UniverSeg Synthetic Data Generation

## Instructions for Claude Code

These instructions describe how to reproduce the synthetic segmentation task generation from "UniverSeg: Universal Medical Image Segmentation" (arXiv:2304.06131, Butoi et al., ICCV 2023). The generation adapts SynthMorph's random shapes procedure (Hoffmann et al., 2022) to produce synthetic binary segmentation tasks on-the-fly during training.

The code is **dimension-agnostic** (2D or 3D) from the start. UniverSeg operated in 2D at 128x128, but the underlying SynthMorph procedure is natively 3D, and the pipeline generalizes trivially.

---

## Goal

Build a PyTorch Dataset/generator that produces synthetic segmentation tasks **on-the-fly** with no disk storage. Each call yields a (image, label) pair from a procedurally generated task. The generator must:

1. Support both 2D (H, W) and 3D (D, H, W) spatial dimensions.
2. Be deterministic per task (same task seed = same anatomy) while varying across subjects.
3. Be fast enough that data generation does not bottleneck GPU training.

---

## Architecture Overview

```
SyntheticTaskGenerator
  Maintains a pool of "task seeds" (one per task).
  Each seed deterministically defines an anatomy (base label map).
  On each __getitem__ call:
    1. Pick a task seed
    2. Regenerate (or cache) the base label map
    3. Apply a random deformation -> subject label
    4. Synthesize intensity image from label
    5. Pick a random foreground label -> binary mask
    6. Return (image, binary_mask)
```

**Why on-the-fly:** SynthMorph itself generates data at every mini-batch with no pre-stored dataset. UniverSeg pre-generated 1,000 tasks x 100 subjects stored in LMDB, but on-the-fly is preferable because:
- No disk footprint (3D at 128^3 x 1,000 tasks x 100 subjects = ~250 GB).
- Effectively infinite subject diversity (every call produces a new deformation).
- Simpler pipeline (no separate generation phase).
- Matches SynthMorph's own approach.

The tradeoff is CPU cost per sample. Mitigate by caching base label maps (one per task, ~2 MB each for 128^3) and only generating deformations + images on the fly.

---

## Environment Setup

```bash
pip install torch numpy scipy
```

Optional reference implementations:

```bash
# PyTorch reimplementation of SynthMorph (label generation + training)
git clone https://github.com/matt-kh/synthmorph-torch.git

# Dalca lab PyTorch synthesis/augmentation library
pip install git+https://github.com/dalcalab/voxynth.git

# Fast Perlin noise (numpy, 2D and 3D)
pip install git+https://github.com/pvigier/perlin-numpy.git
```

### Reference Repositories

- synthmorph-torch (github.com/matt-kh/synthmorph-torch): PyTorch label-from-noise, image synthesis. See synthmorph_shapes.ipynb.
- voxynth (github.com/dalcalab/voxynth): PyTorch synthesis utilities from the same lab.
- VoxelMorph (github.com/voxelmorph/voxelmorph): Original TF SynthMorph. train_synthmorph.py generates on-the-fly.
- neurite (github.com/adalca/neurite): TF/Keras layers for label-from-noise, image-from-labels.
- SynthMorph demos (synthmorph.voxelmorph.net): Official Colab notebooks (shapes demo, affine demo).
- perlin-numpy (github.com/pvigier/perlin-numpy): 2D/3D Perlin noise in numpy.
- UniverSeg (github.com/JJGO/UniverSeg): Model architecture, inference only.

---

## Generation Pipeline

Three stages, all dimension-agnostic. ndim is 2 or 3, shape is e.g. (128, 128) or (128, 128, 128).

### Stage 1: Base Label Map

Generate a discrete label map with num_labels blob-like regions.

Algorithm (from SynthMorph, Section II-B of Hoffmann et al. 2022):
1. Sample num_labels independent noise volumes of size shape from N(0, 1).
2. Smooth each with a Gaussian kernel. Sample sigma per-volume from U(sigma_min, sigma_max) to get shapes at varying spatial scales.
3. (Optional) Warp each smoothed noise volume with a small random deformation before argmax, adding irregularity to region boundaries.
4. Argmax across the num_labels volumes at each voxel -> discrete label map with values in {0, ..., num_labels-1}.

```python
import numpy as np
from scipy.ndimage import gaussian_filter

def generate_base_label_map(shape, num_labels=16, sigma_range=(5.0, 15.0), rng=None):
    if rng is None:
        rng = np.random.default_rng()
    noise_stack = np.zeros((num_labels, *shape), dtype=np.float32)
    for j in range(num_labels):
        raw = rng.standard_normal(shape).astype(np.float32)
        sigma = rng.uniform(*sigma_range)
        noise_stack[j] = gaussian_filter(raw, sigma=sigma)
    return np.argmax(noise_stack, axis=0).astype(np.int16)
```

Cost for 3D: ~0.5-1s for 16 volumes of 128^3. Cached per task, so paid once.

### Stage 2: Random Deformation (Subject Variation)

Apply a random smooth deformation field to produce a subject-specific variant.

```python
from scipy.ndimage import gaussian_filter, map_coordinates

def random_deformation_field(shape, sigma_def=2.0, sigma_smooth=8.0, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    ndim = len(shape)
    flow = rng.normal(0, sigma_def, size=(ndim, *shape)).astype(np.float32)
    for d in range(ndim):
        flow[d] = gaussian_filter(flow[d], sigma=sigma_smooth)
    return flow

def apply_deformation(volume, flow, order=0):
    shape = volume.shape
    ndim = len(shape)
    grid = np.mgrid[tuple(slice(0, s) for s in shape)].astype(np.float64)
    coords = grid + flow.astype(np.float64)
    return map_coordinates(
        volume.astype(np.float64), coords, order=order, mode='reflect'
    ).astype(volume.dtype)
```

Cost for 3D: map_coordinates on 128^3 takes ~0.1-0.3s. This is the main per-sample cost. For GPU acceleration, see the grid_sample approach below.

### Stage 3: Intensity Synthesis

```python
def synthesize_image(label_map, num_labels=16, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    shape = label_map.shape
    image = np.zeros(shape, dtype=np.float32)

    # Per-region intensity (GMM-like)
    for l in range(num_labels):
        mask = (label_map == l)
        if not mask.any():
            continue
        mu = rng.uniform(0.0, 1.0)
        sigma = rng.uniform(0.01, 0.05)
        image[mask] = rng.normal(mu, sigma, size=mask.sum()).astype(np.float32)

    # Global Gaussian noise
    noise_std = rng.uniform(0.0, 0.05)
    image += rng.normal(0, noise_std, size=shape).astype(np.float32)

    # Smooth spatial noise (portable Perlin substitute, works in any dim)
    smooth_noise = rng.standard_normal(shape).astype(np.float32)
    smooth_noise = gaussian_filter(smooth_noise, sigma=15.0)
    sn_range = smooth_noise.max() - smooth_noise.min()
    if sn_range > 0:
        smooth_noise = 2.0 * (smooth_noise - smooth_noise.min()) / sn_range - 1.0
    amplitude = rng.uniform(0.0, 0.1)
    image += amplitude * smooth_noise

    return np.clip(image, 0.0, 1.0)
```

Why smoothed Gaussian noise instead of Perlin: perlin-numpy only supports 2D/3D with shape constraints. Smoothed Gaussian with large sigma produces equivalent low-frequency texture and works in any dimension. For true Perlin, use perlin_numpy.generate_perlin_noise_2d or generate_perlin_noise_3d.

---

## On-the-Fly Generator (PyTorch Dataset)

```python
import torch
from torch.utils.data import Dataset

class SyntheticTaskDataset(Dataset):
    def __init__(
        self,
        num_tasks=1000,
        num_labels=16,
        shape=(128, 128),
        sigma_range=(5.0, 15.0),
        sigma_def=2.0,
        sigma_smooth=8.0,
        epoch_length=10000,
        master_seed=42,
        max_cache_size=None,  # None = cache all; set e.g. 200 for 3D
    ):
        self.num_tasks = num_tasks
        self.num_labels = num_labels
        self.shape = shape
        self.sigma_range = sigma_range
        self.sigma_def = sigma_def
        self.sigma_smooth = sigma_smooth
        self.epoch_length = epoch_length
        self.max_cache_size = max_cache_size

        master_rng = np.random.default_rng(master_seed)
        self.task_seeds = [int(master_rng.integers(0, 2**31)) for _ in range(num_tasks)]

        self._label_cache = {}
        self._cache_order = []

    def __len__(self):
        return self.epoch_length

    def _get_base_label(self, task_idx):
        if task_idx not in self._label_cache:
            rng = np.random.default_rng(self.task_seeds[task_idx])
            label = generate_base_label_map(
                shape=self.shape, num_labels=self.num_labels,
                sigma_range=self.sigma_range, rng=rng,
            )
            if self.max_cache_size and len(self._label_cache) >= self.max_cache_size:
                evict = self._cache_order.pop(0)
                del self._label_cache[evict]
            self._label_cache[task_idx] = label
            self._cache_order.append(task_idx)
        else:
            if task_idx in self._cache_order:
                self._cache_order.remove(task_idx)
                self._cache_order.append(task_idx)
        return self._label_cache[task_idx]

    def __getitem__(self, idx):
        rng = np.random.default_rng()  # fresh randomness each call
        task_idx = rng.integers(0, self.num_tasks)
        base_label = self._get_base_label(task_idx)

        flow = random_deformation_field(self.shape, self.sigma_def, self.sigma_smooth, rng=rng)
        subject_label = apply_deformation(base_label, flow, order=0)
        image = synthesize_image(subject_label, self.num_labels, rng=rng)

        fg_label = rng.integers(0, self.num_labels)
        binary_label = (subject_label == fg_label).astype(np.float32)

        image_t = torch.from_numpy(image).unsqueeze(0)
        label_t = torch.from_numpy(binary_label).unsqueeze(0)
        return image_t, label_t

    def get_support_set(self, task_idx, fg_label, num_support, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        base_label = self._get_base_label(task_idx)
        images, labels = [], []
        for _ in range(num_support):
            flow = random_deformation_field(self.shape, self.sigma_def, self.sigma_smooth, rng=rng)
            subj_label = apply_deformation(base_label, flow, order=0)
            img = synthesize_image(subj_label, self.num_labels, rng=rng)
            binary = (subj_label == fg_label).astype(np.float32)
            images.append(torch.from_numpy(img).unsqueeze(0))
            labels.append(torch.from_numpy(binary).unsqueeze(0))
        return torch.stack(images), torch.stack(labels)

    def sample_episode(self, num_support=64, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        task_idx = rng.integers(0, self.num_tasks)
        fg_label = rng.integers(0, self.num_labels)
        base_label = self._get_base_label(task_idx)

        flow = random_deformation_field(self.shape, self.sigma_def, self.sigma_smooth, rng=rng)
        q_label = apply_deformation(base_label, flow, order=0)
        q_image = synthesize_image(q_label, self.num_labels, rng=rng)
        q_binary = (q_label == fg_label).astype(np.float32)

        s_images, s_labels = self.get_support_set(task_idx, fg_label, num_support, rng=rng)

        return (
            torch.from_numpy(q_image).unsqueeze(0),
            torch.from_numpy(q_binary).unsqueeze(0),
            s_images, s_labels,
        )
```

### Usage

```python
# 2D
dataset_2d = SyntheticTaskDataset(num_tasks=1000, shape=(128, 128), epoch_length=10000)

# 3D
dataset_3d = SyntheticTaskDataset(num_tasks=1000, shape=(128, 128, 128), epoch_length=10000, max_cache_size=200)

# DataLoader
from torch.utils.data import DataLoader
loader = DataLoader(dataset_2d, batch_size=1, num_workers=4)

# Full episode for UniverSeg training
q_img, q_lbl, s_imgs, s_lbls = dataset_2d.sample_episode(num_support=64)
```

---

## Performance Considerations

### Timing Estimates (CPU, single-threaded)

| Operation                          | 2D (128x128) | 3D (128^3)     |
|------------------------------------|-------------|----------------|
| Base label map generation          | ~50 ms      | ~500 ms-1s     |
| Random deformation field           | ~10 ms      | ~200 ms        |
| Apply deformation (map_coordinates)| ~5 ms       | ~200 ms        |
| Image synthesis                    | ~10 ms      | ~300 ms        |
| **Total per sample**               | **~75 ms**  | **~700 ms-1.5s** |
| Per sample with cached base label  | ~25 ms      | ~700 ms        |

### Faster Generation Strategies

1. **Cache base label maps** (already implemented). Cost paid once per task.

2. **num_workers > 0 in DataLoader.** Each worker generates on CPU in parallel. 4 workers = ~4x throughput. Usually sufficient for 2D.

3. **GPU deformation with grid_sample** (recommended for 3D):

```python
import torch
import torch.nn.functional as F

def apply_deformation_torch(volume_t, flow_t):
    """
    GPU-accelerated deformation.
    Args:
        volume_t: (1, 1, *shape) tensor on GPU
        flow_t: (1, ndim, *shape) displacement in voxels
    Returns:
        warped: (1, 1, *shape) tensor
    """
    shape = volume_t.shape[2:]
    ndim = len(shape)
    vectors = [torch.arange(0, s, device=volume_t.device, dtype=torch.float32) for s in shape]
    grid = torch.stack(torch.meshgrid(*vectors, indexing='ij'))
    coords = grid + flow_t[0]
    for d in range(ndim):
        coords[d] = 2.0 * coords[d] / (shape[d] - 1) - 1.0

    if ndim == 2:
        gs_grid = coords.permute(1, 2, 0).unsqueeze(0)[..., [1, 0]]
    elif ndim == 3:
        gs_grid = coords.permute(1, 2, 3, 0).unsqueeze(0)[..., [2, 1, 0]]

    return F.grid_sample(volume_t, gs_grid, mode='nearest', padding_mode='border', align_corners=True)
```

4. **GPU deformation field generation.** Replace scipy gaussian_filter with torch convolution using a Gaussian kernel + generate noise on GPU. Moves entire per-sample path to GPU.

5. **Pre-generate deformation pool.** Store ~500 random fields in GPU memory; randomly select/compose at training time.

6. **Lower resolution for 3D.** 64^3 or 96^3 instead of 128^3 reduces cost 4-8x.

### Recommended Configuration

| Setting          | 2D             | 3D                              |
|------------------|----------------|----------------------------------|
| shape            | (128, 128)     | (128, 128, 128) or (96, 96, 96) |
| num_workers      | 4              | 4-8                              |
| Base label cache | All 1000 (~250 MB) | LRU 100-200 (~400-800 MB)   |
| Deformation      | CPU (scipy)    | GPU (grid_sample) recommended    |

---

## Parameter Reference

### Base Label Map

| Parameter    | Value         | Notes                                    |
|-------------|---------------|------------------------------------------|
| num_labels  | 16            | Regions per task                         |
| sigma_range | (5.0, 15.0)   | Gaussian smoothing; controls region size |
| shape       | (128, 128) or (128, 128, 128) | Paper uses 128x128 for 2D |

### Deformation Field

| Parameter    | Value     | Notes                                     |
|-------------|-----------|-------------------------------------------|
| sigma_def   | 1.0-3.0   | Displacement magnitude. Paper: alpha in [1, 2.5] |
| sigma_smooth | 6.0-10.0 | Smoothness. Paper: sigma in [7, 8]        |

### Image Synthesis

| Parameter              | Value         |
|------------------------|---------------|
| Per-region mu          | U(0, 1)      |
| Per-region sigma       | U(0.01, 0.05)|
| Gaussian noise sigma   | U(0, 0.05)   |
| Smooth noise sigma_spatial | ~15       |
| Smooth noise amplitude | U(0, 0.1)    |

### Training-Time Augmentations (not part of generation, applied after)

**Task augmentations** (consistent across query + all support):

| Augmentation              | p    | Parameters                                      |
|--------------------------|------|--------------------------------------------------|
| Flip intensities (1-img) | 0.50 | -                                                |
| Flip labels (swap fg/bg) | 0.50 | -                                                |
| Horizontal/vertical flip | 0.50 | -                                                |
| Sobel edge of labels     | 0.50 | -                                                |
| Affine                   | 0.50 | rot [0,360], translate [0,0.2], scale [0.8,1.1]  |
| Brightness/contrast      | 0.50 | brightness [-0.1,0.1], contrast [0.8,1.2]        |
| Elastic warp             | 0.25 | alpha [1,2], sigma [6,8]                          |
| Gaussian blur            | 0.50 | k=5, sigma [0.1,1.1]                             |
| Gaussian noise           | 0.50 | mu [0,0.05], sigma^2 [0,0.05]                    |
| Sharpness                | 0.50 | factor 5                                          |

**In-task augmentations** (independent per support entry):

| Augmentation              | p    | Parameters                                      |
|--------------------------|------|--------------------------------------------------|
| Affine                   | 0.50 | rot [0,360], translate [0,0.2], scale [0.8,1.1]  |
| Brightness/contrast      | 0.25 | brightness [-0.1,0.1], contrast [0.5,1.5]        |
| Gaussian blur            | 0.25 | k=5, sigma [0.1,1.1]                             |
| Gaussian noise           | 0.25 | mu [0,0.05], sigma^2 [0,0.05]                    |
| Sharpness                | 0.25 | factor 5                                          |
| Elastic warp             | 0.80 | alpha [1,2.5], sigma [7,8]                        |

---

## Binary Segmentation Tasks

UniverSeg trains on binary segmentation. At each step, one of 16 labels is randomly selected as foreground. This is already handled in __getitem__ and sample_episode above. Each label per task = one binary sub-task, so 1,000 base tasks yield up to 16,000 effective binary tasks.

---

## Validation

### What to Look For

- Base labels: ~16 distinct blob-like regions with organic curved boundaries (compare Figure 12).
- Across subjects: same anatomy, shifted/deformed boundaries.
- Images: distinct intensity per region, subtle smooth texture.
- 3D: check mid-slices along all axes for coherent structure.

### Common Issues

| Problem                    | Cause               | Fix                              |
|---------------------------|---------------------|-----------------------------------|
| Too few visible regions    | sigma_range too high | Lower to (3, 10)                 |
| Regions too small/speckled | sigma_range too low  | Raise min to 5+                  |
| Deformation folding        | sigma_def too high   | Lower to 1.0-2.0                |
| Subjects identical         | sigma_def too low    | Raise to 2.0+                   |
| 3D too slow                | CPU bottleneck       | GPU grid_sample; lower res; workers |
| Memory issues (3D cache)   | Full cache too large | Use max_cache_size=100-200       |

---

## Integration with Training

```python
for step in range(num_train_steps):
    # Sample from synthetic or medical
    if rng.random() < synthetic_ratio:
        q_img, q_lbl, s_imgs, s_lbls = synthetic_dataset.sample_episode(num_support=64)
    else:
        q_img, q_lbl, s_imgs, s_lbls = medical_dataset.sample_episode(...)

    # Task augmentation (same transform to all)
    q_img, q_lbl, s_imgs, s_lbls = task_augment(q_img, q_lbl, s_imgs, s_lbls)

    # In-task augmentation (independent per entry)
    q_img, q_lbl = intask_augment(q_img, q_lbl)
    for i in range(len(s_imgs)):
        s_imgs[i], s_lbls[i] = intask_augment(s_imgs[i], s_lbls[i])

    pred = model(q_img, s_imgs, s_lbls)
    loss = soft_dice_loss(pred, q_lbl)
    loss.backward()
    optimizer.step()
```