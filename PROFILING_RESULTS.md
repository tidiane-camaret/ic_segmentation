# Training Profiling Results

**Date:** 2026-02-12
**Config:** `experiment=60_2_levels`, batch_size=4
**GPU:** NVIDIA RTX 6000 Ada Generation

## Summary of Findings

### 1. Full Training Iteration Breakdown

| Component | Time (ms) | % of Iteration |
|-----------|-----------|----------------|
| **Backward pass** | 322.3ms | **69.9%** |
| Forward pass | 128.4ms | 27.8% |
| Compute loss | 4.9ms | 1.1% |
| Optimizer step | 2.5ms | 0.5% |
| Data loading | 2.4ms | 0.5% |
| Data to device | 0.2ms | <0.1% |
| **Total iteration** | **461.3ms** | 100% |

### 2. Forward Pass Breakdown (113ms total)

| Component | Time (ms) | % of Forward |
|-----------|-----------|--------------|
| **L1: context_sampling** | 32.1ms | **27.9%** |
| FE: img_blocks_all | 14.2ms | 12.4% |
| **L1: target_sampling** | 11.6ms | **10.0%** |
| FE: extract_batch_total | 9.2ms | 8.0% |
| L0: context_sampling | 8.6ms | 7.5% |
| L1: attention | 7.0ms | 6.1% |
| L0: attention | 6.0ms | 5.2% |
| L0: target_sampling | 3.7ms | 3.2% |
| L0: extract_context_features | 3.6ms | 3.1% |
| FE: msk_blocks_all | 2.9ms | 2.6% |
| Others | <3% each | |

### 3. Key Bottlenecks Identified

#### A. **Backward Pass (70% of iteration time)**
The backward pass takes 2.5x longer than the forward pass, which is unusual. This suggests:
- Memory pressure causing GPU stalls
- Large intermediate tensors requiring gradient storage
- Non-fused operations in the computational graph

#### B. **Sampling Operations (48% of forward pass)**
Combined sampling (L0 + L1, target + context) takes ~56ms:
- Level 1 has higher costs due to 16 patches vs 9 patches
- `torch.multinomial` and gather operations cause GPU-CPU sync points
- Each context image (3 total) is sampled separately

#### C. **Feature Extraction (23% of forward pass)**
ICLEncoder takes ~26ms total:
- Image blocks: 14.2ms
- Mask blocks: 2.9ms
- extract_batch_total: 9.2ms

## Root Cause Analysis

### Why GPU utilization fluctuates (20-90%):

1. **CPU-GPU synchronization in sampling**: `torch.multinomial` requires synchronization
2. **Memory-bound operations**: Large feature maps (32x32x256) require frequent memory access
3. **Sequential level processing**: Each level waits for the previous to complete
4. **Backward pass memory pressure**: Storing gradients for all sampling operations

### Why GPU time accessing memory is high (0-100%):

1. **Large intermediate tensors**: Features [B, N, D] with N=1024 tokens
2. **Non-contiguous memory access** in patch extraction
3. **Skip connections** in decoder require storing encoder activations

## Recommendations

### High Impact (Estimated >30% speedup)

1. **Use Mixed Precision (AMP)**
   ```python
   from torch.cuda.amp import autocast, GradScaler
   scaler = GradScaler()
   with autocast():
       outputs = model(...)
       loss = compute_loss(...)
   scaler.scale(loss).backward()
   scaler.step(optimizer)
   ```
   - Reduces memory by 50% for activations
   - Speeds up matmul operations on Ada GPUs

2. **Gradient Checkpointing for Backward Pass**
   ```python
   # In backbone attention
   from torch.utils.checkpoint import checkpoint
   attended = checkpoint(self.attention, encoded, coords, is_context)
   ```
   - Trades compute for memory, but reduces memory pressure
   - Can improve GPU utilization by avoiding memory stalls

3. **Fuse Sampling Operations**
   - Process all context images in parallel instead of loop
   - Use vectorized gather instead of per-context extraction

### Medium Impact (10-20% speedup)

4. **Reduce Sampling Overhead**
   - Use `replacement=True` in multinomial (slightly different semantics but faster)
   - Pre-compute patch extraction indices

5. **Optimize Feature Extraction**
   - Use channels_last memory format for conv layers
   ```python
   model = model.to(memory_format=torch.channels_last)
   images = images.to(memory_format=torch.channels_last)
   ```

6. **Batch Normalization Fusion**
   - Convert BatchNorm to GroupNorm (better for small batches)
   - Or fuse BN into conv weights for inference

### Low Impact (5-10% speedup)

7. **Pin DataLoader Memory** (already fast, but good practice)
   ```python
   DataLoader(..., pin_memory=True)
   ```

8. **Compile Model with torch.compile** (PyTorch 2.0+)
   ```python
   model = torch.compile(model, mode="reduce-overhead")
   ```

## Implementation Priority

1. **First**: Add AMP training support - biggest impact, minimal code change
2. **Second**: Add gradient checkpointing to attention layers
3. **Third**: Fuse context sampling loop into batched operation
4. **Fourth**: Optimize patch extraction with pre-computed indices

## Optimizations Applied (2026-02-12)

### 1. Gumbel-Top-K Sampling (sampling.py)
Replaced `torch.multinomial` with Gumbel-Top-K for faster, sync-free sampling:
```python
# Old (slow - requires CPU-GPU sync):
probs = F.softmax(flat_weights, dim=1)
indices = torch.multinomial(probs, K, replacement=False)

# New (fast - Gumbel-Top-K, no sync):
gumbel = -torch.log(-torch.log(torch.rand_like(flat_weights) + 1e-10) + 1e-10)
scores = flat_weights + gumbel
_, indices = torch.topk(scores, K, dim=1)
```
**Result**: Sampling operations 37-42% faster, forward pass ~44% faster.

### 2. Grid-Sample Patch Extraction (patch_icl.py)
Replaced advanced indexing with `F.grid_sample` for efficient backward pass:
```python
# Old (slow backward due to scatter_add):
patch_features = features[batch_indices, flat_indices]

# New (efficient backward):
patch_features = F.grid_sample(features_spatial, grid, mode='bilinear', ...)
```
**Result**: Backward pass 2.4x faster (262ms → 111ms), backward/forward ratio 0.73x (was 4.2x).

### Summary After Optimizations

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Forward | 128ms | 152ms | -19% (grid_sample overhead) |
| Backward | 322ms | 111ms | **66% faster** |
| Backward/Forward ratio | 2.5x | 0.73x | **Backward now faster than forward** |
| Total iteration | ~461ms | ~263ms | **~43% faster** |

## Data Loading Analysis

Data loading is NOT a bottleneck (2.4ms, 0.5% of iteration):
- In-memory cache is working correctly
- NumWorkers (12) is sufficient
- Augmentations are applied lazily during `__getitem__`
