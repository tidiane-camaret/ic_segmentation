# Neuroverse3D Training Pipeline

This document provides comprehensive instructions for training the Neuroverse3D model for in-context learning on 3D medical images.

## Overview

Neuroverse3D is a memory-efficient In-Context Learning (ICL) model designed for 3D medical imaging that supports large context sizes without proportional memory scaling. It can perform tasks like segmentation and image transformation without retraining.

## Table of Contents

1. [Installation](#installation)
2. [Data Preparation](#data-preparation)
3. [Training](#training)
4. [Evaluation](#evaluation)
5. [Pre-trained Models](#pre-trained-models)

## Installation

### 1. Install Dependencies

```bash
pip install pytorch-lightning wandb nibabel scipy
```

### 2. Clone Neuroverse3D Repository

The actual Neuroverse3D model needs to be obtained from the official repository:

```bash
cd /path/to/repos
git clone https://github.com/jiesihu/Neuroverse3D
export PYTHONPATH="/path/to/Neuroverse3D:$PYTHONPATH"
```

### 3. Replace Placeholder Model

Copy the actual model files into your project:

```bash
# Option 1: Add to PYTHONPATH (recommended)
export PYTHONPATH="/path/to/Neuroverse3D:$PYTHONPATH"

# Option 2: Copy model files
cp -r /path/to/Neuroverse3D/neuroverse3D/* src/neuroverse3d/
```

## Data Preparation

### nnUNet Format

Neuroverse3D requires data in nnUNet format:

```
data/neuroverse3d/
├── imagesTr/
│   ├── case_001_0000.nii.gz
│   ├── case_002_0000.nii.gz
│   └── ...
├── labelsTr/
│   ├── case_001.nii.gz
│   ├── case_002.nii.gz
│   └── ...
└── dataset.json
```

### Convert Your Data

Use the provided data preparation script:

```bash
python scripts/prepare_neuroverse3d_data.py \
    --input-dir /path/to/raw/data \
    --output-dir data/neuroverse3d \
    --dataset-name MyDataset \
    --modality MRI
```

**Arguments:**
- `--input-dir`: Directory containing your raw NIfTI files
- `--output-dir`: Where to save the nnUNet-formatted data
- `--image-pattern`: Glob pattern for image files (default: `*_img.nii.gz`)
- `--mask-pattern`: Glob pattern for mask files (default: `*_gt.nii.gz`)
- `--dataset-name`: Name of your dataset
- `--modality`: Imaging modality (MRI, CT, etc.)

### Manual Data Preparation

If you prefer to prepare data manually:

1. **Create directory structure:**
   ```bash
   mkdir -p data/neuroverse3d/{imagesTr,labelsTr}
   ```

2. **Name files using nnUNet convention:**
   - Images: `case_XXX_0000.nii.gz` (XXX = case number)
   - Labels: `case_XXX.nii.gz`

3. **Ensure matching dimensions:**
   - Images and masks must have the same spatial dimensions
   - Recommended resolution: 128×128×128 (will be resized automatically)

## Training

Training follows a two-stage approach:

### Stage 1: Initial Training with Fixed Context

Train the model with a fixed number of context examples (e.g., 3):

```bash
python scripts/train_neuroverse3d.py \
    --stage 1 \
    --data-dir data/neuroverse3d \
    --context-size 3 \
    --epochs 50 \
    --lr 0.00001 \
    --batch-size 1 \
    --num-gpus 1 \
    --checkpoint-dir checkpoints/neuroverse3d/stage1
```

**Key Parameters:**
- `--stage 1`: Stage 1 training
- `--context-size 3`: Use 3 context examples
- `--epochs 50`: Train for 50 epochs
- `--lr 0.00001`: Learning rate (1e-5)
- `--num-gpus`: Number of GPUs to use

### Stage 2: Fine-tuning with Variable Context

Fine-tune the Stage 1 model with variable context sizes (2-9 examples):

```bash
python scripts/train_neuroverse3d.py \
    --stage 2 \
    --data-dir data/neuroverse3d \
    --checkpoint checkpoints/neuroverse3d/stage1/best.ckpt \
    --epochs 100 \
    --lr 0.000002 \
    --batch-size 1 \
    --num-gpus 1 \
    --checkpoint-dir checkpoints/neuroverse3d/stage2
```

**Key Parameters:**
- `--stage 2`: Stage 2 training
- `--checkpoint`: Path to Stage 1 checkpoint
- `--epochs 100`: Train for 100 epochs
- `--lr 0.000002`: Lower learning rate (2e-6)
- Context size is randomly sampled between 2-9

### Advanced Options

**Multi-GPU Training:**
```bash
python scripts/train_neuroverse3d.py \
    --stage 1 \
    --data-dir data/neuroverse3d \
    --num-gpus 4 \
    --batch-size 2
```

**Mixed Precision Training:**
```bash
python scripts/train_neuroverse3d.py \
    --stage 1 \
    --data-dir data/neuroverse3d \
    --mixed-precision
```

**Custom Dataset Selection:**
```bash
python scripts/train_neuroverse3d.py \
    --stage 1 \
    --data-dir data/neuroverse3d \
    --datasets Dataset001 Dataset002 Dataset003 \
    --sample-rate 0.5
```

**Disable W&B Logging:**
```bash
python scripts/train_neuroverse3d.py \
    --stage 1 \
    --data-dir data/neuroverse3d \
    --no-wandb
```

## Evaluation

### Using the Trained Model

After training, evaluate the model on your test set:

```python
import sys
sys.path.append("/path/to/Neuroverse3D")

from neuroverse3D.lightning_model import LightningModel
from src.neuroverse3d.dataloader import Neuroverse3DDataset
import torch
import nibabel as nib
import numpy as np

# Load trained model
checkpoint_path = 'checkpoints/neuroverse3d/stage2/best.ckpt'
model = LightningModel.load_from_checkpoint(checkpoint_path)
model.eval()
model.to('cuda')

# Load test data
test_dataset = Neuroverse3DDataset(
    data_dir='data/neuroverse3d',
    mode='val',
    augment=False
)

# Get a test sample
sample = test_dataset[0]
target_in = sample['image'].unsqueeze(0).cuda()
target_out = sample['label']

# Create context (use other samples as context)
context_samples = [test_dataset[i] for i in range(1, 4)]
context_in = torch.stack([s['image'] for s in context_samples]).unsqueeze(0).cuda()
context_out = torch.stack([s['label'] for s in context_samples]).unsqueeze(0).cuda()

# Run inference
with torch.no_grad():
    pred_mask = model.forward(target_in, context_in, context_out, gs=2)

# Save prediction
pred_mask = pred_mask.cpu().numpy()[0, 0]
pred_nii = nib.Nifti1Image(pred_mask, affine=np.eye(4))
nib.save(pred_nii, 'prediction.nii.gz')
```

### Computing Metrics

```python
from sklearn.metrics import balanced_accuracy_score
import numpy as np

# Binarize predictions
pred_binary = (pred_mask > 0.5).astype(int)
target_binary = target_out.numpy()[0]

# Compute Dice score
intersection = (pred_binary * target_binary).sum()
dice = (2.0 * intersection) / (pred_binary.sum() + target_binary.sum())

print(f"Dice Score: {dice:.4f}")
```

## Pre-trained Models

### Download Pre-trained Checkpoint

The Neuroverse3D team provides a pre-trained checkpoint:

```python
# Download from Baidu Netdisk (access code: rue4)
# Or from the GitHub repository releases
checkpoint_path = '/path/to/neuroverse3D.ckpt'

# Load pre-trained model
from neuroverse3D.lightning_model import LightningModel
model = LightningModel.load_from_checkpoint(checkpoint_path)
```

### Fine-tune Pre-trained Model

You can fine-tune the pre-trained model on your own data:

```bash
python scripts/train_neuroverse3d.py \
    --stage 1 \
    --data-dir data/your_dataset \
    --checkpoint /path/to/neuroverse3D.ckpt \
    --epochs 20 \
    --lr 0.000001
```

## Troubleshooting

### Common Issues

**1. Out of Memory Error**

- Reduce batch size: `--batch-size 1`
- Use mixed precision: `--mixed-precision`
- Reduce context size: `--context-size 2`

**2. Model Not Found**

Make sure you've cloned the Neuroverse3D repository and added it to PYTHONPATH:

```bash
export PYTHONPATH="/path/to/Neuroverse3D:$PYTHONPATH"
```

**3. Data Loading Errors**

Verify your data is in the correct nnUNet format:

```bash
ls data/neuroverse3d/imagesTr/
# Should show: case_001_0000.nii.gz, case_002_0000.nii.gz, ...

ls data/neuroverse3d/labelsTr/
# Should show: case_001.nii.gz, case_002.nii.gz, ...
```

**4. Slow Training**

- Increase number of workers: `--num-workers 8`
- Use multiple GPUs: `--num-gpus 4`
- Enable mixed precision: `--mixed-precision`

## References

- **Neuroverse3D GitHub**: https://github.com/jiesihu/Neuroverse3D
- **Paper**: *Towards Robust In-Context Learning for Medical Image Segmentation via Data Synthesis*
- **nnUNet Format**: https://github.com/MIC-DKFZ/nnUNet

## Citation

If you use Neuroverse3D in your research, please cite:

```bibtex
@article{neuroverse3d,
  title={Towards Robust In-Context Learning for Medical Image Segmentation via Data Synthesis},
  author={...},
  journal={...},
  year={2024}
}
```
