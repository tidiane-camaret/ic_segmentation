# In-Context Medical Image Segmentation

This repository contains a framework for evaluating and comparing different approaches to in-context medical image segmentation. The framework supports various models including simple baselines, fine-tuned models, and large foundation models like SegGPT.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/tidiane-camaret/ic_segmentation
cd ic_segmentation
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate
```

3. Install dependencies


4. For SegGPT support, add the [cloned Painter repo](https://github.com/baaivision/Painter/) to your Python path:
```bash
export PYTHONPATH="/path/to/Painter/SegGPT/SegGPT_inference:$PYTHONPATH"
```

## Project Structure

```
.
├── src/                    # Source code
│   ├── evaluator.py       # Evaluation framework
│   └── segmentation_models.py            # Model implementations
├── scripts/               # Utility scripts
│   ├── read_nii_files.ipynb  # Data exploration
│   ├── create_dataset.ipynb  # Dataset creation
│   └── eval_seg_model.py  # Model evaluation
└── data/                  # Dataset directory
```

## Supported Models

### CopyPrompt
A baseline model that simply copies the prompt mask as the prediction.

### FineTunedModel
A model that fine-tunes a pre-trained segmentation model on each prompt image/mask pair before making predictions.

### SegGPT
Integration with the SegGPT model for in-context segmentation.

### Neuroverse3D
Memory-efficient In-Context Learning model for 3D medical imaging. Supports training and evaluation on volumetric data. See [Neuroverse3D Training Guide](docs/NEUROVERSE3D_TRAINING.md) for detailed instructions.

## Usage

### Evaluation

1. Data Preparation:
```bash
# Explore the NIFTI files
scripts/read_nii_files.ipynb

# Create the evaluation dataset
scripts/create_dataset.ipynb
```

2. Model Evaluation:
```bash
# Run evaluation on all models
python scripts/eval_seg_model.py
```

### Training Neuroverse3D

1. Prepare data in nnUNet format:
```bash
python scripts/prepare_neuroverse3d_data.py \
    --input-dir /path/to/raw/data \
    --output-dir data/neuroverse3d \
    --dataset-name MyDataset
```

2. Stage 1 Training (fixed context):
```bash
python scripts/train_neuroverse3d.py \
    --stage 1 \
    --data-dir data/neuroverse3d \
    --context-size 3 \
    --epochs 50 \
    --lr 0.00001
```

3. Stage 2 Training (variable context):
```bash
python scripts/train_neuroverse3d.py \
    --stage 2 \
    --checkpoint checkpoints/neuroverse3d/stage1/best.ckpt \
    --epochs 100 \
    --lr 0.000002
```

For detailed training instructions, see the [Neuroverse3D Training Guide](docs/NEUROVERSE3D_TRAINING.md).

## Evaluation Metrics

The framework provides:
- Dice coefficient
- Inference time measurements
- Visualization of results
- Integration with Weights & Biases for experiment tracking : https://wandb.ai/tidiane/ic_segmentation/workspace

## Computing Resources

For GPU access on the cluster:
```bash
srun -p ml_gpu-rtx2080 --time=3:00:00 --pty bash
```

## Contributing

1. Fork the repository
2. Create a new branch for your feature
3. Submit a pull request

## License

MIT

