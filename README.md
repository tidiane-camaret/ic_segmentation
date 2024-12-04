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

## Usage

1. Data Preparation:
```bash
# Explore the NIFTI files
python scripts/read_nii_files.ipynb

# Create the evaluation dataset
python scripts/create_dataset.ipynb
```

2. Model Evaluation:
```bash
# Run evaluation on all models
python scripts/eval_seg_model.py
```

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

