#!/bin/bash
#SBATCH -p ml_gpu-rtx2080
#SBATCH -c 20
#SBATCH --mem 24000
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00

# run with sbatch scripts/slurm_batch.sh
cd ic_segmentation
# uv run accelerate launch --mixed_precision=fp16 scripts/train.py experiment=high_capacity cluster=dlclarge
uv run scripts/totalseg_3d_to_2d.py cluster=dlclarge
uv run scripts/extract_dinov3_features.py cluster=dlclarge
