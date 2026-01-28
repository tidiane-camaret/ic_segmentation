#!/bin/bash
#SBATCH -p ml_gpu-rtx2080
#SBATCH -c 20
#SBATCH --mem 24000
#SBATCH --gres=gpu:2
#SBATCH --time=10:00:00

cd ic_segmentation
uv run accelerate launch scripts/train.py cluster=dlclarge experiment=attention
