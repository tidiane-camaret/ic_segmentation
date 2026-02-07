#!/bin/bash
#SBATCH -p ml_gpu-rtx2080
#SBATCH -c 20
#SBATCH --mem 24000
#SBATCH --gres=gpu:2
#SBATCH --time=12:00:00

# run with sbatch scripts/slurm_batch.sh
# uv run scripts/totalseg_3d_to_2d.py cluster=dlclarge
# uv run scripts/extract_dinov3_features.py cluster=dlclarge
uv run accelerate launch --mixed_precision fp16 --multi_gpu scripts/train.py experiment=30_medsegbench cluster=dlclarge

