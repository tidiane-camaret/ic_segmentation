#!/bin/bash
#SBATCH -p ml_gpu-rtx2080
#SBATCH -c 20
#SBATCH --mem 48000  # Doubled from 24GB to 48GB
#SBATCH --gres=gpu:2
#SBATCH --time=12:00:00


# interactive session : srun -p ml_gpu-rtx2080 -c 20 --mem 48000 --gres=gpu:2 --time=12:00:00 --pty bash 
# NCCL debugging and timeout settings
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
export TORCH_NCCL_TRACE_BUFFER_SIZE=100000
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# Increase NCCL timeout (default is 600s, increase to 1800s = 30min)
export NCCL_TIMEOUT=1800

# Ensure clean GPU state
nvidia-smi

# run with sbatch scripts/slurm_batch.sh
# uv run scripts/totalseg_3d_to_2d.py cluster=dlclarge
# uv run scripts/extract_dinov3_features.py cluster=dlclarge

# uv run accelerate launch --multi_gpu scripts/train.py experiment=40_slidingwindow cluster=dlclarge
uv run accelerate launch --multi_gpu scripts/eval.py experiment=40_slidingwindow cluster=dlclarge # dataset=totalseg2d val_label_ids=val
