#!/bin/bash
#SBATCH -p ml_gpu-rtx2080
#SBATCH -c 20
#SBATCH --mem 24000 
#SBATCH --gres=gpu:1
#SBATCH --time=30:00


# interactive session : srun -p ml_gpu-rtx2080 -c 20 --mem 48000 --gres=gpu:2 --time=2:00:00 --pty bash 
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

# run with sbatch scripts/slurm_eval.sh

#uv run accelerate launch --multi_gpu scripts/eval.py experiment=60_2_levels cluster=dlclarge dataset=medsegbench checkpoint=/work/dlclarge2/ndirt-SegFM3D/ic_segmentation/results/checkpoints/deep-feather-217/best_model.pt
uv run accelerate launch \
    scripts/eval.py \
    max_labels=50 \
    experiment=88_entropy \
    cluster=dlclarge \
    checkpoint=/work/dlclarge2/ndirt-SegFM3D/ic_segmentation/results/checkpoints/2026-02-15_wooing-balloon-274/best_model.pt