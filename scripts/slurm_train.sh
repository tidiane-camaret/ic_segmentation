#!/bin/bash
#SBATCH -p ml_gpu-rtx2080
#SBATCH -c 20
#SBATCH --mem 48000
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00


# interactive session : 
#srun -p ml_gpu-rtx2080 -c 20 --mem 48000 --gres=gpu:2 --time=12:00:00 --pty bash 
#srun -p ml_gpu-rtx2080 -c 20 --mem 24000 --time=4:00:00 --pty bash 

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

# run with sbatch scripts/slurm_train.sh
# uv run scripts/totalseg_3d_to_2d_every_n_slice.py cluster=dlclarge max_files_3d_to_2d=500
uv run accelerate launch \
    --multi_gpu \
    --num_processes=2 \
    scripts/train.py \
    experiment=131_synthmorph \
    cluster=dlclarge \
    #checkpoint=/work/dlclarge2/ndirt-SegFM3D/ic_segmentation/results/checkpoints/2026-02-15_alluring-ring-271/best_model.pt
