#!/bin/bash
# Example script for training Neuroverse3D model
# This script demonstrates the complete training workflow

set -e  # Exit on error

echo "=========================================="
echo "Neuroverse3D Training Example"
echo "=========================================="

# Configuration
DATA_DIR="data/neuroverse3d"
CHECKPOINT_DIR="checkpoints/neuroverse3d"
PROJECT_NAME="ic_segmentation"

# Step 1: Prepare data (if not already done)
echo ""
echo "Step 1: Preparing data..."
if [ ! -d "$DATA_DIR/imagesTr" ]; then
    python scripts/prepare_neuroverse3d_data.py \
        --input-dir /nfs/data/nii/data1/Analysis/camaret___in_context_segmentation/ANALYSIS_20251122/results \
        --output-dir $DATA_DIR \
        --dataset-name BraTS \
        --modality MRI
else
    echo "Data already prepared, skipping..."
fi

# Step 2: Stage 1 Training
echo ""
echo "Step 2: Stage 1 Training (fixed context size)..."
python scripts/train_neuroverse3d.py \
    --stage 1 \
    --data-dir $DATA_DIR \
    --context-size 3 \
    --epochs 50 \
    --lr 0.00001 \
    --batch-size 1 \
    --num-gpus 1 \
    --num-workers 4 \
    --checkpoint-dir $CHECKPOINT_DIR/stage1 \
    --project-name $PROJECT_NAME \
    --mixed-precision

echo ""
echo "Stage 1 training complete!"
echo "Best checkpoint saved to: $CHECKPOINT_DIR/stage1/"

# Step 3: Stage 2 Training
echo ""
echo "Step 3: Stage 2 Training (variable context size)..."

# Find the best checkpoint from Stage 1
STAGE1_CHECKPOINT=$(ls -t $CHECKPOINT_DIR/stage1/*.ckpt | grep -v last | head -1)
echo "Using Stage 1 checkpoint: $STAGE1_CHECKPOINT"

python scripts/train_neuroverse3d.py \
    --stage 2 \
    --data-dir $DATA_DIR \
    --checkpoint $STAGE1_CHECKPOINT \
    --epochs 100 \
    --lr 0.000002 \
    --batch-size 1 \
    --num-gpus 1 \
    --num-workers 4 \
    --checkpoint-dir $CHECKPOINT_DIR/stage2 \
    --project-name $PROJECT_NAME \
    --mixed-precision

echo ""
echo "=========================================="
echo "Training Complete!"
echo "=========================================="
echo "Stage 1 checkpoint: $STAGE1_CHECKPOINT"
echo "Stage 2 checkpoints: $CHECKPOINT_DIR/stage2/"
echo ""
echo "You can now use the trained model for inference or evaluation."
