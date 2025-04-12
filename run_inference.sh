#!/bin/bash

# Stop if any command fails
set -e

# === CONFIG ===
ENV_NAME="mvdetr"
CHECKPOINT="aug_deform_trans_lr0.0005_baseR0.1_neck128_out0_alpha1.0_id0_drop0.0_dropcam0.0_worldRK4_10_imgRK12_10_2021-08-02_17-28-02"
MVDETR_DIR="external/MVDeTr"
DATASET="wildtrack"

# === Step 1: Activate environment ===
echo "Activating conda environment '$ENV_NAME'..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# === Step 2: Run inference ===
echo "Running inference with MVDeTr..."
cd "$MVDETR_DIR"

python main.py \
  --dataset $DATASET \
  --resume $CHECKPOINT \
  --batch_size 1 \
  --arch resnet18 \
  --world_feat deform_trans \
  --bottleneck_dim 128 \
  --outfeat_dim 0 \
  --world_reduce 4 \
  --world_kernel_size 10 \
  --img_reduce 12 \
  --img_kernel_size 10 \
  --visualize

cd ../../

echo "Inference completed."
