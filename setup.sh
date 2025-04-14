#!/bin/bash

# Stop if any command fails
set -e

# === CONFIG ===
ENV_NAME="mvdetr"
ENV_FILE="environment_mvdetr.yml"
MVDETR_REPO="https://github.com/hou-yz/MVDeTr.git"
MVDETR_DIR="external/MVDeTr"
MODEL_DIR="external/MVDeTr/logs/wildtrack"
MODEL_CHECKPOINT="aug_deform_trans_lr0.0005_baseR0.1_neck128_out0_alpha1.0_id0_drop0.0_dropcam0.0_worldRK4_10_imgRK12_10_2021-08-02_17-28-02"
MODEL_FILE="MultiviewDetector.pth"
MODEL_GDOWN_ID="10SqNu2JPTNu0ZJKGWvyp2Syqq776ztI9"
DATA_DIR="Data/Wildtrack"
DATA_ZIP="wildtrack.zip"
KAGGLE_USERNAME="juliusmaliwat"
KAGGLE_KEY="6a89e1598210c75a2387475e2695ea50"
KAGGLE_URL="https://www.kaggle.com/api/v1/datasets/download/juliusmaliwat/wildtrack"

# === Step 0: Check dependencies ===
echo "=== Step 0: Check dependencies ==="
if ! command -v conda &> /dev/null; then
  echo "Error: Conda is not installed. Please install Miniconda or Anaconda first."
  exit 1
fi

# === Step 1: Clone MVDeTr repository ===
echo "=== Step 1: Clone MVDeTr repository ==="
if [ ! -d "$MVDETR_DIR" ]; then
  mkdir -p external
  git clone "$MVDETR_REPO" "$MVDETR_DIR"
else
  echo "MVDeTr repository already cloned."
fi

# Ensure MVDeTr is a Python package by adding __init__.py
INIT_FILE="$MVDETR_DIR/__init__.py"
if [ ! -f "$INIT_FILE" ]; then
  touch "$INIT_FILE"
  echo "# Makes MVDeTr a Python package" > "$INIT_FILE"
  echo "__version__ = '0.1'" >> "$INIT_FILE"
  echo "Added __init__.py to MVDeTr."
else
  echo "__init__.py already exists in MVDeTr."
fi


# === Step 2: Create Conda environment ===
echo "=== Step 2: Create Conda environment ==="
if conda env list | grep -q "$ENV_NAME"; then
  echo "Conda environment '$ENV_NAME' already exists."
else
  conda env create -f "$ENV_FILE"
fi

# Activate Conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# === Step 3: Build MVDeTr CUDA extensions ===
echo "=== Step 3: Build MVDeTr CUDA extensions ==="
cd "$MVDETR_DIR/multiview_detector/models/ops"
bash make.sh
cd ../../../../../

# === Step 4: Download Wildtrack dataset via Kaggle ===
echo "=== Step 4: Download Wildtrack dataset ==="
if [ ! -d "$DATA_DIR" ]; then
  mkdir -p Data
  echo "Downloading Wildtrack dataset from Kaggle..."
  curl -L -u $KAGGLE_USERNAME:$KAGGLE_KEY -o "Data/$DATA_ZIP" "$KAGGLE_URL"
  echo "Unzipping Wildtrack dataset..."
  unzip "Data/$DATA_ZIP" -d Data/
  rm "Data/$DATA_ZIP"
  echo "Wildtrack dataset downloaded and extracted."
else
  echo "Wildtrack dataset already exists."
fi

# === Step 5: Download pretrained model ===
echo "=== Step 5: Download pretrained model ==="
MODEL_PATH="$MODEL_DIR/$MODEL_CHECKPOINT/$MODEL_FILE"

if [ ! -f "$MODEL_PATH" ]; then
  mkdir -p "$(dirname "$MODEL_PATH")"
  echo "Downloading pretrained model..."
  gdown "$MODEL_GDOWN_ID" -O "$MODEL_PATH"
else
  echo "Pretrained model already exists at $MODEL_PATH."
fi

echo "=== Setup completed successfully ==="
echo "To start working, activate the environment with:"
echo "conda activate $ENV_NAME"
