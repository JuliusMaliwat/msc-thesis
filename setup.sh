#!/bin/bash

# Stop the script if any command fails
set -e

# === CONFIGURATION ===
ENV_NAME="mvdetr"
ENV_FILE="environment_mvdetr.yml"
MVDETR_REPO="https://github.com/hou-yz/MVDeTr.git"
MVDETR_DIR="external/MVDeTr"
MODEL_DIR="models/wildtrack"
MODEL_FILE="MultiviewDetector.pth"
MODEL_GDOWN_ID="10SqNu2JPTNu0ZJKGWvyp2Syqq776ztI9"
DATA_DIR="Data/Wildtrack"
DATA_ZIP="wildtrack.zip"
DATA_GDOWN_ID="1pdCvSOqtEtFPi0P_16GX2YeliYg5OtTS"

# === Step 0: Check for required commands ===
echo "=== Step 0: Check dependencies ==="
if ! command -v conda &> /dev/null; then
  echo "Error: Conda is not installed. Please install Miniconda or Anaconda first."
  exit 1
fi

if ! command -v gdown &> /dev/null; then
  echo "Error: gdown is not installed. Installing..."
  pip install gdown
fi

# === Step 1: Clone MVDeTr repository ===
echo "=== Step 1: Clone MVDeTr repository ==="
if [ ! -d "$MVDETR_DIR" ]; then
  mkdir -p external
  git clone "$MVDETR_REPO" "$MVDETR_DIR"
else
  echo "MVDeTr repository already cloned."
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
cd "$MVDETR_DIR"
bash make.sh
cd ../../

# === Step 4: Download Wildtrack dataset ===
echo "=== Step 4: Download Wildtrack dataset ==="
if [ ! -d "$DATA_DIR" ]; then
  mkdir -p Data
  echo "Downloading Wildtrack dataset..."
  gdown "$DATA_GDOWN_ID" -O "Data/$DATA_ZIP"
  echo "Unzipping Wildtrack dataset..."
  unzip "Data/$DATA_ZIP" -d Data/
  rm "Data/$DATA_ZIP"
  echo "Wildtrack dataset downloaded and extracted."
else
  echo "Wildtrack dataset already exists."
fi

# === Step 5: Download pretrained model ===
echo "=== Step 5: Download pretrained model ==="
if [ ! -f "$MODEL_DIR/$MODEL_FILE" ]; then
  mkdir -p "$MODEL_DIR"
  gdown "$MODEL_GDOWN_ID" -O "$MODEL_DIR/$MODEL_FILE"
else
  echo "Pretrained model already exists."
fi

echo "=== Setup completed successfully ==="
echo "To start working, activate the environment with:"
echo "conda activate $ENV_NAME"
