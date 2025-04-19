#!/bin/bash

set -e


# ======================================
# Parse input arguments
# ======================================

while [[ "$#" -gt 0 ]]; do
  case $1 in
    --config_name) CONFIG_NAME="$2"; shift ;;
    *) echo "Unknown parameter passed: $1"; exit 1 ;;
  esac
  shift
done

if [ -z "$CONFIG_NAME" ]; then
  echo "Usage: bash run_experiment.sh --config_name <config_name>"
  exit 1
fi

# Build the path to the config file
CONFIG_PATH="config/${CONFIG_NAME}.yaml"

# Check if the config file exists
if [ ! -f "$CONFIG_PATH" ]; then
  echo "Error: Config file not found at path: $CONFIG_PATH"
  exit 1
fi

# Generate timestamped experiment directory based on config name
TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
EXPERIMENT_DIR="experiments/${TIMESTAMP}_${CONFIG_NAME}"

# ======================================
# Setup experiment folder
# ======================================

mkdir -p "$EXPERIMENT_DIR"

export PYTHONPATH=$(pwd)/src:$(pwd)/external/MVDeTr


echo "Starting experiment in: $EXPERIMENT_DIR"
echo "Using config file: $CONFIG_PATH"

# Copy the config file to the experiment folder for reproducibility
cp "$CONFIG_PATH" "$EXPERIMENT_DIR/config.yaml"

# Extract detector name from the config file using yq
DETECTOR=$(conda run -n tracking_env yq -r '.detection.name' "$CONFIG_PATH")

echo "Selected detector: $DETECTOR"

# ======================================
# Step 1: Detection
# ======================================


if [ "$DETECTOR" = "mvdetr" ]; then
  echo "Running detection with MVDetr..."
  stdbuf -oL -eL env HOME=$(pwd) conda run -n mvdetr_env python -W ignore scripts/run_detection.py --experiment_dir "$EXPERIMENT_DIR"
elif [ "$DETECTOR" = "another_model" ]; then
  echo "Running detection with Another Model..."
  env conda run -n another_env python -u scripts/run_detection.py --experiment_dir "$EXPERIMENT_DIR"
else
  echo "Error: Detector not recognized: $DETECTOR"
  exit 1
fi


echo "Detection completed. Detections saved in: $EXPERIMENT_DIR/detections.txt"

# ======================================
# Step 2: Tracking
# ======================================

echo "Running tracking..."
env PYTHONWARNINGS="ignore::UserWarning" conda run -n tracking_env python scripts/run_tracker.py --experiment_dir "$EXPERIMENT_DIR"

echo "Tracking completed. Results saved in: $EXPERIMENT_DIR/tracking_output.txt"

# ======================================
# Done
# ======================================

echo "Experiment completed successfully."
echo "All results are saved in: $EXPERIMENT_DIR"
