# src/tracking_framework/utils/io.py

import yaml
import pandas as pd

def load_bev_txt(file_path):
    """
    Load BEV detections or tracking results from a .txt file.

    Args:
        file_path (str): Path to the input .txt file.

    Returns:
        list: List of detections or tracking results.
              Format: [[frame_id, x, y], ...] or [[frame_id, x, y, track_id], ...]
    """
    data = []

    with open(file_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                frame_id, x, y = map(float, parts)
                data.append([int(frame_id), x, y])
            elif len(parts) == 4:
                frame_id, x, y, track_id = map(float, parts)
                data.append([int(frame_id), x, y, int(track_id)])
            else:
                raise ValueError(f"Unexpected format in line: {line}")

    return data

def save_bev_txt(data, file_path):
    """
    Save BEV detections or tracking results to a .txt file.

    Args:
        data (list): List of detections or tracking results.
                     Format: [[frame_id, x, y], ...] or [[frame_id, x, y, track_id], ...]
        file_path (str): Path to the output .txt file.
    """
    with open(file_path, "w") as f:
        for entry in data:
            line = " ".join(str(int(val)) if isinstance(val, float) and val.is_integer() else str(val) for val in entry)
            f.write(f"{line}\n")


def save_metrics(metrics, output_path):
    """
    Save evaluation metrics to a YAML file.

    Args:
        metrics (dict): Dictionary of evaluation metrics.
        output_path (str): Path to the output YAML file.
    """
    with open(output_path, "w") as f:
        yaml.safe_dump(metrics, f)


def save_dataframe(df, file_path, sep="\t"):
    """
    Save a dataframe to a text file.

    Args:
        df (pd.DataFrame): DataFrame to save.
        file_path (str): Output file path.
        sep (str): Separator (default: tab).
    """
    df.to_csv(file_path, sep=sep, index=False)


