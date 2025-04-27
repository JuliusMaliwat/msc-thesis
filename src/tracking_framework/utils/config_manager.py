# src/tracking_framework/utils/config_manager.py

import os
import yaml

class ConfigManager:
    def __init__(self, experiment_dir):
        self.experiment_dir = experiment_dir

        # Standard files in every experiment
        self.config_path = os.path.join(experiment_dir, "config.yaml")
        self.detections_path = os.path.join(experiment_dir, "detections.txt")
        self.tracking_output_path = os.path.join(experiment_dir, "tracking_output.txt")
        self.embeddings_path = os.path.join(experiment_dir, "embeddings.npy")  # if needed
        self.detection_metrics_output_path = os.path.join(experiment_dir, "detection_metrics.yaml")
        self.tracking_metrics_output_path = os.path.join(experiment_dir, "tracking_metrics.yaml")
        self.tracking_events_output_path = os.path.join(experiment_dir, "tracking_events.txt")

        # Optional future extensions
        self.logs_dir = os.path.join(experiment_dir, "logs")
        self.results_dir = os.path.join(experiment_dir, "results")

        # Load YAML immediately
        self.config = self._load_config()

    def _load_config(self):
        if not os.path.isfile(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, "r") as f:
            return yaml.safe_load(f)

    # Shortcut accessors
    def dataset_config(self):
        return self.config.get("dataset", {})

    def detection_config(self):
        return self.config.get("detection", {})

    def tracking_config(self):
        return self.config.get("tracking", {})

    # Optional: helper for safe path creation
    def ensure_dirs(self):
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
