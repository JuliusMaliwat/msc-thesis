import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import argparse
from tracking_framework.utils.config_manager import ConfigManager
from tracking_framework.factory import DatasetFactory, DetectionFactory
from tracking_framework.utils.io import save_bev_txt, save_metrics
from tracking_framework.utils.evaluator import evaluate_detection


def parse_args():
    parser = argparse.ArgumentParser(description="Run detection step")
    parser.add_argument("--experiment_dir", type=str, required=True, help="Path to the experiment directory")
    return parser.parse_args()

def main():
    args = parse_args()
    config = ConfigManager(args.experiment_dir)

    # Build dataset
    dataset = DatasetFactory.build(config.dataset_config()["name"])
    dataset.load()

    # Build detector
    detection_cfg = config.detection_config()
    detector = DetectionFactory.build(detection_cfg["name"], detection_cfg.get("params", {}))

    # Run detection
    print(f"Running detection using {detection_cfg['name']}...")
    detections = detector.run(dataset)

    # Save detections
    save_bev_txt(detections, config.detections_path)
    print(f"Detections saved to: {config.detections_path}")


    # === Evaluation ===
    print("Evaluating detection results...")

    ground_truth = dataset.get_ground_truth(split="test", with_tracking=False)

    metrics = evaluate_detection(detections, ground_truth)
    save_metrics(metrics, config.detection_metrics_output_path)


    print(f"Detection evaluation results saved to: {config.detection_metrics_output_path}")
    print(metrics)  # Optional: print to console too

if __name__ == "__main__":
    main()
