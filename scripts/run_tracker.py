from tracking_framework.utils.config_manager import ConfigManager
from tracking_framework.factory import DatasetFactory, TrackerFactory
from tracking_framework.utils.io import load_bev_txt, save_bev_txt, save_metrics, save_dataframe
from tracking_framework.utils.evaluator import evaluate_tracking

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run tracking step")
    parser.add_argument("--experiment_dir", type=str, required=True, help="Path to the experiment directory")
    return parser.parse_args()

def main():
    args = parse_args()
    config = ConfigManager(args.experiment_dir)

    # Build dataset
    dataset = DatasetFactory.build(config.dataset_config()["name"])
    dataset.load()

    # Build tracker
    tracker_cfg = config.tracking_config()
    tracker = TrackerFactory.build(tracker_cfg["name"], tracker_cfg.get("params", {}))

    # Load detections
    detections = load_bev_txt(config.detections_path)

    # Run tracking
    print(f"Running tracking using {tracker_cfg['name']}...")
    results = tracker.run(detections, dataset)

    # Save results
    save_bev_txt(results, config.tracking_output_path)
    print(f"Tracking results saved to: {config.tracking_output_path}")

    # === Evaluation ===
    print("Evaluating tracking results...")

    ground_truth = dataset.get_ground_truth(split="test", with_tracking=True)

    print("len test ground truth: ", len(ground_truth))
    print("len results: ", len(results))

    metrics, error_log_df = evaluate_tracking(results, ground_truth)
    save_metrics(metrics, config.tracking_metrics_output_path)
    print(f"Tracking evaluation results saved to: {config.tracking_metrics_output_path}")
    print(metrics)  

    save_dataframe(error_log_df, config.tracking_events_output_path)
    print(f"Tracking events saved to: {config.tracking_events_output_path}")

if __name__ == "__main__":
    main()
