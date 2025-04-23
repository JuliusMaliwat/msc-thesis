import sys
import os

class MVDetrWrapper:
    """
    MVDeTr wrapper that directly calls MVDeTr main function within the same process.
    """

    def __init__(self, params):
        """
        Initialize MVDeTr wrapper.
        
        Args:
            params (dict): Not used.
        """
        # Add MVDeTr to Python path
        mvdet_dir = "external/MVDeTr"
        sys.path.append(mvdet_dir)

        # Import main after appending the path
        from main import main as mvdetr_main
        self.mvdetr_main = mvdetr_main

    def run(self, dataset):
        """
        Run MVDeTr inference directly.

        Args:
            dataset (object): Dataset object with attribute 'name'.

        Returns:
            list: Detections in standard format [[frame_id, x, y], ...]
        """
        from argparse import Namespace

        # === CONFIG by dataset ===
        if dataset.name == "wildtrack":
            checkpoint = "aug_deform_trans_lr0.0005_baseR0.1_neck128_out0_alpha1.0_id0_drop0.0_dropcam0.0_worldRK4_10_imgRK12_10_2021-08-02_17-28-02"
            args = Namespace(
                dataset="wildtrack",
                resume=checkpoint,
                batch_size=1,
                arch="resnet18",
                world_feat="deform_trans",
                bottleneck_dim=128,
                outfeat_dim=0,
                world_reduce=4,
                world_kernel_size=10,
                img_reduce=12,
                img_kernel_size=10,
                visualize=True,

                reID=False,
                semi_supervised=0,
                id_ratio=0,
                cls_thres=0.6,
                alpha=1.0,
                use_mse=False,
                num_workers=4,
                dropout=0.0,
                dropcam=0.0,
                epochs=10,
                lr=5e-4,
                base_lr_ratio=0.1,
                weight_decay=1e-4,
                seed=2021,
                deterministic=False,
                augmentation=True
            )

        elif dataset.name == "multiviewx":
            checkpoint = "pretrained_multiviewx"
            args = Namespace(
                dataset="multiviewx",
                resume=checkpoint,
                batch_size=1,
                arch="resnet18",
                world_feat="deform_trans",
                bottleneck_dim=128,
                outfeat_dim=0,
                world_reduce=4,
                world_kernel_size=10,
                img_reduce=12,
                img_kernel_size=10,
                visualize=False,

                reID=False,
                semi_supervised=0,
                id_ratio=0,
                cls_thres=0.6,
                alpha=1.0,
                use_mse=False,
                num_workers=4,
                dropout=0.0,
                dropcam=0.0,
                epochs=10,
                lr=5e-4,
                base_lr_ratio=0.1,
                weight_decay=1e-4,
                seed=2021,
                deterministic=False,
                augmentation=True
            )
        else:
            raise ValueError(f"Unsupported dataset: {dataset.name}")

        # === RUN inference ===
        original_cwd = os.getcwd()
        os.chdir("external/MVDeTr")
        try:
            self.mvdetr_main(args)
            print(f"Running MVDeTr inference on {dataset.name}...")
        finally:
            os.chdir(original_cwd)

        print("MVDeTr inference completed. Parsing output...")
        detections = self._parse_mvdet_output(dataset.name, checkpoint)
        return detections

    def _parse_mvdet_output(self, dataset_name, checkpoint):
        """
        Parse MVDeTr output and convert to pipeline standard format.

        Returns:
            list: Detections [[frame_id, x, y], ...]
        """
        output_file = os.path.join("external/MVDeTr/logs", dataset_name, checkpoint, "test.txt")

        if not os.path.exists(output_file):
            raise FileNotFoundError(f"MVDeTr output not found at: {output_file}")

        detections = []
        with open(output_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 3:
                    continue
                frame_id = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                detections.append([frame_id, x, y])

        return detections
