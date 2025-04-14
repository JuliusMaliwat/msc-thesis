import sys
import os

class MVDetrWrapper:
    """
    MVDetr wrapper that directly calls MVDetr main function within the same process.
    """

    def __init__(self, params):
        """
        Initialize MVDetr wrapper.
        
        Args:
            params (dict): Not used (checkpoint and params are fixed).
        """
        # Add MVDetr to Python path
        mvdet_dir = "external/MVDeTr"
        sys.path.append(mvdet_dir)

        # Import main after appending the path
        from main import main as mvdetr_main
        self.mvdetr_main = mvdetr_main

        # Fixed parameters
        self.dataset = "wildtrack"
        self.checkpoint = "aug_deform_trans_lr0.0005_baseR0.1_neck128_out0_alpha1.0_id0_drop0.0_dropcam0.0_worldRK4_10_imgRK12_10_2021-08-02_17-28-02"

    def run(self, dataset):
        """
        Run MVDetr inference directly.

        Args:
            dataset (object): Dataset object (not used, MVDetr loads its own dataset).

        Returns:
            list: Detections in standard format [[frame_id, x, y, score], ...]
        """
        import sys
        from argparse import Namespace

        # Prepare arguments for MVDetr main
        args = Namespace(
            dataset=self.dataset,
            resume=self.checkpoint,
            batch_size=1,
            arch="resnet18",
            world_feat="deform_trans",
            bottleneck_dim=128,
            outfeat_dim=0,
            world_reduce=4,
            world_kernel_size=10,
            img_reduce=12,
            img_kernel_size=10,
            visualize=True
        )

        print("Running MVDetr inference internally...")
        self.mvdetr_main(args)

        print("MVDetr inference completed. Parsing output...")

        # Parse the MVDetr output to standard format
        detections = self._parse_mvdet_output()

        return detections

    def _parse_mvdet_output(self):
        """
        Parse MVDetr output and convert to pipeline standard format.

        Returns:
            list: Detections [[frame_id, x, y], ...]
        """
        output_file = os.path.join("external/MVDeTr/outputs", self.dataset, self.checkpoint, "test.txt")

        if not os.path.exists(output_file):
            raise FileNotFoundError(f"MVDetr output not found at: {output_file}")

        detections = []

        with open(output_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 3:
                    continue  # skip incomplete lines

                frame_id = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                detections.append([frame_id, x, y])

        return detections

