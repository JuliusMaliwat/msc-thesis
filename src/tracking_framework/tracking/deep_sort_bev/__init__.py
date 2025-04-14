import os
import cv2
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import shutil

from tracking.utils.projection import Projector
from tracking.utils.io import load_bev_txt

def create_crops(detections_path, images_dir, intrinsics_dir, extrinsics_dir, output_dir, metadata_output_path):
    """
    Generate cropped images and metadata from BEV detections.

    Args:
        detections_path (str): Path to the BEV detections txt file.
        images_dir (str): Path to the original images.
        intrinsics_dir (str): Path to the camera intrinsics.
        extrinsics_dir (str): Path to the camera extrinsics.
        output_dir (str): Path to save cropped images.
        metadata_output_path (str): Path to save metadata CSV.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load BEV detections
    detections_dict = load_bev_txt(detections_path)

    # Initialize projector
    projector = Projector(intrinsics_dir, extrinsics_dir)

    crop_metadata = []

    print("✅ Starting crop generation...")
    for frame_id in tqdm(sorted(detections_dict.keys()), desc="Frames"):
        for x_idx, y_idx in detections_dict[frame_id]:
            projections = projector.project_bev_detection_to_cameras(x_idx, y_idx)

            for proj in projections:
                cam_id = proj["cam_id"]
                bbox = proj["bbox"]
                x1, y1, x2, y2 = bbox

                # Path to original image
                img_path = os.path.join(images_dir, f"C{cam_id + 1}", f"{frame_id:08}.png")
                if not os.path.exists(img_path):
                    continue

                img = cv2.imread(img_path)
                if img is None:
                    continue

                # Validate bbox dimensions
                h, w = img.shape[:2]
                if x1 < 0 or y1 < 0 or x2 > w or y2 > h or x2 <= x1 or y2 <= y1:
                    continue

                # Crop and save
                crop = img[y1:y2, x1:x2]
                crop_filename = f"{frame_id:08}_cam{cam_id + 1}_{x_idx}_{y_idx}.jpg"
                crop_path = os.path.join(output_dir, crop_filename)

                cv2.imwrite(crop_path, crop)

                # Append metadata
                crop_metadata.append({
                    "frame": frame_id,
                    "x": x_idx,
                    "y": y_idx,
                    "cam_id": cam_id,
                    "crop_path": crop_path
                })

    # Save metadata CSV
    df_metadata = pd.DataFrame(crop_metadata)
    df_metadata.to_csv(metadata_output_path, index=False)

    # Optionally, zip the crops directory
    shutil.make_archive(output_dir, 'zip', output_dir)

    print(f"✅ Crops saved to {output_dir}/")
    print(f"✅ Metadata saved to {metadata_output_path}")
    print(f"✅ Archive created: {output_dir}.zip")
