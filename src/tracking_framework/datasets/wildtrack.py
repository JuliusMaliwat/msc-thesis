import os
import xml.etree.ElementTree as ET
import numpy as np
import cv2
import json
from tqdm import tqdm
import numpy as np
from tracking_framework.datasets.base_dataset import BaseDataset

class WildtrackDataset(BaseDataset):
    """
    Wildtrack dataset implementation.
    Handles loading camera parameters, images, and projections.
    """

    # Constants for BEV grid and bbox size
    ORIGIN_X = -3.0
    ORIGIN_Y = -9.0
    GRID_STEP = 0.025

    BBOX_WIDTH = 0.5  # meters
    BBOX_HEIGHT = 1.8  # meters

    NB_WIDTH = 480
    NB_HEIGHT = 1440

    IMG_WIDTH = 1920
    IMG_HEIGHT = 1080


    def __init__(self):

        # Hardcoded dataset paths
        self.name = 'wildtrack'
        base_dir = "Data/Wildtrack"
        self.images_dir = os.path.join(base_dir, "Image_subsets")
        self.annotations_dir = os.path.join(base_dir, "annotations_positions")
        self.intrinsics_dir = os.path.join(base_dir, "calibrations/intrinsic_zero")
        self.extrinsics_dir = os.path.join(base_dir, "calibrations/extrinsic")
        self.rectangles_path = os.path.join(base_dir, "rectangles.pom")
        self.visibility_map = os.path.join(base_dir, "visibility_map.npy")
        # Loaded resources (to be populated by load())
        self.rvecs = []
        self.tvecs = []
        self.camera_matrices = []
        self.dist_coeffs = []
        self.frames = []
        self.visibility_map = None


    def load(self):
        """
        Load intrinsics, extrinsics, frame IDs, and visibility map.
        """
        self.rvecs, self.tvecs = self._load_extrinsics(self.extrinsics_dir)
        self.camera_matrices, self.dist_coeffs = self._load_intrinsics(self.intrinsics_dir)
        self.frames = self._load_frame_ids()
        self.visibility_map = self._load_visibility_map()


    def load_image(self, frame_id, cam_id):
        """
        Load an image for a given frame and camera.

        Args:
            frame_id (int): Frame ID
            cam_id (int): Camera ID

        Returns:
            np.ndarray: Image array
        """
        img_path = os.path.join(self.images_dir, f"C{cam_id + 1}", f"{frame_id:08}.png")
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")

        img = cv2.imread(img_path)
        if img is None:
            raise IOError(f"Failed to read image: {img_path}")

        return img

    def project_bev_to_image(self, x_idx, y_idx, cam_id = None):
        """
        Project BEV grid coordinate to 2D bounding boxes in all cameras.

        Args:
            x_idx (int): X index in the BEV grid
            y_idx (int): Y index in the BEV grid

        Returns:
            list of dict: Per-camera projected bounding boxes and corners
        """
        cx = self.ORIGIN_X + self.GRID_STEP * x_idx
        cy = self.ORIGIN_Y + self.GRID_STEP * y_idx
        cz = 0.0  # Ground plane

        results = []
        cameras_to_process = range(len(self.camera_matrices)) if cam_id is None else [cam_id]
        for cam_id in cameras_to_process:
            R, _ = cv2.Rodrigues(self.rvecs[cam_id])
            T = self.tvecs[cam_id].reshape(3, 1)
            camera_matrix = self.camera_matrices[cam_id]
            dist_coeff = self.dist_coeffs[cam_id]

            # Compute direction and orthogonal vector
            camera_pos = -R.T @ T
            dir_vec = np.array([cx, cy, cz]) - camera_pos.flatten()
            dir_vec /= np.linalg.norm(dir_vec)
            ortho = np.array([-dir_vec[1], dir_vec[0]])

            # Define 3D bounding box corners
            bottom_center = np.array([cx, cy, cz])
            w = self.BBOX_WIDTH / 2
            h = self.BBOX_HEIGHT

            corners_world = [
                bottom_center + np.array([-ortho[0]*w, -ortho[1]*w, 0]),
                bottom_center + np.array([ ortho[0]*w,  ortho[1]*w, 0]),
                bottom_center + np.array([ ortho[0]*w,  ortho[1]*w, h]),
                bottom_center + np.array([-ortho[0]*w, -ortho[1]*w, h])
            ]

            # Project to image
            corners_2d, _ = cv2.projectPoints(
                np.array(corners_world),
                self.rvecs[cam_id],
                self.tvecs[cam_id],
                camera_matrix,
                dist_coeff
            )
            corners_2d = corners_2d.reshape(-1, 2)

            x_min, y_min = corners_2d.min(axis=0)
            x_max, y_max = corners_2d.max(axis=0)

            bbox = [int(x_min), int(y_min), int(x_max), int(y_max)]

            results.append({
                "cam_id": cam_id,
                "bbox": bbox,
                "corners": corners_2d
            })

        return results

    def get_crop_from_bev(self, frame_id, x_idx, y_idx):
        """
        Get cropped images for a given BEV detection.

        Args:
            frame_id (int): Frame ID
            x_idx (int): X index in the BEV grid
            y_idx (int): Y index in the BEV grid

        Returns:
            list: List of cropped image arrays, one per camera
        """
        crops = []

        projections = self.project_bev_to_image(x_idx, y_idx)

        for proj in projections:
            cam_id = proj["cam_id"]
            bbox = proj["bbox"]
            x1, y1, x2, y2 = bbox

            try:
                img = self.load_image(frame_id, cam_id)
            except (FileNotFoundError, IOError):
                continue

            h, w = img.shape[:2]
            if x1 < 0 or y1 < 0 or x2 > w or y2 > h or x2 <= x1 or y2 <= y1:
                continue

            crop = img[y1:y2, x1:x2]
            crops.append(crop)

        return crops

    def get_frame_ids(self):
        """
        Get list of available frame IDs in the dataset.

        Returns:
            list of int: List of frame IDs
        """
        return self.frames
    def get_train_frame_ids(self):
        """
        Get list of frame IDs for training split.

        Returns:
            list of int: List of frame IDs used for training.
        """
        return [frame_id for frame_id in self.frames if frame_id <= 1799]

    def get_test_frame_ids(self):
        """
        Get list of frame IDs for test split.

        Returns:
            list of int: List of frame IDs used for evaluation/test.
        """
        # Wildtrack ufficiale: test set dai frame 1800 in avanti
        return [frame_id for frame_id in self.frames if frame_id >= 1800]

    def get_camera_ids(self):
        """
        Get list of available camera IDs in the dataset.

        Returns:
            list of int: List of camera IDs
        """
        # Wildtrack ha 7 camere numerate da 0 a 6
        return list(range(7))

    def get_ground_truth(self, split=None, with_tracking=False):
        """
        Extract ground truth detections or tracking annotations from WildTrack dataset.

        Args:
            split (str or None): 'train', 'test', or None for all data.
            with_tracking (bool): If True, includes track IDs (personID).

        Returns:
            list: Ground truth list:
                - if with_tracking=False: [[frame_id, x_idx, y_idx], ...]
                - if with_tracking=True: [[frame_id, x_idx, y_idx, track_id], ...]
        """
        annotations_dir = self.annotations_dir
        ground_truth = []

        # Determine valid frame IDs based on split
        valid_frame_ids = None
        if split == "train":
            valid_frame_ids = set(self.get_train_frame_ids())
        elif split == "test":
            valid_frame_ids = set(self.get_test_frame_ids())

        # Process annotation files
        for filename in sorted(os.listdir(annotations_dir)):
            if not filename.endswith('.json'):
                continue

            frame_id = int(filename.replace(".json", ""))

            if valid_frame_ids is not None and frame_id not in valid_frame_ids:
                continue

            json_path = os.path.join(annotations_dir, filename)
            with open(json_path, "r") as f:
                annotations = json.load(f)

            for person in annotations:
                positionID = person["positionID"]
                x_idx, y_idx = self._position_id_to_bev_indices(positionID)

                if with_tracking:
                    track_id = person["personID"]
                    ground_truth.append([frame_id, x_idx, y_idx, track_id])
                else:
                    ground_truth.append([frame_id, x_idx, y_idx])
        print(f"Loading {len(ground_truth)} for split {split}")
        return ground_truth



    # =========================
    # Private helper functions
    # =========================

    @staticmethod
    def _load_extrinsics(extrinsics_dir):
        rvecs, tvecs = [], []

        files = sorted(os.listdir(extrinsics_dir))
        for file in files:
            if file.endswith('.xml'):
                tree = ET.parse(os.path.join(extrinsics_dir, file))
                root = tree.getroot()

                rvec_text = root.find('rvec').text.strip()
                tvec_text = root.find('tvec').text.strip()

                rvec = np.fromstring(rvec_text, sep=' ')
                tvec = np.fromstring(tvec_text, sep=' ') / 100.0  # Convert to meters

                rvecs.append(rvec)
                tvecs.append(tvec)

        return rvecs, tvecs

    @staticmethod
    def _load_intrinsics(intrinsics_dir):
        camera_matrices, dist_coeffs = [], []

        files = sorted(os.listdir(intrinsics_dir))
        for file in files:
            if file.endswith('.xml'):
                tree = ET.parse(os.path.join(intrinsics_dir, file))
                root = tree.getroot()

                camera_data = root.find('camera_matrix').find('data').text.strip()
                camera_matrix = np.fromstring(camera_data, sep=' ').reshape((3, 3))

                dist_data = root.find('distortion_coefficients').find('data').text.strip()
                dist_coeff = np.fromstring(dist_data, sep=' ')

                camera_matrices.append(camera_matrix)
                dist_coeffs.append(dist_coeff)

        return camera_matrices, dist_coeffs

    def _load_frame_ids(self):
        frames = set()
        cam_dirs = [os.path.join(self.images_dir, d) for d in os.listdir(self.images_dir) if os.path.isdir(os.path.join(self.images_dir, d))]

        for cam_dir in cam_dirs:
            for f in os.listdir(cam_dir):
                if f.endswith(".png"):
                    frame_num = int(f.replace(".png", ""))
                    frames.add(frame_num)

        return sorted(list(frames))

    def _position_id_to_bev_indices(self, positionID):
        x_idx = positionID % self.NB_WIDTH
        y_idx = positionID // self.NB_WIDTH
        return int(x_idx), int(y_idx)



    def _load_visibility_map(self):
        """
        Load a weighted visibility map from rectangles.pom using projected bbox area as quality indicator.
        Returns:
            np.ndarray: [NB_HEIGHT, NB_WIDTH] map with weighted visibility scores
        """
        if os.path.exists(self.visibility_map):
            return np.load(self.visibility_map)
        
        print("First time visibility")
        visibility_map = np.zeros((self.NB_HEIGHT, self.NB_WIDTH), dtype=float)

        # Wildtrack camera resolution
        img_w, img_h = 1920, 1080

        with open(self.rectangles_path, "r") as f:
            for line in tqdm(f, desc="Parsing rectangles.pom"):
                if not line.startswith("RECTANGLE"):
                    continue

                parts = line.strip().split()
                if len(parts) < 3:
                    continue

                cam_id = int(parts[1])
                pos_id = int(parts[2])
                visible = "notvisible" not in line

                if not visible:
                    continue

                x_idx, y_idx = self._position_id_to_bev_indices(pos_id)
                if not (0 <= x_idx < self.NB_WIDTH and 0 <= y_idx < self.NB_HEIGHT):
                    continue

                # Proietta solo per la camera specifica
                projection = self.project_bev_to_image(x_idx, y_idx, cam_id=cam_id)
                if not projection:
                    continue

                proj = projection[0]  # Singola proiezione
                x1, y1, x2, y2 = proj["bbox"]

                # Scarta se bbox è fuori immagine o malformato
                if x1 < 0 or y1 < 0 or x2 > img_w or y2 > img_h or x2 <= x1 or y2 <= y1:
                    area = 0
                else:
                    area = (x2 - x1) * (y2 - y1)


                visibility_map[y_idx, x_idx] +=  np.log1p(area)

        visibility_map /= np.max(visibility_map)


        return visibility_map





