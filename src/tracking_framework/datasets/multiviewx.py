import os
import json
import numpy as np
import cv2
import xml.etree.ElementTree as ET

from tracking_framework.datasets.base_dataset import BaseDataset


class MultiviewXDataset(BaseDataset):
    ORIGIN_X = 0.0
    ORIGIN_Y = 0.0
    GRID_STEP = 0.025  
    BBOX_WIDTH = 0.32  
    BBOX_HEIGHT = 1.8  
    NB_WIDTH = 1000  

    def __init__(self):
        super().__init__()
        self.name = 'multiviewx'
        self.base_dir = "Data/MultiviewX"
        self.images_dir = os.path.join(self.base_dir, "Image_subsets")
        self.annotations_dir = os.path.join(self.base_dir, "annotations_positions")
        self.intrinsics_dir = os.path.join(self.base_dir, "calibrations/intrinsic")
        self.extrinsics_dir = os.path.join(self.base_dir, "calibrations/extrinsic")

        self.rvecs = []
        self.tvecs = []
        self.camera_matrices = []
        self.dist_coeffs = []
        self.frames = []

    def load(self):
        self.rvecs, self.tvecs = self._load_extrinsics()
        self.camera_matrices, self.dist_coeffs = self._load_intrinsics()
        self.frames = self._load_frame_ids()

    def _load_extrinsics(self):
        rvecs, tvecs = [], []
        files = sorted(os.listdir(self.extrinsics_dir))

        for file in files:
            if not file.endswith('.xml'):
                continue

            path = os.path.join(self.extrinsics_dir, file)
            tree = ET.parse(path)
            root = tree.getroot()

            rvec_data = root.find('rvec').find('data').text
            tvec_data = root.find('tvec').find('data').text

            rvec = np.fromstring(rvec_data.strip(), sep=' ')
            tvec = np.fromstring(tvec_data.strip(), sep=' ')

            rvecs.append(rvec)
            tvecs.append(tvec)

        return rvecs, tvecs



    def _load_intrinsics(self):
        camera_matrices, dist_coeffs = [], []
        files = sorted(os.listdir(self.intrinsics_dir))
        for file in files:
            if file.endswith('.xml'):
                path = os.path.join(self.intrinsics_dir, file)
                tree = ET.parse(path)
                root = tree.getroot()
                camera_matrix = np.fromstring(root.find('camera_matrix').find('data').text.strip(), sep=' ').reshape(3, 3)
                dist_coeff = np.fromstring(root.find('distortion_coefficients').find('data').text.strip(), sep=' ')
                camera_matrices.append(camera_matrix)
                dist_coeffs.append(dist_coeff)
        return camera_matrices, dist_coeffs


    def load_image(self, frame_id, cam_id):
        img_path = os.path.join(self.images_dir, f"C{cam_id + 1}", f"{frame_id:04d}.png")
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
        img = cv2.imread(img_path)
        if img is None:
            raise IOError(f"Failed to read image: {img_path}")
        return img

    def project_bev_to_image(self, x_idx, y_idx):
        cx = self.ORIGIN_X + self.GRID_STEP * x_idx
        cy = self.ORIGIN_Y + self.GRID_STEP * y_idx
        cz = 0.0  # ground plane

        results = []
        for cam_id in range(6):
            R, _ = cv2.Rodrigues(self.rvecs[cam_id])
            T = self.tvecs[cam_id].reshape(3, 1)
            camera_matrix = self.camera_matrices[cam_id]
            dist_coeff = self.dist_coeffs[cam_id]

            camera_pos = -R.T @ T  
            dir_vec = np.array([cx, cy, cz]) - camera_pos.flatten()
            dir_vec /= np.linalg.norm(dir_vec)

            ortho = np.array([-dir_vec[1], dir_vec[0]])  

            bottom_center = np.array([cx, cy, cz])
            w = self.BBOX_WIDTH / 2
            h = self.BBOX_HEIGHT

            corners_world = [
                bottom_center + np.array([-ortho[0]*w, -ortho[1]*w, 0]),
                bottom_center + np.array([ ortho[0]*w,  ortho[1]*w, 0]),
                bottom_center + np.array([ ortho[0]*w,  ortho[1]*w, h]),
                bottom_center + np.array([-ortho[0]*w, -ortho[1]*w, h]),
            ]

            corners_2d, _ = cv2.projectPoints(np.array(corners_world), self.rvecs[cam_id], self.tvecs[cam_id], camera_matrix, dist_coeff)
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
        crops = []
        projections = self.project_bev_to_image(x_idx, y_idx)
        for proj in projections:
            cam_id = proj["cam_id"]
            x1, y1, x2, y2 = proj["bbox"]
            try:
                img = self.load_image(frame_id, cam_id)
            except (FileNotFoundError, IOError):
                continue
            h, w = img.shape[:2]

            if x2 <= 0 or x1 >= w or y2 <= 0 or y1 >= h:
                continue  

            x1_clipped = max(0, x1)
            y1_clipped = max(0, y1)
            x2_clipped = min(w, x2)
            y2_clipped = min(h, y2)

            if x2_clipped > x1_clipped and y2_clipped > y1_clipped:
                crop = img[y1_clipped:y2_clipped, x1_clipped:x2_clipped]
                crops.append(crop)

        return crops


    def get_frame_ids(self):
        return self.frames

    def get_train_frame_ids(self):
        return [fid for fid in self.frames if fid <= 359]

    def get_test_frame_ids(self):
        return [fid for fid in self.frames if fid > 359]


    def get_camera_ids(self):
        return list(range(6))

    def get_ground_truth(self, split=None, with_tracking=False):
        ground_truth = []
        valid_frame_ids = None
        if split == "train":
            valid_frame_ids = set(self.get_train_frame_ids())
        elif split == "test":
            valid_frame_ids = set(self.get_test_frame_ids())

        for fid in self.frames:
            if valid_frame_ids is not None and fid not in valid_frame_ids:
                continue
            ann_path = os.path.join(self.annotations_dir, f"{fid:05d}.json")
            if not os.path.exists(ann_path):
                continue
            with open(ann_path, 'r') as f:
                annotations = json.load(f)
            for obj in annotations:
                pos = obj['positionID']
                x_idx, y_idx = self._position_id_to_bev_indices(pos)
                if with_tracking:
                    track_id = obj['personID']
                    ground_truth.append([fid, x_idx, y_idx, track_id])
                else:
                    ground_truth.append([fid, x_idx, y_idx])
        return ground_truth

    def _load_frame_ids(self):
        frame_files = os.listdir(self.annotations_dir)
        frames = []
        for fname in frame_files:
            if fname.endswith('.json'):
                fid = int(fname.replace(".json", ""))
                frames.append(fid)
        return sorted(frames)

    def _position_id_to_bev_indices(self, positionID):
        x_idx = positionID % self.NB_WIDTH
        y_idx = positionID // self.NB_WIDTH
        return int(x_idx), int(y_idx)
