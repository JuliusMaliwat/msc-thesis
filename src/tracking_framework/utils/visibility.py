import numpy as np
import cv2

def distance_camera_to_bev(dataset, cam_id, cx, cy, cz=0.0):
    R, _ = cv2.Rodrigues(dataset.rvecs[cam_id])
    T = dataset.tvecs[cam_id].reshape(3, 1)
    cam_center = -R.T @ T
    pos = np.array([cx, cy, cz])
    return np.linalg.norm(pos - cam_center.flatten())

def compute_frame_visibility_scores(dataset, detections_in_frame):
    visibility_scores = {}

    coords_world = {
        (x, y): (
            dataset.ORIGIN_X + dataset.GRID_STEP * x,
            dataset.ORIGIN_Y + dataset.GRID_STEP * y
        )
        for (x, y) in detections_in_frame
    }

    projections = {
        (x, y): dataset.project_bev_to_image(x, y)
        for (x, y) in detections_in_frame
    }

    distances = {
        (x, y): np.zeros(dataset.N_CAMERAS, dtype=np.float32)
        for (x, y) in detections_in_frame
    }
    for (x, y), (cx, cy) in coords_world.items():
        for cam_id in range(dataset.N_CAMERAS):
            distances[(x, y)][cam_id] = distance_camera_to_bev(dataset, cam_id, cx, cy)

    for (x, y), ref_projs in projections.items():
        base_pos = (x, y)
        cx, cy = coords_world[base_pos]

        for proj in ref_projs:
            cam_id = proj["cam_id"]
            x1, y1, x2, y2 = proj["bbox"]

            if x2 <= x1 or y2 <= y1 or x1 < 0 or y1 < 0 or x2 > dataset.IMG_WIDTH or y2 > dataset.IMG_HEIGHT:
                continue

            depth_ref = distances[(x, y)][cam_id]
            occ_mask = np.zeros((dataset.IMG_HEIGHT, dataset.IMG_WIDTH), dtype=bool)

            for (ox, oy) in detections_in_frame:
                if (ox, oy) == base_pos:
                    continue
                other_proj = projections.get((ox, oy), [])
                other_proj = [p for p in other_proj if p["cam_id"] == cam_id]
                if not other_proj:
                    continue
                ox1, oy1, ox2, oy2 = other_proj[0]["bbox"]
                if ox2 <= ox1 or oy2 <= oy1:
                    continue
                depth_other = distances[(ox, oy)][cam_id]
                if depth_other >= depth_ref:
                    continue
                occ_mask[oy1:oy2, ox1:ox2] = True

            occ_patch = occ_mask[y1:y2, x1:x2]
            visible = np.sum(~occ_patch)
            visibility_scores[(x, y, cam_id)] = np.sqrt(visible)

    return visibility_scores


