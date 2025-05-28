from tracking_framework.tracking.base_tracker import BaseTracker
from tracking_framework.tracking.deep_sort_bev.deep_kalman import DeepKalmanBoxTracker
from tracking_framework.tracking.deep_sort_bev.embedder import Embedder
from tracking_framework.utils.visibility import compute_frame_visibility_scores

from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import numpy as np

class DeepSortBEVTracker(BaseTracker):
    """
    DeepSORT BEV tracker with motion gating and appearance matching.
    """

    def __init__(self, params):
        super().__init__(params)

        self.dist_thresh = params.get("dist_thresh", 0.4)
        self.max_age = params.get("max_age", 4)
        self.gating_dist = params.get("gating_dist", 40.0)

        # tentativo weighted embeddings
        self.use_weighted_embedding = params.get("use_weighted_embedding", False)

        # tentativo 
        self.use_visibility_switching = params.get("use_visibility_switching", False)
        self.visibility_switching_thresh = params.get("visibility_switching_thresh", 300.0)

        self.trackers = []

        # Components
        self.embedder = Embedder()

    def run(self, detections, dataset):
        """
        Runs the DeepSORT-BEV tracking algorithm with visibility-aware switching between motion and appearance cost.

        Args:
            detections (list): List of detections formatted as [frame_id, x, y].
            dataset (object): Dataset interface providing images, intrinsics/extrinsics, and crops.

        Returns:
            list: Tracking results [[frame_id, x, y, track_id], ...]
        """
        results = []
        matching_log = []  # [frame_id, track_id, det_idx, metric_used]

        embedding_dict = self._compute_embeddings(detections, dataset)
        detections_by_frame = _detections_to_frame_dict(detections)

        for frame_id in sorted(detections_by_frame.keys()):
            frame_dets = detections_by_frame[frame_id]
            vis_scores = compute_frame_visibility_scores(dataset, [(x, y) for (x, y) in frame_dets])

            predicted = [tracker.predict() for tracker in self.trackers]
            unmatched_trackers = list(range(len(self.trackers)))
            unmatched_detections = list(range(len(frame_dets)))
            matches = []

            if predicted and frame_dets:
                dists_pos = cdist(np.array(predicted), np.array(frame_dets))
                gating_mask = dists_pos < self.gating_dist

                feats_det = np.stack([
                    embedding_dict.get((frame_id, x, y), np.zeros(512))
                    for (x, y) in frame_dets
                ])
                feats_trk = np.stack([
                    tracker.get_mean_embedding()
                    for tracker in self.trackers
                ])

                dists_app = np.ones((len(feats_trk), len(feats_det)))

                for t_idx, tracker in enumerate(self.trackers):
                    for d_idx, (x, y) in enumerate(frame_dets):
                        if not gating_mask[t_idx, d_idx]:
                            continue

                        vis = sum(v for (xx, yy, _), v in vis_scores.items() if xx == x and yy == y)

                        if self.use_visibility_switching and vis < self.visibility_switching_thresh:
                            pred_pos = predicted[t_idx]
                            det_pos = np.array([x, y])
                            cost = np.linalg.norm(pred_pos - det_pos) / self.gating_dist
                            dists_app[t_idx, d_idx] = cost
                        else:
                            emb_t = feats_trk[t_idx]
                            emb_d = feats_det[d_idx]
                            cosine = 1 - np.dot(emb_t, emb_d) / (np.linalg.norm(emb_t) * np.linalg.norm(emb_d) + 1e-6)
                            cost = cosine / 2
                            dists_app[t_idx, d_idx] = cost

                trk_idx, det_idx = linear_sum_assignment(dists_app)

                for t, d in zip(trk_idx, det_idx):
                    if dists_app[t, d] < self.dist_thresh:
                        matches.append((t, d))

                unmatched_trackers = [i for i in unmatched_trackers if i not in trk_idx]
                unmatched_detections = [i for i in unmatched_detections if i not in det_idx]

            updated_trackers = []

            for t, d in matches:
                tracker = self.trackers[t]
                tracker.update(frame_dets[d])
                x, y = frame_dets[d]
                emb = embedding_dict.get((frame_id, x, y), np.zeros(512))

                metric_used = "appearance"
                if self.use_visibility_switching:
                    vis = sum(v for (xx, yy, _), v in vis_scores.items() if xx == x and yy == y)
                    if vis < self.visibility_switching_thresh:
                        metric_used = "motion"
                    else:
                        tracker.update_appearance(emb)
                else:
                    tracker.update_appearance(emb)

                results.append([frame_id, x, y, tracker.id])
                matching_log.append([frame_id, tracker.id, d, metric_used])
                updated_trackers.append(tracker)

            for d in unmatched_detections:
                x, y = frame_dets[d]
                emb = embedding_dict.get((frame_id, x, y), np.zeros(512))
                tracker = DeepKalmanBoxTracker([x, y], emb)
                updated_trackers.append(tracker)
                results.append([frame_id, x, y, tracker.id])

            for t in unmatched_trackers:
                tracker = self.trackers[t]
                tracker.time_since_update += 1
                if tracker.time_since_update < self.max_age:
                    updated_trackers.append(tracker)

            self.trackers = updated_trackers

        return results



    def _compute_embeddings(self, detections, dataset):
        embedding_dict = {}
        print("Embedding...")
        missing_crops = 0

        detections_by_frame = _detections_to_frame_dict(detections)

        for frame_id, dets in detections_by_frame.items():
            vis_scores = compute_frame_visibility_scores(dataset, dets)  # returns (x, y, cam_id) → vis

            for x, y in dets:
                crops = dataset.get_crop_from_bev(frame_id, x, y)  # list of {"image": ..., "cam_id": ...}

                if not crops:
                    missing_crops += 1
                    embedding_dict[(frame_id, x, y)] = np.zeros(512)
                    continue

                crop_embeddings = []
                weights = []

                for crop in crops:
                    img = crop["image"]
                    cam_id = crop["cam_id"]

                    try:
                        emb = self.embedder.compute_single(img)
                    except Exception as e:
                        print(f"⚠️ Error processing crop for ({frame_id}, {x}, {y}, cam {cam_id}): {e}")
                        emb = np.zeros(512)

                    crop_embeddings.append(emb)

                    if self.use_weighted_embedding:
                        weight = vis_scores.get((x, y, cam_id), 0.0)
                        weights.append(weight)

                if self.use_weighted_embedding and weights:
                    weights = np.array(weights, dtype=float)
                    if weights.sum() > 0:
                        weights /= weights.sum()
                        embedding = np.average(np.vstack(crop_embeddings), axis=0, weights=weights)
                    else:
                        embedding = np.mean(crop_embeddings, axis=0)
                else:
                    embedding = np.mean(crop_embeddings, axis=0)

                embedding_dict[(frame_id, x, y)] = embedding

        print(f"Detections senza crops: {missing_crops}/{len(detections)}")
        return embedding_dict




# ========================
# Private helper functions
# ========================

def _detections_to_frame_dict(detections):
    """
    Organize detections into a dictionary by frame.

    Args:
        detections (list): List of detections [[frame_id, x, y], ...]

    Returns:
        dict: Mapping from frame_id to list of detections
    """
    frame_dict = {}
    for det in detections:
        frame_id = det[0]
        if frame_id not in frame_dict:
            frame_dict[frame_id] = []
        frame_dict[frame_id].append(det[1:])
    return frame_dict
