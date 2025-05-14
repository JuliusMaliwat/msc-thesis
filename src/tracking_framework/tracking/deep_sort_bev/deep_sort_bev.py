from tracking_framework.tracking.base_tracker import BaseTracker
from tracking_framework.tracking.deep_sort_bev.deep_kalman import DeepKalmanBoxTracker
from tracking_framework.tracking.deep_sort_bev.embedder import Embedder

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
        self.use_visibility = params.get("use_visibility", False)
        self.visibility_weight = params.get("visibility_weight", 0.4)
        self.vis_score_thresh = params.get("vis_score_thresh", 0.65)
        self.trackers = []

        # Components
        self.embedder = Embedder()

    def run(self, detections, dataset):
        """
        Run DeepSORT tracking algorithm.

        Args:
            detections (list): List of detections [[frame_id, x, y], ...]
            dataset (object): Dataset object with camera intrinsics/extrinsics, images, etc.

        Returns:
            list: Tracking results [[frame_id, x, y, track_id], ...]
        """
        results = []

        # Step 1: Extract crops and compute embeddings
        print("Extracting crops and computing embeddings...")
        embedding_dict = self._compute_embeddings(detections, dataset)

        # Step 2: Prepare detections by frame
        print("Prepare detections by frame...")
        detections_by_frame = _detections_to_frame_dict(detections)


        print("Starting algo...")
        for frame_id in sorted(detections_by_frame.keys()):
            frame_dets = detections_by_frame[frame_id]

            # Predict tracker positions
            predicted = [tracker.predict() for tracker in self.trackers]
            unmatched_trackers = list(range(len(self.trackers)))
            unmatched_detections = list(range(len(frame_dets)))

            matches = [] 
 
            if predicted and frame_dets:
                # Motion gating (Euclidean distance)
                dists_pos = cdist(np.array(predicted), np.array(frame_dets))
                gating_mask = dists_pos < self.gating_dist

                # Appearance matching (Cosine distance)
                feats_det = np.stack([
                    embedding_dict.get((frame_id, x, y), np.zeros(512))
                    for (x, y) in frame_dets
                ])

                feats_trk = np.stack([
                    tracker.get_mean_embedding()
                    for tracker in self.trackers
                ])

                dists_app = cdist(feats_trk, feats_det, metric="cosine")

                if self.use_visibility:
                    print("Visibility!")
                    dataset.load_visibility()
                    for i, (x, y) in enumerate(frame_dets):
                        if 0 <= y < dataset.visibility_map.shape[0] and 0 <= x < dataset.visibility_map.shape[1]:
                            vis_score = dataset.visibility_map[int(y), int(x)]
                            if vis_score < self.vis_score_thresh:
                                dists_app[:, i] += ((1 - vis_score) ** 2) * self.visibility_weight
                        else:
                            print("SOno fuori dalla visiblity")
                dists_app[~gating_mask] = 1.0  # Penalize impossible matches

                trk_idx, det_idx = linear_sum_assignment(dists_app)

                for t, d in zip(trk_idx, det_idx):
                    if dists_app[t, d] < self.dist_thresh:
                        matches.append((t, d))

                # Update unmatched lists
                unmatched_trackers = [i for i in unmatched_trackers if i not in trk_idx]
                unmatched_detections = [i for i in unmatched_detections if i not in det_idx]

            updated_trackers = []

            # Update matched trackers
            for t, d in matches:
                tracker = self.trackers[t]
                tracker.update(frame_dets[d])
                emb = embedding_dict.get((frame_id, frame_dets[d][0], frame_dets[d][1]), np.zeros(512))
                tracker.update_appearance(emb)
                results.append([frame_id, frame_dets[d][0], frame_dets[d][1], tracker.id])
                updated_trackers.append(tracker)

            # Create new trackers for unmatched detections
            for d in unmatched_detections:
                emb = embedding_dict.get((frame_id, frame_dets[d][0], frame_dets[d][1]), np.zeros(512))
                tracker = DeepKalmanBoxTracker(frame_dets[d], emb)
                updated_trackers.append(tracker)
                results.append([frame_id, frame_dets[d][0], frame_dets[d][1], tracker.id])

            # Keep alive unmatched trackers if within max_age
            for t in unmatched_trackers:
                tracker = self.trackers[t]
                tracker.time_since_update += 1
                if tracker.time_since_update < self.max_age:
                    updated_trackers.append(tracker)

            self.trackers = updated_trackers

        return results

    def _compute_embeddings(self, detections, dataset):
        """
        Compute appearance embeddings for all detections, one at a time to save memory.

        Args:
            detections (list): List of detections [[frame_id, x, y], ...]
            dataset (object): Dataset object with images and camera parameters

        Returns:
            dict: Mapping from (frame_id, x, y) to embedding vector
        """
        embedding_dict = {}
        print("Embedding...")
        missing_crops = 0

        for frame_id, x, y in detections:
            crops = dataset.get_crop_from_bev(frame_id, x, y)

            if not crops:
                missing_crops += 1
                embedding = np.zeros(512)
                embedding_dict[(frame_id, x, y)] = embedding
                continue

            crop_embeddings = []
            weights = []

            for crop in crops:
                h, w = crop.shape[:2]
                area = h * w
                try:
                    emb = self.embedder.compute_single(crop)
                except Exception as e:
                    print(f"⚠️ Error processing crop for ({frame_id}, {x}, {y}): {e}")
                    emb = np.zeros(512)
                crop_embeddings.append(emb)
                weights.append(area)

            weights = np.array(weights, dtype=float)
            weights /= weights.sum()
            embedding = np.average(np.vstack(crop_embeddings), axis=0, weights=weights)

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
