from tracking_framework.tracking.base_tracker import BaseTracker
from tracking_framework.tracking.deep_sort_bev.embedder import Embedder
from tracking_framework.utils.visibility import compute_frame_visibility_scores
from tracking_framework.tracking.deep_sort_bev.tracking_strategy.factory import build_tracking_strategy


from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import numpy as np

class DeepSortBEVTracker(BaseTracker):
    """
    DeepSORT BEV tracker with motion gating and appearance matching.
    """


    def __init__(self, params):
        super().__init__(params)

        self.max_age = params.get("max_age", 4)
        self.use_weighted_embedding = params.get("use_weighted_embedding", False)
        
        self.embedder = Embedder()
        self.strategy = build_tracking_strategy(params)
        self.trackers = []

    def run(self, detections, dataset):
        results = []
        # Precompute all embeddings for detections
        embedding_dict = self._compute_embeddings(detections, dataset)
        detections_by_frame = _detections_to_frame_dict(detections)

        for frame_id in sorted(detections_by_frame.keys()):
            frame_dets = detections_by_frame[frame_id]

            # Strategy-specific step: handles prediction, association, update
            self.trackers, frame_results = self.strategy.track(
                frame_id=frame_id,
                trackers=self.trackers,
                detections=frame_dets,
                embeddings=embedding_dict,
                dataset=dataset,
                max_age=self.max_age
            )

            results.extend(frame_results)

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
