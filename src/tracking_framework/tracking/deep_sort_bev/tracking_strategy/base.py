import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

class DeepSortBaseStrategy:
    def __init__(self, params, tracker_cls):
        self.dist_thresh = params.get("dist_thresh", 0.4)
        self.gating_dist = params.get("gating_dist", 40.0)
        self.tracker_cls = tracker_cls

    def track(self, *, frame_id, trackers, detections, embeddings, dataset, max_age):
        predicted = [trk.predict() for trk in trackers]
        results = []

        unmatched_trackers = list(range(len(trackers)))
        unmatched_detections = list(range(len(detections)))
        matches = []

        if predicted and detections:
            # Motion gating
            dists_pos = cdist(np.array(predicted), np.array(detections))
            gating_mask = dists_pos < self.gating_dist

            # Embedding matching
            feats_det = np.stack([
                embeddings.get((frame_id, x, y), np.zeros(512))
                for (x, y) in detections
            ])
            feats_trk = np.stack([
                trk.get_mean_embedding() for trk in trackers
            ])

            dists_app = cdist(feats_trk, feats_det, metric="cosine")
            dists_app[~gating_mask] = 1.0  # Penalize impossible matches

            trk_idx, det_idx = linear_sum_assignment(dists_app)

            for t, d in zip(trk_idx, det_idx):
                if dists_app[t, d] < self.dist_thresh:
                    matches.append((t, d))

            unmatched_trackers = [i for i in range(len(trackers)) if i not in {t for t, _ in matches}]
            unmatched_detections = [i for i in range(len(detections)) if i not in {d for _, d in matches}]

        updated_trackers = []

        # Update matched
        for t, d in matches:
            trk = trackers[t]
            x, y = detections[d]
            emb = embeddings.get((frame_id, x, y), np.zeros(512))
            trk.update([x, y])
            trk.update_appearance(emb)
            updated_trackers.append(trk)
            results.append([frame_id, x, y, trk.id])

        # New trackers
        for d in unmatched_detections:
            x, y = detections[d]
            emb = embeddings.get((frame_id, x, y), np.zeros(512))
            new_trk = self.tracker_cls([x, y], emb)
            updated_trackers.append(new_trk)
            results.append([frame_id, x, y, new_trk.id])

        # Age out old trackers
        for t in unmatched_trackers:
            trk = trackers[t]
            trk.time_since_update += 1
            if trk.time_since_update < max_age:
                updated_trackers.append(trk)

        return updated_trackers, results
