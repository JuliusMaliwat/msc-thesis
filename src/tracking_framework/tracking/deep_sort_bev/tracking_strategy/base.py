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

        if not predicted or not detections:
            return trackers, results

        dists_pos = cdist(np.array(predicted), np.array(detections))
        gating_mask = dists_pos < self.gating_dist

        feats_det = np.stack([
            embeddings.get((frame_id, x, y), np.zeros(512))
            for (x, y) in detections
        ])
        feats_trk = np.stack([
            trk.get_mean_embedding() for trk in trackers
        ])

        dists_app = cdist(feats_trk, feats_det, metric="cosine")
        dists_app[~gating_mask] = 1.0  

        trk_idx, det_idx = linear_sum_assignment(dists_app)
        matches = [(t, d) for t, d in zip(trk_idx, det_idx) if dists_app[t, d] < self.dist_thresh]

        matched_trks = {t for t, _ in matches}
        matched_dets = {d for _, d in matches}

        updated_trackers = []

        for t, d in matches:
            trk = trackers[t]
            x, y = detections[d]
            emb = embeddings.get((frame_id, x, y), np.zeros(512))
            trk.update([x, y])
            trk.update_appearance(emb)
            updated_trackers.append(trk)
            results.append([frame_id, x, y, trk.id])

        for d_idx, (x, y) in enumerate(detections):
            if d_idx in matched_dets:
                continue
            emb = embeddings.get((frame_id, x, y), np.zeros(512))
            new_trk = self.tracker_cls([x, y], emb)
            updated_trackers.append(new_trk)
            results.append([frame_id, x, y, new_trk.id])

        for t_idx, trk in enumerate(trackers):
            if t_idx in matched_trks:
                continue
            trk.time_since_update += 1
            if trk.time_since_update < max_age:
                updated_trackers.append(trk)

        return updated_trackers, results
