import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from tracking_framework.utils.visibility import compute_frame_visibility_scores

class VisibilitySwitchingStrategy:
    def __init__(self, params, tracker_cls):
        self.dist_thresh = params.get("dist_thresh", 0.4)
        self.gating_dist = params.get("gating_dist", 40.0)
        self.visibility_switching_thresh = params.get("visibility_switching_thresh", 200.0)
        self.tracker_cls = tracker_cls

    def track(self, *, frame_id, trackers, detections, embeddings, dataset, max_age):
        """
        Performs matching and update using visibility-aware cost switching strategy.

        Returns:
            updated_trackers: list of updated trackers
            frame_results: list of results in the format [frame_id, x, y, track_id]
        """

        predicted = [trk.predict() for trk in trackers]
        results = []

        if not predicted or not detections:
            return trackers, results

        vis_scores = compute_frame_visibility_scores(dataset, detections)  # (x, y, cam_id) → score

        dists_pos = cdist(np.array(predicted), np.array(detections))
        gating_mask = dists_pos < self.gating_dist

        feats_det = np.stack([
            embeddings.get((frame_id, x, y), np.zeros(512))
            for (x, y) in detections
        ])
        feats_trk = np.stack([
            trk.get_mean_embedding() for trk in trackers
        ])

        dists_mix = np.ones((len(trackers), len(detections)))

        for t_idx, trk in enumerate(trackers):
            for d_idx, (x, y) in enumerate(detections):
                if not gating_mask[t_idx, d_idx]:
                    continue

                vis = sum(v for (xx, yy, _), v in vis_scores.items() if xx == x and yy == y)

                if vis < self.visibility_switching_thresh:
                    # Use motion cost
                    pred_pos = predicted[t_idx]
                    cost = np.linalg.norm(pred_pos - np.array([x, y])) / self.gating_dist
                else:
                    # Use appearance cost
                    emb_t = feats_trk[t_idx]
                    emb_d = feats_det[d_idx]
                    cosine = 1 - np.dot(emb_t, emb_d) / (np.linalg.norm(emb_t) * np.linalg.norm(emb_d) + 1e-6)
                    cost = cosine / 2

                dists_mix[t_idx, d_idx] = cost

        trk_idx, det_idx = linear_sum_assignment(dists_mix)
        matches = [(t, d) for t, d in zip(trk_idx, det_idx) if dists_mix[t, d] < self.dist_thresh]

        matched_trks = {t for t, _ in matches}
        matched_dets = {d for _, d in matches}

        updated_trackers = []

        for t, d in matches:
            trk = trackers[t]
            x, y = detections[d]
            emb = embeddings.get((frame_id, x, y), np.zeros(512))
            vis = sum(v for (xx, yy, _), v in vis_scores.items() if xx == x and yy == y)

            trk.update([x, y])
            if vis >= self.visibility_switching_thresh:
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
