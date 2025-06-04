import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from tracking_framework.utils.visibility import compute_frame_visibility_scores


class VisibilitySwitchingStrategy:
    def __init__(self, params, tracker_cls):
        # Single matching threshold: used for both appearance and motion costs
        self.dist_thresh = params.get("dist_thresh", 0.4)
        # Gating distance (raw Euclidean) to pre‐filter impossible pairs
        self.gating_dist = params.get("gating_dist", 40.0)
        # Visibility threshold to decide between appearance vs. motion cost
        self.visibility_switching_thresh = params.get("visibility_switching_thresh", 200.0)
        self.tracker_cls = tracker_cls

    def track(self, *, frame_id, trackers, detections, embeddings, dataset, max_age):
        """
        Visibility‐aware matching:
          - If a detection’s visibility ≥ threshold, cost = cosine distance.
          - Otherwise, cost = normalized motion distance (Euclidean / gating_dist).
          - Any pair with raw distance ≥ gating_dist is excluded (cost = 1.0).

        Returns:
          updated_trackers: list of tracker instances after update
          results: list of [frame_id, x, y, track_id]
        """
        # 1) Predict new positions for existing trackers
        predicted = [trk.predict() for trk in trackers]
        results = []

        num_trk = len(trackers)
        num_det = len(detections)
        unmatched_trackers = list(range(num_trk))
        unmatched_detections = list(range(num_det))
        matches = []

        # 2) Compute visibility mask for all detections (always initialize)
        vis_scores = compute_frame_visibility_scores(dataset, detections)
        vis_mask = [
            any(
                v >= self.visibility_switching_thresh
                for (xx, yy, _), v in vis_scores.items()
                if xx == x and yy == y
            )
            for (x, y) in detections
        ]  # length = num_det

        # 3) Only proceed with matching if we have both predictions and detections
        if predicted and detections:
            # 3.1) Motion gating: raw Euclidean distances
            preds = np.array(predicted)            # shape: (num_trk, 2)
            dets = np.array(detections)            # shape: (num_det, 2)
            dists_pos = cdist(preds, dets)         # (num_trk, num_det)
            gating_mask = dists_pos < self.gating_dist

            # 3.2) Appearance embeddings and cosine distances
            feats_trk = np.stack([trk.get_mean_embedding() for trk in trackers])  # (num_trk, 512)
            feats_det = np.stack([
                embeddings.get((frame_id, x, y), np.zeros(512))
                for (x, y) in detections
            ])  # (num_det, 512)
            dists_app = cdist(feats_trk, feats_det, metric="cosine")              # (num_trk, num_det)

            # 3.3) Build mixed cost matrix
            # Normalize motion by gating_dist → values in [0,1) when gated
            motion_norm = dists_pos / self.gating_dist

            # Initialize all costs to 1.0 (≥ dist_thresh, so excluded)
            cost_mix = np.ones((num_trk, num_det), dtype=float)

            # Fill only where gating_mask is True
            gated_idxs = np.where(gating_mask)
            for t, d in zip(*gated_idxs):
                if vis_mask[d]:
                    cost_mix[t, d] = dists_app[t, d]
                else:
                    cost_mix[t, d] = motion_norm[t, d]

            # 3.4) Before Hungarian, replace any NaN/inf in cost_mix with 1.0
            cost_mix = np.nan_to_num(cost_mix, nan=1.0, posinf=1.0, neginf=1.0)

            # 3.5) Hungarian matching on cost_mix
            trk_idx, det_idx = linear_sum_assignment(cost_mix)
            for t, d in zip(trk_idx, det_idx):
                if cost_mix[t, d] < self.dist_thresh:
                    matches.append((t, d))

            matched_trks = {t for t, _ in matches}
            matched_dets = {d for _, d in matches}

            # 3.6) Determine unmatched lists
            unmatched_trackers = [i for i in range(num_trk) if i not in matched_trks]
            unmatched_detections = [j for j in range(num_det) if j not in matched_dets]

        # 4) Update matched trackers
        updated_trackers = []
        for t, d in matches:
            trk = trackers[t]
            x, y = detections[d]
            emb = embeddings.get((frame_id, x, y), np.zeros(512))
            # Always update position
            trk.update([x, y])
            # Update appearance only if detection was visible
            if vis_mask[d]:
                trk.update_appearance(emb)
            updated_trackers.append(trk)
            results.append([frame_id, x, y, trk.id])

        # 5) Create new trackers for unmatched detections
        for d in unmatched_detections:
            x, y = detections[d]
            # Only assign an embedding if detection is visible
            emb = embeddings.get((frame_id, x, y), np.zeros(512)) if vis_mask[d] else None
            new_trk = self.tracker_cls([x, y], emb)
            updated_trackers.append(new_trk)
            results.append([frame_id, x, y, new_trk.id])

        # 6) Age out unmatched trackers
        for t in unmatched_trackers:
            trk = trackers[t]
            trk.time_since_update += 1
            if trk.time_since_update < max_age:
                updated_trackers.append(trk)

        return updated_trackers, results
