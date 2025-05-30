import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from tracking_framework.utils.visibility import compute_frame_visibility_scores


class TwoStageMatchingStrategy:
    def __init__(self, params, tracker_cls):
        self.dist_thresh = params.get("dist_thresh", 0.4)
        self.gating_dist = params.get("gating_dist", 40.0)
        self.visibility_switching_thresh = params.get("visibility_switching_thresh", 200.0)
        self.tracker_cls = tracker_cls

    def track(self, *, frame_id, trackers, detections, embeddings, dataset, max_age):
        """
        Performs two-stage matching: first by appearance (visible), then by motion (remaining).

        Returns:
            updated_trackers: list of updated trackers
            frame_results: list of [frame_id, x, y, track_id]
        """

        predicted = [trk.predict() for trk in trackers]
        results = []

        if not predicted or not detections:
            return trackers, results

        vis_scores = compute_frame_visibility_scores(dataset, detections)

        dists_pos = cdist(np.array(predicted), np.array(detections))
        gating_mask = dists_pos < self.gating_dist

        feats_det = np.stack([
            embeddings.get((frame_id, x, y), np.zeros(512))
            for (x, y) in detections
        ])
        feats_trk = np.stack([
            trk.get_mean_embedding() for trk in trackers
        ])

        num_trk, num_det = len(trackers), len(detections)
        stage1_cost = np.ones((num_trk, num_det))  # Appearance cost
        stage2_cost = np.ones((num_trk, num_det))  # Motion cost

        for t_idx, trk in enumerate(trackers):
            for d_idx, (x, y) in enumerate(detections):
                if not gating_mask[t_idx, d_idx]:
                    continue

                vis = sum(v for (xx, yy, _), v in vis_scores.items() if xx == x and yy == y)

                emb_t = feats_trk[t_idx]
                emb_d = feats_det[d_idx]
                cosine = 1 - np.dot(emb_t, emb_d) / (np.linalg.norm(emb_t) * np.linalg.norm(emb_d) + 1e-6)
                stage1_cost[t_idx, d_idx] = cosine / 2

                pred_pos = predicted[t_idx]
                stage2_cost[t_idx, d_idx] = np.linalg.norm(pred_pos - np.array([x, y])) / self.gating_dist

        # Stage 1: Match only visible detections by appearance
        matches = []
        matched_trks = set()
        matched_dets = set()

        vis_mask = [
            any(v >= self.visibility_switching_thresh for (xx, yy, _), v in vis_scores.items() if xx == x and yy == y)
            for (x, y) in detections
        ]

        stage1_mask = np.ones((num_trk, num_det), dtype=bool)
        for d_idx, visible in enumerate(vis_mask):
            if not visible:
                stage1_mask[:, d_idx] = False

        cost_stage1 = np.where(stage1_mask, stage1_cost, 1.0)
        trk_idx, det_idx = linear_sum_assignment(cost_stage1)
        for t, d in zip(trk_idx, det_idx):
            if cost_stage1[t, d] < self.dist_thresh:
                matches.append((t, d))
                matched_trks.add(t)
                matched_dets.add(d)

        # Stage 2: Fallback on motion for unmatched
        unmatched_trks = [i for i in range(num_trk) if i not in matched_trks]
        unmatched_dets = [i for i in range(num_det) if i not in matched_dets]

        if unmatched_trks and unmatched_dets:
            cost_stage2 = stage2_cost[np.ix_(unmatched_trks, unmatched_dets)]
            t_idx2, d_idx2 = linear_sum_assignment(cost_stage2)

            for t_rel, d_rel in zip(t_idx2, d_idx2):
                t = unmatched_trks[t_rel]
                d = unmatched_dets[d_rel]
                if stage2_cost[t, d] < self.dist_thresh:
                    matches.append((t, d))
                    matched_trks.add(t)
                    matched_dets.add(d)

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
