import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from tracking_framework.utils.visibility import compute_frame_visibility_scores


class TwoStageMatchingStrategy:
    def __init__(self, params, tracker_cls):
        # Appearance threshold (cosine distance)
        self.app_dist_thresh = params.get("app_dist_thresh", 0.4)
        # Motion threshold (Euclidean distance), same as SORT default
        self.motion_dist_thresh = params.get("motion_dist_thresh", 40.0)
        # Gating distance to pre‐filter matches by motion
        self.gating_dist = params.get("gating_dist", 40.0)
        # Visibility threshold for deciding which detections use appearance first
        self.visibility_switching_thresh = params.get("visibility_switching_thresh", 200.0)
        self.tracker_cls = tracker_cls

    def track(self, *, frame_id, trackers, detections, embeddings, dataset, max_age):
        """
        Two‐stage matching:
          1. Appearance‐based matching for sufficiently visible detections.
          2. Motion‐based (Euclidean) matching for the remaining trackers & detections.

        Returns:
          updated_trackers: list of tracker instances after update
          results: list of [frame_id, x, y, track_id]
        """
        # 1) Predict new positions (for gating / motion costs)
        predicted = [trk.predict() for trk in trackers]
        results = []

        num_trk = len(trackers)
        num_det = len(detections)
        unmatched_trackers = list(range(num_trk))
        unmatched_detections = list(range(num_det))
        matches = []

        if predicted and detections:
            # 2) Compute visibility scores for each detection
            vis_scores = compute_frame_visibility_scores(dataset, detections)

            preds = np.array(predicted)       # shape: (num_trk, 2)
            dets = np.array(detections)       # shape: (num_det, 2)

            # 3) Motion gating: Euclidean distance between preds and dets
            dists_pos = cdist(preds, dets)    # shape: (num_trk, num_det)
            gating_mask = dists_pos < self.gating_dist

            # 4) Appearance embeddings (no zero‐padding needed since num_trk, num_det > 0 here)
            feats_trk = np.stack([trk.get_mean_embedding() for trk in trackers])
            feats_det = np.stack([
                embeddings.get((frame_id, x, y), np.zeros(512))
                for (x, y) in detections
            ])

            # 5) Build cost matrices
            # Initialize to large cost by default
            stage1_cost = np.full((num_trk, num_det), fill_value=1.0, dtype=float)
            stage2_cost = np.full((num_trk, num_det), fill_value=self.motion_dist_thresh * 2.0, dtype=float)

            for t_idx in range(num_trk):
                for d_idx in range(num_det):
                    if not gating_mask[t_idx, d_idx]:
                        continue

                    # (a) Appearance cost = cosine distance
                    emb_t = feats_trk[t_idx]
                    emb_d = feats_det[d_idx]
                    cosine = 1.0 - (np.dot(emb_t, emb_d) /
                                   (np.linalg.norm(emb_t) * np.linalg.norm(emb_d) + 1e-6))
                    stage1_cost[t_idx, d_idx] = cosine

                    # (b) Motion cost = raw Euclidean distance
                    stage2_cost[t_idx, d_idx] = dists_pos[t_idx, d_idx]

            # --- Stage 1: Appearance matching on visible detections only ---
            # Determine which detections are “visible” enough
            vis_mask = [
                any(
                    v >= self.visibility_switching_thresh
                    for (xx, yy, _), v in vis_scores.items()
                    if xx == x and yy == y
                )
                for (x, y) in detections
            ]
            # Broadcast vis_mask to (num_trk, num_det)
            vis_broadcast = np.broadcast_to(np.array(vis_mask)[None, :], (num_trk, num_det))
            # If a detection is not visible, force its cost to a large value > app_dist_thresh
            cost_stage1 = np.where(vis_broadcast, stage1_cost, 1.0)

            trk_idx1, det_idx1 = linear_sum_assignment(cost_stage1)
            matched_trks = set()
            matched_dets = set()

            for t, d in zip(trk_idx1, det_idx1):
                if cost_stage1[t, d] < self.app_dist_thresh:
                    matches.append((t, d))
                    matched_trks.add(t)
                    matched_dets.add(d)

            # --- Stage 2: Motion matching for the remaining ---
            unmatched_trks = [i for i in range(num_trk) if i not in matched_trks]
            unmatched_dets = [j for j in range(num_det) if j not in matched_dets]

            if unmatched_trks and unmatched_dets:
                # Build a submatrix of costs for unmatched rows/cols
                sub_cost = stage2_cost[np.ix_(unmatched_trks, unmatched_dets)]
                trk_idx2_rel, det_idx2_rel = linear_sum_assignment(sub_cost)
                for rel_t, rel_d in zip(trk_idx2_rel, det_idx2_rel):
                    t = unmatched_trks[rel_t]
                    d = unmatched_dets[rel_d]
                    if stage2_cost[t, d] < self.motion_dist_thresh:
                        print("Match con motion!")
                        matches.append((t, d))
                        matched_trks.add(t)
                        matched_dets.add(d)

            unmatched_trackers = [i for i in range(num_trk) if i not in matched_trks]
            unmatched_detections = [j for j in range(num_det) if j not in matched_dets]

        # 6) Update matched trackers
        updated_trackers = []
        for t, d in matches:
            trk = trackers[t]
            x, y = detections[d]
            emb = embeddings.get((frame_id, x, y), np.zeros(512))
            trk.update([x, y])
            trk.update_appearance(emb)
            updated_trackers.append(trk)
            results.append([frame_id, x, y, trk.id])

        # 7) Initialize new trackers for unmatched detections
        for d in unmatched_detections:
            x, y = detections[d]
            emb = embeddings.get((frame_id, x, y), np.zeros(512))
            new_trk = self.tracker_cls([x, y], emb)
            updated_trackers.append(new_trk)
            results.append([frame_id, x, y, new_trk.id])

        # 8) Age out unmatched trackers
        for t in unmatched_trackers:
            trk = trackers[t]
            trk.time_since_update += 1
            if trk.time_since_update < max_age:
                updated_trackers.append(trk)

        return updated_trackers, results
