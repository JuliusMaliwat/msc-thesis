import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


class TwoStageMatchingStrategy:
    def __init__(self, params, tracker_cls):
        self.app_dist_thresh = params.get("app_dist_thresh", 0.4)
        self.motion_dist_thresh = params.get("motion_dist_thresh", 40.0)
        self.gating_dist = params.get("gating_dist", 40.0)
        self.tracker_cls = tracker_cls

    def track(self, *, frame_id, trackers, detections, embeddings, dataset, max_age):
        predicted = [trk.predict() for trk in trackers]
        results = []

        num_trk = len(trackers)
        num_det = len(detections)
        unmatched_trackers = list(range(num_trk))
        unmatched_detections = list(range(num_det))
        matches = []

        if predicted and detections:
            preds = np.array(predicted)
            dets = np.array(detections)

            dists_pos = cdist(preds, dets)
            gating_mask = dists_pos < self.gating_dist

            feats_trk = np.stack([trk.get_mean_embedding() for trk in trackers])
            feats_det = np.stack([
                embeddings.get((frame_id, x, y), np.zeros(512))
                for (x, y) in detections
            ])

            stage1_cost = np.full((num_trk, num_det), fill_value=1.0, dtype=float)
            stage2_cost = np.full((num_trk, num_det), fill_value=self.motion_dist_thresh * 2.0, dtype=float)

            for t_idx in range(num_trk):
                for d_idx in range(num_det):
                    if not gating_mask[t_idx, d_idx]:
                        continue

                    emb_t = feats_trk[t_idx]
                    emb_d = feats_det[d_idx]
                    cosine = 1.0 - (np.dot(emb_t, emb_d) /
                                   (np.linalg.norm(emb_t) * np.linalg.norm(emb_d) + 1e-6))
                    stage1_cost[t_idx, d_idx] = cosine
                    stage2_cost[t_idx, d_idx] = dists_pos[t_idx, d_idx]

            # --- Stage 1: Appearance matching (NO visibility filtering) ---
            trk_idx1, det_idx1 = linear_sum_assignment(stage1_cost)
            matched_trks = set()
            matched_dets = set()

            for t, d in zip(trk_idx1, det_idx1):
                if stage1_cost[t, d] < self.app_dist_thresh:
                    matches.append((t, d))
                    matched_trks.add(t)
                    matched_dets.add(d)

            # --- Stage 2: Motion matching ---
            unmatched_trks = [i for i in range(num_trk) if i not in matched_trks]
            unmatched_dets = [j for j in range(num_det) if j not in matched_dets]

            if unmatched_trks and unmatched_dets:
                print("qualcosa è rimasto!")
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

        # --- Update matched trackers ---
        updated_trackers = []
        for t, d in matches:
            trk = trackers[t]
            x, y = detections[d]
            emb = embeddings.get((frame_id, x, y), np.zeros(512))
            trk.update([x, y])
            trk.update_appearance(emb)
            updated_trackers.append(trk)
            results.append([frame_id, x, y, trk.id])

        # --- Initialize new trackers ---
        for d in unmatched_detections:
            x, y = detections[d]
            emb = embeddings.get((frame_id, x, y), np.zeros(512))
            new_trk = self.tracker_cls([x, y], emb)
            updated_trackers.append(new_trk)
            results.append([frame_id, x, y, new_trk.id])

        # --- Age out unmatched trackers ---
        for t in unmatched_trackers:
            trk = trackers[t]
            trk.time_since_update += 1
            if trk.time_since_update < max_age:
                updated_trackers.append(trk)

        return updated_trackers, results
