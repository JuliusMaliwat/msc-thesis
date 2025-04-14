from tracking_framework.tracking.base_tracker import BaseTracker
from tracking_framework.tracking.sort_bev.kalman import KalmanBoxTracker
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

class SortBEV(BaseTracker):
    """
    Simple SORT tracking algorithm for BEV detections.
    """

    def __init__(self, params):
        super().__init__(params)
        self.max_age = params.get("max_age", 30)
        self.dist_thresh = params.get("dist_thresh", 40)
        self.trackers = []

    def run(self, detections, dataset):
        """
        Run SORT tracking algorithm.

        Args:
            detections (list): List of detections [[frame_id, x, y], ...]
            dataset (object): Dataset object (not used here, placeholder for interface consistency)

        Returns:
            list: Tracking results [[frame_id, x, y, track_id], ...]
        """
        results = []

        # Organize detections by frame
        detections_by_frame = _detections_to_frame_dict(detections)

        for frame_id in sorted(detections_by_frame.keys()):
            frame_dets = detections_by_frame[frame_id]

            # Predict existing tracker positions
            predicted = [tracker.predict() for tracker in self.trackers]
            unmatched_trackers = list(range(len(self.trackers)))
            unmatched_detections = list(range(len(frame_dets)))

            matches = []

            if predicted and frame_dets:
                # Compute distance between predictions and current detections
                dists = cdist(np.array(predicted), np.array(frame_dets))

                # Apply Hungarian algorithm
                trk_idx, det_idx = linear_sum_assignment(dists)

                for t, d in zip(trk_idx, det_idx):
                    if dists[t, d] < self.dist_thresh:
                        matches.append((t, d))

                # Update lists of unmatched trackers and detections
                unmatched_trackers = [i for i in unmatched_trackers if i not in trk_idx]
                unmatched_detections = [i for i in unmatched_detections if i not in det_idx]

            updated_trackers = []

            # Update matched trackers
            for t, d in matches:
                tracker = self.trackers[t]
                tracker.update(frame_dets[d])
                results.append([frame_id, frame_dets[d][0], frame_dets[d][1], tracker.id])
                updated_trackers.append(tracker)

            # Create new trackers for unmatched detections
            for d in unmatched_detections:
                tracker = KalmanBoxTracker(frame_dets[d])
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
