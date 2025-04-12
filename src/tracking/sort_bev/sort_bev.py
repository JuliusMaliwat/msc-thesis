import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from tracking.utils.kalman import KalmanBoxTracker

class SortBEV:
    """
    Simple SORT (Simple Online and Realtime Tracking) algorithm 
    for BEV (Bird's Eye View) detections.
    """

    def __init__(self, max_age=5, dist_thresh=30):
        """
        Args:
            max_age (int): Maximum number of frames to keep track without updates.
            dist_thresh (float): Distance threshold for association.
        """
        self.trackers = []
        self.max_age = max_age
        self.dist_thresh = dist_thresh

    def update(self, detections):
        """
        Update tracker with new detections.

        Args:
            detections (list of list): List of detections [[x, y], ...]

        Returns:
            list of tuple: Tracking results [(x, y, track_id), ...]
        """
        updated_trackers = []
        results = []

        # Predict the new positions of existing trackers
        predicted_positions = [tracker.predict() for tracker in self.trackers]
        unmatched_trackers = list(range(len(self.trackers)))
        unmatched_detections = list(range(len(detections)))

        # Data association (tracker <-> detection)
        if len(predicted_positions) > 0 and len(detections) > 0:
            distances = cdist(predicted_positions, detections)
            tracker_indices, detection_indices = linear_sum_assignment(distances)

            for t_idx, d_idx in zip(tracker_indices, detection_indices):
                if distances[t_idx, d_idx] < self.dist_thresh:
                    self.trackers[t_idx].update(detections[d_idx])
                    results.append((
                        detections[d_idx][0],
                        detections[d_idx][1],
                        self.trackers[t_idx].id
                    ))
                    updated_trackers.append(self.trackers[t_idx])

                    unmatched_trackers.remove(t_idx)
                    unmatched_detections.remove(d_idx)

        # Initialize new trackers for unmatched detections
        for d_idx in unmatched_detections:
            new_tracker = KalmanBoxTracker(detections[d_idx])
            new_tracker.update(detections[d_idx])
            updated_trackers.append(new_tracker)
            results.append((
                detections[d_idx][0],
                detections[d_idx][1],
                new_tracker.id
            ))

        # Keep unmatched trackers if they are still "alive"
        for t_idx in unmatched_trackers:
            tracker = self.trackers[t_idx]
            tracker.time_since_update += 1
            if tracker.time_since_update < self.max_age:
                updated_trackers.append(tracker)

        self.trackers = updated_trackers
        return results
