import numpy as np
from tracking_framework.tracking.sort_bev.kalman import KalmanBoxTracker

class DeepKalmanBoxTracker(KalmanBoxTracker):
    """
    Extended KalmanBoxTracker with appearance feature memory for DeepSORT-BEV.

    Attributes:
        appearance_features (list): List of feature vectors.
    """

    def __init__(self, initial_position, initial_embedding):
        """
        Initialize the tracker with initial position and appearance embedding.

        Args:
            initial_position (list or np.ndarray): Initial [x, y] position in BEV.
            initial_embedding (np.ndarray): Initial appearance feature vector (512-dim).
        """
        super().__init__(initial_position)
        self.appearance_features = [initial_embedding]

    def get_mean_embedding(self):
        """
        Compute the mean appearance embedding of the tracker.

        Returns:
            np.ndarray: L2-normalized mean appearance embedding.
        """
        if not self.appearance_features:
            return np.zeros(512)

        feats = np.stack(self.appearance_features)
        feats = feats / np.linalg.norm(feats, axis=1, keepdims=True)

        return feats.mean(axis=0)

    def update_appearance(self, new_embedding):
        """
        Update the appearance feature memory with a new embedding.

        Args:
            new_embedding (np.ndarray): New appearance feature vector.
        """
        self.appearance_features.append(new_embedding)

        # Optional: keep buffer size under control
        if len(self.appearance_features) > 30:
            self.appearance_features.pop(0)
