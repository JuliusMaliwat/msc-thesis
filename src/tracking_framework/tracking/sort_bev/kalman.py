import numpy as np

class KalmanBoxTracker:
    """
    Kalman Filter based tracker for 2D positions (x, y) in BEV.

    State vector: [x, y, dx, dy]
    """
    count = 0

    def __init__(self, initial_position):
        # Initialize state
        self.kf = self._init_kalman()
        self.kf[:2, 0] = initial_position  # Initialize position (x, y)

        # Tracker variables
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []

    def _init_kalman(self):
        """
        Initialize the Kalman Filter matrices.
        """
        # State: [x, y, dx, dy]^T
        kf = np.zeros((4, 1))

        # State transition matrix
        self.F = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        # Observation matrix
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])

        # Initial covariance matrix
        self.P = np.eye(4) * 1000

        # Process noise covariance
        self.Q = np.eye(4)

        # Measurement noise covariance
        self.R = np.eye(2)

        return kf

    def predict(self):
        """
        Predict the next state.
        """
        self.kf = self.F @ self.kf
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.kf[:2].flatten()

    def update(self, measurement):
        """
        Update the state with a new measurement.
        """
        z = np.reshape(measurement, (2, 1))
        y = z - self.H @ self.kf
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.kf = self.kf + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        self.time_since_update = 0  # Reset since update

    def get_state(self):
        """
        Return the current estimated position.
        """
        return self.kf[:2].flatten()
