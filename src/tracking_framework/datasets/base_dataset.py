class BaseDataset:
    """
    Abstract base class for datasets in the tracking framework.
    """

    def __init__(self, params):
        """
        Initialize dataset with given parameters.

        Args:
            params (dict): Dataset-specific parameters (paths, etc.)
        """
        self.params = params

    def load(self):
        """
        Load dataset resources (intrinsics, extrinsics, etc.)
        """
        raise NotImplementedError("Subclasses must implement the 'load' method.")

    def load_image(self, frame_id, cam_id):
        """
        Load an image for a given frame and camera.

        Args:
            frame_id (int): Frame ID
            cam_id (int): Camera ID

        Returns:
            np.ndarray: Image array
        """
        raise NotImplementedError("Subclasses must implement 'load_image' method.")

    def project_bev_to_image(self, x_idx, y_idx):
        """
        Project BEV detection grid coordinate to 2D bounding boxes in all cameras.

        Args:
            x_idx (int): X index in the BEV grid
            y_idx (int): Y index in the BEV grid

        Returns:
            list of dict: Per-camera projected bounding boxes and corners
        """
        raise NotImplementedError("Subclasses must implement 'project_bev_to_image' method.")

    def get_crop_from_bev(self, frame_id, x_idx, y_idx):
        """
        Get cropped images for a given BEV detection.

        Args:
            frame_id (int): Frame ID
            x_idx (int): X index in the BEV grid
            y_idx (int): Y index in the BEV grid

        Returns:
            list: List of cropped image arrays, one per camera
        """
        raise NotImplementedError("Subclasses must implement 'get_crop_from_bev' method.")

    def get_frame_ids(self):
        """
        Get list of available frame IDs in the dataset.

        Returns:
            list of int: List of frame IDs
        """
        raise NotImplementedError("Subclasses must implement 'get_frame_ids' method.")

    def get_train_frame_ids(self):
        """
        Get list of frame IDs for training split.

        Returns:
            list of int: List of frame IDs used for training.
        """
        raise NotImplementedError("Subclasses must implement 'get_train_frame_ids' method.")

    def get_test_frame_ids(self):
        """
        Get list of frame IDs for test split.

        Returns:
            list of int: List of frame IDs used for evaluation/test.
        """
        raise NotImplementedError("Subclasses must implement 'get_test_frame_ids' method.")

    def get_camera_ids(self):
        """
        Get list of available camera IDs in the dataset.

        Returns:
            list of int: List of camera IDs
        """
        raise NotImplementedError("Subclasses must implement 'get_camera_ids' method.")
    
    
    def get_ground_truth(self, with_track_id=False):
        """
        Get ground truth annotations.

        Args:
            with_track_id (bool): If True, return tracking format [[frame_id, x, y, track_id], ...].
                                If False, return detection format [[frame_id, x, y], ...].

        Returns:
            list: Ground truth annotations.
        """

