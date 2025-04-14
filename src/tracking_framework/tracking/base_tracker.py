class BaseTracker:
    def __init__(self, params):
        self.params = params

    def run(self, detections, dataset):
        raise NotImplementedError("Subclasses must implement this method")
