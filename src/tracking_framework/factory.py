class DetectionFactory:
    @staticmethod
    def build(name, params = None):

        if name == "mvdetr":
            from tracking_framework.detection.mvdetr_wrapper import MVDetrWrapper
            return MVDetrWrapper(params)

        else:
            raise ValueError(f"Unknown detector name: {name}")
        


class TrackerFactory:
    @staticmethod
    def build(name, params = None):
        if name == "sort_bev":
            from tracking_framework.tracking.sort_bev.sort_bev import SortBEV
            return SortBEV(params)

        elif name == "deep_sort_bev":
            from tracking_framework.tracking.deep_sort_bev.deep_sort_bev import DeepSortBEVTracker
            return DeepSortBEVTracker(params)

        else:
            raise ValueError(f"Unknown tracker name: {name}")



class DatasetFactory:
    @staticmethod
    def build(name, params = None):
        if name == "wildtrack":
            from tracking_framework.datasets.wildtrack import WildtrackDataset
            return WildtrackDataset()

        elif name == 'multiviewx':
            from tracking_framework.datasets.multiviewx import MultiviewXDataset
            return MultiviewXDataset()

        else:
            raise ValueError(f"Unknown dataset name: {name}")
