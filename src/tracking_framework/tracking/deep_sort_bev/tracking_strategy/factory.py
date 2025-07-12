from tracking_framework.tracking.deep_sort_bev.deep_kalman import DeepKalmanBoxTracker

def build_tracking_strategy(params):
    strategy_type = params.get("tracking_strategy", "base")

    if strategy_type == "base":
        from .base import DeepSortBaseStrategy
        return DeepSortBaseStrategy(params, tracker_cls=DeepKalmanBoxTracker)

    elif strategy_type == "visibility_switching":
        from .visibility_switching import VisibilitySwitchingStrategy
        return VisibilitySwitchingStrategy(params, tracker_cls=DeepKalmanBoxTracker)

    elif strategy_type == "two_stage_visibility":
        from .two_stage_visibility import TwoStageMatchingStrategy
        return TwoStageMatchingStrategy(params, tracker_cls=DeepKalmanBoxTracker)

    else:
        raise ValueError(f"Unknown tracking strategy: {strategy_type}")
