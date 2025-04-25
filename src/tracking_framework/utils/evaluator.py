import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from collections import defaultdict

def evaluate_detection(predictions, ground_truth, threshold=0.5, unit_scale=0.025):
    """
    Evaluate detection results in BEV using MODA, MODP, Recall, and Precision.

    Args:
        predictions (list): List of predicted detections [[frame_id, x, y], ...]
        ground_truth (list): List of ground truth annotations [[frame_id, x, y], ...]
        threshold (float): Distance threshold in meters for considering a match.
        unit_scale (float): Scaling factor from BEV units to meters (e.g., 0.025).

    Returns:
        dict: Dictionary with MODA, MODP, Recall, Precision, TP, FP, FN, and GT count.
    """
    # Convert lists to dicts grouped by frame
    pred_dict = _group_by_frame(predictions)
    gt_dict = _group_by_frame(ground_truth)

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_gt = 0
    total_score = 0.0

    for frame_id in sorted(set(gt_dict.keys()).union(pred_dict.keys())):
        gt = gt_dict.get(frame_id, [])
        pred = pred_dict.get(frame_id, [])

        # Convert to meters
        gt_m = np.array(gt) * unit_scale
        pred_m = np.array(pred) * unit_scale

        total_gt += len(gt_m)

        if len(gt_m) == 0 and len(pred_m) == 0:
            continue

        if len(gt_m) == 0:
            total_fp += len(pred_m)
            continue

        if len(pred_m) == 0:
            total_fn += len(gt_m)
            continue

        # Compute distance matrix
        dists = cdist(gt_m, pred_m)
        dists[dists > threshold] = 1e6  # Large value for distances above threshold

        row_ind, col_ind = linear_sum_assignment(dists)
        matches = [(r, c) for r, c in zip(row_ind, col_ind) if dists[r, c] <= threshold]

        tp = len(matches)
        fp = len(pred_m) - tp
        fn = len(gt_m) - tp

        total_tp += tp
        total_fp += fp
        total_fn += fn

        for r, c in matches:
            score = 1 - (cdist([gt_m[r]], [pred_m[c]])[0][0] / threshold)
            total_score += score

    moda = (1 - (total_fn + total_fp) / total_gt) * 100 if total_gt > 0 else 0.0
    modp = (total_score / total_tp) * 100 if total_tp > 0 else 0.0
    recall = (total_tp / total_gt) * 100 if total_gt > 0 else 0.0
    precision = (total_tp / (total_tp + total_fp)) * 100 if (total_tp + total_fp) > 0 else 0.0

    return {
        "MODA": round(float(moda),2),
        "MODP": round(float(modp), 2),
        "Recall": round(float(recall), 2),
        "Precision": round(float(precision), 2),
        "TP": int(total_tp),
        "FP": int(total_fp),
        "FN": int(total_fn),
        "GT": int(total_gt)
    }


def evaluate_tracking(predictions, ground_truth, threshold=1.0, unit_scale=0.025):
    """
    Evaluates BEV tracking using motmetrics.

    Args:
        predictions: List of tracking predictions in the format [[frame, x, y, id], ...]
        ground_truth: List of ground truth annotations in the format [[frame, x, y, id], ...]
        threshold: Maximum matching distance in meters (default is 1.0 meter)
        unit_scale: Conversion factor from BEV units to meters (default is 0.025)

    Returns:
        dict: Evaluation metrics including MOTA, IDF1, IDP, IDR, FP, FN, ID switches, and total GT count.
    """
    print(f"Len ground_truth dentro evaluate_tracking prima del group by frame: {len(ground_truth)}")  

    gt_by_frame = _group_by_frame(ground_truth)
    pred_by_frame = _group_by_frame(predictions)
    print(f"Len ground_truth dentro evaluate_tracking DOPO il group by frame: {len(ground_truth)}")  

    import motmetrics as mm

    acc = mm.MOTAccumulator(auto_id=True)
    all_frames = sorted(set(gt_by_frame.keys()) | set(pred_by_frame.keys()))

    for frame in all_frames:
        gt_data = gt_by_frame.get(frame, [])
        pred_data = pred_by_frame.get(frame, [])

        gt_ids = [int(gt[2]) for gt in gt_data]
        pred_ids = [int(p[2]) for p in pred_data]

        gt_coords = np.array([gt[:2] for gt in gt_data]) * unit_scale
        pred_coords = np.array([p[:2] for p in pred_data]) * unit_scale

        if len(gt_coords) == 0 and len(pred_coords) == 0:
            acc.update([], [], [])
        elif len(gt_coords) == 0:
            acc.update([], pred_ids, [])
        elif len(pred_coords) == 0:
            acc.update(gt_ids, [], [])
        else:
            dists = mm.distances.norm2squared_matrix(gt_coords, pred_coords, max_d2=threshold**2)
            acc.update(gt_ids, pred_ids, np.sqrt(dists))

    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=mm.metrics.motchallenge_metrics, name="tracker")
    result = summary.loc["tracker"].to_dict()
    print(f"Len ground_truth dentro evaluate_tracking appena prima di return: {len(ground_truth)}")  

    return {
        "MOTA": round(result["mota"] * 100, 2),
        "IDF1": round(result["idf1"] * 100, 2),
        "IDP": round(result["idp"] * 100, 2),
        "IDR": round(result["idr"] * 100, 2),
        "Recall": round(result["recall"] * 100, 2),
        "Precision": round(result["precision"] * 100, 2),
        "ID Switches": int(result["num_switches"]),
        "FP": int(result["num_false_positives"]),
        "FN": int(result["num_misses"]),
        "GT": len(ground_truth)
    }

def _group_by_frame(data):
    """
    Group tracking data by frame.

    Args:
        data (list): List of entries [[frame_id, x, y, track_id], ...]

    Returns:
        dict: Mapping from frame_id to list of [x, y, track_id]
    """
    grouped = defaultdict(list)
    for entry in data:
        frame_id = entry[0]
        grouped[frame_id].append(entry[1:])  # Take all except frame_id
    return grouped


