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
    Evaluate tracking results in BEV using MOTA, IDF1, and related metrics.

    Args:
        predictions (list): Tracking predictions [[frame_id, x, y, track_id], ...]
        ground_truth (list): Ground truth annotations [[frame_id, x, y, track_id], ...]
        threshold (float): Distance threshold in meters for matching (default 1.0m to match motmetrics).
        unit_scale (float): Scale from BEV units to meters (default 0.025).

    Returns:
        dict: Evaluation results with MOTA, IDF1, Recall, Precision, ID Switches, TP, FP, FN, GT count.
    """
    gt_dict = _group_by_frame(ground_truth)
    pred_dict = _group_by_frame(predictions)

    total_gt = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0
    id_switches = 0

    # Mapping from GT ID to last matched predicted ID
    last_matches = {}

    for frame_id in sorted(set(gt_dict.keys()).union(pred_dict.keys())):
        gt_frame = gt_dict.get(frame_id, [])
        pred_frame = pred_dict.get(frame_id, [])

        # Extract positions and IDs
        gt_positions = np.array([gt[:2] for gt in gt_frame]) * unit_scale
        gt_ids = [gt[2] for gt in gt_frame]

        pred_positions = np.array([pred[:2] for pred in pred_frame]) * unit_scale
        pred_ids = [pred[2] for pred in pred_frame]

        total_gt += len(gt_positions)

        if len(gt_positions) == 0 and len(pred_positions) == 0:
            continue
        if len(gt_positions) == 0:
            total_fp += len(pred_positions)
            continue
        if len(pred_positions) == 0:
            total_fn += len(gt_positions)
            continue

        # Distance matrix
        dists = cdist(gt_positions, pred_positions)
        dists[dists > threshold] = 1e6  # Penalize distant matches

        # Hungarian matching
        row_ind, col_ind = linear_sum_assignment(dists)
        matches = [(r, c) for r, c in zip(row_ind, col_ind) if dists[r, c] <= threshold]

        matched_gt = set()
        matched_pred = set()

        for r, c in matches:
            gt_id = gt_ids[r]
            pred_id = pred_ids[c]

            # Check ID switches
            if gt_id in last_matches:
                if last_matches[gt_id] != pred_id:
                    id_switches += 1

            last_matches[gt_id] = pred_id

            matched_gt.add(r)
            matched_pred.add(c)

        tp = len(matches)
        fp = len(pred_positions) - tp
        fn = len(gt_positions) - tp

        total_tp += tp
        total_fp += fp
        total_fn += fn

    mota = (1 - (total_fp + total_fn + id_switches) / total_gt) * 100 if total_gt > 0 else 0.0
    precision = (total_tp / (total_tp + total_fp)) * 100 if (total_tp + total_fp) > 0 else 0.0
    recall = (total_tp / total_gt) * 100 if total_gt > 0 else 0.0

    # IDF1 calculation
    idf1 = (2 * total_tp / (2 * total_tp + total_fp + total_fn)) * 100 if (2 * total_tp + total_fp + total_fn) > 0 else 0.0

    return {
        "MOTA": round(float(mota), 2),
        "IDF1": round(float(idf1), 2),
        "Recall": round(float(recall), 2),
        "Precision": round(float(precision), 2),
        "ID Switches": int(id_switches),
        "TP": int(total_tp),
        "FP": int(total_fp),
        "FN": int(total_fn),
        "GT": int(total_gt)
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


