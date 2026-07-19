"""Compute persistent-object and full-frame current-time detection metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(r"C:\work\auto")
LOW_LEVEL = ROOT / "results" / "sivp_strengthening" / "end_to_end_low_level" / "heldout"
OUTPUT = ROOT / "results" / "sivp_strengthening"
YAW_THRESHOLD = 0.0169726458
IOU_THRESHOLDS = np.round(np.arange(0.50, 0.96, 0.05), 2)
SEED = 20260719
REPLICATES = 10_000


def iou_matrix(predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
    if len(predictions) == 0 or len(targets) == 0:
        return np.zeros((len(predictions), len(targets)))
    top_left = np.maximum(predictions[:, None, :2], targets[None, :, :2])
    bottom_right = np.minimum(predictions[:, None, 2:], targets[None, :, 2:])
    size = np.maximum(bottom_right - top_left, 0)
    intersection = size[:, :, 0] * size[:, :, 1]
    pred_area = np.maximum(predictions[:, 2] - predictions[:, 0], 0) * np.maximum(predictions[:, 3] - predictions[:, 1], 0)
    target_area = np.maximum(targets[:, 2] - targets[:, 0], 0) * np.maximum(targets[:, 3] - targets[:, 1], 0)
    return intersection / np.maximum(pred_area[:, None] + target_area[None, :] - intersection, 1e-12)


def match_all(predictions: pd.DataFrame, targets: pd.DataFrame):
    flags = {float(threshold): np.zeros(len(predictions), dtype=bool) for threshold in IOU_THRESHOLDS}
    matched_keys = {float(threshold): set() for threshold in IOU_THRESHOLDS}
    target_groups = {(frame_id, group): values for (frame_id, group), values in targets.groupby(["frame_id", "analysis_group"], sort=False)}
    for (frame_id, group), pred_group in predictions.groupby(["frame_id", "analysis_group"], sort=False):
        target_group = target_groups.get((frame_id, group))
        if target_group is None:
            continue
        pred_group = pred_group.sort_values("confidence", ascending=False)
        pred_boxes = pred_group[["pred_x1", "pred_y1", "pred_x2", "pred_y2"]].to_numpy(float)
        target_boxes = target_group[["target_x1", "target_y1", "target_x2", "target_y2"]].to_numpy(float)
        overlaps = iou_matrix(pred_boxes, target_boxes)
        orders = np.argsort(overlaps, axis=1)[:, ::-1]
        for threshold in IOU_THRESHOLDS:
            threshold = float(threshold)
            used: set[int] = set()
            for local_prediction, prediction_index in enumerate(pred_group.index):
                chosen = next((int(index) for index in orders[local_prediction]
                               if index not in used and overlaps[local_prediction, index] >= threshold), None)
                if chosen is None:
                    continue
                used.add(chosen)
                flags[threshold][int(prediction_index)] = True
                target = target_group.iloc[chosen]
                matched_keys[threshold].add((str(frame_id), str(target.target_track_uuid)))
    return flags, matched_keys


def interpolated_ap(scores: np.ndarray, flags: np.ndarray, target_count: int) -> float:
    if target_count == 0:
        return float("nan")
    order = np.argsort(scores)[::-1]
    tp = np.cumsum(flags[order].astype(float))
    fp = np.cumsum((~flags[order]).astype(float))
    recall = tp / target_count
    precision = tp / np.maximum(tp + fp, 1e-12)
    return float(np.mean([precision[recall >= level].max() if np.any(recall >= level) else 0.0 for level in np.linspace(0, 1, 101)]))


def evaluate(predictions: pd.DataFrame, targets: pd.DataFrame, scope: str) -> dict[str, float | int]:
    predictions = predictions.reset_index(drop=True)
    targets = targets.reset_index(drop=True)
    if scope == "persistent_object":
        targets = targets[targets.persistent_object.astype(bool)]
    target_count = len(targets)
    frame_count = max(predictions.frame_id.nunique(), targets.frame_id.nunique(), 1)
    ap_values = []
    scores = predictions.confidence.to_numpy(float)
    flags_by_threshold, matches_by_threshold = match_all(predictions, targets)
    for threshold in IOU_THRESHOLDS:
        flags = flags_by_threshold[float(threshold)]
        ap_values.append(interpolated_ap(scores, flags, target_count))
    flags50 = flags_by_threshold[0.5]
    true_positive = int(flags50.sum())
    false_positive = int((~flags50).sum())
    matched50 = matches_by_threshold[0.5]
    new_targets = targets[targets.new_object.astype(bool)]
    missed_new = sum((str(row.frame_id), str(row.target_track_uuid)) not in matched50 for row in new_targets.itertuples(index=False))
    disappeared = predictions.source_object_disappeared.astype(bool).to_numpy()
    disappeared_fp = int((disappeared & ~flags50).sum())
    return {
        "target_objects": target_count, "predictions": len(predictions), "frames": frame_count,
        "ap_50_95": float(np.nanmean(ap_values)) if target_count else float("nan"),
        "ap50": ap_values[0] if target_count else float("nan"),
        "ap75": ap_values[5] if target_count else float("nan"),
        "precision": true_positive / max(true_positive + false_positive, 1),
        "recall": true_positive / max(target_count, 1),
        "false_positives_per_frame": false_positive / frame_count,
        "persistent_object_recall": true_positive / max(target_count, 1) if scope == "persistent_object" else float("nan"),
        "new_object_miss_fraction": missed_new / max(len(new_targets), 1) if scope == "full_frame" else float("nan"),
        "disappeared_object_false_positive_fraction": disappeared_fp / max(int(disappeared.sum()), 1),
    }


def strata(predictions: pd.DataFrame, targets: pd.DataFrame, detailed: bool):
    yield "overall", "all", predictions, targets
    if not detailed:
        return
    for group in ("vehicle", "pedestrian_cyclist"):
        yield "class", group, predictions[predictions.analysis_group == group], targets[targets.analysis_group == group]
    bins = [(0, 20, "0_20m"), (20, 40, "20_40m"), (40, np.inf, "40m_plus")]
    for lower, upper, label in bins:
        selected_targets = targets[(targets.distance_m >= lower) & (targets.distance_m < upper)]
        keys = set(zip(selected_targets.frame_id.astype(str), selected_targets.target_track_uuid.astype(str)))
        selected_predictions = predictions[
            [(str(frame), str(track)) in keys for frame, track in zip(predictions.frame_id, predictions.source_track_uuid)]
        ]
        yield "distance", label, selected_predictions, selected_targets


def bootstrap(values: np.ndarray):
    rng = np.random.default_rng(SEED)
    draws = np.empty(REPLICATES)
    for index in range(REPLICATES):
        draws[index] = np.median(rng.choice(values, len(values), replace=True))
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--detector", choices=["rtdetr_l"])
    args = parser.parse_args()
    paths = sorted(LOW_LEVEL.glob("*/*.predictions.parquet"))
    paths = ([path for path in paths if path.parent.name == args.detector] if args.detector else
             [path for path in paths if path.parent.name in {"yolo11n", "yolo11s"}])
    expected = 50 if args.detector else 100
    if len(paths) != expected:
        raise RuntimeError(f"Expected {expected} end-to-end prediction shards, found {len(paths)}")
    paths = [path for index, path in enumerate(paths) if index % args.shard_count == args.shard_index]
    rows = []
    for path_index, prediction_path in enumerate(paths, 1):
        target_path = prediction_path.with_name(prediction_path.name.replace(".predictions.parquet", ".targets.parquet"))
        predictions = pd.read_parquet(prediction_path)
        targets = pd.read_parquet(target_path)
        detector, log_id = str(predictions.detector.iloc[0]), str(predictions.log_id.iloc[0])
        for delta_ms in sorted(predictions.delta_ms.unique()):
            delta_predictions = predictions[predictions.delta_ms == delta_ms]
            delta_targets = targets[targets.delta_ms == delta_ms]
            for yaw_stratum in ("all_yaw", "high_yaw"):
                if yaw_stratum == "high_yaw":
                    valid_frames = set(delta_targets.loc[delta_targets.yaw_rate_rad_s.abs() > YAW_THRESHOLD, "frame_id"])
                    current_predictions = delta_predictions[delta_predictions.frame_id.isin(valid_frames)]
                    current_targets = delta_targets[delta_targets.frame_id.isin(valid_frames)]
                else:
                    current_predictions, current_targets = delta_predictions, delta_targets
                if current_targets.empty:
                    continue
                detailed = yaw_stratum == "all_yaw" and int(delta_ms) == 300
                for model in sorted(current_predictions.model.unique()):
                    model_predictions = current_predictions[current_predictions.model == model]
                    for stratum_type, stratum, stratum_predictions, stratum_targets in strata(model_predictions, current_targets, detailed and model in {"T5", "T6"}):
                        for scope in ("persistent_object", "full_frame"):
                            result = evaluate(stratum_predictions, stratum_targets, scope)
                            rows.append({"detector": detector, "log_id": log_id, "model": model,
                                         "delta_ms": int(delta_ms), "yaw_stratum": yaw_stratum,
                                         "stratum_type": stratum_type, "stratum": stratum, "scope": scope, **result})
        print(json.dumps({"completed": path_index, "total": len(paths), "detector": detector, "log": log_id}), flush=True)
    per_log = pd.DataFrame(rows)
    if args.shard_count > 1:
        shard_dir = OUTPUT / "end_to_end_metric_shards"
        shard_dir.mkdir(parents=True, exist_ok=True)
        suffix = f"_{args.detector}" if args.detector else ""
        per_log.to_csv(shard_dir / f"shard_{args.shard_index}_of_{args.shard_count}{suffix}.csv", index=False, float_format="%.12g")
        print(json.dumps({"shard": args.shard_index, "per_log_rows": len(per_log)}), flush=True)
        return
    per_log.to_csv(OUTPUT / "end_to_end_per_log_metrics.csv", index=False, float_format="%.12g")
    group = ["detector", "model", "delta_ms", "yaw_stratum", "stratum_type", "stratum", "scope"]
    metrics = ["ap_50_95", "ap50", "ap75", "precision", "recall", "false_positives_per_frame",
               "persistent_object_recall", "new_object_miss_fraction", "disappeared_object_false_positive_fraction"]
    aggregation = {metric: (metric, "median") for metric in metrics}
    aggregation.update({"eligible_logs": ("log_id", "nunique"), "target_objects": ("target_objects", "sum"),
                        "predictions": ("predictions", "sum"), "frames": ("frames", "sum")})
    summary = per_log.groupby(group, as_index=False).agg(**aggregation)
    summary[summary.scope == "persistent_object"].to_csv(OUTPUT / "persistent_object_detection_metrics.csv", index=False, float_format="%.12g")
    summary[summary.scope == "full_frame"].to_csv(OUTPUT / "full_frame_detection_metrics.csv", index=False, float_format="%.12g")

    contrasts = []
    principal = per_log[(per_log.stratum_type == "overall") & per_log.model.isin(["T4", "T5", "T6"])]
    for keys, frame in principal.groupby(["detector", "delta_ms", "yaw_stratum", "scope"]):
        detector, delta_ms, yaw_stratum, scope = keys
        for metric in ("ap_50_95", "ap50", "recall"):
            pivot = frame.pivot(index="log_id", columns="model", values=metric)
            for candidate in ("T4", "T6"):
                paired = pivot[["T5", candidate]].dropna()
                if paired.empty:
                    continue
                difference = paired[candidate] - paired.T5
                relative = difference / paired.T5.replace(0, np.nan)
                low, high = bootstrap(difference.to_numpy())
                try:
                    p_value = float(wilcoxon(difference).pvalue)
                except ValueError:
                    p_value = 1.0
                contrasts.append({"detector": detector, "delta_ms": delta_ms, "yaw_stratum": yaw_stratum,
                                  "scope": scope, "metric": metric, "candidate": candidate, "reference": "T5",
                                  "eligible_logs": len(paired), "paired_absolute_difference": float(np.median(difference)),
                                  "paired_relative_difference": float(np.nanmedian(relative)),
                                  "percent_logs_candidate_improved": float((difference > 0).mean() * 100),
                                  "bootstrap_ci_low": low, "bootstrap_ci_high": high, "wilcoxon_p": p_value})
    pd.DataFrame(contrasts).to_csv(OUTPUT / "end_to_end_bootstrap_intervals.csv", index=False, float_format="%.12g")
    print(json.dumps({"per_log_rows": len(per_log), "summary_rows": len(summary)}))


if __name__ == "__main__":
    main()
