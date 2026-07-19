"""Evaluate the once-frozen causal gate on untouched extension logs."""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(r"C:\work\auto")
sys.path.insert(0, str(ROOT / "scripts"))
OUTPUT = ROOT / "results" / "sivp_strengthening"
from recalculate_av2_sample import center_distance, iou, track_history
FEATURES = ["detector_confidence", "track_age", "number_historical_detections", "association_confidence",
            "estimated_3d_velocity_m_s", "velocity_dispersion_m_s", "acceleration_estimate_m_s2",
            "lidar_point_count", "source_depth_m", "depth_uncertainty_m", "source_box_area_px2",
            "ego_yaw_rate_rad_s", "delta_ms"]
SEED = 20260719
REPLICATES = 10_000
YAW_THRESHOLD = 0.0169726458


def bootstrap(values: np.ndarray):
    rng = np.random.default_rng(SEED)
    draws = np.empty(REPLICATES)
    for index in range(REPLICATES):
        draws[index] = np.median(rng.choice(values, len(values), replace=True))
    return float(np.quantile(draws, .025)), float(np.quantile(draws, .975))


def main() -> None:
    specification = json.loads((OUTPUT / "causal_gate_model_spec.json").read_text())
    threshold = float(specification["frozen_threshold"])
    model = joblib.load(OUTPUT / "frozen_causal_gate.joblib")
    rows = []
    for detector_dir in sorted((OUTPUT / "extension_propagation").glob("*")):
        detector = detector_dir.name
        for path in sorted(detector_dir.glob("*.parquet")):
            frame = pd.read_parquet(path)
            checkpoint_path = (OUTPUT / "extension_detector_checkpoints" / detector / f"{path.stem}.json"
                               if detector != "rtdetr_l" else OUTPUT / "third_detector_checkpoints" / "extension" / f"{path.stem}.json")
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            confidence_by_index = {index: float(detection["confidence"]) for index, detection in enumerate(checkpoint["detections"])}
            detections_by_timestamp = defaultdict(list)
            for index, detection in enumerate(checkpoint["detections"]):
                detections_by_timestamp[int(detection["timestamp_ns"])].append((index, detection))
            b3 = frame[frame.model == "B3"].copy().set_index("comparison_id")
            b5 = frame[frame.model == "B5"].copy().set_index("comparison_id")
            common = b3.index.intersection(b5.index)
            b3, b5 = b3.loc[common], b5.loc[common]
            feature = pd.DataFrame(index=common)
            feature["detector_confidence"] = [confidence_by_index.get(int(str(value).split(":")[3]), np.nan) for value in common]
            feature["track_age"] = b3.track_age
            feature["number_historical_detections"] = b3.track_age
            association_confidence = []
            for comparison_id in common:
                index = int(str(comparison_id).split(":")[3])
                detection = checkpoint["detections"][index]
                source_timestamp = int(detection["timestamp_ns"])
                history = track_history(detections_by_timestamp, source_timestamp, index, detection)
                if len(history) < 2:
                    association_confidence.append(0.0)
                    continue
                current_box = np.asarray([detection["x1"], detection["y1"], detection["x2"], detection["y2"]], float)
                previous = history[-2][2]
                previous_box = np.asarray([previous["x1"], previous["y1"], previous["x2"], previous["y2"]], float)
                overlap = iou(current_box, previous_box)
                distance = center_distance(current_box, previous_box) / math.hypot(1550, 2048)
                association_confidence.append(float(np.clip(overlap * math.exp(-5 * distance), 0, 1)))
            feature["association_confidence"] = association_confidence
            feature["estimated_3d_velocity_m_s"] = b3.object_velocity_norm_m_s
            feature["velocity_dispersion_m_s"] = np.nan
            feature["acceleration_estimate_m_s2"] = np.nan
            feature["lidar_point_count"] = b3.lidar_point_count
            feature["source_depth_m"] = b3.estimated_depth_m
            feature["depth_uncertainty_m"] = b3.depth_dispersion_m
            feature["source_box_area_px2"] = (b3.source_x2 - b3.source_x1).clip(lower=0) * (b3.source_y2 - b3.source_y1).clip(lower=0)
            feature["ego_yaw_rate_rad_s"] = b3.yaw_rate_rad_s
            feature["delta_ms"] = b3.delta_ms
            probability = model.predict_proba(feature[FEATURES])[:, 1]
            choose_b5 = probability >= threshold
            for index, comparison_id in enumerate(common):
                base = b3.loc[comparison_id]
                selected = b5.loc[comparison_id] if choose_b5[index] else base
                rows.append({"comparison_id": comparison_id, "log_id": base.log_id, "detector": detector,
                             "delta_ms": int(base.delta_ms), "yaw_rate_rad_s": base.yaw_rate_rad_s,
                             "gate_probability": probability[index], "gate_selected": "B5" if choose_b5[index] else "B3",
                             "B3_error": base.normalized_center_error, "B5_error": b5.loc[comparison_id].normalized_center_error,
                             "G2_error": selected.normalized_center_error, "B3_iou": base.iou, "B5_iou": b5.loc[comparison_id].iou,
                             "G2_iou": selected.iou, "B3_recall": base.recall_iou_0_5,
                             "B5_recall": b5.loc[comparison_id].recall_iou_0_5, "G2_recall": selected.recall_iou_0_5,
                             "future_information_input": False})
    low_level = pd.DataFrame(rows)
    overall = low_level.copy(); overall["endpoint"] = "all_horizons_all_yaw"
    primary = low_level[(low_level.delta_ms == 300) & (low_level.yaw_rate_rad_s.abs() > YAW_THRESHOLD)].copy()
    primary["endpoint"] = "300ms_high_yaw"
    evaluation = pd.concat([overall, primary], ignore_index=True)
    per_log = evaluation.groupby(["detector", "log_id", "endpoint"], as_index=False).agg(
        comparisons=("comparison_id", "nunique"), B3_error=("B3_error", "median"), B5_error=("B5_error", "median"),
        G2_error=("G2_error", "median"), B3_iou=("B3_iou", "median"), B5_iou=("B5_iou", "median"),
        G2_iou=("G2_iou", "median"), B3_recall=("B3_recall", "mean"), B5_recall=("B5_recall", "mean"),
        G2_recall=("G2_recall", "mean"), selected_B5_percent=("gate_selected", lambda values: (values == "B5").mean() * 100))
    per_log["G2_minus_B3_error"] = per_log.G2_error - per_log.B3_error
    per_log["G2_minus_B3_iou"] = per_log.G2_iou - per_log.B3_iou
    per_log["G2_minus_B3_recall"] = per_log.G2_recall - per_log.B3_recall
    per_log.to_csv(OUTPUT / "causal_gate_per_log.csv", index=False, float_format="%.12g")
    results, intervals = [], []
    for (detector, endpoint), frame in per_log.groupby(["detector", "endpoint"]):
        difference = frame.G2_minus_B3_error.to_numpy()
        low, high = bootstrap(difference)
        results.append({"detector": detector, "endpoint": endpoint, "logs": len(frame), "comparisons": int(frame.comparisons.sum()),
                        "median_B3_error": frame.B3_error.median(), "median_B5_error": frame.B5_error.median(),
                        "median_G2_error": frame.G2_error.median(), "median_G2_minus_B3_error": np.median(difference),
                        "median_G2_minus_B3_iou": frame.G2_minus_B3_iou.median(),
                        "recall_loss_percentage_points": -frame.G2_minus_B3_recall.median() * 100,
                        "extension_logs_improved_percent": (difference < 0).mean() * 100,
                        "selected_B5_percent": frame.selected_B5_percent.median()})
        try:
            p_value = float(wilcoxon(difference).pvalue)
        except ValueError:
            p_value = 1.0
        intervals.append({"detector": detector, "endpoint": endpoint, "contrast": "G2_minus_B3_normalized_center_error",
                          "median_difference": np.median(difference), "bootstrap_ci_low": low,
                          "bootstrap_ci_high": high, "wilcoxon_p": p_value, "replicates": REPLICATES})
    pd.DataFrame(results).to_csv(OUTPUT / "causal_gate_results.csv", index=False, float_format="%.12g")
    pd.DataFrame(intervals).to_csv(OUTPUT / "causal_gate_bootstrap.csv", index=False, float_format="%.12g")
    print(json.dumps({"rows": len(low_level), "detectors": sorted(low_level.detector.unique()), "threshold": threshold}))


if __name__ == "__main__":
    main()
