"""Run prespecified propagation controls on the deterministic audit sample."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from recalculate_av2_sample import LogData, center_distance, iou


ROOT = Path(r"C:\work\auto")
OUT = ROOT / "results" / "av2_confirmation"
SEED = 20260717


def project_with_relative(log: LogData, box: np.ndarray, depth: float, relative: np.ndarray) -> np.ndarray:
    points = log.backproject_box(box, depth)
    transformed = (relative @ np.c_[points, np.ones(len(points))].T).T[:, :3]
    projected = log.project_camera_box(transformed)
    return box.copy() if projected is None else projected


def main() -> None:
    sample = pd.read_parquet(OUT / "row_level_recalculation.parquet").sort_values("comparison_id").reset_index(drop=True)
    split_map = pd.read_csv(OUT / "cohort_source_metadata.csv").set_index("log_id").split.astype(str).to_dict()
    logs = {log_id: LogData(split_map[log_id], log_id) for log_id in sample.log_id.unique()}
    rng = np.random.default_rng(SEED)
    velocities = np.vstack(sample.estimated_object_velocity_json.map(json.loads).map(np.asarray))
    shuffled = velocities[rng.permutation(len(velocities))]
    mismatched = np.roll(velocities, 17, axis=0)
    other_relative = []
    for index, row in sample.iterrows():
        candidates = sample.index[sample.log_id != row.log_id]
        other = sample.loc[candidates[index % len(candidates)]]
        other_relative.append(np.asarray(json.loads(other.relative_camera_transform_json), float))
    records = []
    for index, row in sample.iterrows():
        log = logs[row.log_id]; box = np.asarray(json.loads(row.source_detector_box_json), float); target = np.asarray(json.loads(row.target_annotation_box_json), float)
        velocity = velocities[index]; horizon = row.delta_ms / 1000.0; depth = float(row.estimated_depth_m)
        valid_b4 = np.asarray(json.loads(row.recomputed_B4_box_json), float); valid_b5 = np.asarray(json.loads(row.recomputed_B5_box_json), float)
        shifted_target = int(row.target_timestamp_ns) + 100_000_000
        closest = min(log.poses, key=lambda value: abs(value-shifted_target))
        controls = {
            "valid_B4": valid_b4,
            "valid_B5": valid_b5,
            "time_shifted_ego_pose": log.propagate_geometry(box, depth, int(row.source_timestamp_ns), closest, velocity*horizon),
            "ego_motion_from_another_log": project_with_relative(log, box, depth, other_relative[index]),
            "shuffled_object_velocity": log.propagate_geometry(box, depth, int(row.source_timestamp_ns), int(row.target_timestamp_ns), shuffled[index]*horizon),
            "reversed_object_velocity": log.propagate_geometry(box, depth, int(row.source_timestamp_ns), int(row.target_timestamp_ns), -velocity*horizon),
            "incorrect_depth": log.propagate_geometry(box, depth*2.0, int(row.source_timestamp_ns), int(row.target_timestamp_ns), velocity*horizon),
            "mismatched_track_history": log.propagate_geometry(box, depth, int(row.source_timestamp_ns), int(row.target_timestamp_ns), mismatched[index]*horizon),
            "no_ego_motion_correction": box.copy(),
            "no_object_motion_correction": np.asarray(json.loads(row.recomputed_B3_box_json), float),
        }
        for control, prediction in controls.items():
            records.append({"comparison_id": row.comparison_id, "log_id": row.log_id, "detector": row.detector,
                            "delta_ms": row.delta_ms, "control": control, "center_error_px": center_distance(prediction,target),
                            "normalized_center_error": center_distance(prediction,target)/np.hypot(log.width,log.height), "iou": iou(prediction,target)})
    metrics = pd.DataFrame(records)
    per_log = metrics.groupby(["detector","delta_ms","control","log_id"],as_index=False).agg(
        median_normalized_center_error=("normalized_center_error","median"), median_center_error_px=("center_error_px","median"), median_iou=("iou","median"))
    summary = per_log.groupby(["detector","delta_ms","control"],as_index=False).agg(
        logs=("log_id","nunique"), median_normalized_center_error=("median_normalized_center_error","median"),
        median_center_error_px=("median_center_error_px","median"), median_iou=("median_iou","median"))
    summary["sample_comparisons"] = len(sample)
    summary["scope"] = "deterministic_invalid_run_audit_sample_not_confirmatory"
    summary["same_eligible_comparisons"] = True
    summary.to_csv(OUT / "negative_controls_recovered.csv",index=False,float_format="%.12g")
    print(summary[(summary.delta_ms==300)&summary.control.isin(["valid_B4","valid_B5","time_shifted_ego_pose","no_ego_motion_correction","no_object_motion_correction"])].to_string(index=False))


if __name__ == "__main__":
    main()
