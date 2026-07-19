"""Generate all-detection current-time predictions and evaluation targets.

Every method receives detections only through the source image timestamp. The
target-time annotation projection is written to a separate evaluation table and
is never passed into propagation or tracker state.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

ROOT = Path(r"C:\work\auto")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from latency_compensation.causal_trackers import ByteTrackAdapter, OCSortAdapter
from recalculate_av2_sample import LogData, damping, image_velocity_prediction, kalman_prediction, track_history
from run_av2_corrected_propagation import coco_group, source_evaluation_association

AV2 = ROOT / "results" / "av2_confirmation"
OUTPUT = ROOT / "results" / "sivp_strengthening" / "end_to_end_low_level"
DELTAS = (100, 200, 300, 400, 500)


def atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def annotation_distance(log: LogData, annotation_timestamp_ns: int, track_uuid: str) -> float:
    frame = log.annotations_by_ts.get(int(annotation_timestamp_ns))
    if frame is None or frame.empty:
        return float("nan")
    selected = frame[frame.track_uuid.astype(str) == str(track_uuid)]
    if selected.empty:
        return float("nan")
    row = selected.iloc[0]
    return float(np.hypot(float(row.tx_m), float(row.ty_m)))


def process(detector: str, split: str, log_id: str, role: str, pairs: pd.DataFrame,
            category_mapping: dict[str, str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    checkpoint_path = (AV2 / "checkpoints" / detector / role / f"{log_id}.json"
                       if detector != "rtdetr_l" else OUTPUT.parent / "third_detector_checkpoints" / role / f"{log_id}.json")
    payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    detections = payload["detections"]
    by_timestamp: dict[int, list[tuple[int, dict]]] = {}
    for index, detection in enumerate(detections):
        by_timestamp.setdefault(int(detection["timestamp_ns"]), []).append((index, detection))
    pair_map = {(int(row.source_image_timestamp_ns), int(row.delta_ms)): row for row in pairs.itertuples(index=False)}
    log = LogData(split, log_id)
    byte, oc = ByteTrackAdapter(), OCSortAdapter()
    depth_cache: dict[int, tuple] = {}
    prediction_rows, target_rows = [], []

    for source_timestamp in sorted(by_timestamp):
        indexed = by_timestamp[source_timestamp]
        local_detections = [detection for _, detection in indexed]
        start = time.perf_counter_ns()
        byte_assigned = byte.update(source_timestamp, local_detections)
        byte_update_ms = (time.perf_counter_ns() - start) / 1e6 / max(len(indexed), 1)
        start = time.perf_counter_ns()
        oc_assigned = oc.update(source_timestamp, local_detections)
        oc_update_ms = (time.perf_counter_ns() - start) / 1e6 / max(len(indexed), 1)
        available = [delta for delta in DELTAS if (source_timestamp, delta) in pair_map]
        if not available:
            continue
        first_pair = pair_map[(source_timestamp, available[0])]
        source_annotation_timestamp = int(first_pair.source_timestamp_ns)
        source_targets = log.projected_targets(source_annotation_timestamp, source_timestamp)
        source_ids = {track_uuid for track_uuid, _, _ in source_targets}
        source_association = source_evaluation_association(indexed, source_targets, category_mapping)

        prepared = {}
        for local_index, (global_index, detection) in enumerate(indexed):
            class_id = int(detection["class_id"])
            source_box = np.asarray([detection["x1"], detection["y1"], detection["x2"], detection["y2"]], float)
            history = track_history(by_timestamp, source_timestamp, global_index, detection)
            if global_index not in depth_cache:
                depth_cache[global_index] = log.depth(source_box, source_timestamp, class_id)
            depth, dispersion, point_count, _, _ = depth_cache[global_index]
            city_positions = []
            for history_timestamp, history_index, history_detection in history:
                history_box = np.asarray([history_detection["x1"], history_detection["y1"],
                                          history_detection["x2"], history_detection["y2"]], float)
                if history_index not in depth_cache:
                    depth_cache[history_index] = log.depth(history_box, history_timestamp, class_id)
                history_depth = depth_cache[history_index][0]
                cx, cy = (history_box[0] + history_box[2]) / 2, (history_box[1] + history_box[3]) / 2
                camera = np.asarray([[(cx - log.K[0, 2]) / log.K[0, 0] * history_depth,
                                      (cy - log.K[1, 2]) / log.K[1, 1] * history_depth, history_depth]])
                city_positions.append(log.cam_points_to_city(camera, history_timestamp)[0])
            object_velocity = np.zeros(3)
            if len(city_positions) >= 2:
                dt = max((history[-1][0] - history[-2][0]) / 1e9, 1e-6)
                object_velocity = (city_positions[-1] - city_positions[-2]) / dt
                speed = float(np.linalg.norm(object_velocity[:2]))
                object_velocity *= min(1.0, 30.0 / max(speed, 1e-9))
            prepared[local_index] = (global_index, detection, source_box, history, depth, dispersion,
                                     point_count, object_velocity, damping(point_count, dispersion, len(history)))

        for delta_ms in available:
            pair = pair_map[(source_timestamp, delta_ms)]
            target_image_timestamp = int(pair.target_image_timestamp_ns)
            target_annotation_timestamp = int(pair.target_timestamp_ns)
            targets = log.projected_targets(target_annotation_timestamp, target_image_timestamp)
            target_ids = {track_uuid for track_uuid, _, _ in targets}
            source_pose, target_pose = log.pose(source_timestamp), log.pose(target_image_timestamp)
            yaw_source = Rotation.from_matrix(source_pose[:3, :3]).as_euler("xyz")[2]
            yaw_target = Rotation.from_matrix(target_pose[:3, :3]).as_euler("xyz")[2]
            yaw_change = float((yaw_target - yaw_source + np.pi) % (2 * np.pi) - np.pi)
            yaw_rate = yaw_change / (delta_ms / 1000)
            frame_id = f"{log_id}:{detector}:{source_timestamp}:{delta_ms}"
            for track_uuid, category, target_box in targets:
                target_rows.append({
                    "frame_id": frame_id, "log_id": log_id, "detector": detector,
                    "source_timestamp_ns": source_timestamp, "target_timestamp_ns": target_image_timestamp,
                    "target_annotation_timestamp_ns": target_annotation_timestamp, "delta_ms": delta_ms,
                    "yaw_rate_rad_s": yaw_rate, "target_track_uuid": track_uuid,
                    "target_category": category, "analysis_group": category_mapping.get(category, "other"),
                    "target_x1": target_box[0], "target_y1": target_box[1],
                    "target_x2": target_box[2], "target_y2": target_box[3],
                    "distance_m": annotation_distance(log, target_annotation_timestamp, track_uuid),
                    "persistent_object": track_uuid in source_ids, "new_object": track_uuid not in source_ids,
                    "evaluation_only": True,
                })
            for local_index, values in prepared.items():
                global_index, detection, source_box, history, depth, dispersion, point_count, object_velocity, weight = values
                source_track_uuid = source_association.get(global_index)
                start = time.perf_counter_ns(); t0 = source_box.copy(); runtime_t0 = (time.perf_counter_ns() - start) / 1e6
                start = time.perf_counter_ns(); t1, _ = image_velocity_prediction(history, target_image_timestamp); runtime_t1 = (time.perf_counter_ns() - start) / 1e6
                start = time.perf_counter_ns(); t2, _ = kalman_prediction(history, target_image_timestamp); runtime_t2 = (time.perf_counter_ns() - start) / 1e6
                start = time.perf_counter_ns(); byte_track = byte_assigned.get(local_index); t3 = source_box.copy() if byte_track is None else byte_track.forecast(target_image_timestamp); runtime_t3 = byte_update_ms + (time.perf_counter_ns() - start) / 1e6
                start = time.perf_counter_ns(); oc_track = oc_assigned.get(local_index); t4 = source_box.copy() if oc_track is None else oc_track.forecast(target_image_timestamp); runtime_t4 = oc_update_ms + (time.perf_counter_ns() - start) / 1e6
                start = time.perf_counter_ns(); t5 = log.propagate_geometry(source_box, depth, source_timestamp, target_image_timestamp, np.zeros(3)); runtime_t5 = (time.perf_counter_ns() - start) / 1e6
                start = time.perf_counter_ns(); t6 = log.propagate_geometry(source_box, depth, source_timestamp, target_image_timestamp, object_velocity * (delta_ms / 1000) * weight); runtime_t6 = (time.perf_counter_ns() - start) / 1e6
                for model, prediction, runtime_ms in (("T0", t0, runtime_t0), ("T1", t1, runtime_t1),
                                                       ("T2", t2, runtime_t2), ("T3", t3, runtime_t3),
                                                       ("T4", t4, runtime_t4), ("T5", t5, runtime_t5),
                                                       ("T6", t6, runtime_t6)):
                    prediction_rows.append({
                        "prediction_id": f"{frame_id}:{global_index}:{model}", "frame_id": frame_id,
                        "log_id": log_id, "detector": detector, "model": model,
                        "source_timestamp_ns": source_timestamp, "target_timestamp_ns": target_image_timestamp,
                        "delta_ms": delta_ms, "yaw_rate_rad_s": yaw_rate, "detection_index": global_index,
                        "confidence": float(detection["confidence"]), "class_id": int(detection["class_id"]),
                        "analysis_group": coco_group(int(detection["class_id"])),
                        "source_track_uuid": source_track_uuid,
                        "source_object_persistent": source_track_uuid in target_ids if source_track_uuid else False,
                        "source_object_disappeared": source_track_uuid not in target_ids if source_track_uuid else False,
                        "pred_x1": prediction[0], "pred_y1": prediction[1],
                        "pred_x2": prediction[2], "pred_y2": prediction[3],
                        "runtime_ms": runtime_ms, "track_age": len(history),
                        "lidar_point_count": point_count, "depth_dispersion_m": dispersion,
                        "future_detection_input": False, "future_geometry_input": False,
                        "association_history_max_timestamp_ns": source_timestamp,
                    })
    return pd.DataFrame(prediction_rows), pd.DataFrame(target_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--detector", choices=["yolo11n", "yolo11s", "rtdetr_l"])
    parser.add_argument("--role", choices=["development", "model_selection", "heldout"], default="heldout")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--log-limit", type=int)
    args = parser.parse_args()
    filename = {"development": "av2_development_logs.txt", "model_selection": "av2_model_selection_logs.txt", "heldout": "av2_heldout_logs.txt"}[args.role]
    wanted = [line.strip() for line in (ROOT / "configs" / filename).read_text().splitlines() if line.strip()]
    metadata = pd.read_csv(AV2 / "cohort_source_metadata.csv")
    split_by_log = {str(row.log_id): str(row.split) for row in metadata.itertuples(index=False)}
    wanted = [value for index, value in enumerate(wanted) if index % args.shard_count == args.shard_index]
    if args.log_limit:
        wanted = wanted[: args.log_limit]
    pairs = pd.read_parquet(AV2 / "latency_pairs.parquet")
    pairs["source_image_timestamp_ns"] = pairs.source_image_path.map(lambda value: int(Path(value).stem))
    pairs["target_image_timestamp_ns"] = pairs.target_image_path.map(lambda value: int(Path(value).stem))
    category_mapping = pd.read_csv(AV2 / "class_mapping.csv").set_index("av2_category").analysis_group.to_dict()
    detectors = [args.detector] if args.detector else ["yolo11n", "yolo11s"]
    for detector in detectors:
        for log_id in wanted:
            prediction_path = OUTPUT / args.role / detector / f"{log_id}.predictions.parquet"
            target_path = OUTPUT / args.role / detector / f"{log_id}.targets.parquet"
            if prediction_path.exists() and target_path.exists():
                print(json.dumps({"detector": detector, "log": log_id, "reused": True}))
                continue
            log_pairs = pairs[pairs.log_id.astype(str) == log_id]
            predictions, targets = process(detector, split_by_log[log_id], log_id, args.role, log_pairs, category_mapping)
            atomic_parquet(prediction_path, predictions)
            atomic_parquet(target_path, targets)
            print(json.dumps({"detector": detector, "log": log_id, "predictions": len(predictions),
                              "targets": len(targets), "reused": False}))


if __name__ == "__main__":
    main()
