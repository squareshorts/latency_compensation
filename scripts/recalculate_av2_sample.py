"""Independent AV2 row-level recalculation for the invalid-run audit.

This deliberately does not call ``run_confirmation.propagate_log``.  It uses
preserved detector boxes plus raw AV2 calibration, poses, lidar and annotations
to recompute a deterministic 1,000-comparison held-out sample.
"""

from __future__ import annotations

import json
import math
import platform
import sys
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation


ROOT = Path(r"C:\work\auto")
OUT = ROOT / "results" / "av2_confirmation"
DATA = ROOT / "data" / "av2" / "sensor"
CAM = "ring_front_center"
DELTAS = (100, 200, 300, 400, 500)
MODELS = ("B0", "B1", "B2", "B3", "B4", "B5")
TOLERANCE = 1e-6


def pose_matrix(row) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat([row.qx, row.qy, row.qz, row.qw]).as_matrix()
    matrix[:3, 3] = [row.tx_m, row.ty_m, row.tz_m]
    return matrix


def box_state(box: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = box
    return np.asarray([(x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, 2.0), max(y2 - y1, 2.0)])


def state_box(state: np.ndarray, width: int = 1550, height: int = 2048) -> np.ndarray:
    cx, cy, w, h = state
    return np.asarray([np.clip(cx - w / 2, 0, width - 1), np.clip(cy - h / 2, 0, height - 1),
                       np.clip(cx + w / 2, 0, width - 1), np.clip(cy + h / 2, 0, height - 1)])


def iou(a: np.ndarray, b: np.ndarray) -> float:
    ix1, iy1 = np.maximum(a[:2], b[:2]); ix2, iy2 = np.minimum(a[2:], b[2:])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / max(area_a + area_b - inter, 1e-12)


def center_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(box_state(a)[:2] - box_state(b)[:2]))


def json_array(value) -> str:
    return json.dumps(np.asarray(value, float).tolist(), separators=(",", ":"))


def coco_group(class_id: int) -> str:
    if class_id in {2, 5, 7}: return "vehicle"
    if class_id == 0: return "pedestrian_cyclist"
    return "other"


class LogData:
    def __init__(self, split: str, log_id: str):
        self.split, self.log_id = split, log_id
        self.root = DATA / split / log_id
        poses = pd.read_feather(self.root / "city_SE3_egovehicle.feather")
        self.poses = {int(row.timestamp_ns): pose_matrix(row) for row in poses.itertuples(index=False)}
        intr = pd.read_feather(self.root / "calibration" / "intrinsics.feather").set_index("sensor_name").loc[CAM]
        self.width, self.height = int(intr.width_px), int(intr.height_px)
        self.K = np.asarray([[intr.fx_px, 0, intr.cx_px], [0, intr.fy_px, intr.cy_px], [0, 0, 1]], float)
        ext = pd.read_feather(self.root / "calibration" / "egovehicle_SE3_sensor.feather").set_index("sensor_name").loc[CAM]
        self.ego_from_cam = pose_matrix(ext)
        self.cam_from_ego = np.linalg.inv(self.ego_from_cam)
        self.annotations = pd.read_feather(self.root / "annotations.feather")
        self.annotations_by_ts = {int(ts): g for ts, g in self.annotations.groupby("timestamp_ns")}
        self.lidar_paths = sorted((self.root / "sensors" / "lidar").glob("*.feather"))
        self.lidar_timestamps = np.asarray([int(path.stem) for path in self.lidar_paths], dtype=np.int64)
        self._projected_lidar: dict[int, tuple[np.ndarray, np.ndarray, int]] = {}
        self._target_cache: dict[tuple[int, int], list[tuple[str, str, np.ndarray]]] = {}

    def pose(self, timestamp: int) -> np.ndarray:
        return self.poses[int(timestamp)]

    def projected_lidar(self, camera_timestamp: int) -> tuple[np.ndarray, np.ndarray, int]:
        camera_timestamp = int(camera_timestamp)
        if camera_timestamp in self._projected_lidar:
            return self._projected_lidar[camera_timestamp]
        index = int(np.argmin(np.abs(self.lidar_timestamps - camera_timestamp)))
        lidar_timestamp = int(self.lidar_timestamps[index])
        points = pd.read_feather(self.lidar_paths[index])[["x", "y", "z"]].to_numpy(float)
        transform = self.cam_from_ego @ np.linalg.inv(self.pose(camera_timestamp)) @ self.pose(lidar_timestamp)
        points_h = np.c_[points, np.ones(len(points))]
        camera = (transform @ points_h.T).T[:, :3]
        positive = camera[:, 2] > 0.1
        camera = camera[positive]
        uvw = (self.K @ camera.T).T
        uv = uvw[:, :2] / uvw[:, 2:3]
        valid = (uv[:, 0] >= 0) & (uv[:, 0] < self.width) & (uv[:, 1] >= 0) & (uv[:, 1] < self.height)
        result = (uv[valid], camera[valid, 2], lidar_timestamp)
        self._projected_lidar[camera_timestamp] = result
        return result

    def depth(self, box: np.ndarray, timestamp: int, class_id: int) -> tuple[float, float, int, int, str]:
        uv, z, lidar_timestamp = self.projected_lidar(timestamp)
        inside = (uv[:, 0] >= box[0]) & (uv[:, 0] <= box[2]) & (uv[:, 1] >= box[1]) & (uv[:, 1] <= box[3])
        values = z[inside]
        if len(values):
            median = float(np.median(values)); dispersion = float(1.4826 * np.median(np.abs(values - median)))
            return median, dispersion, int(len(values)), lidar_timestamp, "source_lidar_median"
        typical_height = {0: 1.7, 1: 1.5, 2: 1.5, 3: 1.5, 5: 3.0, 7: 3.0}.get(class_id, 1.7)
        fallback = float(self.K[1, 1] * typical_height / max(box[3] - box[1], 2.0))
        return float(np.clip(fallback, 2.0, 120.0)), np.nan, 0, lidar_timestamp, "geometry_fallback_no_points"

    def cam_points_to_city(self, points: np.ndarray, timestamp: int) -> np.ndarray:
        return (self.pose(timestamp) @ self.ego_from_cam @ np.c_[points, np.ones(len(points))].T).T[:, :3]

    def city_points_to_cam(self, points: np.ndarray, timestamp: int) -> np.ndarray:
        return (self.cam_from_ego @ np.linalg.inv(self.pose(timestamp)) @ np.c_[points, np.ones(len(points))].T).T[:, :3]

    def backproject_box(self, box: np.ndarray, depth: float) -> np.ndarray:
        pixels = np.asarray([[box[0], box[1]], [box[2], box[1]], [box[2], box[3]], [box[0], box[3]]], float)
        x = (pixels[:, 0] - self.K[0, 2]) / self.K[0, 0] * depth
        y = (pixels[:, 1] - self.K[1, 2]) / self.K[1, 1] * depth
        return np.c_[x, y, np.full(4, depth)]

    def project_camera_box(self, camera_points: np.ndarray) -> np.ndarray | None:
        valid = camera_points[:, 2] > 0.1
        if valid.sum() < 2: return None
        uvw = (self.K @ camera_points[valid].T).T
        uv = uvw[:, :2] / uvw[:, 2:3]
        return np.asarray([np.clip(uv[:, 0].min(), 0, self.width - 1), np.clip(uv[:, 1].min(), 0, self.height - 1),
                           np.clip(uv[:, 0].max(), 0, self.width - 1), np.clip(uv[:, 1].max(), 0, self.height - 1)])

    def propagate_geometry(self, box: np.ndarray, depth: float, source_ts: int, target_ts: int, displacement_city: np.ndarray) -> np.ndarray:
        city = self.cam_points_to_city(self.backproject_box(box, depth), source_ts) + displacement_city
        projected = self.project_camera_box(self.city_points_to_cam(city, target_ts))
        return box.copy() if projected is None else projected

    def projected_targets(self, annotation_ts: int, image_ts: int) -> list[tuple[str, str, np.ndarray]]:
        key = (int(annotation_ts), int(image_ts))
        if key in self._target_cache: return self._target_cache[key]
        output = []
        unit = np.asarray([[1,1,1],[1,-1,1],[1,-1,-1],[1,1,-1],[-1,1,1],[-1,-1,1],[-1,-1,-1],[-1,1,-1]], float)
        for row in self.annotations_by_ts.get(int(annotation_ts), pd.DataFrame()).itertuples(index=False):
            rotation = Rotation.from_quat([row.qx, row.qy, row.qz, row.qw]).as_matrix()
            vertices_ego = (unit * np.asarray([row.length_m, row.width_m, row.height_m]) / 2) @ rotation.T + np.asarray([row.tx_m, row.ty_m, row.tz_m])
            city = (self.pose(annotation_ts) @ np.c_[vertices_ego, np.ones(8)].T).T[:, :3]
            box = self.project_camera_box(self.city_points_to_cam(city, image_ts))
            if box is not None and box[2] > box[0] and box[3] > box[1]:
                output.append((str(row.track_uuid), str(row.category), box))
        self._target_cache[key] = output
        return output


def track_history(detections_by_ts: dict[int, list[tuple[int, dict]]], timestamp: int, detection_index: int, detection: dict) -> list[tuple[int, int, dict]]:
    current_box = np.asarray([detection["x1"], detection["y1"], detection["x2"], detection["y2"]], float)
    history = [(timestamp, detection_index, detection)]
    for previous_ts in sorted((value for value in detections_by_ts if value < timestamp and timestamp - value <= 600_000_000), reverse=True):
        candidates = [(idx, row) for idx, row in detections_by_ts[previous_ts] if int(row["class_id"]) == int(detection["class_id"])]
        if not candidates: continue
        scored = []
        for idx, row in candidates:
            box = np.asarray([row["x1"], row["y1"], row["x2"], row["y2"]], float)
            scored.append((iou(current_box, box), center_distance(current_box, box) / math.hypot(1550, 2048), idx, row, box))
        best = max(scored, key=lambda value: (value[0], -value[1]))
        if best[0] < 0.10 and best[1] > 0.15: continue
        history.append((previous_ts, best[2], best[3])); current_box = best[4]
        if len(history) == 4: break
    return list(reversed(history))


def image_velocity_prediction(history: list[tuple[int, int, dict]], target_ts: int) -> tuple[np.ndarray, np.ndarray]:
    current = history[-1][2]
    current_box = np.asarray([current["x1"], current["y1"], current["x2"], current["y2"]], float)
    if len(history) < 2: return current_box, np.zeros(4)
    previous = history[-2]
    previous_box = np.asarray([previous[2]["x1"], previous[2]["y1"], previous[2]["x2"], previous[2]["y2"]], float)
    dt = (history[-1][0] - previous[0]) / 1e9
    velocity = (box_state(current_box) - box_state(previous_box)) / max(dt, 1e-6)
    horizon = (target_ts - history[-1][0]) / 1e9
    return state_box(box_state(current_box) + velocity * horizon), velocity


def kalman_prediction(history: list[tuple[int, int, dict]], target_ts: int) -> tuple[np.ndarray, np.ndarray]:
    x = None; p = np.eye(8) * 100.0; previous_ts = None
    for timestamp, _, row in history:
        box = np.asarray([row["x1"], row["y1"], row["x2"], row["y2"]], float); z = box_state(box)
        if x is None: x = np.r_[z, np.zeros(4)]; previous_ts = timestamp; continue
        dt = (timestamp - previous_ts) / 1e9; previous_ts = timestamp
        f = np.eye(8); f[:4, 4:] = np.eye(4) * dt
        x = f @ x; p = f @ p @ f.T + np.eye(8) * 4.0
        h = np.c_[np.eye(4), np.zeros((4,4))]; s = h @ p @ h.T + np.eye(4) * 9.0
        gain = p @ h.T @ np.linalg.inv(s); x += gain @ (z - h @ x); p = (np.eye(8) - gain @ h) @ p
    horizon = (target_ts - history[-1][0]) / 1e9
    predicted = x[:4] + x[4:] * horizon
    return state_box(predicted), x[4:].copy()


def original_stored_predictions(payload: dict) -> list[dict[int, dict[str, np.ndarray]]]:
    history = {}; result = []
    for row in payload["detections"]:
        box = np.asarray([row["x1"], row["y1"], row["x2"], row["y2"]], float)
        past = history.setdefault(int(row["class_id"]), []); by_delta = {}
        for delta in DELTAS:
            velocity = box.copy() if not past else box + (box - past[-1]) * delta / 100.0
            by_delta[delta] = {"B0": box.copy(), "B1": velocity.copy(), "B2": velocity.copy(), "B3": box.copy(), "B4": velocity.copy(), "B5": box + 0.65 * (velocity - box)}
        result.append(by_delta); past.append(box); history[int(row["class_id"])] = past[-4:]
    return result


def damping(point_count: int, dispersion: float, age: int) -> float:
    lidar = min(1.0, point_count / 10.0); depth = 0.0 if np.isnan(dispersion) else 1.0 / (1.0 + max(0.0, dispersion) / 5.0); track = min(1.0, age / 3.0)
    return float(np.clip(lidar * depth * track, 0, 1))


def main() -> None:
    heldout = {line.strip() for line in (ROOT / "configs" / "av2_heldout_logs.txt").read_text().splitlines() if line.strip()}
    metadata = pd.read_csv(OUT / "cohort_source_metadata.csv")
    selected_meta = metadata[metadata.log_id.astype(str).isin(heldout)].sort_values(["p75_yaw_rate_rad_s", "log_id"])
    selected = pd.concat([selected_meta.head(5), selected_meta.tail(5)]).drop_duplicates("log_id")
    motion_group = {str(row.log_id): ("low_motion" if index < 5 else "high_motion") for index, row in enumerate(selected.itertuples(index=False))}
    split_map = {str(row.log_id): str(row.split) for row in selected.itertuples(index=False)}
    pairs = pd.read_parquet(OUT / "latency_pairs.parquet")
    pairs["source_image_timestamp_ns"] = pairs.source_image_path.map(lambda value: int(Path(value).stem))
    pairs["target_image_timestamp_ns"] = pairs.target_image_path.map(lambda value: int(Path(value).stem))
    pair_lookup = pairs[pairs.log_id.astype(str).isin(split_map)].set_index(["log_id", "source_image_timestamp_ns", "delta_ms"])
    rows = []
    for log_id in selected.log_id.astype(str):
        log = LogData(split_map[log_id], log_id)
        for detector in ("yolo11n", "yolo11s"):
            checkpoint = OUT / "checkpoints" / detector / "heldout" / f"{log_id}.json"
            payload = json.loads(checkpoint.read_text(encoding="utf-8")); detections = payload["detections"]
            by_ts = defaultdict(list)
            for index, detection in enumerate(detections): by_ts[int(detection["timestamp_ns"])].append((index, detection))
            stored = original_stored_predictions(payload)
            candidates = []
            for index, detection in enumerate(detections):
                timestamp = int(detection["timestamp_ns"]); box = np.asarray([detection["x1"], detection["y1"], detection["x2"], detection["y2"]], float)
                if (box[2]-box[0])*(box[3]-box[1]) < 400: continue
                if all((log_id, timestamp, delta) in pair_lookup.index for delta in DELTAS): candidates.append((index, detection))
            if not candidates: raise RuntimeError(f"No candidates: {log_id} {detector}")
            positions = np.linspace(0, len(candidates)-1, min(80, len(candidates)), dtype=int)
            accepted = 0
            for position in positions:
                index, detection = candidates[int(position)]; source_ts = int(detection["timestamp_ns"]); class_id = int(detection["class_id"])
                source_box = np.asarray([detection["x1"], detection["y1"], detection["x2"], detection["y2"]], float)
                history = track_history(by_ts, source_ts, index, detection)
                depth, dispersion, point_count, lidar_ts, depth_source = log.depth(source_box, source_ts, class_id)
                city_positions = []
                for hist_ts, _, hist_detection in history:
                    hist_box = np.asarray([hist_detection["x1"], hist_detection["y1"], hist_detection["x2"], hist_detection["y2"]], float)
                    hist_depth, _, _, _, _ = log.depth(hist_box, hist_ts, class_id)
                    center = np.asarray([[(hist_box[0]+hist_box[2])/2, (hist_box[1]+hist_box[3])/2, 1.0]])
                    cam = np.asarray([[(center[0,0]-log.K[0,2])/log.K[0,0]*hist_depth, (center[0,1]-log.K[1,2])/log.K[1,1]*hist_depth, hist_depth]])
                    city_positions.append(log.cam_points_to_city(cam, hist_ts)[0])
                object_velocity = np.zeros(3)
                if len(city_positions) >= 2:
                    dt = (history[-1][0] - history[-2][0]) / 1e9
                    object_velocity = (city_positions[-1] - city_positions[-2]) / max(dt, 1e-6)
                    speed = np.linalg.norm(object_velocity[:2])
                    if speed > 30: object_velocity *= 30 / speed
                weight = damping(point_count, dispersion, len(history))
                for delta in DELTAS:
                    pair = pair_lookup.loc[(log_id, source_ts, delta)]
                    if isinstance(pair, pd.DataFrame): pair = pair.iloc[0]
                    target_ts = int(pair.target_image_timestamp_ns); annotation_ts = int(pair.target_timestamp_ns)
                    targets = log.projected_targets(annotation_ts, target_ts)
                    same_group = [target for target in targets if pd.read_csv(OUT / "class_mapping.csv").set_index("av2_category").analysis_group.to_dict().get(target[1], "other") == coco_group(class_id)]
                    pool = same_group or targets
                    if not pool: continue
                    target_id, target_category, target_box = min(pool, key=lambda target: center_distance(source_box, target[2]))
                    b1, image_velocity = image_velocity_prediction(history, target_ts)
                    b2, kalman_velocity = kalman_prediction(history, target_ts)
                    b3 = log.propagate_geometry(source_box, depth, source_ts, target_ts, np.zeros(3))
                    displacement = object_velocity * (delta / 1000.0)
                    b4 = log.propagate_geometry(source_box, depth, source_ts, target_ts, displacement)
                    b5 = log.propagate_geometry(source_box, depth, source_ts, target_ts, displacement * weight)
                    recomputed = {"B0": source_box.copy(), "B1": b1, "B2": b2, "B3": b3, "B4": b4, "B5": b5}
                    source_pose, target_pose = log.pose(source_ts), log.pose(target_ts)
                    relative = log.cam_from_ego @ np.linalg.inv(target_pose) @ source_pose @ log.ego_from_cam
                    yaw_s = Rotation.from_matrix(source_pose[:3,:3]).as_euler("xyz")[2]; yaw_t = Rotation.from_matrix(target_pose[:3,:3]).as_euler("xyz")[2]
                    yaw_change = float((yaw_t-yaw_s+np.pi)%(2*np.pi)-np.pi)
                    record = {
                        "comparison_id": f"{log_id}:{detector}:{source_ts}:{index}:{delta}", "log_id": log_id, "detector": detector,
                        "motion_group": motion_group[log_id], "delta_ms": delta, "source_timestamp_ns": source_ts,
                        "target_timestamp_ns": target_ts, "target_annotation_timestamp_ns": annotation_ts,
                        "source_detector_box_json": json_array(source_box), "target_annotation_box_json": json_array(target_box),
                        "target_track_uuid": target_id, "target_category": target_category,
                        "source_ego_pose_json": json_array(source_pose), "target_ego_pose_json": json_array(target_pose),
                        "relative_camera_transform_json": json_array(relative), "estimated_depth_m": depth,
                        "depth_dispersion_m": dispersion, "lidar_point_count": point_count, "lidar_timestamp_ns": lidar_ts,
                        "depth_source": depth_source, "track_history_json": json.dumps([{"timestamp_ns": t, "detector_index": i, "box": [d['x1'],d['y1'],d['x2'],d['y2']]} for t,i,d in history], separators=(",",":")),
                        "track_age": len(history), "estimated_image_velocity_json": json_array(image_velocity),
                        "estimated_image_velocity_norm_px_s": float(np.linalg.norm(image_velocity[:2])),
                        "estimated_kalman_velocity_json": json_array(kalman_velocity),
                        "estimated_kalman_velocity_norm_px_s": float(np.linalg.norm(kalman_velocity[:2])),
                        "estimated_object_velocity_json": json_array(object_velocity),
                        "estimated_object_velocity_norm_m_s": float(np.linalg.norm(object_velocity[:2])),
                        "uncertainty_damping_weight": weight, "ego_translation_m": float(np.linalg.norm(target_pose[:3,3]-source_pose[:3,3])),
                        "ego_yaw_change_rad": yaw_change, "ego_yaw_rate_rad_s": yaw_change/(delta/1000),
                        "ego_predicted_pixel_displacement": center_distance(source_box, b3),
                    }
                    for model in MODELS:
                        stored_box = stored[index][delta][model]; recomputed_box = recomputed[model]; diff = np.abs(stored_box-recomputed_box)
                        record[f"stored_{model}_box_json"] = json_array(stored_box); record[f"recomputed_{model}_box_json"] = json_array(recomputed_box)
                        record[f"max_abs_difference_{model}_px"] = float(diff.max())
                    rows.append(record)
                accepted += 1
                if accepted == 10: break
            if accepted < 10: raise RuntimeError(f"Only {accepted} accepted sources: {log_id} {detector}")
    result = pd.DataFrame(rows)
    if len(result) < 1000: raise RuntimeError(f"Sample too small: {len(result)}")
    result.to_parquet(OUT / "row_level_recalculation.parquet", index=False)
    summary = []
    for model in MODELS:
        difference = result[f"max_abs_difference_{model}_px"]
        summary.append({"model": model, "comparisons": len(result), "tolerance_px": TOLERANCE,
                        "agreements_within_tolerance": int((difference <= TOLERANCE).sum()),
                        "agreement_percent": float((difference <= TOLERANCE).mean()*100),
                        "median_max_abs_difference_px": float(difference.median()), "maximum_abs_difference_px": float(difference.max()),
                        "stored_output_valid": bool((difference <= TOLERANCE).all())})
    pd.DataFrame(summary).to_csv(OUT / "row_level_recalculation_summary.csv", index=False, float_format="%.12g")
    metrics = {
        "ego_translation_m": result.ego_translation_m, "ego_yaw_change_rad": result.ego_yaw_change_rad.abs(),
        "ego_predicted_pixel_displacement": result.ego_predicted_pixel_displacement,
        "historical_image_velocity_norm_px_s": result.estimated_image_velocity_norm_px_s,
        "kalman_velocity_norm_px_s": result.estimated_kalman_velocity_norm_px_s,
        "object_motion_estimate_norm_m_s": result.estimated_object_velocity_norm_m_s,
        "valid_depth_indicator": (result.lidar_point_count > 0).astype(float), "valid_depth_m": result.estimated_depth_m,
        "track_age": result.track_age.astype(float), "uncertainty_damping_weight": result.uncertainty_damping_weight,
    }
    distribution, audit = [], []
    for name, series in metrics.items():
        values = pd.to_numeric(series, errors="coerce"); nonnull = values.dropna()
        distribution.append({"metric": name, "count": len(values), "nonnull": len(nonnull), "null_percent": values.isna().mean()*100,
                             "zero_percent": (nonnull==0).mean()*100 if len(nonnull) else np.nan, "unique_values": nonnull.nunique(),
                             "minimum": nonnull.min(), "p05": nonnull.quantile(.05), "median": nonnull.median(), "p95": nonnull.quantile(.95), "maximum": nonnull.max()})
        audit.append({"metric": name, "all_null": len(nonnull)==0, "all_zero": bool(len(nonnull) and (nonnull==0).all()),
                      "constant": nonnull.nunique() <= 1, "nontrivial": bool(nonnull.nunique()>1 and (nonnull!=0).any()),
                      "sample_scope": "1000 deterministic heldout comparisons; both detectors; 10 logs; all five horizons"})
    pd.DataFrame(audit).to_csv(OUT / "motion_input_audit.csv", index=False)
    pd.DataFrame(distribution).to_csv(OUT / "motion_input_distributions.csv", index=False, float_format="%.12g")
    print(json.dumps({"comparisons": len(result), "logs": result.log_id.nunique(), "detectors": result.detector.nunique(),
                      "all_horizons": sorted(result.delta_ms.unique().tolist()), "B4_agreement_percent": summary[4]["agreement_percent"]}))


if __name__ == "__main__":
    main()
