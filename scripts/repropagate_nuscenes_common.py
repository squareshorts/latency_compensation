"""Repropagate nuScenes detector boxes with the AV2 B3/B5 definitions.

This uses preserved detector outputs. It does not run detector inference. GT
instance tokens remain evaluation-only and are not used for motion history.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pyquaternion import Quaternion

ROOT = Path(r"C:\work\auto")
SDK = ROOT / "external" / "nuscenes-devkit" / "python-sdk"
sys.path.insert(0, str(SDK))
from nuscenes.nuscenes import NuScenes

ARCHIVE = ROOT / "archive" / "negative_routes" / "dual_loop_visual_residual"
sys.path.insert(0, str(ARCHIVE))
from run_dual_loop_500ms import box_metrics, box_plane_points, corners_bbox, iou, oracle_camera_transform, project_points

SOURCE = ROOT / "results" / "nuscenes_500ms_reproduction" / "propagated_boxes.parquet"
OUT = ROOT / "results" / "object_motion_harm" / "nuscenes_common_b3_b5.parquet"


def center_distance(a, b):
    ac = np.asarray([(a[0]+a[2])/2, (a[1]+a[3])/2]); bc = np.asarray([(b[0]+b[2])/2, (b[1]+b[3])/2])
    return float(np.linalg.norm(ac-bc))


def damping(points: int, dispersion: float, age: int) -> float:
    lidar = min(1.0, points / 10.0)
    depth = 0.0 if not np.isfinite(dispersion) else 1.0 / (1.0 + max(0.0, dispersion) / 5.0)
    return float(np.clip(lidar * depth * min(1.0, age / 3.0), 0, 1))


def camera_center(box, depth, intrinsic):
    x = (box[0]+box[2])/2; y = (box[1]+box[3])/2
    return np.linalg.inv(intrinsic) @ np.asarray([x, y, 1.0]) * depth


def camera_to_global(point, pose, calibration):
    r_ce = Quaternion(calibration["rotation"]).rotation_matrix; t_ce = np.asarray(calibration["translation"], float)
    r_pose = Quaternion(pose["rotation"]).rotation_matrix; t_pose = np.asarray(pose["translation"], float)
    return r_pose @ (r_ce @ point + t_ce) + t_pose


def global_displacement_to_target_camera(displacement, target_pose, calibration):
    r_ce = Quaternion(calibration["rotation"]).rotation_matrix
    r_target = Quaternion(target_pose["rotation"]).rotation_matrix
    return r_ce.T @ r_target.T @ displacement


def main() -> None:
    raw = pd.read_parquet(SOURCE)
    rows = raw[raw.model == "B0_stale"].drop_duplicates("record_id").sort_values(["scene_name", "source_timestamp_us", "record_id"]).copy()
    nusc = NuScenes(version="v1.0-mini", dataroot=str(ROOT / "data" / "nuscenes"), verbose=False)
    history_by_scene: dict[str, list] = {}
    output = []
    for scene, scene_rows in rows.groupby("scene_name", sort=True):
        processed = []
        for row in scene_rows.itertuples(index=False):
            source_box = np.asarray(json.loads(row.source_box_json), float)
            target_box = np.asarray(json.loads(row.target_annotation_box_json), float)
            source_sd = nusc.get("sample_data", row.source_frame_token); target_sd = nusc.get("sample_data", row.target_frame_token)
            calibration = nusc.get("calibrated_sensor", source_sd["calibrated_sensor_token"])
            intrinsic = np.asarray(calibration["camera_intrinsic"], float)
            source_pose = nusc.get("ego_pose", source_sd["ego_pose_token"]); target_pose = nusc.get("ego_pose", target_sd["ego_pose_token"])
            a, b = oracle_camera_transform(source_pose, target_pose, calibration)
            source_points = box_plane_points(source_box, float(row.depth_m), np.linalg.inv(intrinsic))
            b3_points = (a @ source_points.T).T + b
            b3 = np.asarray(corners_bbox(project_points(b3_points, intrinsic)), float)
            if not np.isfinite(b3).all(): b3 = source_box.copy()

            current = {"timestamp": int(row.source_timestamp_us), "box": source_box, "class": str(row.class_name),
                       "depth": float(row.depth_m), "pose": source_pose, "calibration": calibration, "intrinsic": intrinsic}
            track = [current]; current_box = source_box.copy(); last_overlap = np.nan; last_distance = np.nan
            prior_times = sorted({item["timestamp"] for item in processed if 0 < current["timestamp"] - item["timestamp"] <= 600_000}, reverse=True)
            for timestamp in prior_times:
                candidates = [item for item in processed if item["timestamp"] == timestamp and item["class"] == current["class"]]
                if not candidates: continue
                scored = [(iou(current_box, item["box"]), center_distance(current_box, item["box"]) / math.hypot(1600,900), item) for item in candidates]
                overlap, distance, best = max(scored, key=lambda value: (value[0], -value[1]))
                if overlap < 0.10 and distance > 0.15: continue
                track.append(best); current_box = best["box"]
                last_overlap, last_distance = overlap, distance
                if len(track) == 4: break
            track = list(reversed(track))
            positions = [camera_to_global(camera_center(item["box"], item["depth"], item["intrinsic"]), item["pose"], item["calibration"]) for item in track]
            velocity = np.zeros(3); velocity_dispersion = np.nan
            if len(track) >= 2:
                dt = (track[-1]["timestamp"] - track[-2]["timestamp"]) / 1e6
                velocity = (positions[-1] - positions[-2]) / max(dt, 1e-6)
                speed = np.linalg.norm(velocity[:2])
                if speed > 30: velocity *= 30/speed
                intervals = []
                for left, right, p0, p1 in zip(track[:-1], track[1:], positions[:-1], positions[1:]):
                    dt_i = (right["timestamp"] - left["timestamp"]) / 1e6
                    if dt_i > 0: intervals.append((p1-p0)/dt_i)
                if len(intervals) > 1:
                    values = np.asarray(intervals)
                    velocity_dispersion = float(np.sqrt(np.mean(np.sum((values[:,:2]-np.median(values[:,:2],axis=0))**2,axis=1))))
                elif intervals:
                    velocity_dispersion = 0.0
            weight = damping(int(row.lidar_points), float(row.depth_uncertainty_m), len(track))
            elapsed = (int(row.target_timestamp_us) - int(row.source_timestamp_us)) / 1e6
            object_delta_camera = global_displacement_to_target_camera(velocity * elapsed * weight, target_pose, calibration)
            b5 = np.asarray(corners_bbox(project_points(b3_points + object_delta_camera, intrinsic)), float)
            if not np.isfinite(b5).all(): b5 = b3.copy()
            source_sample = nusc.get("sample", source_sd["sample_token"]); target_sample = nusc.get("sample", target_sd["sample_token"])
            def instance_translation(sample):
                for token in sample["anns"]:
                    ann = nusc.get("sample_annotation", token)
                    if ann["instance_token"] == row.instance_token: return np.asarray(ann["translation"], float)
                return None
            source_translation, target_translation = instance_translation(source_sample), instance_translation(target_sample)
            true_speed = np.nan if source_translation is None or target_translation is None else float(np.linalg.norm((target_translation-source_translation)[:2]) / max(elapsed,1e-6))
            ego_speed = float(np.linalg.norm((np.asarray(target_pose["translation"])-np.asarray(source_pose["translation"]))[:2]) / max(elapsed,1e-6))
            association_confidence = float(np.clip((0 if np.isnan(last_overlap) else last_overlap) * math.exp(-5*(0 if np.isnan(last_distance) else last_distance)),0,1))
            common_group = "pedestrian_cyclist" if row.class_name in {"person","bicycle","motorcycle"} else "vehicle"
            base = {"scene_name": scene, "record_id": row.record_id, "source_frame_token": row.source_frame_token,
                    "source_timestamp_us": int(row.source_timestamp_us), "target_timestamp_us": int(row.target_timestamp_us),
                    "latency_ms": 500, "detector": "yolo11n", "class_name": row.class_name, "object_group": row.object_group,
                    "confidence": float(row.confidence), "depth_m": float(row.depth_m), "depth_uncertainty_m": float(row.depth_uncertainty_m),
                    "lidar_points": int(row.lidar_points), "yaw_rate_rad_s": float(row.yaw_rate_deg_s) * math.pi / 180,
                    "track_age": len(track), "estimated_speed_m_s": float(np.linalg.norm(velocity[:2])), "true_speed_m_s": true_speed,
                    "velocity_dispersion_m_s": velocity_dispersion, "association_confidence": association_confidence,
                    "ego_speed_m_s": ego_speed, "common_object_group": common_group, "damping_weight": weight,
                    "future_geometry_input": False}
            for model, prediction in (("B3", b3), ("B5", b5)):
                output.append({**base, "model": model, "predicted_box_json": json.dumps(prediction.tolist()), **box_metrics(prediction, target_box)})
            processed.append(current)
        history_by_scene[scene] = processed
    result = pd.DataFrame(output)
    result["recall_iou_0_7"] = (result.iou >= .7).astype(int)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(OUT, index=False)
    print(result.groupby("model")[["normalized_center_error", "iou", "recall_iou_0_3", "recall_iou_0_5", "recall_iou_0_7"]].median())


if __name__ == "__main__":
    main()
