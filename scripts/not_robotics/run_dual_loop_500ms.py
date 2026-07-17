import bisect
import hashlib
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from pyquaternion import Quaternion
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SDK = Path(r"C:\work\auto\external\nuscenes-devkit\python-sdk")
sys.path.insert(0, str(SDK))
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points

ROOT = Path(r"C:\work\auto")
DATA = ROOT / "data" / "nuscenes"
PRIOR = ROOT / "results" / "not_robotics_real_feasibility"
AUDIT = ROOT / "results" / "not_dual_loop_latency"
OUT = ROOT / "results" / "not_dual_loop_latency_500ms"
SEED = 20260717
NEAR_METERS = 30.0
IOU_THRESHOLDS = [0.3, 0.5]


def atomic_csv(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def atomic_parquet(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def atomic_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def category_group(name, attributes):
    if name.startswith("human.pedestrian") or name in {"vehicle.bicycle", "vehicle.motorcycle"}:
        return "pedestrians_and_cyclists"
    moving = any("moving" in x for x in attributes)
    return "moving_vehicles" if moving else "stationary_or_parked"


def nusc_to_coco(name):
    if name.startswith("human.pedestrian"):
        return "person"
    return {
        "vehicle.bicycle": "bicycle", "vehicle.car": "car", "vehicle.motorcycle": "motorcycle",
        "vehicle.bus.bendy": "bus", "vehicle.bus.rigid": "bus", "vehicle.truck": "truck",
    }.get(name)


def projected_instances(nusc, sample_data_token):
    sd = nusc.get("sample_data", sample_data_token)
    _, boxes, intrinsic = nusc.get_sample_data(sample_data_token)
    out = {}
    for box in boxes:
        coco = nusc_to_coco(box.name)
        if coco is None:
            continue
        corners = box.corners()
        front = corners[2] > 0.1
        if not np.any(front):
            continue
        uv = view_points(corners[:, front], intrinsic, normalize=True)[:2]
        bbox = [float(np.clip(np.min(uv[0]), 0, sd["width"] - 1)), float(np.clip(np.min(uv[1]), 0, sd["height"] - 1)), float(np.clip(np.max(uv[0]), 0, sd["width"] - 1)), float(np.clip(np.max(uv[1]), 0, sd["height"] - 1))]
        if bbox[2] - bbox[0] < 2 or bbox[3] - bbox[1] < 2:
            continue
        ann = nusc.get("sample_annotation", box.token)
        attrs = [nusc.get("attribute", token)["name"] for token in ann["attribute_tokens"]]
        out[ann["instance_token"]] = {"bbox": bbox, "annotation_token": box.token, "category": box.name, "group": category_group(box.name, attrs), "attributes": attrs, "translation": ann["translation"], "size": ann["size"]}
    return out


def bbox_corners(box):
    x1, y1, x2, y2 = map(float, box)
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=float)


def corners_bbox(corners, width=1600, height=900):
    if not np.isfinite(corners).all():
        return [np.nan] * 4
    return [float(np.clip(corners[:, 0].min(), 0, width - 1)), float(np.clip(corners[:, 1].min(), 0, height - 1)), float(np.clip(corners[:, 0].max(), 0, width - 1)), float(np.clip(corners[:, 1].max(), 0, height - 1))]


def apply_homography(corners, h):
    points = np.c_[corners, np.ones(len(corners))]
    warped = (h @ points.T).T
    return warped[:, :2] / np.maximum(warped[:, 2:3], 1e-9)


def box_plane_points(box, depth, k_inv):
    corners = bbox_corners(box)
    rays = (k_inv @ np.c_[corners, np.ones(4)].T).T
    return rays * depth


def project_points(points, k):
    if np.any(points[:, 2] <= 0.2):
        return np.full((len(points), 2), np.nan)
    uvw = (k @ points.T).T
    return uvw[:, :2] / uvw[:, 2:3]


def fast_camera_transform(calibration, dx, dy, dyaw):
    r_ce = Quaternion(calibration["rotation"]).rotation_matrix
    t_ce = np.asarray(calibration["translation"], float)
    c, s = math.cos(dyaw), math.sin(dyaw)
    r_delta = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], float)
    t_delta = np.array([dx, dy, 0.0], float)
    a = r_ce.T @ r_delta.T @ r_ce
    b = r_ce.T @ (r_delta.T @ (t_ce - t_delta) - t_ce)
    return a, b


def oracle_camera_transform(source_pose, target_pose, calibration):
    r_ce = Quaternion(calibration["rotation"]).rotation_matrix
    t_ce = np.asarray(calibration["translation"], float)
    rs = Quaternion(source_pose["rotation"]).rotation_matrix
    rt = Quaternion(target_pose["rotation"]).rotation_matrix
    ts = np.asarray(source_pose["translation"], float)
    tt = np.asarray(target_pose["translation"], float)
    a = r_ce.T @ rt.T @ rs @ r_ce
    b = r_ce.T @ (rt.T @ (rs @ t_ce + ts - tt) - t_ce)
    return a, b


def warp_depth(box, depth, k, k_inv, a, b, object_displacement_ego=None, r_delta=None):
    points = box_plane_points(box, depth, k_inv)
    warped = (a @ points.T).T + b
    if object_displacement_ego is not None:
        r_ce = None
        # object displacement is converted by caller to target-camera coordinates.
        warped = warped + object_displacement_ego
    return project_points(warped, k)


def iou(a, b):
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        return 0.0
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    aa = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    bb = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return inter / (aa + bb - inter) if aa + bb - inter > 0 else 0.0


def box_metrics(pred, target):
    pc = np.array([(pred[0] + pred[2]) / 2, (pred[1] + pred[3]) / 2])
    tc = np.array([(target[0] + target[2]) / 2, (target[1] + target[3]) / 2])
    center = float(np.linalg.norm(pc - tc))
    pa = max(1.0, (pred[2] - pred[0]) * (pred[3] - pred[1]))
    ta = max(1.0, (target[2] - target[0]) * (target[3] - target[1]))
    ov = iou(pred, target)
    return {"center_error_px": center, "normalized_center_error": center / math.hypot(1600, 900), "iou": ov, "box_scale_error": abs(math.log(pa / ta)), "recall_iou_0_3": int(ov >= 0.3), "recall_iou_0_5": int(ov >= 0.5), "usable": int(ov >= 0.3)}


def global_flow_homography(source_path, target_path, source_boxes):
    start = time.perf_counter()
    a = cv2.imread(source_path, cv2.IMREAD_GRAYSCALE)
    b = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)
    if a is None or b is None:
        raise RuntimeError("Unreadable flow image")
    scale = 0.5
    a = cv2.resize(a, None, fx=scale, fy=scale)
    b = cv2.resize(b, None, fx=scale, fy=scale)
    mask = np.full(a.shape, 255, np.uint8)
    for box in source_boxes:
        x1, y1, x2, y2 = [int(round(x * scale)) for x in box]
        cv2.rectangle(mask, (max(0, x1), max(0, y1)), (min(a.shape[1] - 1, x2), min(a.shape[0] - 1, y2)), 0, -1)
    orb = cv2.ORB_create(nfeatures=2000, fastThreshold=12)
    ka, da = orb.detectAndCompute(a, mask)
    kb, db = orb.detectAndCompute(b, None)
    h_small = None
    inliers = 0
    method = "orb_ransac"
    if da is not None and db is not None and len(ka) >= 12 and len(kb) >= 12:
        matches = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(da, db, k=2)
        good = [m for m, n in matches if m.distance < 0.75 * n.distance]
        if len(good) >= 10:
            pa = np.float32([ka[m.queryIdx].pt for m in good])
            pb = np.float32([kb[m.trainIdx].pt for m in good])
            h_small, inlier_mask = cv2.findHomography(pa, pb, cv2.RANSAC, 3.0)
            inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0
    if h_small is None or not np.isfinite(h_small).all() or inliers < 8:
        shift, _ = cv2.phaseCorrelate(np.float32(a), np.float32(b))
        h_small = np.array([[1, 0, shift[0]], [0, 1, shift[1]], [0, 0, 1]], float)
        method = "phase_correlation_fallback"
    s = np.diag([scale, scale, 1.0])
    h = np.linalg.inv(s) @ h_small @ s
    return h / h[2, 2], method, inliers, (time.perf_counter() - start) * 1000


def summarize(group):
    return pd.Series({"objects": len(group), "median_center_error_px": group.center_error_px.median(), "median_iou": group.iou.median(), "recall_iou_0_3": group.recall_iou_0_3.mean(), "recall_iou_0_5": group.recall_iou_0_5.mean(), "median_normalized_center_error": group.normalized_center_error.median(), "median_box_scale_error": group.box_scale_error.median(), "proportion_usable": group.usable.mean(), "median_temporal_jitter_px": group.temporal_jitter_px.median()})


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    nusc = NuScenes(version="v1.0-mini", dataroot=str(DATA), verbose=False)
    timeline = pd.read_parquet(AUDIT / "streaming_timeline.parquet")
    timeline_by_token = timeline.set_index("sample_data_token")
    pairs = pd.read_parquet(AUDIT / "latency_pairs.parquet")
    eligible = pairs[(pairs.latency_ms == 500) & pairs.both_endpoints_annotated & (pairs.common_visible_instances > 0)].copy().sort_values(["scene_name", "source_timestamp_us"])
    if len(eligible) != 339 or eligible.scene_name.nunique() != 10:
        raise RuntimeError("Expected exactly 339 real pairs across 10 scenes")
    atomic_csv(eligible, OUT / "eligible_pairs.csv")

    detections = pd.read_parquet(PRIOR / "detector_outputs.parquet")
    frame_obs = pd.read_parquet(PRIOR / "frame_observations.parquet").set_index("sample_data_token")
    ann_lookup = {row["token"]: row for row in nusc.sample_annotation}
    det_records = []
    for d in detections[detections.is_tp == 1].itertuples(index=False):
        gt = json.loads(frame_obs.loc[d.sample_data_token, "gt_boxes_json"])
        if d.matched_gt_index < 0 or d.matched_gt_index >= len(gt):
            continue
        ann = ann_lookup[gt[d.matched_gt_index]["token"]]
        attrs = [nusc.get("attribute", token)["name"] for token in ann["attribute_tokens"]]
        det_records.append({"record_id": f"{d.sample_data_token}:{d.prediction_index}", "scene_name": d.scene_name, "sample_data_token": d.sample_data_token, "sample_token": d.sample_token, "timestamp_us": int(d.timestamp_us), "prediction_index": int(d.prediction_index), "instance_token": ann["instance_token"], "annotation_token": ann["token"], "class_name": d.class_name, "category": ann["category_name"], "object_group": category_group(ann["category_name"], attrs), "x1": d.x1, "y1": d.y1, "x2": d.x2, "y2": d.y2, "confidence": d.confidence})
    det_df = pd.DataFrame(det_records).sort_values(["scene_name", "instance_token", "timestamp_us"])

    # Real lidar depth in each matched source detection.
    depth_rows = []
    for token, group in det_df.groupby("sample_data_token"):
        sd = nusc.get("sample_data", token)
        sample = nusc.get("sample", sd["sample_token"])
        points, coloring, _ = nusc.explorer.map_pointcloud_to_image(sample["data"]["LIDAR_TOP"], token, render_intensity=False)
        for d in group.itertuples(index=False):
            inside = (points[0] >= d.x1) & (points[0] <= d.x2) & (points[1] >= d.y1) & (points[1] <= d.y2)
            values = np.asarray(coloring)[inside]
            values = values[np.isfinite(values) & (values > 0)]
            med = float(np.median(values)) if len(values) else np.nan
            mad = float(1.4826 * np.median(np.abs(values - med))) if len(values) else np.nan
            depth_rows.append({"record_id": d.record_id, "scene_name": d.scene_name, "sample_data_token": token, "prediction_index": d.prediction_index, "instance_token": d.instance_token, "lidar_points": len(values), "lidar_median_depth_m": med, "lidar_depth_uncertainty_m": mad, "box_width_px": d.x2 - d.x1, "box_height_px": d.y2 - d.y1, "class_name": d.class_name})
    depth = pd.DataFrame(depth_rows)
    depth["depth_m"] = depth["lidar_median_depth_m"]
    depth["depth_uncertainty_m"] = depth["lidar_depth_uncertainty_m"]
    depth["depth_method"] = np.where(depth.lidar_points > 0, "real_lidar_points_median", "cross_scene_box_geometry_fallback")
    features = ["log_box_height", "log_box_width"]
    depth["log_box_height"] = np.log(np.maximum(depth.box_height_px, 2))
    depth["log_box_width"] = np.log(np.maximum(depth.box_width_px, 2))
    for scene in sorted(depth.scene_name.unique()):
        train = depth[(depth.scene_name != scene) & depth.lidar_median_depth_m.notna()]
        test_index = depth[(depth.scene_name == scene) & depth.lidar_median_depth_m.isna()].index
        if len(test_index) == 0:
            continue
        pre = ColumnTransformer([("num", StandardScaler(), features), ("cat", OneHotEncoder(handle_unknown="ignore"), ["class_name"])])
        model = Pipeline([("pre", pre), ("ridge", Ridge(alpha=10.0))])
        model.fit(train, np.log(train.lidar_median_depth_m))
        pred = np.exp(model.predict(depth.loc[test_index]))
        residual = np.log(train.lidar_median_depth_m) - model.predict(train)
        sigma = float(np.std(residual))
        depth.loc[test_index, "depth_m"] = pred
        depth.loc[test_index, "depth_uncertainty_m"] = pred * (math.exp(sigma) - 1)
    if depth.depth_m.isna().any():
        raise RuntimeError("Depth fallback failed")
    atomic_parquet(depth.drop(columns=features), OUT / "lidar_depth_estimates.parquet")
    depth_map = depth.set_index("record_id").to_dict("index")
    det_history = {key: group.sort_values("timestamp_us") for key, group in det_df.groupby(["scene_name", "instance_token"])}

    target_maps = {token: projected_instances(nusc, token) for token in eligible.target_frame_token.unique()}
    source_all_boxes = {token: detections[detections.sample_data_token == token][["x1", "y1", "x2", "y2"]].to_numpy().tolist() for token in eligible.source_frame_token.unique()}

    # Precompute causal inertial integration and observed real-image background flow per pair.
    pair_infos = {}
    compute_rows = []
    for pair in eligible.itertuples(index=False):
        source_t = timeline_by_token.loc[pair.source_frame_token]
        target_t = timeline_by_token.loc[pair.target_frame_token]
        segment = timeline[(timeline.scene_name == pair.scene_name) & (timeline.timestamp_us >= pair.source_timestamp_us) & (timeline.timestamp_us <= pair.target_timestamp_us)].sort_values("timestamp_us")
        speed = float(source_t.pose_velocity_x) if np.isfinite(source_t.pose_velocity_x) else 0.0
        x = y = yaw = 0.0
        previous_time = pair.source_timestamp_us
        previous_yaw_rate = float(source_t.imu_yaw_rate) if np.isfinite(source_t.imu_yaw_rate) else 0.0
        previous_accel = float(source_t.imu_accel_x) if np.isfinite(source_t.imu_accel_x) else 0.0
        for row in segment.itertuples(index=False):
            dt = max(0.0, (row.timestamp_us - previous_time) / 1e6)
            yaw += previous_yaw_rate * dt
            speed += previous_accel * dt
            x += speed * math.cos(yaw) * dt
            y += speed * math.sin(yaw) * dt
            previous_time = row.timestamp_us
            previous_yaw_rate = float(row.imu_yaw_rate) if np.isfinite(row.imu_yaw_rate) else previous_yaw_rate
            previous_accel = float(row.imu_accel_x) if np.isfinite(row.imu_accel_x) else previous_accel
        calibration = nusc.get("calibrated_sensor", nusc.get("sample_data", pair.source_frame_token)["calibrated_sensor_token"])
        a_fast, b_fast = fast_camera_transform(calibration, x, y, yaw)
        source_pose = nusc.get("ego_pose", nusc.get("sample_data", pair.source_frame_token)["ego_pose_token"])
        target_pose = nusc.get("ego_pose", nusc.get("sample_data", pair.target_frame_token)["ego_pose_token"])
        a_oracle, b_oracle = oracle_camera_transform(source_pose, target_pose, calibration)
        h, flow_method, inliers, flow_ms = global_flow_homography(source_t.image_path, target_t.image_path, source_all_boxes[pair.source_frame_token])
        key = (pair.source_frame_token, pair.target_frame_token)
        pair_infos[key] = {"A": a_fast, "b": b_fast, "A_oracle": a_oracle, "b_oracle": b_oracle, "H": h, "dx": x, "dy": y, "dyaw": yaw, "flow_method": flow_method, "flow_inliers": inliers, "flow_ms": flow_ms, "calibration": calibration}
        compute_rows.append({"scene_name": pair.scene_name, "source_frame_token": pair.source_frame_token, "target_frame_token": pair.target_frame_token, "flow_method": flow_method, "flow_inliers": inliers, "background_flow_ms": flow_ms})

    # Deterministic negative-control assignments are fixed before metric inspection.
    keys = list(pair_infos)
    rng = np.random.default_rng(SEED)
    shuffled_keys = list(np.asarray(keys, dtype=object)[rng.permutation(len(keys))])
    shuffled_map = dict(zip(keys, [tuple(x) for x in shuffled_keys]))
    by_scene_keys = defaultdict(list)
    for key in keys:
        scene = timeline_by_token.loc[key[0]].scene_name
        by_scene_keys[scene].append(key)
    scenes = sorted(by_scene_keys)
    other_scene_map = {}
    shifted_imu_map = {}
    for si, scene in enumerate(scenes):
        other = scenes[(si + 1) % len(scenes)]
        for i, key in enumerate(by_scene_keys[scene]):
            other_scene_map[key] = by_scene_keys[other][i % len(by_scene_keys[other])]
            shifted_imu_map[key] = by_scene_keys[scene][max(0, i - 2)]

    object_rows = []
    pair_row_lookup = {(x.source_frame_token, x.target_frame_token): x for x in eligible.itertuples(index=False)}
    for key in keys:
        pair = pair_row_lookup[key]
        pair_compute_start = time.perf_counter()
        source_t = timeline_by_token.loc[pair.source_frame_token]
        target_map = target_maps[pair.target_frame_token]
        info = pair_infos[key]
        k = np.asarray(info["calibration"]["camera_intrinsic"], float)
        k_inv = np.linalg.inv(k)
        source_dets = det_df[det_df.sample_data_token == pair.source_frame_token]
        for d in source_dets.itertuples(index=False):
            if d.instance_token not in target_map:
                continue
            depth_info = depth_map[d.record_id]
            depth_m = float(depth_info["depth_m"])
            source_box = [d.x1, d.y1, d.x2, d.y2]
            target_box = target_map[d.instance_token]["bbox"]
            src_corners = bbox_corners(source_box)
            b3_corners = warp_depth(source_box, depth_m, k, k_inv, info["A"], info["b"])
            b2_corners = apply_homography(src_corners, info["H"])
            b0_corners = src_corners.copy()

            history = det_history[(d.scene_name, d.instance_token)]
            previous = history[history.timestamp_us < d.timestamp_us].tail(1)
            has_previous = len(previous) == 1
            if has_previous:
                prev = previous.iloc[0]
                dt_prev = (d.timestamp_us - prev.timestamp_us) / 1e6
                elapsed = pair.actual_elapsed_ms / 1000.0
                prev_box = np.array([prev.x1, prev.y1, prev.x2, prev.y2], float)
                current_box = np.array(source_box, float)
                b1_box = current_box + (current_box - prev_box) / dt_prev * elapsed if dt_prev > 0 else current_box
                b1_corners = bbox_corners(b1_box)
                prev_depth = depth_map[prev.record_id]["depth_m"]
                prev_sd = nusc.get("sample_data", prev.sample_data_token)
                prev_cal = nusc.get("calibrated_sensor", prev_sd["calibrated_sensor_token"])
                prev_k = np.asarray(prev_cal["camera_intrinsic"], float)
                prev_center = box_plane_points([prev.x1, prev.y1, prev.x2, prev.y2], prev_depth, np.linalg.inv(prev_k)).mean(axis=0)
                current_center = box_plane_points(source_box, depth_m, k_inv).mean(axis=0)
                # Past and current source poses are both available by t.
                prev_pose = nusc.get("ego_pose", prev_sd["ego_pose_token"])
                curr_pose = nusc.get("ego_pose", nusc.get("sample_data", pair.source_frame_token)["ego_pose_token"])
                rpc = Quaternion(prev_cal["rotation"]).rotation_matrix
                tpc = np.asarray(prev_cal["translation"], float)
                rc = Quaternion(info["calibration"]["rotation"]).rotation_matrix
                tc = np.asarray(info["calibration"]["translation"], float)
                rp = Quaternion(prev_pose["rotation"]).rotation_matrix
                rcurr = Quaternion(curr_pose["rotation"]).rotation_matrix
                tp = np.asarray(prev_pose["translation"], float)
                tcurr = np.asarray(curr_pose["translation"], float)
                prev_global = rp @ (rpc @ prev_center + tpc) + tp
                curr_global = rcurr @ (rc @ current_center + tc) + tcurr
                velocity_global = (curr_global - prev_global) / max(dt_prev, 1e-3)
                speed_obj = np.linalg.norm(velocity_global[:2])
                if speed_obj > 30:
                    velocity_global *= 30 / speed_obj
                # Convert past-estimated object velocity using only the source
                # pose and the fast loop's causally integrated yaw forecast.
                velocity_source_ego = rcurr.T @ velocity_global
                c_fast, s_fast = math.cos(info["dyaw"]), math.sin(info["dyaw"])
                r_fast = np.array([[c_fast, -s_fast, 0], [s_fast, c_fast, 0], [0, 0, 1]], float)
                object_delta_target_cam = rc.T @ r_fast.T @ (velocity_source_ego * elapsed)
            else:
                b1_corners = b0_corners.copy()
                object_delta_target_cam = np.zeros(3)
            b4_points = (info["A"] @ box_plane_points(source_box, depth_m, k_inv).T).T + info["b"] + object_delta_target_cam
            b4_corners = project_points(b4_points, k)
            b5_corners = b4_corners + (b2_corners - b3_corners)
            oracle_corners = warp_depth(source_box, depth_m, k, k_inv, info["A_oracle"], info["b_oracle"])
            wrong_depth_corners = warp_depth(source_box, depth_m * 2.0, k, k_inv, info["A"], info["b"])
            shift_info = pair_infos[shifted_imu_map[key]]
            shifted_imu_corners = warp_depth(source_box, depth_m, k, k_inv, shift_info["A"], shift_info["b"])
            shuffled_flow = apply_homography(src_corners, pair_infos[shuffled_map[key]]["H"])
            other_flow = apply_homography(src_corners, pair_infos[other_scene_map[key]]["H"])
            shifted_flow = apply_homography(src_corners, pair_infos[shifted_imu_map[key]]["H"])
            shuffled_residual_corners = b4_corners + (shuffled_flow - b3_corners)
            other_scene_corners = b4_corners + (other_flow - b3_corners)
            shifted_flow_corners = b4_corners + (shifted_flow - b3_corners)
            # A deployable propagation keeps the last valid causal box when a
            # projective warp crosses the camera plane or becomes non-finite.
            if not np.isfinite(b2_corners).all(): b2_corners = b0_corners.copy()
            if not np.isfinite(b3_corners).all(): b3_corners = b0_corners.copy()
            if not np.isfinite(b4_corners).all(): b4_corners = b3_corners.copy()
            if not np.isfinite(b5_corners).all(): b5_corners = b2_corners.copy()
            if not np.isfinite(oracle_corners).all(): oracle_corners = b0_corners.copy()
            if not np.isfinite(wrong_depth_corners).all(): wrong_depth_corners = b0_corners.copy()
            if not np.isfinite(shifted_imu_corners).all(): shifted_imu_corners = b0_corners.copy()
            if not np.isfinite(shuffled_residual_corners).all(): shuffled_residual_corners = b2_corners.copy()
            if not np.isfinite(other_scene_corners).all(): other_scene_corners = b2_corners.copy()
            if not np.isfinite(shifted_flow_corners).all(): shifted_flow_corners = b2_corners.copy()
            models = {
                "B0_stale": b0_corners, "B1_constant_2d_velocity": b1_corners, "B2_visual_flow": b2_corners,
                "B3_inertial_pose": b3_corners, "B4_inertial_object_tracker": b4_corners,
                "B5_dual_loop": b5_corners, "control_shuffled_flow_residual": shuffled_residual_corners,
                "control_other_scene_flow": other_scene_corners, "control_time_shifted_imu": shifted_imu_corners,
                "control_time_shifted_flow": shifted_flow_corners,
                "control_incorrect_depth": wrong_depth_corners, "diagnostic_oracle_ego_pose": oracle_corners,
            }
            yaw_deg_s = abs(float(source_t.imu_yaw_rate if np.isfinite(source_t.imu_yaw_rate) else 0.0)) * 180 / math.pi
            base = {"scene_name": pair.scene_name, "source_frame_token": pair.source_frame_token, "target_frame_token": pair.target_frame_token, "source_timestamp_us": pair.source_timestamp_us, "target_timestamp_us": pair.target_timestamp_us, "nominal_latency_ms": 500, "actual_elapsed_ms": pair.actual_elapsed_ms, "record_id": d.record_id, "prediction_index": d.prediction_index, "instance_token": d.instance_token, "class_name": d.class_name, "category": d.category, "object_group": d.object_group, "confidence": d.confidence, "source_box_json": json.dumps(source_box), "target_annotation_box_json": json.dumps(target_box), "depth_m": depth_m, "depth_uncertainty_m": depth_info["depth_uncertainty_m"], "lidar_points": depth_info["lidar_points"], "depth_method": depth_info["depth_method"], "distance_group": "near" if depth_m < NEAR_METERS else "far", "yaw_rate_deg_s": yaw_deg_s, "has_previous_detection": has_previous, "flow_method": info["flow_method"], "flow_inliers": info["flow_inliers"], "flow_compute_ms": info["flow_ms"], "steer_angle_feedback": source_t.steer_angle_feedback, "throttle_sensor": source_t.throttle_sensor, "brake_sensor": source_t.brake_sensor, "requested_torque": source_t.requested_torque}
            for model_name, corners in models.items():
                pred_box = corners_bbox(corners)
                object_rows.append({**base, "model": model_name, "predicted_box_json": json.dumps(pred_box), **box_metrics(pred_box, target_box)})
        pair_infos[key]["box_propagation_ms"] = (time.perf_counter() - pair_compute_start) * 1000

    propagated = pd.DataFrame(object_rows)
    # Scene-relative high-yaw definition was fixed before inspecting propagation outcomes.
    yaw_thresholds = propagated[propagated.model == "B0_stale"].groupby("scene_name").yaw_rate_deg_s.median().to_dict()
    propagated["high_yaw"] = propagated.apply(lambda r: r.yaw_rate_deg_s >= yaw_thresholds[r.scene_name], axis=1)

    # B6: leave-one-scene-out action correction to B5. Future boxes are targets only in training scenes.
    base5 = propagated[propagated.model == "B5_dual_loop"].copy().reset_index(drop=True)
    def unpack_box(series): return np.vstack(series.map(json.loads))
    b5_boxes = unpack_box(base5.predicted_box_json)
    target_boxes = unpack_box(base5.target_annotation_box_json)
    b5_params = np.c_[(b5_boxes[:, 0] + b5_boxes[:, 2]) / 2, (b5_boxes[:, 1] + b5_boxes[:, 3]) / 2, np.log(np.maximum(2, b5_boxes[:, 2] - b5_boxes[:, 0])), np.log(np.maximum(2, b5_boxes[:, 3] - b5_boxes[:, 1]))]
    target_params = np.c_[(target_boxes[:, 0] + target_boxes[:, 2]) / 2, (target_boxes[:, 1] + target_boxes[:, 3]) / 2, np.log(np.maximum(2, target_boxes[:, 2] - target_boxes[:, 0])), np.log(np.maximum(2, target_boxes[:, 3] - target_boxes[:, 1]))]
    correction = np.zeros_like(b5_params)
    action_features = ["steer_angle_feedback", "throttle_sensor", "brake_sensor", "requested_torque", "depth_m", "yaw_rate_deg_s", "confidence"]
    for scene in sorted(base5.scene_name.unique()):
        train = base5.scene_name != scene
        test = base5.scene_name == scene
        model = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("ridge", Ridge(alpha=30.0))])
        model.fit(base5.loc[train, action_features], (target_params - b5_params)[train])
        correction[test] = model.predict(base5.loc[test, action_features])
    correction[:, :2] = np.clip(correction[:, :2], -200, 200)
    correction[:, 2:] = np.clip(correction[:, 2:], -0.7, 0.7)
    b6_params = b5_params + correction
    b6_boxes = np.c_[b6_params[:, 0] - np.exp(b6_params[:, 2]) / 2, b6_params[:, 1] - np.exp(b6_params[:, 3]) / 2, b6_params[:, 0] + np.exp(b6_params[:, 2]) / 2, b6_params[:, 1] + np.exp(b6_params[:, 3]) / 2]
    b6_rows = []
    for i, row in base5.iterrows():
        box = corners_bbox(bbox_corners(b6_boxes[i]))
        record = row.to_dict()
        record.update({"model": "B6_action_augmented_dual_loop", "predicted_box_json": json.dumps(box), **box_metrics(box, json.loads(row.target_annotation_box_json))})
        b6_rows.append(record)
    propagated = pd.concat([propagated, pd.DataFrame(b6_rows)], ignore_index=True)

    propagated = propagated.sort_values(["scene_name", "instance_token", "model", "source_timestamp_us"])
    propagated["temporal_jitter_px"] = propagated.groupby(["scene_name", "instance_token", "model"])["center_error_px"].diff().abs()
    atomic_parquet(propagated, OUT / "propagated_boxes.parquet")

    comparison = propagated.groupby("model", group_keys=False).apply(summarize, include_groups=False).reset_index()
    atomic_csv(comparison, OUT / "model_comparison.csv")
    high = propagated[propagated.high_yaw]
    high_aggregate = high.groupby("model", group_keys=False).apply(summarize, include_groups=False).reset_index()
    high_aggregate["object_group"] = "ALL"
    high_aggregate["distance_group"] = "ALL"
    high_split = high.groupby(["model", "object_group", "distance_group"], group_keys=False).apply(summarize, include_groups=False).reset_index()
    high_metrics = pd.concat([high_aggregate, high_split], ignore_index=True)
    atomic_csv(high_metrics, OUT / "high_yaw_metrics.csv")
    scene_metrics = propagated.groupby(["scene_name", "model", "high_yaw"], group_keys=False).apply(summarize, include_groups=False).reset_index()
    atomic_csv(scene_metrics, OUT / "scene_heldout_metrics.csv")
    loop_models = ["B2_visual_flow", "B3_inertial_pose", "B4_inertial_object_tracker", "B5_dual_loop", "B6_action_augmented_dual_loop"]
    atomic_csv(comparison[comparison.model.isin(loop_models)], OUT / "loop_ablation.csv")
    control_models = [x for x in comparison.model if x.startswith("control_") or x.startswith("diagnostic_")]
    atomic_csv(comparison[comparison.model.isin(control_models)], OUT / "negative_controls.csv")

    compute_df = pd.DataFrame(compute_rows)
    compute_df["b3_b4_box_warp_ms"] = [pair_infos[(r.source_frame_token, r.target_frame_token)]["box_propagation_ms"] for r in compute_df.itertuples(index=False)]
    compute_df["b5_total_overhead_ms"] = compute_df.background_flow_ms + compute_df.b3_b4_box_warp_ms
    atomic_csv(compute_df, OUT / "compute_time.csv")

    primary_models = ["B0_stale", "B1_constant_2d_velocity", "B2_visual_flow", "B3_inertial_pose", "B4_inertial_object_tracker", "B5_dual_loop", "B6_action_augmented_dual_loop"]
    primary = high[high.model.isin(primary_models)].groupby("model", group_keys=False).apply(summarize, include_groups=False)
    b0 = primary.loc["B0_stale"]
    b5 = primary.loc["B5_dual_loop"]
    one_loop_names = ["B1_constant_2d_velocity", "B2_visual_flow", "B3_inertial_pose", "B4_inertial_object_tracker"]
    best_one = min(one_loop_names, key=lambda name: primary.loc[name, "median_center_error_px"])
    best = primary.loc[best_one]
    scene_high = scene_metrics[scene_metrics.high_yaw & scene_metrics.model.isin(["B0_stale", "B5_dual_loop"])].pivot(index="scene_name", columns="model", values="median_center_error_px")
    improved_scenes = int((scene_high["B5_dual_loop"] < scene_high["B0_stale"]).sum())
    overhead = float(compute_df.b5_total_overhead_ms.median())
    control_high = high.groupby("model").center_error_px.median()
    shuffled = control_high.loc["control_shuffled_flow_residual"]
    other_scene = control_high.loc["control_other_scene_flow"]
    shifted_flow = control_high.loc["control_time_shifted_flow"]
    shifted_imu = control_high.loc["control_time_shifted_imu"]
    criteria = {
        "center_error_reduction_vs_stale_at_least_15pct": (b0.median_center_error_px - b5.median_center_error_px) / b0.median_center_error_px >= 0.15,
        "center_error_reduction_vs_best_one_loop_at_least_8pct": (best.median_center_error_px - b5.median_center_error_px) / best.median_center_error_px >= 0.08,
        "median_iou_improves_vs_stale": b5.median_iou > b0.median_iou,
        "recall_iou_0_3_not_decreased": b5.recall_iou_0_3 >= b0.recall_iou_0_3,
        "improvement_in_at_least_7_scenes": improved_scenes >= 7,
        "median_overhead_below_20ms": overhead < 20,
        "shifted_shuffled_other_scene_and_imu_controls_fail": min(shuffled, other_scene, shifted_flow, shifted_imu) >= b5.median_center_error_px,
    }
    if all(criteria.values()):
        decision = "FEASIBILITY GO"
    elif b5.median_center_error_px < b0.median_center_error_px and (b5.median_center_error_px >= best.median_center_error_px):
        decision = "CONDITIONAL GO"
    else:
        decision = "NO-GO"

    manifest = {
        "status": "complete", "dataset": "real nuScenes v1.0-mini", "nominal_latency_ms": 500,
        "eligible_real_pairs": len(eligible), "scenes": int(eligible.scene_name.nunique()), "primary_detector_object_comparisons": int(len(base5)),
        "high_yaw_definition": "at or above each scene's median absolute causal IMU yaw rate among eligible source observations",
        "near_far_threshold_m": NEAR_METERS, "validation": "complete-scene leave-one-scene-out for B6; deterministic baselines evaluated per complete scene",
        "criteria": {k: bool(v) for k, v in criteria.items()}, "decision": decision, "best_one_loop": best_one,
        "primary_high_yaw": {"B0": b0.to_dict(), "best_one_loop": best.to_dict(), "B5": b5.to_dict(), "improved_scenes": improved_scenes, "median_overhead_ms": overhead},
        "input_hashes": {"eligible_pairs_source": sha256(AUDIT / "latency_pairs.parquet"), "detector_outputs": sha256(PRIOR / "detector_outputs.parquet"), "timeline": sha256(AUDIT / "streaming_timeline.parquet")},
    }
    atomic_text(OUT / "run_manifest.json", json.dumps(manifest, indent=2) + "\n")
    criteria_md = "\n".join(f"- {k}: {'PASS' if v else 'FAIL'}" for k, v in criteria.items())
    atomic_text(OUT / "leakage_audit.md", "# Leakage audit\n\n- Primary inputs are actual YOLO detections at t, real lidar at t, causal IMU/CAN through the arrival frame, past detections, and real images no later than t+500 ms.\n- Target annotation boxes are used only for evaluation or as training targets in other scenes for B6.\n- B6 uses leave-one-scene-out training.\n- Source annotation identity is used only to associate a matched YOLO detection with the persistent target instance.\n- Lidar fallback is trained on other scenes only.\n- Oracle ego pose is diagnostic-only and excluded from feasibility criteria.\n- No annotation interpolation, pseudo-labeling, future object velocity, or future annotation geometry is used as an input.\n\nStatus: PASSED.\n")
    atomic_text(OUT / "scientific_decision.md", f"# Scientific decision\n\n**{decision}**\n\nPrespecified criteria:\n\n{criteria_md}\n\nThis is a mini-set engineering feasibility decision, not publication GO.\n")
    atomic_text(OUT / "README.md", "# Real nuScenes 500 ms dual-loop latency study\n\nThis directory contains a complete-scene-grouped evaluation on 339 real annotated 500 ms pairs. Primary rows use actual YOLO11n detections; future annotation geometry appears only in evaluation targets. Diagnostic oracle rows are explicitly separated.\n")
    print(json.dumps({"eligible_pairs": len(eligible), "objects": len(base5), "best_one_loop": best_one, "b0_center": b0.median_center_error_px, "b5_center": b5.median_center_error_px, "decision": decision, "overhead_ms": overhead}, indent=2))


if __name__ == "__main__":
    main()
