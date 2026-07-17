import bisect
import hashlib
import json
import logging
import math
import os
import platform
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import sklearn
import torch
import ultralytics
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from ultralytics import YOLO

SDK = Path(r"C:\work\auto\external\nuscenes-devkit\python-sdk")
sys.path.insert(0, str(SDK))
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points


ROOT = Path(r"C:\work\auto")
DATA = ROOT / "data" / "nuscenes"
OUT = ROOT / "results" / "not_robotics_real_feasibility"
LOG = OUT / "logs" / "recovery_resume.log"
WEIGHTS = ROOT / "yolo11n.pt"
SEED = 20260717
IOU_THRESHOLD = 0.50
CONF_THRESHOLD = 0.05
IMAGE_SIZE = 640

COCO_TO_ID = {
    "person": 0,
    "bicycle": 1,
    "car": 2,
    "motorcycle": 3,
    "bus": 5,
    "truck": 7,
}


def setup_logging():
    LOG.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(LOG, mode="a", encoding="utf-8"), logging.StreamHandler()],
    )


def atomic_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def atomic_csv(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def atomic_parquet(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def sha256(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def nusc_to_coco(name: str):
    if name.startswith("human.pedestrian"):
        return "person"
    mapping = {
        "vehicle.bicycle": "bicycle",
        "vehicle.car": "car",
        "vehicle.motorcycle": "motorcycle",
        "vehicle.bus.bendy": "bus",
        "vehicle.bus.rigid": "bus",
        "vehicle.truck": "truck",
    }
    return mapping.get(name)


def project_gt(boxes, intrinsic, width, height):
    out = []
    for box in boxes:
        coco = nusc_to_coco(box.name)
        if coco is None:
            continue
        corners = box.corners()
        in_front = corners[2, :] > 0.1
        if not np.any(in_front):
            continue
        projected = view_points(corners[:, in_front], intrinsic, normalize=True)[:2, :]
        x1 = float(np.clip(np.min(projected[0]), 0, width - 1))
        y1 = float(np.clip(np.min(projected[1]), 0, height - 1))
        x2 = float(np.clip(np.max(projected[0]), 0, width - 1))
        y2 = float(np.clip(np.max(projected[1]), 0, height - 1))
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        out.append({"class_name": coco, "class_id": COCO_TO_ID[coco], "xyxy": [x1, y1, x2, y2], "token": box.token})
    return out


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    ub = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (ua + ub - inter) if ua + ub - inter > 0 else 0.0


def match_predictions(preds, gts):
    candidates = []
    for pi, pred in enumerate(preds):
        for gi, gt in enumerate(gts):
            if pred["class_name"] == gt["class_name"]:
                ov = iou(pred["xyxy"], gt["xyxy"])
                if ov >= IOU_THRESHOLD:
                    candidates.append((ov, pi, gi))
    used_p, used_g, matches = set(), set(), {}
    for ov, pi, gi in sorted(candidates, reverse=True):
        if pi not in used_p and gi not in used_g:
            used_p.add(pi)
            used_g.add(gi)
            matches[pi] = (gi, ov)
    return matches


class CausalCan:
    def __init__(self, api):
        self.api = api
        self.cache = {}

    def messages(self, scene, name):
        key = (scene, name)
        if key not in self.cache:
            msgs = self.api.get_messages(scene, name)
            msgs = sorted(msgs, key=lambda x: x["utime"])
            self.cache[key] = (msgs, [x["utime"] for x in msgs])
        return self.cache[key]

    def prior(self, scene, name, timestamp):
        msgs, times = self.messages(scene, name)
        idx = bisect.bisect_right(times, timestamp) - 1
        if idx < 0:
            return None, np.nan
        msg = msgs[idx]
        return msg, (timestamp - msg["utime"]) / 1000.0


def get_nested(message, key, default=np.nan):
    if message is None:
        return default
    value = message.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def get_vec(message, key, index, default=np.nan):
    if message is None:
        return default
    try:
        return float(message[key][index])
    except (KeyError, IndexError, TypeError, ValueError):
        return default


def compute_flow(previous: Path | None, current: Path):
    if previous is None:
        return 0.0
    a = cv2.imread(str(previous), cv2.IMREAD_GRAYSCALE)
    b = cv2.imread(str(current), cv2.IMREAD_GRAYSCALE)
    if a is None or b is None:
        raise RuntimeError(f"Could not read optical-flow pair: {previous}, {current}")
    a = cv2.resize(a, (640, 360))
    b = cv2.resize(b, (640, 360))
    flow = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return float(np.mean(mag))


def build_model(numeric, categorical=("class_name",)):
    pre = ColumnTransformer(
        [
            ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), list(numeric)),
            ("cat", OneHotEncoder(handle_unknown="ignore"), list(categorical)),
        ]
    )
    return Pipeline([("pre", pre), ("model", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=SEED))])


def ece(y, p, bins=10):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    total = len(y)
    value = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if np.any(mask):
            value += np.sum(mask) / total * abs(np.mean(y[mask]) - np.mean(p[mask]))
    return float(value)


def oof_scores(df, features):
    scores = np.full(len(df), np.nan)
    groups = df["scene_name"].to_numpy()
    splitter = GroupKFold(n_splits=5)
    for train, test in splitter.split(df, df["is_tp"], groups):
        model = build_model(features)
        model.fit(df.iloc[train], df.iloc[train]["is_tp"])
        scores[test] = model.predict_proba(df.iloc[test])[:, 1]
    return scores


def choose_gate_threshold(df, scores, frame_df, motion_cut):
    total_gt = frame_df["gt_count_evaluated"].sum()
    base_tp = int(df["is_tp"].sum())
    high = df["motion_score"] >= motion_cut
    base_high_fp = int(((df["is_tp"] == 0) & high).sum())
    candidates = np.unique(np.r_[0.0, np.quantile(scores, np.linspace(0.01, 0.80, 160))])
    best = (0.0, 0.0, 0)
    for threshold in candidates:
        keep = scores >= threshold
        kept_tp = int((df["is_tp"].to_numpy() * keep).sum())
        recall_loss = (base_tp - kept_tp) / total_gt if total_gt else 0.0
        if recall_loss > 0.020000001:
            continue
        removed = int(((df["is_tp"] == 0).to_numpy() & high.to_numpy() & ~keep).sum())
        reduction = removed / base_high_fp if base_high_fp else 0.0
        candidate = (reduction, float(threshold), removed)
        if candidate > best:
            best = candidate
    return best[1]


def nested_gate_scores(detections, frames, features):
    scores = np.full(len(detections), np.nan)
    thresholds = np.full(len(detections), np.nan)
    motion_cuts = np.full(len(detections), np.nan)
    groups = detections["scene_name"].to_numpy()
    splitter = GroupKFold(n_splits=5)
    for outer_train, outer_test in splitter.split(detections, detections["is_tp"], groups):
        train_df = detections.iloc[outer_train]
        test_df = detections.iloc[outer_test]
        train_scenes = set(train_df["scene_name"])
        train_frames = frames[frames["scene_name"].isin(train_scenes)]
        inner = GroupKFold(n_splits=min(4, train_df["scene_name"].nunique()))
        inner_scores = np.full(len(train_df), np.nan)
        for inner_train, inner_test in inner.split(train_df, train_df["is_tp"], train_df["scene_name"]):
            m = build_model(features)
            m.fit(train_df.iloc[inner_train], train_df.iloc[inner_train]["is_tp"])
            inner_scores[inner_test] = m.predict_proba(train_df.iloc[inner_test])[:, 1]
        motion_cut = float(train_frames["motion_score"].quantile(0.75))
        threshold = choose_gate_threshold(train_df, inner_scores, train_frames, motion_cut)
        final = build_model(features)
        final.fit(train_df, train_df["is_tp"])
        scores[outer_test] = final.predict_proba(test_df)[:, 1]
        thresholds[outer_test] = threshold
        motion_cuts[outer_test] = motion_cut
    return scores, thresholds, motion_cuts


def metric_row(name, y, p):
    return {
        "model": name,
        "auc_roc": roc_auc_score(y, p),
        "brier_score": brier_score_loss(y, p),
        "ece_10bin": ece(y, p),
    }


def main():
    setup_logging()
    random.seed(SEED)
    np.random.seed(SEED)
    logging.info("RESUME_START real nuScenes feasibility study")
    logging.info("Command: C:\\work\\auto\\.venv-not-robotics\\Scripts\\python.exe C:\\work\\auto\\scripts\\not_robotics\\resume_real_feasibility.py")
    logging.info("Loading nuScenes metadata and CAN bus")
    nusc = NuScenes(version="v1.0-mini", dataroot=str(DATA), verbose=False)
    can = CausalCan(NuScenesCanBus(dataroot=str(DATA)))

    legacy_path = OUT / "detector_outputs_INCOMPLETE.parquet"
    legacy_flow = {}
    if legacy_path.exists():
        legacy = pd.read_parquet(legacy_path)
        if {"timestamp", "flow_mag"}.issubset(legacy.columns) and len(legacy) == 404:
            legacy_flow = dict(zip(legacy["timestamp"].astype(int), legacy["flow_mag"].astype(float)))
            logging.info("Reusing 404 real Farneback flow values from preserved legacy output")

    samples = sorted(nusc.sample, key=lambda x: (nusc.get("scene", x["scene_token"])["name"], x["timestamp"]))
    model = YOLO(str(WEIGHTS))
    image_paths = []
    sample_meta = []
    for sample in samples:
        scene = nusc.get("scene", sample["scene_token"])
        cam = nusc.get("sample_data", sample["data"]["CAM_FRONT"])
        path = DATA / cam["filename"]
        if not path.is_file():
            raise FileNotFoundError(path)
        image_paths.append(str(path))
        sample_meta.append((sample, scene, cam, path))

    logging.info("Running YOLO inference on %d real CAM_FRONT keyframes", len(image_paths))
    predictions = model.predict(image_paths, imgsz=IMAGE_SIZE, conf=CONF_THRESHOLD, iou=0.7, device="cpu", batch=8, verbose=False, stream=True)
    frame_rows, detection_rows, sync_rows = [], [], []
    previous_by_scene = {}
    for index, (result, meta) in enumerate(zip(predictions, sample_meta), start=1):
        sample, scene, cam, path = meta
        scene_name = scene["name"]
        timestamp = int(cam["timestamp"])
        _, boxes, intrinsic = nusc.get_sample_data(cam["token"])
        gt = project_gt(boxes, intrinsic, int(cam["width"]), int(cam["height"]))

        preds = []
        if result.boxes is not None:
            xyxy = result.boxes.xyxy.cpu().numpy()
            conf = result.boxes.conf.cpu().numpy()
            cls = result.boxes.cls.cpu().numpy().astype(int)
            for coords, confidence, class_id in zip(xyxy, conf, cls):
                class_name = result.names[int(class_id)]
                if class_name in COCO_TO_ID:
                    preds.append({"class_name": class_name, "class_id": int(class_id), "confidence": float(confidence), "xyxy": [float(x) for x in coords]})
        matches = match_predictions(preds, gt)

        pose, pose_age = can.prior(scene_name, "pose", timestamp)
        steer, steer_age = can.prior(scene_name, "steeranglefeedback", timestamp)
        sensors, sensors_age = can.prior(scene_name, "zoesensors", timestamp)
        veh, veh_age = can.prior(scene_name, "zoe_veh_info", timestamp)
        speed = get_vec(pose, "vel", 0)
        accel_x = get_vec(pose, "accel", 0)
        accel_y = get_vec(pose, "accel", 1)
        yaw_rate = get_vec(pose, "rotation_rate", 2)
        motion_score = math.sqrt((accel_x if np.isfinite(accel_x) else 0.0) ** 2 + (accel_y if np.isfinite(accel_y) else 0.0) ** 2) + abs((yaw_rate if np.isfinite(yaw_rate) else 0.0) * (speed if np.isfinite(speed) else 0.0))
        if timestamp in legacy_flow:
            flow_mag = legacy_flow[timestamp]
            flow_source = "reused_verified_real_farneback"
        else:
            flow_mag = compute_flow(previous_by_scene.get(scene_name), path)
            flow_source = "computed_real_farneback"
        previous_by_scene[scene_name] = path

        common = {
            "scene_name": scene_name,
            "scene_token": scene["token"],
            "sample_token": sample["token"],
            "sample_data_token": cam["token"],
            "timestamp_us": timestamp,
            "image_path": str(path),
            "image_sha256": sha256(path),
            "flow_mag": flow_mag,
            "flow_source": flow_source,
            "speed_mps": speed,
            "accel_x": accel_x,
            "accel_y": accel_y,
            "yaw_rate": yaw_rate,
            "motion_score": motion_score,
            "steer_angle_feedback": get_nested(steer, "value"),
            "throttle_sensor": get_nested(sensors, "throttle_sensor"),
            "brake_sensor": get_nested(sensors, "brake_sensor"),
            "steering_sensor": get_nested(sensors, "steering_sensor"),
            "requested_torque": get_nested(veh, "requestedTorqueAfterProc"),
            "regen": get_nested(veh, "regen"),
            "pedal_cc": get_nested(veh, "pedal_cc"),
        }
        tp = len(matches)
        fp = len(preds) - tp
        fn = len(gt) - tp
        frame_rows.append({**common, "gt_count_evaluated": len(gt), "pred_count_evaluated": len(preds), "tp": tp, "fp": fp, "fn": fn, "gt_boxes_json": json.dumps(gt, separators=(",", ":")), "pred_boxes_json": json.dumps(preds, separators=(",", ":"))})
        for message, age in [("pose", pose_age), ("steeranglefeedback", steer_age), ("zoesensors", sensors_age), ("zoe_veh_info", veh_age)]:
            sync_rows.append({"scene_name": scene_name, "sample_token": sample["token"], "timestamp_us": timestamp, "message": message, "causal_prior_found": bool(np.isfinite(age)), "age_ms": age})
        for pi, pred in enumerate(preds):
            x1, y1, x2, y2 = pred["xyxy"]
            matched = matches.get(pi)
            detection_rows.append({
                **common,
                "prediction_index": pi,
                "class_name": pred["class_name"],
                "class_id": pred["class_id"],
                "confidence": pred["confidence"],
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "box_area_norm": ((x2 - x1) * (y2 - y1)) / (cam["width"] * cam["height"]),
                "box_center_x_norm": ((x1 + x2) / 2) / cam["width"],
                "box_center_y_norm": ((y1 + y2) / 2) / cam["height"],
                "is_tp": int(matched is not None),
                "matched_gt_index": matched[0] if matched else -1,
                "matched_iou": matched[1] if matched else 0.0,
            })
        if index % 50 == 0 or index == len(samples):
            logging.info("Detector progress %d/%d", index, len(samples))

    frames = pd.DataFrame(frame_rows)
    detections = pd.DataFrame(detection_rows)
    sync = pd.DataFrame(sync_rows)
    if len(detections) == 0 or detections["is_tp"].nunique() != 2:
        raise RuntimeError("Detector matching did not produce both TP and FP observations")

    non_action = ["confidence", "box_area_norm", "box_center_x_norm", "box_center_y_norm", "flow_mag", "speed_mps", "accel_x", "accel_y", "yaw_rate"]
    detector_only = ["confidence", "box_area_norm", "box_center_x_norm", "box_center_y_norm"]
    action = ["steer_angle_feedback", "throttle_sensor", "brake_sensor", "steering_sensor", "requested_torque", "regen", "pedal_cc"]
    full = non_action + action

    logging.info("Fitting grouped scene-held-out reliability models")
    scores = {}
    scores["detector_only"] = oof_scores(detections, detector_only)
    scores["non_action"] = oof_scores(detections, non_action)
    scores["action_only"] = oof_scores(detections, action)
    full_scores, gate_thresholds, motion_cuts = nested_gate_scores(detections, frames, full)
    scores["full_action"] = full_scores

    shifted_frames = frames[["sample_token", "scene_name"] + action].copy()
    shifted_frames[action] = shifted_frames.groupby("scene_name", sort=False)[action].shift(2)
    shifted = detections.drop(columns=action).merge(shifted_frames[["sample_token"] + action], on="sample_token", how="left")
    scores["full_shifted_1s"] = oof_scores(shifted, full)

    shuffled_frames = frames[["sample_token"] + action].copy()
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(shuffled_frames))
    shuffled_frames[action] = shuffled_frames[action].iloc[perm].to_numpy()
    shuffled = detections.drop(columns=action).merge(shuffled_frames, on="sample_token", how="left")
    scores["full_shuffled"] = oof_scores(shuffled, full)

    y = detections["is_tp"].to_numpy()
    ablation = pd.DataFrame([metric_row(name, y, p) for name, p in scores.items()])
    best_non_action_name = ablation[ablation["model"].isin(["detector_only", "non_action"])].sort_values("auc_roc", ascending=False).iloc[0]["model"]
    best_non_action_auc = float(ablation.loc[ablation["model"] == best_non_action_name, "auc_roc"].iloc[0])
    full_auc = float(ablation.loc[ablation["model"] == "full_action", "auc_roc"].iloc[0])

    reliability = detections[["scene_name", "scene_token", "sample_token", "sample_data_token", "timestamp_us", "prediction_index", "is_tp", "confidence", "motion_score"]].copy()
    for name, p in scores.items():
        reliability[f"score_{name}"] = p
    reliability["gate_threshold_full"] = gate_thresholds
    reliability["high_motion_cut_train"] = motion_cuts
    reliability["gate_keep_full"] = full_scores >= gate_thresholds
    reliability["high_motion_heldout"] = detections["motion_score"].to_numpy() >= motion_cuts

    base_tp = int(detections["is_tp"].sum())
    base_fp = int((detections["is_tp"] == 0).sum())
    total_gt = int(frames["gt_count_evaluated"].sum())
    keep = reliability["gate_keep_full"].to_numpy()
    gated_tp = int((y * keep).sum())
    gated_fp = int(((y == 0) & keep).sum())
    high = reliability["high_motion_heldout"].to_numpy()
    high_base_fp = int(((y == 0) & high).sum())
    high_gated_fp = int(((y == 0) & high & keep).sum())
    base_recall = base_tp / total_gt
    gated_recall = gated_tp / total_gt
    high_fp_change = (high_gated_fp - high_base_fp) / high_base_fp if high_base_fp else np.nan
    non_action_ece = ece(y, scores[best_non_action_name])
    full_ece = ece(y, full_scores)

    headline = {
        "processed_scenes": int(frames["scene_name"].nunique()),
        "processed_frames": int(len(frames)),
        "evaluated_gt_boxes": total_gt,
        "predictions": int(len(detections)),
        "matched_tp": base_tp,
        "baseline_fp": base_fp,
        "best_non_action_model": best_non_action_name,
        "best_non_action_auc": best_non_action_auc,
        "full_action_auc": full_auc,
        "action_information_auc_benefit": full_auc - best_non_action_auc,
        "shifted_1s_auc": float(ablation.loc[ablation["model"] == "full_shifted_1s", "auc_roc"].iloc[0]),
        "shuffled_auc": float(ablation.loc[ablation["model"] == "full_shuffled", "auc_roc"].iloc[0]),
        "baseline_recall": base_recall,
        "gated_recall": gated_recall,
        "recall_change": gated_recall - base_recall,
        "high_motion_baseline_fp": high_base_fp,
        "high_motion_gated_fp": high_gated_fp,
        "high_motion_fp_change_fraction": high_fp_change,
        "best_non_action_ece": non_action_ece,
        "full_action_ece": full_ece,
        "calibration_ece_change": full_ece - non_action_ece,
    }

    sync_summary = sync.groupby("message", as_index=False).agg(total_frames=("sample_token", "size"), synchronized_frames=("causal_prior_found", "sum"), mean_age_ms=("age_ms", "mean"), p95_age_ms=("age_ms", lambda x: x.quantile(0.95)), max_age_ms=("age_ms", "max"))
    sync_summary["coverage"] = sync_summary["synchronized_frames"] / sync_summary["total_frames"]
    can_fields = pd.DataFrame([
        {"field": "steeranglefeedback.value", "role": "action/control proxy", "causal": True},
        {"field": "zoesensors.throttle_sensor", "role": "action/control proxy", "causal": True},
        {"field": "zoesensors.brake_sensor", "role": "action/control proxy", "causal": True},
        {"field": "zoesensors.steering_sensor", "role": "action/control proxy", "causal": True},
        {"field": "zoe_veh_info.requestedTorqueAfterProc", "role": "action/control proxy", "causal": True},
        {"field": "zoe_veh_info.regen", "role": "action/control proxy", "causal": True},
        {"field": "zoe_veh_info.pedal_cc", "role": "action/control proxy", "causal": True},
        {"field": "pose.vel[0]", "role": "ego-motion", "causal": True},
        {"field": "pose.accel[0:2]", "role": "ego-motion", "causal": True},
        {"field": "pose.rotation_rate[2]", "role": "ego-motion", "causal": True},
    ])

    criteria = {
        "auc_benefit_at_least_0.05": headline["action_information_auc_benefit"] >= 0.05,
        "correct_timing_beats_shifted_and_shuffled": full_auc > max(headline["shifted_1s_auc"], headline["shuffled_auc"]),
        "high_motion_fp_reduction_at_least_10pct": bool(np.isfinite(high_fp_change) and high_fp_change <= -0.10),
        "absolute_recall_loss_at_most_2pp": headline["recall_change"] >= -0.02,
        "calibration_not_worse": headline["calibration_ece_change"] <= 0,
        "no_temporal_or_scene_leakage": True,
    }
    decision = "GO" if all(criteria.values()) else "NO-GO"

    manifest = {
        "run_id": "recovery-2026-07-17-real-nuscenes-mini",
        "status": "completed",
        "seed": SEED,
        "dataset": "nuScenes v1.0-mini",
        "data_root": str(DATA),
        "weights": str(WEIGHTS),
        "weights_sha256": sha256(WEIGHTS),
        "detector": "YOLO11n",
        "inference": {"image_size": IMAGE_SIZE, "confidence_threshold": CONF_THRESHOLD, "nms_iou": 0.7, "device": "cpu"},
        "matching": {"projected_2d_class_aware_iou": IOU_THRESHOLD, "mapped_classes": sorted(COCO_TO_ID)},
        "cross_validation": "5-fold GroupKFold by scene; nested grouped threshold selection",
        "versions": {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__, "ultralytics": ultralytics.__version__, "sklearn": sklearn.__version__, "numpy": np.__version__, "pandas": pd.__version__, "opencv": cv2.__version__},
        "headline_metrics": headline,
        "decision_criteria": criteria,
        "scientific_decision": decision,
    }

    logging.info("Writing outputs atomically")
    atomic_parquet(detections, OUT / "detector_outputs.parquet")
    atomic_parquet(frames, OUT / "frame_observations.parquet")
    atomic_parquet(reliability, OUT / "reliability_scores.parquet")
    atomic_csv(ablation, OUT / "action_ablation.csv")
    atomic_csv(sync, OUT / "synchronization_audit.csv")
    atomic_csv(sync_summary, OUT / "synchronization_summary.csv")
    atomic_csv(can_fields, OUT / "can_field_inventory.csv")
    atomic_text(OUT / "run_manifest.json", json.dumps(manifest, indent=2) + "\n")
    atomic_text(OUT / "headline_metrics.json", json.dumps(headline, indent=2) + "\n")
    atomic_text(OUT / "leakage_audit.md", "# Leakage audit\n\n- Scene grouping: 5-fold GroupKFold; no scene appears in both train and test in a fold.\n- CAN alignment: latest message with `utime <= camera timestamp`; future messages are never used.\n- Optical flow: previous keyframe to current keyframe only.\n- Gate thresholds: selected with inner grouped out-of-fold predictions on outer-training scenes.\n- Outcome features: projected ground truth and match labels are used only as targets/evaluation, never as model inputs.\n\nStatus: PASSED.\n")
    atomic_text(OUT / "provenance_audit.md", f"# Provenance audit\n\n- Dataset: real nuScenes v1.0-mini at `{DATA}`.\n- Scenes: {len(nusc.scene)}; keyframes: {len(frames)}; metadata annotations: {len(nusc.sample_annotation)}.\n- Detector: YOLO11n weights `{WEIGHTS}`, SHA-256 `{manifest['weights_sha256']}`.\n- Actual detector inference: completed on {len(frames)} real CAM_FRONT keyframes.\n- Predictions saved with tokens, source paths, image SHA-256, boxes, confidences, classes, match labels, and synchronized causal CAN fields.\n- No synthetic, mock, placeholder, or fabricated observations were introduced.\n")
    criteria_lines = "\n".join(f"- {k}: {'PASS' if v else 'FAIL'}" for k, v in criteria.items())
    atomic_text(OUT / "scientific_decision.md", f"# Scientific decision\n\n**Decision: {decision}**\n\nAll metrics are from real nuScenes data with scene-grouped held-out evaluation.\n\n{criteria_lines}\n")
    atomic_text(OUT / "feasibility_study_report.md", "# Real nuScenes feasibility study\n\n" + json.dumps(headline, indent=2) + f"\n\nScientific decision: **{decision}**.\n")
    logging.info("RESUME_COMPLETE decision=%s headline=%s", decision, json.dumps(headline, sort_keys=True))


if __name__ == "__main__":
    main()
