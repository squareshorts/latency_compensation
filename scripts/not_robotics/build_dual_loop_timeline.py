import bisect
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SDK = Path(r"C:\work\auto\external\nuscenes-devkit\python-sdk")
sys.path.insert(0, str(SDK))
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
from nuscenes.nuscenes import NuScenes

ROOT = Path(r"C:\work\auto")
DATA = ROOT / "data" / "nuscenes"
OUT = ROOT / "results" / "not_dual_loop_latency"
PRIOR = ROOT / "results" / "not_robotics_real_feasibility"
HORIZONS_MS = [100, 200, 300, 400, 500]


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


class CausalCan:
    def __init__(self, api):
        self.api = api
        self.cache = {}

    def prior(self, scene, message, timestamp):
        key = (scene, message)
        if key not in self.cache:
            records = sorted(self.api.get_messages(scene, message), key=lambda x: x["utime"])
            self.cache[key] = (records, [x["utime"] for x in records])
        records, times = self.cache[key]
        index = bisect.bisect_right(times, timestamp) - 1
        if index < 0:
            return None, np.nan
        return records[index], (timestamp - records[index]["utime"]) / 1000.0


def scalar(message, key):
    if message is None:
        return np.nan
    try:
        return float(message[key])
    except (KeyError, TypeError, ValueError):
        return np.nan


def vector(message, key, index):
    if message is None:
        return np.nan
    try:
        return float(message[key][index])
    except (KeyError, IndexError, TypeError, ValueError):
        return np.nan


def scene_condition(description):
    text = description.lower()
    if "night" in text:
        return "night"
    if "rain" in text or "wet" in text:
        return "rain_or_wet"
    return "day_or_unspecified"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    nusc = NuScenes(version="v1.0-mini", dataroot=str(DATA), verbose=False)
    can = CausalCan(NuScenesCanBus(dataroot=str(DATA)))
    annotation_to_instance = {row["token"]: row["instance_token"] for row in nusc.sample_annotation}

    timeline_rows = []
    scene_frame_tokens = {}
    scene_instances = {}
    for scene in nusc.scene:
        scene_name = scene["name"]
        condition = scene_condition(scene["description"])
        first_sample = nusc.get("sample", scene["first_sample_token"])
        sample_data = nusc.get("sample_data", first_sample["data"]["CAM_FRONT"])
        chain = []
        while True:
            chain.append(sample_data)
            if not sample_data["next"]:
                break
            sample_data = nusc.get("sample_data", sample_data["next"])
        scene_frame_tokens[scene_name] = [row["token"] for row in chain]
        keyframes = [row for row in chain if row["is_key_frame"]]
        key_times = [row["timestamp"] for row in keyframes]
        key_instance_map = {}
        for row in keyframes:
            sample = nusc.get("sample", row["sample_token"])
            key_instance_map[row["token"]] = sorted(annotation_to_instance[token] for token in sample["anns"])
        scene_instances[scene_name] = key_instance_map

        for index, row in enumerate(chain):
            timestamp = int(row["timestamp"])
            image_path = DATA / row["filename"]
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            pose = nusc.get("ego_pose", row["ego_pose_token"])
            calibration = nusc.get("calibrated_sensor", row["calibrated_sensor_token"])
            nearest_key_index = int(np.argmin(np.abs(np.asarray(key_times, dtype=np.int64) - timestamp)))
            nearest_key = keyframes[nearest_key_index]
            imu, imu_age = can.prior(scene_name, "ms_imu", timestamp)
            pose_can, pose_age = can.prior(scene_name, "pose", timestamp)
            steer, steer_age = can.prior(scene_name, "steeranglefeedback", timestamp)
            sensors, sensors_age = can.prior(scene_name, "zoesensors", timestamp)
            vehicle, vehicle_age = can.prior(scene_name, "zoe_veh_info", timestamp)
            timeline_rows.append({
                "scene_name": scene_name,
                "scene_token": scene["token"],
                "scene_description": scene["description"],
                "condition": condition,
                "frame_index": index,
                "sample_data_token": row["token"],
                "sample_token": row["sample_token"],
                "timestamp_us": timestamp,
                "image_path": str(image_path),
                "is_key_frame": bool(row["is_key_frame"]),
                "previous_frame_token": row["prev"],
                "next_frame_token": row["next"],
                "width": int(row["width"]),
                "height": int(row["height"]),
                "ego_pose_token": row["ego_pose_token"],
                "ego_translation_json": json.dumps(pose["translation"], separators=(",", ":")),
                "ego_rotation_json": json.dumps(pose["rotation"], separators=(",", ":")),
                "calibrated_sensor_token": row["calibrated_sensor_token"],
                "camera_translation_json": json.dumps(calibration["translation"], separators=(",", ":")),
                "camera_rotation_json": json.dumps(calibration["rotation"], separators=(",", ":")),
                "camera_intrinsic_json": json.dumps(calibration["camera_intrinsic"], separators=(",", ":")),
                "nearest_annotated_frame_token": nearest_key["token"],
                "nearest_annotated_timestamp_us": int(nearest_key["timestamp"]),
                "nearest_annotation_offset_ms": (int(nearest_key["timestamp"]) - timestamp) / 1000.0,
                "visible_instance_tokens_json": json.dumps(key_instance_map.get(row["token"], []), separators=(",", ":")),
                "annotation_count": len(key_instance_map.get(row["token"], [])),
                "imu_message_timestamp_us": int(imu["utime"]) if imu else np.nan,
                "imu_age_ms": imu_age,
                "imu_accel_x": vector(imu, "linear_accel", 0),
                "imu_accel_y": vector(imu, "linear_accel", 1),
                "imu_accel_z": vector(imu, "linear_accel", 2),
                "imu_yaw_rate": vector(imu, "rotation_rate", 2),
                "pose_message_timestamp_us": int(pose_can["utime"]) if pose_can else np.nan,
                "pose_age_ms": pose_age,
                "pose_velocity_x": vector(pose_can, "vel", 0),
                "pose_accel_x": vector(pose_can, "accel", 0),
                "pose_accel_y": vector(pose_can, "accel", 1),
                "pose_yaw_rate": vector(pose_can, "rotation_rate", 2),
                "steer_message_timestamp_us": int(steer["utime"]) if steer else np.nan,
                "steer_age_ms": steer_age,
                "steer_angle_feedback": scalar(steer, "value"),
                "sensor_message_timestamp_us": int(sensors["utime"]) if sensors else np.nan,
                "sensor_age_ms": sensors_age,
                "throttle_sensor": scalar(sensors, "throttle_sensor"),
                "brake_sensor": scalar(sensors, "brake_sensor"),
                "steering_sensor": scalar(sensors, "steering_sensor"),
                "vehicle_message_timestamp_us": int(vehicle["utime"]) if vehicle else np.nan,
                "vehicle_age_ms": vehicle_age,
                "requested_torque": scalar(vehicle, "requestedTorqueAfterProc"),
                "regen": scalar(vehicle, "regen"),
                "pedal_cc": scalar(vehicle, "pedal_cc"),
                "front_left_wheel_speed": scalar(vehicle, "FL_wheel_speed"),
                "front_right_wheel_speed": scalar(vehicle, "FR_wheel_speed"),
                "rear_left_wheel_speed": scalar(vehicle, "RL_wheel_speed"),
                "rear_right_wheel_speed": scalar(vehicle, "RR_wheel_speed"),
            })

    timeline = pd.DataFrame(timeline_rows).sort_values(["scene_name", "frame_index"]).reset_index(drop=True)
    if len(timeline) != 2342 or int(timeline["is_key_frame"].sum()) != 404:
        raise RuntimeError(f"Unexpected timeline counts: {len(timeline)} frames, {timeline['is_key_frame'].sum()} keyframes")
    for message_column in ["imu_message_timestamp_us", "pose_message_timestamp_us", "steer_message_timestamp_us", "sensor_message_timestamp_us", "vehicle_message_timestamp_us"]:
        valid = timeline[message_column].notna()
        if not (timeline.loc[valid, message_column] <= timeline.loc[valid, "timestamp_us"]).all():
            raise RuntimeError(f"Future CAN leakage in {message_column}")
    atomic_parquet(timeline, OUT / "streaming_timeline.parquet")

    latency_rows = []
    audit_rows = []
    for scene_name, group in timeline.groupby("scene_name", sort=False):
        group = group.sort_values("timestamp_us").reset_index(drop=True)
        times = group["timestamp_us"].to_numpy(np.int64)
        scene_latency = []
        for source_index, source in group.iterrows():
            source_instances = set(json.loads(source["visible_instance_tokens_json"]))
            for horizon in HORIZONS_MS:
                ideal = int(source["timestamp_us"] + horizon * 1000)
                target_index = bisect.bisect_right(times.tolist(), ideal) - 1
                if target_index <= source_index:
                    continue
                target = group.iloc[target_index]
                target_instances = set(json.loads(target["visible_instance_tokens_json"]))
                row = {
                    "scene_name": scene_name,
                    "latency_ms": horizon,
                    "source_frame_token": source["sample_data_token"],
                    "source_timestamp_us": int(source["timestamp_us"]),
                    "source_is_annotated": bool(source["is_key_frame"]),
                    "target_frame_token": target["sample_data_token"],
                    "target_timestamp_us": int(target["timestamp_us"]),
                    "target_is_annotated": bool(target["is_key_frame"]),
                    "actual_elapsed_ms": (int(target["timestamp_us"]) - int(source["timestamp_us"])) / 1000.0,
                    "availability_to_frame_offset_ms": (ideal - int(target["timestamp_us"])) / 1000.0,
                    "both_endpoints_annotated": bool(source["is_key_frame"] and target["is_key_frame"]),
                    "common_visible_instances": len(source_instances & target_instances) if source["is_key_frame"] and target["is_key_frame"] else 0,
                }
                latency_rows.append(row)
                scene_latency.append(row)
        scene_latency = pd.DataFrame(scene_latency)
        for horizon in HORIZONS_MS:
            rows = scene_latency[scene_latency["latency_ms"] == horizon]
            annotated = rows[rows["both_endpoints_annotated"]]
            usable = annotated[annotated["common_visible_instances"] > 0]
            audit_rows.append({
                "scene_name": scene_name,
                "latency_ms": horizon,
                "real_frame_pairs": len(rows),
                "annotated_source_pairs": int(rows["source_is_annotated"].sum()),
                "annotated_target_pairs": int(rows["target_is_annotated"].sum()),
                "both_endpoints_annotated_pairs": len(annotated),
                "pairs_with_persistent_visible_instance": len(usable),
                "persistent_instance_comparisons": int(usable["common_visible_instances"].sum()),
                "median_actual_elapsed_ms": float(rows["actual_elapsed_ms"].median()) if len(rows) else np.nan,
                "median_availability_to_frame_offset_ms": float(rows["availability_to_frame_offset_ms"].median()) if len(rows) else np.nan,
                "primary_endpoint_observable": bool(horizon == 300 and len(usable) > 0),
            })
    latency_pairs = pd.DataFrame(latency_rows)
    timing_audit = pd.DataFrame(audit_rows)
    atomic_parquet(latency_pairs, OUT / "latency_pairs.parquet")
    atomic_csv(timing_audit, OUT / "timing_audit.csv")

    detector = pd.read_parquet(PRIOR / "detector_outputs.parquet")
    frames = pd.read_parquet(PRIOR / "frame_observations.parquet")
    data_inventory = pd.DataFrame([
        {"asset": "CAM_FRONT images", "path": str(DATA / "samples" / "CAM_FRONT") + ";" + str(DATA / "sweeps" / "CAM_FRONT"), "count": len(timeline), "status": "reused_verified_real"},
        {"asset": "annotated CAM_FRONT keyframes", "path": str(DATA / "v1.0-mini" / "sample_data.json"), "count": int(timeline["is_key_frame"].sum()), "status": "reused_verified_real"},
        {"asset": "nuScenes scenes", "path": str(DATA / "v1.0-mini" / "scene.json"), "count": timeline["scene_name"].nunique(), "status": "reused_verified_real"},
        {"asset": "sample annotations", "path": str(DATA / "v1.0-mini" / "sample_annotation.json"), "count": len(nusc.sample_annotation), "status": "reused_verified_real_keyframes_only"},
        {"asset": "CAN JSON files", "path": str(DATA / "can_bus"), "count": len(list((DATA / "can_bus").rglob("*.json"))), "status": "reused_verified_real"},
        {"asset": "YOLO11n matched predictions", "path": str(PRIOR / "detector_outputs.parquet"), "count": len(detector), "status": "reused_verified_real_keyframes_only"},
        {"asset": "YOLO11n processed keyframes", "path": str(PRIOR / "frame_observations.parquet"), "count": len(frames), "status": "reused_verified_real"},
        {"asset": "YOLO11n weights", "path": str(ROOT / "yolo11n.pt"), "count": 1, "status": "reused_verified_real_inference"},
    ])
    atomic_csv(data_inventory, OUT / "data_inventory.csv")

    primary = timing_audit[timing_audit["latency_ms"] == 300]
    primary_pairs = int(primary["pairs_with_persistent_visible_instance"].sum())
    all_counts = timing_audit.groupby("latency_ms", as_index=False).agg(both_endpoints_annotated_pairs=("both_endpoints_annotated_pairs", "sum"), pairs_with_persistent_visible_instance=("pairs_with_persistent_visible_instance", "sum"), persistent_instance_comparisons=("persistent_instance_comparisons", "sum"))
    counts_text = all_counts.to_csv(index=False)
    reuse_audit = f"""# Reuse audit

- The recovered detector-error reliability study remains untouched as a negative audit under `{PRIOR}`.
- Reused 2,342 verified real CAM_FRONT images, 404 annotated keyframes, 18,538 keyframe annotations, calibrated camera records, per-frame ego poses, parsed CAN streams, and 8,946 real YOLO11n keyframe predictions.
- No archive download or full archive-integrity test was repeated.
- No image, detection, pose, CAN record, object box, or annotation was simulated or interpolated.
- Algorithm latency is represented only by causal source-to-real-frame timing pairs.

## Endpoint observability

nuScenes v1.0-mini supplies object annotations and persistent instance tokens only on keyframes. Intermediate CAM_FRONT sweeps have real images, calibrated sensors, and ego poses, but no object boxes or instance identities. Consequently, a box-propagation endpoint is evaluable only when both the acquisition frame and the frame current at detector availability are annotated keyframes.

```csv
{counts_text}```

At the prespecified 300 ms endpoint, the number of real pairs with annotations and persistent visible instances at both ends is **{primary_pairs}**. Interpolating boxes or identities into sweeps would simulate missing observations and was therefore not performed.
"""
    atomic_text(OUT / "reuse_audit.md", reuse_audit)

    leakage = """# Leakage audit

- CAN/IMU lookup uses `bisect_right(timestamp) - 1`; every retained message timestamp is less than or equal to its camera timestamp.
- Detector availability uses the latest real camera frame at or before `t + delay`; no later image is used.
- Object identities are populated only on genuine annotated keyframes.
- No annotation interpolation, pseudo-labeling, synthetic detections, or future object velocity is used.
- The intended validation grouping is complete nuScenes scenes.

Status: PASSED for the completed timeline and timing audit. Propagation-model leakage cannot be audited because those models were not run.
"""
    atomic_text(OUT / "leakage_audit.md", leakage)

    novelty = """# Novelty assessment

The proposed framing—fast inertial/action prediction plus slower visual-residual correction for streaming detector latency—is distinct from claiming invention of ego-motion compensation. The current dataset can support real camera-motion and timing experiments, but nuScenes mini cannot validate the prespecified 100–400 ms object-box endpoints without interpolated object annotations. Therefore preemptive box compensation, complementary-loop benefit, high-turn improvement, and an operating envelope remain unassessed.
"""
    atomic_text(OUT / "novelty_assessment.md", novelty)

    decision = f"""# Scientific decision

**Decision: BLOCKED; publication route NO-GO on nuScenes v1.0-mini alone.**

The prespecified principal endpoint requires persistent object boxes at acquisition and 300 ms evaluation times. The real dataset provides zero such 300 ms pairs. Producing the requested center-error, IoU, recall, fragmentation, jitter, dual-loop ablations, or success decision would require interpolated/pseudo-labeled observations, which the study forbids.

This is an evidence-availability result, not evidence that the dual-loop method fails. A dataset with densely annotated video frames or an approved annotation protocol is required.
"""
    atomic_text(OUT / "scientific_decision.md", decision)

    readme = """# Action-conditioned dual-loop latency compensation

Status: partially completed and scientifically blocked.

Completed artifacts cover verified-data reuse, all-frame streaming chronology, per-frame pose/calibration, causal CAN/IMU alignment, detector-availability emulation, and annotation-endpoint observability. Propagation metrics and figures were intentionally not generated because the primary 300 ms object endpoint has no real annotations.
"""
    atomic_text(OUT / "README.md", readme)

    blocked_paths = [
        "motion_forecasts.parquet", "propagated_boxes.parquet", "latency_metrics.csv", "scene_heldout_metrics.csv",
        "action_ablation.csv", "loop_ablation.csv", "operating_envelope.csv", "failure_boundary.csv",
    ]
    manifest = {
        "study": "action-conditioned dual-loop latency compensation for streaming object detection",
        "status": "blocked_after_observability_audit",
        "data_root": str(DATA),
        "real_scenes": int(timeline["scene_name"].nunique()),
        "real_cam_front_frames": int(len(timeline)),
        "annotated_keyframes": int(timeline["is_key_frame"].sum()),
        "latency_horizons_ms": HORIZONS_MS,
        "primary_endpoint": "300 ms high-yaw propagated-box center error",
        "primary_real_annotated_pairs": primary_pairs,
        "blocker": "nuScenes object annotations and instance tokens exist only on keyframes; no real 300 ms annotated endpoint pairs",
        "prohibited_workaround_not_used": "annotation or object-identity interpolation into sweeps",
        "blocked_output_paths": [str(OUT / name) for name in blocked_paths],
        "input_hashes": {
            "prior_detector_outputs_sha256": sha256(PRIOR / "detector_outputs.parquet"),
            "prior_frame_observations_sha256": sha256(PRIOR / "frame_observations.parquet"),
            "yolo11n_sha256": sha256(ROOT / "yolo11n.pt"),
        },
    }
    atomic_text(OUT / "run_manifest.json", json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"frames": len(timeline), "keyframes": int(timeline["is_key_frame"].sum()), "latency_pairs": len(latency_pairs), "primary_300ms_usable_pairs": primary_pairs, "decision": "BLOCKED"}, indent=2))


if __name__ == "__main__":
    main()
