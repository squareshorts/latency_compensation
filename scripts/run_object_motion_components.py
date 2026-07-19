"""Causal component substitutions and object-level AV2 harm diagnostics.

M0/M1 are read from the corrected propagation outputs. M2--M5 use only
source-time or earlier information. M6/M7 alone use future annotation motion
and are labeled nondeployable diagnostics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation

ROOT = Path(r"C:\work\auto")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from latency_compensation.object_motion_harm import validate_component_substitution
from recalculate_av2_sample import LogData, center_distance, iou, track_history

SOURCE = ROOT / "results" / "av2_confirmation" / "corrected_checkpoints"
CHECKPOINTS = ROOT / "results" / "av2_confirmation" / "checkpoints"
OUT = ROOT / "results" / "object_motion_harm"
MODELS = ("M0", "M1", "M2", "M3", "M4", "M5", "M6", "M7")
METRICS = ("normalized_center_error", "iou", "recall_iou_0_3", "recall_iou_0_5", "recall_iou_0_7")


def box(row, prefix: str) -> np.ndarray:
    return np.asarray([row[f"{prefix}_x1"], row[f"{prefix}_y1"], row[f"{prefix}_x2"], row[f"{prefix}_y2"]], float)


def box_center(value: np.ndarray) -> np.ndarray:
    return np.asarray([(value[0] + value[2]) / 2, (value[1] + value[3]) / 2], float)


def metrics(prediction: np.ndarray, target: np.ndarray, width: int, height: int) -> dict[str, float]:
    overlap = iou(prediction, target)
    return {
        "normalized_center_error": center_distance(prediction, target) / math.hypot(width, height),
        "iou": overlap,
        "recall_iou_0_3": float(overlap >= 0.3),
        "recall_iou_0_5": float(overlap >= 0.5),
        "recall_iou_0_7": float(overlap >= 0.7),
    }


def city_point_from_box(log: LogData, value: np.ndarray, depth: float, timestamp: int) -> np.ndarray:
    center = box_center(value)
    camera = np.asarray([[(center[0] - log.K[0, 2]) / log.K[0, 0] * depth,
                          (center[1] - log.K[1, 2]) / log.K[1, 1] * depth, depth]], float)
    return log.cam_points_to_city(camera, int(timestamp))[0]


def velocity(positions: list[np.ndarray], timestamps: list[int]) -> tuple[np.ndarray, float, np.ndarray]:
    if len(positions) < 2:
        return np.zeros(3), np.nan, np.zeros(3)
    interval = []
    for left, right, t0, t1 in zip(positions[:-1], positions[1:], timestamps[:-1], timestamps[1:]):
        dt = (int(t1) - int(t0)) / 1e9
        if dt > 0:
            interval.append((right - left) / dt)
    if not interval:
        return np.zeros(3), np.nan, np.zeros(3)
    values = np.asarray(interval)
    latest = values[-1].copy()
    speed = np.linalg.norm(latest[:2])
    if speed > 30:
        latest *= 30 / speed
    dispersion = float(np.sqrt(np.mean(np.sum((values[:, :2] - np.median(values[:, :2], axis=0)) ** 2, axis=1)))) if len(values) > 1 else 0.0
    acceleration = (values[-1] - values[-2]) / max((timestamps[-1] - timestamps[-2]) / 1e9, 1e-6) if len(values) > 1 else np.zeros(3)
    return latest, dispersion, acceleration


def annotation_row(log: LogData, timestamp: int, track_uuid: str):
    rows = log.annotations_by_ts.get(int(timestamp))
    if rows is None:
        return None
    found = rows[rows.track_uuid.astype(str) == str(track_uuid)]
    return None if found.empty else found.iloc[0]


def annotation_city_center(log: LogData, annotation) -> np.ndarray | None:
    if annotation is None:
        return None
    point = np.asarray([annotation.tx_m, annotation.ty_m, annotation.tz_m, 1.0], float)
    return (log.pose(int(annotation.timestamp_ns)) @ point)[:3]


def projected_track_box(log: LogData, annotation_ts: int, image_ts: int, track_uuid: str) -> np.ndarray | None:
    for uuid, _, projected in log.projected_targets(int(annotation_ts), int(image_ts)):
        if uuid == str(track_uuid):
            return projected
    return None


def source_table(frame: pd.DataFrame) -> pd.DataFrame:
    sources = frame.sort_values("delta_ms").drop_duplicates(["source_timestamp_ns", "evaluation_track_uuid", "source_index"]).copy()
    return sources.sort_values("source_timestamp_ns").reset_index(drop=True)


def prepare_file(path: Path) -> pd.DataFrame:
    columns = [
        "comparison_id", "log_id", "role", "detector", "source_timestamp_ns", "target_timestamp_ns",
        "source_annotation_timestamp_ns", "target_annotation_timestamp_ns", "delta_ms", "model", "class_id",
        "target_category", "evaluation_track_uuid", "source_x1", "source_y1", "source_x2", "source_y2",
        "target_x1", "target_y1", "target_x2", "target_y2", "pred_x1", "pred_y1", "pred_x2", "pred_y2",
        "estimated_depth_m", "depth_dispersion_m", "lidar_point_count", "track_age", "image_velocity_norm_px_s",
        "object_velocity_norm_m_s", "uncertainty_damping_weight", "ego_translation_m", "yaw_change_rad",
        "yaw_rate_rad_s", "normalized_center_error", "iou", "recall_iou_0_3", "recall_iou_0_5", "recall_iou_0_7",
    ]
    data = pd.read_parquet(path, columns=columns)
    base = data[data.model == "B3"].drop(columns=["model"]).copy()
    base["source_index"] = base.comparison_id.str.split(":").str[3].astype(int)
    for model in ("B3", "B5"):
        selected = data[data.model == model][["comparison_id", "pred_x1", "pred_y1", "pred_x2", "pred_y2", *METRICS]].copy()
        selected = selected.rename(columns={column: f"{model}_{column}" for column in selected.columns if column != "comparison_id"})
        if model == "B3":
            base = base.drop(columns=["pred_x1", "pred_y1", "pred_x2", "pred_y2", *METRICS]).merge(selected, on="comparison_id", how="inner", validate="one_to_one")
        else:
            base = base.merge(selected, on="comparison_id", how="inner", validate="one_to_one")
    return base


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file-limit", type=int)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard-index must be in [0, shard-count)")
    OUT.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_csv(ROOT / "results" / "av2_confirmation" / "cohort_source_metadata.csv")
    split_map = metadata.set_index(metadata.log_id.astype(str)).split.astype(str).to_dict()
    threshold = float(json.loads((ROOT / "results" / "av2_confirmation" / "high_yaw_threshold.json").read_text())["threshold_abs_yaw_rate_rad_s"])
    suffix = "" if args.shard_count == 1 else f".shard-{args.shard_index:02d}-of-{args.shard_count:02d}"
    object_path = OUT / f"object_level_diagnostics{suffix}.parquet"
    writer = None
    component_log_rows, interaction_rows, mechanism_chunks = [], [], []
    files = sorted(SOURCE.glob("*/*/*.propagation.parquet"))
    files = [path for index, path in enumerate(files) if index % args.shard_count == args.shard_index]
    if args.file_limit:
        files = files[:args.file_limit]

    for file_number, path in enumerate(files, start=1):
        frame = prepare_file(path)
        if frame.empty:
            continue
        log_id, role, detector = str(frame.log_id.iloc[0]), str(frame.role.iloc[0]), str(frame.detector.iloc[0])
        log = LogData(split_map[log_id], log_id)
        payload = json.loads((CHECKPOINTS / detector / role / f"{log_id}.json").read_text())
        detections = payload["detections"]
        by_ts: dict[int, list[tuple[int, dict]]] = defaultdict(list)
        for index, detection in enumerate(detections):
            by_ts[int(detection["timestamp_ns"])].append((index, detection))

        sources = source_table(frame)
        source_records = {int(row.source_index): row for row in sources.itertuples(index=False)}
        depth_lookup = {int(row.source_index): float(row.estimated_depth_m) for row in sources.itertuples(index=False)}
        uuid_lookup = {int(row.source_index): str(row.evaluation_track_uuid) for row in sources.itertuples(index=False)}
        same_track = {uuid: group.sort_values("source_timestamp_ns") for uuid, group in sources.groupby("evaluation_track_uuid", observed=True)}
        feature_lookup: dict[int, dict] = {}

        for source in sources.itertuples(index=False):
            index, source_ts = int(source.source_index), int(source.source_timestamp_ns)
            detection = detections[index]
            source_box = np.asarray([detection["x1"], detection["y1"], detection["x2"], detection["y2"]], float)
            deployed_history = track_history(by_ts, source_ts, index, detection)
            dep_positions, dep_times, history_uuids = [], [], []
            for hist_ts, hist_index, hist_detection in deployed_history:
                hist_box = np.asarray([hist_detection["x1"], hist_detection["y1"], hist_detection["x2"], hist_detection["y2"]], float)
                hist_depth = depth_lookup.get(int(hist_index), float(source.estimated_depth_m))
                dep_positions.append(city_point_from_box(log, hist_box, hist_depth, hist_ts))
                dep_times.append(int(hist_ts)); history_uuids.append(uuid_lookup.get(int(hist_index)))
            deployed_velocity, velocity_dispersion, acceleration = velocity(dep_positions, dep_times)

            track_rows = same_track[str(source.evaluation_track_uuid)]
            causal = track_rows[(track_rows.source_timestamp_ns <= source_ts) & (track_rows.source_timestamp_ns >= source_ts - 600_000_000)].tail(4)
            m2_positions, m3_positions, oracle_times = [], [], []
            for hist in causal.itertuples(index=False):
                hist_box = np.asarray([hist.source_x1, hist.source_y1, hist.source_x2, hist.source_y2], float)
                hist_depth = float(hist.estimated_depth_m)
                m2_positions.append(city_point_from_box(log, hist_box, hist_depth, int(hist.source_timestamp_ns)))
                gt_box = projected_track_box(log, int(hist.source_annotation_timestamp_ns), int(hist.source_timestamp_ns), str(hist.evaluation_track_uuid))
                m3_positions.append(city_point_from_box(log, gt_box if gt_box is not None else hist_box, hist_depth, int(hist.source_timestamp_ns)))
                oracle_times.append(int(hist.source_timestamp_ns))
            m2_velocity, _, _ = velocity(m2_positions, oracle_times)
            m3_velocity, _, _ = velocity(m3_positions, oracle_times)

            annotations = log.annotations[(log.annotations.track_uuid.astype(str) == str(source.evaluation_track_uuid)) &
                                          (log.annotations.timestamp_ns <= int(source.source_annotation_timestamp_ns))].sort_values("timestamp_ns").tail(4)
            gt_positions = [annotation_city_center(log, row) for _, row in annotations.iterrows()]
            gt_times = annotations.timestamp_ns.astype(int).tolist()
            m4_velocity, gt_velocity_dispersion, gt_acceleration = velocity(gt_positions, gt_times)
            source_annotation = annotation_row(log, int(source.source_annotation_timestamp_ns), str(source.evaluation_track_uuid))
            source_city = annotation_city_center(log, source_annotation)
            true_depth = np.nan
            if source_city is not None:
                true_depth = float(log.city_points_to_cam(np.asarray([source_city]), source_ts)[0, 2])

            association_iou, association_distance = np.nan, np.nan
            if len(deployed_history) >= 2:
                previous = deployed_history[-2][2]
                previous_box = np.asarray([previous["x1"], previous["y1"], previous["x2"], previous["y2"]], float)
                association_iou = iou(source_box, previous_box)
                association_distance = center_distance(source_box, previous_box) / math.hypot(log.width, log.height)
            association_confidence = float(np.clip((0 if np.isnan(association_iou) else association_iou) * math.exp(-5 * (0 if np.isnan(association_distance) else association_distance)), 0, 1))
            known_history = [value for value in history_uuids[:-1] if value is not None]
            incorrect_association = bool(known_history and any(value != str(source.evaluation_track_uuid) for value in known_history))
            feature_lookup[index] = {
                "detector_confidence": float(detection.get("confidence", detection.get("score", np.nan))),
                "number_historical_detections": len(deployed_history), "association_confidence": association_confidence,
                "association_iou": association_iou, "association_center_distance_normalized": association_distance,
                "estimated_3d_velocity_x_m_s": deployed_velocity[0], "estimated_3d_velocity_y_m_s": deployed_velocity[1],
                "estimated_3d_velocity_z_m_s": deployed_velocity[2], "velocity_dispersion_m_s": velocity_dispersion,
                "acceleration_estimate_m_s2": float(np.linalg.norm(acceleration[:2])), "oracle_past_velocity_x_m_s": m4_velocity[0],
                "oracle_past_velocity_y_m_s": m4_velocity[1], "oracle_past_speed_m_s": float(np.linalg.norm(m4_velocity[:2])),
                "oracle_velocity_dispersion_m_s": gt_velocity_dispersion, "oracle_acceleration_m_s2": float(np.linalg.norm(gt_acceleration[:2])),
                "true_source_depth_m": true_depth, "incorrect_association": incorrect_association,
                "m2_velocity": m2_velocity, "m3_velocity": m3_velocity, "m4_velocity": m4_velocity,
                "deployed_velocity": deployed_velocity, "source_city": source_city,
            }

        object_rows, long_component = [], []
        interaction_accumulator: dict[tuple, list[float]] = defaultdict(list)
        for row in frame.itertuples(index=False):
            source_box = np.asarray([row.source_x1, row.source_y1, row.source_x2, row.source_y2], float)
            target_box = np.asarray([row.target_x1, row.target_y1, row.target_x2, row.target_y2], float)
            b3 = np.asarray([row.B3_pred_x1, row.B3_pred_y1, row.B3_pred_x2, row.B3_pred_y2], float)
            b5 = np.asarray([row.B5_pred_x1, row.B5_pred_y1, row.B5_pred_x2, row.B5_pred_y2], float)
            feature = feature_lookup[int(row.source_index)]
            target_annotation = annotation_row(log, int(row.target_annotation_timestamp_ns), str(row.evaluation_track_uuid))
            target_city = annotation_city_center(log, target_annotation)
            future_displacement = np.zeros(3) if feature["source_city"] is None or target_city is None else target_city - feature["source_city"]
            horizon = int(row.delta_ms) / 1000.0
            true_depth = feature["true_source_depth_m"] if np.isfinite(feature["true_source_depth_m"]) and feature["true_source_depth_m"] > 0.1 else float(row.estimated_depth_m)
            predictions = {
                "M0": b3, "M1": b5,
                "M2": log.propagate_geometry(source_box, float(row.estimated_depth_m), int(row.source_timestamp_ns), int(row.target_timestamp_ns), feature["m2_velocity"] * horizon * float(row.uncertainty_damping_weight)),
                "M3": log.propagate_geometry(source_box, float(row.estimated_depth_m), int(row.source_timestamp_ns), int(row.target_timestamp_ns), feature["m3_velocity"] * horizon * float(row.uncertainty_damping_weight)),
                "M4": log.propagate_geometry(source_box, float(row.estimated_depth_m), int(row.source_timestamp_ns), int(row.target_timestamp_ns), feature["m4_velocity"] * horizon * float(row.uncertainty_damping_weight)),
                "M5": log.propagate_geometry(source_box, true_depth, int(row.source_timestamp_ns), int(row.target_timestamp_ns), feature["deployed_velocity"] * horizon * float(row.uncertainty_damping_weight)),
                "M6": log.propagate_geometry(source_box, float(row.estimated_depth_m), int(row.source_timestamp_ns), int(row.target_timestamp_ns), future_displacement),
                "M7": log.propagate_geometry(source_box, true_depth, int(row.source_timestamp_ns), int(row.target_timestamp_ns), future_displacement),
            }
            model_metrics = {model: metrics(pred, target_box, log.width, log.height) for model, pred in predictions.items()}
            validate_component_substitution(pd.DataFrame({
                "comparison_id": [row.comparison_id] * len(MODELS), "model": MODELS,
                "future_geometry_input": [model in {"M6", "M7"} for model in MODELS],
            }))
            true_speed = float(np.linalg.norm(future_displacement[:2]) / max(horizon, 1e-6))
            estimated_velocity = feature["deployed_velocity"]
            true_velocity = future_displacement / max(horizon, 1e-6)
            denominator = max(np.linalg.norm(estimated_velocity[:2]) * np.linalg.norm(true_velocity[:2]), 1e-9)
            cosine = float(np.clip(np.dot(estimated_velocity[:2], true_velocity[:2]) / denominator, -1, 1))
            direction_error = float(np.degrees(np.arccos(cosine))) if denominator > 1e-8 else np.nan
            speed_ratio = float(np.linalg.norm(estimated_velocity[:2]) / max(np.linalg.norm(true_velocity[:2]), 1e-6))
            depth_relative_error = float(abs(float(row.estimated_depth_m) - true_depth) / max(true_depth, 1e-6))
            harm = model_metrics["M1"]["normalized_center_error"] - model_metrics["M0"]["normalized_center_error"]
            flags = {
                "incorrect_association": bool(feature["incorrect_association"]),
                "velocity_sign_error": bool(cosine < 0), "velocity_magnitude_error": bool(speed_ratio < 0.5 or speed_ratio > 2.0),
                "velocity_direction_error": bool(np.isfinite(direction_error) and direction_error > 45),
                "unstable_short_track_extrapolation": bool(int(row.track_age) < 3 and feature["velocity_dispersion_m_s"] > 2),
                "depth_induced_projection_error": bool(depth_relative_error > 0.25),
                "missed_object_acceleration": bool(abs(true_speed - feature["oracle_past_speed_m_s"]) > 2),
                "detector_box_jitter_interpreted_as_motion": bool(feature["velocity_dispersion_m_s"] > 3 and feature["oracle_velocity_dispersion_m_s"] < 1),
                "clipping_or_boundary_error": bool(np.any(np.isclose(b5, [0, 0, log.width - 1, log.height - 1], atol=1e-6))),
                "class_specific_failure": False, "distance_dependent_failure": bool(true_depth > 50 and harm > 0),
            }
            primary_failure = "unclassified"
            for name in ("incorrect_association", "velocity_sign_error", "velocity_direction_error", "velocity_magnitude_error",
                         "detector_box_jitter_interpreted_as_motion", "unstable_short_track_extrapolation", "depth_induced_projection_error",
                         "missed_object_acceleration", "clipping_or_boundary_error", "distance_dependent_failure"):
                if flags[name]:
                    primary_failure = name; break
            object_record = {
                "comparison_id": row.comparison_id, "log_id": log_id, "role": role, "detector": detector,
                "source_timestamp_ns": int(row.source_timestamp_ns), "target_timestamp_ns": int(row.target_timestamp_ns),
                "delta_ms": int(row.delta_ms), "high_yaw": abs(float(row.yaw_rate_rad_s)) >= threshold,
                "track_age": int(row.track_age), "number_historical_detections": feature["number_historical_detections"],
                "association_confidence": feature["association_confidence"], "detector_confidence": feature["detector_confidence"],
                "estimated_image_velocity_px_s": float(row.image_velocity_norm_px_s),
                "estimated_3d_velocity_m_s": float(row.object_velocity_norm_m_s),
                "estimated_3d_velocity_x_m_s": feature["estimated_3d_velocity_x_m_s"],
                "estimated_3d_velocity_y_m_s": feature["estimated_3d_velocity_y_m_s"],
                "velocity_dispersion_m_s": feature["velocity_dispersion_m_s"], "acceleration_estimate_m_s2": feature["acceleration_estimate_m_s2"],
                "source_depth_m": float(row.estimated_depth_m), "true_source_depth_m": true_depth,
                "depth_uncertainty_m": float(row.depth_dispersion_m) if np.isfinite(row.depth_dispersion_m) else np.nan,
                "depth_relative_error": depth_relative_error, "lidar_point_count": int(row.lidar_point_count),
                "object_distance_m": true_depth, "object_class": str(row.target_category),
                "source_box_area_px2": float(max(row.source_x2 - row.source_x1, 0) * max(row.source_y2 - row.source_y1, 0)),
                "object_displacement_m": float(np.linalg.norm(future_displacement[:2])), "true_object_speed_m_s": true_speed,
                "ego_translation_m": float(row.ego_translation_m), "ego_yaw_change_rad": float(row.yaw_change_rad),
                "ego_yaw_rate_rad_s": float(row.yaw_rate_rad_s), "uncertainty_damping_weight": float(row.uncertainty_damping_weight),
                "velocity_direction_error_deg": direction_error, "velocity_speed_ratio": speed_ratio,
                "B3_error": model_metrics["M0"]["normalized_center_error"], "B5_error": model_metrics["M1"]["normalized_center_error"],
                "delta_harm": harm, "B3_iou": model_metrics["M0"]["iou"], "B5_iou": model_metrics["M1"]["iou"],
                "primary_failure_mechanism": primary_failure, **flags,
            }
            for model in MODELS:
                for metric_name, value in model_metrics[model].items():
                    object_record[f"{model}_{metric_name}"] = value
            object_rows.append(object_record)
            for model in MODELS:
                long_component.append({"comparison_id": row.comparison_id, "log_id": log_id, "role": role, "detector": detector,
                                       "delta_ms": int(row.delta_ms), "high_yaw": object_record["high_yaw"], "model": model,
                                       "future_geometry_input": model in {"M6", "M7"}, **model_metrics[model]})

            # Full 2x2x2x2 factorial on a deterministic 1% diagnostic sample.
            if int.from_bytes(hashlib.sha256(row.comparison_id.encode()).digest()[:4], "big") % 100 == 0:
                for association_oracle in (0, 1):
                    for depth_oracle in (0, 1):
                        for velocity_oracle in (0, 1):
                            for ego_transform in (0, 1):
                                chosen_depth = true_depth if depth_oracle else float(row.estimated_depth_m)
                                chosen_velocity = feature["m2_velocity"] if association_oracle else feature["deployed_velocity"]
                                displacement = future_displacement if velocity_oracle else chosen_velocity * horizon * float(row.uncertainty_damping_weight)
                                target_ts = int(row.target_timestamp_ns) if ego_transform else int(row.source_timestamp_ns)
                                prediction = log.propagate_geometry(source_box, chosen_depth, int(row.source_timestamp_ns), target_ts, displacement)
                                interaction_accumulator[(association_oracle, depth_oracle, velocity_oracle, ego_transform)].append(
                                    metrics(prediction, target_box, log.width, log.height)["normalized_center_error"]
                                )

        objects = pd.DataFrame(object_rows)
        table = pa.Table.from_pandas(objects, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(object_path, table.schema, compression="zstd")
        writer.write_table(table)
        mechanism_chunks.append(objects[["detector", "role", "delta_ms", "high_yaw", "delta_harm", "primary_failure_mechanism",
                                         "object_class", "object_distance_m"]])
        components = pd.DataFrame(long_component)
        for keys, group in components.groupby(["role", "detector", "log_id", "delta_ms", "high_yaw", "model"], observed=True):
            rec = dict(zip(["role", "detector", "log_id", "delta_ms", "high_yaw", "model"], keys)); rec["comparisons"] = len(group)
            for metric_name in METRICS:
                rec[metric_name] = float(group[metric_name].median())
            component_log_rows.append(rec)
        for factors, values in interaction_accumulator.items():
            interaction_rows.append({"role": role, "detector": detector, "log_id": log_id,
                                     "association_oracle": factors[0], "depth_oracle": factors[1], "velocity_oracle": factors[2],
                                     "ego_motion_transform": factors[3], "sample_comparisons": len(values),
                                     "median_normalized_center_error": float(np.median(values))})
        print(json.dumps({"file": file_number, "total": len(files), "log": log_id, "detector": detector, "objects": len(objects)}), flush=True)

    if writer is not None:
        writer.close()
    component_logs = pd.DataFrame(component_log_rows)
    aggregate = []
    for keys, group in component_logs.groupby(["role", "detector", "delta_ms", "high_yaw", "model"], observed=True):
        rec = dict(zip(["role", "detector", "delta_ms", "high_yaw", "model"], keys)); rec["logs"] = group.log_id.nunique(); rec["comparisons"] = group.comparisons.sum()
        for metric_name in METRICS:
            rec[f"median_of_log_medians_{metric_name}"] = float(group[metric_name].median())
        aggregate.append(rec)
    component = pd.DataFrame(aggregate).sort_values(["role", "detector", "delta_ms", "high_yaw", "model"])
    component.to_csv(OUT / f"component_substitution{suffix}.csv", index=False, float_format="%.12g")
    component_logs.to_csv(OUT / f"component_log_rows{suffix}.csv", index=False, float_format="%.12g")
    component_logs[component_logs.model.isin(["M0", "M1", "M4", "M6", "M7"])].to_csv(OUT / f"oracle_diagnostics{suffix}.csv", index=False, float_format="%.12g")

    effects = []
    for keys, group in component.groupby(["role", "detector", "delta_ms", "high_yaw"], observed=True):
        indexed = group.set_index("model")
        for candidate, reference, interpretation in [
            ("M2", "M1", "association"), ("M3", "M2", "detector_history_boxes"), ("M4", "M3", "2D_vs_3D_history"),
            ("M5", "M1", "source_depth"), ("M6", "M0", "object_motion_in_principle"), ("M7", "M0", "full_oracle_ceiling")]:
            if candidate not in indexed.index or reference not in indexed.index:
                continue
            c = indexed.loc[candidate, "median_of_log_medians_normalized_center_error"]
            r = indexed.loc[reference, "median_of_log_medians_normalized_center_error"]
            effects.append({**dict(zip(["role", "detector", "delta_ms", "high_yaw"], keys)), "candidate": candidate, "reference": reference,
                            "isolated_component": interpretation, "absolute_error_difference": c-r, "relative_error_difference": c/r-1})
    pd.DataFrame(effects).to_csv(OUT / f"component_effects_by_detector{suffix}.csv", index=False, float_format="%.12g")

    mechanisms = pd.concat(mechanism_chunks, ignore_index=True)
    harmed = mechanisms[mechanisms.delta_harm > 0]
    mechanism_summary = harmed.groupby(["detector", "role", "primary_failure_mechanism"], observed=True).agg(
        harmed_objects=("delta_harm", "size"), median_delta_harm=("delta_harm", "median")
    ).reset_index()
    totals = harmed.groupby(["detector", "role"], observed=True).size().rename("total_harmed").reset_index()
    mechanism_summary = mechanism_summary.merge(totals, on=["detector", "role"])
    mechanism_summary["percent_of_harmed"] = mechanism_summary.harmed_objects / mechanism_summary.total_harmed * 100
    mechanism_summary.to_csv(OUT / f"failure_mechanism_summary{suffix}.csv", index=False, float_format="%.12g")
    interactions = pd.DataFrame(interaction_rows)
    interactions.to_csv(OUT / f"component_interactions{suffix}.csv", index=False, float_format="%.12g")
    print(json.dumps({"object_rows": int(pq.ParquetFile(object_path).metadata.num_rows), "component_log_rows": len(component_logs),
                      "factorial_log_cells": len(interactions)}))


if __name__ == "__main__":
    main()
