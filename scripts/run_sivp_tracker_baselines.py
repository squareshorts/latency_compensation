"""Run causal prediction-only tracker baselines on frozen AV2 comparisons."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"C:\work\auto")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from latency_compensation.causal_trackers import ByteTrackAdapter, OCSortAdapter
from recalculate_av2_sample import center_distance, iou

SOURCE = ROOT / "results" / "av2_confirmation"
OUTPUT = ROOT / "results" / "sivp_strengthening" / "tracker_low_level"
MODEL_MAP = {"B0": "T0", "B1": "T1", "B2": "T2", "B3": "T5", "B5": "T6"}


def atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float | int]:
    overlap = iou(prediction, target)
    return {
        "center_error_px": center_distance(prediction, target),
        "normalized_center_error": center_distance(prediction, target) / np.hypot(1550, 2048),
        "iou": overlap,
        "recall_iou_0_3": int(overlap >= 0.3),
        "recall_iou_0_5": int(overlap >= 0.5),
        "recall_iou_0_7": int(overlap >= 0.7),
    }


def process(detector: str, log_id: str) -> pd.DataFrame:
    corrected_path = (SOURCE / "corrected_checkpoints" / detector / "heldout" / f"{log_id}.propagation.parquet"
                      if detector != "rtdetr_l" else
                      ROOT / "results" / "sivp_strengthening" / "third_detector_propagation" / "heldout" / f"{log_id}.parquet")
    frozen = pd.read_parquet(corrected_path)
    selected = frozen[frozen.model.isin(MODEL_MAP)].copy()
    selected["model"] = selected.model.map(MODEL_MAP)
    selected["future_detection_input"] = False
    selected["association_history_max_timestamp_ns"] = selected.source_timestamp_ns
    base_columns = list(selected.columns)

    comparisons = frozen[frozen.model == "B0"].copy()
    comparisons["detection_index"] = comparisons.comparison_id.map(lambda value: int(str(value).split(":")[3]))
    requested = {(int(row.source_timestamp_ns), int(row.detection_index), int(row.delta_ms)): row
                 for row in comparisons.itertuples(index=False)}
    requested_by_detection: dict[tuple[int, int], list[int]] = {}
    for source_timestamp, detection_index, delta_ms in requested:
        requested_by_detection.setdefault((source_timestamp, detection_index), []).append(delta_ms)
    checkpoint_path = (SOURCE / "checkpoints" / detector / "heldout" / f"{log_id}.json"
                       if detector != "rtdetr_l" else
                       ROOT / "results" / "sivp_strengthening" / "third_detector_checkpoints" / "heldout" / f"{log_id}.json")
    payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    by_timestamp: dict[int, list[tuple[int, dict]]] = {}
    for index, detection in enumerate(payload["detections"]):
        by_timestamp.setdefault(int(detection["timestamp_ns"]), []).append((index, detection))

    byte, oc = ByteTrackAdapter(), OCSortAdapter()
    rows: list[dict] = []
    for timestamp in sorted(by_timestamp):
        indexed = by_timestamp[timestamp]
        local_detections = [detection for _, detection in indexed]
        start = time.perf_counter_ns()
        byte_assigned = byte.update(timestamp, local_detections)
        byte_update_ms = (time.perf_counter_ns() - start) / 1e6 / max(len(indexed), 1)
        start = time.perf_counter_ns()
        oc_assigned = oc.update(timestamp, local_detections)
        oc_update_ms = (time.perf_counter_ns() - start) / 1e6 / max(len(indexed), 1)
        for local_index, (global_index, detection) in enumerate(indexed):
            for delta_ms in requested_by_detection.get((timestamp, global_index), []):
                reference = requested[(timestamp, global_index, delta_ms)]
                target_timestamp = int(reference.target_timestamp_ns)
                source_box = np.asarray([detection["x1"], detection["y1"], detection["x2"], detection["y2"]], float)
                target_box = np.asarray([reference.target_x1, reference.target_y1, reference.target_x2, reference.target_y2], float)
                for model, assigned, update_ms in (("T3", byte_assigned, byte_update_ms), ("T4", oc_assigned, oc_update_ms)):
                    start = time.perf_counter_ns()
                    track = assigned.get(local_index)
                    prediction = source_box.copy() if track is None else track.forecast(target_timestamp)
                    runtime_ms = update_ms + (time.perf_counter_ns() - start) / 1e6
                    row = reference._asdict()
                    row.update({
                        "model": model,
                        "pred_x1": prediction[0], "pred_y1": prediction[1],
                        "pred_x2": prediction[2], "pred_y2": prediction[3],
                        "runtime_ms": runtime_ms,
                        "track_age": 1 if track is None else track.hits,
                        "future_geometry_input": False,
                        "future_detection_input": False,
                        "association_history_max_timestamp_ns": timestamp,
                        **metrics(prediction, target_box),
                    })
                    rows.append(row)
    tracker = pd.DataFrame(rows)
    if len(tracker) != len(comparisons) * 2:
        raise RuntimeError(f"Tracker row mismatch for {detector}/{log_id}: {len(tracker)} != {len(comparisons) * 2}")
    for column in base_columns:
        if column not in tracker:
            tracker[column] = np.nan
    result = pd.concat([selected, tracker[selected.columns]], ignore_index=True)
    counts = result.groupby("comparison_id").model.nunique()
    if not (counts == 7).all():
        raise RuntimeError(f"Incomplete T0-T6 model sets for {detector}/{log_id}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--detector", choices=["yolo11n", "yolo11s", "rtdetr_l"])
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--log-limit", type=int)
    args = parser.parse_args()
    logs = [line.strip() for line in (ROOT / "configs" / "av2_heldout_logs.txt").read_text().splitlines() if line.strip()]
    logs = [value for index, value in enumerate(logs) if index % args.shard_count == args.shard_index]
    if args.log_limit:
        logs = logs[: args.log_limit]
    detectors = [args.detector] if args.detector else ["yolo11n", "yolo11s"]
    for detector in detectors:
        for log_id in logs:
            destination = OUTPUT / detector / f"{log_id}.parquet"
            if destination.exists():
                print(json.dumps({"detector": detector, "log": log_id, "reused": True}))
                continue
            result = process(detector, log_id)
            atomic_parquet(destination, result)
            print(json.dumps({"detector": detector, "log": log_id, "rows": len(result), "reused": False}))


if __name__ == "__main__":
    main()
