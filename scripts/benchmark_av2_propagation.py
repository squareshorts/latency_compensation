"""Warm microbenchmark of B0--B5 propagation math on the audit sample."""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd

from recalculate_av2_sample import LogData, box_state, image_velocity_prediction, json_array, kalman_prediction, state_box


ROOT = Path(r"C:\work\auto")
OUT = ROOT / "results" / "av2_confirmation"
REPETITIONS = 2000
WARMUP = 200


def parse_history(text: str):
    result = []
    for item in json.loads(text):
        box = item["box"]
        result.append((int(item["timestamp_ns"]), int(item["detector_index"]),
                       {"x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3]}))
    return result


def main() -> None:
    sample = pd.read_parquet(OUT / "row_level_recalculation.parquet")
    unique = sample.sort_values("comparison_id").drop_duplicates(["log_id", "detector", "source_timestamp_ns"]).head(1000).copy()
    split_map = pd.read_csv(OUT / "cohort_source_metadata.csv").set_index("log_id").split.astype(str).to_dict()
    logs = {log_id: LogData(split_map[log_id], log_id) for log_id in unique.log_id.unique()}
    prepared = []
    for row in unique.itertuples(index=False):
        source = np.asarray(json.loads(row.source_detector_box_json), float)
        velocity = np.asarray(json.loads(row.estimated_object_velocity_json), float)
        prepared.append((row, source, parse_history(row.track_history_json), velocity, logs[row.log_id]))

    def operation(model, item):
        row, source, history, velocity, log = item
        if model == "B0": return source.copy()
        if model == "B1": return image_velocity_prediction(history, int(row.target_timestamp_ns))[0]
        if model == "B2": return kalman_prediction(history, int(row.target_timestamp_ns))[0]
        if model == "B3": return log.propagate_geometry(source, float(row.estimated_depth_m), int(row.source_timestamp_ns), int(row.target_timestamp_ns), np.zeros(3))
        displacement = velocity * (int(row.delta_ms) / 1000.0)
        if model == "B5": displacement *= float(row.uncertainty_damping_weight)
        return log.propagate_geometry(source, float(row.estimated_depth_m), int(row.source_timestamp_ns), int(row.target_timestamp_ns), displacement)

    rows = []
    for model in ("B0", "B1", "B2", "B3", "B4", "B5"):
        for index in range(WARMUP): operation(model, prepared[index % len(prepared)])
        elapsed = np.empty(REPETITIONS)
        checksum = 0.0
        for index in range(REPETITIONS):
            start = time.perf_counter_ns(); output = operation(model, prepared[index % len(prepared)]); elapsed[index] = (time.perf_counter_ns()-start)/1e6
            checksum += float(output[0])
        rows.append({"model": model, "repetitions": REPETITIONS, "warmup_repetitions": WARMUP,
                     "median_ms_per_object": np.median(elapsed), "p95_ms_per_object": np.quantile(elapsed,.95),
                     "minimum_ms_per_object": elapsed.min(), "maximum_ms_per_object": elapsed.max(), "checksum": checksum})
    runtime = pd.DataFrame(rows)
    total_detections = total_frames = 0
    for log_id in unique.log_id.unique():
        for detector in ("yolo11n", "yolo11s"):
            payload = json.loads((OUT / "checkpoints" / detector / "heldout" / f"{log_id}.json").read_text())
            total_detections += len(payload["detections"]); total_frames += int(payload["frames"])
    objects_per_frame = total_detections / total_frames
    runtime["representative_objects_per_frame"] = objects_per_frame
    b0_median = float(runtime.loc[runtime.model=="B0", "median_ms_per_object"].iloc[0])
    b0_p95 = float(runtime.loc[runtime.model=="B0", "p95_ms_per_object"].iloc[0])
    runtime["median_overhead_ms_per_object_vs_B0"] = runtime.median_ms_per_object - b0_median
    runtime["p95_overhead_ms_per_object_vs_B0"] = runtime.p95_ms_per_object - b0_p95
    runtime["estimated_median_frame_overhead_ms"] = runtime.median_overhead_ms_per_object_vs_B0 * objects_per_frame
    runtime["estimated_p95_frame_overhead_ms"] = runtime.p95_overhead_ms_per_object_vs_B0 * objects_per_frame
    runtime["hardware"] = platform.platform() + "; processor=" + platform.processor()
    runtime["scope"] = "propagation_math_with_precomputed_detector_history_depth_and_poses_detector_inference_excluded"
    runtime.to_csv(OUT / "runtime_recovery.csv", index=False, float_format="%.12g")
    method = f"""# AV2 propagation runtime recovery method

No valid per-model propagation timing was present in the original logs. The
invalid generator logged detector inference duration and propagation row count,
but not B0–B5 runtime.

A warmed microbenchmark was therefore run on deterministic inputs from the
1,000-comparison recalculation sample. Each B0–B5 branch ran {REPETITIONS}
timed repetitions after {WARMUP} warmups. Detector inference, disk I/O, lidar
loading, and pose/depth acquisition were excluded; detector history, depth and
poses were precomputed. The representative subset contained
{objects_per_frame:.6f} detector objects per frame. Per-frame overhead is the
measured per-object overhead multiplied by this observed object count.

Hardware: {platform.platform()}; processor={platform.processor()}.

These timings characterize the corrected propagation mathematics only. They
are audit measurements, not confirmatory runtime evidence for the invalid
original run.
"""
    (OUT / "runtime_method.md").write_text(method, encoding="utf-8")
    print(runtime[["model","median_ms_per_object","p95_ms_per_object","estimated_median_frame_overhead_ms","estimated_p95_frame_overhead_ms"]].to_string(index=False))


if __name__ == "__main__":
    main()
