"""Quantify coordinate duplication in the original AV2 propagation run.

The original Parquets did not store coordinates or comparison IDs.  This audit
reads their exact row order and deterministically reconstructs the coordinates
written by ``run_confirmation.propagate_log`` from the preserved detector JSON.
It does not run detector inference or the propagation wrapper.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"C:\work\auto")
OUT = ROOT / "results" / "av2_confirmation"
MODELS = ["stale", "constant_velocity", "kalman", "ego_motion_only", "B4", "B5"]
LABEL = {"stale": "B0", "constant_velocity": "B1", "kalman": "B2", "ego_motion_only": "B3", "B4": "B4", "B5": "B5"}
REFS = ["stale", "constant_velocity", "kalman", "ego_motion_only"]


def reconstructed_predictions(detections: list[dict]) -> tuple[dict[int, list[int]], list[dict[int, dict[str, np.ndarray]]]]:
    indices_by_timestamp: dict[int, list[int]] = defaultdict(list)
    predictions: list[dict[int, dict[str, np.ndarray]]] = []
    history: dict[int, list[np.ndarray]] = {}
    for index, detection in enumerate(detections):
        timestamp = int(detection["timestamp_ns"])
        indices_by_timestamp[timestamp].append(index)
        box = np.asarray([detection["x1"], detection["y1"], detection["x2"], detection["y2"]], dtype="<f8")
        past = history.setdefault(int(detection["class_id"]), [])
        by_delta = {}
        for delta in (100, 200, 300, 400, 500):
            velocity = box.copy()
            if past:
                velocity = box + (box - past[-1]) * delta / 100.0
            by_delta[delta] = {
                "stale": box.copy(),
                "constant_velocity": velocity.copy(),
                "kalman": velocity.copy(),
                "ego_motion_only": box.copy(),
                "B4": velocity.copy(),
                "B5": box + (velocity - box) * 0.65,
            }
        predictions.append(by_delta)
        past.append(box)
        history[int(detection["class_id"])] = past[-4:]
    return indices_by_timestamp, predictions


def main() -> None:
    files = sorted((OUT / "checkpoints").glob("*/*/*.propagation.parquet"))
    arrays: dict[tuple[str, str, int, str], list[np.ndarray]] = defaultdict(list)
    comparison_ids: dict[tuple[str, str, int], set[str]] = defaultdict(set)
    hashes = defaultdict(hashlib.sha256)
    missing_coordinate_columns = Counter()

    for parquet_path in files:
        detector, role = parquet_path.parts[-3], parquet_path.parts[-2]
        detector_path = parquet_path.with_suffix("").with_suffix(".json")
        payload = json.loads(detector_path.read_text(encoding="utf-8"))
        indices_by_ts, predictions = reconstructed_predictions(payload["detections"])
        frame = pd.read_parquet(parquet_path)
        missing_coordinate_columns[(detector, role)] += int(not {"pred_x1", "pred_y1", "pred_x2", "pred_y2", "comparison_id"}.issubset(frame.columns))
        if len(frame) % 6:
            raise RuntimeError(f"Model rows not divisible by six: {parquet_path}")
        first = frame.iloc[::6].reset_index(drop=True)
        model_matrix = frame["model"].to_numpy().reshape(-1, 6)
        if not np.all(model_matrix == np.asarray(MODELS)):
            raise RuntimeError(f"Unexpected model ordering: {parquet_path}")
        occurrence = Counter()
        per_file: dict[tuple[int, str], list[np.ndarray]] = defaultdict(list)
        for row in first.itertuples(index=False):
            timestamp, delta = int(row.source_timestamp_ns), int(row.delta_ms)
            ordinal = occurrence[(timestamp, delta)]
            occurrence[(timestamp, delta)] += 1
            detection_indices = indices_by_ts[timestamp]
            if ordinal >= len(detection_indices):
                raise RuntimeError(f"Cannot join Parquet row to detector: {parquet_path}, {timestamp}, {delta}, {ordinal}")
            detector_index = detection_indices[ordinal]
            comparison_id = f"{payload['log_id']}:{timestamp}:{detector_index}:{delta}"
            comparison_ids[(detector, role, delta)].add(comparison_id)
            for model in MODELS:
                per_file[(delta, model)].append(predictions[detector_index][delta][model])
        for (delta, model), values in per_file.items():
            block = np.vstack(values).astype("<f8", copy=False)
            key = (detector, role, delta, model)
            arrays[key].append(block)
            hashes[key].update(block.tobytes(order="C"))

    audit_rows, hash_rows = [], []
    for detector in ("yolo11n", "yolo11s"):
        for role in ("development", "model_selection", "heldout"):
            for delta in (100, 200, 300, 400, 500):
                stacked = {model: np.vstack(arrays[(detector, role, delta, model)]) for model in MODELS}
                for model in MODELS:
                    values = stacked[model]
                    row = {
                        "detector": detector, "split": role, "delta_ms": delta,
                        "model": model, "model_label": LABEL[model],
                        "row_count": len(values),
                        "unique_object_comparison_ids": len(comparison_ids[(detector, role, delta)]),
                        "unique_predicted_box_coordinates": len(np.unique(values, axis=0)),
                        "coordinate_source": "reconstructed_from_preserved_detector_json_and_original_generator_joined_to_parquet_order",
                        "coordinates_stored_in_original_parquet": False,
                    }
                    for ref in REFS:
                        difference = np.abs(values - stacked[ref])
                        exactly_equal = np.all(difference == 0.0, axis=1)
                        row[f"exactly_equal_to_{LABEL[ref]}_count"] = int(exactly_equal.sum())
                        row[f"exactly_equal_to_{LABEL[ref]}_percent"] = float(exactly_equal.mean() * 100.0)
                        row[f"mean_coordinate_abs_difference_vs_{LABEL[ref]}"] = float(difference.mean())
                        row[f"max_coordinate_abs_difference_vs_{LABEL[ref]}"] = float(difference.max())
                    audit_rows.append(row)
                    hash_rows.append({
                        "detector": detector, "split": role, "delta_ms": delta,
                        "model": model, "model_label": LABEL[model], "row_count": len(values),
                        "sha256_ordered_float64_coordinate_array": hashes[(detector, role, delta, model)].hexdigest(),
                        "coordinate_source": row["coordinate_source"],
                    })

    audit = pd.DataFrame(audit_rows)
    hashes_frame = pd.DataFrame(hash_rows)
    audit.to_csv(OUT / "propagation_distinctness_audit.csv", index=False, float_format="%.12g")
    hashes_frame.to_csv(OUT / "model_coordinate_hashes.csv", index=False)
    explicit_pairs = [("B0", "B1"), ("B0", "B3"), ("B1", "B4"), ("B3", "B4"), ("B4", "B5")]
    pair_lines = []
    for left, right in explicit_pairs:
        left_hash = hashes_frame[hashes_frame.model_label == left].sort_values(["detector", "split", "delta_ms"])["sha256_ordered_float64_coordinate_array"].to_numpy()
        right_hash = hashes_frame[hashes_frame.model_label == right].sort_values(["detector", "split", "delta_ms"])["sha256_ordered_float64_coordinate_array"].to_numpy()
        identical_groups = int(np.sum(left_hash == right_hash))
        pair_lines.append(f"- {left} versus {right}: identical ordered-coordinate hashes in {identical_groups}/30 detector/split/latency groups.")
    report = """# Duplicate prediction audit

## Classification

**INVALID RUN — the propagation outputs are degenerate.**

The original metric Parquets were inspected at row level. They do not contain
predicted coordinates or object-comparison IDs. To quantify the defect, this
audit reconstructed the exact coordinate arrays produced by the generating
branch from the preserved detector JSON and joined them to original Parquet row
order. This is possible because each comparison has the fixed six-model order.

## Explicit model-pair tests

""" + "\n".join(pair_lines) + """

## Cause

`run_confirmation.py` assigns B2 and B4 directly from the B1 `velocity` array,
and assigns B3 from the B0 `stale` array. It never calls the repository's
Kalman filter, ego-motion transform, lidar-depth path, historical 3D
object-motion path, or uncertainty-damping function. B5 is a fixed 0.65 blend
rather than the frozen uncertainty-aware branch. The exact ties are duplicated
model outputs, not a scientific result.

## Schema defect

All 300 original Parquets omit `comparison_id`, target/source box coordinates,
track identity, pose/depth provenance, and motion inputs. Consequently the
stored metric values cannot independently prove which coordinates generated
them; the generator reconstruction and source trace establish the defect.
"""
    (OUT / "duplicate_prediction_audit.md").write_text(report, encoding="utf-8")
    print(json.dumps({"files": len(files), "audit_rows": len(audit), "hash_rows": len(hashes_frame),
                      "B1_equals_B4_all_groups": bool((hashes_frame[hashes_frame.model_label == "B1"].sort_values(["detector", "split", "delta_ms"])["sha256_ordered_float64_coordinate_array"].to_numpy() == hashes_frame[hashes_frame.model_label == "B4"].sort_values(["detector", "split", "delta_ms"])["sha256_ordered_float64_coordinate_array"].to_numpy()).all())}))


if __name__ == "__main__":
    main()
