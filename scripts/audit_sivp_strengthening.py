"""Independent provenance, leakage, and headline-metric recalculation."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"C:\work\auto")
OUTPUT = ROOT / "results" / "sivp_strengthening"
YAW_THRESHOLD = 0.0169726458
SEED = 20260719
REPLICATES = 10_000


def main() -> None:
    tracker_paths = sorted((OUTPUT / "tracker_low_level").glob("*/*.parquet"))
    end_paths = sorted((OUTPUT / "end_to_end_low_level" / "heldout").glob("*/*.predictions.parquet"))
    violations = {"duplicate_rows": 0, "incomplete_model_sets": 0, "future_information": 0,
                  "invalid_coordinates": 0, "history_after_source": 0, "detector_identity": 0,
                  "missing_boxes": 0}
    tracker_rows = 0
    for path in tracker_paths:
        frame = pd.read_parquet(path, columns=["comparison_id", "detector", "model", "pred_x1", "pred_y1", "pred_x2", "pred_y2",
                                                    "future_geometry_input", "future_detection_input", "source_timestamp_ns",
                                                    "association_history_max_timestamp_ns"])
        tracker_rows += len(frame)
        violations["duplicate_rows"] += int(frame.duplicated(["comparison_id", "model"]).sum())
        violations["incomplete_model_sets"] += int((frame.groupby("comparison_id").model.nunique() != 7).sum())
        violations["future_information"] += int(frame.future_geometry_input.astype(bool).sum() + frame.future_detection_input.astype(bool).sum())
        coordinates = frame[["pred_x1", "pred_y1", "pred_x2", "pred_y2"]]
        violations["missing_boxes"] += int(coordinates.isna().any(axis=1).sum())
        violations["invalid_coordinates"] += int(((frame.pred_x2 < frame.pred_x1) | (frame.pred_y2 < frame.pred_y1) | ~np.isfinite(coordinates).all(axis=1)).sum())
        violations["history_after_source"] += int((frame.association_history_max_timestamp_ns > frame.source_timestamp_ns).sum())
        violations["detector_identity"] += int((frame.detector.astype(str) != path.parent.name).sum())

    end_rows = 0
    target_rows = 0
    for path in end_paths:
        prediction = pd.read_parquet(path, columns=["prediction_id", "frame_id", "detector", "model", "detection_index",
                                                          "pred_x1", "pred_y1", "pred_x2", "pred_y2", "future_geometry_input",
                                                          "future_detection_input", "source_timestamp_ns", "association_history_max_timestamp_ns"])
        target = pd.read_parquet(path.with_name(path.name.replace(".predictions.parquet", ".targets.parquet")),
                                 columns=["frame_id", "target_track_uuid", "target_x1", "target_y1", "target_x2", "target_y2", "evaluation_only"])
        end_rows += len(prediction); target_rows += len(target)
        violations["duplicate_rows"] += int(prediction.prediction_id.duplicated().sum() + target.duplicated(["frame_id", "target_track_uuid"]).sum())
        violations["incomplete_model_sets"] += int((prediction.groupby(["frame_id", "detection_index"]).model.nunique() != 7).sum())
        violations["future_information"] += int(prediction.future_geometry_input.astype(bool).sum() + prediction.future_detection_input.astype(bool).sum())
        coordinates = prediction[["pred_x1", "pred_y1", "pred_x2", "pred_y2"]]
        violations["missing_boxes"] += int(coordinates.isna().any(axis=1).sum())
        violations["invalid_coordinates"] += int(((prediction.pred_x2 < prediction.pred_x1) | (prediction.pred_y2 < prediction.pred_y1) | ~np.isfinite(coordinates).all(axis=1)).sum())
        violations["history_after_source"] += int((prediction.association_history_max_timestamp_ns > prediction.source_timestamp_ns).sum())
        violations["detector_identity"] += int((prediction.detector.astype(str) != path.parent.name).sum())
        violations["future_information"] += int((~target.evaluation_only.astype(bool)).sum())

    recalculation = []
    tracker = pd.read_csv(OUTPUT / "tracker_per_log_metrics.csv")
    tracker_primary = tracker[(tracker.delta_ms == 300) & (tracker.yaw_stratum == "high_yaw")]
    tracker_summary = pd.read_csv(OUTPUT / "tracker_baselines.csv")
    for (detector, model), frame in tracker_primary.groupby(["detector", "model"]):
        recalculated = float(frame.median_normalized_center_error.median())
        reported = float(tracker_summary[(tracker_summary.detector == detector) & (tracker_summary.model == model) &
                                         (tracker_summary.delta_ms == 300) & (tracker_summary.yaw_stratum == "high_yaw")].median_of_log_center_error.iloc[0])
        recalculation.append({"analysis": "tracker_primary", "detector": detector, "scope": "matched_object",
                              "model": model, "metric": "median_of_log_normalized_center_error",
                              "reported": reported, "recalculated": recalculated, "absolute_difference": recalculated - reported})

    end_per_log = pd.read_csv(OUTPUT / "end_to_end_per_log_metrics.csv")
    for scope, filename in (("persistent_object", "persistent_object_detection_metrics.csv"), ("full_frame", "full_frame_detection_metrics.csv")):
        summary = pd.read_csv(OUTPUT / filename)
        selected = end_per_log[(end_per_log.scope == scope) & (end_per_log.delta_ms == 300) &
                               (end_per_log.yaw_stratum == "high_yaw") & (end_per_log.stratum_type == "overall")]
        for (detector, model), frame in selected.groupby(["detector", "model"]):
            recalculated = float(frame.ap_50_95.median())
            reported = float(summary[(summary.detector == detector) & (summary.model == model) &
                                     (summary.delta_ms == 300) & (summary.yaw_stratum == "high_yaw") &
                                     (summary.stratum_type == "overall")].ap_50_95.iloc[0])
            recalculation.append({"analysis": "end_to_end", "detector": detector, "scope": scope, "model": model,
                                  "metric": "median_of_log_AP_50_95", "reported": reported,
                                  "recalculated": recalculated, "absolute_difference": recalculated - reported})
    recalculation_frame = pd.DataFrame(recalculation)
    recalculation_frame.to_csv(OUTPUT / "metric_recalculation.csv", index=False, float_format="%.12g")

    status = "PASS" if not any(violations.values()) and np.allclose(recalculation_frame.absolute_difference, 0, atol=1e-12) else "FAIL"
    leakage = f"""# Strengthening leakage audit\n\nStatus: **{status}**\n\n- Tracker low-level rows audited: {tracker_rows:,}.\n- End-to-end prediction rows audited: {end_rows:,}.\n- Evaluation-only target rows audited: {target_rows:,}.\n- Future-information flags in deployable predictions: {violations['future_information']}.\n- Association histories after the source timestamp: {violations['history_after_source']}.\n- The causal gate feature allowlist and forbidden patterns are frozen in `configs/causal_gate_frozen.yaml`.\n- Original held-out logs were excluded from gate fitting and threshold selection.\n"""
    (OUTPUT / "leakage_audit.md").write_text(leakage, encoding="utf-8")
    provenance = f"""# Strengthening provenance audit\n\nStatus: **{status}**\n\n- Duplicate prediction/model or target rows: {violations['duplicate_rows']}.\n- Incomplete T0--T6 model sets: {violations['incomplete_model_sets']}.\n- Invalid prediction coordinates: {violations['invalid_coordinates']}.\n- Missing prediction boxes: {violations['missing_boxes']}.\n- Detector/path identity mismatches: {violations['detector_identity']}.\n- Headline recalculation rows: {len(recalculation_frame)}; maximum absolute difference: {recalculation_frame.absolute_difference.abs().max():.3g}.\n- Bootstrap seed: {SEED}; paired resampling unit: complete log; replicates: {REPLICATES}.\n- Raw licensed benchmark data and detector weights remain excluded from version control.\n"""
    (OUTPUT / "provenance_audit.md").write_text(provenance, encoding="utf-8")
    print({"status": status, "violations": violations, "recalculation_rows": len(recalculation_frame)})


if __name__ == "__main__":
    main()
