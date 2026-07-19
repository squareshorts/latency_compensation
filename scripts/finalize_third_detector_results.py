"""Assemble required RT-DETR-L robustness result tables."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(r"C:\work\auto")
OUTPUT = ROOT / "results" / "sivp_strengthening"


def main() -> None:
    tracker_summary = pd.read_csv(OUTPUT / "tracker_baselines.csv")
    tracker_summary = tracker_summary[tracker_summary.detector == "rtdetr_l"].copy()
    tracker_long = tracker_summary.melt(
        id_vars=["detector", "delta_ms", "yaw_stratum", "model", "tracker_name", "eligible_logs", "comparisons"],
        value_vars=["median_of_log_center_error", "median_of_log_iou", "median_of_log_recall_0_5"],
        var_name="metric", value_name="value")
    tracker_long["scope"] = "matched_object"
    end_frames = []
    for scope, filename in (("persistent_object", "third_detector_persistent_object_detection_metrics.csv"),
                            ("full_frame", "third_detector_full_frame_detection_metrics.csv")):
        frame = pd.read_csv(OUTPUT / filename)
        frame = frame[(frame.stratum_type == "overall") & (frame.stratum == "all")]
        long = frame.melt(id_vars=["detector", "model", "delta_ms", "yaw_stratum", "eligible_logs", "target_objects", "predictions"],
                          value_vars=["ap_50_95", "ap50", "ap75", "precision", "recall", "false_positives_per_frame"],
                          var_name="metric", value_name="value")
        long["scope"] = scope
        end_frames.append(long)
    metrics = pd.concat([tracker_long, *end_frames], ignore_index=True, sort=False)
    metrics.to_csv(OUTPUT / "third_detector_metrics.csv", index=False, float_format="%.12g")

    tracker_per_log = pd.read_csv(OUTPUT / "tracker_per_log_metrics.csv")
    tracker_per_log = tracker_per_log[tracker_per_log.detector == "rtdetr_l"].copy()
    tracker_per_log["scope"] = "matched_object"
    end_per_log = pd.read_csv(OUTPUT / "third_detector_end_to_end_per_log_metrics.csv")
    pd.concat([tracker_per_log, end_per_log], ignore_index=True, sort=False).to_csv(
        OUTPUT / "third_detector_per_log_metrics.csv", index=False, float_format="%.12g")

    tracker_bootstrap = pd.read_csv(OUTPUT / "tracker_bootstrap_intervals.csv")
    tracker_bootstrap = tracker_bootstrap[tracker_bootstrap.detector == "rtdetr_l"].copy()
    tracker_bootstrap["scope"] = "matched_object"
    end_bootstrap = pd.read_csv(OUTPUT / "third_detector_end_to_end_bootstrap.csv")
    pd.concat([tracker_bootstrap, end_bootstrap], ignore_index=True, sort=False).to_csv(
        OUTPUT / "third_detector_bootstrap.csv", index=False, float_format="%.12g")

    checkpoints = []
    for path in sorted((OUTPUT / "third_detector_checkpoints" / "heldout").glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        checkpoints.append({"log_id": payload["log_id"], "frames": payload["frames"],
                            "detections": len(payload["detections"]), "inference_seconds": payload["inference_seconds"],
                            "fps": payload["frames"] / payload["inference_seconds"]})
    timing = pd.DataFrame(checkpoints)
    manifest = json.loads((OUTPUT / "third_detector_manifest.json").read_text())
    manifest.update({"heldout_logs_completed": len(timing), "heldout_frames": int(timing.frames.sum()),
                     "median_log_inference_seconds": float(timing.inference_seconds.median()),
                     "median_log_fps": float(timing.fps.median())})
    (OUTPUT / "third_detector_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print({"metric_rows": len(metrics), "heldout_logs": len(timing)})


if __name__ == "__main__":
    main()
