"""Reviewer-facing failure sensitivity and tracker mechanism analyses."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"C:\work\auto")
DIAGNOSTICS = ROOT / "results" / "object_motion_harm" / "object_level_diagnostics.parquet"
TRACKERS = ROOT / "results" / "sivp_strengthening" / "tracker_low_level"
OUTPUT = ROOT / "results" / "sivp_strengthening"


def summarize(frame: pd.DataFrame, dimension: str, value: str) -> dict:
    per_log = frame.groupby(["detector", "log_id"], as_index=False).agg(
        b3=("B3_error", "median"), b5=("B5_error", "median"), comparisons=("comparison_id", "nunique"))
    per_log["difference"] = per_log.b5 - per_log.b3
    per_log["relative"] = per_log.difference / per_log.b3.replace(0, np.nan)
    return {"dimension": dimension, "stratum": value, "eligible_logs": per_log.log_id.nunique(),
            "comparisons": int(per_log.comparisons.sum()), "median_B3_error": float(per_log.b3.median()),
            "median_B5_error": float(per_log.b5.median()), "median_absolute_difference": float(per_log.difference.median()),
            "median_relative_difference": float(per_log.relative.median()),
            "percent_logs_B5_improved": float((per_log.difference < 0).mean() * 100)}


def main() -> None:
    columns = ["comparison_id", "log_id", "role", "detector", "delta_ms", "high_yaw", "track_age",
               "detector_confidence", "association_confidence", "velocity_sign_error", "source_depth_m",
               "object_class", "primary_failure_mechanism", "B3_error", "B5_error"]
    data = pd.read_parquet(DIAGNOSTICS, columns=columns)
    heldout = data[data.role == "heldout"].copy()
    heldout["distance_bin"] = pd.cut(heldout.source_depth_m, [0, 20, 40, np.inf], labels=["0_20m", "20_40m", "40m_plus"])
    heldout["confidence_bin"] = pd.cut(heldout.detector_confidence, [0, .5, .75, 1.01], labels=["low", "medium", "high"], include_lowest=True)
    heldout["track_age_bin"] = np.where(heldout.track_age >= 3, "3_plus", heldout.track_age.astype(str))
    heldout["association_bin"] = pd.cut(heldout.association_confidence, [-np.inf, .25, .5, np.inf], labels=["low", "medium", "high"])
    heldout["velocity_sign_consistency"] = np.where(heldout.velocity_sign_error, "sign_error", "sign_consistent")
    heldout["yaw_bin"] = np.where(heldout.high_yaw, "high_yaw", "lower_yaw")
    definitions = [("latency_ms", "delta_ms"), ("yaw", "yaw_bin"), ("distance", "distance_bin"),
                   ("detector_confidence", "confidence_bin"), ("track_age", "track_age_bin"),
                   ("class", "object_class"), ("association_confidence", "association_bin"),
                   ("velocity_sign_consistency", "velocity_sign_consistency"), ("detector", "detector")]
    rows = []
    for dimension, column in definitions:
        for value, frame in heldout.groupby(column, observed=True):
            if len(frame):
                rows.append(summarize(frame, dimension, str(value)))
    pd.DataFrame(rows).to_csv(OUTPUT / "failure_sensitivity.csv", index=False, float_format="%.12g")

    primary = heldout[(heldout.delta_ms == 300) & heldout.high_yaw].copy()
    tracker_frames = []
    for path in sorted(TRACKERS.glob("*/*.parquet")):
        if path.parent.name not in {"yolo11n", "yolo11s"}:
            continue
        frame = pd.read_parquet(path, columns=["comparison_id", "model", "normalized_center_error"])
        tracker_frames.append(frame[frame.model.isin(["T3", "T4", "T5", "T6"])])
    trackers = pd.concat(tracker_frames, ignore_index=True)
    merged = trackers.merge(primary[["comparison_id", "log_id", "detector", "primary_failure_mechanism"]],
                            on="comparison_id", how="inner", validate="many_to_one")
    mechanism_per_log = merged.groupby(["detector", "primary_failure_mechanism", "model", "log_id"], as_index=False).agg(
        median_normalized_center_error=("normalized_center_error", "median"), comparisons=("comparison_id", "nunique"))
    mechanism = mechanism_per_log.groupby(["detector", "primary_failure_mechanism", "model"], as_index=False).agg(
        eligible_logs=("log_id", "nunique"), comparisons=("comparisons", "sum"),
        median_of_log_center_error=("median_normalized_center_error", "median"))
    pivot = mechanism.pivot_table(index=["detector", "primary_failure_mechanism"], columns="model",
                                  values="median_of_log_center_error").reset_index()
    for model in ("T3", "T4", "T6"):
        pivot[f"{model}_relative_to_B3"] = pivot[model] / pivot.T5 - 1
    pivot.to_csv(OUTPUT / "tracker_failure_mechanisms.csv", index=False, float_format="%.12g")

    architecture = pd.read_csv(OUTPUT / "tracker_bootstrap_intervals.csv")
    architecture = architecture[(architecture.delta_ms == 300) & (architecture.yaw_stratum == "high_yaw") &
                                architecture.candidate.isin(["T4", "T6"])]
    architecture.to_csv(OUTPUT / "architecture_interaction.csv", index=False, float_format="%.12g")
    print({"sensitivity_rows": len(rows), "mechanism_rows": len(pivot), "merged_comparisons": merged.comparison_id.nunique()})


if __name__ == "__main__":
    main()
