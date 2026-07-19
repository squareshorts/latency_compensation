"""Aggregate causal tracker baselines with log-cluster inference."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(r"C:\work\auto")
sys.path.insert(0, str(ROOT / "src"))
SOURCE = ROOT / "results" / "sivp_strengthening" / "tracker_low_level"
OUTPUT = ROOT / "results" / "sivp_strengthening"
YAW_THRESHOLD = 0.0169726458
SEED = 20260719
REPLICATES = 10_000
LABELS = {"T0": "stale", "T1": "constant_image_velocity", "T2": "SORT_Kalman",
          "T3": "ByteTrack_prediction", "T4": "OC_SORT_prediction", "T5": "B3_ego_only", "T6": "B5_ego_object"}


def bootstrap(values: np.ndarray, statistic=np.median) -> tuple[float, float]:
    rng = np.random.default_rng(SEED)
    draws = np.empty(REPLICATES)
    for index in range(REPLICATES):
        draws[index] = statistic(rng.choice(values, len(values), replace=True))
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def main() -> None:
    paths = sorted(SOURCE.glob("*/*.parquet"))
    if len(paths) not in {100, 150}:
        raise RuntimeError(f"Expected 100 or 150 tracker shards, found {len(paths)}")
    frames = [pd.read_parquet(path) for path in paths]
    data = pd.concat(frames, ignore_index=True)
    data["yaw_stratum"] = np.where(data.yaw_rate_rad_s.abs() > YAW_THRESHOLD, "high_yaw", "all_yaw")
    all_yaw = data.copy()
    all_yaw["yaw_stratum"] = "all_yaw"
    high_yaw = data[data.yaw_rate_rad_s.abs() > YAW_THRESHOLD].copy()
    high_yaw["yaw_stratum"] = "high_yaw"
    expanded = pd.concat([all_yaw, high_yaw], ignore_index=True)
    group = ["detector", "log_id", "delta_ms", "yaw_stratum", "model"]
    per_log = expanded.groupby(group, as_index=False).agg(
        comparisons=("comparison_id", "nunique"),
        median_normalized_center_error=("normalized_center_error", "median"),
        median_iou=("iou", "median"),
        recall_iou_0_5=("recall_iou_0_5", "mean"),
        median_runtime_ms=("runtime_ms", "median"),
    )
    per_log["tracker_name"] = per_log.model.map(LABELS)
    per_log.to_csv(OUTPUT / "tracker_per_log_metrics.csv", index=False, float_format="%.12g")

    summary = per_log.groupby(["detector", "delta_ms", "yaw_stratum", "model", "tracker_name"], as_index=False).agg(
        eligible_logs=("log_id", "nunique"), comparisons=("comparisons", "sum"),
        median_of_log_center_error=("median_normalized_center_error", "median"),
        median_of_log_iou=("median_iou", "median"), median_of_log_recall_0_5=("recall_iou_0_5", "median"),
    )
    summary.to_csv(OUTPUT / "tracker_baselines.csv", index=False, float_format="%.12g")

    contrasts = []
    for keys, frame in per_log.groupby(["detector", "delta_ms", "yaw_stratum"]):
        detector, delta_ms, yaw_stratum = keys
        pivot_error = frame.pivot(index="log_id", columns="model", values="median_normalized_center_error")
        pivot_iou = frame.pivot(index="log_id", columns="model", values="median_iou")
        for candidate in [f"T{x}" for x in range(7) if x != 5]:
            paired = pivot_error[["T5", candidate]].dropna()
            if paired.empty:
                continue
            absolute = paired[candidate] - paired.T5
            relative = absolute / paired.T5.replace(0, np.nan)
            iou_pair = pivot_iou[["T5", candidate]].dropna()
            iou_difference = iou_pair[candidate] - iou_pair.T5
            low, high = bootstrap(relative.dropna().to_numpy())
            try:
                wilcoxon_p = float(wilcoxon(absolute.to_numpy()).pvalue)
            except ValueError:
                wilcoxon_p = 1.0
            contrasts.append({
                "detector": detector, "delta_ms": delta_ms, "yaw_stratum": yaw_stratum,
                "candidate": candidate, "candidate_name": LABELS[candidate], "reference": "T5",
                "eligible_logs": len(paired), "paired_absolute_difference": float(np.median(absolute)),
                "paired_relative_difference": float(np.median(relative)),
                "median_within_log_candidate_error": float(np.median(paired[candidate])),
                "median_within_log_B3_error": float(np.median(paired.T5)),
                "percent_logs_candidate_improved": float((absolute < 0).mean() * 100),
                "bootstrap_ci_low": low, "bootstrap_ci_high": high,
                "wilcoxon_p": wilcoxon_p, "paired_iou_difference": float(np.median(iou_difference)),
            })
    interval = pd.DataFrame(contrasts)
    interval.to_csv(OUTPUT / "tracker_bootstrap_intervals.csv", index=False, float_format="%.12g")

    runtime = data.groupby(["detector", "model"], as_index=False).agg(
        observations=("runtime_ms", "size"), median_runtime_ms=("runtime_ms", "median"),
        p95_runtime_ms=("runtime_ms", lambda values: values.quantile(0.95)),
    )
    runtime["tracker_name"] = runtime.model.map(LABELS)
    runtime.to_csv(OUTPUT / "tracker_runtime.csv", index=False, float_format="%.12g")

    primary = summary[(summary.delta_ms == 300) & (summary.yaw_stratum == "high_yaw") & summary.model.isin(["T0", "T1", "T2", "T3", "T4"])]
    best = primary.sort_values(["detector", "median_of_log_center_error"]).groupby("detector").first().reset_index()
    print(json.dumps({"rows": len(data), "comparisons": int(data.comparison_id.nunique()),
                      "best_tracker_by_detector": dict(zip(best.detector, best.model))}))


if __name__ == "__main__":
    main()
