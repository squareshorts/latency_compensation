"""Consolidate sharded end-to-end metrics and run log-cluster contrasts."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(r"C:\work\auto")
OUTPUT = ROOT / "results" / "sivp_strengthening"
SEED = 20260719
REPLICATES = 10_000


def bootstrap(values: np.ndarray):
    rng = np.random.default_rng(SEED)
    draws = np.empty(REPLICATES)
    for index in range(REPLICATES):
        draws[index] = np.median(rng.choice(values, len(values), replace=True))
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--detector", choices=["rtdetr_l"])
    args = parser.parse_args()
    pattern = "shard_*_of_4_rtdetr_l.csv" if args.detector else "shard_*_of_4.csv"
    paths = sorted((OUTPUT / "end_to_end_metric_shards").glob(pattern))
    if len(paths) != 4:
        raise RuntimeError(f"Expected four metric shards, found {len(paths)}")
    per_log = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    per_log_name = "third_detector_end_to_end_per_log_metrics.csv" if args.detector else "end_to_end_per_log_metrics.csv"
    per_log.to_csv(OUTPUT / per_log_name, index=False, float_format="%.12g")
    group = ["detector", "model", "delta_ms", "yaw_stratum", "stratum_type", "stratum", "scope"]
    metrics = ["ap_50_95", "ap50", "ap75", "precision", "recall", "false_positives_per_frame",
               "persistent_object_recall", "new_object_miss_fraction", "disappeared_object_false_positive_fraction"]
    aggregation = {metric: (metric, "median") for metric in metrics}
    aggregation.update({"eligible_logs": ("log_id", "nunique"), "target_objects": ("target_objects", "sum"),
                        "predictions": ("predictions", "sum"), "frames": ("frames", "sum")})
    summary = per_log.groupby(group, as_index=False).agg(**aggregation)
    persistent_name = "third_detector_persistent_object_detection_metrics.csv" if args.detector else "persistent_object_detection_metrics.csv"
    full_name = "third_detector_full_frame_detection_metrics.csv" if args.detector else "full_frame_detection_metrics.csv"
    summary[summary.scope == "persistent_object"].to_csv(OUTPUT / persistent_name, index=False, float_format="%.12g")
    summary[summary.scope == "full_frame"].to_csv(OUTPUT / full_name, index=False, float_format="%.12g")
    contrasts = []
    principal = per_log[(per_log.stratum_type == "overall") & per_log.model.isin(["T4", "T5", "T6"])]
    for keys, frame in principal.groupby(["detector", "delta_ms", "yaw_stratum", "scope"]):
        detector, delta_ms, yaw_stratum, scope = keys
        for metric in ("ap_50_95", "ap50", "recall"):
            pivot = frame.pivot(index="log_id", columns="model", values=metric)
            for candidate in ("T4", "T6"):
                paired = pivot[["T5", candidate]].dropna()
                if paired.empty:
                    continue
                difference = paired[candidate] - paired.T5
                relative = difference / paired.T5.replace(0, np.nan)
                low, high = bootstrap(difference.to_numpy())
                try:
                    p_value = float(wilcoxon(difference).pvalue)
                except ValueError:
                    p_value = 1.0
                contrasts.append({"detector": detector, "delta_ms": delta_ms, "yaw_stratum": yaw_stratum,
                                  "scope": scope, "metric": metric, "candidate": candidate, "reference": "T5",
                                  "eligible_logs": len(paired), "paired_absolute_difference": float(np.median(difference)),
                                  "paired_relative_difference": float(np.nanmedian(relative)),
                                  "percent_logs_candidate_improved": float((difference > 0).mean() * 100),
                                  "bootstrap_ci_low": low, "bootstrap_ci_high": high, "wilcoxon_p": p_value})
    interval_name = "third_detector_end_to_end_bootstrap.csv" if args.detector else "end_to_end_bootstrap_intervals.csv"
    pd.DataFrame(contrasts).to_csv(OUTPUT / interval_name, index=False, float_format="%.12g")
    print({"per_log_rows": len(per_log), "summary_rows": len(summary)})


if __name__ == "__main__":
    main()
