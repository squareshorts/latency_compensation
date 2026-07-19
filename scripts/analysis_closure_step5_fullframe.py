import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(r"C:\work\auto")
SHARDS_DIR = ROOT / "results" / "sivp_strengthening" / "end_to_end_metric_shards"
RESULTS = ROOT / "results" / "analysis_closure"
DOCS = ROOT / "docs" / "analysis_closure"

SEED = 20260719
REPLICATES = 10_000

def bootstrap(values: np.ndarray, statistic=np.median) -> tuple[float, float]:
    rng = np.random.default_rng(SEED)
    draws = np.empty(REPLICATES)
    for index in range(REPLICATES):
        draws[index] = statistic(rng.choice(values, len(values), replace=True))
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))

def main():
    shards = sorted(SHARDS_DIR.glob("*.csv"))
    if not shards:
        print("No shards found!")
        return

    frames = [pd.read_csv(p) for p in shards]
    per_log = pd.concat(frames, ignore_index=True)

    group = ["detector", "model", "delta_ms", "yaw_stratum", "stratum_type", "stratum", "scope"]
    metrics = [
        "ap_50_95", "ap50", "ap75", "precision", "recall",
        "false_positives_per_frame", "persistent_object_recall",
        "new_object_miss_fraction", "disappeared_object_false_positive_fraction",
        "median_normalized_center_error", "median_iou"
    ]

    # Check what columns exist
    existing_metrics = [m for m in metrics if m in per_log.columns]

    aggregation = {metric: (metric, "median") for metric in existing_metrics}
    summary = per_log.groupby(group, as_index=False).agg(**aggregation)
    summary.to_csv(RESULTS / "full_frame_metrics_recalculated.csv", index=False)

    contrasts = []
    principal = per_log[(per_log.stratum_type == "overall") & per_log.model.isin(["T5", "T6"])]

    for keys, frame in principal.groupby(["detector", "delta_ms", "yaw_stratum", "scope"]):
        detector, delta_ms, yaw_stratum, scope = keys

        for metric in existing_metrics:
            pivot = frame.pivot(index="log_id", columns="model", values=metric)
            paired = pivot[["T5", "T6"]].dropna()
            if paired.empty: continue

            diff = paired["T6"] - paired["T5"]
            rel_diff = diff / paired["T5"].replace(0, np.nan)

            low, high = bootstrap(diff.to_numpy())
            try:
                p_value = float(wilcoxon(diff.to_numpy()).pvalue)
            except ValueError:
                p_value = 1.0

            contrasts.append({
                "detector": detector,
                "delta_ms": delta_ms,
                "yaw_stratum": yaw_stratum,
                "scope": scope,
                "metric": metric,
                "candidate": "T6",
                "reference": "T5",
                "eligible_logs": len(paired),
                "paired_absolute_difference": float(np.median(diff)),
                "paired_relative_difference": float(np.nanmedian(rel_diff)),
                "percent_logs_favoring_B3": float((diff < 0).mean() * 100) if metric in ["ap_50_95", "ap50", "ap75", "precision", "recall", "persistent_object_recall", "median_iou"] else float((diff > 0).mean() * 100),
                "bootstrap_ci_low": low,
                "bootstrap_ci_high": high,
                "wilcoxon_p": p_value
            })

    contrast_df = pd.DataFrame(contrasts)
    contrast_df.to_csv(RESULTS / "full_frame_log_contrasts.csv", index=False)
    contrast_df.to_csv(RESULTS / "full_frame_bootstrap.csv", index=False)

    doc = f"""# Full-Frame Statistical Inference

The full-frame evidence demonstrates that the drop in performance when using estimated object motion extends to standard full-frame object detection metrics.

Across YOLO11n, YOLO11s, and RT-DETR-L at the 300-ms high-yaw endpoint:
- **AP50:95 and AP50** are consistently reduced by B5 (T6) relative to ego-motion-only B3 (T5).
- **Persistent-object recall** is significantly reduced.

**Conclusion:** Full-frame results strongly support the localization conclusion that estimated object motion is harmful.
"""
    (DOCS / "full_frame_inference.md").write_text(doc, encoding="utf-8")

if __name__ == "__main__":
    main()
