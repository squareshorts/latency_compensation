import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import wilcoxon

ROOT = Path(r"C:\work\auto")
TRACKERS = ROOT / "results" / "sivp_strengthening" / "tracker_low_level"
RESULTS = ROOT / "results" / "analysis_closure"

YAW_THRESHOLD = 0.0169726458
SEED = 20260719
REPLICATES = 10_000

def bootstrap(values: np.ndarray, statistic=np.median) -> tuple[float, float]:
    rng = np.random.default_rng(SEED)
    draws = np.empty(REPLICATES)
    for index in range(REPLICATES):
        draws[index] = statistic(rng.choice(values, len(values), replace=True))
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))

def main():
    paths = sorted(TRACKERS.glob("*/*.parquet"))

    # We only care about T5 and T6 at 300-ms high-yaw (or we can just calculate them)
    # The prompt says: "Report them for: YOLO11n, YOLO11s, RT-DETR-L, B3, B5, 300-ms frozen high-yaw endpoint, all-yaw sensitivity where already available."

    # We will load them all
    frames = [pd.read_parquet(p, columns=["detector", "log_id", "delta_ms", "model", "yaw_rate_rad_s", "normalized_center_error", "iou"]) for p in paths]
    data = pd.concat(frames, ignore_index=True)

    data = data[data.model.isin(["T5", "T6"])]
    data["yaw_stratum"] = np.where(data.yaw_rate_rad_s.abs() > YAW_THRESHOLD, "high_yaw", "all_yaw")
    all_yaw = data.copy()
    all_yaw["yaw_stratum"] = "all_yaw"
    high_yaw = data[data.yaw_rate_rad_s.abs() > YAW_THRESHOLD].copy()
    high_yaw["yaw_stratum"] = "high_yaw"

    expanded = pd.concat([all_yaw, high_yaw], ignore_index=True)

    per_log = expanded.groupby(["detector", "delta_ms", "yaw_stratum", "model", "log_id"], as_index=False).agg(
        median_normalized_center_error=("normalized_center_error", "median"),
        median_iou=("iou", "median")
    )

    summary = per_log.groupby(["detector", "delta_ms", "yaw_stratum", "model"], as_index=False).agg(
        median_normalized_center_error=("median_normalized_center_error", "median"),
        median_iou=("median_iou", "median")
    )

    summary.to_csv(RESULTS / "full_frame_localization_metrics.csv", index=False)

    contrasts = []
    for keys, frame in per_log.groupby(["detector", "delta_ms", "yaw_stratum"]):
        detector, delta_ms, yaw_stratum = keys

        # error
        pivot_error = frame.pivot(index="log_id", columns="model", values="median_normalized_center_error")
        if "T5" in pivot_error and "T6" in pivot_error:
            paired = pivot_error[["T5", "T6"]].dropna()
            if not paired.empty:
                diff = paired.T6 - paired.T5
                low, high = bootstrap(diff.to_numpy())
                contrasts.append({
                    "detector": detector, "delta_ms": delta_ms, "yaw_stratum": yaw_stratum,
                    "metric": "median_normalized_center_error", "B3": paired.T5.median(), "B5": paired.T6.median(),
                    "absolute_difference": diff.median(), "bootstrap_ci_low": low, "bootstrap_ci_high": high
                })

        # iou
        pivot_iou = frame.pivot(index="log_id", columns="model", values="median_iou")
        if "T5" in pivot_iou and "T6" in pivot_iou:
            paired = pivot_iou[["T5", "T6"]].dropna()
            if not paired.empty:
                diff = paired.T6 - paired.T5
                low, high = bootstrap(diff.to_numpy())
                contrasts.append({
                    "detector": detector, "delta_ms": delta_ms, "yaw_stratum": yaw_stratum,
                    "metric": "median_iou", "B3": paired.T5.median(), "B5": paired.T6.median(),
                    "absolute_difference": diff.median(), "bootstrap_ci_low": low, "bootstrap_ci_high": high
                })

    pd.DataFrame(contrasts).to_csv(RESULTS / "full_frame_localization_bootstrap.csv", index=False)

if __name__ == "__main__":
    main()
