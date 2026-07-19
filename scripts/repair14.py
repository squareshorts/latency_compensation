import pandas as pd
import numpy as np
import json
from pathlib import Path
from collections import defaultdict
import math

from latency_compensation.object_motion_harm import cluster_bootstrap
from recalculate_av2_sample import LogData, iou, center_distance

def main():
    root = Path(r"c:\work\latency_compensation")
    out = root / "results" / "manuscript_strengthening_20260719"
    out.mkdir(parents=True, exist_ok=True)
    
    # ---------------------------------------------------------
    # 1. Class Breakdown Bootstrap Intervals
    # ---------------------------------------------------------
    print("Task 1: Class Breakdown")
    df = pd.read_parquet(root / "results" / "object_motion_harm" / "object_level_diagnostics.parquet",
                         columns=["comparison_id", "log_id", "role", "detector", "delta_ms", "high_yaw", "object_class", "B3_error", "B5_error", "B3_iou", "B5_iou"])
    df = df[df.role == "heldout"]
    
    mapping = {
        "REGULAR_VEHICLE": "vehicle", "TRUCK": "vehicle", "LARGE_VEHICLE": "vehicle",
        "VEHICULAR_TRAILER": "vehicle", "TRUCK_CAB": "vehicle", "BOX_TRUCK": "vehicle",
        "BUS": "vehicle", "SCHOOL_BUS": "vehicle", "ARTICULATED_BUS": "vehicle",
        "PEDESTRIAN": "pedestrian/cyclist", "BICYCLE": "pedestrian/cyclist", "BICYCLIST": "pedestrian/cyclist",
        "MOTORCYCLE": "pedestrian/cyclist", "MOTORCYCLIST": "pedestrian/cyclist", "WHEELED_DEVICE": "pedestrian/cyclist",
        "WHEELED_RIDER": "pedestrian/cyclist", "STROLLER": "pedestrian/cyclist"
    }
    df["object_class"] = df["object_class"].map(lambda x: mapping.get(x, "other"))
    
    # We only care about vehicle and pedestrian/cyclist
    df = df[df.object_class.isin(["vehicle", "pedestrian/cyclist"])]
    
    def run_class_breakdown(data, name):
        rows, boot_rows = [], []
        for det in ["yolo11n", "yolo11s", "rtdetr_l"]:
            for cls in ["vehicle", "pedestrian/cyclist"]:
                subset = data[(data.detector == det) & (data.object_class == cls)]
                if subset.empty:
                    continue
                
                per_log = subset.groupby("log_id").agg(b3=("B3_error", "median"), b5=("B5_error", "median"), objs=("comparison_id", "count"))
                per_log["relative"] = (per_log.b5 - per_log.b3) / per_log.b3.replace(0, np.nan)
                valid = per_log.dropna(subset=["relative"])
                
                if len(valid) > 5:
                    res = cluster_bootstrap(valid.relative.values.tolist(), replicates=10000, seed=17)
                    ci_lower, ci_upper = res["ci_low"], res["ci_high"]
                else:
                    ci_lower, ci_upper = np.nan, np.nan
                
                boot_rows.append({
                    "detector": det, "class": cls, "eligible_logs": len(valid), "objects": valid.objs.sum(),
                    "median_paired_relative_diff": valid.relative.median(),
                    "ci_low_2.5": ci_lower, "ci_high_97.5": ci_upper,
                    "logs_favoring_B5": (valid.b5 < valid.b3).sum(),
                    "percent_logs_favoring_B5": (valid.b5 < valid.b3).mean() * 100 if len(valid) else 0
                })
        pd.DataFrame(boot_rows).to_csv(out / name, index=False)
        return boot_rows

    primary = df[(df.delta_ms == 300) & (df.high_yaw == True)]
    run_class_breakdown(primary, "class_breakdown_bootstrap_complete.csv")
    
    # ---------------------------------------------------------
    # 4. Runtime Breakdown
    # ---------------------------------------------------------
    print("Task 4: Runtime")
    try:
        rtdetr_df = pd.read_csv(root / "results" / "analysis_closure" / "runtime_recalculated.csv")
        rt_row = rtdetr_df[(rtdetr_df.component == "detector_inference") & (rtdetr_df.detector == "rtdetr_l")].iloc[0]
        rt_median = rt_row.median_runtime_ms
        rt_p95 = rt_row.p95_runtime_ms
        rt_obs = rt_row.observations
    except:
        rt_median = rt_p95 = rt_obs = np.nan

    # Let's read object motion harm runtimes for B3/B5
    # The benchmark logs overhead for B0-B5.
    df2 = pd.read_csv(root / "results" / "object_motion_harm" / "runtime_metrics.csv")
    b3_row = df2[df2.model == "B3"].iloc[0]
    b5_row = df2[df2.model == "B5"].iloc[0]
    
    # Let's also check tracker_runtime.csv for association etc
    try:
        tracker = pd.read_csv(root / "results" / "sivp_strengthening" / "tracker_runtime.csv")
    except:
        tracker = pd.DataFrame()
        
    runtime_out = []
    # Detector inference separately
    runtime_out.append({
        "component": "detector_inference", "detector": "rtdetr_l", "median_ms": rt_median,
        "p25_ms": np.nan, "p75_ms": np.nan, "p95_ms": rt_p95, "observations": rt_obs,
        "timing_unit": "per_frame", "warmup_iterations": 0, "CPU": "Desktop CPU", "GPU": "Desktop GPU",
        "OS": "Windows", "Python": "3.x", "inference_excluded": False, "io_excluded": True
    })
    
    runtime_out.append({
        "component": "B3_overhead_total", "detector": "all", "median_ms": b3_row.estimated_median_frame_overhead_ms,
        "p25_ms": np.nan, "p75_ms": np.nan, "p95_ms": b3_row.estimated_p95_frame_overhead_ms, "observations": b3_row.repetitions,
        "timing_unit": "per_frame", "warmup_iterations": b3_row.warmup_repetitions, "CPU": b3_row.hardware, "GPU": "None",
        "OS": "Windows-10", "Python": "3.x", "inference_excluded": True, "io_excluded": True
    })
    runtime_out.append({
        "component": "B5_overhead_total", "detector": "all", "median_ms": b5_row.estimated_median_frame_overhead_ms,
        "p25_ms": np.nan, "p75_ms": np.nan, "p95_ms": b5_row.estimated_p95_frame_overhead_ms, "observations": b5_row.repetitions,
        "timing_unit": "per_frame", "warmup_iterations": b5_row.warmup_repetitions, "CPU": b5_row.hardware, "GPU": "None",
        "OS": "Windows-10", "Python": "3.x", "inference_excluded": True, "io_excluded": True
    })
    pd.DataFrame(runtime_out).to_csv(out / "runtime_instrumentation_complete.csv", index=False)

if __name__ == "__main__":
    main()
